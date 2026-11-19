"""Incremental curated CSV -> native ROI/real downbeats -> RGB/MS candidates.

All authoritative artifacts are write-once; queues are append-only hints.
A shared initialization/CUDA failure stops its worker, retaining pending sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
REPO = ROOT / 'H3-posttrain'
SPLITS = ROOT / 'OmniDance_processed/pose_preprocess_outputs/mainline/vlm_filter_235b/splits'
EXPECTED = {'train': 200568, 'val': 11110, 'test': 11261}
SCHEMA = 'curated_incremental_annotation_v1'
FLAGS = {'semantic_approval': False, 'human_review_completed': False, 'formal_training': False}


def raw_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(raw_json(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raw_json(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refusing changed existing artifact: ' + str(path))
        return path
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()
    return path


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(raw_json(value))
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def modules(repo, ms_source=None):
    directory = Path(ms_source).resolve() if ms_source else Path(repo).resolve() / 'scripts_omnidance/exp/ms'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import ms_bar_annotation as bar
    import ms_hand_roi as roi
    import ms_visual_batch as visual
    if any(Path(module.__file__).resolve().parent != directory for module in (bar, roi, visual)):
        raise ValueError('Imported MS source differs from explicitly assigned source directory')
    return bar, roi, visual


class SharedFailure(RuntimeError):
    pass


def recover_video(value, sid, meta):
    if isinstance(value, str) and Path(value).is_absolute():
        return value, None
    anomaly = {'field': 'video', 'original_value': value, 'status': 'source_blocked',
               'reason': 'video is not an explicit absolute path'}
    if not meta or not Path(meta).is_absolute():
        return None, anomaly
    try:
        payload = load(meta)
        if payload.get('video_stem') not in (None, sid):
            return None, {**anomaly, 'reason': 'meta video_stem differs from CSV SID', 'meta': ref(meta)}
        candidates = []
        for key in ('video', 'video_path', 'output_video', 'output_video_path'):
            candidate = payload.get(key)
            if isinstance(candidate, str) and Path(candidate).is_absolute() and Path(candidate).name == sid + '.mp4':
                candidates.append((key, candidate))
        if len({v for _, v in candidates}) != 1:
            return None, {**anomaly, 'reason': 'meta has no unambiguous explicit SID-matching video path', 'meta': ref(meta)}
        key, candidate = candidates[0]
        if not Path(candidate).is_file():
            return None, {**anomaly, 'reason': 'explicit metadata video missing', 'meta': ref(meta)}
        return candidate, {**anomaly, 'status': 'recovered_from_explicit_meta', 'meta': ref(meta),
                           'meta_field': key, 'resolved_video': candidate, 'video_sha256': sha(candidate)}
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return None, {**anomaly, 'reason': 'meta recovery failed: ' + str(error)}


def create_manifest(csv_paths, out, batch_size=500, expected=None, rich_root=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rich_root = Path(rich_root or ROOT / 'OmniDance/text')
    batches, counts, inputs, anomalies = [], {}, {}, []
    ids, group_splits = set(), {}
    ordinal = 0
    manifest = out / 'manifest.jsonl'
    with manifest.open('xb') as all_output:
        for split, csv_path in csv_paths.items():
            csv_path = Path(csv_path).resolve()
            source_ref = ref(csv_path)
            inputs[split] = source_ref
            count, buffer = 0, []
            def flush_batch():
                if not buffer:
                    return
                path = out / 'batches' / (buffer[0]['batch_id'] + '.manifest.jsonl')
                path.parent.mkdir(exist_ok=True)
                with path.open('xb') as stream:
                    for row in buffer:
                        stream.write(raw_json(row))
                batches.append({**ref(path), 'batch_id': buffer[0]['batch_id'], 'split': split,
                                'count': len(buffer), 'first_source_ordinal': buffer[0]['source_ordinal']})
                buffer.clear()
            with csv_path.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                stems = [i for i, key in enumerate(header) if key == 'video_stem']
                if not stems:
                    raise ValueError('CSV has no video_stem: ' + str(csv_path))
                for line, values in enumerate(reader, 2):
                    if len(values) != len(header):
                        raise ValueError('CSV column count mismatch at ' + str(line))
                    sid = values[stems[0]]
                    if any(values[i] != sid for i in stems) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', sid):
                        raise ValueError('Unsafe or conflicting duplicate video_stem')
                    if sid.casefold() in ids:
                        raise ValueError('Duplicate/cross-split source ID: ' + sid)
                    ids.add(sid.casefold())
                    original = dict(zip(header, values))
                    group = original['source_video']
                    if not group:
                        raise ValueError('Missing exact source_video group')
                    if group in group_splits and group_splits[group] != split:
                        raise ValueError('Exact source_video group crosses splits')
                    group_splits[group] = split
                    video, anomaly = recover_video(original['video'], sid, original.get('meta'))
                    audio = original['input_audio']
                    if not Path(audio).is_absolute():
                        raise ValueError('Audio must be an explicit absolute CSV path')
                    batch_id = f'{split}-b{count // batch_size:06d}'
                    row = {'schema': SCHEMA, 'id': sid, 'source_id': sid, 'split': split,
                           'video_path': video, 'audio_path': audio, 'nlf_path': None, 'input_sha256': None,
                           'source_group': group, 'source_row_index': original.get('source_row_index'),
                           'source_ordinal': ordinal, 'split_ordinal': count, 'batch_id': batch_id,
                           'rich_path': str(rich_root / (sid + '.txt')), 'meta_path': original.get('meta'),
                           'source_refs': {'csv': {**source_ref, 'csv_record_number': line - 1,
                                                   'csv_physical_line_end': reader.line_num,
                                                   'raw_row_sha256': digest({'header': header, 'values': values}),
                                                   'row_sha256_basis': 'canonical_original_header_and_raw_values'}},
                           'raw_csv_values': values, 'source_anomaly': anomaly,
                           'source_status': 'source_blocked' if video is None else 'pending', **FLAGS}
                    if anomaly:
                        anomalies.append({'id': sid, 'split': split, **anomaly})
                    all_output.write(raw_json(row))
                    buffer.append(row)
                    count += 1
                    ordinal += 1
                    if len(buffer) == batch_size:
                        flush_batch()
                flush_batch()
            if sha(csv_path) != source_ref['sha256']:
                raise ValueError('CSV changed during manifest creation')
            counts[split] = count
            if expected is not None and count != expected[split]:
                raise ValueError(f'Curated count mismatch for {split}: {count} != {expected[split]}')
            inputs[split]['header'] = header
    summary = {'schema': SCHEMA, 'counts': counts, 'total': ordinal, 'batch_size': batch_size,
               'manifest': ref(manifest), 'inputs': inputs, 'batches': batches, 'source_anomalies': anomalies,
               'exact_id_and_source_video_split_overlap': 0,
               'song_or_near_duplicate_overlap_verified': False,
               'all_media_read_at_manifest_stage': False, 'nlf_is_optional_provenance': True, **FLAGS}
    save(out / 'manifest.summary.json', summary)
    return summary


def iter_rows(summary, shard=0, nshard=1, limit=0, ids=None):
    if not 0 <= shard < nshard or limit < 0:
        raise ValueError('Invalid shard/limit')
    selected = 0
    wanted = set(ids) if ids else None
    for batch in summary['batches']:
        if sha(batch['path']) != batch['sha256']:
            raise ValueError('Batch manifest SHA changed')
        count = 0
        with Path(batch['path']).open() as f:
            for line in f:
                row = json.loads(line)
                count += 1
                if row['split'] != batch['split'] or row['batch_id'] != batch['batch_id']:
                    raise ValueError('Batch/split binding mismatch')
                if row['source_ordinal'] % nshard != shard or (wanted is not None and row['id'] not in wanted):
                    continue
                yield row, batch
                selected += 1
                if limit and selected >= limit:
                    return
        if count != batch['count']:
            raise ValueError('Batch count changed')


def source_dir(out, row):
    return Path(out) / 'sources' / row['split'] / row['id']


def audio_probe(path, ffprobe):
    data = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'a:0', '-show_streams',
                           '-show_format', '-of', 'json', str(path)], check=True, capture_output=True).stdout
    value = json.loads(data)
    if len(value.get('streams', [])) != 1:
        raise ValueError('Actual source audio needs one audio stream')
    return {'probe': value, 'probe_sha256': hashlib.sha256(data).hexdigest()}


def merge_short(windows, minimum=.75):
    result, changes = copy.deepcopy(windows), []
    i = 0
    while len(result) > 1 and i < len(result):
        w = result[i]
        if w['end'] - w['start'] >= minimum:
            i += 1
            continue
        changes.append({'short_window': copy.deepcopy(w), 'minimum_seconds': minimum,
                        'reason': 'insufficient_temporal_context_for_visual_motion'})
        if i + 1 < len(result):
            result[i + 1]['start'] = w['start']
            result[i + 1]['raw_window_indices'] = w['raw_window_indices'] + result[i + 1]['raw_window_indices']
            result.pop(i)
        else:
            result[i - 1]['end'] = w['end']
            result[i - 1]['raw_window_indices'] += w['raw_window_indices']
            result.pop(i)
            i = max(0, i - 1)
    return result, changes


def build_plan(row, batch_ref, source, probe, detection, detection_ref):
    binding = {'id': row['id'], 'input_sha256': None,
               'video_sha256': source['files']['video_path']['sha256'],
               'audio_sha256': source['files']['audio_path']['sha256'],
               'actual_video_pts_sha256': probe['actual_video_pts_sha256'],
               'media_start': 0., 'media_end': probe['media_end'], 'video_pts_offset': probe['video_pts_offset'],
               'video_fps': probe['video_fps'], 'decoded_video_frames': len(probe['times'])}
    if source['files']['nlf_path'] is not None:
        binding['input_sha256'] = source['files']['nlf_path']['sha256']
    if detection['audio_sha256'] != binding['audio_sha256']:
        raise ValueError('Real downbeat audio SHA mismatch')
    original_db = detection.get('downbeats') or []
    db = [float(t) for t in original_db if 0 <= t <= binding['media_end']]
    if any(not math.isfinite(t) for t in db) or any(a >= b for a, b in zip(db, db[1:])):
        raise ValueError('Downbeat timestamps must be finite and increasing')
    raw = []
    if db:
        if db[0] > 0:
            raw.append({'start': 0., 'end': db[0], 'kind': 'prefix_residual'})
        raw.extend({'start': a, 'end': b, 'kind': 'complete_detected_bar', 'downbeat_indices': [i, i+1]}
                   for i, (a, b) in enumerate(zip(db, db[1:])))
        if db[-1] < binding['media_end']:
            raw.append({'start': db[-1], 'end': binding['media_end'], 'kind': 'suffix_residual'})
    else:
        raw = [{'start': 0., 'end': binding['media_end'], 'kind': 'unresolved_no_downbeats'}]
    windows = [{'start': w['start'], 'end': w['end'], 'raw_window_indices': [i]} for i, w in enumerate(raw)]
    status = 'ready'
    if len(db) >= 2:
        if len(windows) > 1 and raw[0]['kind'] == 'prefix_residual' and windows[0]['end'] < 1.5:
            first = windows.pop(0)
            windows[0]['start'] = 0.
            windows[0]['raw_window_indices'] = first['raw_window_indices'] + windows[0]['raw_window_indices']
        if len(windows) > 1 and raw[-1]['kind'] == 'suffix_residual' and windows[-1]['end'] - windows[-1]['start'] < 1.5:
            tail = windows.pop()
            windows[-1]['end'] = tail['end']
            windows[-1]['raw_window_indices'] += tail['raw_window_indices']
    else:
        from motionscript_v2.chunks import fixed_windows
        windows = fixed_windows(0., binding['media_end'], 2., 1.5)
        for w in windows:
            w['raw_window_indices'] = [i for i, r in enumerate(raw) if r['start'] < w['end'] and r['end'] > w['start']]
        status = 'ready_fixed_fallback'
    pre_visual = copy.deepcopy(windows)
    windows, changes = merge_short(windows)
    beat_source = copy.deepcopy(detection['source'])
    beat_source.update(original_downbeats_seconds=original_db,
                       model_timestamps_outside_video_omitted=[v for v in original_db if not 0 <= v <= binding['media_end']])
    music = {'schema': 'motionscript_music_windows_v1', 'id': row['id'], 'binding': binding,
             'strategy': 'model_downbeat_bars', 'status': status, 'raw_windows': raw, 'windows': windows,
             'downbeats_seconds': db, 'provenance': beat_source,
             'window_interval': '[start,end)', 'time_basis': 'global_media_seconds',
             'pre_visual_support_windows': pre_visual, 'visual_support_merges': changes,
             'params': {'policy': 'merge_short_edges', 'min_edge_seconds': 1.5, 'visual_min_seconds': .75},
             'annotation_policy': 'detected_bar_merge_edges_visual_min075_v1',
             'fallback_is_verified_bar': False, 'one_window_is_exactly_one_bar': False,
             'beat_phase_verified': False, 'downbeat_verified': False,
             'audio_video_alignment_verified': False, 'motions_on_beats_verified': False,
             'window_boundaries_are_action_boundaries': False, **FLAGS}
    return {'schema': 'bar_visual_annotation_plan_v1', 'id': row['id'], 'manifest_record': row,
            'binding': binding, 'window_manifest': music, 'windows': copy.deepcopy(windows),
            'plan_status': status, 'data_source': 'curated_full_csv_actual_media_v1',
            'source_refs': {'manifest': {'path': batch_ref['path'], 'sha256': batch_ref['sha256']},
                            'original_csv': row['source_refs']['csv'], 'beat_detection': detection_ref},
            'beat_detection': detection, **FLAGS}


def load_beat_runtime(args):
    for dependency in (args.beat_sitepath, args.beat_audio_sitepath):
        if dependency:
            site = Path(dependency)
            if not site.is_absolute() or not site.is_dir():
                raise SharedFailure('Explicit BeatThis dependency sitepath missing')
            if str(site) not in sys.path:
                sys.path.append(str(site))
    try:
        import beat_this.inference
        source_path = Path(args.beat_checkpoint_source)
        checkpoint_source = load(source_path)
        checkpoint = Path(checkpoint_source['checkpoint_path'])
        if sha(checkpoint) != checkpoint_source['checkpoint_sha256']:
            raise ValueError('BeatThis checkpoint SHA differs')
        if sha(checkpoint_source['license_path']) != checkpoint_source['license_sha256']:
            raise ValueError('BeatThis license SHA differs')
        tracker = beat_this.inference.File2Beats(checkpoint_path=str(checkpoint),
                                               device=args.beat_device, float16=False, dbn=False)
        params = {'method': 'beat_this.inference.File2Beats', 'checkpoint_name': 'final0', 'dbn': False,
                  'device': args.beat_device, 'synthetic': False,
                  'downbeats_origin': 'model_downbeat_head_postprocessed',
                  'checkpoint_sha256': checkpoint_source['checkpoint_sha256'],
                  'library_source_sha256': sha(beat_this.inference.__file__),
                  'library_version': importlib.metadata.version('beat-this'),
                  'checkpoint_source_ref': ref(source_path), 'audio_video_offset_seconds': 'unknown'}
        return tracker, params
    except Exception as error:
        raise SharedFailure('BeatThis shared initialization failed: ' + str(error)) from error


def load_reused_beats(paths, checkpoint_source):
    result = {}
    expected = load(checkpoint_source)['checkpoint_sha256']
    for path in paths:
        path = Path(path)
        receipt_path = path.with_name('run_receipt.json')
        receipt = load(receipt_path)
        if receipt['model']['checkpoint_sha256'] != expected:
            raise ValueError('Reuse BeatThis checkpoint mismatch')
        with path.open() as stream:
            for line in stream:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    break  # A live append-only producer may have one incomplete last line.
                if d.get('status') != 'ok' or not d.get('audio_sha256'):
                    continue
                source = {'method': 'beat_this.inference.File2Beats', 'checkpoint_name': 'final0',
                          'dbn': False, 'device': d.get('params', {}).get('device', 'cpu'), 'synthetic': False,
                          'downbeats_origin': 'model_downbeat_head_postprocessed', 'checkpoint_sha256': expected,
                          'library_source_sha256': receipt['runtime']['beatthis_inference_sha256'],
                          'library_version': d['library_versions']['beat-this'], 'audio_sha256': d['audio_sha256']}
                result[d['audio_sha256']] = {**d, 'source': source,
                    'reused_detection_ref': {**ref(receipt_path), 'detections_path': str(path),
                                             'detection_row_sha256': digest(d)}}
    return result


def finish_cuda_profile(model, directory):
    receipts = {}
    try:
        for name, session in [('detector', model.session_det), ('pose', model.session_pose)]:
            path = Path(session.end_profiling())
            events = load(path)
            providers = Counter(e['args']['provider'] for e in events
                                if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
            if not providers['CUDAExecutionProvider']:
                raise ValueError('No actual CUDA compute in ' + name)
            receipts[name] = {'profile': ref(path), 'node_provider_counts': dict(providers)}
        save(Path(directory) / 'cuda_execution_receipt.json',
             {'status': 'verified_real_cuda', 'profile_ended_after_first_successful_source': True,
              'sessions': receipts, **FLAGS})
    except Exception as error:
        raise SharedFailure('Shared CUDA profile validation failed: ' + str(error)) from error


def run_producer(args):
    summary = load(args.manifest)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bar, roi, visual = modules(args.repo, args.ms_source)
    model_ref = load(args.roi_model_reference) if args.roi_model_reference else roi.model_identity()
    try:
        if args.roi_model_reference:
            model_ref = copy.deepcopy(model_ref)
            for key, identity in model_ref['files'].items():
                try:
                    roi.verify_identity(identity)
                except ValueError:
                    observed = roi.file_identity(identity['path'])
                    if observed['sha256'] != identity['sha256']:
                        raise ValueError('ROI reference content SHA differs: ' + key)
                    model_ref['files'][key] = observed
            model_ref['sha256'] = roi.digest(model_ref['files'])
        roi.verify_model(model_ref)
    except Exception as error:
        raise SharedFailure('ROI shared model identity failed: ' + str(error)) from error
    model_ref = copy.deepcopy(model_ref)
    for key, path in [('roi_source', Path(roi.__file__)),
                      ('layout_source', Path(roi.__file__).with_name('dwpose_full_extract.py'))]:
        if sha(path) != model_ref['files'][key]['sha256']:
            raise SharedFailure('Executed ROI code differs from verified model reference: ' + key)
        model_ref['files'][key] = roi.file_identity(path)
    model_ref['sha256'] = roi.digest(model_ref['files'])
    params = dict(roi.DEFAULT_PARAMS, device=args.device)
    roi_root = out / 'roi'
    roi_root.mkdir(exist_ok=True)
    run_id = f'p{args.shard}-{uuid.uuid4().hex}'
    run_root = out / 'runs' / run_id
    run_root.mkdir(parents=True)
    save(run_root / 'protocol.json', {'schema': SCHEMA, 'mode': 'producer', 'stage': args.stage,
         'manifest_summary': ref(args.manifest), 'counts': summary['counts'], 'total': summary['total'],
         'shard': args.shard, 'nshard': args.nshard, 'roi_model_ref': model_ref, 'roi_params': params,
         'beat_device': args.beat_device, 'pipeline_code': ref(__file__), **FLAGS})
    reuse = load_reused_beats(args.reuse_beats, args.beat_checkpoint_source)
    model, beat_runtime, profile_done = None, None, False
    manifests, counts = {}, Counter()
    for row, batch_ref in iter_rows(summary, args.shard, args.nshard, args.limit, args.ids):
        directory = source_dir(out, row)
        terminal = directory / 'producer.terminal.json'
        if terminal.exists():
            counts['resumed_terminal'] += 1
            append(out / 'ready.jsonl', load(terminal))
            continue
        with visual.bundle_lock(directory):
            try:
                if row['video_path'] is None:
                    raise ValueError('source_blocked: ' + str(row['source_anomaly']))
                source = visual.source_identity(row)
                probe = bar.probe_actual(row['video_path'], args.ffprobe)
                actual_audio = audio_probe(row['audio_path'], args.ffprobe)
                save(directory / 'actual_media.json', {'source_identity': source, 'video_probe': probe,
                                                       'audio_probe': actual_audio, **FLAGS})
                key = batch_ref['path']
                if key not in manifests:
                    manifests[key] = roi.file_identity(key)
                input_row = {'id': row['id'], 'manifest_record': row, 'input_line': row['split_ordinal'] % summary['batch_size'] + 1,
                             'input_row_sha256': digest(row)}
                prepared = roi.prepare_source(input_row, manifests[key], model_ref, params, args.ffprobe, args.ffmpeg)
                roi_directory = roi.ensure_prepared(roi_root, prepared)
                if args.stage == 'prepare':
                    save(directory / 'producer.prepare.json', {'id': row['id'], 'status': 'prepared_not_extracted',
                         'roi_source_ref': ref(roi_directory / 'source_manifest.json'), 'batch_id': row['batch_id'], **FLAGS})
                    counts['prepared'] += 1
                    continue
                if model is None:
                    try:
                        model = roi.load_cuda_model(model_ref, params, run_root / 'cuda_profile')
                    except Exception as error:
                        raise SharedFailure('ROI shared CUDA model initialization failed: ' + str(error)) from error
                roi_manifest = roi.extract_source(roi_directory, prepared, model)
                if not profile_done:
                    finish_cuda_profile(model, run_root / 'cuda_profile')
                    profile_done = True
                audio_sha = source['files']['audio_path']['sha256']
                if audio_sha in reuse:
                    detection = copy.deepcopy(reuse[audio_sha])
                    detection['reused_by_exact_audio_sha256'] = True
                    detection['reused_for_source_id'] = row['id']
                else:
                    if beat_runtime is None:
                        beat_runtime = load_beat_runtime(args)
                        save(run_root / 'beat_model.json', beat_runtime[1])
                    tracker, beat_source = beat_runtime
                    import soundfile
                    info = soundfile.info(row['audio_path'])
                    beats, downbeats = tracker(row['audio_path'])
                    beats, downbeats = [float(t) for t in beats], [float(t) for t in downbeats]
                    duration = info.frames / info.samplerate
                    if any(not math.isfinite(t) or not 0 <= t <= duration + .02 for t in beats + downbeats):
                        raise ValueError('Actual BeatThis timestamps outside decoded audio')
                    if any(a >= b for values in (beats, downbeats) for a, b in zip(values, values[1:])):
                        raise ValueError('Actual BeatThis timestamps not increasing')
                    detection = {'id': row['id'], 'status': 'ok', 'audio_sha256': audio_sha,
                                 'beats': beats, 'downbeats': downbeats, 'duration_seconds': duration,
                                 'sample_rate': info.samplerate, 'channels': info.channels,
                                 'source': {**beat_source, 'audio_sha256': audio_sha},
                                 'model_prediction_returned': True, 'inference_called': True, **FLAGS}
                visual.verify_source(source)
                if sha(row['audio_path']) != audio_sha:
                    raise ValueError('Audio changed during inference')
                save(directory / 'beat_detection.json', detection)
                plan = build_plan(row, batch_ref, source, probe, detection, ref(directory / 'beat_detection.json'))
                bar.validate_row(plan)
                save(directory / 'bar_plan.json', plan)
                value = {'schema': SCHEMA, 'id': row['id'], 'split': row['split'], 'batch_id': row['batch_id'],
                         'status': 'ready', 'roi_status': 'extracted', 'roi_unknowns_allowed': True,
                         'plan_ref': ref(directory / 'bar_plan.json'),
                         'roi_ref': ref(roi_directory / 'roi_manifest.json'), 'source_binding': plan['binding'], **FLAGS}
            except SharedFailure as error:
                save(run_root / 'shared_failure.json', {'status': 'stopped_shared_failure', 'error': str(error),
                     'current_source_id': row['id'], 'remaining_sources_not_marked_failed': True, **FLAGS})
                raise
            except RuntimeError as error:
                if any(w in str(error).lower() for w in ('cuda', 'cudnn', 'cublas', 'executionprovider')):
                    raise SharedFailure('ROI/beat CUDA runtime failure; remaining pending: ' + str(error)) from error
                value = {'schema': SCHEMA, 'id': row['id'], 'split': row['split'], 'batch_id': row['batch_id'],
                         'status': 'failed', 'error': f'{type(error).__name__}: {error}', **FLAGS}
            except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
                value = {'schema': SCHEMA, 'id': row['id'], 'split': row['split'], 'batch_id': row['batch_id'],
                         'status': 'failed', 'error': f'{type(error).__name__}: {error}', **FLAGS}
            value['source_ordinal'] = row['source_ordinal']
            if value['status'] == 'failed':
                value['manifest_record'] = row
            save(terminal, value)
            append(out / 'ready.jsonl', value)
            counts[value['status']] += 1
            append(run_root / 'progress.jsonl', {'id': row['id'], 'counts': dict(counts), 'total': summary['total'], **FLAGS})
    save(run_root / 'completed.json', {'schema': SCHEMA, 'counts': dict(counts), 'total': summary['total'], **FLAGS})
    return {'run_root': str(run_root), 'counts': dict(counts)}


def load_style_reuse(path):
    if not path:
        return {}
    if isinstance(path, (list, tuple)):
        merged = {}
        for source_path in path:
            selected = load_style_reuse(source_path)
            if merged.keys() & selected.keys():
                raise ValueError('Duplicate source ID across historical label files')
            merged.update(selected)
        return merged
    identity = ref(path)
    text = Path(path).read_text()
    if sha(path) != identity['sha256']:
        raise ValueError('Style reuse source changed while reading')
    line_numbers = None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and {'sid', 'label', 'raw'} <= payload.keys():
            records = [payload]
            line_numbers = [next(i for i, line in enumerate(text.splitlines(), 1) if line.strip())]
        else:
            records = payload.get('records', payload) if isinstance(payload, dict) else payload
            if isinstance(records, dict):
                records = [{'id': sid, **row} for sid, row in records.items()]
    except json.JSONDecodeError:
        parsed = [(i, json.loads(line)) for i, line in enumerate(text.splitlines(), 1) if line.strip()]
        line_numbers, records = [i for i, _ in parsed], [r for _, r in parsed]
    import ms_visual_batch as batch
    result, label_files = {}, {}
    if line_numbers is not None:
        label_files[(identity['path'], identity['sha256'])] = dict(zip(line_numbers, records))
    for number, record in enumerate(records):
        sid = record.get('id', record.get('source_id', record.get('sid')))
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', sid) or sid in result:
            raise ValueError('Style reuse requires unique safe source IDs')
        current_sha = record.get('current_video_sha256', record.get('video_sha256', record.get('binding', {}).get('video_sha256')))
        historical_sha = record.get('historical_video_sha256')
        for value in (current_sha, historical_sha):
            if value is not None and not re.fullmatch(r'[0-9a-f]{64}', str(value)):
                raise ValueError('Style reuse supplied invalid current/historical SHA')
        label = record.get('label', record.get('historical_label'))
        if not isinstance(label, dict):
            raise ValueError('Style reuse must bind an actual historical RGB label object')
        source_refs = copy.deepcopy(record.get('source_refs', {}))
        label_ref = source_refs.get('historical_label_record', record.get('historical_label_record'))
        if label_ref is None and line_numbers is not None and {'sid', 'label', 'raw'} <= record.keys():
            label_ref = {**identity, 'sid': sid, 'line': line_numbers[number],
                         'row_sha256': batch.digest(record)}
        if label_ref is None:
            raise ValueError('Historical RGB reuse requires original label file/row SHA and line provenance')
        if label_ref.get('sid') != sid or type(label_ref.get('line')) is not int or label_ref['line'] < 1:
            raise ValueError('Historical label source ID/line mismatch')
        key = (label_ref['path'], label_ref['sha256'])
        if key not in label_files:
            if sha(key[0]) != key[1]:
                raise ValueError('Original historical label file SHA differs')
            with Path(key[0]).open() as stream:
                label_files[key] = {i: json.loads(line) for i, line in enumerate(stream, 1) if line.strip()}
            if sha(key[0]) != key[1]:
                raise ValueError('Original historical label file changed while reading')
        original = label_files[key].get(label_ref['line'])
        if (original is None or original.get('sid', original.get('id')) != sid or original.get('label') != label
                or batch.digest(original) != label_ref.get('row_sha256')):
            raise ValueError('Historical label original row SHA/label/SID binding differs')
        source_refs['historical_label_record'] = label_ref
        historical_path = record.get('historical_video_path', record.get('video_path'))
        if historical_path is None:
            historical_path = str(ROOT / 'OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/videos' / (sid + '.mp4'))
        result[sid] = {'label': label, 'source_refs': source_refs, 'reuse_index': identity,
                       'record_sha256': digest(record), 'current_video_sha256': current_sha,
                       'historical_video_sha256': historical_sha, 'historical_video_path': historical_path}
    return result


def joint_candidate(row, visual_result, style_result, model_sha=None):
    label = style_result.get('dance_style', 'unknown')
    known = isinstance(label, str) and label not in ('', 'unknown') and style_result.get('status') in ('resolved', 'candidate')
    windows = visual_result.get('final', {}).get('json', {}).get('windows', [])
    original_windows = visual_result.get('original_windows', [])
    full = (visual_result.get('status') == 'candidate' and bool(windows)
            and len(windows) == len(original_windows)
            and all(isinstance(w.get('caption'), str) and w['caption'].strip() for w in windows)
            and bool(visual_result.get('coverage', {}).get('full_window_caption')))
    text = visual_result.get('final_timed_caption', '')
    complete = full and known and bool(text.strip())
    return {'schema': SCHEMA, 'id': row['id'], 'source_id': row['id'], 'split': row['split'],
            'batch_id': row['batch_id'], 'video_path': row['video_path'], 'audio_path': row['audio_path'],
            'dance_style_label': label, 'label': {'dance_style': label, 'timed_motion_script': text},
            'target_text': 'Dance style: ' + label + '\n' + text if complete else '',
            'joint_complete': complete, 'all_windows_nonempty': full,
            'style_status': style_result.get('status', 'pending'), 'visual_status': visual_result.get('status'),
            'annotator_model_sha256': model_sha, 'source_binding': visual_result.get('binding'),
            'source_refs': row.get('source_refs'), 'style_result': style_result,
            'raw_rgb_manifest': visual_result.get('raw_rgb_manifest'), 'hand_roi': visual_result.get('hand_roi'),
            'visual_result_ref': visual_result.get('result_ref'),
            'batch_review_status': 'pending_independent_review', 'training_eligible': False, **FLAGS}


def observe_dance_style(row, plan, visual_result, visual_root, args, runtime, reuse):
    try:
        import ms_dance_style as style
    except ImportError:
        return {'id': row['id'], 'status': 'pending', 'dance_style': 'unknown',
                'reason': 'independent_style_module_not_yet_available', **FLAGS}
    refs, rich = {}, ''
    rich_path = row.get('rich_path')
    if rich_path and Path(rich_path).is_file():
        refs['rich'] = ref(rich_path)
        rich = Path(rich_path).read_text()
        if sha(rich_path) != refs['rich']['sha256']:
            raise ValueError('Original rich text changed while reading')
    historical = reuse.get(row['id'])
    if historical and historical['current_video_sha256'] not in (None, plan['binding']['video_sha256']):
        refs['rejected_historical_reuse'] = {'reason': 'current/posthoc_video_SHA_mismatch', **historical}
        historical = None
    if historical:
        refs.update(historical['source_refs'])
        refs['historical_reuse_index'] = historical['reuse_index']
    prepared = style.prepare_style(plan, label=historical['label'] if historical else None, rich_text=rich,
                                    source_refs=refs, historical_video_sha256=historical['historical_video_sha256'] if historical else None,
                                    historical_video_path=historical['historical_video_path'] if historical else None)
    if prepared['status'] == 'resolved':
        return style.observe_style(prepared, None, runtime, args)
    manifest_path = visual_root / 'rgb/manifest.json'
    if not manifest_path.exists():
        return {'id': row['id'], 'status': 'pending', 'dance_style': 'unknown',
                'reason': 'source_RGB_not_prepared', **FLAGS}
    manifest = load(manifest_path)
    actual_video_sha = manifest.get('binding', {}).get('source_binding', {}).get('video_sha256')
    if actual_video_sha != plan['binding']['video_sha256']:
        raise ValueError('Style union RGB video binding differs')
    manifest = {**manifest, 'video_sha256': actual_video_sha}
    if len(manifest.get('frames', [])) < 2:
        return {'id': row['id'], 'status': 'unknown', 'dance_style': 'unknown',
                'reason': 'fewer_than_two_actual_source_RGB_frames', **FLAGS}
    item = style.style_item(prepared, visual_root / 'rgb', manifest,
                            visual_root / 'style/attempt-000001', args)
    return style.observe_style(prepared, item, runtime, args)


def run_consumer(args):
    summary = load(args.manifest)
    if args.tp != 2 or args.batch_size < 1 or args.max_num_seqs < args.batch_size:
        raise ValueError('Native visual consumer requires TP2 and valid batch budget')
    if not 0 <= args.shard < args.nshard or args.poll_seconds <= 0:
        raise ValueError('Invalid consumer shard/watch polling')
    bar, roi, visual = modules(args.repo, args.ms_source)
    out = Path(args.out).resolve()
    args.hand_roi_root = str(out / 'roi')
    visual_out = out / 'visual' / f'shard-{args.shard:04d}'
    visual_out.mkdir(parents=True, exist_ok=True)
    code = bar.code_identity()
    configuration = bar.params(args, code)
    style_module_path = Path(bar.__file__).with_name('ms_dance_style.py')
    configuration['style_module_sha256'] = sha(style_module_path) if style_module_path.exists() else None
    run_root = out / 'runs' / ('v' + str(args.shard) + '-' + uuid.uuid4().hex)
    save(run_root / 'protocol.json', {'schema': SCHEMA, 'mode': 'consumer', 'manifest_summary': ref(args.manifest),
         'total': summary['total'], 'counts': summary['counts'], 'shard': args.shard, 'nshard': args.nshard,
         'configuration': configuration, 'pipeline_code': ref(__file__), **FLAGS})
    try:
        runtime = bar.runtime_loader(args, visual_out, code)
        reuse = load_style_reuse(args.style_reuse)
    except Exception as error:
        raise SharedFailure('Visual shared initialization failed: ' + str(error)) from error
    seen, counts, ref_cache = set(), Counter(), {}
    position = 0
    assigned_total = sum(1 for i in range(summary['total']) if i % args.nshard == args.shard)
    wanted = set(args.ids) if args.ids else None
    if wanted:
        assigned_total = sum(1 for _ in iter_rows(summary, args.shard, args.nshard, ids=wanted))
    if args.limit:
        assigned_total = min(assigned_total, args.limit)
    while len(seen) < assigned_total:
        queue = out / 'ready.jsonl'
        if not queue.exists():
            if not args.watch:
                break
            time.sleep(args.poll_seconds)
            continue
        progressed = False
        with queue.open('rb') as stream:
            stream.seek(position)
            while True:
                begin = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    position = begin
                    break
                position = stream.tell()
                hint = json.loads(line)
                if hint['source_ordinal'] % args.nshard != args.shard or hint['id'] in seen or (wanted and hint['id'] not in wanted):
                    continue
                directory = out / 'sources' / hint['split'] / hint['id']
                terminal = load(directory / 'producer.terminal.json')
                if terminal != hint:
                    raise ValueError('Ready hint differs from immutable producer terminal')
                consumer_terminal = directory / 'consumer.terminal.json'
                if consumer_terminal.exists():
                    previous = load(consumer_terminal)
                    if previous.get('params_sha256') != visual.digest(configuration):
                        raise ValueError('Refusing consumer resume with different code/model/parameters')
                    artifact = previous.get('joint_candidate_ref')
                    if artifact and sha(artifact['path']) != artifact['sha256']:
                        raise ValueError('Resumed joint candidate artifact changed')
                    seen.add(hint['id'])
                    counts['resumed_terminal'] += 1
                    continue
                if terminal['status'] == 'failed':
                    row = terminal['manifest_record']
                    value = {'id': row['id'], 'split': row['split'], 'batch_id': row['batch_id'],
                             'status': 'producer_failed', 'producer_terminal_ref': ref(directory / 'producer.terminal.json'),
                             'manifest_record': row, **FLAGS}
                    value['params_sha256'] = visual.digest(configuration)
                    save(consumer_terminal, value)
                    append(out / 'completed.jsonl', value)
                    notify_batch_terminal(args, summary, value)
                    seen.add(row['id'])
                    counts['producer_failed'] += 1
                    continue
                plan_ref = terminal['plan_ref']
                if sha(plan_ref['path']) != plan_ref['sha256'] or sha(terminal['roi_ref']['path']) != terminal['roi_ref']['sha256']:
                    raise ValueError('Producer terminal plan/ROI artifact SHA changed')
                plan = bar.validate_row(load(plan_ref['path']))
                row = plan['manifest_record']
                if terminal['source_binding'] != plan['binding']:
                    raise ValueError('Producer terminal source binding differs')
                root = visual_out / 'sources' / row['id']
                raw_line = Path(plan_ref['path']).read_bytes()
                binding = {'source_binding': plan['binding'], 'plan_sha256': plan_ref['sha256'],
                           'full_source_row_sha256': visual.digest(plan), 'raw_line_sha256': hashlib.sha256(raw_line).hexdigest(),
                           'manifest_record_sha256': visual.digest(row), 'params_sha256': visual.digest(configuration)}
                if args.stage == 'run':
                    try:
                        runtime()  # Shared load failure must not create a failed source/window.
                    except Exception as error:
                        save(run_root / 'shared_failure.json', {'status': 'stopped_shared_failure', 'error': str(error),
                             'current_source_id': row['id'], 'remaining_sources_not_marked_failed': True, **FLAGS})
                        raise SharedFailure('Visual shared runtime load failed: ' + str(error)) from error
                with visual.bundle_lock(directory):
                    with visual.bundle_lock(root):
                        try:
                            result = bar.process_source(plan, raw_line, root, args, binding, runtime, ref_cache)
                            style_result = observe_dance_style(row, plan, result, root, args, runtime, reuse)
                        except RuntimeError as error:
                            raise SharedFailure('Visual/style shared runtime failure: ' + str(error)) from error
                        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
                            value = {'id': row['id'], 'split': row['split'], 'batch_id': row['batch_id'],
                                     'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                                     'manifest_record': row, **FLAGS}
                        else:
                            model_path = visual_out / 'model.identity.json'
                            model_sha = load(model_path)['sha256'] if model_path.exists() else None
                            joint = joint_candidate(row, result, style_result, model_sha)
                            actual_media = load(directory / 'actual_media.json')
                            joint['source_identity'] = actual_media['source_identity']
                            joint['actual_video_probe'] = actual_media['video_probe']
                            style_path = directory / 'style-results' / (digest(style_result) + '.json')
                            save(style_path, style_result)
                            joint['style_result_ref'] = ref(style_path)
                            snapshot = directory / 'joint-results' / (digest(joint) + '.json')
                            save(snapshot, joint)
                            value = {'id': row['id'], 'split': row['split'], 'batch_id': row['batch_id'],
                                     'status': 'joint_candidate' if joint['joint_complete'] else 'incomplete_candidate',
                                     'joint_complete': joint['joint_complete'], 'joint_candidate_ref': ref(snapshot),
                                     'style_status': style_result['status'], 'visual_status': result['status'], **FLAGS}
                            if args.stage == 'prepare' or style_result['status'] in ('pending', 'prepared') or result['status'] == 'prepared':
                                save(directory / 'consumer-pending' / (digest(value) + '.json'), value)
                                counts['prepared_or_style_pending'] += 1
                                seen.add(row['id'])
                                continue
                        value['params_sha256'] = visual.digest(configuration)
                        save(consumer_terminal, value)
                        append(out / 'completed.jsonl', value)
                        notify_batch_terminal(args, summary, value)
                        counts[value['status']] += 1
                seen.add(row['id'])
                progressed = True
                append(run_root / 'progress.jsonl', {'id': row['id'], 'counts': dict(counts),
                                                   'selected_total': assigned_total, 'total': summary['total'], **FLAGS})
                if len(seen) >= assigned_total:
                    break
        if not args.watch:
            break
        if not progressed and len(seen) < assigned_total:
            time.sleep(args.poll_seconds)
    save(run_root / 'completed.json', {'counts': dict(counts), 'total': summary['total'],
                                      'queue_sources_consumed': len(seen), 'selected_total': assigned_total, **FLAGS})
    return {'run_root': str(run_root), 'counts': dict(counts), 'queue_sources_consumed': len(seen)}


def notify_batch_terminal(args, summary, terminal):
    batch_id = terminal['batch_id']
    batch = next(b for b in summary['batches'] if b['batch_id'] == batch_id)
    directory = Path(args.out) / 'batches' / batch_id
    index = directory / 'terminal_index.jsonl'
    append(index, {'id': terminal['id'], 'split': terminal['split']})
    seen = set()
    with index.open() as stream:
        for line in stream:
            try:
                seen.add(json.loads(line)['id'])
            except json.JSONDecodeError:
                break
    if len(seen) < batch['count']:
        return
    with (directory / '.publish.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (directory / 'batch_ready.json').exists():
            from types import SimpleNamespace
            publish_batches(SimpleNamespace(manifest=args.manifest, out=args.out, batch_ids=[batch_id]))
        fcntl.flock(lock, fcntl.LOCK_UN)


def publish_batches(args):
    summary = load(args.manifest)
    out = Path(args.out).resolve()
    published, pending = [], []
    for batch in summary['batches']:
        if getattr(args, 'batch_ids', None) and batch['batch_id'] not in args.batch_ids:
            continue
        if sha(batch['path']) != batch['sha256']:
            raise ValueError('Batch manifest changed')
        rows = [json.loads(line) for line in Path(batch['path']).read_text().splitlines() if line.strip()]
        records, missing = [], []
        for row in rows:
            directory = source_dir(out, row)
            path = directory / 'consumer.terminal.json'
            if not path.exists():
                missing.append(row['id'])
                continue
            terminal = load(path)
            if terminal['batch_id'] != batch['batch_id'] or terminal['split'] != batch['split']:
                raise ValueError('Consumer batch/split binding differs')
            if terminal.get('joint_candidate_ref'):
                artifact = terminal['joint_candidate_ref']
                if sha(artifact['path']) != artifact['sha256']:
                    raise ValueError('Joint candidate artifact SHA changed')
                record = load(artifact['path'])
                record['label_path'] = artifact['path']
                record['label_sha256'] = artifact['sha256']
            else:
                record = {'id': row['id'], 'source_id': row['id'], 'split': row['split'],
                          'batch_id': row['batch_id'], 'video_path': row['video_path'],
                          'joint_complete': False, 'status': terminal['status'],
                          'label': {'dance_style': 'unknown', 'timed_motion_script': ''},
                          'terminal_ref': ref(path), **FLAGS}
            records.append(record)
        if missing:
            pending.append({'batch_id': batch['batch_id'], 'missing_count': len(missing)})
            continue
        batch_root = out / 'batches' / batch['batch_id']
        batch_ready = {'schema': SCHEMA, 'batch_id': batch['batch_id'], 'split': batch['split'],
                       'source_ids': [r['id'] for r in rows], 'count': len(rows), 'records': records,
                       'source_manifest_ref': batch, 'joint_complete_count': sum(bool(r['joint_complete']) for r in records),
                       'sampling_assumption': {'random_count': 25, 'risk_count': 5,
                                               'engineering_defaults_not_statistical_quality_proof': True},
                       'independent_review_status': 'pending', 'human_confirmed_error_gate': 'pending', **FLAGS}
        path = save(batch_root / 'batch_ready.json', batch_ready)
        jsonl = batch_root / 'completed.jsonl'
        payload = b''.join(raw_json(r) for r in records)
        if jsonl.exists():
            if jsonl.read_bytes() != payload:
                raise ValueError('Refusing changed batch completed JSONL')
        else:
            with jsonl.open('xb') as stream:
                stream.write(payload)
        published.append(ref(path))
    index = {'schema': SCHEMA, 'manifest_summary': ref(args.manifest), 'published': published,
             'pending': pending, 'total_source_count': summary['total'], **FLAGS}
    save(out / 'batch-snapshots' / (digest(index) + '.json'), index)
    return index


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    m = sub.add_parser('manifest')
    m.add_argument('--output-dir', required=True)
    m.add_argument('--train-csv', default=str(SPLITS / 'train.csv'))
    m.add_argument('--val-csv', default=str(SPLITS / 'val.csv'))
    m.add_argument('--test-csv', default=str(SPLITS / 'test.csv'))
    m.add_argument('--batch-size', type=int, default=500)
    m.add_argument('--rich-root', default=str(ROOT / 'OmniDance/text'))
    m.add_argument('--allow-nonproduction-counts', action='store_true')
    for name in ('producer', 'consumer', 'batches'):
        q = sub.add_parser(name)
        q.add_argument('--manifest', required=True, help='manifest.summary.json')
        q.add_argument('--out', required=True)
        q.add_argument('--repo', default=str(REPO))
        q.add_argument('--ms-source', help='Explicit frozen flat MS source directory; imports must originate there')
        q.add_argument('--shard', type=int, default=0)
        q.add_argument('--nshard', type=int, default=1)
        q.add_argument('--limit', type=int, default=0)
        q.add_argument('--ids', nargs='+')
        if name == 'batches':
            q.add_argument('--batch-ids', nargs='+')
        q.add_argument('--stage', choices=('prepare', 'run'), default='run')
        q.add_argument('--ffprobe', default='/opt/ms_tools/ffprobe')
        q.add_argument('--ffmpeg', default='/usr/local/bin/ffmpeg')
        if name == 'producer':
            q.add_argument('--device', type=int, default=0, help='CUDA ordinal within assigned visible GPU')
            q.add_argument('--roi-model-reference')
            q.add_argument('--beat-sitepath', default='/opt/venv_beatthis/lib64/python3.11/site-packages')
            q.add_argument('--beat-audio-sitepath', default='/opt/venv_music/lib/python3.11/site-packages')
            q.add_argument('--beat-device', choices=('cpu', 'cuda'), default='cuda')
            q.add_argument('--beat-checkpoint-source', default=str(REPO / 'models/BeatThis/source_manifest.json'))
            q.add_argument('--reuse-beats', action='append', default=[])
        if name == 'consumer':
            q.add_argument('--model', required=True)
            q.add_argument('--model-reference')
            q.add_argument('--tp', type=int, default=2)
            q.add_argument('--batch-size', type=int, default=2)
            q.add_argument('--max-num-seqs', type=int, default=2)
            q.add_argument('--sample-fps', type=float, default=8.)
            q.add_argument('--max-model-len', type=int, default=16384)
            q.add_argument('--max-pixels', type=int, default=200704)
            q.add_argument('--visual-max-tokens', type=int, default=1800)
            q.add_argument('--text-max-tokens', type=int, default=4096)
            q.add_argument('--gpu-memory-utilization', type=float, default=.88)
            q.add_argument('--boundary-review', choices=('on', 'off'), default='on')
            q.add_argument('--rewrite-reference-repair', choices=('on', 'off'), default='on')
            q.add_argument('--retry-failed', action='store_true')
            q.add_argument('--watch', action='store_true')
            q.add_argument('--poll-seconds', type=float, default=5.)
            q.add_argument('--style-reuse', nargs='+', help='Original labels_s0..s3.jsonl or verified reuse index files')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == 'manifest':
        return create_manifest({'train': args.train_csv, 'val': args.val_csv, 'test': args.test_csv},
                               args.output_dir, args.batch_size,
                               None if args.allow_nonproduction_counts else EXPECTED, args.rich_root)
    if args.mode == 'producer':
        return run_producer(args)
    if args.mode == 'consumer':
        return run_consumer(args)
    return publish_batches(args)


if __name__ == '__main__':
    try:
        print(json.dumps(main(), ensure_ascii=False), flush=True)
    except Exception as error:
        print(json.dumps({'status': 'worker_stopped', 'error': f'{type(error).__name__}: {error}',
                          'remaining_sources_not_marked_failed': isinstance(error, SharedFailure), **FLAGS}),
              file=sys.stderr, flush=True)
        sys.exit(2)
