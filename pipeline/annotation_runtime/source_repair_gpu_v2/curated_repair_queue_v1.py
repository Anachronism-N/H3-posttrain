"""CPU-only immutable failure queue; repair execution requires caller-owned VLM runtime."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import math
import subprocess
import sys
import time
import uuid

SCHEMA = 'curated_repair_queue_v1'
FLAGS = {'semantic_approval': False, 'human_review_completed': False,
         'formal_training': False, 'training_eligible': False}
STAGES = ('media', 'reobserve', 'rewrite', 'styleunknown')


class SourceChanged(ValueError):
    pass


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
        finished = os.fstat(stream.fileno())
    after = path.stat()
    snapshots = [(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                 for s in (before, opened, finished, after)]
    if any(s != snapshots[0] for s in snapshots[1:]):
        raise SourceChanged('Artifact changed while hashing: ' + str(path))
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def identity(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked(ref):
    if sha(ref['path']) != ref['sha256']:
        raise SourceChanged('Source artifact SHA changed: ' + ref['path'])
    return load(ref['path'])


def save_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encoded(value)
    if path.exists():
        if path.read_bytes() != data:
            raise SourceChanged('Refusing changed immutable repair artifact: ' + str(path))
        return
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def replace_state(path, value):
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append(path, value):
    with Path(path).open('ab') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def safe_key(row):
    sid, split = row['id'], row['split']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', sid) or split not in ('train', 'val', 'test'):
        raise ValueError('Unsafe source ID/split')
    return split + '/' + sid


def artifact(ref, production):
    if not Path(ref['path']).resolve().is_relative_to(production):
        raise ValueError('Artifact escapes production directory')
    return checked(ref)


def body_diagnostic(path):
    if not path.exists():
        return {'body_status': 'absent'}, None
    raw = path.read_bytes()
    if not raw.strip():
        return {'body_status': 'empty', 'raw_ref': identity(path)}, None
    try:
        # Production responses are strict JSON. Malformed bodies are not repaired in-place.
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        return {'body_status': 'malformed', 'raw_ref': identity(path)}, None
    return {'body_status': 'present', 'raw_ref': identity(path)}, value


def observation_diagnostic(observation):
    result = {'window_id': observation['window']['window_id'], 'status': observation['status']}
    attempt = observation.get('attempt_path')
    if not attempt:
        result['body_status'] = 'absent'
        result['reason'] = 'no_model_attempt_or_insufficient_actual_frames'
        return result
    root = Path(attempt)
    body, raw = body_diagnostic(root / 'response.raw.txt')
    result.update(body)
    result['attempt_path'] = str(root)
    result['artifact_refs'] = {name: identity(root / name) for name in
        ('request.json', 'generation.json', 'receipt.json') if (root / name).is_file()}
    if (not isinstance(raw, dict) or not isinstance(raw.get('part_support'), list)
            or not isinstance(raw.get('events'), list)):
        result['reason'] = observation.get('error', body['body_status'] + '_response_body')
        return result
    supports = {r['part']: set(r['frame_indices']) for r in raw['part_support']
                if isinstance(r, dict) and isinstance(r.get('part'), str)
                and isinstance(r.get('frame_indices'), list)
                and all(type(i) is int for i in r['frame_indices'])}
    result['support_checklist'] = {part: sorted(indices) for part, indices in supports.items()}
    result['raw_event_count'] = len(raw.get('events', []))
    result['accepted_event_count'] = len(observation.get('json', {}).get('events', []))
    deficits = []
    for event in raw.get('events', []):
        if not isinstance(event, dict):
            continue
        if (not isinstance(event.get('frame_indices'), list)
                or not all(type(i) is int for i in event['frame_indices'])
                or not isinstance(event.get('parts'), list)
                or not all(isinstance(p, str) for p in event['parts'])):
            continue
        refs = set(event['frame_indices'])
        for part in event['parts']:
            local = supports.get(part, set())
            overlap = sorted(refs & local)
            if len(overlap) < 2:
                deficits.append({'raw_event_id': event.get('id'), 'part': part,
                    'event_frame_indices': sorted(refs), 'part_frame_indices': sorted(local),
                    'intersection': overlap, 'technical_reference_mismatch_possible': len(local) >= 2,
                    'current_part_support_insufficient': len(local) < 2,
                    'motion_truth_not_verified': True})
    result['reference_deficits'] = deficits
    result['reason'] = observation.get('error') or (
        'event_part_support_reference_mismatch_or_insufficiency' if deficits else
        'no_admitted_motion_events' if not result['accepted_event_count'] else 'supported_source_events')
    return result


def classify(row, production):
    key = safe_key(row)
    directory = production / 'sources' / key
    terminal_path = directory / 'consumer.terminal.json'
    if load(terminal_path) != row:
        raise SourceChanged('Completed hint differs from immutable consumer terminal: ' + key)
    refs = {'consumer_terminal': identity(terminal_path)}
    producer_path = directory / 'producer.terminal.json'
    producer = load(producer_path)
    refs['producer_terminal'] = identity(producer_path)
    for name in ('plan_ref', 'roi_ref'):
        if producer.get(name):
            artifact(producer[name], production)
            refs[name] = producer[name]
    failures, window_diagnostics = [], []
    if not row.get('joint_candidate_ref'):
        failures.append({'stage': 'media', 'reason': row.get('error') or producer.get('error') or
                         'consumer_failed_without_joint_result', 'original_status': row['status']})
        original_ref = refs['consumer_terminal']
        visual_root = None
    else:
        original_ref = row['joint_candidate_ref']
        joint = artifact(original_ref, production)
        refs['joint_result'] = original_ref
        visual_ref = joint.get('visual_result_ref')
        if not visual_ref:
            failures.append({'stage': 'media', 'reason': 'missing_visual_result_reference'})
            visual_root = None
        else:
            visual = artifact(visual_ref, production)
            refs['visual_result'] = visual_ref
            visual_root = str(Path(visual_ref['path']).parent.parent)
            for name, path in (('prepared', Path(visual_root) / 'prepared.json'),
                               ('prepared_receipt', Path(visual_root) / 'prepared.receipt.json')):
                if path.is_file():
                    refs[name] = identity(path)
            for name in ('raw_rgb_manifest', 'hand_roi', 'model_identity_ref'):
                ref = visual.get(name)
                if ref and ref.get('path') and ref.get('sha256'):
                    artifact(ref, production)
                    refs[name] = {'path': ref['path'], 'sha256': ref['sha256']}
            for observation in visual.get('window_observations', []):
                diagnostic = observation_diagnostic(observation)
                window_diagnostics.append(diagnostic)
                if observation['status'] != 'candidate' or not observation.get('json', {}).get('events'):
                    failures.append({'stage': 'reobserve', 'window_id': diagnostic['window_id'],
                                     'reason': diagnostic['reason'], 'body_status': diagnostic['body_status']})
                manifest = Path(visual_root) / 'windows' / diagnostic['window_id'] / 'manifest.json'
                if manifest.is_file():
                    diagnostic['manifest_ref'] = identity(manifest)
            final = visual.get('final', {})
            if final.get('status') == 'failed' or (not failures and not joint.get('all_windows_nonempty')):
                final_root = final.get('attempt_path')
                final_body, _ = body_diagnostic(Path(final_root) / 'response.raw.txt') if final_root else (
                    {'body_status': 'absent'}, None)
                failures.append({'stage': 'rewrite', 'reason': final.get('error', 'missing_or_partial_final_caption'),
                                 'attempt_path': final_root, **final_body})
                if final_root:
                    for name in ('request.json', 'response.raw.txt', 'generation.json', 'receipt.json'):
                        path = Path(final_root) / name
                        if path.is_file():
                            refs['rewrite_' + name] = identity(path)
        style = joint.get('style_result', {})
        if joint.get('dance_style_label') in (None, '', 'unknown') or style.get('status') not in ('candidate', 'resolved'):
            failures.append({'stage': 'styleunknown', 'reason': style.get('reason', 'style_unknown_or_failed'),
                             'original_status': style.get('status')})
        if joint.get('style_result_ref'):
            artifact(joint['style_result_ref'], production)
            refs['style_result'] = joint['style_result_ref']
    if not failures:
        failures.append({'stage': 'rewrite', 'reason': 'joint_incomplete_without_diagnostic'})
    stages = [stage for stage in STAGES if any(f['stage'] == stage for f in failures)]
    return {'schema': SCHEMA, 'id': row['id'], 'case_id': row['id'], 'source_key': key,
        'split': row['split'], 'batch_id': row.get('batch_id'), 'production': str(production),
        'original_status': row['status'], 'original_result_ref': original_ref,
        'original_result_sha256': original_ref['sha256'], 'completed_row_sha256': digest(row),
        'source_binding': producer.get('source_binding'), 'refs': refs, 'visual_root': visual_root,
        'failures': failures, 'stages': stages, 'stage': stages[0],
        'window_diagnostics': window_diagnostics, 'maximum_extra_source_attempts': 1,
        'maximum_reobserve_requests': sum(f['stage'] == 'reobserve' for f in failures),
        'maximum_rewrite_requests': int('rewrite' in stages or 'reobserve' in stages),
        'repair_policy': 'reobserve_failed_windows_then_one_strict_rewrite_no_reference_autofill',
        'independent_review_status': 'required_after_extra_attempt',
        'risk_review_case': True, 'repair_completed': False, **FLAGS}


def collect(production, out, revision):
    production, out = Path(production).resolve(), Path(out).resolve()
    if out.is_relative_to(production) or production.is_relative_to(out):
        raise ValueError('Repair output must be separate from production artifacts')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', revision):
        raise ValueError('Unsafe global revision')
    out.mkdir(parents=True, exist_ok=True)
    queue = production / 'completed.jsonl'
    state_path = out / 'collector.state.json'
    with (out / 'collector.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = load(state_path) if state_path.exists() else {
            'input_path': str(queue), 'offset': 0, 'seen': {}, 'counts': {}, 'stage_counts': {}}
        stat = queue.stat()
        input_identity = [stat.st_dev, stat.st_ino]
        if state['input_path'] != str(queue) or stat.st_size < state['offset'] or (
                'input_identity' in state and state['input_identity'] != input_identity):
            raise SourceChanged('Completed stream replaced/truncated')
        state['input_identity'] = input_identity
        added = 0
        initial_offset = checkpoint_offset = state['offset']
        with queue.open('rb') as stream:
            stream.seek(state['offset'])
            while True:
                begin = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    state['offset'] = begin
                    break
                row = json.loads(line)
                key, row_sha = safe_key(row), digest(row)
                previous = state['seen'].get(key)
                if previous and previous != row_sha:
                    raise SourceChanged('Source completed result revision changed: ' + key)
                if not previous:
                    complete = row.get('joint_complete') is True and row['status'] == 'joint_candidate'
                    if not complete:
                        job = classify(row, production)
                        job['global_revision'] = revision
                        job['queue_root'] = str(out)
                        job_path = out / 'jobs' / (key + '.json')
                        if job_path.exists():
                            existing = load(job_path)
                            if existing['completed_row_sha256'] != row_sha:
                                raise SourceChanged('Queued original source SHA changed')
                            job = existing
                        else:
                            save_once(job_path, job)
                        # Write-once announcements make a crash between save and state safe to resume.
                        notice = out / 'announced' / (key + '.json')
                        if not notice.exists():
                            append(out / 'ready.jsonl', {'source_key': key, 'job_ref': identity(job_path),
                                   'stage': job['stage'], 'stages': job['stages'], 'global_revision': job['global_revision']})
                            save_once(notice, {'job_ref': identity(job_path)})
                        added += 1
                        state['stage_counts'][job['stage']] = state['stage_counts'].get(job['stage'], 0) + 1
                    name = 'joint_complete' if complete else 'incomplete_or_failed'
                    state['counts'][name] = state['counts'].get(name, 0) + 1
                    state['seen'][key] = row_sha
                state['offset'] = stream.tell()
                if state['offset'] - checkpoint_offset >= 1024 * 1024:
                    replace_state(state_path, state)
                    checkpoint_offset = state['offset']
        if state['offset'] != initial_offset or not state_path.exists():
            replace_state(state_path, state)
        return {'schema': SCHEMA, 'sources_seen': len(state['seen']), 'new_queued_sources': added,
                'counts': state['counts'], 'primary_stage_counts': state['stage_counts'],
                'input_byte_offset': state['offset'], 'queue_path': str(out / 'ready.jsonl'),
                'GPU_dispatch_performed': False, **FLAGS}


def verify_job(job):
    for ref in job['refs'].values():
        verify_bytes(ref)
    checked(job['original_result_ref'])
    if job.get('refs', {}).get('demo_manifest'):
        manifest = checked(job['refs']['demo_manifest'])
        for frame in manifest['frames']:
            verify_bytes({'path': frame['file'], 'sha256': frame['sha256']})
    for diagnostic in job['window_diagnostics']:
        if diagnostic['body_status'] == 'absent' and diagnostic.get('attempt_path') and (
                Path(diagnostic['attempt_path']) / 'response.raw.txt').exists():
            raise SourceChanged('Previously absent original response body appeared')
        for name in ('manifest_ref', 'raw_ref'):
            if diagnostic.get(name):
                checked(diagnostic[name]) if name == 'manifest_ref' else verify_bytes(diagnostic[name])
        for ref in diagnostic.get('artifact_refs', {}).values():
            checked(ref)


def verify_bytes(ref):
    if sha(ref['path']) != ref['sha256']:
        raise SourceChanged('Original raw/evidence SHA changed: ' + ref['path'])


def reserve_attempt(job, out, revision):
    if job.get('queue_root') and Path(out).resolve() != Path(job['queue_root']).resolve():
        raise ValueError('Attempt ledger must remain in original queue root')
    verify_job(job)
    path = Path(out) / 'attempt-budget' / (job['source_key'] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = {'source_key': job['source_key'], 'original_result_sha256': job['original_result_sha256'],
             'global_revision': revision, 'extra_source_attempt': 1, 'maximum_extra_source_attempts': 1}
    try:
        with path.open('xb') as stream:
            stream.write(encoded(claim))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        previous = load(path)
        if previous['original_result_sha256'] != job['original_result_sha256']:
            raise SourceChanged('Reserved source original SHA changed')
        raise ValueError('Extra source attempt already reserved; no redispatch across revisions')
    return claim


def verify_references_by_content(value):
    if isinstance(value, dict):
        if value.get('path') and value.get('sha256'):
            verify_bytes(value)
        for item in value.values():
            verify_references_by_content(item)
    elif isinstance(value, list):
        for item in value:
            verify_references_by_content(item)


def verify_prepared_content(root, plan, bar, ffprobe):
    """Cross-mount verification compares SHA and freshly probed actual PTS, never mutates old stat."""
    root = Path(root)
    prepared = load(root / 'prepared.json')
    binding = load(root / 'binding.json')
    receipt = load(root / 'prepared.receipt.json')
    if (load(root / 'plan.json') != plan or prepared['binding'] != binding
            or receipt['binding'] != binding or receipt['prepared_sha256'] != sha(root / 'prepared.json')):
        raise SourceChanged('Prepared plan/receipt binding changed')
    raw = (root / 'plan.line.raw.json').read_bytes()
    if (hashlib.sha256(raw).hexdigest() != binding['raw_line_sha256']
            or bar.batch.digest(plan) != binding['full_source_row_sha256']):
        raise SourceChanged('Original plan raw SHA changed')
    verify_references_by_content(prepared['source_identity'])
    verify_references_by_content(prepared['source_refs'])
    verify_references_by_content(prepared['raw2d_mask'])
    video = prepared['source_identity']['files']['video_path']
    actual = bar.probe_actual(video['path'], ffprobe)
    old = prepared['probe']
    for name in ('times', 'source_timestamps'):
        if len(actual[name]) != len(old[name]) or any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
                                                       for a, b in zip(actual[name], old[name])):
            raise SourceChanged('Actual decoded video PTS differs')
    if actual['actual_video_pts_sha256'] != plan['binding']['actual_video_pts_sha256']:
        raise SourceChanged('Actual video PTS SHA differs')
    verify_bytes(video)
    union_path = root / 'rgb/manifest.json'
    if sha(union_path) != prepared['union_manifest_sha256']:
        raise SourceChanged('Original union RGB manifest changed')
    union = load(union_path)
    if union['binding'] != binding:
        raise SourceChanged('Original RGB binding changed')
    for frame in union['frames']:
        path = (root / 'rgb' / frame['file']).resolve()
        if not path.is_relative_to((root / 'rgb').resolve()):
            raise ValueError('Original RGB escapes bundle')
        verify_bytes({'path': str(path), 'sha256': frame['sha256']})
        i = frame['index']
        if not math.isclose(frame['time'], actual['times'][i], abs_tol=1e-8) or not math.isclose(
                frame['source_timestamp'], actual['source_timestamps'][i], abs_tol=1e-8):
            raise SourceChanged('RGB actual PTS binding differs')
    roi = prepared['hand_roi']
    if roi.get('available'):
        path = Path(roi['path'])
        for refpath, key in ((path, 'sha256'), (path.with_name('source_manifest.json'), 'source_manifest_sha256'),
                             (path.with_name('extract_receipt.json'), 'extract_receipt_sha256')):
            verify_bytes({'path': str(refpath), 'sha256': roi[key]})
        manifest = load(path)
        if manifest['id'] != plan['id'] or manifest['video_sha256'] != video['sha256']:
            raise SourceChanged('Original ROI source binding differs')
        rp = manifest['probe']
        for values, expected in ((rp['frame_times'], actual['times']),
                                 (rp['actual_video_pts'], actual['source_timestamps'])):
            if len(values) != len(expected) or any(not math.isclose(a, b, abs_tol=1e-8)
                                                   for a, b in zip(values, expected)):
                raise SourceChanged('ROI actual PTS differs')
        for crop in manifest['crops']:
            crop_path = Path(crop['path']).resolve()
            if not crop_path.is_relative_to(path.parent.resolve()):
                raise ValueError('ROI crop escapes original source')
            verify_bytes({'path': str(crop_path), 'sha256': crop['sha256']})
            x0, y0, x1, y1 = crop['crop_xyxy']
            if (not all(type(v) is int for v in (x0, y0, x1, y1))
                    or not (0 <= x0 < x1 <= rp['width'] and 0 <= y0 < y1 <= rp['height'])
                    or crop['source_frame_size'] != [rp['width'], rp['height']]
                    or crop['crop_pixel_size'] != [x1-x0, y1-y0]
                    or crop['ROI_to_source_affine'] != [[1., 0., x0], [0., 1., y0]]
                    or crop['source_to_ROI_affine'] != [[1., 0., -x0], [0., 1., -y0]]
                    or any(crop.get(k) is not False for k in
                           ('resized', 'rotated', 'mirrored', 'crop_adds_native_detail'))):
                raise SourceChanged('ROI native crop geometry differs')
    return prepared


def prepare_demo(root, result_path, window_id, extra_indices, out, revision, bar, args):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('Demo output must be separate from original visual source')
    result_path = Path(result_path).resolve()
    if result_path.parent != root / 'results':
        raise ValueError('Demo result must be explicit original source results artifact')
    original = load(result_path)
    plan = load(root / 'plan.json')
    prepared = verify_prepared_content(root, plan, bar, args.ffprobe)
    source_id = original['id']
    split = original['manifest_record'].get('split', 'train')
    key = safe_key({'id': source_id, 'split': split})
    diagnostic = observation_diagnostic(next(o for o in original['window_observations']
                                             if o['window']['window_id'] == window_id))
    manifest = copy.deepcopy(next(m for m in prepared['windows'] if m['window']['window_id'] == window_id))
    start, end = manifest['window']['start'], manifest['window']['end']
    old_indices = {f['index'] for f in manifest['frames']}
    extra_indices = sorted(set(extra_indices) - old_indices)
    for i in extra_indices:
        if type(i) is not int or not 0 <= i < len(prepared['probe']['times']) or not start <= prepared['probe']['times'][i] < end:
            raise ValueError('Extra frame outside original actual half-open window')
    hand = manifest.get('hand_roi_evidence')
    roi_count = sum(len(hand[p]['crops']) for p in ('left_hand', 'right_hand')) if hand else 0
    if len(old_indices) + len(extra_indices) + roi_count > 16:
        raise ValueError('Demo exceeds original 16-image budget')
    evidence_root = out / 'demo-evidence' / revision / key / window_id
    evidence_root.mkdir(parents=True, exist_ok=True)
    for frame in manifest['frames']:
        frame['file'] = str((root / 'rgb' / frame['file']).resolve())
    video_ref = prepared['source_identity']['files']['video_path']
    frames_receipt = evidence_root / 'extra.frames.json'
    if extra_indices and not frames_receipt.exists():
        expression = '+'.join('eq(n\\,' + str(i) + ')' for i in extra_indices)
        # Native full-resolution decode on CPU, no scale/interpolation or seek-relative indices.
        subprocess.run([args.ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-i', video_ref['path'],
            '-vf', 'select=' + expression, '-vsync', '0', '-q:v', '2',
            str(evidence_root / '%06d.jpg')], check=True, capture_output=True)
        files = sorted(evidence_root.glob('*.jpg'))
        if len(files) != len(extra_indices):
            raise ValueError('Native decoded extra frame count differs')
        extra = [{'index': i, 'time': prepared['probe']['times'][i],
            'source_timestamp': prepared['probe']['source_timestamps'][i],
            'file': str(path), 'sha256': sha(path)} for i, path in zip(extra_indices, files)]
        verify_bytes(video_ref)
        save_once(frames_receipt, {'indices': extra_indices, 'frames': extra,
            'video_ref': {'path': video_ref['path'], 'sha256': video_ref['sha256']},
            'actual_video_pts_sha256': plan['binding']['actual_video_pts_sha256'],
            'native_source_decode_CPU_only': True, 'resized': False})
    if extra_indices:
        extra = load(frames_receipt)
        if extra['indices'] != extra_indices:
            raise SourceChanged('Demo extra frame selection changed')
        for frame in extra['frames']:
            verify_bytes({'path': frame['file'], 'sha256': frame['sha256']})
        manifest['frames'].extend(extra['frames'])
        manifest['frames'].sort(key=lambda f: f['index'])
    manifest['sampling_policy'] = {'original_body_frames_preserved': sorted(old_indices),
        'extra_actual_source_frames': extra_indices, 'maximum_total_images': 16,
        'actual_body_images': len(manifest['frames']), 'actual_ROI_images': roi_count,
        'full_temporal_coverage_requested_not_continuous_video': True}
    new_manifest = evidence_root / 'manifest.json'
    save_once(new_manifest, manifest)
    diagnostic['manifest_ref'] = identity(root / 'windows' / window_id / 'manifest.json')
    refs = {'visual_result': identity(result_path), 'plan_ref': identity(root / 'plan.json'),
            'prepared': identity(root / 'prepared.json'), 'prepared_receipt': identity(root / 'prepared.receipt.json'),
            'binding': identity(root / 'binding.json'), 'raw_rgb_manifest': identity(root / 'rgb/manifest.json'),
            'demo_manifest': identity(new_manifest)}
    final_root = original.get('final', {}).get('attempt_path')
    if final_root:
        for name in ('request.json', 'response.raw.txt', 'generation.json', 'receipt.json'):
            path = Path(final_root) / name
            if path.exists():
                refs['rewrite_' + name] = identity(path)
    job = {'schema': SCHEMA, 'id': source_id, 'case_id': source_id, 'source_key': key,
        'split': split, 'batch_id': original['manifest_record'].get('batch_id', 'legacy-demo'),
        'production': str(root.parent.parent.parent), 'queue_root': str(out), 'global_revision': revision,
        'original_result_ref': identity(result_path), 'original_result_sha256': sha(result_path),
        'source_binding': plan['binding'], 'refs': refs, 'visual_root': str(root),
        'window_diagnostics': [diagnostic], 'stage': 'reobserve', 'stages': ['reobserve'],
        'failures': [{'stage': 'reobserve', 'window_id': window_id,
            'reason': 'user_review_rejects_uncertain_shape_only_motion_script_missing_required_hand_part'}],
        'maximum_extra_source_attempts': 1, 'maximum_reobserve_requests': 1, 'maximum_rewrite_requests': 1,
        'legacy_visual_demo': True, 'verification_policy': 'stable_contents_SHA_and_actual_PTS_not_mount_stat',
        'untouched_windows_AI_error_claim': 'notConfirmed_do_not_automatically_modify',
        'independent_review_status': 'required_after_extra_attempt', 'repair_completed': False, **FLAGS}
    job_path = out / 'jobs' / (key + '.json')
    save_once(job_path, job)
    verify_job(job)
    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
    request = reobserve_request(bar, args, root / 'rgb', manifest, forbidden, diagnostic, job_path, revision)
    preview_path = evidence_root / 'request.preview.json'
    save_once(preview_path, request)
    return {'job_ref': identity(job_path), 'request_preview_ref': identity(preview_path), 'body_frame_indices': [f['index'] for f in manifest['frames']],
            'ROI_images': roi_count, 'GPU_dispatch_performed': False, **FLAGS}


def reobserve_request(bar, args, rgb_root, manifest, forbidden, diagnostic, job_path, revision):
    request = bar.visual_request(args, rgb_root, manifest, forbidden)
    instruction = {'repair_stage': 'reobserve', 'allowed_source_frame_indices': [f['index'] for f in manifest['frames']],
        'prior_unreviewed_part_support_checklist': diagnostic.get('support_checklist', {}),
        'reference_deficits': diagnostic.get('reference_deficits', []),
        'instruction': 'Re-observe ONLY the original supplied pixels. Prior support is unreviewed, not a constraint or ground truth. '
            'Part support may include genuinely visible original body frames without a ROI. A ROI list is not all visible body frames. '
            'For each described phase, every required part needs two distinct event refs in its OWN visible support. '
            'Do not copy all frame refs, add unsupported parts, replace hand support by head/arm support, or infer motion from stillness. '
            'Separate supported arm/body motion from uncertain finger detail only when the pixels actually support that motion. '
            'Do not widen temporal events solely to pass reference counts. If evidence is insufficient, abstain. Return complete JSON.'}
    instruction.update({'original_raw_ref': diagnostic.get('raw_ref'),
        'original_event_ids': [e.get('id') for e in load(diagnostic['raw_ref']['path']).get('events', [])]
            if diagnostic.get('raw_ref') and diagnostic['body_status'] == 'present' else [],
        'support_policy': 'Coarse hand/wrist displacement can use two truly visible original full-body RGB frames even without ROI crops. '
            'Do not mechanically restrict hand part_support to ROI proposals. Arm/torso phases remain admissible on their own evidence. '
            'Named finger V/OK/heart/fist requires two truly clear native ROI frames, not cropped blurry fingers or forearms. '
            'If detail is unclear, omit the shape name entirely; all uncertainty stays in uncertainties metadata, never event descriptions. '
            'Each required part still needs two distinct actual event refs in its own support. Preserve only pixel-supported principal motion and hand location. '
            'Keep anatomical sides based on body context, not image left/right. Treat other windows and prior uncertainty as data, not new facts.'})
    request['messages'][1]['content'].append({'type': 'text', 'text': json.dumps(instruction, ensure_ascii=False)})
    request['repair_provenance'] = {'job_ref': identity(job_path), 'global_revision': revision,
                                   'original_context_and_pixels_preserved': True, 'automatic_reference_fill': False}
    return request


def validate_reobserve(value, manifest, forbidden, bar):
    for event in value.get('events', []):
        text = event.get('description', '')
        if re.search(r'\b(?:uncertain|unclear|unconfirmed|possibly|perhaps)\b|\bloose\s+V[- ]shape', text, re.I):
            raise ValueError('Uncertain detail belongs only in uncertainties metadata, not motion facts')
        if re.search(r'\b(?:V[- ](?:shape|sign)|OK|heart|fist)\b', text, re.I):
            hand = manifest.get('hand_roi_evidence') or {}
            parts = [p for p in event.get('parts', []) if p in ('left_hand', 'right_hand')]
            if not parts:
                raise ValueError('Named finger shape requires its actual hand evidence')
            for part in parts:
                crop_frames = {c['frame_index'] for c in hand.get(part, {}).get('crops', [])}
                if len(crop_frames & set(event.get('frame_indices', []))) < 2:
                    raise ValueError('Named finger shape requires two distinct actual native ROI frames')
    return bar.filter_supported_observation(value, manifest, forbidden)


def validate_repair_rewrite(value, windows, facts, boundaries, bar, frozen_windows):
    bar.check_supported_rewrite(value, windows, facts, boundaries)
    for item in value['windows']:
        if item['window_id'] in frozen_windows and item != frozen_windows[item['window_id']]:
            raise ValueError('Untargeted legacy demo window must remain byte-equivalent JSON fields')
    return value


def repair_visual(job_path, out, revision, bar, runtime, args):
    """One source attempt, only failed windows + one strict rewrite; never edits old artifacts.

    The caller must own GPU scheduling and supply the explicitly frozen bar module and runtime.
    Media/style failures stay in the queue; this API does not fabricate those labels.
    """
    job = load(job_path)
    if args.stage != 'run' or args.retry_failed or args.rewrite_reference_repair != 'off':
        raise ValueError('Require run stage, retry_failed=False, rewrite_reference_repair=off')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', revision):
        raise ValueError('Unsafe global revision')
    out = Path(out).resolve()
    production = Path(job['production']).resolve()
    if out.is_relative_to(production) or production.is_relative_to(out):
        raise ValueError('Repair output must be separate from production')
    if 'media' in job['stages'] or not set(job['stages']) & {'reobserve', 'rewrite'}:
        raise ValueError('Media/style-only cases require separately authorized dispatcher')
    verify_job(job)
    old_root = Path(job['visual_root'])
    plan = checked(job['refs']['plan_ref'])
    prepared = verify_prepared_content(old_root, plan, bar, args.ffprobe)
    original = checked(job['refs']['visual_result'])
    windows = [m['window'] for m in prepared['windows']]
    observations = copy.deepcopy(original['window_observations'])
    root = out / 'revisions' / revision / job['source_key']
    if root.resolve().is_relative_to(old_root.resolve()):
        raise ValueError('Repair root cannot be original visual root')
    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
    failed_ids = {f['window_id'] for f in job['failures'] if f['stage'] == 'reobserve'}
    items, positions = [], []
    for index, manifest in enumerate(prepared['windows']):
        wid = manifest['window']['window_id']
        if job.get('legacy_visual_demo') and wid in failed_ids:
            manifest = checked(job['refs']['demo_manifest'])
        if wid not in failed_ids or len(manifest['frames']) < 2:
            continue
        diagnostic = next(d for d in job['window_diagnostics'] if d['window_id'] == wid)
        request = reobserve_request(bar, args, old_root / 'rgb', manifest, forbidden, diagnostic, job_path, revision)
        item_root = root / 'windows' / wid
        items.append({'root': item_root, 'request': request,
            'validator': lambda value, m=manifest: validate_reobserve(value, m, forbidden, bar),
            'binding': {**original['binding'], 'repair_revision': revision, 'window': manifest['window'],
                        'frames_sha256': bar.batch.digest(manifest['frames'])}})
        positions.append(index)
    if items or original.get('facts'):
        runtime()  # A shared initialization failure must not consume a per-source budget.
    reserve_attempt(job, out, revision)
    save_once(root / 'original_job.json', job)
    results = bar.transactions(items, runtime, args)
    for index, item in zip(positions, items):
        result = results[str(item['root'])]
        if result['status'] == 'candidate' and result['json']['abstain']:
            result = {**result, 'status': 'abstain'}
        observations[index] = {**result, 'window': windows[index], 'attempt_path': str(item['root'])}
    facts = bar.facts_from_observations(windows, observations)
    # Existing boundaries are re-gated, never widened from new support or redispatched.
    boundary_results = copy.deepcopy(original.get('boundary_observations', []))
    specs = {s['id']: s for s in prepared['boundaries']}
    for result in boundary_results:
        if result.get('json', {}).get('relations'):
            relation = result['json']['relations'][0]
            bid = relation.get('id') or relation['boundary_id'].rsplit(':', 1)[0]
            if bid in specs:
                result['json'] = bar.gate_boundary_support(result['json'], specs[bid], windows, observations)
    boundaries = [r for b in boundary_results if b['status'] in ('candidate', 'unknown')
                  for r in b.get('json', {}).get('relations', [])]
    baseline = {'windows': [{'window_id': w['window_id'],
        'caption': ' '.join(f['raw_text'] for f in facts if f['window_id'] == w['window_id']),
        'used_fact_ids': [f['fact_id'] for f in facts if f['window_id'] == w['window_id']],
        'used_boundary_ids': []} for w in windows], 'issues': []}
    final = {'status': 'abstain', 'json': baseline, 'model_call_performed': False, **FLAGS}
    if any(o['status'] in ('failed', 'prepared') for o in observations):
        final = {'status': 'failed', 'error': 'reobservation_failed_or_pending', **FLAGS}
    if facts and not any(o['status'] in ('failed', 'prepared') for o in observations):
        supplied = {'windows': [{'window_id': w['window_id'], 'source_status': o['status'],
            'source_status_is_audit_only': True,
            'admitted_fact_count': sum(f['window_id'] == w['window_id'] for f in facts)}
            for w, o in zip(windows, observations)],
            'facts': [{k: copy.deepcopy(v) for k, v in f.items() if k != 'raw_text'} for f in facts]}
        request = bar.bar_rewrite_request(args, supplied, boundaries)
        frozen_windows = {w['window_id']: copy.deepcopy(w)
            for w in original.get('final', {}).get('json', {}).get('windows', [])
            if job.get('legacy_visual_demo') and w['window_id'] not in failed_ids}
        if frozen_windows:
            request['messages'].append({'role': 'user', 'content': json.dumps({
                'untargeted_windows_must_remain_exactly_unchanged': frozen_windows,
                'instruction': 'Only revise the target window against its new observed facts. Copy the other original window captions and reference fields exactly, not rephrase or relabel them.'}, ensure_ascii=False)})
        request['messages'][0]['content'] += '\nDo not name parts absent from local used facts. Windows without admitted facts MUST stay empty. No stillness filler. No automatic ID copying. Return one complete JSON.'
        request['prompt_sha256'] = hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest()
        request['repair_provenance'] = {'job_ref': identity(job_path), 'global_revision': revision,
            'original_rewrite_refs': {k: v for k, v in job['refs'].items() if k.startswith('rewrite_')},
            'maximum_additional_rewrite_calls': 1, 'automatic_reference_fill': False}
        item = {'root': root / 'rewrite', 'request': request,
            'validator': lambda value: validate_repair_rewrite(value, windows, facts, boundaries, bar, frozen_windows),
            'binding': {**original['binding'], 'repair_revision': revision,
                'facts_sha256': bar.batch.digest(facts), 'boundaries_sha256': bar.batch.digest(boundaries)}}
        final = bar.transactions([item], runtime, args)[str(item['root'])]
        final = {**final, 'attempt_path': str(item['root'])}
    verify_job(job)
    full = final['status'] == 'candidate' and len(final.get('json', {}).get('windows', [])) == len(windows) and all(
        w['caption'].strip() for w in final.get('json', {}).get('windows', []))
    result = {'schema': SCHEMA, 'id': job['id'], 'global_revision': revision,
        'original_job_ref': identity(job_path), 'original_result_ref': job['original_result_ref'],
        'original_source_binding': job['source_binding'], 'window_observations': observations,
        'facts': facts, 'boundary_observations': boundary_results, 'final': final,
        'all_windows_nonempty': full, 'structural_repair_candidate_complete': full and 'styleunknown' not in job['stages'],
        'repair_completed': False,
        'style_label_changed': False, 'independent_review_status': 'pending_independent_vlm_review',
        'new_model_request_budget': len(items) + 1, 'extra_source_attempts': 1,
        'original_raw_or_terminal_modified': False,
        'untouched_windows_AI_error_claim': job.get('untouched_windows_AI_error_claim', 'notReviewed'),
        'verification_policy': 'stable_contents_SHA_and_actual_PTS_not_mount_stat', **FLAGS}
    save_once(root / 'result.json', result)
    append(out / 'review-required.jsonl', {'id': job['id'], 'batch_id': job['batch_id'],
        'result_ref': identity(root / 'result.json'), 'risk_review_case': True, **FLAGS})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production')
    parser.add_argument('--demo-root')
    parser.add_argument('--demo-result')
    parser.add_argument('--window-id', default='w00001')
    parser.add_argument('--extra-frame-indices', nargs='*', type=int, default=[])
    parser.add_argument('--run-job')
    parser.add_argument('--consumer-config')
    parser.add_argument('--ms-source')
    parser.add_argument('--out', required=True)
    parser.add_argument('--global-revision', required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error('poll-seconds must be positive')
    if args.demo_root or args.run_job:
        if not args.consumer_config or not args.ms_source:
            parser.error('demo/run-job require consumer-config and explicitly frozen ms-source')
        if args.production or args.watch or (args.demo_root and args.run_job):
            parser.error('Do not combine collector/watch with demo/run-job')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(Path(args.ms_source).resolve()))
        import curated_pipeline as pipeline
        import ms_bar_annotation as bar
        if Path(bar.__file__).resolve().parent != Path(args.ms_source).resolve():
            parser.error('Imported bar differs from assigned frozen source')
        config = load(args.consumer_config)
        worker = pipeline.parser().parse_args(config['argv'][4:])
        worker.retry_failed = False
        worker.rewrite_reference_repair = 'off'
        if args.demo_root:
            if not args.demo_result:
                parser.error('demo-root requires explicit demo-result')
            result = prepare_demo(args.demo_root, args.demo_result, args.window_id,
                                  args.extra_frame_indices, args.out, args.global_revision, bar, worker)
        else:
            runtime_root = Path(args.out) / 'gpu-runtime' / args.global_revision
            runtime_root.mkdir(parents=True, exist_ok=True)
            runtime = bar.runtime_loader(worker, runtime_root, bar.code_identity())
            result = repair_visual(args.run_job, args.out, args.global_revision, bar, runtime, worker)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if not args.production:
        parser.error('collector requires production')
    while True:
        print(json.dumps(collect(args.production, args.out, args.global_revision), ensure_ascii=False), flush=True)
        if not args.watch:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
