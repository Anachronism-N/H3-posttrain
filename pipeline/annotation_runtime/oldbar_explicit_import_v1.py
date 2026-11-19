#!/usr/bin/env python3
"""CPU-only, single-pass import of immutable old bar MS as unapproved alternatives."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import copy
import csv
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import uuid

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
BAR = ROOT / 'diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1'
RT = BAR / 'runtime'
PLAN_SHA = 'ab7fd0921021d9b414b017dc06fc7111bc9df35d125bd1ef3aac8b762e392353'
MANIFEST_SHA = '50acb9c8ba20fd8550e333e90430e9efd3d4253d553bf79ac1439d4704465afe'
SUMMARY_SHA = 'a1eb129cf8557b4adeb4e11e00ace334ecb1575e43a587d665e33f09410e2bfd'
FLAGS = {'training_eligible': False, 'semantic_approval': False, 'human_review_completed': False,
         'formal_training': False, 'formal_approval': False, 'human_approved': False,
         'training_entry_updated': False, 'independent_review_required': True}
ARTIFACTS = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}


def need(value, message):
    if not value:
        raise ValueError(message)


def raw(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + chr(10)).encode()


def digest(value):
    return hashlib.sha256(raw(value)).hexdigest()


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def snapshot(path):
    s = Path(path).stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
            'device': s.st_dev, 'inode': s.st_ino, 'mode': s.st_mode}


def load(path):
    return json.loads(Path(path).read_bytes())


class Pending(ValueError):
    pass


class StableFiles:
    """Only explicitly shared immutable inputs get process-local stat/SHA reuse."""
    def __init__(self):
        self.cache = {}

    def sha(self, path, shared=False):
        path = Path(path).resolve(strict=True)
        before = snapshot(path)
        if shared and str(path) in self.cache:
            state, value = self.cache[str(path)]
            need(state == before, 'shared input changed during process: ' + str(path))
            return value
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        need(before == snapshot(path), 'file changed while fully hashing: ' + str(path))
        value = h.hexdigest()
        if shared:
            self.cache[str(path)] = (before, value)
        return value

    def ref(self, path, shared=False):
        return {'path': str(Path(path).resolve(strict=True)), 'sha256': self.sha(path, shared)}

    def verify(self, ref, shared=False):
        need(isinstance(ref, dict) and re.fullmatch('[0-9a-f]{64}', str(ref.get('sha256'))), 'invalid SHA reference')
        need(self.sha(ref['path'], shared) == ref['sha256'], 'content SHA mismatch: ' + ref['path'])
        return ref

    def jsonl(self, path, wanted=None, key='id', expected=None):
        path = Path(path).resolve(strict=True)
        before = snapshot(path)
        h, result, seen = hashlib.sha256(), {}, set()
        with path.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                h.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = row.get(key)
                need(isinstance(sid, str) and sid.casefold() not in seen, 'duplicate/invalid source ID: ' + str(path))
                seen.add(sid.casefold())
                if wanted is None or sid in wanted:
                    result[sid] = (row, number, line)
        value = h.hexdigest()
        need(before == snapshot(path), 'input changed while indexing: ' + str(path))
        need(expected is None or expected == value, 'input SHA differs from frozen contract: ' + str(path))
        self.cache[str(path)] = (before, value)
        return result


class Importer:
    def __init__(self, ids=None, cohort='oldbar-explicit-import-v1', ffprobe='/opt/ms_tools/ffprobe'):
        self.files = StableFiles()
        self.cohort, self.ffprobe = cohort, ffprobe
        need(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', cohort), 'unsafe import cohort')
        self.plan_path = BAR / 'plan_full12000.jsonl'
        self.plans = {sid: (row, line, raw_line.rstrip(bytes([13, 10])))
                      for sid, (row, line, raw_line) in self.files.jsonl(self.plan_path, expected=PLAN_SHA).items()}
        need(len(self.plans) == 12000, 'old plan population differs')
        wanted = set(ids) if ids else set(self.plans)
        need(wanted <= self.plans.keys(), 'requested ID absent from original old plan')
        self.ids = [sid for sid in self.plans if sid in wanted]
        self.manifest_path = BAR / 'curated222939_inputs_v1/manifest.jsonl'
        self.curated = self.files.jsonl(self.manifest_path, wanted, expected=MANIFEST_SHA)
        self.summary_path = BAR / 'curated222939_inputs_v1/manifest.summary.json'
        self.files.verify({'path': str(self.summary_path), 'sha256': SUMMARY_SHA}, True)
        self.summary = load(self.summary_path)
        need(len(self.summary['batches']) == 448 and self.summary['total'] == 222939, 'curated population/batches differ')
        self.batches = {r['batch_id']: r for r in self.summary['batches']}
        self.plan_order = {sid: i for i, sid in enumerate(self.plans)}
        self.batch_rows, self.style_rows, self.csv_rows = {}, None, {}
        self.bar = self.style = None

    def modules(self):
        if self.bar is None:
            source = RT / 'source_stable_roi_v3'
            sys.path.insert(0, str(source))
            self.bar = importlib.import_module('ms_bar_annotation')
            need(Path(self.bar.__file__).resolve().parent == source, 'foreign validation module')
            sys.path.insert(0, str(RT / 'source_curated_consumer_v1'))
            self.style = importlib.import_module('ms_dance_style')
        return self.bar

    def current_row(self, sid):
        if sid not in self.curated:
            raise ValueError('source_not_in_current_curated_manifest')
        row, line, raw_line = self.curated[sid]
        self.files.verify({'path': str(self.manifest_path), 'sha256': MANIFEST_SHA}, True)
        need(row['split'] == 'train' and row['id'] == row['source_id'] == sid, 'current source split/identity differs')
        need(line == row['source_ordinal'] + 1 and row['split_ordinal'] == row['source_ordinal'],
             'global manifest line/source/split ordinal differs')
        batch = self.batches[row['batch_id']]
        self.files.verify(batch, True)
        if row['batch_id'] not in self.batch_rows:
            self.batch_rows[row['batch_id']] = self.files.jsonl(batch['path'], expected=batch['sha256'])
        indexed = self.batch_rows[row['batch_id']]
        need(len(indexed) == batch['count'] and sid in indexed, 'curated batch count/membership differs')
        actual, batch_line, batch_raw = indexed[sid]
        need(actual == row and batch_raw == raw_line, 'original batch/global manifest row differs')
        need(batch['split'] == row['split'] and row['source_ordinal'] == batch['first_source_ordinal'] + batch_line - 1,
             'curated ordinal/batch binding differs')
        need(row['batch_id'] == 'train-b%06d' % (row['split_ordinal'] // self.summary['batch_size']), 'split ordinal batch differs')
        csv_ref = row['source_refs']['csv']
        source = self.summary['inputs']['train']
        need(csv_ref['path'] == source['path'] and csv_ref['sha256'] == source['sha256'], 'CSV ref differs from curated summary')
        self.files.verify(csv_ref, True)
        if source['path'] not in self.csv_rows:
            before = snapshot(source['path'])
            wanted = {r[0]['source_refs']['csv'].get('csv_line', r[0]['source_refs']['csv'].get('csv_record_number', 0) + 1)
                      for r in self.curated.values() if r[0]['source_refs']['csv']['path'] == source['path']}
            selected = {}
            with Path(source['path']).open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream)
                header = next(reader)
                need(header == source['header'], 'original CSV header differs')
                for number, values in enumerate(reader, 2):
                    if number in wanted:
                        selected[number] = values
            need(before == snapshot(source['path']), 'CSV changed while parsing')
            self.csv_rows[source['path']] = (header, selected)
        header, selected = self.csv_rows[source['path']]
        number = csv_ref.get('csv_line', csv_ref.get('csv_record_number', 0) + 1)
        values = selected.get(number)
        need(values == row['raw_csv_values'] and digest({'header': header, 'values': values}) == csv_ref['raw_row_sha256'],
             'original CSV record/row SHA differs')
        original = dict(zip(header, values))
        need(all(values[i] == sid for i, name in enumerate(header) if name == 'video_stem'), 'CSV SID differs')
        need(original['video'] == row['video_path'] and original['input_audio'] == row['audio_path']
             and original['source_video'] == row['source_group'], 'CSV source media/group differs')
        ref = {**batch, 'line': batch_line, 'raw_source_row_sha256': bytes_sha(batch_raw)}
        return row, ref, {'path': str(self.manifest_path), 'sha256': MANIFEST_SHA, 'line': line,
                          'raw_source_row_sha256': bytes_sha(raw_line), 'row_sha256': digest(row)}

    def historical_style(self, sid, plan):
        self.modules()
        if self.style_rows is None:
            self.style_rows = {}
            for shard in range(4):
                path = BAR.parent / 'vlm' / ('labels_s%d.jsonl' % shard)
                rows = self.files.jsonl(path, set(self.ids), key='sid')
                for key, (record, line, _) in rows.items():
                    need(key not in self.style_rows, 'duplicate historical style ID')
                    self.style_rows[key] = (record, {**self.files.ref(path, True), 'sid': key,
                                                       'line': line, 'row_sha256': self.bar.batch.digest(record)})
        if sid not in self.style_rows:
            raise Pending('historical_style_label_missing')
        record, ref = self.style_rows[sid]
        self.files.verify(ref, True)
        prepared = self.style.prepare_style(plan, label=record['label'], source_refs={'historical_label_record': ref},
                                            historical_video_sha256=None,
                                            historical_video_path=plan['manifest_record']['video_path'])
        if prepared['status'] != 'resolved' or prepared['source'] != 'historical_RGB_label':
            raise Pending('historical_style_not_resolved_genre')
        need(prepared['metadata']['historical_video_binding'] == 'path_only_unverified'
             and prepared['metadata']['historical_video_binding_verified'] is False, 'historical style binding upgraded')
        return prepared, ref

    def strong(self, path, expected=None, shared=False):
        ref = self.files.ref(path, shared)
        need(expected is None or ref['sha256'] == expected, 'content SHA mismatch: ' + str(path))
        return ref

    def phase(self, phase, root, binding, validator=None, allowed_images=None):
        root = Path(root)
        receipt_ref = self.strong(root / 'receipt.json')
        receipt = load(receipt_ref['path'])
        need(receipt['binding'] == binding and load(root / 'binding.json') == binding, phase + ' receipt binding differs')
        need(set(receipt['files_sha256']) == ARTIFACTS, phase + ' missing six-artifact receipt')
        refs = {name: self.strong(root / name, expected) for name, expected in receipt['files_sha256'].items()}
        request, final = load(root / 'request.json'), load(root / 'final.json')
        self.bar.core.validate_generation(root, request)
        need(receipt['status'] == final['status'] == 'candidate', phase + ' noncandidate receipt')
        if validator:
            parsed = self.bar.parse_response(self.bar.batch.json_bytes((root / 'response.raw.txt').read_text()))
            checked = validator(parsed)
            need(checked == final['json'] == load(root / 'response.json'), phase + ' raw/filter/final differs')
        if allowed_images is not None:
            found = []
            for message in request['messages']:
                if isinstance(message['content'], list):
                    for item in message['content']:
                        if item.get('type') == 'image_url':
                            url = item['image_url']['url']
                            need(url.startswith('data:image/jpeg;base64,'), 'nonlocal request image')
                            found.append(bytes_sha(base64.b64decode(url.split(',', 1)[1], validate=True)))
            need(found and all(v in allowed_images for v in found), phase + ' request image is not original source RGB/native ROI')
        return final, {'phase': phase, 'attempt_path': str(root), 'receipt': receipt_ref, 'artifacts': refs, 'binding': binding}

    @staticmethod
    def structural(result):
        windows = result.get('original_windows', [])
        captions = result.get('final', {}).get('json', {}).get('windows', [])
        observations = result.get('window_observations', [])
        return (result.get('status') == 'candidate' and result.get('final', {}).get('status') == 'candidate'
                and result.get('coverage', {}).get('full_window_caption') is True
                and result.get('trainable_weak_candidate') is True and bool(windows)
                and len(captions) == len(windows) == len(observations)
                and [w.get('window_id') for w in captions] == [o.get('window', {}).get('window_id') for o in observations]
                and all(isinstance(w.get('caption'), str) and w['caption'].strip() for w in captions)
                and all(o.get('status') == 'candidate' and o.get('json', {}).get('events') for o in observations)
                and all(any(f.get('window_id') == w.get('window_id') for f in result.get('facts', [])) for w in captions)
                and isinstance(result.get('final_timed_caption'), str) and bool(result['final_timed_caption'].strip()))

    def verify_evidence(self, result, plan, raw_line, source, current):
        bar = self.modules()
        binding = result['binding']
        need(binding['source_binding'] == plan['binding'] and binding['plan_sha256'] == PLAN_SHA
             and binding['full_source_row_sha256'] == bar.batch.digest(plan)
             and binding['raw_line_sha256'] == bytes_sha(raw_line)
             and binding['manifest_record_sha256'] == bar.batch.digest(plan['manifest_record']), 'old plan binding/row/raw-line differs')
        bar.validate_row(plan)
        need(load(source / 'plan.json') == plan and (source / 'plan.line.raw.json').read_bytes() == raw_line
             and load(source / 'binding.json') == binding, 'source original plan/binding differs')
        prepared_ref = self.strong(source / 'prepared.json')
        prepared = load(prepared_ref['path'])
        prepared_receipt = self.strong(source / 'prepared.receipt.json')
        receipt = load(prepared_receipt['path'])
        need(receipt['binding'] == prepared['binding'] == binding and receipt['prepared_sha256'] == prepared_ref['sha256'],
             'prepared SHA receipt differs')
        identity_ref = self.strong(source / 'source.identity.json')
        identity = load(identity_ref['path'])
        need(identity == prepared['source_identity'], 'original source identity differs')
        media = {}
        for key, asset in identity['files'].items():
            expected = plan['binding'][{'video_path': 'video_sha256', 'audio_path': 'audio_sha256', 'nlf_path': 'input_sha256'}[key]]
            need(asset['path'] == plan['manifest_record'][key] and asset['sha256'] == expected, 'old source identity/plan differs')
            media[key] = self.strong(asset['path'], expected)
            if key in ('video_path', 'audio_path'):
                need(current[key] == asset['path'], 'current media path differs from original source')
                self.strong(current[key], expected)
        need({'video_path', 'audio_path', 'nlf_path'} <= media.keys(), 'original media/NLF evidence missing')
        for ref in prepared['source_refs']:
            self.files.verify(ref, Path(ref['path']).name == 'manifest.jsonl')
        if prepared['raw2d_mask'].get('available'):
            self.files.verify(prepared['raw2d_mask'])
        pts_ref = self.strong(source / 'actual_pts.json')
        pts = load(pts_ref['path'])
        actual = bar.probe_actual(current['video_path'], self.ffprobe)
        need(actual == pts == prepared['probe'] and len(pts['times']) == plan['binding']['decoded_video_frames']
             and pts['actual_video_pts_sha256'] == plan['binding']['actual_video_pts_sha256'], 'current actual PTS differs from original')
        need(result['original_windows'] == plan['windows'] == prepared['original_windows']
             and result['original_window_manifest'] == plan['window_manifest'] == prepared['window_manifest'], 'original windows changed')
        need(result['hand_roi'] == prepared['hand_roi'], 'wrong original ROI reference')
        roi_meta = prepared['hand_roi']
        need(roi_meta['available'] is True, 'native ROI evidence unavailable')
        roi_path = Path(roi_meta['path'])
        self.strong(roi_path, roi_meta['sha256'])
        self.strong(roi_path.with_name('source_manifest.json'), roi_meta['source_manifest_sha256'])
        self.strong(roi_path.with_name('extract_receipt.json'), roi_meta['extract_receipt_sha256'])
        roi, _ = bar.load_hand_roi(plan, pts, str(roi_path))
        gate_ref = self.strong(source / 'roi_gate.json')
        gate = load(gate_ref['path'])
        terminal_ref = self.strong(BAR / 'hand_roi_full12000_v1/terminal' / (plan['id'] + '.json'))
        need(gate == load(terminal_ref['path']) and gate['status'] == 'ready' and gate['roi_manifest'] == str(roi_path), 'ROI terminal/gate differs')
        for path, expected in gate['files'].items():
            self.strong(path, expected)
        union_path = source / 'rgb/manifest.json'
        union_ref = self.strong(union_path, prepared['union_manifest_sha256'])
        need(result['raw_rgb_manifest'] == union_ref, 'original union RGB reference differs')
        union = load(union_path)
        need(union['binding'] == binding, 'RGB union source binding differs')
        image_hashes = set()
        for frame in union['frames']:
            i = frame['index']
            need(frame['time'] == pts['times'][i] and frame['source_timestamp'] == pts['source_timestamps'][i], 'RGB frame global index/PTS differs')
            path = (source / 'rgb' / frame['file']).resolve(strict=True)
            need(path.is_relative_to((source / 'rgb').resolve()), 'RGB frame escapes original source')
            image_hashes.add(self.strong(path, frame['sha256'])['sha256'])
        image_hashes.update(c['sha256'] for c in roi['crops'])
        forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
        observations, phases, window_manifest_refs = [], [], []
        windows = [m['window'] for m in prepared['windows']]
        need(windows == bar.normalized_windows(plan), 'prepared normalized original windows differ')
        union_by_index = {f['index']: f for f in union['frames']}
        need(len(windows) == len(result['original_windows']) == len(result['window_observations']), 'missing original window')
        for manifest, observation in zip(prepared['windows'], result['window_observations']):
            window = manifest['window']
            need(observation['window'] == window, 'window identity differs')
            need(all(f == union_by_index.get(f['index']) and window['start'] <= f['time'] < window['end']
                     for f in manifest['frames']), 'prepared window RGB/half-open PTS differs')
            request = load(Path(observation['attempt_path']) / 'request.json')
            expected_request = bar.visual_request(SimpleNamespace(model=request['model'],
                visual_max_tokens=request['sampling']['max_tokens']), source / 'rgb', manifest, forbidden)
            need(request == expected_request, 'original window request/RGB/nativeROI metadata differs')
            window_manifest_ref = self.strong(source / 'windows' / window['window_id'] / 'manifest.json')
            window_manifest_refs.append(window_manifest_ref)
            saved = load(window_manifest_ref['path'])
            expected = {**manifest, 'binding': binding, 'union_manifest_sha256': prepared['union_manifest_sha256'],
                        'raw2d_mask': prepared['raw2d_mask'], **bar.FLAGS}
            need(saved == expected, 'window prepared manifest differs')
            phase_binding = {**binding, 'window': window, 'frames_sha256': bar.batch.digest(manifest['frames'])}
            final, evidence = self.phase('window', observation['attempt_path'], phase_binding,
                lambda v, m=manifest: bar.filter_supported_observation(v, m, forbidden), image_hashes)
            need({k: v for k, v in observation.items() if k not in ('window', 'attempt_path')} == final, 'window observation differs from original final')
            observations.append({**final, 'window': window, 'attempt_path': observation['attempt_path']})
            phases.append(evidence)
        facts = bar.facts_from_observations(windows, observations)
        need(facts == result['facts'], 'raw observation re-filtered facts differ')
        boundaries = []
        need(len(prepared['boundaries']) == len(result['boundary_observations']), 'missing boundary observation')
        ordered_specs = [s for s in prepared['boundaries'] if not s['usable']] + [s for s in prepared['boundaries'] if s['usable']]
        for spec, observation in zip(ordered_specs, result['boundary_observations']):
            if not spec['usable']:
                metadata = {k: spec[k] for k in ('id', 'left_window_id', 'right_window_id', 'time')}
                expected = {'status': 'unknown', 'json': {'relations': [
                    {'part': part, 'relation': 'unknown', 'left_frame_indices': [], 'right_frame_indices': [],
                     **metadata, 'boundary_id': spec['id'] + ':' + part, 'ai_candidate': True, **bar.FLAGS}
                    for part in bar.core.PARTS],
                    'issues': ['nonadjacent or fewer than two actual frames per side']}, **bar.FLAGS}
                need(observation == expected, 'original no-model unknown boundary differs')
                boundaries.extend(observation['json']['relations'])
                continue
            root = source / 'boundaries' / spec['id']
            attempts = [p for p in root.glob('attempt-*') if (p / 'receipt.json').exists()]
            matches = [p for p in attempts if load(p / 'final.json') == observation]
            need(len(matches) == 1, 'boundary original attempt ambiguous/missing')
            metadata = {k: spec[k] for k in ('id', 'left_window_id', 'right_window_id', 'time')}
            def boundary_validator(value):
                checked = bar.core.check_boundary(value, spec['left'], spec['right'], metadata)
                for rel in checked['relations']:
                    need(rel['relation'] == 'unknown' or not set(rel['left_frame_indices'] + rel['right_frame_indices']) & forbidden.get(rel['part'], set()),
                         'boundary contains forbidden raw2d indices')
                return bar.gate_boundary_support(checked, spec, windows, observations)
            request = load(matches[0] / 'request.json')
            expected_messages = [
                {'role': 'system', 'content': bar.core.BOUNDARY_SYSTEM},
                {'role': 'user', 'content': bar.rgb_content(source / 'rgb',
                    [union_by_index[i] for i in spec['left'] + spec['right']],
                    {'boundary_time': spec['time'], 'left_frame_indices': spec['left'], 'right_frame_indices': spec['right']})}]
            need(request['messages'] == expected_messages, 'original boundary request/RGB/PTS differs')
            final, evidence = self.phase('boundary', matches[0], {**binding, 'boundary': spec}, boundary_validator, image_hashes)
            boundaries.extend(final['json']['relations'])
            phases.append(evidence)
        rewrite_binding = {**binding, 'facts_sha256': bar.batch.digest(facts), 'boundaries_sha256': bar.batch.digest(boundaries)}
        final_result = result['final']
        final_root = Path(final_result['attempt_path'])
        request_root = Path(final_result['reference_repair']['original_attempt_path']) if final_result.get('reference_repair') else final_root
        rewrite_request = load(request_root / 'request.json')
        supplied = {'windows': [{'window_id': w['window_id'], 'source_status': o['status'],
                    'source_status_is_audit_only': True,
                    'admitted_fact_count': sum(f['window_id'] == w['window_id'] for f in facts)}
                    for w, o in zip(windows, observations)],
                    'facts': [{k: copy.deepcopy(v) for k, v in f.items() if k != 'raw_text'} for f in facts]}
        expected_rewrite = bar.bar_rewrite_request(SimpleNamespace(model=rewrite_request['model'],
            text_max_tokens=rewrite_request['sampling']['max_tokens']), supplied, boundaries)
        need(rewrite_request == expected_rewrite, 'original rewrite request/facts/boundary IDs differs')
        if final_result.get('reference_repair'):
            repair = final_result['reference_repair']
            original = Path(repair['original_attempt_path'])
            for name, key in [('request.json', 'original_request_sha256'), ('response.raw.txt', 'original_response_sha256'), ('generation.json', 'original_generation_sha256')]:
                self.strong(original / name, repair[key])
            original_receipt = load(original / 'receipt.json')
            need(original_receipt['binding'] == rewrite_binding and load(original / 'binding.json') == rewrite_binding,
                 'reference repair original binding differs')
            original_final = load(original / 'final.json')
            need(original_final['status'] == original_receipt['status'] == 'failed'
                 and {k: v for k, v in final_result['original_attempt'].items() if k != 'attempt_path'} == original_final,
                 'reference repair original failed result differs')
            original_request = load(original / 'request.json')
            bar.core.validate_generation(original, original_request)
            repair_request = load(Path(final_result['attempt_path']) / 'request.json')
            need(repair_request['reference_repair'] == repair
                 and repair_request['messages'][:len(original_request['messages'])] == original_request['messages']
                 and repair_request['messages'][len(original_request['messages'])]['content'] == (original / 'response.raw.txt').read_text(),
                 'reference repair context/original raw changed')
            missing = {wid: refs for wid, refs in bar.missing_fact_refs(
                bar.parse_response(bar.batch.json_bytes((original / 'response.raw.txt').read_text())), facts).items() if refs}
            need(missing == repair['missing_fact_ids_by_window'], 'reference repair missing fact IDs differ')
            refs = {name: self.strong(original / name, expected) for name, expected in original_receipt['files_sha256'].items()}
            phases.append({'phase': 'rewrite_original_failed', 'attempt_path': str(original),
                           'receipt': self.strong(original / 'receipt.json'), 'artifacts': refs, 'binding': rewrite_binding})
            rewrite_binding['reference_repair_sha256'] = bar.batch.digest(repair)
        final, evidence = self.phase('rewrite', final_result['attempt_path'], rewrite_binding,
            lambda v: bar.check_supported_rewrite(v, windows, facts, boundaries))
        need(final['json'] == final_result['json'], 'rewrite original final differs')
        phases.append(evidence)
        text = bar.core.timed_text({'windows': windows}, final['json']['windows'])
        need(text == result['final_timed_caption'], 'timed MS changed byte-for-byte')
        model_ref = result['model_identity_ref']
        self.files.verify(model_ref)
        return {'old_source_identity': identity_ref, 'old_actual_pts': pts_ref,
                'old_binding': self.strong(source / 'binding.json'), 'old_prepared': prepared_ref,
                'old_prepared_receipt': prepared_receipt, 'old_source_refs': result['source_refs'],
                'old_window_manifests': window_manifest_refs,
                'old_source_plan_file': self.strong(source / 'plan.json'),
                'old_source_plan_rawline_file': self.strong(source / 'plan.line.raw.json'),
                'old_window_manifest': result['original_window_manifest'], 'old_original_windows': result['original_windows'],
                'old_normalized_final_windows': result['final']['json']['windows'], 'old_raw2d_mask': result['raw2d_mask'],
                'old_nlf': media['nlf_path'], 'NLF_provenance_only_not_model_input': True,
                'old_roi_gate': gate_ref, 'old_roi_terminal': terminal_ref, 'old_roi_ref': result['hand_roi'],
                'old_raw_rgb_manifest': union_ref, 'old_model_identity_ref': model_ref, 'old_generation_receipts': phases,
                'current_video': media['video_path'], 'current_audio': media['audio_path'],
                'current_media_content_SHA_matches_old_MS': True, 'current_actual_pts_matches_old_MS': True,
                'full_content_media_identity_verified_sources': 1, 'old_to_current_plan_equality_claimed': False}, load(model_ref['path'])['sha256']

    def import_one(self, sid):
        plan, line, raw_line = self.plans[sid]
        shard = self.plan_order[sid] % 4
        origin = BAR / 'full12000_stable_roi_v1' / ('shard%d' % shard)
        completed = origin / 'completed' / (sid + '.json')
        if not completed.exists():
            raise Pending('old_immutable_completed_missing')
        completed_ref = self.strong(completed)
        result = load(completed)
        if not self.structural(result):
            raise Pending('old_result_not_structurally_complete_nonempty_candidate')
        source = origin / 'sources' / sid
        result_ref = result['result_ref']
        need(Path(result_ref['path']).resolve().is_relative_to(source / 'results'), 'original result reference escapes source')
        self.files.verify(result_ref)
        original = load(result_ref['path'])
        need(all(result.get(k) == v for k, v in original.items()) and result['id'] == sid, 'completed record differs from original result')
        current, batch_ref, manifest_ref = self.current_row(sid)
        provenance, model_sha = self.verify_evidence(result, plan, raw_line, source, current)
        style, style_ref = self.historical_style(sid, plan)
        text, genre = result['final_timed_caption'], style['dance_style']
        provenance.update({'old_plan': {'path': str(self.plan_path), 'sha256': PLAN_SHA, 'line': line,
            'raw_line_sha256': bytes_sha(raw_line), 'full_source_row_sha256': self.bar.batch.digest(plan),
            'manifest_record_sha256': self.bar.batch.digest(plan['manifest_record'])},
            'old_completed_record': completed_ref, 'old_result': result_ref, 'historical_style_label_record': style_ref,
            'historical_video_binding': 'path_only_unverified', 'historical_video_sha256': None,
            'historical_video_SHA_not_invented': True, 'current_curated_summary': self.files.ref(self.summary_path, True),
            'current_curated_manifest': manifest_ref, 'current_curated_batch': batch_ref,
            'current_curated_csv': current['source_refs']['csv']})
        target = 'Dance style: ' + genre + chr(10) + text
        candidate = {'schema': 'old_bar_joint_explicit_alternative_import_v1', 'id': sid, 'source_id': sid,
            'split': current['split'], 'batch_id': self.cohort, 'video_path': current['video_path'], 'audio_path': current['audio_path'],
            'dance_style_label': genre, 'label': {'dance_style': genre, 'timed_motion_script': text}, 'target_text': target,
            'joint_complete': True, 'all_windows_nonempty': True, 'style_status': style['status'], 'visual_status': result['status'],
            'annotator_model_sha256': model_sha, 'source_binding': result['binding'],
            'source_refs': {'current_curated_csv': current['source_refs']['csv'], 'import_provenance': provenance},
            'style_result': style, 'raw_rgb_manifest': result['raw_rgb_manifest'], 'hand_roi': result['hand_roi'],
            'visual_result_ref': result_ref, 'batch_review_status': 'pending_independent_review',
            'edition_id': 'oldbar-explicit-alternative-v1', 'original_curated_batch_id': current['batch_id'],
            'origin_root': str(origin), 'source_hint': str(source),
            'candidate_origin': 'reused_existing_bar_MS_and_historical_pathweak_style',
            'population_status': 'same_curated_source_alternative_not_new_population', 'dedup_key': {'source_id': sid, 'split': current['split']},
            'label_status': 'completed', 'original_timed_motion_sha256': bytes_sha(text.encode()), 'target_text_sha256': bytes_sha(target.encode()),
            'original_timed_motion_unchanged': True, 'original_windows_unchanged': True, **FLAGS}
        validate_candidate_payload(candidate)
        return candidate


def validate_candidate_payload(candidate):
    meta = candidate['style_result']['metadata']
    provenance = candidate['source_refs']['import_provenance']
    need(meta['historical_video_sha256'] is None and provenance['historical_video_sha256'] is None
         and meta['historical_video_binding'] == provenance['historical_video_binding'] == 'path_only_unverified'
         and meta['historical_video_binding_verified'] is False, 'historical style SHA/binding must remain weak')
    need(all(candidate.get(k) is v for k, v in FLAGS.items()), 'candidate approval flags changed')
    text = candidate['label']['timed_motion_script']
    need(candidate['target_text'] == 'Dance style: ' + candidate['dance_style_label'] + chr(10) + text
         and candidate['original_timed_motion_sha256'] == bytes_sha(text.encode())
         and candidate['target_text_sha256'] == bytes_sha(candidate['target_text'].encode()), 'target/timed text SHA differs')
    need(candidate['batch_id'] != candidate['original_curated_batch_id'], 'import cohort cannot impersonate original 500 batch')


def immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raw(value)
    if path.exists():
        need(path.read_bytes() == data, 'same-source different candidate bytes refused: ' + str(path))
        return False
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)
    except FileExistsError:
        need(path.read_bytes() == data, 'same-source different candidate bytes refused: ' + str(path))
        return False
    finally:
        temp.unlink()
    return True


def atomic(path, data):
    path = Path(path)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def run(args):
    out = Path(args.out).resolve()
    protected = [BAR / 'full12000_stable_roi_v1', BAR / 'hand_roi_full12000_v1', BAR / 'hand_roi_gpu_fixed3_v3',
                 BAR / 'curated222939_inputs_v1', ROOT / 'OmniDance_processed', RT]
    need(not any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected), 'output overlaps original/frozen inputs')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.import.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        importer = Importer(args.ids.split(',') if args.ids else None, args.cohort, args.ffprobe)
        ids = importer.ids[:args.limit] if args.limit else importer.ids
        counts, dispositions = Counter(), []
        for sid in ids:
            try:
                candidate = importer.import_one(sid)
                written = immutable(out / 'candidates' / (sid + '.json'), candidate)
                status = 'imported' if written else 'already_imported'
                disposition = {'id': sid, 'status': status, 'candidate_ref': importer.files.ref(out / 'candidates' / (sid + '.json')), **FLAGS}
            except Pending as error:
                disposition = {'id': sid, 'status': 'pending', 'reason': str(error), **FLAGS}
            except FileNotFoundError as error:
                disposition = {'id': sid, 'status': 'pending', 'reason': 'evidence_file_missing: ' + str(error), **FLAGS}
            except Exception as error:
                disposition = {'id': sid, 'status': 'excluded', 'reason': type(error).__name__ + ': ' + str(error), **FLAGS}
            counts[disposition['status']] += 1
            dispositions.append(disposition)
            immutable(out / 'dispositions' / sid / (digest(disposition) + '.json'), disposition)
            progress = {'schema': 'oldbar_explicit_import_progress_v1', 'selected_source_count': len(ids),
                        'processed_source_count': len(dispositions), 'status_counts': dict(counts), 'last_source_id': sid, **FLAGS}
            atomic(out / 'progress.json', raw(progress))
            print(json.dumps({'id': sid, 'status': disposition['status'], 'reason': disposition.get('reason')}, ensure_ascii=False), flush=True)
        rejected_ids = {r['id'] for r in dispositions if r['status'] in ('pending', 'excluded')}
        candidates = [load(p) for p in sorted((out / 'candidates').glob('*.json'))
                      if p.stem not in rejected_ids] if (out / 'candidates').exists() else []
        for candidate in candidates:
            validate_candidate_payload(candidate)
        atomic(out / 'candidates.jsonl', b''.join(raw(c) for c in candidates))
        summary = {'schema': 'oldbar_explicit_import_receipt_v1', 'mode': 'single_pass_CPU_no_model',
                   'selected_ids': ids, 'status_counts': dict(counts), 'verified_this_pass_count': counts['imported'] + counts['already_imported'],
                   'candidate_count': len(candidates), 'retained_unselected_candidates_reverified_this_pass': False,
                   'current_failed_sources_not_exported': True, 'plan_sha256': PLAN_SHA, 'curated_manifest_sha256': MANIFEST_SHA,
                   'curated_summary_sha256': SUMMARY_SHA, 'code_ref': importer.files.ref(__file__),
                   'candidate_export_ref': importer.files.ref(out / 'candidates.jsonl'), 'dispositions': dispositions, **FLAGS}
        receipt = out / 'receipts' / (digest(summary) + '.json')
        immutable(receipt, summary)
        atomic(out / 'latest.receipt.json', raw(importer.files.ref(receipt)))
        print(json.dumps({'verified_this_pass_count': summary['verified_this_pass_count'], 'status_counts': dict(counts),
                          'receipt': str(receipt)}, ensure_ascii=False), flush=True)
        return 1 if counts['excluded'] else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=0, help='Maximum original-plan sources to inspect; 0 means full old plan')
    p.add_argument('--out', required=True, help='New isolated import output directory')
    p.add_argument('--once', action='store_true', help='Explicit single pass (also the default; never starts a daemon)')
    p.add_argument('--ids', help='Comma-separated old source IDs, in original plan order')
    p.add_argument('--cohort', default='oldbar-explicit-import-v1')
    p.add_argument('--ffprobe', default='/opt/ms_tools/ffprobe')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    need(args.limit >= 0, 'limit must be nonnegative')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
