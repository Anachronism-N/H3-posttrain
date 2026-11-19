#!/usr/bin/env python3
"""Single-pass derived source-deduplicated oldbar overlay; never a production terminal."""
import argparse
import base64
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

BAR = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
RT = BAR / 'runtime'
IMPORT_SHA = 'aadc2f7578c1e9d986c7dcee86f7f84297dcb34190c81dee1233a375fe53d66b'
SUMMARY_SHA = 'a1eb129cf8557b4adeb4e11e00ace334ecb1575e43a587d665e33f09410e2bfd'
PILOT_SHA = '98611de703011815aee9a586f500462565bbfb12d2a237af30486c231f4aaf17'
REVIEW_WORKER_SHA = 'f5507fe6b42f9d5bda22346d369bebf76c1455d14dea1234433ef16869cd23ef'
KNOWN_OLD_ALTERNATIVE_REJECTIONS = {'train/RAQoxqRxCfMvTHSx': {
    'source': 'explicit_user_review', 'oldbar_original_rejected': True,
    'reason': 'old second window retained uncertain hand shape but lost visible bilateral arm motion',
    'separate_scope_v3_repair_review_must_not_be_reused_for_old_label': True}}
FLAGS = dict(semantic_approval=False, training_eligible=False, training_approved=False, human_approved=False,
             human_review_completed=False, formal_approval=False, formal_training=False,
             training_entry_updated=False, full_pool_updated=False)
STATES = ('complete_candidate', 'partial_candidate', 'failed', 'ready_not_joint', 'terminal_without_result', 'missing')
SIX = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}


def need(value, message):
    if not value:
        raise ValueError(message)


def raw(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + chr(10)).encode()


def old_digest(value):
    return sha_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + chr(10)).encode())


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def key(row):
    sid, split = row.get('source_id', row.get('id')), row.get('split')
    need(isinstance(sid, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', sid) and split in ('train', 'val', 'test'), 'invalid source/split')
    return split + '/' + sid


def stat(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode]


def stable_read(path):
    before = stat(path)
    data = Path(path).read_bytes()
    need(before == stat(path), 'pending_unstable_snapshot: ' + str(path))
    return data, before


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def immutable_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'immutable snapshot bytes changed: ' + str(path))
        return
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            need(path.read_bytes() == data, 'immutable snapshot bytes changed: ' + str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_base_snapshot(out, evidence, base_progress, original_refs):
    data = evidence.base_snapshot_bytes
    need(len(data) == len(original_refs) == 2
         and all(sha_bytes(b) == ref['sha256'] for b, ref in zip(data, original_refs)),
         'immutable snapshot source bytes SHA mismatch')
    snapshot_key = sha_bytes(raw({'original_refs': original_refs}))
    root = Path(out) / 'snapshots' / snapshot_key
    refs = []
    for original, value in zip(original_refs, data):
        path = root / Path(original['path']).name
        immutable_bytes(path, value)
        refs.append({'path': str(path.resolve()), 'sha256': original['sha256']})
    meta_path = root / 'snapshot.json'
    if meta_path.exists():
        meta_bytes, _ = stable_read(meta_path)
        meta = json.loads(meta_bytes)
        need(meta['original_refs'] == original_refs and meta['immutable_refs'] == refs
             and meta['base_snapshot_only'] is True and meta['latest_live_base_claimed'] is False,
             'immutable snapshot metadata differs')
    else:
        meta = {'schema': 'curated_oldbar_immutable_base_snapshot_v1', 'original_refs': original_refs,
                'immutable_refs': refs, 'read_atUTC': evidence.base_snapshot_read_at,
                'updatedUTC': datetime.fromtimestamp(base_progress['updated_unix'], timezone.utc).isoformat(),
                'base_snapshot_only': True, 'latest_live_base_claimed': False}
        immutable_bytes(meta_path, raw(meta))
    meta_ref = {'path': str(meta_path.resolve()), 'sha256': sha_bytes(meta_path.read_bytes())}
    for ref in refs:
        evidence.check(ref, strong=True)
    return {**meta, 'snapshot_metadata_ref': meta_ref}


class Evidence:
    def __init__(self, state=None):
        self.state = state or {'files': {}, 'members': {}}
        self.state.setdefault('files', {})
        self.state.setdefault('members', {})
        self.docs = {}
        self.csv_values = {}
        self.line_documents = {}

    def check(self, ref, strong=False):
        need(isinstance(ref, dict) and Path(ref.get('path', '')).is_absolute()
             and re.fullmatch('[0-9a-f]{64}', str(ref.get('sha256'))), 'explicit absolute SHA reference required')
        path = Path(ref['path']).resolve(strict=True)
        before = stat(path)
        cached = self.state['files'].get(str(path))
        if not strong and cached and cached['stat'] == before:
            actual = cached['sha256']
        else:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            need(before == stat(path), 'pending_file_changed_while_hashing: ' + str(path))
            actual = h.hexdigest()
            self.state['files'][str(path)] = {'stat': before, 'sha256': actual}
        need(actual == ref['sha256'], 'content SHA mismatch: ' + str(path))
        return {'path': str(path), 'sha256': actual}

    def reference(self, path):
        data, _ = stable_read(path)
        ref = {'path': str(Path(path).resolve()), 'sha256': sha_bytes(data)}
        self.check(ref)
        return ref

    def document(self, ref):
        checked = self.check(ref)
        cache_key = (checked['path'], checked['sha256'])
        if cache_key not in self.docs:
            data, _ = stable_read(checked['path'])
            need(sha_bytes(data) == checked['sha256'], 'document changed while reading')
            self.docs[cache_key] = json.loads(data)
        return self.docs[cache_key]

    def visit(self, value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                self.check(value)
            for child in value.values():
                self.visit(child)
        elif isinstance(value, list):
            for child in value:
                self.visit(child)

    def membership(self, descriptor):
        self.check(descriptor)
        bid = descriptor['batch_id']
        cached = self.state['members'].get(bid)
        if not cached or cached['sha256'] != descriptor['sha256']:
            data, _ = stable_read(descriptor['path'])
            need(sha_bytes(data) == descriptor['sha256'], 'original batch changed while reading')
            rows = {}
            for number, line in enumerate(data.splitlines(keepends=True), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = key(row)
                need(identity not in rows and row['batch_id'] == bid and row['split'] == descriptor['split'], 'duplicate/wrong batch member')
                rows[identity] = {'row': row, 'line': number, 'raw_sha256': sha_bytes(line)}
            need(len(rows) == descriptor['count'], 'original batch denominator differs')
            cached = {'sha256': descriptor['sha256'], 'rows': rows}
            self.state['members'][bid] = cached
        return cached['rows']

    def line(self, ref):
        self.check(ref)
        path = ref['path']
        if path not in self.line_documents:
            data, _ = stable_read(path)
            need(sha_bytes(data) == ref['sha256'], 'original row file changed while reading')
            self.line_documents[path] = data.splitlines(keepends=True)
        number = ref['line']
        need(type(number) is int and 1 <= number <= len(self.line_documents[path]), 'original row line outside file')
        data = self.line_documents[path][number - 1]
        return data, json.loads(data)

    def prepare_csv(self, candidates, summary):
        wanted = {}
        for candidate in candidates:
            ref = candidate['source_refs']['import_provenance']['current_curated_csv']
            number = ref.get('csv_line', ref.get('csv_record_number', 0) + 1)
            wanted.setdefault(ref['path'], set()).add(number)
        for path, numbers in wanted.items():
            descriptor = next(r for r in summary['inputs'].values() if r['path'] == path)
            self.check(descriptor)
            before = stat(path)
            values = {}
            with Path(path).open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream)
                header = next(reader)
                need(header == descriptor['header'], 'current CSV header differs')
                for number, row in enumerate(reader, 2):
                    if number in numbers:
                        values[number] = row
            need(before == stat(path), 'pending_CSV_changed_while_reading')
            self.csv_values[path] = (header, values)


def read_base(root, evidence, summary):
    root = Path(root)
    paths = [root / 'progress.json', root / 'candidates.jsonl']
    before = [stat(p) for p in paths]
    read_at = datetime.now(timezone.utc).isoformat()
    progress_before = paths[0].read_bytes()
    candidate_bytes = paths[1].read_bytes()
    progress_after = paths[0].read_bytes()
    need(progress_before == progress_after, 'pending_snapshot_mismatch_progress_generation')
    data = [progress_before, candidate_bytes]
    need(before == [stat(p) for p in paths], 'pending_unstable_base_snapshot')
    second = [p.read_bytes() for p in paths]
    need(before == [stat(p) for p in paths] and [sha_bytes(b) for b in data] == [sha_bytes(b) for b in second],
         'pending_unstable_base_snapshot')
    progress = json.loads(data[0])
    rows = [json.loads(line) for line in data[1].splitlines() if line.strip()]
    need(progress['authoritative_manifest']['sha256'] == SUMMARY_SHA
         and progress['expected_sources'] == summary['total'] and progress['expected_batches'] == len(summary['batches']), 'base population identity differs')
    need(len({key(r) for r in rows}) == len(rows), 'base duplicate source keys')
    state_counts = Counter(r['state'] for r in rows)
    need(all(state_counts[s] == progress['counts'].get(s, 0) for s in STATES if s != 'missing')
         and progress['counts']['missing'] == summary['total'] - len(rows), 'pending_base_cross_snapshot_count_mismatch')
    split_counts = {s: Counter() for s in summary['counts']}
    batch_counts = {d['batch_id']: Counter() for d in summary['batches']}
    for row in rows:
        split_counts[row['split']][row['state']] += 1
        need(row['batch_id'] in batch_counts, 'base unknown original batch')
        batch_counts[row['batch_id']][row['state']] += 1
    for split, expected in summary['counts'].items():
        split_counts[split]['missing'] = expected - sum(split_counts[split].values())
        want = progress['split_counts'].get(split, {})
        need(all(split_counts[split].get(s, 0) == want.get(s, 0) for s in STATES),
             'pending_snapshot_mismatch_split_counts')
    base_batches = {b['batch_id']: b for b in progress['batches']}
    need(len(base_batches) == len(summary['batches']), 'pending_snapshot_mismatch_batch_population')
    for descriptor in summary['batches']:
        bid = descriptor['batch_id']
        batch_counts[bid]['missing'] = descriptor['count'] - sum(batch_counts[bid].values())
        want = base_batches[bid]
        need(want['expected'] == descriptor['count'] and want['split'] == descriptor['split']
             and all(batch_counts[bid].get(s, 0) == want['counts'].get(s, 0) for s in STATES),
             'pending_snapshot_mismatch_original_batch_counts')
    need(sum(bool(r.get('ready_alternatives')) for r in rows) == progress['counts']['ready_unique'],
         'pending_snapshot_mismatch_ready_unique')
    descriptors = {d['batch_id']: d for d in summary['batches']}
    members_by_batch = {}
    for row in rows:
        need(row['batch_id'] in descriptors, 'base original batch unknown')
        if row['batch_id'] not in members_by_batch:
            members_by_batch[row['batch_id']] = evidence.membership(descriptors[row['batch_id']])
        members = members_by_batch[row['batch_id']]
        member = members.get(key(row), {}).get('row')
        need(member and all(row[k] == member[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')),
             'base source original membership differs')
    evidence.base_snapshot_bytes = data
    evidence.base_snapshot_read_at = read_at
    return progress, rows, [{'path': str(p.resolve()), 'sha256': sha_bytes(b)} for p, b in zip(paths, data)]


def completed_imports(root, evidence):
    root = Path(root).resolve()
    found, diagnostics = {}, []
    for directory in sorted((root / 'dispositions').glob('*')):
        entries = []
        for path in sorted(directory.glob('*.json')):
            try:
                ref = evidence.reference(path)
                record = evidence.document(ref)
                need(sha_bytes(raw(record)) == path.stem, 'immutable disposition content-address differs')
                need(record['id'] == directory.name, 'disposition/source directory differs')
                entries.append((record, ref, stat(path)))
            except Exception as error:
                diagnostics.append({'source_id': directory.name, 'state': 'excluded', 'reason': str(error), 'path': str(path)})
        successes = [(r, f, s) for r, f, s in entries if r['status'] in ('imported', 'already_imported')]
        if not successes:
            diagnostics.extend({'source_id': r['id'], 'state': r['status'], 'reason': r.get('reason'), 'disposition_ref': ref}
                               for r, ref, _ in entries)
            continue
        record, disposition_ref, success_stat = max(successes, key=lambda v: v[2][3])
        later_failures = [(r, f) for r, f, s in entries if r['status'] in ('pending', 'excluded') and s[3] >= success_stat[3]]
        if later_failures:
            diagnostics.append({'source_id': record['id'], 'state': 'pending', 'reason': 'later_nonadmitted_disposition',
                                'records': later_failures})
            continue
        try:
            candidate_ref = record['candidate_ref']
            need(Path(candidate_ref['path']).resolve() == root / 'candidates' / (record['id'] + '.json'), 'candidate canonical import path differs')
            candidate = evidence.document(candidate_ref)
            need(candidate['id'] == record['id'] and candidate['source_id'] == record['id'], 'completed disposition candidate ID differs')
            identity = key(candidate)
            need(identity not in found or found[identity]['candidate_ref'] == candidate_ref, 'same source different candidate bytes')
            found[identity] = {'candidate': candidate, 'candidate_ref': candidate_ref, 'disposition_ref': disposition_ref,
                               'disposition': record, 'importer_code_ref': {'path': str(RT / 'oldbar_explicit_import_v1.py'), 'sha256': IMPORT_SHA}}
        except Exception as error:
            diagnostics.append({'source_id': record['id'], 'state': 'excluded', 'reason': str(error), 'disposition_ref': disposition_ref})
    disposed_ids = {Path(r['disposition_ref']['path']).parent.name for r in found.values()}
    disposed_ids.update(directory.name for directory in (root / 'dispositions').glob('*'))
    for path in sorted((root / 'candidates').glob('*.json')):
        if path.stem not in disposed_ids:
            diagnostics.append({'source_id': path.stem, 'state': 'pending',
                                'reason': 'candidate_written_but_completed_disposition_missing', 'path': str(path)})
    return found, diagnostics


def validate_candidate(record, evidence, summary):
    row = record['candidate']
    evidence.check(record['candidate_ref'])
    evidence.check(record['disposition_ref'])
    evidence.check(record['importer_code_ref'])
    need(row['schema'] == 'old_bar_joint_explicit_alternative_import_v1' and row['joint_complete'] is True
         and row['all_windows_nonempty'] is True and row['visual_status'] == 'candidate', 'oldbar not complete explicit candidate')
    need(all(row.get(k) is False for k in FLAGS if k in row) and row['independent_review_required'] is True,
         'candidate training/approval claims forbidden')
    p = row['source_refs']['import_provenance']
    evidence.visit(p)
    descriptor = next((d for d in summary['batches'] if d['batch_id'] == row['original_curated_batch_id']), None)
    need(descriptor and all(p['current_curated_batch'][k] == descriptor[k] for k in descriptor), 'original curated descriptor differs')
    entry = evidence.membership(descriptor).get(key(row))
    need(entry and entry['line'] == p['current_curated_batch']['line']
         and entry['raw_sha256'] == p['current_curated_batch']['raw_source_row_sha256'], 'current original member/line SHA differs')
    member = entry['row']
    need(row['split'] == member['split'] == 'train' and row['batch_id'] != member['batch_id']
         and all(row[k] == member[k] for k in ('video_path', 'audio_path')), 'current curated split/media/import cohort differs')
    need(p['current_curated_csv'] == member['source_refs']['csv'], 'current CSV source ref differs')
    csv_ref = p['current_curated_csv']
    header, csv_values = evidence.csv_values[csv_ref['path']]
    number = csv_ref.get('csv_line', csv_ref.get('csv_record_number', 0) + 1)
    values = csv_values.get(number)
    canonical = {'header': header, 'values': values}
    need(values == member['raw_csv_values'] and sha_bytes(raw(canonical)) == csv_ref['raw_row_sha256'], 'current original CSV row differs')
    result = evidence.document(p['old_result'])
    completed = evidence.document(p['old_completed_record'])
    need(completed['result_ref'] == p['old_result'] == row['visual_result_ref'] and completed['trainable_weak_candidate'] is True
         and all(completed.get(k) == v for k, v in result.items()), 'original immutable completed/result differs')
    need(result['id'] == row['id'] and result['binding'] == row['source_binding']
         and row['label']['timed_motion_script'] == result['final_timed_caption']
         and p['old_original_windows'] == result['original_windows']
         and p['old_normalized_final_windows'] == result['final']['json']['windows'], 'original timed MS/windows/binding differs')
    need(row['hand_roi'] == result['hand_roi'] == p['old_roi_ref'] and row['raw_rgb_manifest'] == result['raw_rgb_manifest'], 'original RGB/native ROI differs')
    need(Path(p['old_result']['path']).parent == Path(row['source_hint']) / 'results'
         and Path(row['source_hint']) == Path(row['origin_root']) / 'sources' / row['id'], 'real origin/source namespace differs')
    need(row['target_text'] == 'Dance style: ' + row['label']['dance_style'] + chr(10) + row['label']['timed_motion_script']
         and row['target_text_sha256'] == sha_bytes(row['target_text'].encode()), 'explicit semantic target differs')
    meta = row['style_result']['metadata']
    need(meta['historical_video_sha256'] is None and meta['historical_video_binding'] == 'path_only_unverified'
         and meta['historical_video_binding_verified'] is False and p['historical_video_sha256'] is None
         and meta['source_refs']['historical_label_record'] == p['historical_style_label_record'], 'historical style binding upgraded')
    historical_raw, historical = evidence.line(p['historical_style_label_record'])
    need(historical['sid'] == row['id'] and old_digest(historical) == p['historical_style_label_record']['row_sha256']
         and historical['label'] == meta['historical_label'] and historical['label']['genre'] == row['label']['dance_style'],
         'historical style original row/label differs')
    plan_raw, plan = evidence.line(p['old_plan'])
    need(sha_bytes(plan_raw.rstrip(bytes([13, 10]))) == p['old_plan']['raw_line_sha256']
         and old_digest(plan) == p['old_plan']['full_source_row_sha256']
         and plan['binding'] == row['source_binding']['source_binding'] and plan['manifest_record'] == result['manifest_record'],
         'original oldplan raw row/source binding differs')
    identity = evidence.document(p['old_source_identity'])
    for name, suffix in [('video_path', 'video'), ('audio_path', 'audio')]:
        ref = p['current_' + suffix]
        need(ref['path'] == row[name] == identity['files'][name]['path']
             and ref['sha256'] == identity['files'][name]['sha256'] == row['source_binding']['source_binding'][suffix + '_sha256'],
             'current media/old source binding differs')
        evidence.check(ref, strong=True)
    for phase in p['old_generation_receipts']:
        receipt = evidence.document(phase['receipt'])
        need(receipt['binding'] == phase['binding'], 'imported phase binding differs')
        for name, ref in phase['artifacts'].items():
            need(receipt['files_sha256'][name] == ref['sha256'], 'imported phase artifact SHA differs')
    return member


def checkpoint(identity):
    weights = sorted(v['sha256'] for n, v in identity['files'].items() if Path(n).suffix in ('.safetensors', '.bin', '.pt', '.gguf'))
    need(weights, 'checkpoint has no real weight inventory')
    return old_digest({'weight_file_content_sha256': weights})


def frozen_review_row(row, manifest):
    return {'source_id': row['id'], 'id': row['id'], 'split': row['split'], 'video_path': row['video_path'],
            'audio_path': row['audio_path'], 'label': row['label'], 'label_status': 'completed', 'hand_roi': row['hand_roi'],
            'annotator_model_sha256': row['annotator_model_sha256'], 'explicit_review_binding': {
                'schema': 'oldbar_explicit_independent_review_v1', 'manifest': manifest,
                'import_candidate_sha256': old_digest(row), 'import_batch_id': row['batch_id'],
                'original_curated_batch_id': row['original_curated_batch_id'], 'label_sha256': old_digest(row['label']),
                'target_text_sha256': row['target_text_sha256'], 'source_binding_sha256': old_digest(row['source_binding']),
                'provenance_sha256': old_digest(row['source_refs']), 'historical_video_sha256': None,
                'historical_video_binding': 'path_only_unverified'}}


class Reviews:
    def __init__(self, root, evidence):
        self.root, self.evidence = Path(root), evidence
        self.rows, self.diagnostics = {}, []
        if not (self.root / 'independent.ledger.jsonl').exists():
            self.diagnostics.append({'state': 'pending_review_ledger_missing'})
            return
        population = evidence.document(evidence.reference(self.root / 'frozen.population.json'))
        self.manifest = population['manifest']
        need(self.manifest['sha256'] == PILOT_SHA and self.manifest['source_count'] == 3, 'old review frozen pilot identity differs')
        evidence.check(self.manifest)
        data, _ = stable_read(self.manifest['path'])
        self.pilot = {key(r): r for r in [json.loads(x) for x in data.splitlines() if x.strip()]}
        self.population = population
        need(population['candidate_rows_sha256'] == [old_digest(r) for r in self.pilot.values()], 'frozen pilot row digests differ')
        ledger_ref = evidence.reference(self.root / 'independent.ledger.jsonl')
        ledger_data, _ = stable_read(ledger_ref['path'])
        for line in ledger_data.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identity = key(row)
            try:
                need(identity not in self.rows and identity in self.pilot, 'review duplicate/unknown source')
                self.rows[identity] = self.verify(row, ledger_ref)
            except Exception as error:
                self.diagnostics.append({'source_key': identity, 'state': 'pending_review_invalid', 'reason': str(error)})

    def verify(self, ledger, ledger_ref):
        e = self.evidence
        pilot = self.pilot[key(ledger)]
        label_sha = old_digest(pilot['label'])
        input_row = frozen_review_row(pilot, self.manifest)
        need(ledger['manifest_sha256'] == PILOT_SHA and ledger['import_candidate_sha256'] == old_digest(pilot)
             and ledger['input_row_sha256'] == old_digest(input_row) and ledger['label_sha256'] == label_sha,
             'review ledger original input/semantic label differs')
        root = Path(ledger['evidence_ref']['path']).parent
        evidence = e.document(ledger['evidence_ref'])
        binding = evidence['binding']
        need(binding['input_row_sha256'] == ledger['input_row_sha256'] and binding['label_sha256'] == label_sha
             and binding['source_id'] == pilot['id'] and binding['split'] == pilot['split'], 'review evidence input binding differs')
        for name, suffix in [('video_path', 'video'), ('audio_path', 'audio')]:
            ref = evidence['source_identity']['files'][name]
            need(ref['path'] == pilot[name] and ref['sha256'] == pilot['source_binding']['source_binding'][suffix + '_sha256'], 'review current media differs')
            e.check(ref, strong=True)
        original_roi = e.document(pilot['hand_roi'])
        original_crops = {c['path']: c for c in original_roi['crops']}
        indices = [f['index'] for f in evidence['frames']]
        need(indices == binding['frame_indices'] and len(set(indices)) == len(indices) == 16
             and len(evidence['hand_rois']) == 6 and ledger['evidence_frames'] == 16 and ledger['native_hand_rois'] == 6,
             'review native evidence count/index differs')
        image_hashes = []
        for frame in evidence['frames']:
            path = (root / frame['file']).resolve()
            need(path.is_relative_to(root.resolve()) and frame['time'] == evidence['probe']['times'][frame['index']]
                 and frame['source_timestamp'] == evidence['probe']['source_timestamps'][frame['index']], 'review RGB/native PTS differs')
            image_hashes.append(e.check({'path': str(path), 'sha256': frame['sha256']})['sha256'])
            for crop in evidence['hand_rois']:
                if crop['frame_index'] == frame['index']:
                    need(crop == original_crops.get(crop['path']) and crop['frame_index'] in indices, 'review ROI not original same-frame native crop')
                    image_hashes.append(e.check(crop)['sha256'])
        receipt = e.document(ledger['review_receipt'])
        transaction = Path(ledger['review_receipt']['path']).parent
        need(receipt['status'] == 'candidate' and set(receipt['files_sha256']) == SIX, 'review requires native six artifacts')
        artifacts = {name: e.check({'path': str(transaction / name), 'sha256': expected}) for name, expected in receipt['files_sha256'].items()}
        bound = e.document(artifacts['binding.json'])
        need(bound == receipt['binding'] and all(bound[k] == binding[k] for k in binding)
             and bound['evidence_sha256'] == ledger['evidence_ref']['sha256'], 'review receipt/evidence binding differs')
        request, generation = e.document(artifacts['request.json']), e.document(artifacts['generation.json'])
        claim = json.loads(request['messages'][1]['content'][0]['text'])
        need(claim['label_to_test'] == pilot['label'] and claim['allowed_source_frame_indices'] == indices, 'review request semantic/native frames differ')
        requested = [sha_bytes(base64.b64decode(c['image_url']['url'].split(',', 1)[1], validate=True))
                     for c in request['messages'][1]['content'] if c['type'] == 'image_url']
        need(requested == image_hashes, 'review request original native RGB/ROI bytes differ')
        need(generation['request_sha256'] == artifacts['request.json']['sha256']
             and generation['raw_response_sha256'] == artifacts['response.raw.txt']['sha256']
             and generation['structured_output_schema_sha256'] == old_digest(request['json_schema'])
             and generation['model_identity_sha256'] == bound['model_sha256'] and generation['model_call_performed'] is True
             and generation['finish_reason'] == 'stop' and 0 < generation['completion_tokens'] < generation['max_tokens']
             and generation['max_tokens'] == request['sampling']['max_tokens'], 'review generation truncated/unbound')
        model_ref = e.reference(self.root / 'reviewer.model.identity.json')
        reviewer = e.document(model_ref)
        annotator_ref = bound['annotator_model_reference']
        annotator_document = e.document(annotator_ref)
        annotator = annotator_document.get('model_identity', annotator_document)
        need(bound['checkpoint_content_independence_verified'] is True
             and bound['annotator_model_sha256'] == pilot['annotator_model_sha256']
             and reviewer['sha256'] == bound['model_sha256']
             and checkpoint(reviewer) == bound['reviewer_checkpoint_sha256']
             and checkpoint(reviewer) != checkpoint(annotator), 'review checkpoint inventories not independent')
        final = e.document(artifacts['final.json'])
        response = e.document(artifacts['response.json'])
        response_raw = json.loads(Path(artifacts['response.raw.txt']['path']).read_bytes())
        need(final['status'] == 'candidate' and final['json'] == response
             and all(response.get(k) == v for k, v in response_raw.items())
             and response['status'] == ledger['status'] in ('pass', 'confirmed_error', 'uncertain'), 'review raw/final/ledger differs')
        code_ref = e.reference(self.root / 'reviewer.code.identity.json')
        code = e.document(code_ref)
        need(code['sha256'] == bound['code_sha256'], 'historical code inventory SHA differs')
        worker = next((v for v in code['files'].values() if Path(v['path']).name == 'oldbar_explicit_review_v1.py'), None)
        need(worker and worker['sha256'] == REVIEW_WORKER_SHA, 'review restored original worker identity differs')
        for ref in code['files'].values():
            e.check(ref)
        need(all(ledger.get(k) is False and receipt.get(k) is False and final.get(k) is False
                 for k in ('semantic_approval', 'human_review_completed', 'formal_approval', 'formal_training', 'training_entry_updated')),
             'review cannot claim human/training approval')
        need(response.get('errors') == ledger.get('errors') and response.get('uncertainties') == ledger.get('uncertainties'),
             'review final/ledger errors or uncertainties differ')
        return {'status': ledger['status'], 'independent': True, 'semantic_label_sha256': label_sha,
                'ledger_ref': ledger_ref, 'review_receipt': ledger['review_receipt'], 'evidence_ref': ledger['evidence_ref'],
                'manifest_ref': self.manifest, 'reviewer_model_ref': model_ref, 'historical_code_inventory_ref': code_ref,
                'historical_code_sha256': bound['code_sha256'], 'current_review_source_used_to_validate_old_receipt': False,
                'restored_exact_original_review_code_inventory_SHA_verified': True,
                'restored_original_worker_sha256': REVIEW_WORKER_SHA,
                'checkpoint_inventory_content_independence_verified': True, 'native_body_frames': 16, 'native_hand_rois': 6,
                'errors': ledger.get('errors', []), 'uncertainties': ledger.get('uncertainties', []), **FLAGS}

    def match(self, candidate):
        reviewed = self.rows.get(key(candidate))
        if (reviewed and reviewed['semantic_label_sha256'] == old_digest(candidate['label'])
                and self.pilot[key(candidate)]['video_path'] == candidate['video_path']):
            pilot = self.pilot[key(candidate)]
            if pilot['source_binding'] == candidate['source_binding'] and pilot['hand_roi'] == candidate['hand_roi']:
                return [reviewed]
        return []


def derive(base_progress, base_rows, imports, members, reviews, summary, diagnostics):
    rows = {key(r): copy.deepcopy(r) for r in base_rows}
    base_complete = {key(r) for r in base_rows if r['state'] == 'complete_candidate'}
    increments = Counter(old_import_verified_unique=len(imports), old_import_overlap_base_unique=0,
                         old_import_new_source_unique=0, old_import_net_complete_unique=0, old_import_selected_unique=0,
                         old_import_individually_reviewed_unique=0, old_import_sampled_pass_unique=0,
                         old_import_known_rejected_alternative_unique=0)
    for identity, record in imports.items():
        candidate, member = record['candidate'], members[identity]
        prior = rows.get(identity)
        increments['old_import_overlap_base_unique'] += bool(prior)
        increments['old_import_new_source_unique'] += not bool(prior)
        source_risk = KNOWN_OLD_ALTERNATIVE_REJECTIONS.get(identity)
        increments['old_import_known_rejected_alternative_unique'] += bool(source_risk)
        matched = reviews.match(candidate)
        increments['old_import_individually_reviewed_unique'] += bool(matched)
        increments['old_import_sampled_pass_unique'] += any(r['status'] == 'pass' for r in matched)
        alternative = {**{k: member[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')},
            'origin': 'oldbar_verified_import', 'kind': 'complete_candidate', 'revision': False,
            'joint_candidate_ref': record['candidate_ref'], 'label': candidate['label'],
            'semantic_label_sha256': old_digest(candidate['label']), 'source_binding': candidate['source_binding'],
            'import_batch_id': candidate['batch_id'], 'original_curated_batch_id': member['batch_id'],
            'import_provenance': candidate['source_refs']['import_provenance'], 'import_disposition_ref': record['disposition_ref'],
            'importer_code_ref': record['importer_code_ref'], 'reviews': matched, 'reviewed': bool(matched),
            'sampled_pass': any(r['status'] == 'pass' for r in matched),
            'pendinghuman': any(r['status'] in ('confirmed_error', 'uncertain') for r in matched),
            'individual_review_status': 'reviewed_exact_semantic_payload' if matched else 'not_individually_reviewed',
            'independent_review_required': True, 'known_source_review_risk': source_risk, **FLAGS}
        if prior is None:
            prior = {**{k: member[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')},
                'state': 'missing', 'selection': None, 'alternatives': [], 'ready_alternatives': [], 'failed_attempts': [],
                'pendinghuman': False, 'overlay_kind': 'candidate_not_training_approved', **FLAGS}
            rows[identity] = prior
        prior.setdefault('base_state', prior['state'])
        prior.setdefault('base_selection', copy.deepcopy(prior['selection']))
        if not any(r.get('joint_candidate_ref') == record['candidate_ref'] for r in prior['alternatives']):
            prior['alternatives'].append(alternative)
        if prior['state'] != 'complete_candidate' and not source_risk:
            prior['selection'] = alternative
            prior['state'] = 'complete_candidate'
            increments['old_import_selected_unique'] += 1
        increments['old_import_net_complete_unique'] += prior['state'] == 'complete_candidate' and identity not in base_complete
        if source_risk:
            prior['known_source_review_risk'] = source_risk
            prior['oldbar_user_rejected_alternative_preserved'] = True
        prior['oldbar_import_present'] = True
        prior['pendinghuman'] = bool(prior.get('pendinghuman') or alternative['pendinghuman'] or source_risk)
        prior['approval_pending'] = True
        prior['oldbar_individual_review_status'] = alternative['individual_review_status']
        prior['base_failures_and_reviews_preserved'] = True
        prior.update(FLAGS)
    counter = Counter({s: 0 for s in STATES})
    splits = {s: Counter({state: 0 for state in STATES}) for s in summary['counts']}
    batches = {d['batch_id']: Counter({state: 0 for state in STATES}) for d in summary['batches']}
    for row in rows.values():
        state = row['state']
        counter[state] += 1
        splits[row['split']][state] += 1
        batches[row['batch_id']][state] += 1
    counter['missing'] += summary['total'] - len(rows)
    for split, expected in summary['counts'].items():
        splits[split]['missing'] += expected - sum(splits[split].values())
    descriptors = []
    for d in summary['batches']:
        c = batches[d['batch_id']]
        c['missing'] += d['count'] - sum(c.values())
        need(sum(c.values()) == d['count'], 'batch source denominator mismatch')
        descriptors.append({'batch_id': d['batch_id'], 'split': d['split'], 'expected': d['count'], 'counts': dict(c),
                            'all_joint_complete': c['complete_candidate'] == d['count'], 'oldbar_subset_approves_original_batch': False})
    need(sum(counter.values()) == summary['total'] and sum(c['missing'] for c in batches.values()) == counter['missing'], 'global denominator mismatch')
    progress = copy.deepcopy(base_progress)
    all_reviewed = {key(r) for r in rows.values() if any(a.get('reviewed') for a in r['alternatives'])}
    all_pass = {key(r) for r in rows.values() if any(a.get('sampled_pass') for a in r['alternatives'])}
    progress.update({'all_pendinghuman_source_unique': sum(bool(r.get('pendinghuman')) for r in rows.values()),
        'all_individually_reviewed_source_unique': len(all_reviewed),
        'all_individually_sampled_pass_source_unique': len(all_pass),
        'base_review_counts_retained_unchanged_oldbar_reviews_reported_separately': True,
        'schema': 'curated_oldbar_derived_overlay_v1', 'base_counts': copy.deepcopy(base_progress['counts']),
        'counts': {**base_progress['counts'], **dict(counter)}, 'split_counts': {s: dict(c) for s, c in splits.items()},
        'batches': descriptors, 'oldbar_incremental_counts': dict(increments),
        'oldbar_import_diagnostics': diagnostics, 'oldbar_review_diagnostics': reviews.diagnostics,
        'base_ready_unique_unchanged': base_progress['counts']['ready_unique'],
        'base_terminal_unique_unchanged': base_progress['counts']['terminal_unique'],
        'oldbar_does_not_create_production_ready_or_terminal': True,
        'all_batches_reviewed': False, 'full_batch_semantic_pass_claimed': False,
        'oldbar_subset_original_batches_review_completed': False,
        'cache_derived_rebuildable_not_authoritative': True, 'import_candidates_export_not_required': True,
        'base_journal_invalid_evidence_preserved': True, 'approval_pending': True,
        'selection_policy': 'preserve_base_complete; show_complete_oldbar_if_base_incomplete; preserve_all_prior_errors', **FLAGS})
    return progress, sorted(rows.values(), key=lambda r: r['source_ordinal'])


def run(args):
    out = Path(args.out).resolve()
    protected = [Path(args.base_root).resolve(), Path(args.import_root).resolve(), Path(args.oldbar_review_root).resolve(), RT,
                 BAR / 'curated222939_inputs_v1']
    need(not any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected), 'output overlaps source namespace')
    cache = Path(args.state_cache).resolve() if args.state_cache else Path('/tmp') / ('oldbar-overlay-' + sha_bytes(str(out).encode())[:20] + '.json')
    need(cache.is_relative_to(Path('/tmp')) and not out.is_relative_to(cache) and not cache.is_relative_to(out), 'cache must be isolated local /tmp JSON')
    out.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(cache.read_bytes()) if cache.exists() else None
        evidence = Evidence(state)
        summary_ref = {'path': str(BAR / 'curated222939_inputs_v1/manifest.summary.json'), 'sha256': SUMMARY_SHA}
        summary = evidence.document(summary_ref)
        try:
            base_progress, base_rows, base_refs = read_base(args.base_root, evidence, summary)
            frozen_base = save_base_snapshot(out, evidence, base_progress, base_refs)
        except Exception as error:
            pending = {'schema': 'curated_oldbar_overlay_pending_v1', 'state': 'pending_stable_base_snapshot',
                       'reason': str(error), 'prior_derived_export_not_replaced': True, **FLAGS}
            atomic(out / 'pending.json', raw(pending))
            print(json.dumps(pending), flush=True)
            return pending
        imports, diagnostics = completed_imports(args.import_root, evidence)
        try:
            evidence.prepare_csv([r['candidate'] for r in imports.values()], summary)
        except Exception as error:
            diagnostics.append({'state': 'pending', 'reason': 'current_CSV_evidence_unavailable: ' + str(error)})
            imports = {}
        members, admitted = {}, {}
        for identity, record in imports.items():
            try:
                members[identity] = validate_candidate(record, evidence, summary)
                admitted[identity] = record
            except Exception as error:
                diagnostics.append({'source_key': identity, 'state': 'excluded', 'reason': str(error),
                                    'candidate_ref': record['candidate_ref'], 'disposition_ref': record['disposition_ref']})
        try:
            reviews = Reviews(args.oldbar_review_root, evidence)
        except Exception as error:
            reviews = Reviews.__new__(Reviews)
            reviews.rows = {}
            reviews.diagnostics = [{'state': 'pending_review_evidence_unavailable', 'reason': str(error)}]
        progress, rows = derive(base_progress, base_rows, admitted, members, reviews, summary, diagnostics)
        progress.update({'base_snapshot_refs': base_refs, 'frozen_base_snapshot': frozen_base,
                         'base_snapshot_only': True, 'latest_live_base_claimed': False, 'oldbar_import_root': str(Path(args.import_root).resolve()),
                         'oldbar_review_root': str(Path(args.oldbar_review_root).resolve()),
                         'state_cache': {'path': str(cache), 'local_only': True, 'authority': False, 'rebuildable': True}})
        try:
            for ref in frozen_base['immutable_refs'] + [frozen_base['snapshot_metadata_ref']]:
                evidence.check(ref, strong=True)
        except Exception as error:
            pending = {'schema': 'curated_oldbar_overlay_pending_v1', 'state': 'pending_stable_base_snapshot',
                       'reason': 'immutable_base_snapshot_changed: ' + str(error),
                       'prior_derived_export_not_replaced': True, **FLAGS}
            atomic(cache, raw(evidence.state))
            atomic(out / 'pending.json', raw(pending))
            print(json.dumps(pending), flush=True)
            return pending
        data = b''.join(raw(r) for r in rows)
        progress['candidate_export_sha256'] = sha_bytes(data)
        atomic(out / 'candidates.jsonl', data)
        atomic(out / 'progress.json', raw(progress))
        atomic(out / 'pending.json', raw({'state': 'no_pending_base_snapshot',
            'candidate_export_sha256': sha_bytes(data), **FLAGS}))
        atomic(cache, raw(evidence.state))
        receipt = {'schema': 'curated_oldbar_overlay_receipt_v1', 'code_ref': evidence.reference(__file__),
                   'base_snapshot_refs': base_refs, 'frozen_base_snapshot': frozen_base,
                   'base_snapshot_only': True, 'latest_live_base_claimed': False, 'candidate_export_sha256': sha_bytes(data),
                   'derived_progress_sha256': sha_bytes(raw(progress)), 'oldbar_incremental_counts': progress['oldbar_incremental_counts'],
                   'completed_disposition_refs': [r['disposition_ref'] for r in admitted.values()],
                   'base_original_invalid_records_preserved': True, **FLAGS}
        atomic(out / 'receipts' / (sha_bytes(raw(receipt)) + '.json'), raw(receipt))
        print(json.dumps({'oldbar_incremental_counts': progress['oldbar_incremental_counts'], 'counts': progress['counts'],
                          'out': str(out)}, ensure_ascii=False), flush=True)
        return progress


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-root', default=str(BAR / 'curated222939_overlay_v2'))
    p.add_argument('--import-root', default=str(BAR / 'oldbar_explicit_import_v1'))
    p.add_argument('--oldbar-review-root', default=str(BAR / 'curated_oldbar3_independent_reviews_v1'))
    p.add_argument('--out', default=str(BAR / 'curated222939_overlay_with_oldbar_v1'))
    p.add_argument('--state-cache')
    p.add_argument('--once', action='store_true', help='Single pass, also the default; no daemon')
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    raise SystemExit(1 if result.get('state') == 'pending_stable_base_snapshot' else 0)
