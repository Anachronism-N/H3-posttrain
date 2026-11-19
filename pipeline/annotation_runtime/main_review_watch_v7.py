"""Frozen v5 contract, finite byte index, exact empty row and terminal reconciliation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

RT = Path(__file__).resolve().parent
ROOT = RT.parent
FROZEN = {
    'main_review_watch_v5.py': '568a4fb896bbdaae65d94e77324bcba546257fd965000fd3c1029ff37d17b039',
    'main_review_watch_v6.py': '0226ce3dd29873aede62a0e4952623f1942a4806f5236f1ddeca1d30d85da711',
    'main_review_history_v1.py': 'b23d698108ea699220b94dc96564fbc17d1ae66afb9ac30332acdc788f6b6484',
    'curated_repair_collector_recovery_v1.py': 'f85efb1441930cfc31bfe4cafacf25e0fdb18aeb9da6ad80fb540b58519237cd',
    'source_repair_gpu_v2/curated_repair_queue_v1.py': '94b00ad0b503053f4b9bd7f7c08097bf9b6480f8a9aa248e971126a149c78b4c',
}
for name, digest in FROZEN.items():
    if hashlib.sha256((RT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen v7 dependency changed: ' + name)
sys.path.insert(0, str(RT))
spec = importlib.util.spec_from_file_location('frozen_main_review_v6_for_v7', RT / 'main_review_watch_v6.py')
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)
v5, recovery = v6.v5, v6.recovery
need, sha, atomic, checked = v5.need, v5.sha, v5.atomic, v5.checked
Membership, ByteCache = v5.Membership, v5.ByteCache
terminal_record = v5.terminal_record
retained_selection, install_history_selection = v5.retained_selection, v5.install_history_selection
reviewer_from_verified_local_model = v5.reviewer_from_verified_local_model
historical_ids = v6.historical_ids
EMPTY_OFFSET = 1192186
EMPTY_SHA = '01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b'
EMPTY_HANDLER_SHA = '5c6c01e2f9ad70fc6515ef43384e81af2cface3b52f716c67496a24e1565752a'
EXPECTED_RECONCILIATION = {
    'train/BBucGIEtwcrDIngP': 'train-b000001',
    'train/dBloPhzUQlCooRtD': 'train-b000001',
    'train/pbUXtdkTIWrvjeXw': 'train-b000003',
    'train/tYhkFFYDNlaEJimA': 'train-b000003',
}


class Incident(v5.Incident):
    def __init__(self, production, manifest, binding, empty_binding=None):
        super().__init__(production, manifest, binding)
        self.empty_binding = empty_binding
        self.empty_row_offsets = []

    def expand(self, begin, line):
        if begin != EMPTY_OFFSET:
            return super().expand(begin, line)
        need(self.empty_binding is not None, 'Empty row lacks explicit receipt binding')
        need(self.production == (ROOT / 'curated222939_production_v1').resolve(), 'Empty row foreign production namespace')
        need(line == b'\n' and hashlib.sha256(line).hexdigest() == EMPTY_SHA, 'Exact empty row bytes changed')
        receipt_ref = self.empty_binding['receipt_ref']
        receipt_path = ROOT / 'curated222939_repair_v1/collector-emptyline-v2/1192186.json'
        need(Path(receipt_ref['path']).resolve() == receipt_path.resolve(), 'Empty receipt outside original repair namespace')
        receipt = checked(receipt_ref)
        expected = {
            'schema': 'verified_empty_physical_jsonl_row_v2',
            'original_stream': str(self.production / 'completed.jsonl'),
            'byte_begin': EMPTY_OFFSET, 'byte_end': EMPTY_OFFSET + 1,
            'raw_size': 1, 'raw_sha256': EMPTY_SHA,
            'contained_source_records': 0, 'reconstructed_source_records': 0,
            'source_terminal_or_label_generated': False, 'source_records_skipped': False,
            'original_log_modified': False, 'original_decode_error_preserved': True,
            'source_population_completeness_inferred': False,
            'unseen_or_lost_terminals_require_separate_reconciliation': True,
            **recovery.q.FLAGS,
        }
        need(all(type(receipt.get(k)) is type(value) and receipt.get(k) == value for k, value in expected.items()), 'Empty receipt span/zero-source/decode-error contract differs')
        raw_ref = receipt['raw_ref']
        raw_path = ROOT / ('curated222939_repair_v1/collector-recovery-v1/badlines/' + str(EMPTY_OFFSET) + '-' + EMPTY_SHA + '.raw')
        need(Path(raw_ref['path']).resolve() == raw_path.resolve() and raw_ref['sha256'] == EMPTY_SHA, 'Empty rawref namespace/SHA differs')
        raw, ref = recovery.stable_read(raw_path)
        need(raw == line and ref == raw_ref, 'Empty original raw evidence differs')
        for name, path, digest in (
                ('collector_base_ref', RT / 'curated_repair_collector_recovery_v1.py', FROZEN['curated_repair_collector_recovery_v1.py']),
                ('emptyline_handler_ref', RT / 'curated_repair_collector_emptyline_v2.py', EMPTY_HANDLER_SHA)):
            need(receipt[name] == {'path': str(path), 'sha256': digest} and sha(path) == digest, 'Empty collector code identity changed')
        stream = self.production / 'completed.jsonl'
        s = stream.stat()
        identity = [s.st_dev, s.st_ino]
        snapshots = receipt['stable_badline_snapshots']
        need(len(snapshots) == 3 and all(x['identity'] == snapshots[0]['identity'] and x['line_sha256'] == EMPTY_SHA for x in snapshots), 'Empty receipt lacks three stable snapshots')
        need(len(snapshots[0]['identity']) == 5 and snapshots[0]['identity'][1] == identity[1] and snapshots[0]['identity'][2] >= EMPTY_OFFSET + 1, 'Empty receipt original source inode/span differs')
        need(self.empty_binding['collector_snapshot_identity'] == snapshots[0]['identity'][:2], 'Empty collector mount-local snapshot identity differs')
        need(self.empty_binding['original_byte_begin'] == begin and self.empty_binding['original_byte_end'] == begin + 1 and self.empty_binding['source_identity'] == identity, 'Empty origin span/identity binding differs')
        need(self.empty_binding['historical_capture_identity'] == snapshots[0]['identity'][:2]
             and self.empty_binding['fresh_current_node_identity'] == identity, 'Empty historical/fresh mount identity differs')
        need(self.empty_binding['cross_mount_device_rebound_content_verified'] is (identity[0] != snapshots[0]['identity'][0]), 'Explicit empty cross-mount content rebound proof differs')
        prefix = self.empty_binding['current_verified_prefix']
        need(prefix['path'] == str(stream) and prefix['identity'] == identity and prefix['bytes'] >= begin + 1, 'Empty current verified original prefix namespace/identity/span differs')
        need(recovery.prefix_sha(stream, prefix['bytes'], identity) == prefix['sha256'], 'Empty current verified original prefix changed')
        recovery.stable_badline(stream, begin, line, identity, .05)
        need(recovery.prefix_sha(stream, prefix['bytes'], identity) == prefix['sha256'], 'Empty prefix changed after three current snapshots')
        if begin not in self.empty_row_offsets:
            self.empty_row_offsets.append(begin)
        return []


class NormalizedReader(v5.NormalizedReader):
    @property
    def empty_row_offsets(self):
        return self.incident.empty_row_offsets if self.incident else []


class RawIndex(v6.RawIndex):
    def __init__(self, *args, reconciliation_mount_bindings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconciliation_mount_bindings = reconciliation_mount_bindings
        self.empty_row_offsets = []
        self.restored_sources = {}
        self.journal_keys = set()
        self.reconciliation_manifest_ref = None

    def scan(self):
        self.pending = None
        dirty = set()
        if not self.stream.exists():
            self.pending = {'offset': self.indexed_offset, 'reason': 'stream_not_published', 'skipped': False}
            return dirty
        stat = self.stream.stat()
        identity = [stat.st_dev, stat.st_ino]
        if self.identity is None:
            self.identity = identity
        need(self.identity == identity and stat.st_size >= self.indexed_offset, 'Original stream replaced/truncated')
        self.check_prefix()
        limit = stat.st_size
        with self.stream.open('rb') as f:
            need([os.fstat(f.fileno()).st_dev, os.fstat(f.fileno()).st_ino] == identity, 'Opened source identity differs')
            f.seek(self.indexed_offset)
            data = f.read(limit - self.indexed_offset)
            need(len(data) == limit - self.indexed_offset, 'Finite source snapshot truncated')
        begin = self.indexed_offset
        consumed = bytearray()
        staged_lines = []
        snapshot_hints = {}
        for line in data.splitlines(keepends=True):
            if not line.endswith(b'\n'):
                self.pending = {'offset': begin, 'reason': 'partial_original_line', 'skipped': False}
                break
            try:
                try:
                    values = [json.loads(line)]
                except (ValueError, UnicodeError):
                    need(self.incident is not None and (begin in recovery.KNOWN or begin == EMPTY_OFFSET), 'unknown_malformed_line')
                    recovery.stable_badline(self.stream, begin, line, identity, .05)
                    values = self.incident.expand(begin, line)
                staged = []
                for value in values:
                    need(isinstance(value, dict) and all(k in value for k in ('id', 'split', 'batch_id', 'status')), 'Raw terminal schema differs')
                    sid, split, bid = value['id'], value['split'], value['batch_id']
                    need(isinstance(sid, str) and sid and bid in self.allowed_batches, 'Raw source outside admitted edition')
                    originals = self.membership.batch(bid)
                    need(sid in originals and split == self.membership.descriptors[bid]['split'], 'Raw hint outside full authoritative membership')
                    old = self.raw.get(bid, {}).get(sid)
                    prior = snapshot_hints.get((bid, sid), old['hint'] if old else None)
                    need(prior is None or prior == value, 'Raw source revision changed without explicit repair')
                    staged.append((bid, sid, value, old))
            except (ValueError, UnicodeError, OSError, KeyError, TypeError, recovery.StreamBusy) as error:
                self.pending = {'offset': begin, 'reason': str(error), 'raw_sha256': hashlib.sha256(line).hexdigest(), 'skipped': False}
                break
            for bid, sid, value, _ in staged:
                snapshot_hints[(bid, sid)] = value
            staged_lines.append((begin, begin + len(line), staged, len(values)))
            consumed.extend(line)
            begin += len(line)
        h = recovery.prefix_hash(self.stream, self.indexed_offset, identity)
        need(h.hexdigest() == self.indexed_sha, 'Previously indexed bytes changed during scan')
        h.update(consumed)
        need(recovery.prefix_sha(self.stream, begin, identity) == h.hexdigest(), 'Read raw snapshot changed before index commit')
        # Commit only after the exact physical prefix was rechecked.
        for start, end, staged, count in staged_lines:
            for bid, sid, value, _ in staged:
                old = self.raw[bid].get(sid)
                need(old is None or old['hint'] == value, 'Same-snapshot source revision differs')
                if old is None:
                    self.raw[bid][sid] = {'hint': value, 'byte_begin': start, 'byte_end': end, 'raw_sha256': hashlib.sha256(data[start - self.indexed_offset:end - self.indexed_offset]).hexdigest(), 'origin': 'original_journal'}
                    dirty.add(bid)
            self.journal_keys.update((b, s) for b, s, _, _ in staged)
            self.lines.append((start, end, [(b, s) for b, s, _, _ in staged]))
            if count == 2 and start not in self.recovered_offsets:
                self.recovered_offsets.append(start)
            if count == 0 and start not in self.empty_row_offsets:
                self.empty_row_offsets.append(start)
        self.indexed_offset, self.indexed_sha = begin, h.hexdigest()
        return dirty

    def verify(self, bid, ids=None):
        previous = v6.terminal_record
        try:
            v6.terminal_record = terminal_record
            return super().verify(bid, ids)
        finally:
            v6.terminal_record = previous

    def reconcile(self, manifest, expected_ref=None):
        restored = Reconciliation(self, manifest, expected_ref)
        staged = []
        for key, record_ref, value, row in restored.records:
            hint = value['hint']
            bid, sid = hint['batch_id'], hint['id']
            need(bid in self.allowed_batches, 'Reconciliation outside admitted original batches')
            old = self.raw[bid].get(sid)
            need(old is None or old['hint'] == hint, 'Reconciliation conflicts with appended journal candidate')
            verified = self.completed[bid].get(sid)
            need(verified is None or verified == row, 'Reconciled label revision changed without explicit repair')
            staged.append((key, record_ref, hint, row, old))
        self.check_prefix()
        dirty = set()
        for key, ref, hint, row, old in staged:
            bid, sid = hint['batch_id'], hint['id']
            if old is None:
                self.raw[bid][sid] = {'hint': hint, 'byte_begin': None, 'byte_end': None,
                    'raw_sha256': None, 'origin': 'explicit_terminal_reconciliation', 'record_ref': ref}
                dirty.add(bid)
            self.completed[bid][sid] = row
            self.verified_keys.add((bid, sid))
            self.restored_sources[key] = {'record_ref': ref, 'original_byte_offset': None,
                'original_consumer_terminal_ref': restored.values[key]['original_consumer_terminal_ref'], 'batch_id': bid,
                'input_candidate_sha256': recovery.q.digest(hint), 'label_sha256': row.get('label_sha256')}
        self.reconciliation_manifest_ref = restored.ref
        return dirty


def progress(index, out, edition, bids, total):
    value = v6.progress(index, out, edition, bids, total)
    value['schema'] = 'curated_independent_review_watch_v7'
    value['indexed_raw_source_count'] = len(index.journal_keys)
    value['verified_journal_source_count'] = len(index.journal_keys & index.verified_keys)
    value['unverified_raw_source_count'] = len(index.journal_keys - index.verified_keys)
    value['admitted_terminal_hint_count'] = sum(len(rows) for rows in index.raw.values())
    value['empty_row_offsets'] = index.empty_row_offsets
    value['empty_physical_rows_encoded_sources'] = 0
    value['empty_physical_rows_reconstructed_sources'] = 0
    value['reconciliation_restored_source_count'] = len(index.restored_sources)
    value['reconciliation_restored_sources'] = index.restored_sources
    value['reconciliation_manifest_ref'] = index.reconciliation_manifest_ref
    value['reconciliation_does_not_assign_original_byte_offsets'] = True
    value['full_population_or_quality_inferred_from_recovery'] = False
    return value


def verify_mount_binding(binding, prefix, raw_ref, expected_path):
    need(binding['original_prefix'] == prefix and binding['raw_prefix_ref'] == raw_ref, 'Fresh mount proof original snapshot/ref changed')
    need(prefix['path'] == expected_path and binding['path'] == expected_path and str(Path(expected_path).resolve()) == expected_path, 'Fresh mount proof original namespace differs')
    historic, current = binding['historical_capture_identity'], binding['fresh_current_node_identity']
    need(historic == prefix['input_identity'] and len(current) == 2 and current[1] == historic[1], 'Fresh mount original inode identity differs')
    need(binding['current_node_hostname'] == os.uname().nodename, 'Fresh mount proof belongs to another node')
    need(binding['cross_mount_device_rebound_content_verified'] is (current[0] != historic[0]) and binding['prefix_content_SHA_verified'] is True, 'Explicit cross-mount prefix content proof missing')
    path = Path(expected_path)
    need(recovery.prefix_sha(path, prefix['byte_length'], current) == prefix['prefix_sha256'], 'Fresh admitted mount original prefix changed')
    raw, ref = recovery.stable_read(raw_ref['path'])
    need(ref == raw_ref and len(raw) == prefix['byte_length'] and ref['sha256'] == prefix['prefix_sha256'], 'Original immutable snapshot raw bytes/length/SHA differ')
    need(recovery.prefix_sha(path, prefix['byte_length'], current) == prefix['prefix_sha256'], 'Original prefix changed after immutable snapshot SHA verification')
    return current


def capture_reconciliation_mount_bindings(manifest):
    admission = checked(recovery.document(manifest)[1])
    production = ROOT / 'curated222939_production_v1'
    need(admission['production'] == str(production), 'Fresh reconciliation mount capture foreign production')
    snapshot = checked(admission['snapshot_ref'])
    prefixes = [(snapshot['original_journal_prefix'], snapshot['raw_journal_ref'], str(production / 'completed.jsonl'))]
    prefixes.extend((prefix, snapshot['raw_terminal_index_refs'][bid], str(production / 'batches' / bid / 'terminal_index.jsonl'))
                    for bid, prefix in snapshot['terminal_index_prefixes'].items())
    result = {}
    for prefix, raw_ref, expected_path in prefixes:
        need(prefix['path'] == expected_path, 'Historical reconciliation prefix foreign original namespace')
        s = Path(expected_path).stat()
        current = [s.st_dev, s.st_ino]
        historic = prefix['input_identity']
        binding = {'path': expected_path, 'original_prefix': prefix, 'raw_prefix_ref': raw_ref,
                   'historical_capture_identity': historic, 'fresh_current_node_identity': current,
                   'current_node_hostname': os.uname().nodename,
                   'cross_mount_device_rebound_content_verified': current[0] != historic[0],
                   'prefix_content_SHA_verified': True}
        verify_mount_binding(binding, prefix, raw_ref, expected_path)
        result[expected_path] = binding
    return result


class Reconciliation:
    def __init__(self, index, manifest, expected_ref=None):
        production = index.production
        need(production == (ROOT / 'curated222939_production_v1').resolve(), 'Reconciliation foreign production root')
        self.document, self.ref = recovery.document(manifest, expected_ref['sha256'] if expected_ref else None)
        if expected_ref:
            need(self.ref == expected_ref, 'Reconciliation manifest ref differs from source binding')
        value = self.document
        namespace = (ROOT / 'curated_terminal_reconciliation_v1').resolve()
        need(Path(manifest).resolve().parent == namespace and Path(manifest).name.startswith('admission-'), 'Frozen reconciliation admission namespace required')
        need(value['schema'] == 'terminal_reconciliation_admission_v1' and value['production'] == str(production), 'Reconciliation admission schema/root differs')
        need(value['manifest_summary_ref'] == index.membership.ref, 'Reconciliation full authoritative manifest differs')
        need(value['quality_approved'] is False and value['pending_independent_review'] is True, 'Recovered data must remain independently unapproved')
        need(len(value['source_keys']) == 4 and set(value['source_keys']) == set(EXPECTED_RECONCILIATION), 'Only four frozen actually missing sources admitted')
        refs = value['record_refs']
        need(len(refs) == 4 and len({r['path'] for r in refs}) == 4, 'Four unique exact record refs required')
        evidence = recovery.Evidence(production, index.membership.ref['path'])
        self.records, self.values = [], {}
        snapshot_ref = value['snapshot_ref']
        snapshot = checked(snapshot_ref)
        need(Path(snapshot_ref['path']).resolve().parent.parent == namespace / 'snapshots' and Path(snapshot_ref['path']).name == 'snapshot.json', 'Reconciliation snapshot outside original recovery namespace')
        need(snapshot['production'] == str(production) and snapshot['manifest_summary_ref'] == index.membership.ref, 'Snapshot foreign original root/manifest')
        prefix = snapshot['original_journal_prefix']
        need(prefix['path'] == str(index.stream), 'Snapshot source_ref outside original production journal')
        need(prefix['input_identity'][1] == index.identity[1] and prefix['byte_length'] <= index.indexed_offset, 'Snapshot original inode/span not part of indexed original stream')
        mount_bindings = index.reconciliation_mount_bindings
        expected_paths = {str(index.stream)} | {str(production / 'batches' / bid / 'terminal_index.jsonl') for bid in snapshot['terminal_index_prefixes']}
        need(mount_bindings is not None and set(mount_bindings) == expected_paths, 'Explicit frozen current-node journal/index mount bindings required')
        current_journal_identity = verify_mount_binding(mount_bindings[str(index.stream)], prefix, snapshot['raw_journal_ref'], str(index.stream))
        need(current_journal_identity == index.identity, 'Reconciliation journal fresh identity differs from reader admission')
        raw_ref = snapshot['raw_journal_ref']
        raw_path = Path(snapshot_ref['path']).parent / 'completed.prefix.raw'
        need(Path(raw_ref['path']).resolve() == raw_path.resolve() and raw_ref['sha256'] == prefix['prefix_sha256'], 'Snapshot raw namespace/SHA differs')
        raw, actual = recovery.stable_read(raw_path)
        need(actual == raw_ref and len(raw) == prefix['byte_length'] and (not raw or raw.endswith(b'\n')), 'Exact snapshotted raw bytes differ')
        keys = {}
        begin = 0
        for line in raw.splitlines(keepends=True):
            try:
                hints = [json.loads(line)]
            except (ValueError, UnicodeError):
                need(index.incident is not None, 'Snapshot malformed line lacks original incident proof')
                hints = index.incident.expand(begin, line)
            for hint in hints:
                key = recovery.q.safe_key(hint)
                bid, sid = hint['batch_id'], hint['id']
                need(bid in index.membership.descriptors and sid in index.membership.batch(bid) and hint['split'] == index.membership.descriptors[bid]['split'], 'Snapshot hint foreign original membership')
                need(key not in keys or keys[key] == hint, 'Snapshot contains conflicting same-SID revision')
                keys[key] = hint
            begin += len(line)
        need(begin == prefix['byte_length'], 'Snapshot contains incomplete physical boundary')
        for ref in refs:
            record = checked(ref)
            hint = record['hint']
            key = recovery.q.safe_key(hint)
            need(key in EXPECTED_RECONCILIATION and key not in self.values, 'Foreign/duplicate reconciliation source')
            bid, sid, split = hint['batch_id'], hint['id'], hint['split']
            need(bid == EXPECTED_RECONCILIATION[key] and split == 'train' and len(index.membership.batch(bid)) == 500, 'Reconciliation split/full500batch differs')
            need(Path(ref['path']).resolve() == namespace / 'records' / (key + '.json'), 'Reconciliation record source namespace differs')
            need(record['schema'] == 'terminal_reconciliation_v1' and record['source_key'] == key and record['source_dedup_key'] == key, 'Reconciliation source/candidate key differs')
            recovery.source_identity(record, sid, split, bid)
            need(record['completed_row_sha256'] == recovery.q.digest(hint), 'Reconciliation exact candidate SHA differs')
            need(record['original_byte_offset'] is None and record['absence_from_that_exact_snapshot'] is True and key not in keys, 'Reconciliation lacks exact-snapshot absence or assigns false byte offset')
            need(record['original_journal_snapshot_ref'] == snapshot_ref and record['original_journal_snapshot'] == prefix, 'Record snapshot identity differs')
            need(record['quality_approved'] is False and record['pending_independent_review'] is True and record['data_complete_candidate'] is True, 'Reconciliation claims quality approval')
            for flag in ('source_reinference_performed', 'original_journal_modified', *recovery.q.FLAGS):
                need(record[flag] is False, 'Reconciliation source/review/training flags changed')
            need(hint['status'] == 'joint_candidate' and hint['joint_complete'] is True, 'Reconciliation requires actual complete joint terminal')
            terminal, checked_refs = evidence.terminal(sid, split, bid)
            need(terminal == hint, 'Reconciliation hint differs from actual consumer terminal')
            directory = production / 'sources' / key
            need(record['original_consumer_terminal_ref'] == checked_refs['consumer_terminal_ref'] and Path(record['original_consumer_terminal_ref']['path']).resolve() == directory / 'consumer.terminal.json', 'Reconciliation original consumer terminal namespace differs')
            joint_ref = hint['joint_candidate_ref']
            need(Path(joint_ref['path']).resolve().parent == directory / 'joint-results' and Path(joint_ref['path']).name == joint_ref['sha256'] + '.json', 'Reconciliation candidate outside original content-SHA namespace')
            joint = checked(joint_ref)
            visual_root = Path(checked_refs['visual_result_ref']['path']).parent.parent
            prepared, prepared_ref = recovery.document(visual_root / 'prepared.json')
            receipt, receipt_ref = recovery.document(visual_root / 'prepared.receipt.json')
            binding = checked(checked_refs['source_binding_ref'])
            need(prepared['binding'] == binding and receipt['binding'] == binding and receipt['prepared_sha256'] == prepared_ref['sha256'], 'Reconciliation prepared visual binding/SHA differs')
            checked_refs.update({'visual_prepared_ref': prepared_ref, 'visual_prepared_receipt_ref': receipt_ref})
            if joint.get('style_result_ref'):
                checked_refs['style_result_ref'] = joint['style_result_ref']
                checked(checked_refs['style_result_ref'])
            need(record['evidence'] == checked_refs, 'Reconciliation all immutable Evidence refs differ')
            member, member_ref = evidence.member(sid, split, bid)
            source = checked(checked_refs['source_identity_ref'])
            original = {'manifest_summary_ref': evidence.summary_ref, 'batch_id': bid,
                'batch_manifest_ref': member_ref, 'manifest_record': member,
                'manifest_record_sha256': recovery.q.digest(member), 'original_csv_ref': member['source_refs']['csv'],
                'original_source_refs': member['source_refs'],
                'original_media_refs': {name: {k: source['files'][name][k] for k in ('path', 'sha256')} for name in ('video_path', 'audio_path')}}
            need(record['current_original_manifest'] == original, 'Reconciliation CSV/media/full member source identity differs')
            index_prefix = record['original_terminal_index_prefix']
            need(index_prefix == snapshot['terminal_index_prefixes'][bid] and index_prefix['path'] == str(production / 'batches' / bid / 'terminal_index.jsonl'), 'Reconciliation terminal_index production namespace differs')
            path = Path(index_prefix['path'])
            index_raw_ref = snapshot['raw_terminal_index_refs'][bid]
            verify_mount_binding(mount_bindings[str(path)], index_prefix, index_raw_ref, str(path))
            need(Path(index_raw_ref['path']).resolve() == Path(snapshot_ref['path']).parent / (bid + '.terminal_index.prefix.raw'), 'Terminal_index rawref namespace differs')
            index_raw, index_actual = recovery.stable_read(index_raw_ref['path'])
            need(index_actual == index_raw_ref and len(index_raw) == index_prefix['byte_length'] and index_actual['sha256'] == index_prefix['prefix_sha256'], 'Terminal_index frozen raw/prefix differs')
            index_rows = [json.loads(line) for line in index_raw.splitlines()]
            need({'id': sid, 'split': split} in index_rows, 'Original terminal_index lacks recovered source notification')
            row = terminal_record(production, terminal, index.membership)
            self.values[key] = record
            self.records.append((key, ref, record, row))
        need(set(self.values) == set(EXPECTED_RECONCILIATION), 'Frozen four-source reconciliation incomplete')
        verify_mount_binding(mount_bindings[str(index.stream)], prefix, snapshot['raw_journal_ref'], str(index.stream))
        for bid, original_prefix in snapshot['terminal_index_prefixes'].items():
            original_path = str(production / 'batches' / bid / 'terminal_index.jsonl')
            verify_mount_binding(mount_bindings[original_path], original_prefix, snapshot['raw_terminal_index_refs'][bid], original_path)
        index.check_prefix()


def validate_binding(binding, args, membership):
    need(binding['schema'] == 'main_review_v7_original_stream_binding_v1' and binding['manifest_ref'] == membership.ref, 'v7 full original source binding differs')
    need(binding['validation_mode'] == 'finite_raw_index_batch_lazy' and binding['remaining_raw_sources_are_unverified'] is True, 'Explicit raw/verified source separation required')
    need(binding['main_prefix']['path'] == str(Path(args.results_root).resolve() / 'completed.jsonl'), 'Original bound input root differs')
    need(binding['annotator_model_ref']['path'] == args.annotator_model_reference and sha(args.annotator_model_reference) == binding['annotator_model_ref']['sha256'], 'Original annotator identity evidence differs')
    need(bool(args.reconciliation_manifest) == bool(binding.get('reconciliation_manifest_ref')), 'Reconciliation CLI/sourcebinding admission differs')
    need(bool(args.reconciliation_manifest) == bool(binding.get('reconciliation_mount_bindings')), 'Reconciliation fixed current-node mount mapping admission differs')
    if args.reconciliation_manifest:
        need(args.reconciliation_manifest == binding['reconciliation_manifest_ref']['path'], 'Reconciliation manifest path differs')
    extra = checked(binding['additional_edition']['manifest'])
    need(extra['original_summary'] == membership.ref and extra['selected_source_count'] == 10000 and extra['source_range'] == {'start': 21000, 'end': 31000}, 'Additional authoritative range differs')
    bids = {b['batch_id'] for b in extra['selected_batches']}
    need(len(bids) == 20 and all(membership.descriptors[b['batch_id']] == b for b in extra['selected_batches']), 'Additional original full batch descriptors differ')
    need(binding['additional_edition']['results_root'] == args.additional_results_root and binding['additional_edition']['out'] == args.additional_out and binding['additional_edition']['manifest']['path'] == args.additional_manifest, 'Independent additional edition binding differs')
    return bids


def main():
    ap = argparse.ArgumentParser()
    for name in ('manifest', 'results-root', 'out', 'model', 'ms-source', 'annotator-model-reference', 'source-binding',
                 'additional-results-root', 'additional-out', 'additional-manifest'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--reconciliation-manifest')
    ap.add_argument('--poll-seconds', type=float, default=60)
    a = ap.parse_args()
    need(a.poll_seconds > 0, 'Positive poll interval required')
    binding = json.loads(Path(a.source_binding).read_text())
    cache = ByteCache()
    recovery.q.sha = cache.sha
    membership = Membership(a.manifest)
    bids = validate_binding(binding, a, membership)
    incident = Incident(a.results_root, a.manifest, binding['incident'], binding['empty_incident'])
    index = RawIndex(a.results_root, membership, incident=incident, binding=binding['main_prefix'],
                     reconciliation_mount_bindings=binding.get('reconciliation_mount_bindings'))
    index.scan()
    while index.pending and any(x in index.pending['reason'] for x in ('not stable', 'Concurrent stream write')):
        time.sleep(a.poll_seconds)
        index.scan()
    need(recovery.BAD_OFFSET in index.recovered_offsets and EMPTY_OFFSET in index.empty_row_offsets, 'Original exact splice/zero-source empty row not proved')
    need(index.pending is None or index.pending['reason'] == 'partial_original_line', 'Unknown original stream record blocks model admission')
    if a.reconciliation_manifest:
        index.reconcile(a.reconciliation_manifest, binding['reconciliation_manifest_ref'])
    need(index.verify('train-b000000', historical_ids(a.out, 'train-b000000')), 'Historical evidence not fully published')
    sys.path.insert(0, a.ms_source)
    import ms_batch_review as review
    review.batch.sha256 = cache.sha
    reviewer = reviewer_from_verified_local_model(review, a.model, a.out, a.annotator_model_reference, binding['model_stage_ref'])
    extra_index = RawIndex(a.additional_results_root, membership, allowed_batches=bids)
    contexts = [(index, Path(a.out), 'main-production-v1', set(membership.descriptors), 222939),
                (extra_index, Path(a.additional_out), 'collaborative-v2', bids, 10000)]
    pending_dirty = {id(index): set(index.raw), id(extra_index): set()}
    baseline = review.sampling_plan
    while True:
        values = []
        for current, out, edition, expected, total in contexts:
            out.mkdir(parents=True, exist_ok=True)
            pending_dirty[id(current)] |= current.scan()
            if not current.pending:
                review.sampling_plan = baseline
                install_history_selection(review, out)
                reviewer.hand_roi_root = str(current.production / 'roi')
                for bid in sorted(pending_dirty[id(current)]):
                    if not current.verify(bid):
                        continue
                    originals = membership.batch(bid)
                    rows = [current.completed[bid].get(sid, {**row, 'label_status': 'pending'}) for sid, row in originals.items()]
                    retained_selection(review, out, bid, [review.normalized_record(r) for r in rows])
                    result = review.review_batch_ready(rows, out, reviewer, batch_id=bid, expected_source_ids=list(originals), sample_count=25, risk_count=5)
                    print(json.dumps({'edition': edition, 'batch': bid, 'verified_terminal_sources': len(current.completed[bid]),
                        'sources_reviewed': result['sampled_reviewed_denominator'], 'AI_error_candidates': result['confirmed_error_count'],
                        'state': result['state'], 'eligible_for_training': result['acceptance']['eligible_for_training']}), flush=True)
                    pending_dirty[id(current)].remove(bid)
            value = progress(current, out, edition, expected, total)
            value['source_binding'] = a.source_binding
            atomic(out / 'progress.json', value)
            if value['all_batches_reviewed'] and value['all_source_labels_completed'] and not current.pending and not current.batch_pending:
                atomic(out / 'run.summary.json', value)
            values.append(value)
        if all(v['all_batches_reviewed'] and v['all_source_labels_completed'] and not v['pending_stream_record'] and not v['batch_pending_evidence'] for v in values):
            break
        time.sleep(a.poll_seconds)


if __name__ == '__main__':
    main()
