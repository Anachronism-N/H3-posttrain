"""Private exact five-byte admission; v7 review/history/resource contracts stay frozen."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

RT = Path(__file__).resolve().parent
ROOT = RT.parent
V7_SHA = '422aabcf2b1e54ceac3b797b4be123ff510d6643d1a9da586e530fcee12bea21'
OFFSET = 2053650
RAW = bytes.fromhex('7465227d0a')
RAW_SHA = '7561b973a7f36b54382420878e894bbb5cd9e7bcd7e5a8db6ec27f883315593c'
SCHEMA = 'main_review_offset2053650_binding_v1'


def module(name, path, digest):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen execution dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


v7 = module('private_frozen_main_review_v7_for_v8', RT / 'main_review_watch_v7.py', V7_SHA)
need, sha, checked, recovery = v7.need, v7.sha, v7.checked, v7.recovery
Membership, ByteCache, terminal_record = v7.Membership, v7.ByteCache, v7.terminal_record
historical_ids, EMPTY_OFFSET = v7.historical_ids, v7.EMPTY_OFFSET
v5, NormalizedReader = v7.v5, v7.NormalizedReader
retained_selection, install_history_selection = v7.retained_selection, v7.install_history_selection
reviewer_from_verified_local_model = v7.reviewer_from_verified_local_model
capture_reconciliation_mount_bindings = v7.capture_reconciliation_mount_bindings


def derive(function, replacements, namespace):
    source = textwrap.dedent(inspect.getsource(function))
    for old, new in replacements:
        need(source.count(old) == 1, 'Frozen minimal adapter anchor differs: ' + old)
        source = source.replace(old, new, 1)
    result = dict(namespace)
    exec(compile(source, str(RT / 'main_review_watch_v8.py') + ':' + function.__name__, 'exec'), result)
    return result[function.__name__]


def prefix_specs(snapshot):
    production = ROOT / 'curated222939_production_v1'
    prefixes = {str(production / 'completed.jsonl'): snapshot['original_journal_prefix']}
    prefixes.update({str(production / 'batches' / bid / 'terminal_index.jsonl'): p
                     for bid, p in snapshot['terminal_index_prefixes'].items()})
    return prefixes


def capture_offset_binding(snapshot_path, collector_sha):
    root = ROOT / 'curated222939_repair_v1/collector-offset2053650-reconciled-v1'
    need(Path(snapshot_path).resolve() == root / 'snapshot.json', 'New preparation outside exact incident namespace')
    snapshot, ref = recovery.document(snapshot_path)
    collector_ref = {'path': str(RT / 'collector_offset2053650_reconciled_v1.py'), 'sha256': collector_sha}
    need(snapshot['wrapper_ref'] == collector_ref, 'Prepared collector identity differs')
    need(sha(collector_ref['path']) == collector_ref['sha256'], 'Prepared collector execution SHA changed')
    bindings = {}
    for path, prefix in prefix_specs(snapshot).items():
        bid = Path(path).parent.name
        raw_path = root / ('completed.prefix.raw' if Path(path).name == 'completed.jsonl' else bid + '.terminal_index.prefix.raw')
        raw_ref = {'path': str(raw_path), 'sha256': prefix['prefix_sha256']}
        s = Path(path).stat()
        current, historic = [s.st_dev, s.st_ino], prefix['input_identity']
        binding = {'path': path, 'original_prefix': prefix, 'raw_prefix_ref': raw_ref,
            'historical_capture_identity': historic, 'fresh_current_node_identity': current,
            'current_node_hostname': os.uname().nodename,
            'cross_mount_device_rebound_content_verified': current[0] != historic[0], 'prefix_content_SHA_verified': True}
        v7.verify_mount_binding(binding, prefix, raw_ref, path)
        bindings[path] = binding
    keys = snapshot['missing_source_keys']
    return {'schema': SCHEMA, 'snapshot_ref': ref, 'collector_ref': collector_ref,
        'record_refs': [recovery.document(root / 'records' / (key + '.json'))[1] for key in keys],
        'receipt_refs': [recovery.document(root / 'receipts' / (key + '.json'))[1] for key in keys],
        'raw_ref': {'path': str(root / 'badline-2053650.raw'), 'sha256': RAW_SHA},
        'mount_bindings': bindings, 'original_byte_begin': OFFSET, 'original_byte_end': OFFSET + len(RAW),
        'fragment_source_association': 'unknown', 'bytes_contained_complete_JSON_sources': 0,
        'reconciliation_sources': 4, 'source_byte_offsets': [None] * 4,
        'quality_approved': False, 'pending_independent_review': True}


class Preparation:
    def __init__(self, binding, production, manifest):
        self.binding = binding
        root = ROOT / 'curated222939_repair_v1/collector-offset2053650-reconciled-v1'
        self.production = Path(production).resolve()
        need(self.production == ROOT / 'curated222939_production_v1', 'New incident foreign production')
        need(binding['schema'] == SCHEMA and binding['original_byte_begin'] == OFFSET
             and binding['original_byte_end'] == OFFSET + 5 and binding['fragment_source_association'] == 'unknown'
             and type(binding['bytes_contained_complete_JSON_sources']) is int and binding['bytes_contained_complete_JSON_sources'] == 0
             and binding['reconciliation_sources'] == 4 and binding['source_byte_offsets'] == [None] * 4
             and binding['quality_approved'] is False and binding['pending_independent_review'] is True,
             'Exact incident zero encoded sources/unknown association contract differs')
        ref = binding['collector_ref']
        need(ref['path'] == str(RT / 'collector_offset2053650_reconciled_v1.py'), 'Foreign collector execution dependency')
        self.collector = module('private_offset2053650_collector_for_review', ref['path'], ref['sha256'])
        m = self.collector
        need(binding['snapshot_ref']['path'] == str(root / 'snapshot.json'), 'Foreign new snapshot namespace')
        snapshot = checked(binding['snapshot_ref'])
        self.snapshot = snapshot
        expected = {'schema': m.SCHEMA, 'manifest_summary_ref': recovery.document(manifest)[1],
            'selected_original_manifest_member_count': 1000, 'indexed_sources': 813, 'shared_exact_terminals': 809,
            'missing_source_keys': sorted(m.EXPECTED), 'fragment_source_association': 'unknown',
            'fragment_kind': 'not_standalone_JSON_object', 'bytes_contained_complete_JSON_sources': 0,
            'original_decode_error_preserved': True, 'original_journal_modified': False,
            'wrapper_ref': ref, 'committed_prefix_sha256': m.PREFIX_SHA,
            'physical_byte_begin': OFFSET, 'physical_byte_end': OFFSET + 5,
            'source_population_completeness_inferred': False, **m.q.FLAGS}
        need(all(type(snapshot.get(k)) is type(v) and snapshot.get(k) == v for k, v in expected.items()), 'Frozen preparation snapshot contract differs')
        need(snapshot['original_code_refs'] == [{'path': str(RT / name), 'sha256': digest} for name, digest in m.CODE_SHAS.items()], 'Original collector proof dependencies differ')
        snapshots = snapshot['stable_badline_snapshots']
        need(len(snapshots) == 3 and all(x['line_sha256'] == RAW_SHA and x['identity'] == snapshots[0]['identity'] for x in snapshots), 'Preparation lacks three identical original snapshots')
        need(len(snapshots[0]['identity']) == 5 and snapshots[0]['identity'][:2] == m.INPUT_IDENTITY and snapshots[0]['identity'][2] >= OFFSET + 5, 'Preparation original inode/span changed')
        need(binding['raw_ref'] == {'path': str(root / 'badline-2053650.raw'), 'sha256': RAW_SHA} and Path(binding['raw_ref']['path']).read_bytes() == RAW, 'Exact prepared badline bytes/SHA differs')
        need(sha(binding['raw_ref']['path']) == RAW_SHA, 'Prepared badline SHA changed')
        prefixes = prefix_specs(snapshot)
        need(set(binding['mount_bindings']) == set(prefixes), 'Exact journal/two-index fresh mount proof required')
        def mounted_prefix(prefix):
            path = prefix['path']
            need(path in prefixes and prefix == prefixes[path], 'Unadmitted frozen prefix')
            bridge = binding['mount_bindings'][path]
            raw_name = 'completed.prefix.raw' if Path(path).name == 'completed.jsonl' else Path(path).parent.name + '.terminal_index.prefix.raw'
            need(bridge['raw_prefix_ref'] == {'path': str(root / raw_name), 'sha256': prefix['prefix_sha256']}, 'Frozen raw prefix evidence namespace differs')
            v7.verify_mount_binding(bridge, prefix, bridge['raw_prefix_ref'], path)
            return Path(bridge['raw_prefix_ref']['path']).read_bytes()
        # Private collector module only: retain historical prefix refs, bind current mount separately.
        m.c.verify_prefix = mounted_prefix
        self.records, self.values = [], {}
        with m.c.verified_hash_cache() as cache:
            evidence = m.r.Evidence(self.production, manifest)
            raw, index_raw, journal_ref, index_refs, _, indexed, missing = m.check_population(self.production, evidence)
            need(snapshot['original_journal_prefix'] == journal_ref and snapshot['terminal_index_prefixes'] == index_refs, 'Frozen finite union proof changed')
            record_refs, receipt_refs = binding['record_refs'], binding['receipt_refs']
            need(len(record_refs) == 4 and len(receipt_refs) == 4, 'Four independent record/receipt refs required')
            for key, record_ref, receipt_ref in zip(missing, record_refs, receipt_refs):
                need(record_ref['path'] == str(root / 'records' / (key + '.json')) and receipt_ref['path'] == str(root / 'receipts' / (key + '.json')), 'Exact source record/receipt namespace differs')
                record, receipt = checked(record_ref), checked(receipt_ref)
                refs, source = m.full_checked(indexed[key], evidence)
                expected_record = {'schema': m.SCHEMA, 'source_key': key, 'hint': indexed[key],
                    'completed_row_sha256': m.q.digest(indexed[key]), 'origin': 'authoritative_terminal_reconciliation',
                    'original_byte_offset': None, 'fragment_source_association': 'unknown', 'absence_from_that_exact_snapshot': True,
                    'original_journal_snapshot_ref': binding['snapshot_ref'], 'evidence': refs, **source,
                    'quality_approved': False, 'pending_independent_review': True, **m.q.FLAGS}
                need(record == expected_record, 'Immutable independent record/member/current media/Evidence differs')
                need(receipt == {'schema': m.SCHEMA, 'source_key': key, 'record_ref': record_ref,
                    'original_byte_offset': None, 'origin': 'authoritative_terminal_reconciliation',
                    'original_journal_snapshot_ref': binding['snapshot_ref'], 'fragment_source_association': 'unknown',
                    'retry_budget_reset': False, **m.q.FLAGS}, 'Independent reconciliation receipt changed')
                self.values[key] = record
                self.records.append((key, record_ref, record))
            need(sum(r['data_complete_candidate'] for r in self.values.values()) == 2, 'Two complete/two failed original terminals required')
            cache.verify_all()
            self.cache = cache
        self.check_physical(OFFSET, RAW)

    def check_physical(self, begin, line):
        need(begin == OFFSET and line == RAW and hashlib.sha256(line).hexdigest() == RAW_SHA, 'Exact five-byte incident offset/bytes changed')
        prefixes = prefix_specs(self.snapshot)
        for path, prefix in prefixes.items():
            bridge = self.binding['mount_bindings'][path]
            v7.verify_mount_binding(bridge, prefix, bridge['raw_prefix_ref'], path)
        stream = self.production / 'completed.jsonl'
        identity = self.binding['mount_bindings'][str(stream)]['fresh_current_node_identity']
        need(recovery.prefix_sha(stream, OFFSET, identity) == self.snapshot['committed_prefix_sha256'], 'Exact incident committed prefix changed')
        recovery.stable_badline(stream, begin, line, identity, .05)
        need(recovery.prefix_sha(stream, OFFSET, identity) == self.snapshot['committed_prefix_sha256'], 'Incident prefix changed after three fresh snapshots')
        self.cache.verify_all()


class Incident(v7.Incident):
    def __init__(self, production, manifest, binding, empty_binding=None, offset_binding=None):
        super().__init__(production, manifest, binding, empty_binding)
        self.preparation = Preparation(offset_binding, production, manifest) if offset_binding else None
        self.fragment_row_offsets = []

    def expand(self, begin, line):
        if begin != OFFSET:
            return super().expand(begin, line)
        need(self.preparation is not None, 'Exact fragment lacks frozen full reconciliation preparation')
        self.preparation.check_physical(begin, line)
        if begin not in self.fragment_row_offsets:
            self.fragment_row_offsets.append(begin)
        return []


_scan = derive(v7.RawIndex.scan, [
    ('begin in recovery.KNOWN or begin == EMPTY_OFFSET', 'begin in recovery.KNOWN or begin == EMPTY_OFFSET or begin == OFFSET'),
    ('if count == 0 and start not in self.empty_row_offsets:', 'if count == 0 and start == EMPTY_OFFSET and start not in self.empty_row_offsets:')], {**vars(v7), 'OFFSET': OFFSET})


class RawIndex(v7.RawIndex):
    scan = _scan

    def reconcile(self, manifest, expected_ref=None):
        dirty = super().reconcile(manifest, expected_ref)
        if self.incident and self.incident.preparation:
            dirty |= self.reconcile_offset()
        return dirty

    def reconcile_offset(self):
        proof = self.incident.preparation
        need(OFFSET in self.incident.fragment_row_offsets and self.indexed_offset >= OFFSET + 5, 'Physical fragment must be verified before independent reconciliation')
        staged = []
        for key, ref, record in proof.records:
            hint = record['hint']
            bid, sid = hint['batch_id'], hint['id']
            need(bid in self.allowed_batches and sid in self.membership.batch(bid), 'Reconciliation foreign original membership')
            old = self.raw[bid].get(sid)
            need(old is None or old['hint'] == hint, 'Reconciliation conflicts with current raw terminal')
            row = terminal_record(self.production, hint, self.membership)
            need(sid not in self.completed[bid] or self.completed[bid][sid] == row, 'Reconciliation changes verified labels')
            staged.append((key, ref, hint, row, old))
        self.check_prefix()
        proof.cache.verify_all()
        dirty = set()
        for key, ref, hint, row, old in staged:
            bid, sid = hint['batch_id'], hint['id']
            if old is None:
                self.raw[bid][sid] = {'hint': hint, 'byte_begin': None, 'byte_end': None, 'raw_sha256': None,
                    'origin': 'explicit_terminal_reconciliation', 'record_ref': ref, 'fragment_source_association': 'unknown'}
                dirty.add(bid)
            self.completed[bid][sid] = row
            self.verified_keys.add((bid, sid))
            self.restored_sources[key] = {'record_ref': ref, 'original_byte_offset': None,
                'original_consumer_terminal_ref': proof.values[key]['evidence']['consumer_terminal_ref'],
                'batch_id': bid, 'input_candidate_sha256': recovery.q.digest(hint), 'label_sha256': row.get('label_sha256'),
                'fragment_source_association': 'unknown', 'origin': 'authoritative_terminal_reconciliation'}
        return dirty


def progress(index, out, edition, bids, total):
    value = v7.progress(index, out, edition, bids, total)
    value['schema'] = 'curated_independent_review_watch_v8'
    value['fragment_row_offsets'] = index.incident.fragment_row_offsets if isinstance(index.incident, Incident) else []
    value['fragment_bytes_contained_complete_JSON_sources'] = 0
    value['fragment_source_association'] = 'unknown'
    value['offset2053650_preparation_ref'] = index.incident.preparation.binding['snapshot_ref'] if isinstance(index.incident, Incident) and index.incident.preparation else None
    value['original_watch_v7_ref'] = {'path': str(RT / 'main_review_watch_v7.py'), 'sha256': V7_SHA}
    value['execution_watch_v8_ref'] = {'path': str(Path(__file__)), 'sha256': sha(__file__)}
    return value


def validate_binding(binding, args, membership):
    bids = v7.validate_binding(binding, args, membership)
    need(binding.get('offset2053650') is not None and binding.get('reconciliation_manifest_ref') is not None, 'Full old and new independent reconciliation proofs required')
    return bids


main = derive(v7.main, [("binding['incident'], binding['empty_incident'])", "binding['incident'], binding['empty_incident'], binding['offset2053650'])")],
    {**vars(v7), 'Incident': Incident, 'RawIndex': RawIndex, 'progress': progress, 'validate_binding': validate_binding})

if __name__ == '__main__':
    main()
