#!/usr/bin/env python3
"""Exact physical incident + bounded authoritative terminal reconciliation; CPU only."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

RT = Path(__file__).resolve().parent
CODE_SHAS = {
    'curated_repair_collector_emptyline_v2.py': '5c6c01e2f9ad70fc6515ef43384e81af2cface3b52f716c67496a24e1565752a',
    'curated_repair_collector_recovery_v1.py': 'f85efb1441930cfc31bfe4cafacf25e0fdb18aeb9da6ad80fb540b58519237cd',
    'source_repair_gpu_v2/curated_repair_queue_v1.py': '94b00ad0b503053f4b9bd7f7c08097bf9b6480f8a9aa248e971126a149c78b4c',
    'curated_terminal_reconciliation_v1.py': '1ef0825d120775c25f628ed1c1374ac008de41b19f1397dfdf6f8b834038d2a8',
}
for name, sha in CODE_SHAS.items():
    if hashlib.sha256((RT / name).read_bytes()).hexdigest() != sha:
        raise RuntimeError('Frozen collector dependency changed: ' + name)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, RT / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


v2 = module('private_emptyline_chain_offset2053650', 'curated_repair_collector_emptyline_v2.py')
r = v2.recovery
c = module('private_bounded_prefix_offset2053650', 'curated_terminal_reconciliation_v1.py')
# Use the original byte-cache implementation with this private collector's Evidence.
c.recovery, c.q = r, r.q
q = r.q
old_recover_line = r.recover_line
SCHEMA = 'exact_terminal_reconciliation_offset2053650_v1'
OFFSET = 2053650
RAW = bytes.fromhex('7465227d0a')
RAW_SHA = '7561b973a7f36b54382420878e894bbb5cd9e7bcd7e5a8db6ec27f883315593c'
INPUT_IDENTITY = [1048620, 9895974566444]
PREFIX_SHA = '7fb515967552cd1916581019b1a13f1f83f07e2da263424a4ed6af93ec80864c'
INDEXED_COUNT = 813
SHARED_COUNT = 809
JOURNAL_LENGTH = 2309736
JOURNAL_SHA = 'e28fd17c77af4e970c449150c008a0acb47b1cb0e227c526af1c58d57b87ea5a'
INDEXES = {
    'train-b000005': ([1048620, 9895975746592], 21000, 'c2e7b53dcb716bab1235cea88d7cf22a6ec72458f2dd4751642fe17491d8192e'),
    'train-b000006': ([1048620, 9895975979936], 13146, 'b07e272c6c1c5601f3defe7a18ac23ceea29ea265beae14411e8cd5a7ccdf477'),
}
EXPECTED = {
    'train/afgToKIsEUTkAyDP': ('train-b000005', 2723, '0f5b43a56bbc660d43f11d2cbf0fa6f3954ca516d433400da23ca56c362a6079'),
    'train/SQQbWIbvUGqzZMGP': ('train-b000005', 2775, '709dc468c11b2d5bf154039e4a391ebc3feca1cabd6165d2ee6bfa1c2a81e7bd'),
    'train/QIWvouIspDKiagWW': ('train-b000005', 2883, '166d4a8fcd806bb906f3762609ca53680c656a53cf51bb9fab424d113256dd75'),
    'train/VHFSAcOmzvvCYiGJ': ('train-b000006', 3140, '244156218a17f42782b25347685156b2dc67951d4211040e9f4277a507f9974c'),
}


def namespace(production, out):
    production, out = Path(production).resolve(), Path(out).resolve()
    r.same(production, RT.parent / 'curated222939_production_v1', 'Incident production namespace changed')
    r.same(out, RT.parent / 'curated222939_repair_v1', 'Incident original queue namespace changed')
    return production, out, out / 'collector-offset2053650-reconciled-v1'


def verify_incident(production, begin, line, snapshots):
    if begin != OFFSET or line != RAW or hashlib.sha256(line).hexdigest() != RAW_SHA:
        raise r.UnrecoverableLine('Unreviewed incident bytes/offset; retain original cursor')
    if (len(snapshots) != 3 or any(s['line_sha256'] != RAW_SHA for s in snapshots)
            or any(s['identity'] != snapshots[0]['identity'] for s in snapshots)):
        raise r.StreamBusy('Incident lacks three equal original stable byte snapshots')
    for sample in snapshots:
        if sample['identity'][:2] != INPUT_IDENTITY or sample['identity'][2] < OFFSET + len(RAW):
            raise q.SourceChanged('Incident snapshots have changed inode/truncated bytes')
    queue = production / 'completed.jsonl'
    r.same(r.prefix_sha(queue, OFFSET, INPUT_IDENTITY), PREFIX_SHA, 'Original committed incident prefix changed')
    r.same(c.prefix_bytes(queue, INPUT_IDENTITY, OFFSET + len(RAW))[OFFSET:], RAW, 'Original incident bytes changed after snapshots')


def frozen_prefix(path, identity, size, sha):
    ref = {'path': str(path), 'input_identity': identity, 'byte_length': size, 'prefix_sha256': sha}
    return c.verify_prefix(ref), ref


def journal_sources(raw, production):
    rows, inherited, stream = {}, {}, io.BytesIO(raw)
    while stream.tell() < len(raw):
        begin = stream.tell()
        line = stream.readline()
        if not line.endswith(b'\n'):
            raise r.UnrecoverableLine('Frozen journal prefix has incomplete tail')
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('Original terminal must be an object')
            values = [value]
        except (ValueError, UnicodeError):
            if begin == OFFSET and line == RAW:
                values = []  # Fragment association is unknown, not a reconstructed source.
            elif begin == v2.EMPTY_OFFSET and line == b'\n':
                values = []
            elif begin == r.BAD_OFFSET and hashlib.sha256(line).hexdigest() == r.BAD_SHA:
                values = [r.document(production / 'sources/train' / sid / 'consumer.terminal.json')[0]
                          for sid in r.KNOWN[r.BAD_OFFSET]['ids']]
                boundary = len(line) - len(q.encoded(values[1]))
                if boundary <= 0 or line[:boundary] != q.encoded(values[0])[:boundary] or line[boundary:] != q.encoded(values[1]):
                    raise q.SourceChanged('Prior reviewed splice source bytes changed')
            else:
                raise r.UnrecoverableLine('Unreviewed malformed frozen journal line at ' + str(begin))
        for row in values:
            key = q.safe_key(row)
            if key in rows:
                r.same(rows[key], row, 'Journal duplicate terminal revision differs')
            rows[key] = row
            if begin < OFFSET:
                inherited[key] = q.digest(row)
    return rows, inherited


def check_population(production, evidence):
    r.same((evidence.summary['total'], evidence.summary['batch_size'], len(evidence.batches)),
           (222939, 500, 448), 'Original authoritative full manifest denominator changed')
    raw, journal_ref = frozen_prefix(production / 'completed.jsonl', INPUT_IDENTITY, JOURNAL_LENGTH, JOURNAL_SHA)
    journal, inherited = journal_sources(raw, production)
    index_raw, index_refs, indexed = {}, {}, {}
    for batch, (identity, size, sha) in INDEXES.items():
        r.same((evidence.batches[batch]['split'], evidence.batches[batch]['count']), ('train', 500), 'Original selected 500-member batch changed')
        index_raw[batch], index_refs[batch] = frozen_prefix(production / 'batches' / batch / 'terminal_index.jsonl', identity, size, sha)
        values, _ = c.decode_index(index_raw[batch], production, batch, evidence)
        if set(indexed) & set(values):
            raise q.SourceChanged('Source terminal indexed across two original batches')
        indexed.update(values)
    r.same(len(indexed), INDEXED_COUNT, 'Frozen finite indexed source denominator changed')
    shared = set(indexed) & set(journal)
    r.same(len(shared), SHARED_COUNT, 'Frozen finite journal/index source union changed')
    for key in shared:
        r.same(indexed[key], journal[key], 'Captured journal differs from current immutable terminal: ' + key)
    missing = sorted(set(indexed) - set(journal))
    r.same(missing, sorted(EXPECTED), 'Frozen finite missing source claim differs')
    return raw, index_raw, journal_ref, index_refs, inherited, indexed, missing


def full_checked(row, evidence):
    key = q.safe_key(row)
    batch, ordinal, sha = EXPECTED[key]
    verified, refs = evidence.terminal(row['id'], row['split'], batch)
    r.same(verified, row, 'Reconciliation terminal differs from original full Evidence')
    r.same(refs['consumer_terminal_ref']['sha256'], sha, 'Exact original consumer terminal SHA changed')
    member, manifest_ref = evidence.member(row['id'], row['split'], batch)
    producer, _ = r.document(refs['producer_terminal_ref']['path'], refs['producer_terminal_ref']['sha256'])
    r.same((member['source_ordinal'], producer['source_ordinal']), (ordinal, ordinal), 'Original source ordinal changed')
    complete = row.get('joint_complete') is True and row['status'] == 'joint_candidate'
    stages = []
    if not complete:
        job = q.classify(row, evidence.production)
        q.verify_job(job)
        stages = job['stages']
    return refs, {'batch_manifest_ref': manifest_ref, 'original_source_ordinal': ordinal,
                  'original_csv_ref': member['source_refs']['csv'], 'data_complete_candidate': complete,
                  'original_repair_stages': stages}


def prepare(production, out, revision, manifest, stable_seconds=0.25):
    production, out, root = namespace(production, out)
    state, state_ref = r.document(out / 'collector.recovery.state.json')
    r.same((state['offset'], state['input_identity'], state['input_prefix_sha256'], state['global_revision']),
           (OFFSET, INPUT_IDENTITY, PREFIX_SHA, revision), 'Preparation must inherit the exact original recovery checkpoint')
    r.same(state['recovered_badline_offsets'], [r.BAD_OFFSET, v2.EMPTY_OFFSET], 'Original two reviewed incidents changed')
    r.same(state['input_path'], str(production / 'completed.jsonl'), 'Preparation inherited original input path changed')
    r.document(state['origin_state_ref']['path'], state['origin_state_ref']['sha256'])
    snapshots = r.stable_badline(production / 'completed.jsonl', OFFSET, RAW, INPUT_IDENTITY, stable_seconds)
    verify_incident(production, OFFSET, RAW, snapshots)
    with c.verified_hash_cache() as cache:
        evidence = r.Evidence(production, manifest)
        r.same(evidence.summary_ref, state['manifest_summary_ref'], 'Original inherited manifest summary changed')
        raw, index_raw, journal_ref, index_refs, inherited, indexed, missing = check_population(production, evidence)
        r.same(inherited, state['seen'], 'Inherited source seen differs from exact original prefix source union')
        checked = {key: full_checked(indexed[key], evidence) for key in missing}
        for ref in [journal_ref, *index_refs.values()]:
            c.verify_prefix(ref)
        cache.verify_all()
        artifact_count = len(cache.files)
    r.same(q.identity(out / 'collector.recovery.state.json'), state_ref, 'Original collector state changed during preparation')
    code_refs = [{'path': str(RT / name), 'sha256': sha} for name, sha in CODE_SHAS.items()]
    snapshot = {'schema': SCHEMA, 'original_journal_prefix': journal_ref, 'terminal_index_prefixes': index_refs,
        'original_collector_state_ref': state_ref, 'manifest_summary_ref': evidence.summary_ref,
        'selected_original_manifest_member_count': 1000, 'indexed_sources': 813, 'shared_exact_terminals': 809,
        'missing_source_keys': missing, 'fragment_source_association': 'unknown',
        'fragment_kind': 'not_standalone_JSON_object', 'bytes_contained_complete_JSON_sources': 0,
        'original_decode_error_preserved': True, 'original_journal_modified': False,
        'original_code_refs': code_refs, 'wrapper_ref': q.identity(__file__),
        'committed_prefix_sha256': PREFIX_SHA, 'physical_byte_begin': OFFSET, 'physical_byte_end': OFFSET + len(RAW),
        'stable_badline_snapshots': snapshots, 'source_population_completeness_inferred': False, **q.FLAGS}
    snapshot_path = root / 'snapshot.json'
    if snapshot_path.exists():
        old = q.load(snapshot_path)
        r.same({k:v for k,v in old.items() if k != 'stable_badline_snapshots'},
               {k:v for k,v in snapshot.items() if k != 'stable_badline_snapshots'}, 'Immutable reconciliation snapshot drifted')
        snapshot = old
    r.save_raw_once(root / 'completed.prefix.raw', raw)
    for batch in INDEXES:
        r.save_raw_once(root / (batch + '.terminal_index.prefix.raw'), index_raw[batch])
    r.save_raw_once(root / 'badline-2053650.raw', RAW)
    q.save_once(snapshot_path, snapshot)
    snapshot_ref = q.identity(snapshot_path)
    for key in missing:
        refs, source = checked[key]
        value = {'schema': SCHEMA, 'source_key': key, 'hint': indexed[key], 'completed_row_sha256': q.digest(indexed[key]),
            'origin': 'authoritative_terminal_reconciliation', 'original_byte_offset': None,
            'fragment_source_association': 'unknown', 'absence_from_that_exact_snapshot': True,
            'original_journal_snapshot_ref': snapshot_ref, 'evidence': refs, **source,
            'quality_approved': False, 'pending_independent_review': True, **q.FLAGS}
        path = root / 'records' / (key + '.json')
        q.save_once(path, value)
        q.save_once(root / 'receipts' / (key + '.json'), {'schema': SCHEMA, 'source_key': key,
            'record_ref': q.identity(path), 'original_byte_offset': None,
            'origin': 'authoritative_terminal_reconciliation', 'original_journal_snapshot_ref': snapshot_ref,
            'fragment_source_association': 'unknown', 'retry_budget_reset': False, **q.FLAGS})
    return {'prepared_source_keys': missing, 'prepared_sources': 4, 'prepared_complete_sources': 2,
        'prepared_failed_sources': 2, 'original_Evidence_terminal_full_checked_sources': len(checked),
        'verified_artifact_count': artifact_count, 'metadata_root': str(root), 'original_collector_state_modified': False,
        'original_journal_modified': False, 'model_calls': 0, 'retry_budget_reset': False, **q.FLAGS}


def load_verified_rows(production, out, evidence):
    """Read-only four-row API; provenance is independent of physical fragment expansion."""
    production, out, root = namespace(production, out)
    snapshot, snapshot_ref = r.document(root / 'snapshot.json')
    verify_incident(production, OFFSET, RAW, snapshot['stable_badline_snapshots'])
    r.same(snapshot['schema'], SCHEMA, 'Unreviewed reconciliation snapshot schema')
    r.same(snapshot['missing_source_keys'], sorted(EXPECTED), 'Prepared source claim differs')
    r.same(snapshot['manifest_summary_ref'], evidence.summary_ref, 'Prepared original manifest differs')
    r.same(snapshot['fragment_source_association'], 'unknown', 'Fragment must not claim any source association')
    r.same(snapshot['bytes_contained_complete_JSON_sources'], 0, 'Fragment must not claim complete JSON sources')
    r.same(snapshot['wrapper_ref'], q.identity(__file__), 'Prepared wrapper source changed')
    with c.verified_hash_cache() as cache:
        raw, index_raw, journal_ref, index_refs, _, indexed, missing = check_population(production, evidence)
        r.same(snapshot['original_journal_prefix'], journal_ref, 'Prepared absence journal snapshot differs')
        r.same(snapshot['terminal_index_prefixes'], index_refs, 'Prepared original index snapshot differs')
        r.same((root / 'completed.prefix.raw').read_bytes(), raw, 'Saved original journal prefix evidence changed')
        for batch in INDEXES:
            r.same((root / (batch + '.terminal_index.prefix.raw')).read_bytes(), index_raw[batch], 'Saved original terminal index evidence changed')
        r.same((root / 'badline-2053650.raw').read_bytes(), RAW, 'Saved original badline evidence changed')
        rows, record_refs = [], []
        for key in missing:
            path = root / 'records' / (key + '.json')
            value, record_ref = r.document(path)
            receipt, _ = r.document(root / 'receipts' / (key + '.json'))
            refs, source = full_checked(indexed[key], evidence)
            expected = {'schema': SCHEMA, 'source_key': key, 'hint': indexed[key], 'completed_row_sha256': q.digest(indexed[key]),
                'origin': 'authoritative_terminal_reconciliation', 'original_byte_offset': None,
                'fragment_source_association': 'unknown', 'absence_from_that_exact_snapshot': True,
                'original_journal_snapshot_ref': snapshot_ref, 'evidence': refs, **source,
                'quality_approved': False, 'pending_independent_review': True, **q.FLAGS}
            r.same(value, expected, 'Immutable authoritative source record/evidence changed')
            r.same(receipt, {'schema': SCHEMA, 'source_key': key, 'record_ref': record_ref, 'original_byte_offset': None,
                'origin': 'authoritative_terminal_reconciliation', 'original_journal_snapshot_ref': snapshot_ref,
                'fragment_source_association': 'unknown', 'retry_budget_reset': False, **q.FLAGS}, 'Immutable source receipt changed')
            rows.append(indexed[key])
            record_refs.append(record_ref)
        for ref in [journal_ref, *index_refs.values()]:
            c.verify_prefix(ref)
        cache.verify_all()
    provenance = {'schema': SCHEMA, 'snapshot_ref': snapshot_ref, 'source_record_refs': record_refs,
        'source_keys': [q.safe_key(row) for row in rows], 'origin': 'authoritative_terminal_reconciliation',
        'fragment_source_association': 'unknown', 'bytes_contained_complete_JSON_sources': 0,
        'reconstructed_from_authoritative_terminal_index': 4, 'original_source_byte_offsets': [None] * 4,
        'original_journal_modified': False, 'quality_approved': False, **q.FLAGS}
    return rows, provenance


def recover_line(production, out, begin, line, evidence, snapshots, rules=r.KNOWN):
    if begin != OFFSET:
        return old_recover_line(production, out, begin, line, evidence, snapshots, rules)
    production, out, root = namespace(production, out)
    verify_incident(production, begin, line, snapshots)
    rows, provenance = load_verified_rows(production, out, evidence)
    authorization = {**provenance, 'physical_byte_begin': OFFSET,
        'physical_byte_end': OFFSET + len(RAW), 'raw_sha256': RAW_SHA}
    q.save_once(root / 'physical-cursor-authorization.json', authorization)
    return rows  # Exact original dictionaries only; original admit owns all dedup/count/job/cursor updates.


r.recover_line = recover_line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('production', 'out', 'global-revision', 'manifest'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--stable-seconds', type=float, default=0.25)
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args()
    if args.stable_seconds < 0 or args.poll_seconds < 0 or (args.prepare and args.watch):
        parser.error('Invalid stability/poll delay or prepare/watch combination')
    if args.prepare:
        result = prepare(args.production, args.out, args.global_revision, args.manifest, args.stable_seconds)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return
    while True:
        result = r.collect(args.production, args.out, args.global_revision, args.manifest, args.stable_seconds)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if not args.watch:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
