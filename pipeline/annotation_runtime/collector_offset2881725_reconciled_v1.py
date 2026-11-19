#!/usr/bin/env python3
"""Exact 3-byte incident, finite 1000-source reconciliation; no model calls."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

RT = Path(__file__).resolve().parent
OLD_WRAPPER_SHA = '09c32ccbceb1436732a68c4915b2d6ceade98c41b25b856bcdf75f43dca67b85'
OLD_WRAPPER = RT / 'collector_offset2053650_reconciled_v1.py'
if hashlib.sha256(OLD_WRAPPER.read_bytes()).hexdigest() != OLD_WRAPPER_SHA:
    raise RuntimeError('Frozen inherited collector changed')
spec = importlib.util.spec_from_file_location('private_inherited_offset2881725', OLD_WRAPPER)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
r, q, c, v2 = old.r, old.q, old.c, old.v2
CODE_SHAS = {**old.CODE_SHAS, OLD_WRAPPER.name: OLD_WRAPPER_SHA}
SELF_REF = q.identity(__file__)
old_recover_line = r.recover_line
SCHEMA = 'exact_terminal_reconciliation_offset2881725_v1'
OFFSET = 2881725
RAW = bytes.fromhex('227d0a')
RAW_SHA = '709019e51e9f27b1ea488ede78241cdc3676186d65840f4e54305998d6226d3a'
INPUT_IDENTITY = [1048620, 9895974566444]
PREFIX_SHA = '16f5e15c28c471a1b5de0ede470424e9f6e863fcab763383f2eae940df3ef3e9'
STATE_SHA = '014dd26a29b528936fab9a3c77bec3b13d7e2c58d2f97958944156b5023ed2ed'
INDEXED_COUNT, SHARED_COUNT = 1000, 999
JOURNAL_LENGTH = 3703687
JOURNAL_SHA = '232403d4b388ed6f218ddcca4038426fca909e62bbc9bfe9585396415286b9e0'
INDEXES = {
    'train-b000007': ([1048620, 9895976196277], 21000, '967e6824fd3523bacbae2844086a3530c274432289886b7cb5ae4baa3efdf335'),
    'train-b000008': ([1048620, 9895976430614], 21000, '007c7823f7a254e345a5bfaf843b90e596fa7f2c3ab2dcc8a6aaaad855d6a32d'),
}
EXPECTED = {
    'train/yvMhDxmBblEUAFto': ('train-b000007', 3822, '82c7b37dbf93492d5d364753f93cef10356171f60c8d400cd7da289dbfe52002'),
}
HISTORY_SNAPSHOT_SHA = '5704be5f91075df26a37f00b7b4b9c97a78e6ab1dd5b72c29e8655c3cb9cb4d6'
HISTORY_SHAS = {
    'train/QIWvouIspDKiagWW': ('6965b2fc985b75a6ff20edb0cd9bae019bfcbba650e03f41440135f4a62858e4', 'd3ceb93063bbdbd6266556a58f9c774bf67a50fc0b381b1d397d7b60330b7991'),
    'train/SQQbWIbvUGqzZMGP': ('65c6113c8837740daa504fa1483c151d2a4bda7eb4ca6d3c888c83d79f78ef74', 'c3f7598b7140ad0037abe23aa7c7a555f64a84881e60a07f63c108f83ffc470b'),
    'train/VHFSAcOmzvvCYiGJ': ('c88d7885d1f848fbebf66aa551660bf73a9d3daa41a926ba76416fa4a765a0dc', 'dc28157608edf5421eaf01850d61d4cae85e3900c6ab9069744ee4013abd3528'),
    'train/afgToKIsEUTkAyDP': ('bd2b5a502a470f0207cc0743a4fd4537d181bddcc488bc82d31b1bcd52107f00', '2013ec348c5c60d70ae0b664ccc3c6c1d2056d43537b1e98310395a5a7a84cb7'),
}


def verify_code():
    for name, sha in CODE_SHAS.items():
        q.verify_bytes({'path': str(RT / name), 'sha256': sha})
    q.verify_bytes(SELF_REF)


def historical_seen(production, out):
    # Inherit the four already-admitted immutable records, not their 925-artifact revalidation.
    _, _, root = old.namespace(production, out)
    snapshot, snapshot_ref = r.document(root / 'snapshot.json', HISTORY_SNAPSHOT_SHA)
    r.same(snapshot['schema'], old.SCHEMA, 'Inherited reconciliation schema changed')
    r.same(snapshot['wrapper_ref'], {'path': str(OLD_WRAPPER), 'sha256': OLD_WRAPPER_SHA}, 'Inherited reviewed collector ref changed')
    r.same(snapshot['missing_source_keys'], sorted(old.EXPECTED), 'Inherited reconciliation source union changed')
    r.same((snapshot['fragment_source_association'], snapshot['bytes_contained_complete_JSON_sources']),
           ('unknown', 0), 'Inherited fragment source association changed')
    seen = {}
    for key, (record_sha, receipt_sha) in HISTORY_SHAS.items():
        value, record_ref = r.document(root / 'records' / (key + '.json'), record_sha)
        receipt, _ = r.document(root / 'receipts' / (key + '.json'), receipt_sha)
        r.same(value['source_key'], key, 'Inherited source key changed')
        r.same(q.safe_key(value['hint']), key, 'Inherited hint source changed')
        r.same(value['completed_row_sha256'], q.digest(value['hint']), 'Inherited terminal digest changed')
        r.same(value['original_journal_snapshot_ref'], snapshot_ref, 'Inherited absence snapshot changed')
        r.same((value['original_byte_offset'], value['fragment_source_association']), (None, 'unknown'), 'Inherited source claimed physical offset')
        r.same(receipt['record_ref'], record_ref, 'Inherited receipt record changed')
        r.same(receipt['original_journal_snapshot_ref'], snapshot_ref, 'Inherited receipt absence changed')
        r.same(receipt['retry_budget_reset'], False, 'Inherited retry budget reset claim')
        seen[key] = value['completed_row_sha256']
    return seen


def namespace(production, out):
    production, out = Path(production).resolve(), Path(out).resolve()
    r.same(production, RT.parent / 'curated222939_production_v1', 'Incident production namespace changed')
    r.same(out, RT.parent / 'curated222939_repair_v1', 'Incident original queue namespace changed')
    return production, out, out / 'collector-offset2881725-reconciled-v1'


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
            elif begin == old.OFFSET and line == old.RAW and hashlib.sha256(line).hexdigest() == old.RAW_SHA:
                values = []  # Previously reconciled source records are inherited separately.
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
    for key, digest in historical_seen(production, RT.parent / 'curated222939_repair_v1').items():
        if key in inherited:
            r.same(inherited[key], digest, 'Inherited journal/reconciled terminal differs')
        inherited[key] = digest
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
    state, state_ref = r.document(out / 'collector.recovery.state.json', STATE_SHA)
    r.same((state['offset'], state['input_identity'], state['input_prefix_sha256'], state['global_revision']),
           (OFFSET, INPUT_IDENTITY, PREFIX_SHA, revision), 'Preparation must inherit the exact original recovery checkpoint')
    r.same(state['recovered_badline_offsets'], [r.BAD_OFFSET, v2.EMPTY_OFFSET, old.OFFSET], 'Original three reviewed incidents changed')
    r.same(state['input_path'], str(production / 'completed.jsonl'), 'Preparation inherited original input path changed')
    r.document(state['origin_state_ref']['path'], state['origin_state_ref']['sha256'])
    snapshots = r.stable_badline(production / 'completed.jsonl', OFFSET, RAW, INPUT_IDENTITY, stable_seconds)
    verify_incident(production, OFFSET, RAW, snapshots)
    with c.verified_hash_cache() as cache:
        verify_code()
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
        'selected_original_manifest_member_count': 1000, 'indexed_sources': INDEXED_COUNT, 'shared_exact_terminals': SHARED_COUNT,
        'missing_source_keys': missing, 'fragment_source_association': 'unknown',
        'fragment_kind': 'not_standalone_JSON_object', 'bytes_contained_complete_JSON_sources': 0,
        'original_decode_error_preserved': True, 'original_journal_modified': False,
        'original_code_refs': code_refs, 'wrapper_ref': SELF_REF,
        'inherited_reviewed_incident_offsets': [r.BAD_OFFSET, v2.EMPTY_OFFSET, old.OFFSET],
        'historical_reconciled_sources_inherited_without_rechecking_fullchain': len(HISTORY_SHAS),
        'committed_prefix_sha256': PREFIX_SHA, 'physical_byte_begin': OFFSET, 'physical_byte_end': OFFSET + len(RAW),
        'stable_badline_snapshots': snapshots, 'source_population_completeness_inferred': False, **q.FLAGS}
    snapshot_path = root / 'snapshot.json'
    if snapshot_path.exists():
        saved_snapshot = q.load(snapshot_path)
        r.same({k:v for k,v in saved_snapshot.items() if k != 'stable_badline_snapshots'},
               {k:v for k,v in snapshot.items() if k != 'stable_badline_snapshots'}, 'Immutable reconciliation snapshot drifted')
        snapshot = saved_snapshot
    r.save_raw_once(root / 'completed.prefix.raw', raw)
    for batch in INDEXES:
        r.save_raw_once(root / (batch + '.terminal_index.prefix.raw'), index_raw[batch])
    r.save_raw_once(root / 'badline-2881725.raw', RAW)
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
    return {'prepared_source_keys': missing, 'prepared_sources': len(missing),
        'prepared_complete_sources': sum(source['data_complete_candidate'] for _, source in checked.values()),
        'prepared_failed_sources': sum(not source['data_complete_candidate'] for _, source in checked.values()),
        'indexed_sources': INDEXED_COUNT, 'shared_exact_terminals': SHARED_COUNT, 'original_Evidence_terminal_full_checked_sources': len(checked),
        'verified_artifact_count': artifact_count, 'metadata_root': str(root), 'original_collector_state_modified': False,
        'original_journal_modified': False, 'model_calls': 0, 'retry_budget_reset': False, **q.FLAGS}


def load_verified_rows(production, out, evidence):
    """Read-only source API; provenance is independent of physical fragment expansion."""
    production, out, root = namespace(production, out)
    snapshot, snapshot_ref = r.document(root / 'snapshot.json')
    verify_incident(production, OFFSET, RAW, snapshot['stable_badline_snapshots'])
    r.same(snapshot['schema'], SCHEMA, 'Unreviewed reconciliation snapshot schema')
    r.same(snapshot['missing_source_keys'], sorted(EXPECTED), 'Prepared source claim differs')
    r.same(snapshot['manifest_summary_ref'], evidence.summary_ref, 'Prepared original manifest differs')
    r.same(snapshot['fragment_source_association'], 'unknown', 'Fragment must not claim any source association')
    r.same(snapshot['bytes_contained_complete_JSON_sources'], 0, 'Fragment must not claim complete JSON sources')
    r.same(snapshot['wrapper_ref'], SELF_REF, 'Prepared wrapper source changed')
    with c.verified_hash_cache() as cache:
        verify_code()
        raw, index_raw, journal_ref, index_refs, _, indexed, missing = check_population(production, evidence)
        r.same(snapshot['original_journal_prefix'], journal_ref, 'Prepared absence journal snapshot differs')
        r.same(snapshot['terminal_index_prefixes'], index_refs, 'Prepared original index snapshot differs')
        r.same((root / 'completed.prefix.raw').read_bytes(), raw, 'Saved original journal prefix evidence changed')
        for batch in INDEXES:
            r.same((root / (batch + '.terminal_index.prefix.raw')).read_bytes(), index_raw[batch], 'Saved original terminal index evidence changed')
        r.same((root / 'badline-2881725.raw').read_bytes(), RAW, 'Saved original badline evidence changed')
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
        'reconstructed_from_authoritative_terminal_index': len(rows), 'original_source_byte_offsets': [None] * len(rows),
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
