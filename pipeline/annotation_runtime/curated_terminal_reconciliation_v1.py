#!/usr/bin/env python3
"""One-shot CPU reconciliation; original journals, terminals and labels stay untouched."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import uuid

RT = Path(__file__).resolve().parent
BASE = RT / 'curated_repair_collector_recovery_v1.py'
QUEUE = RT / 'source_repair_gpu_v2/curated_repair_queue_v1.py'
BASE_SHA = 'f85efb1441930cfc31bfe4cafacf25e0fdb18aeb9da6ad80fb540b58519237cd'
QUEUE_SHA = '94b00ad0b503053f4b9bd7f7c08097bf9b6480f8a9aa248e971126a149c78b4c'
for _path, _sha in ((BASE, BASE_SHA), (QUEUE, QUEUE_SHA)):
    if hashlib.sha256(_path.read_bytes()).hexdigest() != _sha:
        raise RuntimeError('Frozen reconciliation dependency changed: ' + str(_path))
_spec = importlib.util.spec_from_file_location('terminal_reconciliation_frozen_evidence', BASE)
recovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recovery)
q = recovery.q
SCHEMA = 'terminal_reconciliation_v1'
SELF_REF = {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
EMPTY_OFFSET = 1192186
EMPTY_SHA = '01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b'
DEFAULT_BATCHES = tuple('train-b%06d' % i for i in range(4))


class ByteCache:
    def __init__(self):
        self.files = {}

    def sha(self, path):
        path = Path(path).absolute()
        before = recovery.snapshot(path.stat())
        key = str(path)
        if key in self.files:
            previous, digest = self.files[key]
            recovery.same(before, previous, 'Previously SHA-verified artifact changed: ' + key)
            return digest
        h = hashlib.sha256()
        with path.open('rb') as stream:
            recovery.same(recovery.snapshot(os.fstat(stream.fileno())), before, 'Hash opened artifact changed')
            for block in iter(lambda: stream.read(8388608), b''):
                h.update(block)
            recovery.same(recovery.snapshot(os.fstat(stream.fileno())), before, 'Hash artifact changed during read')
        recovery.same(recovery.snapshot(path.stat()), before, 'Hash artifact replaced after read')
        self.files[key] = (before, h.hexdigest())
        return h.hexdigest()

    def verify_all(self):
        for path in list(self.files):
            self.sha(path)


@contextmanager
def verified_hash_cache():
    cache, original, original_read = ByteCache(), q.sha, recovery.stable_read
    documents = {}
    def stable_cached_read(path):
        path = Path(path).absolute()
        key = str(path)
        if key in documents:
            stat, raw, ref = documents[key]
            recovery.same(recovery.snapshot(path.stat()), stat, 'Previously read immutable document changed: ' + key)
            return raw, ref
        stat = recovery.snapshot(path.stat())
        with path.open('rb') as stream:
            recovery.same(recovery.snapshot(os.fstat(stream.fileno())), stat, 'Immutable document replaced before read')
            raw = stream.read()
            recovery.same(recovery.snapshot(os.fstat(stream.fileno())), stat, 'Immutable document changed during read')
        recovery.same(recovery.snapshot(path.stat()), stat, 'Immutable document replaced after read')
        digest = hashlib.sha256(raw).hexdigest()
        if key in cache.files:
            recovery.same(cache.files[key], (stat, digest), 'Previously SHA-verified document differs')
        cache.files[key] = (stat, digest)
        ref = {'path': str(path), 'sha256': digest}
        documents[key] = (stat, raw, ref)
        return raw, ref
    q.sha, recovery.stable_read = cache.sha, stable_cached_read
    try:
        yield cache
    finally:
        q.sha, recovery.stable_read = original, original_read


def prefix_bytes(path, identity, length):
    path = Path(path)
    recovery.original_stream(path, identity, length)
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        recovery.same([opened.st_dev, opened.st_ino], identity, 'Opened prefix inode changed')
        raw = stream.read(length)
        recovery.same(len(raw), length, 'Prefix truncated during read')
        finished = os.fstat(stream.fileno())
        recovery.same([finished.st_dev, finished.st_ino], identity, 'Prefix inode changed during read')
    recovery.original_stream(path, identity, length)
    return raw


def capture_prefix(path):
    path = Path(path).resolve()
    stat = path.stat()
    identity, length = [stat.st_dev, stat.st_ino], stat.st_size
    raw = prefix_bytes(path, identity, length)
    recovery.same(prefix_bytes(path, identity, length), raw, 'Prefix changed between stable reads')
    if raw and not raw.endswith(bytes([10])):
        raise recovery.StreamBusy('Incomplete terminal/index tail; no partial row admitted')
    return raw, {'path': str(path), 'input_identity': identity, 'byte_length': length,
                 'prefix_sha256': hashlib.sha256(raw).hexdigest()}


def verify_prefix(ref):
    raw = prefix_bytes(ref['path'], ref['input_identity'], ref['byte_length'])
    recovery.same(hashlib.sha256(raw).hexdigest(), ref['prefix_sha256'], 'Original snapshotted prefix changed')
    return raw


def checked_terminal(production, hint, evidence):
    key = q.safe_key(hint)
    if hint['split'] != 'train':
        raise q.SourceChanged('Only original train curated sources admitted')
    evidence.member(hint['id'], hint['split'], hint['batch_id'])
    terminal, ref = recovery.document(production / 'sources' / key / 'consumer.terminal.json')
    recovery.same(terminal, hint, 'Journal/index hint differs from immutable consumer terminal: ' + key)
    q.verify_bytes(ref)
    return ref


def decode_journal(raw, production, evidence):
    rows, incidents, physical_rows = {}, [], 0
    stream = io.BytesIO(raw)
    while stream.tell() < len(raw):
        begin = stream.tell()
        line = stream.readline()
        physical_rows += 1
        try:
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError('Non-object terminal')
            values = [decoded]
        except (ValueError, UnicodeError):
            digest = hashlib.sha256(line).hexdigest()
            rule = recovery.KNOWN.get(begin)
            if rule and digest == rule['sha256'] and len(line) == rule['size']:
                pairs = [evidence.terminal(sid, rule['split'], rule['batch_id']) for sid in rule['ids']]
                boundary = len(line) - len(q.encoded(pairs[1][0]))
                if (boundary <= 0 or line[boundary:] != q.encoded(pairs[1][0])
                        or line[:boundary] != q.encoded(pairs[0][0])[:boundary]):
                    raise q.SourceChanged('Known badline does not equal original immutable two-source splice')
                values = [pair[0] for pair in pairs]
                incident = {'kind': 'exact_original_two_terminal_splice', 'source_keys': [q.safe_key(v) for v in values],
                            'terminal_evidence': [refs for _, refs in pairs], 'fragment_boundary': boundary}
            elif begin == EMPTY_OFFSET and line == bytes([10]) and digest == EMPTY_SHA:
                values = []
                incident = {'kind': 'verified_empty_physical_row', 'contained_source_records': 0,
                            'reconstructed_source_records': 0, 'population_completeness_inferred': False}
            else:
                raise recovery.UnrecoverableLine('Unreviewed malformed journal row at byte %d sha256=%s size=%d' % (begin, digest, len(line)))
            incidents.append({**incident, 'original_byte_begin': begin, 'original_byte_end': begin + len(line),
                              'raw_sha256': digest, 'raw_size': len(line), 'original_decode_error_preserved': True})
        for row in values:
            checked_terminal(production, row, evidence)
            key = q.safe_key(row)
            if key in rows:
                recovery.same(rows[key], row, 'Duplicate journal source revision differs')
            rows[key] = row
    return rows, incidents, physical_rows


def decode_index(raw, production, batch, evidence):
    descriptor = evidence.batches[batch]
    lines = raw.splitlines()
    if len(lines) > descriptor['count']:
        raise q.SourceChanged('Terminal index exceeds finite original batch member bound')
    rows, refs = {}, {}
    for number, line in enumerate(lines, 1):
        try:
            hint = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise recovery.UnrecoverableLine('Malformed terminal index %s line %d' % (batch, number)) from error
        if not isinstance(hint, dict) or set(hint) != {'id', 'split'} or hint['split'] != descriptor['split']:
            raise q.SourceChanged('Terminal index schema/split differs from original notification')
        key = q.safe_key(hint)
        if hint['split'] != 'train':
            raise q.SourceChanged('Index outside original train curated source scope')
        evidence.member(hint['id'], hint['split'], batch)
        terminal, ref = recovery.document(production / 'sources' / key / 'consumer.terminal.json')
        recovery.source_identity(terminal, hint['id'], hint['split'], batch)
        checked_terminal(production, terminal, evidence)
        if key in rows:
            recovery.same(rows[key], terminal, 'Repeated index source revision differs')
        rows[key], refs[key] = terminal, ref
    return rows, refs


def complete_evidence(evidence, row):
    terminal, refs = evidence.terminal(row['id'], row['split'], row['batch_id'])
    recovery.same(terminal, row, 'Immutable terminal changed during complete source verification')
    member, manifest_ref = evidence.member(row['id'], row['split'], row['batch_id'])
    joint, _ = recovery.document(refs['joint_result_ref']['path'], refs['joint_result_ref']['sha256'])
    source, _ = recovery.document(refs['source_identity_ref']['path'], refs['source_identity_ref']['sha256'])
    visual_root = Path(refs['visual_result_ref']['path']).parent.parent
    prepared, prepared_ref = recovery.document(visual_root / 'prepared.json')
    receipt, receipt_ref = recovery.document(visual_root / 'prepared.receipt.json')
    binding, _ = recovery.document(refs['source_binding_ref']['path'], refs['source_binding_ref']['sha256'])
    recovery.same(prepared['binding'], binding, 'Prepared visual source binding differs')
    recovery.same(receipt['binding'], binding, 'Visual receipt source binding differs')
    recovery.same(receipt['prepared_sha256'], prepared_ref['sha256'], 'Visual prepared receipt SHA differs')
    refs.update({'visual_prepared_ref': prepared_ref, 'visual_prepared_receipt_ref': receipt_ref})
    if joint.get('style_result_ref'):
        refs['style_result_ref'] = joint['style_result_ref']
        q.verify_bytes(refs['style_result_ref'])
    return refs, {'manifest_summary_ref': evidence.summary_ref, 'batch_id': row['batch_id'],
                  'batch_manifest_ref': manifest_ref, 'manifest_record': member,
                  'manifest_record_sha256': q.digest(member), 'original_csv_ref': member['source_refs']['csv'],
                  'original_source_refs': member['source_refs'],
                  'original_media_refs': {name: {k: source['files'][name][k] for k in ('path', 'sha256')}
                                          for name in ('video_path', 'audio_path')}}


def check_record(value, production, evidence):
    if value.get('schema') != SCHEMA or value.get('original_byte_offset', 'missing') is not None:
        raise q.SourceChanged('Reconciliation schema/virtual original byte offset rejected')
    if value.get('absence_from_that_exact_snapshot') is not True:
        raise q.SourceChanged('Reconciliation lacks exact-snapshot absence proof')
    if value.get('pending_independent_review') is not True or value.get('quality_approved') is not False:
        raise q.SourceChanged('Reconciliation is not an unapproved candidate')
    for flag in q.FLAGS:
        recovery.same(value.get(flag), False, 'Reconciliation training/approval flag changed')
    row = value['hint']
    recovery.source_identity(value, row['id'], row['split'], row['batch_id'])
    key = q.safe_key(row)
    recovery.same(value['source_key'], key, 'Derived source key changed')
    recovery.same(value['source_dedup_key'], key, 'Derived source dedup key changed')
    recovery.same(value['completed_row_sha256'], q.digest(row), 'Derived terminal digest changed')
    if row['status'] != 'joint_candidate' or row.get('joint_complete') is not True:
        raise q.SourceChanged('Only complete data candidates can be reconciled')
    refs, original = complete_evidence(evidence, row)
    recovery.same(value['evidence'], refs, 'Previously reconciled source evidence changed')
    recovery.same(value['current_original_manifest'], original, 'Previously reconciled original membership changed')
    recovery.same(value['original_consumer_terminal_ref'], refs['consumer_terminal_ref'], 'Derived original terminal reference differs')
    snapshot, _ = recovery.document(value['original_journal_snapshot_ref']['path'], value['original_journal_snapshot_ref']['sha256'])
    raw = verify_prefix(snapshot['original_journal_prefix'])
    cache = getattr(evidence, '_absence_snapshots', None)
    if cache is None:
        cache = evidence._absence_snapshots = {}
    cache_key = q.digest(snapshot['original_journal_prefix'])
    if cache_key not in cache:
        cache[cache_key] = decode_journal(raw, production, evidence)[0]
    if key in cache[cache_key]:
        raise q.SourceChanged('Reconciled source was present in its claimed absence snapshot')
    recovery.same(value['original_journal_snapshot'], snapshot['original_journal_prefix'], 'Derived original journal snapshot differs')


def save_projection(path, raw):
    if path.exists() and path.read_bytes() == raw:
        return
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reconcile(production, manifest, out, batches=DEFAULT_BATCHES):
    production, manifest, out = Path(production).resolve(), Path(manifest).resolve(), Path(out).resolve()
    if production.is_relative_to(out) or out.is_relative_to(production) or manifest.is_relative_to(out) or out.is_relative_to(manifest.parent):
        raise ValueError('Derived output must not overlap original production/manifest')
    batches = tuple(batches)
    if not batches or len(set(batches)) != len(batches):
        raise ValueError('Explicit unique finite original batches required')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.reconciliation.lock').open('a+b') as lock, verified_hash_cache() as cache:
        fcntl.flock(lock, fcntl.LOCK_EX)
        q.verify_bytes(SELF_REF)
        evidence = recovery.Evidence(production, manifest)
        recovery.same((evidence.summary['total'], evidence.summary['batch_size'], len(evidence.batches)),
                      (222939, 500, 448), 'Full original 222939/448 authoritative manifest required')
        for batch in batches:
            info = evidence.batches.get(batch)
            if not info or info['split'] != 'train' or info['count'] != 500:
                raise ValueError('Only explicit original full 500-member train batches admitted')
            raw_members, member_ref = recovery.stable_read(info['path'])
            recovery.same(member_ref['sha256'], info['sha256'], 'Full batch manifest SHA changed')
            members = [json.loads(line) for line in raw_members.splitlines()]
            recovery.same(len(members), 500, 'Full original batch member count differs')
            for member in members:
                evidence.member(member['id'], 'train', batch)
            q.verify_bytes(member_ref)
        # Every previous accepted snapshot remains bound to its exact original prefix.
        for path in sorted((out / 'snapshots').glob('*/snapshot.json')):
            old, _ = recovery.document(path)
            recovery.same(old['production'], str(production), 'Reconciliation production namespace changed')
            recovery.same(old['manifest_summary_ref'], evidence.summary_ref, 'Reconciliation original manifest changed')
            verify_prefix(old['original_journal_prefix'])
            for ref in old['terminal_index_prefixes'].values():
                verify_prefix(ref)
        capture_started_utc = datetime.now(timezone.utc).isoformat()
        journal_raw, journal_ref = capture_prefix(production / 'completed.jsonl')
        index_raw, index_refs = {}, {}
        for batch in batches:
            index_raw[batch], index_refs[batch] = capture_prefix(production / 'batches' / batch / 'terminal_index.jsonl')
        captured_utc = datetime.now(timezone.utc).isoformat()
        binding = {'production': str(production), 'manifest_summary_ref': evidence.summary_ref,
                   'original_journal_prefix': journal_ref, 'terminal_index_prefixes': index_refs,
                   'selected_original_batches': list(batches)}
        fingerprint = q.digest(binding)
        directory = out / 'snapshots' / fingerprint
        recovery.save_raw_once(directory / 'completed.prefix.raw', journal_raw)
        for batch in batches:
            recovery.save_raw_once(directory / (batch + '.terminal_index.prefix.raw'), index_raw[batch])
        try:
            journal, incidents, physical_rows = decode_journal(journal_raw, production, evidence)
            indexed, index_terminal_refs = {}, {}
            for batch in batches:
                values, refs = decode_index(index_raw[batch], production, batch, evidence)
                for key, row in values.items():
                    if key in indexed:
                        raise q.SourceChanged('Source indexed across multiple batches')
                    indexed[key], index_terminal_refs[key] = row, refs[key]
            previous = {}
            for path in sorted((out / 'records' / 'train').glob('*.json')):
                value, _ = recovery.document(path)
                check_record(value, production, evidence)
                recovery.same(path, out / 'records' / (value['source_key'] + '.json'), 'Wrong derived record filename')
                key = value['source_key']
                receipt = {'schema': SCHEMA, 'source_dedup_key': key, 'record_ref': q.identity(path),
                           'original_consumer_terminal_ref': value['original_consumer_terminal_ref'],
                           'original_journal_snapshot_ref': value['original_journal_snapshot_ref'],
                           'original_byte_offset': None, **q.FLAGS}
                q.save_once(out / 'receipts' / (key + '.json'), receipt)
                previous[key] = value
            projection = out / 'records.jsonl'
            if projection.exists():
                projected, _ = capture_prefix(projection)
                projected_keys = set()
                for line in projected.splitlines():
                    value = json.loads(line)
                    key = value['source_key']
                    if key in projected_keys or key not in previous or value != previous[key]:
                        raise q.SourceChanged('Derived projection contains duplicate/unknown/changed source')
                    projected_keys.add(key)
            missing = sorted(set(indexed) - set(journal))
            verified, unresolved = {}, []
            for key in missing:
                row = indexed[key]
                if row['status'] != 'joint_candidate' or row.get('joint_complete') is not True:
                    unresolved.append({'source_key': key, 'reason': 'immutable_terminal_not_complete_joint_candidate',
                                       'original_consumer_terminal_ref': index_terminal_refs[key]})
                    continue
                refs, original = complete_evidence(evidence, row)
                verified[key] = (row, refs, original)
            for ref in [journal_ref, *index_refs.values()]:
                verify_prefix(ref)
            cache.verify_all()
        except (ValueError, KeyError, TypeError, OSError) as error:
            q.save_once(directory / 'rejection.json', {'schema': SCHEMA, **binding, 'accepted': False,
                'reason': type(error).__name__ + ': ' + str(error), 'raw_journal_ref': q.identity(directory / 'completed.prefix.raw'),
                'original_sources_modified': False, **q.FLAGS})
            raise
        badlines = []
        for incident in incidents:
            begin, end = incident['original_byte_begin'], incident['original_byte_end']
            path = directory / ('badline-%d.raw' % begin)
            recovery.save_raw_once(path, journal_raw[begin:end])
            badlines.append({**incident, 'raw_ref': q.identity(path)})
        snapshot_path = directory / 'snapshot.json'
        if snapshot_path.exists():
            snapshot, _ = recovery.document(snapshot_path)
            recovery.same({k: snapshot[k] for k in binding}, binding, 'Immutable reconciliation snapshot binding changed')
        else:
            snapshot = {'schema': SCHEMA, **binding, 'snapshot_time_utc': captured_utc, 'snapshot_capture_started_utc': capture_started_utc,
                        'raw_journal_ref': q.identity(directory / 'completed.prefix.raw'),
                        'raw_terminal_index_refs': {batch: q.identity(directory / (batch + '.terminal_index.prefix.raw')) for batch in batches},
                        'badlines': badlines, 'reconciliation_code_ref': SELF_REF,
                        'frozen_evidence_code_ref': {'path': str(BASE), 'sha256': BASE_SHA},
                        'frozen_queue_code_ref': {'path': str(QUEUE), 'sha256': QUEUE_SHA}}
            q.save_once(snapshot_path, snapshot)
        snapshot_ref = q.identity(snapshot_path)
        created = []
        for key, (row, refs, original) in verified.items():
            if key in previous:
                recovery.same(previous[key]['hint'], row, 'Reconciled source terminal revision differs')
                continue
            value = {'schema': SCHEMA, 'id': row['id'], 'split': row['split'], 'batch_id': row['batch_id'],
                     'source_key': key, 'source_dedup_key': key, 'hint': row, 'completed_row_sha256': q.digest(row),
                     'original_consumer_terminal_ref': refs['consumer_terminal_ref'], 'evidence': refs,
                     'current_original_manifest': original, 'original_journal_snapshot': journal_ref,
                     'original_journal_snapshot_ref': snapshot_ref, 'original_terminal_index_prefix': index_refs[row['batch_id']],
                     'absence_from_that_exact_snapshot': True, 'original_byte_offset': None,
                     'data_complete_candidate': True, 'quality_approved': False, 'pending_independent_review': True,
                     'source_reinference_performed': False, 'original_journal_modified': False,
                     'reconciliation_policy': 'immutable_original_terminal_full_chain_and_exact_original_prefix_absence', **q.FLAGS}
            path = out / 'records' / (key + '.json')
            q.save_once(path, value)
            q.save_once(out / 'receipts' / (key + '.json'), {'schema': SCHEMA, 'source_dedup_key': key,
                       'record_ref': q.identity(path), 'original_consumer_terminal_ref': refs['consumer_terminal_ref'],
                       'original_journal_snapshot_ref': snapshot_ref, 'original_byte_offset': None, **q.FLAGS})
            previous[key] = value
            created.append(key)
        # This is a rebuildable projection of immutable records, not a substituted original journal.
        save_projection(out / 'records.jsonl', b''.join(q.encoded(previous[key]) for key in sorted(previous)))
        denominator = []
        denominator_started_utc = datetime.now(timezone.utc).isoformat()
        source_sha_checked = set(previous) | set(verified) | {key for incident in incidents for key in incident.get('source_keys', [])}
        for batch in batches:
            members = evidence.members[batch]
            keys = set(members)
            terminal_present = {key for key in keys if (production / 'sources' / key / 'consumer.terminal.json').is_file()}
            known = keys & set(indexed)
            full_checked = keys & source_sha_checked
            denominator.append({'batch_id': batch, 'original_manifest_member_count': len(keys),
                'batch_manifest_ref': {'path': evidence.batches[batch]['path'], 'sha256': evidence.batches[batch]['sha256']},
                'terminal_index_unique_count': len(known), 'journal_unique_count': len(keys & set(journal)),
                'consumer_terminal_path_present_count': len(terminal_present),
                'consumer_terminal_path_missing_count': len(keys - terminal_present),
                'members_not_in_terminal_index': sorted(keys - known),
                'terminal_present_not_indexed': sorted(terminal_present - known),
                'fully_source_SHA_checked_this_run_count': len(full_checked),
                'members_not_fully_source_SHA_checked_this_run_count': len(keys - full_checked),
                'complete_population_or_quality_approval_inferred': False})
        summary = {'schema': SCHEMA, 'snapshot_ref': snapshot_ref, 'snapshot_time_utc': snapshot['snapshot_time_utc'],
            'source_dedup_key': 'original_split/source_id', 'original_journal_physical_rows': physical_rows,
            'original_journal_unique_sources': len(journal), 'terminal_index_unique_sources': len(indexed),
            'missing_from_exact_journal_snapshot_count': len(missing), 'missing_source_keys': missing,
            'data_complete_candidate_verified_count': len(verified), 'new_reconciled_source_count': len(created),
            'new_reconciled_source_keys': created, 'cumulative_reconciled_unique_sources': len(previous),
            'unresolved_missing_sources': unresolved, 'still_missing_without_verified_derived_record_count': len(unresolved),
            'badlines': badlines, 'original_batch_membership_denominator': denominator,
            'terminal_path_existence_observation_started_utc': denominator_started_utc,
            'terminal_path_existence_observation_finished_utc': datetime.now(timezone.utc).isoformat(),
            'terminal_path_existence_is_not_immutable_terminal_or_label_verification': True,
            'selected_original_member_count': sum(d['original_manifest_member_count'] for d in denominator),
            'full_original_manifest_total': evidence.summary['total'], 'full_original_manifest_batch_count': len(evidence.batches),
            'nonselected_original_members_not_reconciled_count': evidence.summary['total'] - sum(d['original_manifest_member_count'] for d in denominator),
            'entire_dataset_success_claimed': False, 'quality_approved_source_count': 0,
            'GPU_dispatch_performed': False, 'new_source_or_label_generated': False, **q.FLAGS}
        for ref in [journal_ref, *index_refs.values()]:
            verify_prefix(ref)
        cache.verify_all()
        # Each invocation has an immutable receipt; the top summary is only a latest projection.
        q.save_once(directory / ('summary-' + q.digest(summary) + '.json'), summary)
        q.replace_state(out / 'summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production', default=str(RT.parent / 'curated222939_production_v1'))
    parser.add_argument('--manifest', default=str(RT.parent / 'curated222939_inputs_v1/manifest.summary.json'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch', action='append', help='Explicit original full train batch; default batch0..3')
    args = parser.parse_args()
    print(json.dumps(reconcile(args.production, args.manifest, args.out, args.batch or DEFAULT_BATCHES),
                     ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
