#!/usr/bin/env python3
"""CPU-only exact, evidence-checked recovery over the original byte cursor."""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import uuid

QUEUE_MODULE = Path(__file__).parent / 'source_repair_gpu_v2/curated_repair_queue_v1.py'
_spec = importlib.util.spec_from_file_location('immutable_repair_queue', QUEUE_MODULE)
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)
SCHEMA = 'curated_repair_collector_recovery_v1'
BAD_OFFSET = 532233
BAD_SHA = 'd74abe92289ada7d68da9f97c5444e177ae91b94ee76dd8be955a6556137e280'
KNOWN = {BAD_OFFSET: {'sha256': BAD_SHA, 'size': 940, 'split': 'train',
    'batch_id': 'train-b000001', 'ids': ['MVuipekTKQDQprcC', 'ytykxzHnSDAecSEP']}}


class StreamBusy(RuntimeError):
    pass


class UnrecoverableLine(ValueError):
    pass


def snapshot(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def stable_read(path):
    path = Path(path)
    before = path.stat()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read()
        finished = os.fstat(stream.fileno())
    after = path.stat()
    if any(snapshot(s) != snapshot(before) for s in (opened, finished, after)):
        raise q.SourceChanged('Immutable artifact changed during read: ' + str(path))
    return raw, {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}


def document(path, expected=None):
    raw, ref = stable_read(path)
    if expected is not None and ref['sha256'] != expected:
        raise q.SourceChanged('Evidence SHA mismatch: ' + str(path))
    return json.loads(raw), ref


def visual_digest(value):
    return hashlib.sha256((json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n').encode()).hexdigest()


def same(value, expected, reason):
    if value != expected:
        raise q.SourceChanged(reason)


def source_identity(value, sid, split, batch):
    same((value.get('id'), value.get('split'), value.get('batch_id')),
         (sid, split, batch), 'Source ID/split/batch mismatch')
    if 'source_id' in value:
        same(value['source_id'], sid, 'source_id mismatch')


def original_stream(path, identity, offset):
    stat = path.stat()
    if [stat.st_dev, stat.st_ino] != identity or stat.st_size < offset:
        raise q.SourceChanged('Original completed stream replaced/truncated')
    return stat


def prefix_hash(path, end, identity):
    original_stream(path, identity, end)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        if [os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino] != identity:
            raise q.SourceChanged('Opened completed stream identity differs')
        left = end
        while left:
            block = stream.read(min(left, 1024 * 1024))
            if not block:
                raise q.SourceChanged('Original prefix truncated')
            h.update(block)
            left -= len(block)
    original_stream(path, identity, end)
    return h


def prefix_sha(path, end, identity):
    return prefix_hash(path, end, identity).hexdigest()


def stable_badline(path, begin, line, identity, delay):
    """A newline alone does not prove that a cross-mount concurrent write finished."""
    samples = []
    for i in range(3):
        if i:
            time.sleep(delay)
        before = original_stream(path, identity, begin + len(line))
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            stream.seek(begin)
            actual = stream.readline()
            finished = os.fstat(stream.fileno())
        after = original_stream(path, identity, begin + len(line))
        if any(snapshot(s) != snapshot(before) for s in (opened, finished, after)) or actual != line:
            raise StreamBusy('Bad line not stable; retain original cursor and retry')
        samples.append({'identity': list(snapshot(after)), 'line_sha256': hashlib.sha256(actual).hexdigest()})
    if any(s['identity'] != samples[0]['identity'] for s in samples[1:]):
        raise StreamBusy('Concurrent stream write during bad-line snapshots; retry')
    return samples


class Evidence:
    def __init__(self, production, manifest):
        self.production = Path(production).resolve()
        self.summary, self.summary_ref = document(manifest)
        self.batches = {b['batch_id']: b for b in self.summary['batches']}
        if len(self.batches) != len(self.summary['batches']) or sum(b['count'] for b in self.batches.values()) != self.summary['total']:
            raise ValueError('Authoritative batch summary count mismatch')
        self.members = {}
        self.protocols = None

    def member(self, sid, split, batch):
        info = self.batches.get(batch)
        if not info or info['split'] != split:
            raise UnrecoverableLine('Unknown authoritative batch/split')
        if batch not in self.members:
            raw, ref = stable_read(info['path'])
            same(ref['sha256'], info['sha256'], 'Authoritative batch SHA changed')
            rows = [json.loads(line) for line in raw.splitlines()]
            members = {}
            for row in rows:
                source_identity(row, row['id'], split, batch)
                key = q.safe_key(row)
                if key in members:
                    raise ValueError('Duplicate authoritative batch member')
                members[key] = row
            same(len(members), info['count'], 'Authoritative batch count mismatch')
            self.members[batch] = members
        key = q.safe_key({'id': sid, 'split': split})
        if key not in self.members[batch]:
            raise UnrecoverableLine('No authoritative source membership: ' + key)
        return self.members[batch][key], {'path': info['path'], 'sha256': info['sha256']}

    def artifact(self, ref, within=None):
        root = self.production if within is None else within
        if not Path(ref['path']).resolve().is_relative_to(root):
            raise q.SourceChanged('Evidence path escapes expected source directory')
        return document(ref['path'], ref['sha256'])

    def params(self, expected, visual_root):
        # Production protocol saves sorted JSON; reconstruct only the insertion order
        # specified by its SHA-verified params() AST, never invent parameter values.
        receipt, receipt_ref = document(visual_root.parent.parent / 'load.receipt.json')
        code = receipt['code_sha256']
        for path, sha in code.items():
            q.verify_bytes({'path': path, 'sha256': sha})
        bar_path = next(Path(path) for path in code if Path(path).name == 'ms_bar_annotation.py')
        tree = ast.parse(bar_path.read_bytes())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'params')
        ret = next(n for n in fn.body if isinstance(n, ast.Return))
        if not isinstance(ret.value, ast.Dict) or not all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in ret.value.keys):
            raise q.SourceChanged('Unknown production params AST')
        keys = [k.value for k in ret.value.keys] + ['style_module_sha256']
        if self.protocols is None:
            self.protocols = []
            for path in sorted((self.production / 'runs').glob('v*/protocol.json')):
                protocol, ref = document(path)
                if protocol.get('mode') == 'consumer':
                    self.protocols.append((protocol, ref))
        for protocol, ref in self.protocols:
            configuration = protocol['configuration']
            if set(configuration) != set(keys) or configuration['code_sha256'] != code:
                continue
            ordered = {k: configuration[k] for k in keys}
            ordered['code_sha256'] = code
            if visual_digest(ordered) == expected:
                same(protocol['manifest_summary'], self.summary_ref, 'Consumer protocol manifest changed')
                return {'consumer_protocol_ref': ref, 'load_receipt_ref': receipt_ref,
                        'params_sha256': expected, 'configuration_canonical_sha256': q.digest(configuration),
                        'params_AST_ref': {'path': str(bar_path), 'sha256': code[str(bar_path)]}}
        raise q.SourceChanged('No authentic consumer configuration matches terminal params SHA')

    def terminal(self, sid, split, batch):
        key = q.safe_key({'id': sid, 'split': split})
        directory = self.production / 'sources' / key
        row, terminal_ref = document(directory / 'consumer.terminal.json')
        source_identity(row, sid, split, batch)
        member, manifest_ref = self.member(sid, split, batch)
        producer, producer_ref = document(directory / 'producer.terminal.json')
        source_identity(producer, sid, split, batch)
        refs = {'consumer_terminal_ref': terminal_ref, 'producer_terminal_ref': producer_ref,
                'manifest_summary_ref': self.summary_ref, 'manifest_ref': manifest_ref,
                'manifest_row_sha256': q.digest(member)}
        # Only evidence-rich joint terminals are admitted by this exact incident.
        # A new/unknown terminal shape requires another reviewed recovery rule.
        if not row.get('joint_candidate_ref') or not producer.get('plan_ref'):
            raise UnrecoverableLine('Exact incident terminal lacks verifiable JOINT/plan')
        plan, plan_ref = self.artifact(producer['plan_ref'], directory)
        same(plan['id'], sid, 'Plan ID changed')
        same(plan['manifest_record'], member, 'Plan differs from authoritative manifest member')
        same(plan['source_refs']['manifest'], manifest_ref, 'Plan original manifest reference changed')
        same(plan['source_refs']['original_csv'], member['source_refs']['csv'], 'Plan CSV sourceRefs changed')
        same(producer['source_binding'], plan['binding'], 'Producer source.binding mismatch')
        joint, joint_ref = self.artifact(row['joint_candidate_ref'], directory / 'joint-results')
        source_identity(joint, sid, split, batch)
        for field in ('joint_complete', 'style_status', 'visual_status'):
            same(row.get(field), joint.get(field), 'Terminal/JOINT status mismatch: ' + field)
        same(joint['source_refs'], member['source_refs'], 'JOINT sourceRefs mismatch')
        visual, visual_ref = self.artifact(joint['visual_result_ref'])
        same(visual['id'], sid, 'VISUAL ID mismatch')
        same(visual['manifest_record'], member, 'VISUAL manifest member mismatch')
        binding = joint['source_binding']
        same(binding, visual['binding'], 'JOINT/VISUAL source.binding mismatch')
        same(binding, {'source_binding': plan['binding'], 'plan_sha256': plan_ref['sha256'],
            'full_source_row_sha256': visual_digest(plan), 'raw_line_sha256': plan_ref['sha256'],
            'manifest_record_sha256': visual_digest(member), 'params_sha256': row['params_sha256']},
            'JOINT source.binding/params/full-row SHA mismatch')
        visual_root = Path(visual_ref['path']).parent.parent
        stored_binding, binding_ref = document(visual_root / 'binding.json')
        same(stored_binding, binding, 'Immutable binding.json mismatch')
        stored_refs, stored_refs_ref = document(visual_root / 'source.refs.json')
        same([{k: ref[k] for k in ('path', 'sha256')} for ref in stored_refs],
             [{k: ref[k] for k in ('path', 'sha256')} for ref in plan['source_refs'].values()],
             'Immutable source.refs.json mismatch')
        q.verify_references_by_content(stored_refs)
        source, source_ref = document(visual_root / 'source.identity.json')
        same(source['sha256'], joint['source_identity']['sha256'], 'JOINT source identity SHA mismatch')
        same(source['files'].get('nlf_path'), joint['source_identity']['files'].get('nlf_path'),
             'JOINT NLF identity mismatch')
        for name in ('video_path', 'audio_path'):
            same({k: source['files'][name][k] for k in ('path', 'sha256')},
                 {k: joint['source_identity']['files'][name][k] for k in ('path', 'sha256')},
                 'JOINT source media content identity mismatch')
            ref = source['files'][name]
            same(ref['path'], member[name], 'Source media manifest path mismatch')
            same(ref['sha256'], plan['binding'][name.replace('_path', '_sha256')], 'Source media binding SHA mismatch')
            q.verify_bytes(ref)
        q.verify_references_by_content(member['source_refs'])
        for name in ('roi_ref',):
            if producer.get(name):
                self.artifact(producer[name])
        if joint.get('style_result_ref'):
            style, _ = self.artifact(joint['style_result_ref'], directory / 'style-results')
            same(style['id'], sid, 'STYLE ID mismatch')
        refs.update({'plan_ref': plan_ref, 'joint_result_ref': joint_ref, 'visual_result_ref': visual_ref,
                     'source_binding_ref': binding_ref, 'source_refs_ref': stored_refs_ref,
                     'source_identity_ref': source_ref, **self.params(row['params_sha256'], visual_root)})
        # Detect a last-moment immutable replacement after completing the chain.
        for name, ref in refs.items():
            if name.endswith('_ref'):
                q.verify_bytes(ref)
        return row, refs


def save_raw_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        same(path.read_bytes(), data, 'Original bad-line evidence changed')
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


def recover_line(production, out, begin, line, evidence, snapshots, rules=KNOWN):
    rule = rules.get(begin)
    line_sha = hashlib.sha256(line).hexdigest()
    raw_path = out / 'collector-recovery-v1' / 'badlines' / (str(begin) + '-' + line_sha + '.raw')
    save_raw_once(raw_path, line)
    if not rule or line_sha != rule['sha256'] or len(line) != rule['size']:
        raise UnrecoverableLine('Unreviewed malformed source line at original byte offset ' + str(begin))
    # Exact interleaving: prefix of one serialized terminal + one entire terminal.
    # No regex ID search, no JSON repair, and no guessed missing field.
    rows = [evidence.terminal(sid, rule['split'], rule['batch_id']) for sid in rule['ids']]
    boundary = len(line) - len(q.encoded(rows[1][0]))
    if boundary <= 0 or line[boundary:] != q.encoded(rows[1][0]) or line[:boundary] != q.encoded(rows[0][0])[:boundary]:
        raise UnrecoverableLine('Known SHA does not match the two immutable serialized terminal fragments')
    raw_ref = q.identity(raw_path)
    hints = []
    for index, (row, refs) in enumerate(rows):
        key = q.safe_key(row)
        value = {'schema': SCHEMA, 'source_key': key, 'hint': row,
                 'completed_row_sha256': q.digest(row), 'evidence': refs,
                 'badline_ref': raw_ref, 'source_input_path': str(production / 'completed.jsonl'),
                 'source_input_byte_begin': begin, 'source_input_byte_end': begin + len(line),
                 'recovered_terminal_index': index, 'recovered_terminal_count': len(rows),
                 'stable_badline_snapshots': snapshots,
                 'recovery_policy': 'exact_offset_SHA_then_authoritative_manifest_and_immutable_terminal_chain',
                 **q.FLAGS}
        hint_path = out / 'collector-recovery-v1' / 'hints' / (key + '.json')
        # Stable snapshots are evidence of the first recovery, not overwritten on replay.
        if hint_path.exists():
            old = q.load(hint_path)
            old_without = {k: v for k, v in old.items() if k != 'stable_badline_snapshots'}
            same(old_without, {k: v for k, v in value.items() if k != 'stable_badline_snapshots'}, 'Derived terminal evidence changed')
            value = old
        q.save_once(hint_path, value)
        notice = out / 'collector-recovery-v1' / 'announced' / (key + '.json')
        if not notice.exists():
            q.append(out / 'collector-recovery-v1' / 'derived-hints.jsonl', value)
            q.save_once(notice, {'hint_ref': q.identity(hint_path)})
        hints.append(row)
    return hints


def validate_prefix(queue, origin):
    seen, counts = {}, {}
    with queue.open('rb') as stream:
        while stream.tell() < origin['offset']:
            line = stream.readline()
            if not line.endswith(b'\n') or stream.tell() > origin['offset']:
                raise q.SourceChanged('Inherited cursor is not a complete source-line boundary')
            row = json.loads(line)
            key, digest = q.safe_key(row), q.digest(row)
            if key in seen:
                same(seen[key], digest, 'Inherited prefix source revision changed')
                continue
            seen[key] = digest
            name = 'joint_complete' if row.get('joint_complete') is True and row['status'] == 'joint_candidate' else 'incomplete_or_failed'
            counts[name] = counts.get(name, 0) + 1
    same(seen, origin['seen'], 'Inherited state seen does not match original source prefix')
    same(counts, origin['counts'], 'Inherited state counts do not match original source prefix')


def admit(row, production, out, revision, state):
    key, row_sha = q.safe_key(row), q.digest(row)
    terminal, _ = document(production / 'sources' / key / 'consumer.terminal.json')
    same(terminal, row, 'Normal/derived hint differs from immutable consumer terminal: ' + key)
    previous = state['seen'].get(key)
    if previous:
        same(previous, row_sha, 'Completed source revision changed: ' + key)
        return 0
    complete = row.get('joint_complete') is True and row['status'] == 'joint_candidate'
    added = 0
    if not complete:
        job = q.classify(row, production)
        job.update(global_revision=revision, queue_root=str(out))
        job_path = out / 'jobs' / (key + '.json')
        if job_path.exists():
            existing = q.load(job_path)
            same(existing['completed_row_sha256'], row_sha, 'Queued original row SHA changed')
            same(existing['global_revision'], revision, 'Existing job global revision changed')
            same(Path(existing['queue_root']).resolve(), out, 'Existing queue root changed')
            job = existing
        else:
            q.save_once(job_path, job)
            added = 1
        notice = out / 'announced' / (key + '.json')
        if not notice.exists():
            q.append(out / 'ready.jsonl', {'source_key': key, 'job_ref': q.identity(job_path),
                'stage': job['stage'], 'stages': job['stages'], 'global_revision': revision})
            q.save_once(notice, {'job_ref': q.identity(job_path)})
        state['stage_counts'][job['stage']] = state['stage_counts'].get(job['stage'], 0) + 1
    name = 'joint_complete' if complete else 'incomplete_or_failed'
    state['counts'][name] = state['counts'].get(name, 0) + 1
    state['seen'][key] = row_sha
    return added


def collect(production, out, revision, manifest, stable_seconds=0.25, rules=KNOWN):
    production, out = Path(production).resolve(), Path(out).resolve()
    if production.is_relative_to(out) or out.is_relative_to(production):
        raise ValueError('Repair output must remain separate from production')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', revision):
        raise ValueError('Unsafe revision')
    if stable_seconds < 0:
        raise ValueError('Negative stability delay')
    out.mkdir(parents=True, exist_ok=True)
    queue, original_path = production / 'completed.jsonl', out / 'collector.state.json'
    state_path = out / 'collector.recovery.state.json'
    with (out / 'collector.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        origin, origin_ref = document(original_path)
        input_identity = origin.get('input_identity')
        if input_identity is None or origin['input_path'] != str(queue):
            raise q.SourceChanged('Inherited state lacks original stream identity')
        original_stream(queue, input_identity, origin['offset'])
        evidence = Evidence(production, manifest)
        if state_path.exists():
            state = q.load(state_path)
            same(state['origin_state_ref'], origin_ref, 'Original collector state drifted after recovery binding')
            same(state['input_identity'], input_identity, 'Recovered stream identity drifted')
            same(state['input_path'], str(queue), 'Recovered input_path must remain original stream')
            same(state['global_revision'], revision, 'Recovery global revision changed')
            same(state['manifest_summary_ref'], evidence.summary_ref, 'Recovery manifest changed')
            same(prefix_sha(queue, state['offset'], input_identity), state['input_prefix_sha256'], 'Completed source prefix changed')
        else:
            validate_prefix(queue, origin)
            state = copy.deepcopy(origin)
            state.update(schema=SCHEMA, global_revision=revision, origin_state_ref=origin_ref,
                manifest_summary_ref=evidence.summary_ref, recovered_badline_offsets=[],
                input_prefix_sha256=prefix_sha(queue, origin['offset'], input_identity))
            q.save_once(out / 'collector-recovery-v1/original-state.json', origin)
            q.replace_state(state_path, state)
        checkpoint_offset, checkpoint_sha = state['offset'], state['input_prefix_sha256']
        input_hasher = prefix_hash(queue, state['offset'], input_identity)
        same(input_hasher.hexdigest(), checkpoint_sha, 'Source prefix changed before opening scan')
        added = processed = checkpoint_sources = 0
        waiting = None
        def checkpoint():
            nonlocal checkpoint_offset, checkpoint_sha
            # Never bind the original state or source cursor to a derived stream.
            original_stream(queue, input_identity, state['offset'])
            same(q.identity(original_path), origin_ref, 'Original collector state changed concurrently')
            same(prefix_sha(queue, checkpoint_offset, input_identity), checkpoint_sha,
                 'Previously committed source prefix changed during scan')
            actual_prefix_sha = prefix_sha(queue, state['offset'], input_identity)
            same(actual_prefix_sha, input_hasher.hexdigest(), 'Read source bytes changed before checkpoint')
            state['input_prefix_sha256'] = actual_prefix_sha
            q.replace_state(state_path, state)
            checkpoint_offset, checkpoint_sha = state['offset'], state['input_prefix_sha256']
        with queue.open('rb') as stream:
            original_stream(queue, input_identity, state['offset'])
            same([os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino], input_identity, 'Opened source identity differs')
            stream.seek(state['offset'])
            while True:
                begin = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    waiting = 'partial_line' if line else 'tail'
                    break
                try:
                    row = json.loads(line)
                    rows = [row]
                except (ValueError, UnicodeError):
                    try:
                        snapshots = stable_badline(queue, begin, line, input_identity, stable_seconds)
                    except StreamBusy:
                        waiting = 'badline_snapshot_busy'
                        break
                    try:
                        rows = recover_line(production, out, begin, line, evidence, snapshots, rules)
                    except UnrecoverableLine as error:
                        checkpoint()
                        raise UnrecoverableLine(str(error) + '; cursor retained at ' + str(begin)) from error
                    if begin not in state['recovered_badline_offsets']:
                        state['recovered_badline_offsets'].append(begin)
                for row in rows:
                    added += admit(row, production, out, revision, state)
                    processed += 1
                    checkpoint_sources += 1
                input_hasher.update(line)
                state['offset'] = stream.tell()
                if checkpoint_sources >= 16:
                    checkpoint()
                    checkpoint_sources = 0
        checkpoint()
        return {'schema': SCHEMA, 'sources_seen': len(state['seen']), 'new_queued_sources': added,
                'processed_source_hints': processed, 'counts': state['counts'],
                'primary_stage_counts': state['stage_counts'], 'input_byte_offset': state['offset'],
                'input_path': str(queue), 'state_path': str(state_path), 'waiting': waiting,
                'recovered_badline_offsets': state['recovered_badline_offsets'],
                'queue_path': str(out / 'ready.jsonl'), 'GPU_dispatch_performed': False, **q.FLAGS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--global-revision', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--stable-seconds', type=float, default=0.25)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args()
    while True:
        result = collect(args.production, args.out, args.global_revision, args.manifest, args.stable_seconds)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
