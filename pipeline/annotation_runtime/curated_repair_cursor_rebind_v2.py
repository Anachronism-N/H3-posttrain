"""Explicit CPU-only cross-mount cursor migration; normal worker checks remain strict."""
import argparse
import copy
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cursor_rebind_q', RT / 'source_repair_gpu_v3/curated_repair_queue_v1.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def rebind_state(out, revision, expected_state_ref, prefix_sha256, node):
    out = Path(out).resolve()
    state_path = out / 'worker.state.json'
    ready = out / 'ready.jsonl'
    if Path(expected_state_ref['path']).resolve() != state_path:
        raise q.SourceChanged('Expected cursor is not the queue worker state')
    state = q.checked(expected_state_ref)
    if state['input_path'] != str(ready) or state['schema'] != 'curated_repair_watch_v2':
        raise q.SourceChanged('Queue cursor schema/path differs')
    before = ready.stat()
    previous_identity = state['input_identity']
    current_identity = [before.st_dev, before.st_ino]
    if (len(previous_identity) != 2 or previous_identity[1] != before.st_ino or
            previous_identity[0] == before.st_dev):
        raise q.SourceChanged('Explicit migration requires same inode and changed mount device')
    offset = state['offset']
    if type(offset) is not int or offset <= 0 or offset > before.st_size:
        raise q.SourceChanged('Invalid consumed cursor offset')
    with ready.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if [opened.st_dev, opened.st_ino] != current_identity:
            raise q.SourceChanged('Ready stream replaced during cursor migration')
        prefix = stream.read(offset)
    if len(prefix) != offset or not prefix.endswith(b'\n') or hashlib.sha256(prefix).hexdigest() != prefix_sha256:
        raise q.SourceChanged('Consumed ready prefix SHA/complete-line boundary differs')
    seen = {}
    counts = Counter()
    checked_terminal_refs = []
    for line in prefix.splitlines():
        hint = json.loads(line)
        key = hint['source_key']
        ref = hint['job_ref']
        if (Path(ref['path']).resolve() != out / 'jobs' / (key + '.json') or
                hint['global_revision'] != revision):
            raise q.SourceChanged('Consumed ready job path/revision differs')
        job = q.checked(ref)
        if (q.safe_key(job) != key or job['source_key'] != key or
                Path(job['queue_root']).resolve() != out or job['global_revision'] != revision):
            raise q.SourceChanged('Consumed ready checked job identity differs')
        q.verify_job(job)
        if key in seen:
            if seen[key] != ref['sha256']:
                raise q.SourceChanged('Consumed ready duplicate source SHA differs')
            continue
        terminal_path = out / 'worker-terminals' / (key + '.json')
        notice_path = out / 'worker-announced' / (key + '.json')
        terminal_ref = q.identity(terminal_path)
        terminal = q.checked(terminal_ref)
        notice = q.checked(q.identity(notice_path))
        if (terminal != notice or terminal['original_job_ref'] != ref or
                terminal['source_key'] != key or terminal['id'] != job['id'] or
                terminal['batch_id'] != job['batch_id'] or terminal['global_revision'] != revision):
            raise q.SourceChanged('Consumed terminal/notice/job binding differs')
        if terminal.get('joint_result_ref'):
            joint = q.checked(terminal['joint_result_ref'])
            if (joint['original_job_ref'] != ref or joint['id'] != job['id'] or
                    joint['global_revision'] != revision or joint['status'] != terminal['status'] or
                    joint['joint_complete'] != terminal['joint_complete']):
                raise q.SourceChanged('Consumed joint/terminal binding differs')
            budget = q.load(out / 'attempt-budget' / (key + '.json'))
            if (budget['source_key'] != key or budget['global_revision'] != revision or
                    budget['original_result_sha256'] != job['original_result_sha256'] or
                    budget['extra_source_attempt'] != 1 or budget['maximum_extra_source_attempts'] != 1):
                raise q.SourceChanged('Consumed repair budget binding differs')
        checked_terminal_refs.append(terminal_ref)
        seen[key] = ref['sha256']
        counts[terminal['status']] += 1
    if state['seen'] != seen or state['counts'] != dict(counts):
        raise q.SourceChanged('Consumed prefix differs from complete seen/count cursor')
    after = ready.stat()
    if ([after.st_dev, after.st_ino] != current_identity or after.st_size < before.st_size or
            q.sha(state_path) != expected_state_ref['sha256']):
        raise q.SourceChanged('Ready stream/state changed during cursor migration')
    with ready.open('rb') as stream:
        if hashlib.sha256(stream.read(offset)).hexdigest() != prefix_sha256:
            raise q.SourceChanged('Consumed prefix changed during cursor migration')
    audit = out / 'cursor-migrations-v2' / (expected_state_ref['sha256'] + '-' + node)
    q.save_once(audit / 'state.before.json', state)
    q.save_once(audit / 'consumed-prefix.json', {'byte_count': offset, 'sha256': prefix_sha256,
                                               'raw_utf8': prefix.decode('utf-8')})
    migrated = copy.deepcopy(state)
    migrated.update(input_identity=current_identity, cursor_node=node,
                    consumed_prefix_sha256=prefix_sha256,
                    cursor_migration_original_state_ref=q.identity(audit / 'state.before.json'))
    q.replace_state(state_path, migrated)
    receipt = {'schema': 'curated_repair_cursor_migration_v2', 'global_revision': revision,
               'node': node, 'original_state_ref': expected_state_ref,
               'state_before_ref': q.identity(audit / 'state.before.json'),
               'state_after_ref': q.identity(state_path), 'old_identity': previous_identity,
               'new_identity': current_identity, 'validated_prefix_bytes': offset,
               'validated_prefix_sha256': prefix_sha256, 'checked_terminal_refs': checked_terminal_refs,
               'source_count': len(seen), 'offset_changed': False, 'budget_changed': False,
               'original_evidence_modified': False, 'VLM_dispatch_performed': False, **q.FLAGS}
    q.save_once(audit / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--global-revision', required=True)
    parser.add_argument('--expected-state-sha256', required=True)
    parser.add_argument('--consumed-prefix-sha256', required=True)
    args = parser.parse_args()
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        parser.error('Only the explicitly admitted N6 mount may receive this cursor')
    out = Path(args.out).resolve()
    with (out / 'resident-worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = rebind_state(out, args.global_revision,
                               {'path': str(out / 'worker.state.json'), 'sha256': args.expected_state_sha256},
                               args.consumed_prefix_sha256, 'N6')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
