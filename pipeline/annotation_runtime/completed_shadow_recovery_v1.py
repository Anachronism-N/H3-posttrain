"""CPU-only, fail-closed independent completion journal and collector migration."""
from __future__ import annotations
import argparse
import ast
import contextlib
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time
import uuid

SCHEMA = 'completed_shadow_recovery_v1'
COLLECTOR_SHA = '94b00ad0b503053f4b9bd7f7c08097bf9b6480f8a9aa248e971126a149c78b4c'

class Changed(ValueError):
    pass

def enc(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def read_stable(path):
    path = Path(path)
    before = path.stat()
    with path.open('rb') as f:
        opened = os.fstat(f.fileno())
        data = f.read()
        done = os.fstat(f.fileno())
    after = path.stat()
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len({fields(s) for s in (before, opened, done, after)}) != 1:
        raise Changed('Immutable artifact changed during read: '+str(path))
    return data

def load(path):
    return json.loads(read_stable(path))

def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha_bytes(read_stable(path))}

def checked(reference):
    data = read_stable(reference['path'])
    if sha_bytes(data) != reference['sha256']:
        raise Changed('Immutable reference SHA changed: '+reference['path'])
    return data

def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError('No progress writing journal')
        view = view[n:]

def once_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_stable(path) != data:
            raise Changed('Immutable recovery artifact differs: '+str(path))
        return
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temp, path)
        fsync_dir(path.parent)
    finally:
        temp.unlink()

def once(path, value):
    once_bytes(path, enc(value))

def replace(path, value):
    path = Path(path)
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    once_bytes(temp, enc(value))
    os.replace(temp, path)
    fsync_dir(path.parent)

def host():
    return {'hostname': socket.gethostname(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

@contextlib.contextmanager
def directory_lock(path, timeout=0):
    path = Path(path)
    deadline = time.monotonic()+timeout
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise Changed('Writer lock exists; no automatic stale-lock deletion: '+str(path))
            time.sleep(.01)
    try:
        yield
    finally:
        path.rmdir()

def safe_append(path, value, timeout=30):
    """Cross-process serialization without O_APPEND; no stale-lock stealing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with directory_lock(path.with_name(path.name+'.writer-lock'), timeout):
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.lseek(fd, 0, os.SEEK_END)
            write_all(fd, enc(value))
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_dir(path.parent)

def safe_key(row):
    import re
    if (not isinstance(row, dict) or row.get('split') not in ('train', 'val', 'test')
            or not isinstance(row.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', row['id'])):
        raise Changed('Unsafe source key')
    return row['split']+'/'+row['id']

def terminal_evidence(production, row):
    key = safe_key(row)
    terminal = production/'sources'/key/'consumer.terminal.json'
    data = read_stable(terminal)
    if json.loads(data) != row:
        raise Changed('Completed row differs from immutable terminal: '+key)
    result = {'source_key': key, 'terminal_ref': {'path': str(terminal), 'sha256': sha_bytes(data)},
              'canonical_row_sha256': sha_bytes(enc(row))}
    if row.get('joint_candidate_ref'):
        joint = row['joint_candidate_ref']
        if not Path(joint['path']).resolve().is_relative_to(production):
            raise Changed('Joint reference escapes production')
        checked(joint)
        result['joint_ref'] = joint
    return result

def source_snapshot(path):
    path = Path(path)
    with path.open('rb') as f:
        before = os.fstat(f.fileno())
        data = f.read(before.st_size)
        after = os.fstat(f.fileno())
    current = path.stat()
    identity = [before.st_dev, before.st_ino]
    if [after.st_dev, after.st_ino] != identity or [current.st_dev, current.st_ino] != identity or after.st_size < before.st_size or current.st_size < before.st_size or len(data) != before.st_size:
        raise Changed('Source replaced/truncated during snapshot')
    return data, identity

def make_plan(production, shadow, manifest_summary, offset, keys):
    production, shadow = Path(production).resolve(), Path(shadow).resolve()
    if shadow.is_relative_to(production) or production.is_relative_to(shadow):
        raise Changed('Shadow must be separate from production')
    shadow.mkdir(parents=True, exist_ok=True)
    source = production/'completed.jsonl'
    data, identity = source_snapshot(source)
    if offset and data[offset-1:offset] != b'\n':
        raise Changed('Bad offset is not a physical-line boundary')
    end = data.find(b'\n', offset)
    if end < 0:
        raise Changed('Bad line is incomplete; keep pending')
    raw = data[offset:end+1]
    try:
        json.loads(raw)
    except (ValueError, UnicodeError):
        pass
    else:
        raise Changed('Recovery plan requires malformed source bytes')
    rows, evidence = [], []
    for key in keys:
        row = load(production/'sources'/key/'consumer.terminal.json')
        if safe_key(row) != key:
            raise Changed('Terminal source key differs')
        rows.append(row)
        evidence.append(terminal_evidence(production, row))
    # Only this fully demonstrated two-write splice is admitted; no fragment parsing.
    if len(rows) != 2:
        raise Changed('Only explicitly proved two-terminal splice supported')
    first, second = [checked(e['terminal_ref']) for e in evidence]
    prefix_length = len(raw)-len(second)
    if not 0 < prefix_length < len(first) or raw != first[:prefix_length]+second:
        raise Changed('Unknown bytes: not exact terminal-prefix + complete-terminal splice')
    summary_ref = ref(manifest_summary)
    summary = json.loads(checked(summary_ref))
    batches = {row['batch_id'] for row in rows}
    if len(batches) != 1:
        raise Changed('Recovered terminals do not share fixed batch')
    batch_id = next(iter(batches))
    batch = next(b for b in summary['batches'] if b['batch_id'] == batch_id)
    manifest_ref = {'path': batch['path'], 'sha256': batch['sha256']}
    members = [json.loads(line) for line in checked(manifest_ref).splitlines()]
    index_ref = ref(production/'batches'/batch_id/'terminal_index.jsonl')
    index = [json.loads(line) for line in checked(index_ref).splitlines()]
    expected = {safe_key(r) for r in members}
    if len(members) != batch['count'] or len(expected) != batch['count'] or {safe_key(r) for r in index} != expected:
        raise Changed('Batch manifest/index fixed membership mismatch')
    if not set(keys).issubset(expected):
        raise Changed('Recovered terminal outside fixed batch')
    raw_path = shadow/'evidence'/('bad-'+str(offset)+'-'+sha_bytes(raw)+'.raw')
    once_bytes(raw_path, raw)
    plan = {'schema': SCHEMA, 'source_path': str(source), 'source_identity': identity,
            'bad_offset': offset, 'physical_line': data[:offset].count(b'\n')+1,
            'bad_raw_ref': ref(raw_path), 'bad_raw_length': len(raw),
            'prefix_before_bad_sha256': sha_bytes(data[:offset]), 'terminal_prefix_length': prefix_length,
            'terminal_evidence': evidence, 'batch_id': batch_id, 'batch_member_count': len(members),
            'manifest_summary_ref': summary_ref, 'batch_manifest_ref': manifest_ref, 'terminal_index_ref': index_ref,
            'normalization': 'derive full canonical rows only from SHA-checked immutable terminals; raw bytes retained',
            'production_mutated': False, 'GPU_dispatch_performed': False}
    once(shadow/'recovery.plan.json', plan)
    return plan

def initial_state(source, identity):
    return {'schema': SCHEMA, 'host': host(), 'source_path': str(source), 'source_identity': identity,
            'source_offset': 0, 'source_observed_size': 0, 'source_observed_sha256': sha_bytes(b''),
            'shadow_offset': 0, 'shadow_sha256': sha_bytes(b''), 'physical_lines': 0, 'normalized_source_rows': 0}

def validate_snapshot(state, source, data, identity):
    if state['host'] != host() or state['source_path'] != str(source) or state['source_identity'] != identity:
        raise Changed('Source host/path/inode binding changed; explicit migration required')
    size = state['source_observed_size']
    if len(data) < size or sha_bytes(data[:size]) != state['source_observed_sha256']:
        raise Changed('Source prior bytes changed/truncated, including pending half-line')

def apply_transaction(shadow, state, txn):
    queue = shadow/'completed.shadow.jsonl'
    payload = checked(txn['payload_ref'])
    old = queue.read_bytes() if queue.exists() else b''
    start = state['shadow_offset']
    if len(old) < start or sha_bytes(old[:start]) != state['shadow_sha256'] or old[start:] != payload[:len(old)-start] or len(old) > start+len(payload):
        raise Changed('Shadow differs from committed prefix/WAL; never truncate unknown bytes')
    fd = os.open(str(queue), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.lseek(fd, len(old), os.SEEK_SET)
        write_all(fd, payload[len(old)-start:])
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(shadow)
    next_state = txn['next_state']
    if sha_bytes(read_stable(queue)) != next_state['shadow_sha256']:
        raise Changed('Shadow post-write content mismatch')
    replace(shadow/'shadow.state.json', next_state)
    return next_state

def sync(production, shadow):
    production, shadow = Path(production).resolve(), Path(shadow).resolve()
    if shadow.is_relative_to(production) or production.is_relative_to(shadow):
        raise Changed('Shadow must remain independent')
    shadow.mkdir(parents=True, exist_ok=True)
    with directory_lock(shadow/'sync.writer-lock'):
        source = production/'completed.jsonl'
        data, identity = source_snapshot(source)
        state_path = shadow/'shadow.state.json'
        state = load(state_path) if state_path.exists() else initial_state(source, identity)
        validate_snapshot(state, source, data, identity)
        plan_path = shadow/'recovery.plan.json'
        if state.get('recovery_plan_ref'):
            plan = json.loads(checked(state['recovery_plan_ref']))
        elif plan_path.exists():
            if state['source_offset']:
                raise Changed('Cannot introduce a new recovery plan after journal commit')
            state['recovery_plan_ref'] = ref(plan_path)
            plan = json.loads(checked(state['recovery_plan_ref']))
        else:
            plan = None
        if plan and (plan['source_path'] != str(source) or plan['source_identity'] != identity
                     or sha_bytes(data[:plan['bad_offset']]) != plan['prefix_before_bad_sha256']):
            raise Changed('Recovery plan source binding/prefix differs')
        # Immutable WAL remains after commit; replay only matches the exact old checkpoint.
        txn_path = shadow/'transactions'/('from-'+sha_bytes(enc(state))+'.json')
        if txn_path.exists():
            txn = load(txn_path)
            if txn['before_state_sha256'] != sha_bytes(enc(state)):
                raise Changed('WAL original checkpoint differs')
            validate_snapshot(txn['next_state'], source, data, identity)
            state = apply_transaction(shadow, state, txn)
        queue = shadow/'completed.shadow.jsonl'
        old = read_stable(queue) if queue.exists() else b''
        if len(old) != state['shadow_offset'] or sha_bytes(old) != state['shadow_sha256']:
            raise Changed('Unexplained shadow change; no automatic truncation')
        chunks, receipts = [], []
        position = state['source_offset']
        line_count = state['physical_lines']
        normalized = state['normalized_source_rows']
        failure = None
        while True:
            end = data.find(b'\n', position)
            if end < 0:
                break
            raw = data[position:end+1]
            try:
                row = json.loads(raw)
                terminal_evidence(production, row)
                chunks.append(raw)
            except (ValueError, UnicodeError, OSError) as error:
                try:
                    if not plan or position != plan['bad_offset'] or sha_bytes(raw) != plan['bad_raw_ref']['sha256'] or checked(plan['bad_raw_ref']) != raw:
                        raise Changed('Unrecognized line or unavailable terminal; never skip: '+str(error))
                    for name in ('manifest_summary_ref', 'batch_manifest_ref', 'terminal_index_ref'):
                        checked(plan[name])
                    derived = []
                    for evidence in plan['terminal_evidence']:
                        row = json.loads(checked(evidence['terminal_ref']))
                        if terminal_evidence(production, row) != evidence:
                            raise Changed('Recovery terminal/joint evidence changed')
                        derived.append(enc(row))
                    first, second = [checked(e['terminal_ref']) for e in plan['terminal_evidence']]
                    if raw != first[:plan['terminal_prefix_length']]+second:
                        raise Changed('Bad source raw no longer exact demonstrated splice')
                    chunks.append(b''.join(derived))
                    normalized += len(derived)
                    receipts.append({'source_offset': position, 'source_length': len(raw), 'bad_raw_ref': plan['bad_raw_ref'],
                                     'derived_rows_sha256': [sha_bytes(r) for r in derived], 'terminal_evidence': plan['terminal_evidence']})
                except (ValueError, UnicodeError, OSError) as rejected:
                    raw_path = shadow/'diagnostics'/('pending-'+str(position)+'-'+sha_bytes(raw)+'.raw')
                    once_bytes(raw_path, raw)
                    failure = {'schema': SCHEMA, 'source_offset': position, 'physical_line': line_count+1,
                               'reason': str(rejected), 'raw_ref': ref(raw_path), 'pending': True, 'skipped': False}
                    once(raw_path.with_suffix('.json'), failure)
                    break
            position = end+1
            line_count += 1
        payload = b''.join(chunks)
        if payload or not state_path.exists() or state['source_observed_size'] != len(data):
            next_state = copy.deepcopy(state)
            next_state.update(source_offset=position, source_observed_size=len(data), source_observed_sha256=sha_bytes(data),
                              shadow_offset=len(old)+len(payload), shadow_sha256=sha_bytes(old+payload),
                              physical_lines=line_count, normalized_source_rows=normalized)
            payload_path = shadow/'transactions'/('payload-'+str(state['source_offset'])+'-'+sha_bytes(payload)+'.raw')
            once_bytes(payload_path, payload)
            txn = {'schema': SCHEMA, 'before_state_sha256': sha_bytes(enc(state)), 'payload_ref': ref(payload_path),
                   'normalizations': receipts, 'next_state': next_state}
            txn_path = shadow/'transactions'/('from-'+sha_bytes(enc(state))+'.json')
            once(txn_path, txn)
            state = apply_transaction(shadow, state, txn)
        if failure:
            raise Changed(json.dumps(failure, ensure_ascii=False))
        return {k: v for k, v in state.items() if k not in ('host',)}

def migrate(production, shadow, out, expected_sha):
    production, shadow, out = [Path(p).resolve() for p in (production, shadow, out)]
    source = production/'completed.jsonl'
    queue = shadow/'completed.shadow.jsonl'
    state_path = out/'collector.state.json'
    with (out/'collector.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw_state = read_stable(state_path)
        state = json.loads(raw_state)
        if sha_bytes(raw_state) != expected_sha or state['input_path'] != str(source):
            raise Changed('Expected original collector state SHA/path differs')
        raw, identity = source_snapshot(source)
        shadow_state = load(shadow/'shadow.state.json')
        validate_snapshot(shadow_state, source, raw, identity)
        if state.get('input_identity') != identity:
            raise Changed('Original collector source identity differs')
        offset = state['offset']
        old_prefix = raw[:offset]
        queue_data = read_stable(queue)
        if len(queue_data) != shadow_state['shadow_offset'] or sha_bytes(queue_data) != shadow_state['shadow_sha256']:
            raise Changed('Migration shadow content differs from committed journal')
        new_prefix = queue_data[:offset]
        if len(old_prefix) != offset or new_prefix != old_prefix or offset and not old_prefix.endswith(b'\n'):
            raise Changed('Cannot migrate checkpoint across nonidentical prefix')
        rows = [json.loads(line) for line in old_prefix.splitlines(keepends=True)]
        # Independent immutable reads are bounded; FUSE metadata latency must not serialize all 754 sources.
        with ThreadPoolExecutor(max_workers=8) as pool:
            evidence = list(pool.map(lambda row: terminal_evidence(production, row), rows))
        seen = {}
        for e in evidence:
            key = e['source_key']
            if key in seen and seen[key] != e['canonical_row_sha256']:
                raise Changed('Conflicting prefix source revision')
            seen[key] = e['canonical_row_sha256']
        if seen != state['seen']:
            raise Changed('Original seen ledger differs from exact prefix immutable terminals')
        next_state = copy.deepcopy(state)
        stat = queue.stat()
        next_state.update(input_path=str(queue), input_identity=[stat.st_dev, stat.st_ino])
        receipt = {'schema': SCHEMA, 'original_state_sha256': expected_sha, 'original_state_ref':
                   {'path': str(shadow/'migration'/'collector.original.state.json'), 'sha256': expected_sha},
                   'new_state_sha256': sha_bytes(enc(next_state)), 'new_state_ref':
                   {'path': str(shadow/'migration'/'collector.migrated.state.json'), 'sha256': sha_bytes(enc(next_state))},
                   'identical_prefix_bytes': offset, 'identical_prefix_sha256': sha_bytes(old_prefix),
                   'source_identity': identity, 'shadow_identity': next_state['input_identity'],
                   'sources_seen': len(seen), 'seen_terminal_evidence': evidence,
                   'preserved_fields': [k for k in state if k not in ('input_path', 'input_identity')],
                   'budget_jobs_ready_mutated': False, 'original_stream_mutated': False}
        once_bytes(shadow/'migration'/'collector.original.state.json', raw_state)
        once(shadow/'migration'/'collector.migrated.state.json', next_state)
        once(shadow/'migration'/'receipt.json', receipt)
        replace(state_path, next_state)
        return {k: v for k, v in receipt.items() if k != 'seen_terminal_evidence'}

def collector_module(path):
    path = Path(path).resolve()
    data = read_stable(path)
    if sha_bytes(data) != COLLECTOR_SHA:
        raise Changed('Frozen collector source SHA differs')
    spec = importlib.util.spec_from_file_location('frozen_completion_collector', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(data)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect')
    lines = data.decode().splitlines(keepends=True)
    code = ''.join(lines[function.lineno-1:function.end_lineno])
    code = code.replace('def collect(production, out, revision):', 'def collect(production, out, revision, completed_stream):', 1)
    if code.count("queue = production / 'completed.jsonl'") != 1:
        raise Changed('Frozen collector layout differs')
    code = code.replace("queue = production / 'completed.jsonl'", 'queue = Path(completed_stream).resolve()', 1)
    exec(compile(code, str(path)+'::completed-stream-derivative', 'exec'), module.__dict__)
    return module

def collect(production, shadow, out, revision, module_path):
    shadow, out = Path(shadow).resolve(), Path(out).resolve()
    with directory_lock(shadow/'sync.writer-lock'):
        queue = shadow/'completed.shadow.jsonl'
        receipt = load(shadow/'migration'/'receipt.json')
        state = load(out/'collector.state.json')
        if state['input_path'] != str(queue) or state.get('input_identity') != receipt['shadow_identity'] or state['offset'] < receipt['identical_prefix_bytes']:
            raise Changed('Collector state is not explicitly migrated shadow lineage')
        ss = load(shadow/'shadow.state.json')
        data = read_stable(queue)
        if len(data) != ss['shadow_offset'] or sha_bytes(data) != ss['shadow_sha256']:
            raise Changed('Collector shadow content differs from committed journal')
        module = collector_module(module_path)
        return module.collect(production, out, revision, queue)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'sync', 'migrate', 'collect', 'watch'))
    parser.add_argument('--production', required=True)
    parser.add_argument('--shadow-root', required=True)
    parser.add_argument('--completed-stream')
    parser.add_argument('--out')
    parser.add_argument('--manifest-summary')
    parser.add_argument('--bad-offset', type=int, default=532233)
    parser.add_argument('--terminal-key', action='append')
    parser.add_argument('--expected-state-sha')
    parser.add_argument('--collector-source', default=str(Path(__file__).parent/'source_repair_gpu_v2/curated_repair_queue_v1.py'))
    parser.add_argument('--global-revision', default='repair-20261025-v1')
    parser.add_argument('--poll-seconds', type=float, default=20)
    args = parser.parse_args(argv)
    if args.completed_stream and Path(args.completed_stream).resolve() != Path(args.shadow_root).resolve()/'completed.shadow.jsonl':
        parser.error('completed-stream must be this independent shadow journal')
    if args.mode == 'plan':
        result = make_plan(args.production, args.shadow_root, args.manifest_summary, args.bad_offset, args.terminal_key or [])
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if args.mode == 'migrate':
        print(json.dumps(migrate(args.production, args.shadow_root, args.out, args.expected_state_sha), ensure_ascii=False), flush=True)
        return
    while True:
        result = {}
        if args.mode in ('sync', 'watch'):
            result['shadow'] = sync(args.production, args.shadow_root)
        if args.mode in ('collect', 'watch'):
            result['collector'] = collect(args.production, args.shadow_root, args.out, args.global_revision, args.collector_source)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.mode != 'watch':
            return
        time.sleep(args.poll_seconds)

if __name__ == '__main__':
    main()
