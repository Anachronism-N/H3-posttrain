"""One-shot N2G0 producer parking and exact unstarted-source handoff."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import subprocess
import time
import types

RT = Path(__file__).resolve().parent
B = RT.parent
OLD_CONFIG = RT / 'curated_collaborative3_n2p0_v1_launch.json'
OLD_SHA = '54a262c7f2a69b2385382340ea03004481198804c6e1327d72db6ccde34689a0'
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
SOURCE_ROOT = B / 'curated222939_collaborative_v3'
KEYS = ('id', 'split', 'batch_id', 'source_ordinal')
PHASE = 'caption_producer_cached_n2p0_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    return {'path': str(Path(path)), 'sha256': sha(path)}


def load(path):
    return json.loads(Path(path).read_bytes())


def checked(value):
    if sha(value['path']) != value['sha256']:
        raise RuntimeError('Immutable binding changed: ' + value['path'])
    return load(value['path'])


def write_once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return ref(path)


def native_base():
    path = RT / 'resource_manager.py'
    if sha(path) != BASE_SHA:
        raise RuntimeError('Original native manager changed')
    spec = importlib.util.spec_from_file_location('_caption_cache_native_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_node(cfg):
    if cfg.get('node') != 'N2' or cfg.get('gpus') != [0] or cfg.get('mode') != 'producer':
        raise RuntimeError('Only fixed N2G0 producer is authorized')
    # Use the current original node map; never broaden its scope.
    spec = importlib.util.spec_from_file_location('_caption_cache_original_plan', RT / 'curated_collaborative3_plan_v1.py')
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    if plan.HOSTS['N2'] not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Wrong actual node')


def old_config():
    cfg = checked({'path': str(OLD_CONFIG), 'sha256': OLD_SHA})
    if (cfg['node'] != 'N2' or cfg['gpus'] != [0] or cfg['collab3_member'] != 'n2p0'
            or cfg['out'] != str(SOURCE_ROOT) or cfg['targetcount'] != 7500
            or cfg['source_range'] != {'start': 32000, 'end': 62000}
            or cfg['source_selection'] != {'shard': 0, 'nshard': 4, 'limit': 0}):
        raise RuntimeError('Original fixed producer scope changed')
    return cfg


def validate_meta(rows, original):
    inventory = {r['id']: r for r in original}
    seen = set()
    ordinals = []
    for row in rows:
        if (set(row) != set(KEYS) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', row['id'])
                or row['id'].casefold() in seen or row != inventory.get(row['id'])
                or row['split'] != 'train' or type(row['source_ordinal']) is not int
                or not 32000 <= row['source_ordinal'] < 62000 or row['source_ordinal'] % 4):
            raise RuntimeError('Remaining source differs from fixed original metadata')
        seen.add(row['id'].casefold())
        ordinals.append(row['source_ordinal'])
    if ordinals != sorted(ordinals):
        raise RuntimeError('Remaining selection must retain manifest order')
    return rows


def split_remaining(original, root):
    root = Path(root)
    # Directory creation itself is a started claim, including empty/partial dirs.
    started = set()
    sources = root / 'sources'
    if sources.exists():
        for split in sources.iterdir():
            if split.is_dir():
                started.update(p.name for p in split.iterdir())
    roi = root / 'roi'
    if roi.exists():
        started.update(p.name for p in roi.iterdir())
    ready = root / 'ready' / 'producer-0000.jsonl'
    if ready.exists():
        data = ready.read_bytes()
        complete = data[:data.rfind(b'\n') + 1]
        for line in complete.splitlines():
            value = json.loads(line)
            if not isinstance(value.get('id'), str):
                raise RuntimeError('Existing ready record has no source identity')
            started.add(value['id'])
    return [r for r in original if r['id'] not in started], [r for r in original if r['id'] in started]


def ready_prefix(path):
    data = Path(path).read_bytes() if Path(path).exists() else b''
    length = data.rfind(b'\n') + 1
    complete = data[:length]
    for line in complete.splitlines():
        json.loads(line)
    return {'path': str(path), 'complete_bytes': length, 'sha256': hashlib.sha256(complete).hexdigest(),
            'complete_lines': len(complete.splitlines()), 'trailing_bytes': len(data) - length}


def verify_prefix(value):
    with Path(value['path']).open('rb') as stream:
        prefix = stream.read(value['complete_bytes'])
    if len(prefix) != value['complete_bytes'] or hashlib.sha256(prefix).hexdigest() != value['sha256']:
        raise RuntimeError('Original complete ready prefix changed')


def exact_identity(base, expected, current):
    if any(expected[k] != current[k] for k in (*base.IDENTITY_KEYS, 'ppid')):
        raise RuntimeError('Exact process identity/PPID changed')
    if current['uid'] != os.getuid() or current['state'] == 'Z':
        raise RuntimeError('Process is not a live owned identity')


def proc_inventory(base):
    values = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                value = base.identity(int(path.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
            values[value['pid']] = value
    return values


def descendants(values, pid):
    result = {pid}
    while True:
        added = {p for p, v in values.items() if v['ppid'] in result} - result
        if not added:
            return result
        result.update(added)


def lock_state(base, parent=None):
    path = Path('/tmp/ms12k_gpu_0.lock')
    if not path.exists():
        if parent is not None:
            raise RuntimeError('Native parent lock is missing')
        return {'exists': False, 'active': False}
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size:
        raise RuntimeError('Unsafe native GPU lock')
    key = '%02x:%02x:%d' % (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
    lines = [line.split() for line in Path('/proc/locks').read_text().splitlines() if key in line.split()]
    if parent is None and lines:
        raise RuntimeError('Native GPU lock still owned')
    if parent is not None and (len(lines) != 1 or lines[0][1:4] != ['FLOCK', 'ADVISORY', 'WRITE']
                              or int(lines[0][4]) != parent):
        raise RuntimeError('Exact native parent does not own sole FLOCK')
    return {'exists': True, 'active': bool(lines), 'device': info.st_dev, 'inode': info.st_ino}


def old_owner_absence(base, proof):
    values = proc_inventory(base)
    child = proof['old_child']
    expected_tree = {v['pid']: v for v in proof['old_tree_identities']}
    for value in values.values():
        if value['state'] == 'Z':
            continue
        previous = expected_tree.get(value['pid'])
        if (previous is not None and previous['start_ticks'] == value['start_ticks']) or value['argv'] == child['argv']:
            raise RuntimeError('Original source owner/tree remains live')
        if str(OLD_CONFIG) in value['argv']:
            parent = proof['old_parent']
            exact_identity(base, parent, value)
            if str(RT / 'curated_collaborative3_resource_manager_v1.py') not in value['argv']:
                raise RuntimeError('Original native parent argv changed')
            if 'do_wait' not in (Path('/proc') / str(value['pid']) / 'wchan').read_text():
                raise RuntimeError('Safe original parent is not waiting on restored holder')
    running = [a for a in base.apps() if a['uuid'] == proof['gpu_uuid']]
    if any(a['pid'] in expected_tree and a['pid'] in values
           and expected_tree[a['pid']]['start_ticks'] == values[a['pid']]['start_ticks'] for a in running):
        raise RuntimeError('Original GPU tree still present')
    return values


def preflight_old(base, cfg):
    queue = Path(cfg['queue'])
    admission, live = load(queue / 'admission.json'), load(queue / 'live.json')
    if (admission['config_sha256'] != OLD_SHA or admission['manager_sha256'] != BASE_SHA
            or admission['node'] != 'N2' or admission['gpus'] != [0]
            or admission['producer'] != cfg['producer'] or admission['out'] != cfg['out']
            or live['argv'] != cfg['argv'] or live['phase'] != cfg['phase']):
        raise RuntimeError('Original admission/live does not bind fixed old config')
    parent, child = admission['manager_identity'], live['child_identity']
    if child is None:
        raise RuntimeError('Original live child evidence missing')
    exact_identity(base, parent, live['manager_identity'])
    exact_identity(base, parent, base.identity(parent['pid']))
    exact_identity(base, child, base.identity(child['pid']))
    if (child['ppid'] != parent['pid'] or child['argv'] != cfg['argv']
            or child['CVD'] != ['CUDA_VISIBLE_DEVICES=0']
            or child['cgroup'] != parent['cgroup']
            or parent['argv'] != cfg['argv'][:3] + [str(RT / 'curated_collaborative3_resource_manager_v1.py'),
                '--config', str(OLD_CONFIG), '--config-sha', OLD_SHA, '--manager-sha', BASE_SHA]):
        raise RuntimeError('Original native parent/child chain mismatch')
    if (parent['sig_ign'] | parent['sig_blk']) & (1 << (signal.SIGTERM - 1)):
        raise RuntimeError('Original native parent TERM unavailable')
    lock_state(base, parent['pid'])
    lease = Path('/tmp/od_reserve_0').lstat()
    if not stat.S_ISREG(lease.st_mode) or lease.st_uid != os.getuid() or lease.st_size or time.time()-lease.st_mtime > 30:
        raise RuntimeError('Original native lease heartbeat unavailable')
    inventory = proc_inventory(base)
    inventory[child['pid']] = child
    tree = descendants(inventory, child['pid'])
    gpu_uuid = base.uuid_for(0)
    running = [a for a in base.apps() if a['uuid'] == gpu_uuid]
    if len(running) != 1 or running[0]['pid'] not in tree or gpu_uuid != cfg['holders'][0]['uuid']:
        raise RuntimeError('Sole GPU app must belong to exact old child tree, not a holder')
    return {'old_parent': parent, 'old_child': child, 'old_tree_pids': sorted(tree),
            'old_tree_identities': [inventory[pid] for pid in sorted(tree)], 'gpu_uuid': gpu_uuid,
            'old_config': ref(OLD_CONFIG), 'old_admission': ref(queue / 'admission.json'),
            'old_live': ref(queue / 'live.json')}


def safe_gate(base, cfg, proof):
    queue = Path(cfg['queue'])
    safe = load(queue / 'resources_safe.json')
    if safe['phase'] != cfg['phase'] or not all(safe.get(k) is True for k in
            ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')):
        raise RuntimeError('Original resources are not safely parked')
    restored = load(queue / 'holder_restored_gpu0.json')
    owner = dict(restored['holder_identity'], gpu=0, uuid=restored['uuid'])
    if restored['gpu'] != 0 or restored['uuid'] != proof['gpu_uuid']:
        raise RuntimeError('Restored GPU binding mismatch')
    old_owner_absence(base, proof)
    base.check_holder(owner, base.identity(owner['pid']), 0, base.uuid_for(0), base.apps())
    lock_state(base)
    lease = base.lease_snapshot(Path('/tmp/od_reserve_0'))
    budget = base.headroom()
    if budget['headroom'] < base.SAFETY + base.LOAD:
        raise RuntimeError('Fresh native 160GiB admission gate fails')
    return owner, lease, budget


def native_args(cfg, ids_path):
    args = cfg['argv'][4:]
    index = args.index('--member')
    args = args[:index] + args[index+2:]
    return args + ['--source-ids-file', str(ids_path)]


def prepare(args):
    cfg = old_config()
    fixed_node(cfg)
    base = native_base()
    if base.headroom()['headroom'] < base.SAFETY + base.LOAD:
        raise RuntimeError('Fresh native 160GiB gate fails before parking')
    import shutil
    if shutil.disk_usage('/tmp').free < 8 * 2**30:
        raise RuntimeError('Local cache free space below 8GiB before parking')
    for suffix in ('_remaining.json', '_ids.txt', '_handoff.json', '_holder_capture.json', '_launch.json'):
        if (RT / (PHASE + suffix)).exists():
            raise RuntimeError('New immutable producer handoff artifact already exists')
    if (B / PHASE).exists() or (B / 'queues' / PHASE).exists():
        raise RuntimeError('New producer execution or queue already exists')
    claim = RT / (PHASE + '_one_time_claim.json')
    if claim.exists():
        raise RuntimeError('Prior producer handoff claim retained; no automatic retry')
    try:
        return prepare_once(args, cfg, base)
    except BaseException as error:
        failure = RT / (PHASE + '_handoff_failed.json')
        if claim.exists() and not failure.exists():
            write_once(failure, dict(reason=repr(error), no_automatic_retry=True,
                       old_config_resume_forbidden=True, old_outputs_and_claims_preserved=True))
        raise


def prepare_once(args, cfg, base):
    claim = RT / (PHASE + '_one_time_claim.json')
    write_once(claim, {'old_config': ref(OLD_CONFIG), 'old_config_resume_forbidden': True,
                       'attempt_unix_ns': time.time_ns(), 'no_retry_after_failure': True})
    original = checked(cfg['selected_sources'])
    if len(original) != 7500:
        raise RuntimeError('Original selected count changed')
    validate_meta(original, original)
    selected_native = base.selected_sources
    base.selected_sources = lambda value: ([r['id'] for r in original], 222939)
    try:
        base.check_sources(cfg, str(OLD_CONFIG), OLD_SHA, BASE_SHA)
    finally:
        base.selected_sources = selected_native
    spec = importlib.util.spec_from_file_location('_caption_cache_old_producer_model', RT / 'curated_resource_manager.py')
    curated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(curated)
    check_model = types.FunctionType(curated.check_model.__code__,
                                    dict(curated.check_model.__globals__, base=base), 'check_model')
    check_model(cfg)  # Original ROI model/code identity gate once, never per source.
    proof = preflight_old(base, cfg)
    # Only the exact native parent receives TERM; its native finally parks the holder.
    fd = os.pidfd_open(proof['old_parent']['pid'])
    try:
        exact_identity(base, proof['old_parent'], base.identity(proof['old_parent']['pid']))
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    deadline = time.monotonic() + args.wait_seconds
    queue = Path(cfg['queue'])
    while not (queue / 'resources_safe.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Native parking timeout; claim retained, no automatic retry/signals')
        time.sleep(1)
    owner, lease, budget = safe_gate(base, cfg, proof)
    remaining, excluded = split_remaining(original, SOURCE_ROOT)
    validate_meta(remaining, original)
    if not remaining:
        raise RuntimeError('No unstarted sources remain')
    prefix = ready_prefix(SOURCE_ROOT / 'ready' / 'producer-0000.jsonl')
    if prefix['trailing_bytes']:
        raise RuntimeError('Ready has incomplete trailing bytes; retain unchanged and require review')
    selected_path = RT / (PHASE + '_remaining.json')
    selection = write_once(selected_path, remaining)
    ids_path = RT / (PHASE + '_ids.txt')
    with ids_path.open('x') as stream:
        stream.write(''.join(row['id'] + '\n' for row in remaining))
        stream.flush()
        os.fsync(stream.fileno())
    proof.update({'schema': 'caption_producer_cache_handoff_v1', 'old_queue': cfg['queue'],
                  'old_resource_safe': ref(queue / 'resources_safe.json'),
                  'restored_holder': ref(queue / 'holder_restored_gpu0.json'),
                  'old_selected': cfg['selected_sources'], 'remaining_meta': selection, 'ids_file': ref(ids_path),
                  'excluded_started_meta': excluded, 'old_ready_prefix': prefix,
                  'source_root': str(SOURCE_ROOT), 'old_config_resume_forbidden': True,
                  'sole_ready_writer': 'n2p0', 'claim': ref(claim), 'checked_unix_ns': time.time_ns(),
                  'source_owner_absence': True, 'old_gpu_tree_absent': True,
                  'holder': owner, 'budget': budget})
    proof_path = RT / (PHASE + '_handoff.json')
    handoff = write_once(proof_path, proof)
    capture_path = RT / (PHASE + '_holder_capture.json')
    lease_value = None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid, lease.st_mode, lease.st_size, lease.st_mtime_ns]
    capture = write_once(capture_path, {'captured_unix_ns': time.time_ns(), 'node': 'N2', 'mode': 'producer',
                        'gpus': [0], 'holders': [owner], 'leases': [lease_value], 'signals_sent': False})
    worker = ref(RT / 'caption_producer_cached_worker_v1.py')
    manager = ref(RT / 'caption_producer_cached_manager_v1.py')
    core = ref(RT / 'caption_producer_stage_cache_v1.py')
    config_path = RT / (PHASE + '_launch.json')
    new = {**cfg, 'phase': PHASE, 'holders': [owner], 'holder_capture': capture,
           'out': str(B / PHASE), 'queue': str(B / 'queues' / PHASE), 'reuse_existing_results': False,
           'producer': worker, 'resource_manager': manager, 'plan': selection,
           'targetcount': len(remaining), 'remaining_meta': selection, 'ids_file': ref(ids_path),
           'source_handoff': handoff, 'handoff_code': ref(__file__), 'cache_code': core, 'old_config': ref(OLD_CONFIG),
           'source_root': str(SOURCE_ROOT), 'source_selection': {'shard': 0, 'nshard': 4, 'limit': 0},
           'native_argv': native_args(cfg, ids_path),
           'argv': cfg['argv'][:3] + [worker['path'], '--config', str(config_path)],
           'selected_sources': selection, 'old_config_resume_forbidden': True,
           'config_path': str(config_path), 'handoff_code': ref(__file__)}
    bindings = dict(cfg['source_sha256'])
    for binding in (worker, manager, core, ref(__file__), selection, ref(ids_path), handoff, capture, ref(OLD_CONFIG)):
        bindings[binding['path']] = binding['sha256']
    new['source_sha256'] = bindings
    launch = write_once(config_path, new)
    return {'config': launch, 'handoff': handoff, 'remaining': len(remaining), 'excluded_started': len(excluded),
            'launch_performed': False, 'old_config_resume_forbidden': True}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--park-and-prepare', action='store_true', required=True)
    parser.add_argument('--wait-seconds', type=int, default=300)
    args = parser.parse_args(argv)
    if args.wait_seconds <= 0:
        raise RuntimeError('Positive bounded parking wait required')
    return prepare(args)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
