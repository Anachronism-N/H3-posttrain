"""One-shot N2G1 (collab3 n2p1) producer parking -> K-way parallel native producer handoff.
Only --park-and-prepare signals: pidfd TERM to the exact verified native parent; its native finally restores
the holder.  Remaining = original 7500 minus every started source (source dir, ROI dir, or ready line).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import time
import types

import caption_producer_cache_handoff_v1 as cache

RT = Path(__file__).resolve().parent
B = RT.parent
MEMBER, NODE, GPU, SHARD, NSHARD, K = 'n2p1', 'N2', 1, 1, 4, 3
OLD_CONFIG = RT / 'curated_collaborative3_n2p1_v1_launch.json'
OLD_SHA = '4d71793223e38292ea0da39a27dcd28a464daa306c424cf10d85b477360d3e8b'
OLD_MANAGER = RT / 'curated_collaborative3_resource_manager_v1.py'
BASE_SHA = cache.BASE_SHA
SOURCE_ROOT = cache.SOURCE_ROOT
KEYS = cache.KEYS
PHASE = 'caption_producer_parallel_n2p1_v1'
READY = SOURCE_ROOT / 'ready' / ('producer-%04d.jsonl' % SHARD)
LOCK = Path('/tmp/ms12k_gpu_%d.lock' % GPU)
LEASE = Path('/tmp/od_reserve_%d' % GPU)
CACHE_HANDOFF_SHA = '9b96eff099a5d2c72deed182779bd75775d52b0272b2622d0456a2ce85ed1334'

sha, ref, load, checked, write_once, native_base = cache.sha, cache.ref, cache.load, cache.checked, cache.write_once, cache.native_base
exact_identity, proc_inventory, descendants, ready_prefix, verify_prefix = (
    cache.exact_identity, cache.proc_inventory, cache.descendants, cache.ready_prefix, cache.verify_prefix)


def fixed_node(cfg):
    if cfg.get('node') != NODE or cfg.get('gpus') != [GPU] or cfg.get('mode') != 'producer':
        raise RuntimeError('Only fixed N2G1 producer is authorized')
    spec = importlib.util.spec_from_file_location('_caption_parallel_plan', RT / 'curated_collaborative3_plan_v1.py')
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    if plan.MEMBERS[MEMBER] != (NODE, 'producer', [GPU], SHARD, NSHARD):
        raise RuntimeError('Original collab3 member map changed')
    if plan.HOSTS[NODE] not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Wrong actual node')


def old_config():
    if sha(cache.Path(cache.__file__)) != CACHE_HANDOFF_SHA:
        raise RuntimeError('Frozen cache handoff helper changed')
    cfg = checked({'path': str(OLD_CONFIG), 'sha256': OLD_SHA})
    if (cfg['node'] != NODE or cfg['gpus'] != [GPU] or cfg['collab3_member'] != MEMBER
            or cfg['out'] != str(SOURCE_ROOT) or cfg['targetcount'] != 7500
            or cfg['source_range'] != {'start': 32000, 'end': 62000}
            or cfg['source_selection'] != {'shard': SHARD, 'nshard': NSHARD, 'limit': 0}
            or cfg['env'].get('CUDA_VISIBLE_DEVICES') != str(GPU)):
        raise RuntimeError('Original fixed producer scope changed')
    return cfg


def validate_meta(rows, original):
    inventory = {r['id']: r for r in original}
    seen, ordinals = set(), []
    for row in rows:
        if (set(row) != set(KEYS) or row != inventory.get(row['id']) or row['id'].casefold() in seen
                or row['split'] != 'train' or type(row['source_ordinal']) is not int
                or not 32000 <= row['source_ordinal'] < 62000 or row['source_ordinal'] % NSHARD != SHARD):
            raise RuntimeError('Remaining source differs from fixed original metadata')
        seen.add(row['id'].casefold())
        ordinals.append(row['source_ordinal'])
    if ordinals != sorted(ordinals):
        raise RuntimeError('Remaining selection must retain manifest order')
    return rows


def split_remaining(original, root=SOURCE_ROOT):
    """Directory creation is a started claim (source or ROI dir), as is any complete ready line."""
    root = Path(root)
    wanted = {r['id'] for r in original}
    started = set()
    for split in ('train', 'val', 'test'):
        directory = root / 'sources' / split
        if directory.exists():
            with os.scandir(directory) as entries:
                started.update(e.name for e in entries if e.name in wanted)
    if (root / 'roi').exists():
        with os.scandir(root / 'roi') as entries:
            started.update(e.name for e in entries if e.name in wanted)
    ready = root / 'ready' / ('producer-%04d.jsonl' % SHARD)
    if ready.exists():
        data = ready.read_bytes()
        for line in data[:data.rfind(b'\n') + 1].splitlines():
            value = json.loads(line)
            if not isinstance(value.get('id'), str):
                raise RuntimeError('Existing ready record has no source identity')
            started.add(value['id'])
    return [r for r in original if r['id'] not in started], [r for r in original if r['id'] in started]


def partition(rows, k=K):
    """Round-robin keeps every child near the global manifest frontier (consumers need ordinal order)."""
    parts = [rows[i::k] for i in range(k)]
    if any(not p for p in parts) or sorted((r['id'] for p in parts for r in p)) != sorted(r['id'] for r in rows):
        raise RuntimeError('Exact nonempty disjoint K-way partition required')
    return parts


def lock_state(parent=None):
    if not LOCK.exists():
        if parent is not None:
            raise RuntimeError('Native parent lock is missing')
        return {'exists': False, 'active': False}
    info = LOCK.lstat()
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


def fresh_lease():
    lease = LEASE.lstat()
    if not stat.S_ISREG(lease.st_mode) or lease.st_uid != os.getuid() or lease.st_size or time.time() - lease.st_mtime > 30:
        raise RuntimeError('Native lease heartbeat unavailable')


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
            exact_identity(base, proof['old_parent'], value)
            if str(OLD_MANAGER) not in value['argv']:
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
            or admission['node'] != NODE or admission['gpus'] != [GPU]
            or admission['producer'] != cfg['producer'] or admission['out'] != cfg['out']
            or live['argv'] != cfg['argv'] or live['phase'] != cfg['phase']):
        raise RuntimeError('Original admission/live does not bind fixed old config')
    if (queue / 'resources_safe.json').exists():
        raise RuntimeError('Original producer already parked/exited; this one-shot handoff only parks a live owner')
    parent, child = admission['manager_identity'], live['child_identity']
    if child is None:
        raise RuntimeError('Original live child evidence missing')
    exact_identity(base, parent, live['manager_identity'])
    exact_identity(base, parent, base.identity(parent['pid']))
    exact_identity(base, child, base.identity(child['pid']))
    if (child['ppid'] != parent['pid'] or child['argv'] != cfg['argv']
            or child['CVD'] != ['CUDA_VISIBLE_DEVICES=%d' % GPU] or child['cgroup'] != parent['cgroup']
            or parent['argv'] != cfg['argv'][:3] + [str(OLD_MANAGER), '--config', str(OLD_CONFIG),
                                                    '--config-sha', OLD_SHA, '--manager-sha', BASE_SHA]):
        raise RuntimeError('Original native parent/child chain mismatch')
    if (parent['sig_ign'] | parent['sig_blk']) & (1 << (signal.SIGTERM - 1)):
        raise RuntimeError('Original native parent TERM unavailable')
    lock_state(parent['pid'])
    fresh_lease()
    inventory = proc_inventory(base)
    inventory[child['pid']] = child
    tree = descendants(inventory, child['pid'])
    gpu_uuid = base.uuid_for(GPU)
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
    restored = load(queue / ('holder_restored_gpu%d.json' % GPU))
    owner = dict(restored['holder_identity'], gpu=GPU, uuid=restored['uuid'])
    if restored['gpu'] != GPU or restored['uuid'] != proof['gpu_uuid']:
        raise RuntimeError('Restored GPU binding mismatch')
    old_owner_absence(base, proof)
    base.check_holder(owner, base.identity(owner['pid']), GPU, base.uuid_for(GPU), base.apps())
    lock_state()
    lease = base.lease_snapshot(LEASE)
    budget = base.headroom()
    if budget['headroom'] < base.SAFETY + base.LOAD:
        raise RuntimeError('Fresh native 160GiB admission gate fails')
    return owner, lease, budget


def native_args(cfg, ids_path):
    args = cfg['argv'][4:]
    index = args.index('--member')
    if args[index + 1] != MEMBER:
        raise RuntimeError('Original member argv changed')
    return args[:index] + args[index + 2:] + ['--source-ids-file', str(ids_path)]


def capacity(base):
    if base.headroom()['headroom'] < base.SAFETY + base.LOAD:
        raise RuntimeError('Fresh native 160GiB gate fails before parking')
    if shutil.disk_usage('/tmp').free < 8 * 2**30:
        raise RuntimeError('Local /tmp free space below 8GiB before parking')


def artifacts():
    names = ['_remaining.json', '_handoff.json', '_holder_capture.json', '_launch.json']
    names += ['_ids_child%d.txt' % i for i in range(K)]
    return [RT / (PHASE + n) for n in names]


def prepare(args):
    cfg = old_config()
    fixed_node(cfg)
    base = native_base()
    capacity(base)
    for path in artifacts():
        if path.exists():
            raise RuntimeError('New immutable producer handoff artifact already exists')
    if (B / PHASE).exists() or (B / 'queues' / PHASE).exists():
        raise RuntimeError('New producer execution or queue already exists')
    claim = RT / (PHASE + '_one_time_claim.json')
    if claim.exists():
        raise RuntimeError('Prior producer handoff claim retained; no automatic retry')
    # Read-only identity/model checks precede the claim; failures here neither signal nor seal.
    original = checked(cfg['selected_sources'])
    if len(original) != 7500:
        raise RuntimeError('Original selected count changed')
    validate_meta(original, original)
    proof = preflight_old(base, cfg)
    if args.check_only:
        return {'status': 'old_producer_checked', 'old_parent': proof['old_parent']['pid'],
                'old_child': proof['old_child']['pid'], 'gpu_tree': proof['old_tree_pids'], 'signals_sent': False}
    write_once(claim, {'old_config': ref(OLD_CONFIG), 'old_config_resume_forbidden': True,
                       'attempt_unix_ns': time.time_ns(), 'no_retry_after_failure': True, 'K': K})
    try:
        return prepare_once(args, cfg, base, original, claim)
    except BaseException as error:
        failure = RT / (PHASE + '_handoff_failed.json')
        if not failure.exists():
            write_once(failure, dict(reason=repr(error), no_automatic_retry=True,
                       old_config_resume_forbidden=True, old_outputs_and_claims_preserved=True))
        raise


def prepare_once(args, cfg, base, original, claim):
    proof = preflight_old(base, cfg)
    capacity(base)
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
    remaining, excluded = split_remaining(original)
    validate_meta(remaining, original)
    if not remaining:
        raise RuntimeError('No unstarted sources remain')
    prefix = ready_prefix(READY)
    if prefix['trailing_bytes']:
        raise RuntimeError('Ready has incomplete trailing bytes; retain unchanged and require review')
    selection = write_once(RT / (PHASE + '_remaining.json'), remaining)
    ids_refs = []
    for i, part in enumerate(partition(remaining)):
        path = RT / (PHASE + '_ids_child%d.txt' % i)
        with path.open('x') as stream:
            stream.write(''.join(r['id'] + '\n' for r in part))
            stream.flush()
            os.fsync(stream.fileno())
        ids_refs.append(ref(path))
    proof.update({'schema': 'caption_producer_parallel_handoff_v1', 'old_queue': cfg['queue'],
                  'old_resource_safe': ref(queue / 'resources_safe.json'),
                  'restored_holder': ref(queue / ('holder_restored_gpu%d.json' % GPU)),
                  'old_selected': cfg['selected_sources'], 'remaining_meta': selection, 'child_ids_files': ids_refs,
                  'excluded_started_meta': excluded, 'old_ready_prefix': prefix, 'source_root': str(SOURCE_ROOT),
                  'old_config_resume_forbidden': True, 'ready_writer': str(READY), 'ready_writers': 'K native flock appenders',
                  'claim': ref(claim), 'checked_unix_ns': time.time_ns(), 'source_owner_absence': True,
                  'old_gpu_tree_absent': True, 'holder': owner, 'budget': budget, 'K': K})
    handoff = write_once(RT / (PHASE + '_handoff.json'), proof)
    lease_value = None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid, lease.st_mode, lease.st_size, lease.st_mtime_ns]
    capture = write_once(RT / (PHASE + '_holder_capture.json'), {'captured_unix_ns': time.time_ns(), 'node': NODE,
                         'mode': 'producer', 'gpus': [GPU], 'holders': [owner], 'leases': [lease_value], 'signals_sent': False})
    worker = ref(RT / 'caption_producer_parallel_worker_v1.py')
    manager = ref(RT / 'caption_producer_parallel_manager_v1.py')
    config_path = RT / (PHASE + '_launch.json')
    new = {**cfg, 'phase': PHASE, 'holders': [owner], 'holder_capture': capture,
           'out': str(B / PHASE), 'queue': str(B / 'queues' / PHASE), 'reuse_existing_results': False,
           'producer': worker, 'resource_manager': manager, 'plan': selection, 'targetcount': len(remaining),
           'remaining_meta': selection, 'child_ids_files': ids_refs, 'parallel_children': K,
           'source_handoff': handoff, 'handoff_code': ref(__file__), 'old_config': ref(OLD_CONFIG),
           'source_root': str(SOURCE_ROOT), 'source_selection': {'shard': SHARD, 'nshard': NSHARD, 'limit': 0},
           'child_native_argv': [native_args(cfg, r['path']) for r in ids_refs],
           'argv': cfg['argv'][:3] + [worker['path'], '--config', str(config_path)],
           'selected_sources': selection, 'old_config_resume_forbidden': True, 'config_path': str(config_path)}
    bindings = dict(cfg['source_sha256'])
    for binding in [worker, manager, ref(__file__), ref(cache.__file__), selection, handoff, capture, ref(OLD_CONFIG)] + ids_refs:
        bindings[binding['path']] = binding['sha256']
    new['source_sha256'] = bindings
    launch = write_once(config_path, new)
    return {'config': launch, 'handoff': handoff, 'remaining': len(remaining), 'excluded_started': len(excluded),
            'children': [sum(1 for _ in open(r['path'])) for r in ids_refs], 'launch_performed': False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--park-and-prepare', action='store_true')
    action.add_argument('--check-only', action='store_true')
    parser.add_argument('--wait-seconds', type=int, default=300)
    args = parser.parse_args(argv)
    if args.wait_seconds <= 0:
        raise RuntimeError('Positive bounded parking wait required')
    return prepare(args)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
