"""N8 shard0 prepare-ahead: frozen native lifecycle with exact safe handoff.

Plan checks are CPU-only. Build requires real predecessor restoration receipts.
Signals, leases and restoration during execution remain in the original base main.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import time
import sys

RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
NODE = 'N8'
GPUS = [4, 5]
PHASE = 'curated_consumer_prepare_ahead_n8s0_v1'
AHEAD = RT / 'curated_consumer_prepare_ahead_v1.py'
AHEAD_SHA = '02d792a4bf9aa7977dcbbc783fedcd95f4a06a0b85c223b6c34a5e898d4a1d49'
TTL_NS = 1200 * 10**9
FROZEN = {
    'resource_manager.py': '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6',
    'curated_resource_manager.py': '767fedabe779081b57b09d060a908194f6892dedf58929c868ef67deeefe8a61',
    'curated_consumer_resource_manager.py': '48c942baddf67396b28403b859eb459f6529bef52d6a705ddfbb02592a6f684f',
    'curated_consumer_image_cache_v1.py': '69eca5c2feadc4e00cbf4ef3ad51a4dbb0027f17c70f1b46990d3b0b8f7490d0',
    'curated_request_image_cache_v1.py': '8e40c5f784ebd548a2ab7d27450cb75a8d4e145576886ed690330925b51c79e3',
    'curated_consumer_prepare_ahead_v1.py': AHEAD_SHA}
for name, expected in FROZEN.items():
    if hashlib.sha256((RT / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen consumer dependency changed: ' + name)
spec = importlib.util.spec_from_file_location('native_n8s0_prepare_ahead_lifecycle', RT / 'curated_consumer_resource_manager.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
c = native.c
original_validate = c.base.validate_pair
OLD = {0: ('curated_consumer_s0_v1_launch.json',
    'e2ffce23db1d2bce352e8f564fba6f17c73041463d509611850a8d5c5c069597')}
BASE_SHA = FROZEN['resource_manager.py']
MANAGER = PREPARE = SELF
NEW_FIELDS = {'original_consumer_config_ref', 'resource_manager_ref', 'preparation_ref',
              'handoff', 'fresh_capture'}
CHANGED = {'holders', 'argv', 'phase', 'queue', 'producer', 'source_sha256'}
original_check_sources = c.base.check_sources
original_configure_restore = c.base.configure_local_restore
original_acquire_lease = c.base.acquire_lease
original_release_holder = c.base.release_holder


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': c.base.sha(path)}


def load_ref(reference, expected_path=None):
    path = Path(reference['path'])
    if (not path.is_absolute() or str(path.resolve()) != str(path)
            or (expected_path is not None and path != expected_path)):
        raise RuntimeError('Receipt reference path mismatch')
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
        raise RuntimeError('Unsafe receipt file type/owner')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise RuntimeError('Receipt SHA changed: ' + str(path))
    return json.loads(raw)


def original_config(shard):
    if type(shard) is not int or shard not in OLD:
        raise RuntimeError('Only original N8 consumer shard0 GPU4,5 admitted')
    name, digest = OLD[shard]
    reference = {'path': str(RT / name), 'sha256': digest}
    return load_ref(reference, RT / name), reference


def plan_config(shard):
    old, reference = original_config(shard)
    cfg = copy.deepcopy(old)
    cfg.update(original_consumer_config_ref=reference, resource_manager_ref=ref(MANAGER),
               preparation_ref=ref(PREPARE), handoff=None, fresh_capture=None)
    cfg['argv'][3] = str(AHEAD)
    cfg['phase'] = PHASE
    cfg['queue'] = str(RT.parent / 'queues' / cfg['phase'])
    cfg['producer'] = {'path': cfg['argv'][3], 'sha256': AHEAD_SHA}
    for name, digest in FROZEN.items():
        cfg['source_sha256'][str(RT / name)] = digest
    for reference in (cfg['original_consumer_config_ref'], cfg['resource_manager_ref'],
                      cfg['preparation_ref'], cfg['original_full_manifest'], cfg['plan']):
        cfg['source_sha256'][reference['path']] = reference['sha256']
    return cfg


def validate_semantics(cfg):
    shard = cfg['source_selection']['shard']
    old, expected = original_config(shard)
    if set(cfg) != set(old) | NEW_FIELDS or cfg['original_consumer_config_ref'] != expected:
        raise RuntimeError('Exact original consumer fields/config required')
    for key in set(old) - CHANGED:
        if cfg[key] != old[key]:
            raise RuntimeError('Native semantic/scheduling contract changed: ' + key)
    argv = list(old['argv'])
    argv[3] = str(AHEAD)
    if cfg['argv'] != argv or cfg['node'] != NODE or cfg['gpus'] != GPUS or cfg['mode'] != 'consumer':
        raise RuntimeError('Only native N8 shard0 consumer execution entrypoint may change')
    if (cfg['phase'] != PHASE
            or cfg['queue'] != str(RT.parent / 'queues' / cfg['phase'])):
        raise RuntimeError('New isolated consumer resource queue required')
    if cfg['producer'] != {'path': argv[3], 'sha256': AHEAD_SHA}:
        raise RuntimeError('Frozen prepare-ahead execution wrapper required')
    for name, digest in old['source_sha256'].items():
        if cfg['source_sha256'].get(name) != digest:
            raise RuntimeError('Original source binding changed: ' + name)
    for name, digest in FROZEN.items():
        if (cfg['source_sha256'].get(str(RT / name)) != digest
                or c.base.sha(RT / name) != digest):
            raise RuntimeError('Frozen consumer dependency changed: ' + name)
    for key, path in (('resource_manager_ref', MANAGER), ('preparation_ref', PREPARE),
                      ('original_consumer_config_ref', RT / OLD[shard][0])):
        reference = cfg[key]
        if (reference != ref(path)
                or cfg['source_sha256'].get(str(path)) != reference['sha256']):
            raise RuntimeError('New lifecycle/config source binding changed: ' + key)
    expected_sources = dict(old['source_sha256'])
    expected_sources.update({str(RT / name): digest for name, digest in FROZEN.items()})
    for reference in (cfg['original_consumer_config_ref'], cfg['resource_manager_ref'],
                      cfg['preparation_ref'], cfg['original_full_manifest'], cfg['plan']):
        expected_sources[reference['path']] = reference['sha256']
    if cfg['handoff'] is not None:
        for reference in cfg['handoff']['receipt_refs']:
            existing = expected_sources.get(reference['path'])
            if existing is not None and existing != reference['sha256']:
                raise RuntimeError('Handoff must not re-sign original source SHA')
            expected_sources[reference['path']] = reference['sha256']
    if cfg['source_sha256'] != expected_sources:
        raise RuntimeError('Only necessary frozen execution/handoff source fields may be added')


def check_plan(cfg):
    # Source/model validation is CPU-only and must precede any holder or lock IO.
    validate_semantics(cfg)
    oldref = cfg['original_consumer_config_ref']
    original_check_sources(cfg, oldref['path'], oldref['sha256'], BASE_SHA)
    for key in ('original_full_manifest', 'manifest_summary', 'modelref'):
        reference = cfg[key]
        if c.base.sha(reference['path']) != reference['sha256']:
            raise RuntimeError('Original input/model reference changed: ' + key)
    c.check_model(cfg)


def same_identity(current, expected):
    return all(current[key] == expected[key] for key in c.base.IDENTITY_KEYS)


def handoff_gate(cfg, transition_ref, require_binding=False, require_parent=False):
    old, oldref = original_config(cfg['source_selection']['shard'])
    transition = load_ref(transition_ref)
    if not Path(transition_ref['path']).is_relative_to(RT.parent):
        raise RuntimeError('Main transition must remain in current experiment')
    queue = Path(old['queue'])
    if (transition.get('schema') != 'curated_consumer_prefetch_n8s0_transition_v1'
            or transition.get('node') != NODE or transition.get('shard') != old['source_selection']['shard']
            or transition.get('reason') != 'intentional_reschedule'
            or transition.get('worker_exit_claimed', False) is not False
            or transition.get('old_config_ref') != oldref):
        raise RuntimeError('Exact intentional-reschedule main transition required')
    admission_ref = transition['admission_ref']
    safe_ref = transition['safe_ref']
    live_ref = transition['live_ref']
    admission = load_ref(admission_ref, queue / 'admission.json')
    safe = load_ref(safe_ref, queue / 'resources_safe.json')
    live = load_ref(live_ref, queue / 'live.json')
    if (safe.get('phase') != old['phase'] or not all(safe.get(key) is True for key in
            ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled'))):
        raise RuntimeError('Original exact manager resources are not safe/restored')
    if (admission.get('config_sha256') != oldref['sha256']
            or admission.get('manager_sha256') != BASE_SHA
            or admission.get('source_sha256') != old['source_sha256']
            or admission.get('model_identity_sha256') != old['modelref']['model_identity_sha256']
            or any(admission.get(key) != old[key] for key in ('phase','node','gpus','out','producer','plan','targetcount'))):
        raise RuntimeError('Original manager admission/config binding changed')
    if (len(admission['holders']) != len(old['holders'])
            or any(admitted['gpu'] != owner['gpu'] or admitted['uuid'] != owner['uuid']
                   or not same_identity(admitted['identity'], owner)
                   for admitted, owner in zip(admission['holders'], old['holders']))):
        raise RuntimeError('Original admitted holder inventory changed')
    identity = admission['manager_identity']
    expected_argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(RT / 'curated_consumer_resource_manager.py'),
                     '--config', oldref['path'], '--config-sha', oldref['sha256'], '--manager-sha', BASE_SHA]
    try:
        current_manager = c.base.identity(identity['pid'])
    except FileNotFoundError:
        current_manager = None
    if (identity['argv'] != expected_argv or identity['uid'] != os.getuid()
            or transition.get('old_manager_identity') != identity
            or not same_identity(live['manager_identity'], identity)
            or (current_manager is None and require_parent)
            or (current_manager is not None and (not same_identity(current_manager, identity)
                or (require_parent and current_manager['state'] in ('Z', 'X'))))):
        raise RuntimeError('Exact original manager must remain in restored-holder parent wait')
    if (live.get('phase') != old['phase'] or live.get('argv') != old['argv']
            or live.get('targetcount') != old['targetcount']):
        raise RuntimeError('Original live worker binding changed')
    child = live['child_identity']
    if (child is None or child['argv'] != old['argv'] or child['ppid'] != identity['pid']
            or child['uid'] != identity['uid'] or child['cgroup'] != identity['cgroup']
            or child['CVD'] != ['CUDA_VISIBLE_DEVICES=' + old['env']['CUDA_VISIBLE_DEVICES']]):
        raise RuntimeError('Exact original worker identity required')
    try:
        current_child = c.base.identity(child['pid'])
    except FileNotFoundError:
        current_child = None
    if (current_child is not None and same_identity(current_child, child)
            and current_child['state'] not in ('Z', 'X')):
        raise RuntimeError('Original consumer worker still alive')
    outcome_ref = transition['outcome_ref']
    if Path(outcome_ref['path']) not in (queue / 'failure.json', queue / 'worker_exit.json'):
        raise RuntimeError('Transition must bind an actual old-manager outcome')
    outcome = load_ref(outcome_ref)
    if outcome.get('phase') != old['phase']:
        raise RuntimeError('Old manager outcome phase mismatch')
    if Path(outcome_ref['path']).name == 'failure.json':
        if (outcome.get('worker_started') is not True or outcome.get('unknown_jobs_not_signaled') is not True
                or outcome.get('partial_outputs_preserved') is not True
                or not isinstance(outcome.get('error'), str) or not outcome['error']):
            raise RuntimeError('Actual old worker stop receipt required')
    elif (outcome.get('targetcount') != old['targetcount'] or outcome.get('producer') != old['producer']
            or outcome.get('plan') != old['plan'] or outcome.get('source_selection') != old['source_selection']
            or type(outcome.get('exit_code')) is not int or outcome.get('source_unchanged') is not True
            or outcome.get('input_count') != old['input_count']):
        raise RuntimeError('Old worker outcome identity mismatch')
    restored_refs = transition['restored_holder_refs']
    if len(restored_refs) != len(old['gpus']):
        raise RuntimeError('Both original holder restoration receipts required')
    restores = []
    admission_ns = Path(admission_ref['path']).stat().st_mtime_ns
    safe_ns = Path(safe_ref['path']).stat().st_mtime_ns
    outcome_ns = Path(outcome_ref['path']).stat().st_mtime_ns
    for gpu, reference, admitted in zip(old['gpus'], restored_refs, admission['holders']):
        restored = load_ref(reference, queue / ('holder_restored_gpu%d.json' % gpu))
        restored_identity = restored['holder_identity']
        if (restored.get('gpu') != gpu or restored.get('uuid') != admitted['uuid']
                or admitted['gpu'] != gpu or not same_identity(restored['original_identity'], admitted['identity'])
                or restored.get('managed_parent_wait') is not True
                or restored.get('restore_argv') != c.base.RESTORE_ARGV
                or restored_identity['argv'] != c.base.RESTORE_ARGV
                or restored_identity['uid'] != identity['uid']
                or restored_identity['exe'] != identity['exe']
                or restored_identity['cgroup'] != identity['cgroup']
                or restored_identity['state'] in ('Z', 'X')
                or (restored_identity['sig_ign'] | restored_identity['sig_blk']) & (1 << 14)
                or restored_identity['ppid'] != identity['pid']
                or restored_identity['CVD'] != ['CUDA_VISIBLE_DEVICES=' + str(gpu)]
                or not outcome_ns <= Path(reference['path']).stat().st_mtime_ns <= safe_ns):
            raise RuntimeError('Exact restored-holder provenance changed: GPU%d' % gpu)
        restores.append(restored)
    checked_ns = transition.get('checked_unix_ns')
    if (type(checked_ns) is not int or not admission_ns <= Path(live_ref['path']).stat().st_mtime_ns <= outcome_ns <= safe_ns <= checked_ns <= time.time_ns()):
        raise RuntimeError('Main transition/safe/outcome chronology mismatch')
    references = [oldref, transition_ref, admission_ref, live_ref, outcome_ref, safe_ref] + restored_refs
    if transition.get('signal_capture_ref') is not None:
        signal_ref = transition['signal_capture_ref']
        if not Path(signal_ref['path']).is_relative_to(RT.parent):
            raise RuntimeError('Signal-capture reference outside current experiment')
        load_ref(signal_ref)
        references.append(signal_ref)
    if require_binding:
        if cfg['handoff'] != {'transition_ref': transition_ref, 'receipt_refs': references}:
            raise RuntimeError('Bound predecessor handoff receipt set changed')
        for reference in references:
            if cfg['source_sha256'].get(reference['path']) != reference['sha256']:
                raise RuntimeError('Handoff receipt not source-bound')
    return restores, references


def stat_value(st):
    return None if st is None else {'device': st.st_dev, 'inode': st.st_ino, 'uid': st.st_uid,
        'mode': st.st_mode, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def check_stat_rows(rows):
    fields = {'device', 'inode', 'uid', 'mode', 'size', 'mtime_ns'}
    if len(rows) != 2 or [row.get('gpu') for row in rows] != GPUS:
        raise RuntimeError('Exactly ordered GPU4,5 lock/lease snapshots required')
    for row in rows:
        if set(row) != {'gpu', 'lock', 'lease'} or row['lock'] is None:
            raise RuntimeError('Complete lock/lease snapshot required')
        for name in ('lock', 'lease'):
            value = row[name]
            if value is not None and (set(value) != fields
                    or any(type(v) is not int for v in value.values())
                    or not stat.S_ISREG(value['mode']) or value['uid'] != os.getuid() or value['size'] != 0):
                raise RuntimeError('Lock/lease must use complete regular/owned/empty integer-ns identity')


def lease_snapshot(path):
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None
    if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size != 0
            or time.time_ns() - st.st_mtime_ns <= TTL_NS):
        raise RuntimeError('Unsafe or active lease before fresh native admission')
    return st


def fresh_capture(cfg, restores):
    if cfg['node'] != NODE or cfg['gpus'] != GPUS or len(restores) != 2:
        raise RuntimeError('Fresh capture requires exact original N8 GPU4,5 scope')
    if c.HOSTS[NODE] not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Fresh prepare-ahead construction is N8-only')
    locks, descriptors, owners, snapshots = [], [], [], []
    try:
        for gpu in cfg['gpus']:
            path = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_size != 0:
                raise RuntimeError('Unsafe original lock file')
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            locks.append(fd)
            if stat_value(os.fstat(fd)) != stat_value(before):
                raise RuntimeError('Lock inode changed before capture')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            snapshots.append({'gpu': gpu, 'lock': stat_value(before),
                              'lease': stat_value(lease_snapshot(Path('/tmp/od_reserve_%d' % gpu)))})
        for gpu, restored in zip(cfg['gpus'], restores):
            uuid = c.base.uuid_for(gpu)
            if uuid != restored['uuid']:
                raise RuntimeError('Restored physical GPU UUID changed')
            running = c.base.apps()
            apps = [app for app in running if app['uuid'] == uuid]
            expected = dict(restored['holder_identity'], gpu=gpu, uuid=uuid)
            if [app['pid'] for app in apps] != [expected['pid']]:
                raise RuntimeError('Only the exact restored project holder may occupy GPU%d' % gpu)
            current = c.base.identity(expected['pid'])
            if (not same_identity(current, expected) or current['ppid'] != expected['ppid']
                    or current['state'] in ('Z', 'X')):
                raise RuntimeError('Restored-holder PID/identity changed')
            c.base.check_holder(expected, current, gpu, uuid, running)
            fd = os.pidfd_open(expected['pid'])
            descriptors.append(fd)
            if select.select([fd], [], [], 0)[0]:
                raise RuntimeError('Restored holder already exited')
            owners.append(dict(current, gpu=gpu, uuid=uuid))
        budget = c.base.headroom()
        if budget['headroom'] < c.base.required_budget(cfg):
            raise RuntimeError('Fresh headroom below native 160GiB admission budget')
        for gpu, owner, captured in zip(cfg['gpus'], owners, snapshots):
            current = c.base.identity(owner['pid'])
            c.base.check_holder(owner, current, gpu, c.base.uuid_for(gpu), c.base.apps())
            if (not same_identity(current, owner) or current['ppid'] != owner['ppid']
                    or stat_value(Path('/tmp/ms12k_gpu_%d.lock' % gpu).lstat()) != captured['lock']
                    or stat_value(lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))) != captured['lease']):
                raise RuntimeError('Fresh holder/lock/lease changed during capture')
        budget = c.base.headroom()
        if budget['headroom'] < c.base.required_budget(cfg):
            raise RuntimeError('Fresh headroom changed below native 160GiB budget')
        return {'node': NODE, 'gpus': cfg['gpus'], 'holders': owners, 'locks_leases': snapshots,
                'headroom': budget, 'captured_unix_ns': time.time_ns(), 'boot_id':
                Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'signals_sent': False, 'gpu_started': False, 'lock_lease_contents_modified': False}
    finally:
        for fd in descriptors:
            os.close(fd)
        for fd in locks:
            os.close(fd)


def compare_capture(cfg, captured, fresh):
    if (type(captured.get('captured_unix_ns')) is not int
            or not 0 <= time.time_ns() - captured['captured_unix_ns'] <= TTL_NS
            or captured.get('node') != NODE or captured.get('gpus') != cfg['gpus']
            or captured.get('signals_sent') is not False or captured.get('gpu_started') is not False
            or captured.get('lock_lease_contents_modified') is not False
            or captured.get('headroom', {}).get('headroom', 0) < c.base.required_budget(cfg)
            or fresh.get('headroom', {}).get('headroom', 0) < c.base.required_budget(cfg)
            or captured['boot_id'] != fresh['boot_id'] or captured['locks_leases'] != fresh['locks_leases']
            or cfg['holders'] != captured['holders'] or len(captured['holders']) != len(fresh['holders'])
            or any(not same_identity(now, before) or now['ppid'] != before['ppid']
                   or now['gpu'] != before['gpu'] or now['uuid'] != before['uuid']
                   for now, before in zip(fresh['holders'], captured['holders']))):
        raise RuntimeError('Bound fresh holder/lock/lease/headroom capture changed or expired')
    check_stat_rows(captured['locks_leases'])


def capture_lifetime(cfg):
    captured = cfg['fresh_capture']
    if (type(captured.get('captured_unix_ns')) is not int
            or not 0 <= time.time_ns() - captured['captured_unix_ns'] <= TTL_NS
            or captured.get('node') != NODE or captured.get('gpus') != GPUS
            or captured.get('boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise RuntimeError('N8 handoff capture expired or boot/physical scope changed')
    check_stat_rows(captured['locks_leases'])


def validate(cfg):
    check_plan(cfg)
    original_validate(cfg)
    restores, _ = handoff_gate(cfg, cfg['handoff']['transition_ref'], require_binding=True, require_parent=True)
    compare_capture(cfg, cfg['fresh_capture'], fresh_capture(cfg, restores))


def check_sources(cfg, config_path, config_sha, manager_sha):
    original_check_sources(cfg, config_path, config_sha, manager_sha)
    validate_semantics(cfg)
    handoff_gate(cfg, cfg['handoff']['transition_ref'], require_binding=True)


def entry_capture(entry, cfg):
    capture_lifetime(cfg)
    handoff_gate(cfg, cfg['handoff']['transition_ref'], require_binding=True)
    captured = next(row for row in cfg['fresh_capture']['locks_leases'] if row['gpu'] == entry['gpu'])
    if stat_value(Path('/tmp/ms12k_gpu_%d.lock' % entry['gpu']).lstat()) != captured['lock']:
        raise RuntimeError('Captured lock inode/nanoseconds changed before native admission')
    return captured


def configure_restore(entry, cfg):
    captured = entry_capture(entry, cfg)
    if stat_value(entry['old']) != captured['lease']:
        raise RuntimeError('Captured lease changed before native restore configuration')
    original_configure_restore(entry, cfg)
    entry['ahead_handoff_config'] = cfg


def acquire_lease(entry):
    cfg = entry['ahead_handoff_config']
    captured = entry_capture(entry, cfg)
    if stat_value(lease_snapshot(entry['reserve'])) != captured['lease']:
        raise RuntimeError('Captured lease changed before native lease acquisition')
    original_acquire_lease(entry)
    entry['ahead_owned_lease_stat'] = stat_value(os.fstat(entry['leasefd']))


def release_holder(entry):
    cfg = entry['ahead_handoff_config']
    entry_capture(entry, cfg)
    # Native heartbeat may advance mtime after acquisition. Stable ownership must
    # remain exact; the current path/FD must agree and carry a recent heartbeat.
    expected = entry['ahead_owned_lease_stat']
    current = stat_value(os.fstat(entry['leasefd']))
    stable = ('device', 'inode', 'uid', 'mode', 'size')
    if (stat_value(entry['reserve'].lstat()) != current
            or any(current[key] != expected[key] for key in stable)
            or not 0 <= time.time_ns() - current['mtime_ns'] <= 60 * 10**9):
        raise RuntimeError('Owned lease identity or current heartbeat changed before native holder release')
    original_release_holder(entry)


def build(args):
    cfg = plan_config(0)
    check_plan(cfg)
    original_validate(cfg)
    out = Path(args.out)
    if (not out.is_absolute() or str(out.resolve()) != str(out) or out.parent != RT
            or not out.name.startswith(PHASE + '_') or out.suffix != '.json'
            or out.exists() or out.is_symlink() or Path(cfg['queue']).exists()):
        raise RuntimeError('Unique absolute runtime launch and unused N8s0 queue required')
    transition_ref = {'path': args.transition, 'sha256': args.transition_sha}
    restores, references = handoff_gate(cfg, transition_ref, require_parent=True)
    capture = fresh_capture(cfg, restores)
    cfg.update(holders=capture['holders'], fresh_capture=capture,
               handoff={'transition_ref': transition_ref, 'receipt_refs': references})
    for reference in references:
        existing = cfg['source_sha256'].get(reference['path'])
        if existing is not None and existing != reference['sha256']:
            raise RuntimeError('Handoff attempted to re-sign an existing source')
        cfg['source_sha256'][reference['path']] = reference['sha256']
    validate(cfg)
    if Path(cfg['queue']).exists():
        raise RuntimeError('Fixed N8s0 queue appeared; no launch written')
    with out.open('x') as stream:
        json.dump(cfg, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return {'status': 'fresh_launch_saved_once', 'config': ref(out),
            'resource_manager_ref': cfg['resource_manager_ref'], 'preparation_ref': cfg['preparation_ref'],
            'queue': cfg['queue'], 'base_manager_sha256': BASE_SHA,
            'signals_sent': False, 'gpu_started': False,
            'manager_argv': ['/opt/venv_eval/bin/python', '-B', '-u', str(SELF),
                '--self-sha', c.base.sha(SELF), '--ahead-sha', AHEAD_SHA,
                '--config', str(out), '--config-sha', c.base.sha(out), '--manager-sha', BASE_SHA]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check-plan', action='store_true')
    actions.add_argument('--build', action='store_true')
    actions.add_argument('--config')
    parser.add_argument('--self-sha', required=True)
    parser.add_argument('--ahead-sha', required=True)
    parser.add_argument('--config-sha')
    parser.add_argument('--manager-sha')
    parser.add_argument('--transition')
    parser.add_argument('--transition-sha')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if c.base.sha(SELF) != args.self_sha or args.ahead_sha != AHEAD_SHA or c.base.sha(AHEAD) != AHEAD_SHA:
        raise RuntimeError('Caller frozen N8 lifecycle/prepare-ahead SHA differs')
    if args.check_plan:
        if any((args.out, args.transition, args.transition_sha, args.config_sha, args.manager_sha)):
            parser.error('--check-plan cannot construct or execute a launch')
        cfg = plan_config(0)
        check_plan(cfg)
        value = {'status': 'cpu_plan_sources_model_verified', 'node': NODE, 'gpus': GPUS,
                 'targetcount': cfg['targetcount'], 'original_consumer_config_ref': cfg['original_consumer_config_ref'],
                 'resource_manager_ref': cfg['resource_manager_ref'], 'preparation_ref': cfg['preparation_ref'],
                 'queue': cfg['queue'], 'launch_written': False, 'signals_sent': False,
                 'gpu_started': False, 'gpu_lock_or_lease_accessed': False}
    elif args.build:
        if not all((args.out, args.transition, args.transition_sha)) or any((args.config_sha, args.manager_sha)):
            parser.error('--build requires --out --transition --transition-sha only')
        value = build(args)
    else:
        if (not all((args.config_sha, args.manager_sha)) or args.manager_sha != BASE_SHA
                or any((args.out, args.transition, args.transition_sha))):
            parser.error('--config requires native --config-sha and frozen --manager-sha only')
        previous = sys.argv
        try:
            sys.argv = [str(SELF), '--config', args.config, '--config-sha', args.config_sha,
                        '--manager-sha', BASE_SHA]
            return c.base.main()
        finally:
            sys.argv = previous
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


c.base.validate_pair = validate
c.base.check_sources = check_sources
c.base.configure_local_restore = configure_restore
c.base.acquire_lease = acquire_lease
c.base.release_holder = release_holder
if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'n8s0_prepare_ahead_refused', 'error': str(error),
                          'signals_sent_by_wrapper': False}), file=sys.stderr, flush=True)
        sys.exit(2)
