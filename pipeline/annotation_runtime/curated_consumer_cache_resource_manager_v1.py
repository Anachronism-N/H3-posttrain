"""Native N1 consumer lifecycle; only execution entrypoint and IO protocol differ."""
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

RT = Path(__file__).resolve().parent
FROZEN = {
    'resource_manager.py': '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6',
    'curated_resource_manager.py': '767fedabe779081b57b09d060a908194f6892dedf58929c868ef67deeefe8a61',
    'curated_consumer_resource_manager.py': '48c942baddf67396b28403b859eb459f6529bef52d6a705ddfbb02592a6f684f',
    'curated_consumer_image_cache_v1.py': '69eca5c2feadc4e00cbf4ef3ad51a4dbb0027f17c70f1b46990d3b0b8f7490d0',
    'curated_request_image_cache_v1.py': '8e40c5f784ebd548a2ab7d27450cb75a8d4e145576886ed690330925b51c79e3'}
for name, expected in FROZEN.items():
    if hashlib.sha256((RT / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen consumer dependency changed: ' + name)
spec = importlib.util.spec_from_file_location('native_cached_consumer_lifecycle', RT / 'curated_consumer_resource_manager.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
c = native.c
original_validate = c.base.validate_pair
OLD = {
    2: ('curated_consumer_s2_v1_launch.json', '5185fe7ec43efc33f4a623c08eba741a174a8216840ff7c42ebb6eea2be72902'),
    3: ('curated_consumer_s3_v1_launch.json', '727f58edd49c48481f806f154b1db0b0ad8435259b2dcf8a42c0155ed933c251')}


BASE_SHA = FROZEN['resource_manager.py']
PREPARE = RT / 'prepare_curated_consumer_cache_v1.py'
MANAGER = Path(__file__).resolve()
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
        raise RuntimeError('Only N1 native consumer shards2/3 admitted')
    name, digest = OLD[shard]
    reference = {'path': str(RT / name), 'sha256': digest}
    return load_ref(reference, RT / name), reference


def plan_config(shard):
    old, reference = original_config(shard)
    cfg = copy.deepcopy(old)
    cfg.update(original_consumer_config_ref=reference, resource_manager_ref=ref(MANAGER),
               preparation_ref=ref(PREPARE), handoff=None, fresh_capture=None)
    cfg['argv'][3] = str(RT / 'curated_consumer_image_cache_v1.py')
    cfg['phase'] = 'curated_consumer_cache_s%d_v1' % shard
    cfg['queue'] = str(RT.parent / 'queues' / cfg['phase'])
    cfg['producer'] = {'path': cfg['argv'][3], 'sha256': FROZEN['curated_consumer_image_cache_v1.py']}
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
    argv[3] = str(RT / 'curated_consumer_image_cache_v1.py')
    if cfg['argv'] != argv or cfg['node'] != 'N1' or cfg['mode'] != 'consumer':
        raise RuntimeError('Only native N1 consumer execution entrypoint may change')
    if (cfg['phase'] != 'curated_consumer_cache_s%d_v1' % shard
            or cfg['queue'] != str(RT.parent / 'queues' / cfg['phase'])):
        raise RuntimeError('New isolated consumer resource queue required')
    if cfg['producer'] != {'path': argv[3], 'sha256': FROZEN['curated_consumer_image_cache_v1.py']}:
        raise RuntimeError('Verified cache execution wrapper required')
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
    if (transition.get('schema') != 'curated_consumer_cache_transition_v1'
            or transition.get('node') != 'N1' or transition.get('shard') != old['source_selection']['shard']
            or transition.get('reason') != 'intentional_reschedule'
            or transition.get('worker_exit_claimed') is not False
            or transition.get('original_consumer_config_ref') != oldref):
        raise RuntimeError('Exact intentional-reschedule main transition required')
    admission_ref = transition['admission_ref']
    safe_ref = transition['resources_safe_ref']
    live_ref = transition['live_ref']
    admission = load_ref(admission_ref, queue / 'admission.json')
    safe = load_ref(safe_ref, queue / 'resources_safe.json')
    live = load_ref(live_ref, queue / 'live.json')
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
    if (identity['argv'] != expected_argv or transition.get('old_manager_identity') != identity
            or not same_identity(live['manager_identity'], identity)
            or (current_manager is None and require_parent)
            or (current_manager is not None and not same_identity(current_manager, identity))):
        raise RuntimeError('Exact original manager must remain in restored-holder parent wait')
    if (safe.get('phase') != old['phase'] or not all(safe.get(key) is True for key in
            ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled'))):
        raise RuntimeError('Original exact manager resources are not safe/restored')
    if live.get('phase') != old['phase'] or live.get('argv') != old['argv']:
        raise RuntimeError('Original live worker binding changed')
    child = live['child_identity']
    if child is None or child['argv'] != old['argv'] or child['ppid'] != identity['pid']:
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
        if outcome.get('worker_started') is not True or outcome.get('unknown_jobs_not_signaled') is not True:
            raise RuntimeError('Actual old worker stop receipt required')
    elif (outcome.get('targetcount') != old['targetcount'] or outcome.get('producer') != old['producer']
            or outcome.get('plan') != old['plan'] or outcome.get('source_selection') != old['source_selection']):
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
                or restored_identity['ppid'] != identity['pid']
                or restored_identity['CVD'] != ['CUDA_VISIBLE_DEVICES=' + str(gpu)]
                or not admission_ns <= Path(reference['path']).stat().st_mtime_ns <= safe_ns):
            raise RuntimeError('Exact restored-holder provenance changed: GPU%d' % gpu)
        restores.append(restored)
    checked_ns = transition.get('checked_unix_ns')
    if (type(checked_ns) is not int or not admission_ns <= outcome_ns <= safe_ns <= checked_ns <= time.time_ns()):
        raise RuntimeError('Main transition/safe/outcome chronology mismatch')
    references = [oldref, transition_ref, admission_ref, live_ref, outcome_ref, safe_ref] + restored_refs
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


def fresh_capture(cfg, restores):
    if c.HOSTS['N1'] not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Fresh cache construction is N1-only')
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
                              'lease': stat_value(c.base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu)))})
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
                    or stat_value(c.base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))) != captured['lease']):
                raise RuntimeError('Fresh holder/lock/lease changed during capture')
        return {'node': 'N1', 'gpus': cfg['gpus'], 'holders': owners, 'locks_leases': snapshots,
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
            or not 0 <= time.time_ns() - captured['captured_unix_ns'] <= 1200 * 10**9
            or captured.get('node') != 'N1' or captured.get('gpus') != cfg['gpus']
            or captured.get('signals_sent') is not False or captured.get('gpu_started') is not False
            or captured.get('lock_lease_contents_modified') is not False
            or captured.get('headroom', {}).get('headroom', 0) < c.base.required_budget(cfg)
            or captured['boot_id'] != fresh['boot_id'] or captured['locks_leases'] != fresh['locks_leases']
            or cfg['holders'] != captured['holders'] or len(captured['holders']) != len(fresh['holders'])
            or any(not same_identity(now, before) or now['ppid'] != before['ppid']
                   or now['gpu'] != before['gpu'] or now['uuid'] != before['uuid']
                   for now, before in zip(fresh['holders'], captured['holders']))):
        raise RuntimeError('Bound fresh holder/lock/lease/headroom capture changed or expired')
    for row in captured['locks_leases']:
        for name in ('lock', 'lease'):
            if row[name] is not None and any(type(v) is not int for v in row[name].values()):
                raise RuntimeError('Lock/lease stat identity must use exact integer nanoseconds')


def validate(cfg):
    check_plan(cfg)
    original_validate(cfg)
    restores, _ = handoff_gate(cfg, cfg['handoff']['transition_ref'], require_binding=True, require_parent=True)
    compare_capture(cfg, cfg['fresh_capture'], fresh_capture(cfg, restores))


def check_sources(cfg, config_path, config_sha, manager_sha):
    original_check_sources(cfg, config_path, config_sha, manager_sha)
    validate_semantics(cfg)
    handoff_gate(cfg, cfg['handoff']['transition_ref'], require_binding=True)


def configure_restore(entry, cfg):
    captured = next(row for row in cfg['fresh_capture']['locks_leases'] if row['gpu'] == entry['gpu'])
    if (stat_value(Path('/tmp/ms12k_gpu_%d.lock' % entry['gpu']).lstat()) != captured['lock']
            or stat_value(entry['old']) != captured['lease']):
        raise RuntimeError('Captured lock/lease changed before native admission')
    original_configure_restore(entry, cfg)
    entry['cache_handoff_config'] = cfg


def acquire_lease(entry):
    cfg = entry['cache_handoff_config']
    captured = next(row for row in cfg['fresh_capture']['locks_leases'] if row['gpu'] == entry['gpu'])
    if (stat_value(Path('/tmp/ms12k_gpu_%d.lock' % entry['gpu']).lstat()) != captured['lock']
            or stat_value(c.base.lease_snapshot(entry['reserve'])) != captured['lease']):
        raise RuntimeError('Captured lock/lease changed before native lease acquisition')
    original_acquire_lease(entry)


def release_holder(entry):
    cfg = entry['cache_handoff_config']
    if not 0 <= time.time_ns() - cfg['fresh_capture']['captured_unix_ns'] <= 1200 * 10**9:
        raise RuntimeError('Cache handoff capture expired before holder release')
    handoff_gate(cfg, cfg['handoff']['transition_ref'], require_binding=True)
    original_release_holder(entry)


c.base.validate_pair = validate
c.base.check_sources = check_sources
c.base.configure_local_restore = configure_restore
c.base.acquire_lease = acquire_lease
c.base.release_holder = release_holder
if __name__ == '__main__':
    c.base.main()
