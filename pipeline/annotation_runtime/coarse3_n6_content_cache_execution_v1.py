"""Explicit finite-review IO execution: lexical cache keys, unchanged source validators."""
import hashlib
import importlib.util
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
STAT_KEYS = ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
PINNED = {
    'prepare_coarse3_n6_v1.py': '38cbc9cdfd09d8919de1521fced67e2c6a1df674d99da9c844bcd329e4be350c',
    'coarse3_n6_resource_manager_v1.py': 'a7ac544d1faa5587049b3378df502540e72f3f20b2595eeefd99f2f5520418e5',
    'coarse3_n6_driver_v1.py': 'd2a5566a39547a30328621eede9b1d16a14972bb49887c9c2c186d32c81c639e',
    'prepare_curated_coarse_empty_n5_v1.py': '1190df9feed04400391c9d4b7b0df11e9bb40b2ef642754d3dcaaffcdb733319',
    'resource_manager.py': '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6',
}
_cache = {}


def content_sha(path):
    path = Path(path)
    before = path.stat()
    # Key identity is NOT a path-confinement or symlink validator. Reads/stats
    # still follow the supplied path, and all original source validators remain.
    key = (os.path.abspath(os.fspath(path)), *(getattr(before, k) for k in STAT_KEYS))
    if key in _cache:
        return _cache[key]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
        finished = os.fstat(stream.fileno())
    after = path.stat()
    if any(getattr(before, k) != getattr(s, k) for s in (opened, finished, after) for k in STAT_KEYS):
        raise RuntimeError('File changed during full content SHA: ' + str(path))
    _cache[key] = digest.hexdigest()
    return _cache[key]


def module(name, path):
    raw = Path(path).read_bytes()
    expected = PINNED[Path(path).name]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Original execution source SHA changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), value.__dict__)
    return value


def configured():
    for name, digest in PINNED.items():
        if content_sha(RT / name) != digest:
            raise RuntimeError('Frozen original source differs: ' + name)
    import prepare_coarse3_n6_v1 as c
    c.h.sha = content_sha
    c.sha = content_sha
    c.MANAGER = c.DRIVER = SELF
    original_plan = c.source_plan

    def source_plan(m=None, *, probe=True):
        m = m or c.worker_module()
        library = m.libraries()
        # The library loads its own copy of the N5 helper; bind its handle
        # explicitly, without importing any external or old CSV cache seeds.
        library.c.sha = content_sha
        value = original_plan(m, probe=probe)
        value['source_sha256'].update({str(RT / n): h for n, h in PINNED.items()})
        value['source_sha256'][str(SELF)] = content_sha(SELF)
        test = RT / 'test_coarse3_n6_content_cache_execution_v1.py'
        value['source_sha256'][str(test)] = content_sha(test)
        value['content_SHA_cache_execution_ref'] = {'path': str(SELF), 'sha256': content_sha(SELF)}
        value['original_source_validators_unchanged'] = True
        value['content_cache_key_is_not_source_authority'] = True
        return value

    c.source_plan = source_plan
    return c


def fresh_current_validate(manager, cfg):
    c, base = manager.c, manager.base
    plan = c.source_plan()
    c.verify_config(cfg, plan)
    c.check_node()
    c.verify_local_model(cfg)
    prior = c.prior_safe(cfg['transition_ref'], base)
    historical = c.checked(cfg['holder_capture'])
    c.need(historical.get('signals_sent') is False and historical['holders'] == cfg['holders']
           and historical['lease_snapshot_ns'] == cfg['lease_snapshot_ns']
           and historical['budget']['headroom'] >= 160 * 2**30,
           'Historical holder/config/lease capture differs')
    c.require_unreviewed(plan)
    manager.active_config = cfg
    fresh = c.fresh_holder(base, prior)
    c.need(fresh['lease_snapshot_ns'] == cfg['lease_snapshot_ns'], 'Original exact lease changed')
    base.check_holder(cfg['holders'][0], fresh['holders'][0], 3, fresh['holders'][0]['uuid'], base.apps())
    c.need(0 <= time.time_ns() - fresh['captured_unix_ns'] <= 60 * 10**9, 'Current resource capture already stale')
    path = RT / ('coarse3_current_resource_admission_' + str(time.time_ns()) + '.json')
    c.save_once(path, {'config_ref': c.ref(cfg['launch_path']), 'historical_capture_ref': cfg['holder_capture'],
                      'current_verified_capture': fresh, 'source_checks_preceded_current_capture': True})


def current_before_release(manager, entry):
    c, base = manager.c, manager.base
    cfg = manager.active_config
    c.need(cfg is not None and cfg['gpus'] == [3] and entry['gpu'] == 3, 'Current single-GPU scope changed')
    parent = base.identity(os.getpid())
    c.need(parent['argv'] == ['/opt/venv_eval/bin/python', '-B', '-u', str(SELF), '--config',
              cfg['launch_path'], '--config-sha', content_sha(cfg['launch_path']), '--manager-sha', c.BASE_SHA],
           'Exact current resource manager identity changed before release')
    c.need(c.proc_lock_owned(parent['pid'], '/tmp/ms12k_gpu_3.lock'), 'Actual parent FLOCK missing')
    s, fs = entry['reserve'].lstat(), os.fstat(entry['leasefd'])
    c.need((s.st_dev, s.st_ino) == entry['lease_inode'] == (fs.st_dev, fs.st_ino)
           and s.st_uid == os.getuid() and s.st_size == 0
           and c.fd_inode_owned(parent['pid'], s.st_dev, s.st_ino)
           and 0 <= time.time_ns() - s.st_mtime_ns <= 60 * 10**9,
           'Current owned lease FD/inode/heartbeat changed before release')
    c.need(base.headroom()['headroom'] >= base.required_budget(cfg), 'Current headroom below admission')
    base.check_holder(entry['expected'], base.identity(entry['expected']['pid']), 3, base.uuid_for(3), base.apps())
    c.save_once(Path(cfg['queue']) / 'current_before_holder_release.json',
                {'manager_identity': parent, 'config_ref': c.ref(cfg['launch_path']),
                 'current_lease_stat': c.lease_value(s), 'verified_unix_ns': time.time_ns(),
                 'current_FD_kernel_FLOCK_UUID_and_holder_checked': True})


def revision(c, args):
    original = c.checked({'path': str(Path(args.from_launch).resolve()), 'sha256': args.from_sha})
    old_plan = c.checked(original['plan'])
    cfg, plan = copy.deepcopy(original), copy.deepcopy(old_plan)
    launch = Path(args.launch).resolve()
    c.need(launch.parent == RT and launch.name.startswith('coarse3_n6_launch_')
           and not launch.exists(), 'New revision launch required')
    c.need(original['gpus'] == [3] and original['targetcount'] == 3
           and original['producer']['path'] == str(SELF), 'Only prior exact independent-three execution allowed')
    for name, digest in PINNED.items():
        c.need(content_sha(RT / name) == digest, 'Original execution source changed')
        plan['source_sha256'][str(RT / name)] = digest
    plan['source_sha256'][str(SELF)] = content_sha(SELF)
    test = RT / 'test_coarse3_n6_content_cache_execution_v1.py'
    plan['source_sha256'][str(test)] = content_sha(test)
    plan['content_SHA_cache_execution_ref'] = {'path': str(SELF), 'sha256': content_sha(SELF)}
    suffix = str(time.time_ns())
    plan_path = RT / ('coarse3_current_resource_plan_' + suffix + '.json')
    c.save_once(plan_path, plan)
    # This is an explicit source-code revision of an immutable prior plan, not
    # a re-signing of any original source/media/score reference. The manager
    # independently reconstructs the complete current source plan before release.
    cfg.update(launch_path=str(launch), argv=['/opt/venv_vllm/bin/python', '-B', '-u', str(SELF),
                  '--launch', str(launch)], producer=c.ref(SELF), resource_manager=c.ref(SELF),
               env=c.driver_env(launch), plan=c.ref(plan_path),
               queue=str(c.BAR / 'queues' / ('coarse3_n6_independent_current_' + suffix)))
    cfg['source_sha256'].update(plan['source_sha256'])
    cfg['source_sha256'].update({str(plan_path): c.sha(plan_path)})
    base = c.base_module()
    restored = c.checked(c.checked(cfg['transition_ref'])['restored_ref'])
    current = c.fresh_holder(base, {'restored': restored})
    capture_path = RT / ('coarse3_current_resource_capture_' + suffix + '.json')
    c.save_once(capture_path, current)
    cfg.update(holders=current['holders'], lease_snapshot_ns=current['lease_snapshot_ns'], holder_capture=c.ref(capture_path))
    cfg['source_sha256'].update({str(capture_path): c.sha(capture_path)})
    c.save_once(launch, cfg)
    return {'launch_ref': c.ref(launch), 'wrapper_ref': c.ref(SELF), 'signals_sent': False, 'GPU_model_calls': 0,
            'complete_current_sources_and_old_scores_revalidated_by_manager_before_release': True,
            'command': ['/opt/venv_eval/bin/python', '-B', '-u', str(SELF), '--config', str(launch),
                        '--config-sha', c.sha(launch), '--manager-sha', c.BASE_SHA]}


def main():
    c = configured()
    if '--revision-from' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--revision-from', dest='from_launch', required=True)
        parser.add_argument('--from-sha', required=True)
        parser.add_argument('--launch', required=True)
        args = parser.parse_args()
        print(json.dumps(revision(c, args)), flush=True)
    elif '--config' in sys.argv:
        manager = module('coarse3_content_cache_native_manager', RT / 'coarse3_n6_resource_manager_v1.py')
        original_release = manager.base.release_holder
        manager.base.validate_pair = lambda cfg: fresh_current_validate(manager, cfg)
        def release(entry):
            current_before_release(manager, entry)
            original_release(entry)
        manager.base.release_holder = release
        manager.base.main()
    elif '--launch' in sys.argv and '--build' not in sys.argv:
        driver = module('coarse3_content_cache_native_driver', RT / 'coarse3_n6_driver_v1.py')
        driver.main()
    else:
        c.main()


if __name__ == '__main__':
    main()
