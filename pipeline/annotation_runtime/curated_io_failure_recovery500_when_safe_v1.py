"""CPU-only wait for exact predecessor completion and fresh safe GPU admission.

Never signals holders, writes GPU leases, starts a duplicate worker, or edits the
source-prepared config. After readonly checks it execs the verified manager.
"""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import time
import uuid

RT = Path(__file__).resolve().parent
SOURCE = RT / 'curated_io_failure_recovery500_n5g0_source_prepared_v1.json'
SOURCE_SHA = '022d63eee2dc5e3d72cdf2c35921e63b6c090b354cfaa9bd3b7fa34932eae571'
BASE_MANAGER_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
OUT = RT.parent / 'curated222939_io_failure_recovery_v1'
ADMISSION_POINTER = RT / 'curated_io_failure_recovery500_producer_when_safe_admission_v1.json'
SPEC = importlib.util.spec_from_file_location('curated_io_recovery_original_manager_v1', RT / 'curated_io_resource_manager_v1.py')
io = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(io)
c, base = io.c, io.c.base


class WaitSafe(RuntimeError):
    pass


def load(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': base.sha(path)}


def save_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def source_config():
    if base.sha(SOURCE) != SOURCE_SHA or base.sha(RT / 'resource_manager.py') != BASE_MANAGER_SHA:
        raise RuntimeError('Frozen source-prepared/base manager SHA changed')
    cfg = load(SOURCE)
    if (cfg['mode'] != 'producer' or cfg['node'] != 'N5' or cfg['gpus'] != [0]
            or cfg['holders'] != [] or cfg['holder_capture'] is not None
            or cfg['release_authorized'] is not False or Path(cfg['out']) != OUT
            or cfg['source_range'] != {'start': 500, 'end': 503}
            or cfg['source_selection'] != {'shard': 0, 'nshard': 1, 'limit': 3}
            or cfg['targetcount'] != 3):
        raise RuntimeError('Only exact three-source recovery placeholder admitted')
    for path, sha in cfg['source_sha256'].items():
        if base.sha(path) != sha:
            raise RuntimeError('Frozen recovery source/provenance changed: ' + path)
    ids, total = io.selected(cfg)
    if ids != [row['id'] for row in load(cfg['selected_sources']['path'])] or total != 222939:
        raise RuntimeError('Three recovery source selection differs')
    return cfg


def predecessor_gate(cfg):
    reference = cfg['resource_serialization']['predecessor_config']
    if base.sha(reference['path']) != reference['sha256']:
        raise RuntimeError('Collaborative predecessor config SHA changed')
    previous = load(reference['path'])
    queue = Path(previous['queue'])
    paths = {name: queue / name for name in ('admission.json', 'worker_exit.json', 'resources_safe.json', 'holder_restored_gpu0.json')}
    if not all(path.exists() for path in paths.values()):
        raise WaitSafe('Waiting collaborative GPU0 worker_exit + resources_safe + holder-restored receipts')
    admission, finished, safe, restored = [load(paths[name]) for name in paths]
    if (admission['config_sha256'] != reference['sha256'] or admission['node'] != 'N5'
            or admission['gpus'] != [0] or admission['out'] != previous['out']
            or admission['phase'] != previous['phase']):
        raise RuntimeError('Predecessor admission binding changed')
    if (finished.get('exit_code') != 0 or finished.get('phase') != previous['phase']
            or finished.get('targetcount') != 500 or finished.get('input_count') != 222939
            or finished.get('producer') != previous['producer'] or finished.get('plan') != previous['plan']
            or finished.get('source_selection') != previous['source_selection']
            or finished.get('source_unchanged') is not True):
        raise WaitSafe('Predecessor producer lacks exact successful 500-source worker-exit receipt')
    summaries = finished.get('summaries', {})
    if not summaries or not any(value.get('summary', {}).get('processed_source_count') == 500
            and value.get('summary', {}).get('selected_source_count') == 500
            and value.get('selection', {}).get('selected_ids') == [row['id'] for row in load(previous['selected_sources']['path'])]
            for value in summaries.values()):
        raise WaitSafe('Predecessor has no exact completed producer summary')
    if (safe.get('phase') != previous['phase'] or not all(safe.get(key) is True
            for key in ('all_holders_safe', 'locks_released', 'leases_restored', 'unknown_jobs_not_signaled'))
            or paths['resources_safe.json'].stat().st_mtime_ns < paths['worker_exit.json'].stat().st_mtime_ns):
        raise WaitSafe('Predecessor GPU0 holders/leases/locks not explicitly safe')
    if restored.get('gpu') != 0 or restored.get('managed_parent_wait') is not True:
        raise WaitSafe('Predecessor exact GPU0 holder restoration not acknowledged')
    return previous, restored, [ref(path) for path in paths.values()]


def lease_value(st):
    return None if st is None else [st.st_dev, st.st_ino, st.st_uid, st.st_mode, st.st_size, st.st_mtime_ns]


def fresh_capture(cfg, restored=None):
    if c.HOSTS[cfg['node']] not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Recovery waiter executing on wrong physical node')
    owners, leases, descriptors, locks = [], [], [], []
    try:
        for gpu in cfg['gpus']:
            lock = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
            if lock.exists():
                fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
                locks.append(fd)
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size:
                    raise WaitSafe('Unknown GPU lock shape/owner/content')
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise WaitSafe('GPU%d lock still held; leaving it untouched' % gpu) from error
            lease = base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))
            leases.append(lease_value(lease))
            gpu_uuid = base.uuid_for(gpu)
            active = [app for app in base.apps() if app['uuid'] == gpu_uuid]
            if len(active) != 1:
                raise WaitSafe('GPU%d does not have exactly one project holder' % gpu)
            owner = base.identity(active[0]['pid'])
            owner.update(gpu=gpu, uuid=gpu_uuid)
            base.check_holder(owner, base.identity(owner['pid']), gpu, gpu_uuid, base.apps())
            if restored is not None:
                expected = dict(restored['holder_identity'], gpu=gpu, uuid=restored['uuid'])
                base.check_holder(expected, base.identity(owner['pid']), gpu, gpu_uuid, base.apps())
            fd = os.pidfd_open(owner['pid'])
            descriptors.append(fd)
            if select.select([fd], [], [], 0)[0]:
                raise WaitSafe('Exact holder already exited')
            owners.append(owner)
        budget = base.headroom()
        if budget['headroom'] < base.required_budget(cfg):
            raise WaitSafe('Fresh node headroom below native admission budget')
        for gpu, owner, expected in zip(cfg['gpus'], owners, leases):
            base.check_holder(owner, base.identity(owner['pid']), gpu, base.uuid_for(gpu), base.apps())
            if lease_value(base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))) != expected:
                raise WaitSafe('GPU%d lease changed during readonly fresh check' % gpu)
        return {'node': cfg['node'], 'mode': cfg['mode'], 'gpus': cfg['gpus'], 'holders': owners,
                'leases': leases, 'captured_unix_ns': time.time_ns(), 'headroom': budget,
                'locks_readonly_checked': True, 'signals_sent': False, 'gpu_started': False}
    except (OSError, ValueError, KeyError) as error:
        raise WaitSafe('Fresh GPU check unavailable; unknown state untouched: ' + str(error)) from error
    except RuntimeError as error:
        if isinstance(error, WaitSafe):
            raise
        raise WaitSafe('Fresh GPU check not safe: ' + str(error)) from error
    finally:
        for fd in descriptors:
            os.close(fd)
        for fd in locks:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def authorize_producer(cfg, capture, gate_refs):
    if Path(cfg['queue']).exists() or OUT.exists() or ADMISSION_POINTER.exists():
        raise RuntimeError('Recovery queue/output/admission already exists; refusing duplicate or ambiguous launch')
    token = uuid.uuid4().hex
    capture_path = RT / ('curated_io_failure_recovery500_n5g0_fresh_' + token + '.json')
    config_path = RT / ('curated_io_failure_recovery500_n5g0_launch_' + token + '.json')
    save_new(capture_path, capture)
    active = copy.deepcopy(cfg)
    active.update(holders=capture['holders'], holder_capture=ref(capture_path), release_authorized=True,
                  preparation_status='fresh_readonly_admitted_after_collaborative_resources_safe',
                  resource_assignment_status='fresh_exact_gpu0_authorized', source_prepared_reference={'path': str(SOURCE), 'sha256': SOURCE_SHA})
    active['source_sha256'][str(SOURCE)] = SOURCE_SHA
    for reference in gate_refs + [ref(__file__), ref(capture_path)]:
        active['source_sha256'][reference['path']] = reference['sha256']
    active['when_safe_predecessor_receipts'] = gate_refs
    active['resource_serialization']['current_placeholder_launchable'] = True
    active['resource_serialization']['required_before_admission'][0] = 'Producer worker completed and resources_safe confirms holder/lease/lock restoration; restored-holder parent may still wait'
    save_new(config_path, active)
    c.validate(active)
    base.check_sources(active, config_path, base.sha(config_path), BASE_MANAGER_SHA)
    c.check_model(active)
    fresh = fresh_capture(active)
    if (any(any(current[key] != expected[key] for key in base.IDENTITY_KEYS + ('gpu', 'uuid'))
            for current, expected in zip(fresh['holders'], capture['holders']))
            or fresh['leases'] != capture['leases']):
        raise WaitSafe('Exact fresh holder/lease changed before manager exec; no signals sent')
    value = {'status': 'fresh_readonly_admitted', 'source_prepared_ref': {'path': str(SOURCE), 'sha256': SOURCE_SHA},
             'config': ref(config_path), 'capture': ref(capture_path), 'manager': ref(active['resource_manager_reference']['path']),
             'base_manager_sha256': BASE_MANAGER_SHA, 'out': str(OUT), 'queue': active['queue'],
             'predecessor_receipts': gate_refs, 'checked_unix_ns': time.time_ns(),
             'signals_sent_by_waiter': False, 'gpu_started_by_waiter': False}
    save_new(ADMISSION_POINTER, value)
    print(json.dumps(value), flush=True)
    argv = ['/opt/venv_eval/bin/python', '-B', '-u', active['resource_manager_reference']['path'],
            '--config', str(config_path), '--config-sha', base.sha(config_path), '--manager-sha', BASE_MANAGER_SHA]
    os.execv(argv[0], argv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-producer', action='store_true')
    parser.add_argument('--readonly-now', action='store_true')
    parser.add_argument('--self-sha', required=True)
    args = parser.parse_args()
    if base.sha(__file__) != args.self_sha:
        raise RuntimeError('Frozen waiter SHA changed')
    cfg = source_config()
    if not args.wait_producer and not args.readonly_now:
        parser.error('Explicit --wait-producer or --readonly-now required')
    lock = RT / 'curated_io_failure_recovery500_producer_cpu_waiter_v1.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_reason = None
        while True:
            try:
                previous, restored, gate_refs = predecessor_gate(cfg)
                capture = fresh_capture(cfg, restored)
            except WaitSafe as error:
                reason = str(error)
                if reason != last_reason:
                    print(json.dumps({'status': 'waiting_resources_safe', 'reason': reason,
                          'checked_unix_ns': time.time_ns(), 'signals_sent': False, 'gpu_started': False}), flush=True)
                    last_reason = reason
                if args.readonly_now:
                    return
                time.sleep(30)
                continue
            if args.readonly_now:
                print(json.dumps({'status': 'readonly_ready', 'capture': capture, 'signals_sent': False, 'gpu_started': False}), flush=True)
                return
            try:
                authorize_producer(cfg, capture, gate_refs)
            except WaitSafe as error:
                print(json.dumps({'status': 'waiting_fresh_admission', 'reason': str(error),
                      'signals_sent': False, 'gpu_started': False}), flush=True)
                time.sleep(30)


if __name__ == '__main__':
    main()
