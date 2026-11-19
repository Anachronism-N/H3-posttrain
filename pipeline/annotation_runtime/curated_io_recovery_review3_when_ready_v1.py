"""CPU waiter: all 3 recovery terminals + fresh N6G3, then exact manager exec."""
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
import time
import uuid

RT = Path(__file__).resolve().parent
SOURCE = RT / 'curated_io_recovery_review3_n6g3_source_prepared_v1.json'
MANAGER = RT / 'curated_io_recovery_review3_resource_manager_v1.py'
POINTER = RT / 'curated_io_recovery_review3_n6g3_when_ready_admission_v1.json'
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'


def module(path):
    s = importlib.util.spec_from_file_location('recovery3_wait_resource', path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def lease_value(s):
    return None if s is None else [s.st_dev, s.st_ino, s.st_uid, s.st_mode, s.st_size, s.st_mtime_ns]


def fresh_capture(manager):
    base, worker = manager.base, manager.worker
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('CPU review3 waiter must execute on N6')
    lockfd, pidfd = None, None
    try:
        path = Path('/tmp/ms12k_gpu_3.lock')
        if path.exists() or path.is_symlink():
            lockfd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            s = os.fstat(lockfd)
            if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_size:
                raise worker.NotReady('Unknown GPU3 lock shape/owner/content; untouched')
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = base.lease_snapshot(Path('/tmp/od_reserve_3'))
        gpu_uuid = base.uuid_for(3)
        running = base.apps()
        active = [x for x in running if x['uuid'] == gpu_uuid]
        if len(active) != 1:
            raise worker.NotReady('GPU3 does not contain exactly one project holder; unknown active untouched')
        owner = dict(base.identity(active[0]['pid']), gpu=3, uuid=gpu_uuid)
        base.check_holder(owner, base.identity(owner['pid']), 3, gpu_uuid, running)
        pidfd = os.pidfd_open(owner['pid'])
        if select.select([pidfd], [], [], 0)[0]:
            raise worker.NotReady('GPU3 holder already exited')
        budget = base.headroom()
        if budget['headroom'] < base.required_budget({}):
            raise worker.NotReady('N6 headroom below native review admission budget')
        base.check_holder(owner, base.identity(owner['pid']), 3, base.uuid_for(3), base.apps())
        if lease_value(base.lease_snapshot(Path('/tmp/od_reserve_3'))) != lease_value(lease):
            raise worker.NotReady('GPU3 lease changed during read-only capture')
        return dict(node='N6', gpus=[3], holders=[owner], lease=lease_value(lease), headroom=budget,
                    captured_unix_ns=time.time_ns(), signals_sent=False, gpu_started=False, leases_modified=False)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if isinstance(error, worker.NotReady):
            raise
        raise worker.NotReady('GPU3 state not safe/available; leave unknown active untouched: ' + str(error)) from error
    finally:
        if pidfd is not None:
            os.close(pidfd)
        if lockfd is not None:
            fcntl.flock(lockfd, fcntl.LOCK_UN)
            os.close(lockfd)


def save_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def source_config(args):
    # Check all executable dependencies before importing any of them.
    import hashlib
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest(__file__) != args.self_sha or digest(SOURCE) != args.source_prepared_sha:
        raise RuntimeError('Frozen waiter/source-prepared SHA changed')
    cfg = json.loads(SOURCE.read_bytes())
    if (cfg['node'] != 'N6' or cfg['gpus'] != [3] or cfg['holders'] != []
            or cfg['release_authorized'] is not False or cfg['holder_capture'] is not None
            or cfg['env']['CUDA_VISIBLE_DEVICES'] != '3' or cfg['targetcount'] != 3
            or cfg['resource_manager_ref']['path'] != str(MANAGER)
            or cfg['resource_manager_ref']['sha256'] != args.manager_sha):
        raise RuntimeError('Only unadmitted exact N6G3 source-prepared config allowed')
    for path, sha in cfg['source_sha256'].items():
        if digest(path) != sha:
            raise RuntimeError('Frozen review3 input/source changed: ' + path)
    if digest(MANAGER) != args.manager_sha or digest(RT / 'resource_manager.py') != BASE_SHA:
        raise RuntimeError('Frozen recovery3 manager SHA changed')
    m = module(MANAGER)
    if (cfg['queue'] != str(m.QUEUE) or cfg['out'] != str(m.worker.OUT)
            or cfg['phase'] != m.PHASE or cfg['input_count'] != 222939):
        raise RuntimeError('Independent recovery3 queue/output/denominator changed')
    m.selected(cfg)
    m.support.check_model(cfg)
    return cfg, m


def authorize(cfg, capture, references, manager, args):
    base, worker = manager.base, manager.worker
    if manager.QUEUE.exists() or POINTER.exists():
        raise RuntimeError('Recovery3 manager/admission already exists; do not start twice')
    # Consumer terminals and exact holder/lease must both remain unchanged.
    _, current_refs = worker.ready_records()
    if current_refs != references:
        raise worker.NotReady('Recovery terminal/ref changed before manager exec')
    fresh = fresh_capture(manager)
    if (any(fresh['holders'][0][k] != capture['holders'][0][k] for k in base.IDENTITY_KEYS + ('gpu', 'uuid'))
            or fresh['lease'] != capture['lease']):
        raise worker.NotReady('GPU3 holder/lease changed before manager exec; no signals')
    token = uuid.uuid4().hex
    captured = RT / ('curated_io_recovery_review3_n6g3_fresh_' + token + '.json')
    launch = RT / ('curated_io_recovery_review3_n6g3_launch_' + token + '.json')
    save_new(captured, fresh)
    active = copy.deepcopy(cfg)
    active.update(holders=fresh['holders'], holder_capture=worker.ref(captured), release_authorized=True,
                  ready_source_refs=references, source_prepared_ref=worker.ref(SOURCE),
                  preparation_status='fresh_N6G3_after_exact_three_consumer_terminals')
    for reference in references + [worker.ref(captured), worker.ref(SOURCE), worker.ref(__file__)]:
        active['source_sha256'][reference['path']] = reference['sha256']
    save_new(launch, active)
    manager.validate(active)
    base.check_sources(active, launch, base.sha(launch), BASE_SHA)
    manager.support.check_model(active)
    save_new(POINTER, dict(config=worker.ref(launch), capture=worker.ref(captured), manager=worker.ref(MANAGER),
        source_prepared_ref=worker.ref(SOURCE), ready_source_refs=references,
        node='N6', gpus=[3], signals_sent_by_waiter=False, GPU_started_by_waiter=False))
    print('All three recovery terminals verified; exact N6G3 admitted. Executing frozen resource manager.', flush=True)
    argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(MANAGER), '--config', str(launch),
            '--config-sha', base.sha(launch), '--manager-sha', BASE_SHA]
    os.execv(argv[0], argv)


def wait(cfg, manager, args):
    last = None
    while True:
        try:
            _, refs = manager.worker.ready_records()
            capture = fresh_capture(manager)
        except manager.worker.NotReady as error:
            reason = str(error)
            if reason != last:
                print('Waiting without GPU signals: ' + reason, flush=True)
                last = reason
            if args.readonly_now:
                return
            time.sleep(30)
            continue
        if args.readonly_now:
            print('Read-only ready: all three terminals + safe N6G3 project holder/lease/headroom. No GPU started.', flush=True)
            return
        try:
            authorize(cfg, capture, refs, manager, args)
        except manager.worker.NotReady as error:
            print('Waiting without GPU signals: ' + str(error), flush=True)
            time.sleep(30)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--wait', action='store_true')
    g.add_argument('--readonly-now', action='store_true')
    p.add_argument('--source-prepared-sha', required=True)
    p.add_argument('--manager-sha', required=True)
    p.add_argument('--self-sha', required=True)
    args = p.parse_args()
    cfg, m = source_config(args)
    if args.readonly_now:
        wait(cfg, m, args)
        return
    path = RT / 'curated_io_recovery_review3_n6g3_cpu_waiter_v1.lock'
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_size:
            raise RuntimeError('Unknown CPU waiter lock type/owner/content')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.set_inheritable(fd, True)
        wait(cfg, m, args)
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
