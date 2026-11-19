import argparse
import importlib.util
import json
import os
import socket
import stat
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location('bar_resource_manager', Path(__file__).with_name('resource_manager.py'))
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


def snapshot(path):
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None
    return dict(path=str(path), mode=st.st_mode, uid=st.st_uid, size=st.st_size,
                device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns,
                regular=stat.S_ISREG(st.st_mode), owner_self=st.st_uid == os.getuid(),
                stale_1200=time.time() - st.st_mtime > 1200)


def capture(node, gpus, pids):
    if tuple(gpus) not in manager.AUTHORIZED_PAIRS.get(node, ()) or len(pids) != 2:
        raise RuntimeError('capture requires an explicitly authorized pair and two specified PIDs')
    running = manager.apps()
    result = dict(schema='ms_bar_resource_capture_v1', node=node, hostname=socket.gethostname(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  captured_unix=time.time(), gpus=gpus, holders=[], files=[],
                  headroom=manager.headroom(), signals_sent=False)
    for gpu, pid in zip(gpus, pids):
        first = manager.identity(pid)
        uuid = manager.uuid_for(gpu)
        expected = dict(first, gpu=gpu, uuid=uuid)
        manager.check_holder(expected, manager.identity(pid), gpu, uuid, running)
        fd = os.pidfd_open(pid)
        try:
            if manager.select.select([fd], [], [], 0)[0]:
                raise RuntimeError('captured holder has exited')
            manager.check_holder(expected, manager.identity(pid), gpu, manager.uuid_for(gpu), manager.apps())
        finally:
            os.close(fd)
        result['holders'].append(expected)
        reserve = Path('/tmp/od_reserve_%d' % gpu)
        lock = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
        old = manager.lease_snapshot(reserve)
        result['files'].append(dict(gpu=gpu, lock=snapshot(lock), reserve=snapshot(reserve),
                                    reserve_admissible=True, absent_reserve=old is None))
        ls = result['files'][-1]['lock']
        if ls is not None and not (ls['regular'] and ls['owner_self'] and ls['size'] == 0):
            raise RuntimeError('unsafe existing lock file')
    return result


def write_new(path, payload):
    path = Path(path).resolve()
    if not path.is_relative_to(manager.EXPERIMENT):
        raise RuntimeError('output must be inside new experiment')
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')


def build(template_path, capture_path):
    cfg = json.loads(Path(template_path).read_text())
    fresh = json.loads(Path(capture_path).read_text())
    if fresh['node'] != cfg['node'] or fresh['gpus'] != cfg['gpus'] or fresh.get('signals_sent') is not False:
        raise RuntimeError('capture/config pair mismatch')
    if not 0 <= time.time() - fresh['captured_unix'] <= 1200:
        raise RuntimeError('stale capture; recapture specified exact holders')
    cfg['holders'] = fresh['holders']
    cfg['capture'] = dict(path=str(Path(capture_path).resolve()), sha256=manager.sha(capture_path),
                          hostname=fresh['hostname'], boot_id=fresh['boot_id'], captured_unix=fresh['captured_unix'])
    for kind in ('producer', 'plan', 'modelref'):
        path = str(Path(cfg[kind]['path']).resolve())
        cfg[kind]['path'] = path
        cfg[kind]['sha256'] = manager.sha(path)
    model = json.loads(Path(cfg['modelref']['path']).read_text())['model_identity']
    cfg['modelref']['model_identity_sha256'] = model['sha256']
    cfg['source_sha256'] = {str(Path(path).resolve()): manager.sha(path) for path in cfg['source_sha256']}
    cfg['source_sha256'][cfg['producer']['path']] = cfg['producer']['sha256']
    cfg['source_sha256'][cfg['plan']['path']] = cfg['plan']['sha256']
    manager.validate_pair(cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    commands = ap.add_subparsers(dest='command', required=True)
    ca = commands.add_parser('capture')
    ca.add_argument('--node', required=True, choices=list(manager.AUTHORIZED_PAIRS))
    ca.add_argument('--gpus', nargs=2, type=int, required=True)
    ca.add_argument('--pids', nargs=2, type=int, required=True)
    ca.add_argument('--out', required=True)
    bu = commands.add_parser('build')
    bu.add_argument('--template', required=True)
    bu.add_argument('--capture', required=True)
    bu.add_argument('--out', required=True)
    args = ap.parse_args()
    payload = capture(args.node, args.gpus, args.pids) if args.command == 'capture' else build(args.template, args.capture)
    write_new(args.out, payload)
    print(json.dumps(dict(path=args.out, sha256=manager.sha(args.out),
                         manager_sha256=manager.sha(manager.__file__), signals_sent=False)), flush=True)


if __name__ == '__main__':
    main()
