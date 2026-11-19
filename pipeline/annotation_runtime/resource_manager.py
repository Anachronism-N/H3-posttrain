import argparse
import fcntl
import hashlib
import json
import os
import select
import signal
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXPERIMENT = ROOT / 'diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1'
HOLDER = ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py'
SAFETY = 80 * 2**30
LOAD = 80 * 2**30
IDENTITY_KEYS = ('pid', 'uid', 'argv', 'start_ticks', 'cgroup', 'CVD', 'exe')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
    env = (p / 'environ').read_bytes().split(b'\0')
    status = dict(line.split(':', 1) for line in (p / 'status').read_text().splitlines() if ':' in line)
    return dict(pid=pid, uid=p.stat().st_uid, ppid=int(fields[1]),
                start_ticks=int(fields[19]), state=fields[0],
                argv=[x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x],
                cgroup=(p / 'cgroup').read_text().strip(),
                CVD=[x.decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES=')],
                exe=os.readlink(p / 'exe'), sig_ign=int(status['SigIgn'].strip(), 16),
                sig_blk=int(status['SigBlk'].strip(), 16))


def apps():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    return [dict(uuid=row[0], pid=int(row[1]), used_MiB=row[2])
            for line in text.splitlines() if line.strip()
            for row in [[v.strip() for v in line.split(',')]]]


def uuid_for(gpu):
    return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader',
                                    '-i', str(gpu)], text=True).strip()


def headroom():
    p = Path('/sys/fs/cgroup/memory')
    current = int((p / 'memory.current').read_text())
    maximum = (p / 'memory.max').read_text().strip()
    available = int(next(x for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:')).split()[1]) * 1024
    cap = int(maximum) if maximum != 'max' else current + available
    return dict(current=current, maximum=maximum, headroom=min(cap-current, available))


def check_holder(expected, current, gpu, uuid, running):
    for key in IDENTITY_KEYS:
        if current[key] != expected[key]:
            raise RuntimeError('holder identity changed: GPU%d %s' % (gpu, key))
    if current['uid'] != os.getuid() or current['argv'][-1:] != [str(HOLDER)]:
        raise RuntimeError('not an authorized project holder')
    if current['CVD'] != ['CUDA_VISIBLE_DEVICES=' + str(gpu)]:
        raise RuntimeError('holder CVD changed')
    if (current['sig_ign'] | current['sig_blk']) & (1 << (signal.SIGTERM - 1)):
        raise RuntimeError('holder SIGTERM ignored or blocked')
    if uuid != expected['uuid']:
        raise RuntimeError('GPU UUID changed')
    if [r['pid'] for r in running if r['uuid'] == uuid] != [expected['pid']]:
        raise RuntimeError('GPU has changed or nonholder compute processes')


def lease_snapshot(path):
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size != 0:
        raise RuntimeError('unsafe lease type/owner/content: ' + str(path))
    if time.time()-st.st_mtime <= 1200:
        raise RuntimeError('active lease: ' + str(path))
    return st


def check_lease_unchanged(path, old):
    try:
        now = path.lstat()
    except FileNotFoundError:
        if old is None:
            return
        raise RuntimeError('lease disappeared')
    if old is None or (now.st_dev, now.st_ino, now.st_uid, now.st_mode, now.st_size,
                       now.st_mtime_ns) != (old.st_dev, old.st_ino, old.st_uid, old.st_mode,
                                           old.st_size, old.st_mtime_ns):
        raise RuntimeError('lease changed since snapshot')


AUTHORIZED_PAIRS = {
    'N1': ((0, 1), (2, 3)), 'N5': ((0, 1), (6, 7)),
    'N6': ((4, 5), (6, 7)), 'N8': ((4, 5), (6, 7)), 'N3': ((2, 3), (4, 5)),
}
RESTORE_ARGV = ['/opt/venv_eval/bin/python', '-u', str(HOLDER)]


def validate_pair(cfg):
    pair = cfg['gpus']
    if tuple(pair) not in AUTHORIZED_PAIRS.get(cfg['node'], ()):
        raise RuntimeError('outside explicitly authorized GPU pairs')
    if cfg.get('release_authorized') is not True:
        raise RuntimeError('explicit holder SIGTERM authorization required')
    expected = cfg['holders']
    if len(pair) != 2 or len(expected) != 2:
        raise RuntimeError('exactly two GPU holder identities required')
    if len({e['uuid'] for e in expected}) != 2 or len({e['pid'] for e in expected}) != 2:
        raise RuntimeError('duplicate GPU UUID or holder PID')
    if [e['gpu'] for e in expected] != pair:
        raise RuntimeError('holder identities not in physical GPU order')
    if not isinstance(cfg['targetcount'], int) or isinstance(cfg['targetcount'], bool) or cfg['targetcount'] <= 0:
        raise RuntimeError('positive stage targetcount required')
    phase = cfg['phase']
    if not isinstance(phase, str) or not phase or not all(c.isalnum() or c in '_-' for c in phase):
        raise RuntimeError('invalid stage phase')
    if not cfg['argv'] or not all(isinstance(v, str) and v for v in cfg['argv']):
        raise RuntimeError('explicit worker argv required')
    if cfg['env'].get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, pair)):
        raise RuntimeError('worker CVD must exactly match admitted physical pair')
    if not Path(cfg['cwd']).is_absolute():
        raise RuntimeError('absolute worker cwd required')


def required_budget(cfg):
    return SAFETY + LOAD


def configure_local_restore(entry, cfg):
    entry['restore_argv'] = list(RESTORE_ARGV)
    entry['restore_cwd'] = '/tmp'
    env = entry['original_env'].copy()
    for name in ('PYTHONHOME', 'PYTHONPATH', 'CONDA_PREFIX'):
        env.pop(name, None)
    env.update({'OCCUPY_GPU': '0', 'OMP_NUM_THREADS': '2'})
    env['CUDA_VISIBLE_DEVICES'] = str(entry['gpu'])
    entry['restore_env'] = env


def release_holder(entry):
    # Mark the attempt first: a delivered signal may interrupt Python on return.
    entry['released'] = True
    signal.pidfd_send_signal(entry['fd'], signal.SIGTERM)


def refresh_leases(entries, on_error=None):
    for e in entries:
        if e['leasefd'] is None or 'lease_inode' not in e:
            continue
        try:
            now = e['reserve'].lstat()
            if ((now.st_dev, now.st_ino) != e['lease_inode'] or now.st_uid != os.getuid()
                    or not stat.S_ISREG(now.st_mode) or now.st_size != 0):
                raise RuntimeError('owned lease replaced or changed')
            os.utime(e['leasefd'], None)
        except BaseException as error:
            if on_error is None:
                raise
            on_error(e, error)


def reset_holder_signals():
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def restore_step(e, save, holders):
    if e.get('restore_state') == 'blocked':
        return False
    try:
        if e.get('restore_state') == 'safe':
            if e['released']:
                holder = e['restored_holder']
                if holder.poll() is not None:
                    raise RuntimeError('previously restored holder exited before pair unlock')
                check_holder(e['restored_identity'], identity(holder.pid), e['gpu'],
                             uuid_for(e['gpu']), apps())
            return True
        if not e['released']:
            e['restore_state'] = 'safe'
            return True
        if not select.select([e['fd']], [], [], 0)[0]:
            # An attempted SIGTERM may still be pending; do not unlock prematurely.
            if not e.get('original_wait_reported'):
                save('holder_exit_wait_gpu%d.json' % e['gpu'], dict(reason='original holder still alive',
                     lock_and_lease_retained=True, all_pair_leases_heartbeat=True,
                     unknown_jobs_not_signaled=True))
                e['original_wait_reported'] = True
            return False
        running = [r for r in apps() if r['uuid'] == e['uuid']]
        holder = e.get('restored_holder')
        if holder is None:
            if running:
                if not e.get('restore_busy_reported'):
                    save('restore_blocked_gpu%d.json' % e['gpu'], dict(apps=running,
                         unknown_jobs_not_signaled=True, lock_and_lease_retained=True,
                         all_pair_leases_heartbeat=True, managed_parent_wait=True))
                    e['restore_busy_reported'] = True
                return False
            log = (e['queue'] / ('restored_holder_gpu%d.log' % e['gpu'])).open('xb')
            e['restore_log'] = log
            # A failed spawn is blocked permanently; never retry an ambiguous Popen.
            holder = subprocess.Popen(e['restore_argv'], env=e['restore_env'], cwd=e['restore_cwd'],
                                      stdout=log, stderr=subprocess.STDOUT, preexec_fn=reset_holder_signals)
            e['restored_holder'] = holder
            e['restore_started'] = time.monotonic()
            holders.append((holder, log))
            return False
        if holder.poll() is not None:
            raise RuntimeError('restored holder exited; no duplicate holder will be spawned')
        if [r['pid'] for r in running] == [holder.pid]:
            restored = identity(holder.pid)
            if (restored['uid'] != os.getuid() or restored['argv'] != e['restore_argv']
                    or restored['CVD'] != e['identity']['CVD']
                    or (restored['sig_ign'] | restored['sig_blk']) & (1 << (signal.SIGTERM - 1))):
                raise RuntimeError('restored holder argv/CVD mismatch')
            save('holder_restored_gpu%d.json' % e['gpu'], dict(gpu=e['gpu'], uuid=e['uuid'],
                 holder_identity=restored, original_identity=e['identity'],
                 original_argv=e['identity']['argv'], original_cwd=e['original_cwd'],
                 restore_argv=e['restore_argv'], restore_cwd=e['restore_cwd'],
                 original_argv_preserved=False, original_env_preserved=False,
                 restore_reason='authorized_local_python_to_avoid_shared_env_io', managed_parent_wait=True))
            e['restored_identity'] = dict(restored, uuid=e['uuid'])
            e['restore_state'] = 'safe'
            return True
        if any(r['pid'] != holder.pid for r in running) and not e.get('restore_unknown_reported'):
            save('restore_unknown_gpu%d.json' % e['gpu'], dict(apps=running, holder_pid=holder.pid,
                 unknown_jobs_not_signaled=True, all_pair_leases_heartbeat=True,
                 lock_and_lease_retained=True, status='blocked'))
            e['restore_unknown_reported'] = True
        if time.monotonic()-e['restore_started'] > 120 and not e.get('restore_wait_reported'):
            save('restore_wait_gpu%d.json' % e['gpu'], dict(holder_pid=holder.pid,
                 lock_and_lease_retained=True, all_pair_leases_heartbeat=True, no_duplicate_spawn=True))
            e['restore_wait_reported'] = True
        return False
    except BaseException as error:
        e['restore_state'] = 'blocked'
        save('restore_failure_gpu%d.json' % e['gpu'], dict(error=repr(error),
             unknown_jobs_not_signaled=True, lock_and_lease_retained=True,
             all_pair_leases_heartbeat=True, no_duplicate_spawn=True))
        return False


def save_receipt(queue, name, value, phase):
    with (queue / name).open('x') as stream:
        json.dump(value, stream, indent=2)
    # Progress output is not part of the durable resource transaction.
    try:
        print(json.dumps(dict(event=name, phase=phase, path=str(queue/name),
                              utc=time.strftime('%FT%TZ', time.gmtime()))), flush=True)
    except OSError:
        pass



def check_sources(cfg, config_path, config_sha, manager_sha):
    if sha(__file__) != manager_sha or sha(config_path) != config_sha:
        raise RuntimeError('manager or launch config SHA mismatch')
    bindings = dict(cfg['source_sha256'])
    for kind in ('producer', 'plan'):
        ref = cfg[kind]
        if ref['path'] in bindings and bindings[ref['path']] != ref['sha256']:
            raise RuntimeError('conflicting source binding: ' + ref['path'])
        bindings[ref['path']] = ref['sha256']
    if cfg['producer']['path'] not in cfg['argv']:
        raise RuntimeError('worker argv does not name frozen producer')
    for name, digest in bindings.items():
        if not Path(name).is_absolute() or sha(name) != digest:
            raise RuntimeError('source/input changed: ' + name)
    selected_ids, total = selected_sources(cfg)
    if len(selected_ids) != cfg['targetcount']:
        raise RuntimeError('selected source count differs from targetcount')
    if cfg.get('input_count', total) != total:
        raise RuntimeError('full plan count differs from input_count')


def argv_option(cfg, option, default=None):
    args = cfg['argv']
    matches = []
    for index, value in enumerate(args):
        if value == option:
            if index + 1 >= len(args):
                raise RuntimeError('missing worker option value: ' + option)
            matches.append(args[index + 1])
        elif value.startswith(option + '='):
            matches.append(value.split('=', 1)[1])
    if len(matches) > 1:
        raise RuntimeError('duplicate worker option: ' + option)
    return matches[0] if matches else default


def selected_sources(cfg):
    # Matches producer.read_plan: full-plan IDs, stride sharding, then limit.
    selection = {key: int(argv_option(cfg, '--' + key, default))
                 for key, default in (('shard', 0), ('nshard', 1), ('limit', 0))}
    if cfg.get('source_selection', selection) != selection:
        raise RuntimeError('source_selection does not match explicit worker argv')
    shard, nshard, limit = (selection[key] for key in ('shard', 'nshard', 'limit'))
    if nshard <= 0 or not 0 <= shard < nshard or limit < 0:
        raise RuntimeError('invalid source selection')
    argv_plan = argv_option(cfg, '--plan')
    if argv_plan is not None and Path(argv_plan).resolve() != Path(cfg['plan']['path']).resolve():
        raise RuntimeError('worker argv plan differs from SHA-bound plan')
    with Path(cfg['plan']['path']).open(encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row['id'] for row in rows]
    if not ids or not all(isinstance(value, str) and value for value in ids):
        raise RuntimeError('nonempty source IDs required')
    if len({value.casefold() for value in ids}) != len(ids):
        raise RuntimeError('duplicate source ID in full plan')
    selected = ids[shard::nshard]
    return (selected[:limit] if limit else selected), len(ids)


def stage_summaries(cfg, out):
    selected_ids, total = selected_sources(cfg)
    expected_stage = argv_option(cfg, '--stage', 'prepare')
    paths = set(out.glob('snapshots/*/summary.json'))
    paths.update(p for p in (out / 'run.summary.json', out / 'summary.json') if p.exists())
    matched = {}
    for path in paths:
        value = json.loads(path.read_text())
        summary = value.get('summary', value)
        if (summary.get('stage') != expected_stage or summary.get('plan_sha256') != cfg['plan']['sha256']
                or summary.get('total_source_count') != total
                or summary.get('selected_source_count') != cfg['targetcount']):
            continue
        selection = value.get('selection')
        if selection is not None and selection.get('selected_ids') != selected_ids:
            continue
        matched[str(path)] = value
    return matched


def check_model(cfg):
    ref = cfg['modelref']
    if sha(ref['path']) != ref['sha256']:
        raise RuntimeError('model reference file changed')
    reference = json.loads(Path(ref['path']).read_text())['model_identity']
    model = Path('/opt/models/Qwen3-VL-32B-Instruct')
    if reference['sha256'] != ref['model_identity_sha256'] or reference['root'] != str(model):
        raise RuntimeError('model identity reference mismatch')
    inventory = {str(p.relative_to(model)) for p in model.rglob('*') if p.is_file()
                 and not any(s.startswith('.') for s in p.relative_to(model).parts)}
    if len(reference['files']) != 25 or inventory != set(reference['files']):
        raise RuntimeError('model file inventory mismatch')
    for name, value in reference['files'].items():
        if (model / name).stat().st_size != value['stat']['size']:
            raise RuntimeError('model file size changed: ' + name)
    for name in ('config.json', 'model.safetensors.index.json'):
        if sha(model / name) != reference['files'][name]['sha256']:
            raise RuntimeError('model config/index changed')
    if cfg['node'] != 'N1':
        marker = json.loads(Path(str(model) + '.stage_receipt.json').read_text())
        if marker.get('complete') is not True or marker['model_identity_sha256'] != reference['sha256']:
            raise RuntimeError('local staged model identity mismatch')


def wait_generation_gate(cfg, save):
    gate = cfg.get('previous_generation_gate')
    if not gate:
        return
    gq, go = Path(gate['queue']).resolve(), Path(gate['out']).resolve()
    if not gq.is_relative_to(EXPERIMENT / 'queues') or not go.is_relative_to(EXPERIMENT):
        raise RuntimeError('generation gate outside current experiment')
    deadline = time.monotonic() + 3600
    while True:
        if (gq / 'failure.json').exists():
            raise RuntimeError('previous pair failed; coordinator review required')
        exit_path = gq / 'worker_exit.json'
        if exit_path.exists() and json.loads(exit_path.read_text()).get('exit_code') != 0:
            raise RuntimeError('previous pair worker failed')
        admission = gq / 'admission.json'
        if admission.exists():
            value = json.loads(admission.read_text())
            if value['node'] != cfg['node'] or set(value['gpus']) & set(cfg['gpus']):
                raise RuntimeError('generation gate must name other pair on same node')
            if value['model_identity_sha256'] != cfg['modelref']['model_identity_sha256']:
                raise RuntimeError('generation gate model mismatch')
            if Path(value['out']).resolve() != go:
                raise RuntimeError('generation gate output differs from admitted output')
            if gate.get('config_sha256') and value['config_sha256'] != gate['config_sha256']:
                raise RuntimeError('generation gate stage config mismatch')
            admitted_ns = admission.stat().st_mtime_ns
            loads = [Path(gate['load_receipt'])] if gate.get('load_receipt') else list(go.rglob('load.receipt.json'))
            generations = [Path(gate['generation_receipt'])] if gate.get('generation_receipt') else list(go.rglob('generation.json'))
            for p in loads + generations:
                if not p.resolve().is_relative_to(go):
                    raise RuntimeError('generation receipt outside gate output')
            valid = [p for p in generations if p.exists() and p.stat().st_mtime_ns >= admitted_ns
                     and json.loads(p.read_text()).get('model_call_performed') is True]
            current_loads = [p for p in loads if p.exists() and p.stat().st_mtime_ns >= admitted_ns]
            if current_loads and valid:
                save('previous_generation_gate.json', dict(queue=str(gq), generation=str(valid[0]),
                     fresh_budget_required=True, required_bytes=SAFETY + LOAD))
                return
        if time.monotonic() > deadline:
            raise RuntimeError('real generation gate timeout; no holder released')
        time.sleep(5)


def acquire_lease(e):
    check_lease_unchanged(e['reserve'], e['old'])
    flags = os.O_RDWR | os.O_NOFOLLOW
    if e['old'] is None:
        flags |= os.O_CREAT | os.O_EXCL
    e['leasefd'] = os.open(e['reserve'], flags, 0o600)
    lease = os.fstat(e['leasefd'])
    if not stat.S_ISREG(lease.st_mode) or lease.st_uid != os.getuid() or lease.st_size != 0:
        raise RuntimeError('unsafe lease after open')
    if e['old'] and (lease.st_ino, lease.st_dev, lease.st_mtime_ns) != (
            e['old'].st_ino, e['old'].st_dev, e['old'].st_mtime_ns):
        os.close(e['leasefd'])
        e['leasefd'] = None
        raise RuntimeError('lease replaced during admission')
    e['lease_inode'] = (lease.st_dev, lease.st_ino)
    refresh_leases([e])


def interrupted(signum, frame):
    raise InterruptedError('manager received signal %d' % signum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--config-sha', required=True)
    ap.add_argument('--manager-sha', required=True)
    args = ap.parse_args()
    if sha(__file__) != args.manager_sha or sha(args.config) != args.config_sha:
        raise RuntimeError('manager or launch config SHA mismatch')
    cfg = json.loads(Path(args.config).read_text())
    validate_pair(cfg)
    pair = cfg['gpus']
    queue, out = Path(cfg['queue']).resolve(), Path(cfg['out']).resolve()
    if not queue.is_relative_to(EXPERIMENT / 'queues') or not out.is_relative_to(EXPERIMENT):
        raise RuntimeError('paths outside current experiment')
    if queue == out or queue.is_relative_to(out) or out.is_relative_to(queue):
        raise RuntimeError('queue and output must not overlap')
    queue.mkdir(parents=True, exist_ok=False)
    if out.exists() and (not out.is_dir() or cfg.get('reuse_existing_results') is not True):
        raise RuntimeError('existing output requires explicit reuse_existing_results=true')

    def save(name, value):
        save_receipt(queue, name, value, cfg['phase'])

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    locks, entries, holders = [], [], []
    child = None
    started = False
    peak, gpu_peak = 0, {g: 0 for g in pair}
    try:
        check_sources(cfg, args.config, args.config_sha, args.manager_sha)
        check_model(cfg)
        wait_generation_gate(cfg, save)
        budget = headroom()
        needed = required_budget(cfg)
        if budget['headroom'] < needed:
            save('pending_budget.json', dict(budget=budget, required_bytes=needed, no_holder_released=True))
            return
        for gpu, expected in zip(pair, cfg['holders']):
            path = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
            lockfd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            locks.append(lockfd)
            st = os.fstat(lockfd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size != 0:
                raise RuntimeError('unsafe lock type/owner/content')
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reserve = Path('/tmp/od_reserve_%d' % gpu)
            old = lease_snapshot(reserve)
            fd = os.pidfd_open(expected['pid'])
            entry = dict(gpu=gpu, uuid=expected['uuid'], fd=fd, expected=expected,
                         reserve=reserve, old=old, leasefd=None, released=False,
                         original_env=None, original_cwd=None)
            entries.append(entry)
            if select.select([fd], [], [], 0)[0]:
                raise RuntimeError('expected holder already exited')
            current = identity(expected['pid'])
            check_holder(expected, current, gpu, uuid_for(gpu), apps())
            rawenv = (Path('/proc') / str(expected['pid']) / 'environ').read_bytes()
            entry['original_env'] = dict(x.decode().split('=', 1) for x in rawenv.split(b'\0') if x)
            entry['original_cwd'] = os.readlink(Path('/proc') / str(expected['pid']) / 'cwd')
            entry['identity'] = current
            configure_local_restore(entry, cfg)
        check_sources(cfg, args.config, args.config_sha, args.manager_sha)
        budget = headroom()
        if budget['headroom'] < needed:
            raise RuntimeError('fresh headroom below stage admission budget; no holder released')
        save('admission.json', dict(config_sha256=args.config_sha, manager_sha256=args.manager_sha,
             manager_identity=identity(os.getpid()), phase=cfg['phase'], targetcount=cfg['targetcount'],
             node=cfg['node'], gpus=pair, producer=cfg['producer'], plan=cfg['plan'], out=str(out),
             source_sha256=cfg['source_sha256'], budget=budget, safety_bytes=SAFETY,
             load_allowance_bytes=LOAD, admission_required_bytes=needed,
             model_identity_sha256=cfg['modelref']['model_identity_sha256'],
             holders=[dict(gpu=e['gpu'], uuid=e['uuid'], identity=e['identity'],
                           previous_lease_mtime=e['old'].st_mtime if e['old'] else None) for e in entries]))
        for e in entries:
            acquire_lease(e)
        # All pair leases are owned before ANY holder is released.
        for e in entries:
            if select.select([e['fd']], [], [], 0)[0]:
                raise RuntimeError('holder exited before authorization signal')
            check_holder(e['expected'], identity(e['expected']['pid']), e['gpu'], uuid_for(e['gpu']), apps())
        if headroom()['headroom'] < needed:
            raise RuntimeError('admission budget changed before holder release')
        for e in entries:
            release_holder(e)
        for e in entries:
            if not select.select([e['fd']], [], [], 30)[0]:
                raise RuntimeError('authorized holder did not exit')
        uuids = {e['uuid'] for e in entries}
        for _ in range(120):
            refresh_leases(entries)
            if not [r for r in apps() if r['uuid'] in uuids]:
                break
            time.sleep(.25)
        if [r for r in apps() if r['uuid'] in uuids]:
            raise RuntimeError('GPU remains occupied; unknown jobs not signaled')
        out.mkdir(parents=True, exist_ok=cfg.get('reuse_existing_results') is True)
        env = os.environ.copy()
        env.update(cfg['env'])
        peak = budget['current']
        runtime_log = queue / 'worker.runtime.log'
        with (queue / 'resource_samples.jsonl').open('x') as samples, runtime_log.open('xb') as log:
            check_sources(cfg, args.config, args.config_sha, args.manager_sha)
            child = subprocess.Popen(cfg['argv'], env=env, cwd=cfg['cwd'], stdout=log,
                                     stderr=subprocess.STDOUT, preexec_fn=reset_holder_signals)
            started = True
            time.sleep(.2)
            save('live.json', dict(child_identity=identity(child.pid) if child.poll() is None else None,
                 early_exit_code=child.poll(), manager_identity=identity(os.getpid()),
                 phase=cfg['phase'], targetcount=cfg['targetcount'], argv=cfg['argv'], runtime_log=str(runtime_log)))
            while child.poll() is None:
                refresh_leases(entries)
                memory = headroom()
                peak = max(peak, memory['current'])
                if memory['headroom'] < SAFETY and not (queue / 'low_memory_warning.json').exists():
                    save('low_memory_warning.json', dict(memory=memory, no_other_pairs_should_start=True))
                snapshots = []
                for gpu in pair:
                    values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits', '-i', str(gpu)], text=True).strip().split(',')
                    used, util = map(int, values)
                    gpu_peak[gpu] = max(gpu_peak[gpu], used)
                    snapshots.append(dict(gpu=gpu, used_MiB=used, util_pct=util))
                samples.write(json.dumps(dict(utc=time.strftime('%FT%TZ', time.gmtime()), memory=memory,
                                             gpus=snapshots)) + '\n')
                samples.flush()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            code = child.wait()
        child = None
        check_sources(cfg, args.config, args.config_sha, args.manager_sha)
        summaries = stage_summaries(cfg, out)
        save('worker_exit.json', dict(exit_code=code, phase=cfg['phase'], targetcount=cfg['targetcount'],
             input_count=selected_sources(cfg)[1], source_selection=cfg.get('source_selection'), summaries=summaries,
             producer=cfg['producer'], plan=cfg['plan'], source_unchanged=True, cgroup_observed_peak=peak,
             gpu_peak_MiB=gpu_peak, semantic_approval=False, training_entry_updated=False))
        if code != 0:
            raise RuntimeError('stage worker exit ' + str(code))
        if not summaries:
            raise RuntimeError('worker exited zero without matching current-stage summary')
    except BaseException as error:
        save('failure.json', dict(error=repr(error), phase=cfg['phase'], partial_outputs_preserved=True,
                                 worker_started=started, unknown_jobs_not_signaled=True))
    finally:
        # Even on failure, never unlock before BOTH exact holders are safe.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cleanup_errors = set()

        def retained_error(e, error):
            key = e['gpu']
            if key not in cleanup_errors:
                save('lease_cleanup_failure_gpu%d.json' % key, dict(error=repr(error),
                     lock_retained=True, unknown_lease_not_modified=True))
                cleanup_errors.add(key)

        if child is not None and child.poll() is None:
            child.terminate()  # Only our direct Popen child, never any process group.
            deadline = time.monotonic() + 120
            while child.poll() is None:
                refresh_leases(entries, retained_error)
                if time.monotonic() > deadline and not (queue / 'child_cleanup_blocked.json').exists():
                    save('child_cleanup_blocked.json', dict(pid=child.pid, unknown_jobs_not_signaled=True,
                         all_pair_leases_heartbeat=True, lock_and_lease_retained=True))
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        for e in entries:
            e['queue'] = queue
            if e['leasefd'] is not None and 'lease_inode' not in e:
                retained_error(e, RuntimeError('uncertain lease admission ownership'))
        while True:
            refresh_leases(entries, retained_error)
            safe = [restore_step(e, save, holders) for e in entries]
            if all(safe) and not cleanup_errors:
                break
            time.sleep(5)
        for e in entries:
            if e['leasefd'] is not None:
                try:
                    refresh_leases([e])
                    if e['old'] is None:
                        e['reserve'].unlink()
                    else:
                        os.utime(e['leasefd'], ns=(e['old'].st_atime_ns, e['old'].st_mtime_ns))
                except BaseException as error:
                    retained_error(e, error)
                    while True:
                        refresh_leases(entries, retained_error)
                        time.sleep(5)
                os.close(e['leasefd'])
                e['leasefd'] = None
            os.close(e['fd'])
        for fd in locks:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        save('resources_safe.json', dict(phase=cfg['phase'], all_holders_safe=True, locks_released=True,
             leases_restored=True, managed_parent_wait=True, unknown_jobs_not_signaled=True))
        for h, log in holders:
            h.wait()
            log.close()


if __name__ == '__main__':
    main()
