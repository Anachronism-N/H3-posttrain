import argparse
import fcntl
import hashlib
import json
import math
import os
import select
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1/runtime')
import stage_models as stage

ROOT = stage.ROOT
EXPERIMENT = stage.EXPERIMENT
HOLDER = ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py'
SAFETY = 80 * 2**30
LOAD = 80 * 2**30
IDENTITY_KEYS = ('pid', 'uid', 'argv', 'start_ticks', 'cgroup', 'CVD', 'exe')
RESTORE_ARGV = ['/opt/venv_eval/bin/python', '-u', str(HOLDER)]


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
                OCCUPY=[x.decode() for x in env if x.startswith(b'OCCUPY_GPU=')],
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
    if current['cgroup'] != identity(os.getpid())['cgroup'] or current.get('OCCUPY') != ['OCCUPY_GPU=0']:
        raise RuntimeError('holder cgroup or OCCUPY_GPU scope mismatch')
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
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})


def restore_step(e, save, holders):
    if e.get('restore_state') == 'blocked':
        return False
    try:
        if e.get('restore_state') == 'safe':
            if e['released']:
                holder = e['restored_holder']
                if holder.poll() is not None:
                    raise RuntimeError('previously restored holder exited before node unlock')
                check_holder(e['restored_identity'], identity(holder.pid), e['gpu'],
                             uuid_for(e['gpu']), apps())
            else:
                check_holder(e['expected'], identity(e['expected']['pid']), e['gpu'],
                             uuid_for(e['gpu']), apps())
            return True
        if not e['released']:
            check_holder(e['expected'], identity(e['expected']['pid']), e['gpu'],
                         uuid_for(e['gpu']), apps())
            e['restore_state'] = 'safe'
            return True
        if not select.select([e['fd']], [], [], 0)[0]:
            # An attempted SIGTERM may still be pending; do not unlock prematurely.
            if not e.get('original_wait_reported'):
                save('holder_exit_wait_gpu%d.json' % e['gpu'], dict(reason='original holder still alive',
                     lock_and_lease_retained=True, all_node_leases_heartbeat=True,
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
                         all_node_leases_heartbeat=True, managed_parent_wait=True))
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
                 unknown_jobs_not_signaled=True, all_node_leases_heartbeat=True,
                 lock_and_lease_retained=True, status='blocked'))
            e['restore_unknown_reported'] = True
        if time.monotonic()-e['restore_started'] > 120 and not e.get('restore_wait_reported'):
            save('restore_wait_gpu%d.json' % e['gpu'], dict(holder_pid=holder.pid,
                 lock_and_lease_retained=True, all_node_leases_heartbeat=True, no_duplicate_spawn=True))
            e['restore_wait_reported'] = True
        return False
    except BaseException as error:
        e['restore_state'] = 'blocked'
        save('restore_failure_gpu%d.json' % e['gpu'], dict(error=repr(error),
             unknown_jobs_not_signaled=True, lock_and_lease_retained=True,
             all_node_leases_heartbeat=True, no_duplicate_spawn=True))
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


def required_budget(cfg):
    return SAFETY + LOAD


def argv_option(argv, option, default=None):
    values = []
    for index, value in enumerate(argv):
        if value == option:
            if index + 1 >= len(argv):
                raise RuntimeError('missing argv option value: ' + option)
            values.append(argv[index + 1])
        elif value.startswith(option + '='):
            values.append(value.split('=', 1)[1])
    if len(values) > 1:
        raise RuntimeError('duplicate argv option: ' + option)
    return values[0] if values else default


def validate_training_argv(cfg, phase):
    argv = cfg[phase + '_argv']
    producer = cfg['producer']['path']
    if (argv[:3] != ['/opt/venv_music_planner_v1/bin/python', '-m', 'torch.distributed.run']
            or argv_option(argv, '--nproc_per_node') != '8' or argv_option(argv, '--nnodes', '1') != '1'
            or '--standalone' not in argv):
        raise RuntimeError('only standalone single-node eight-process local torchrun is admitted')
    options = argv[argv.index(producer) + 1:]
    required = {'--family': cfg['family'], '--model': str(stage.MODEL_BASE / cfg['family']),
                '--weights-manifest': str(stage.MODEL_BASE / (cfg['family'] + '.stage_receipt.json')),
                '--out': str(Path(cfg[phase + '_result']).parent), '--backend': 'deepspeed'}
    defaults = {'--epochs': '2', '--batch': '32', '--micro': '1', '--accum': '4', '--max-updates': '0'}
    required.update(defaults)
    if phase in ('smoke', 'resume_smoke'):
        required['--max-updates'] = '2' if phase == 'smoke' else '3'
    for option, expected in required.items():
        if argv_option(options, option, defaults.get(option)) != expected:
            raise RuntimeError(phase + ' argv does not match admitted recipe: ' + option)
    resume = argv_option(options, '--resume')
    if phase in ('resume_smoke', 'full'):
        previous = 2 if phase == 'resume_smoke' else 3
        if resume != str(Path(cfg['smoke_result']).parent / ('checkpoint-%07d' % previous)):
            raise RuntimeError('resume must name this run exact preceding admitted checkpoint')
    elif resume is not None:
        raise RuntimeError('initial smoke must start official base fresh')
    if '--check-data-only' in options:
        raise RuntimeError('fake GPU smoke is forbidden')
    if phase in ('smoke', 'resume_smoke') and '--no-verify-reload' not in options:
        raise RuntimeError('ZeRO3 smoke must restore in a new process, never same-engine reload')
    for split in ('train', 'val'):
        path = argv_option(options, '--' + split)
        if not path or path not in cfg['source_sha256']:
            raise RuntimeError('train/val argv requires exact SHA-bound manifest')
        if path != cfg['dataset'][split]['path']:
            raise RuntimeError('all phases must use the same full fixed2s train/val baseline')


def validate_config(cfg):
    stage.validate_scope(cfg['node'], cfg['family'], cfg['host'])
    gpus = cfg['gpus']
    if gpus != list(range(8)) or any(type(gpu) is not int for gpu in gpus):
        raise RuntimeError('only all eight physical GPUs 0..7 are authorized')
    if cfg.get('release_authorized') is not True:
        raise RuntimeError('explicit holder SIGTERM authorization required')
    expected = cfg['holders']
    if (len(expected) != 8 or [e['gpu'] for e in expected] != gpus
            or len({e['uuid'] for e in expected}) != 8 or len({e['pid'] for e in expected}) != 8):
        raise RuntimeError('eight ordered unique holder identities required')
    if cfg['env'].get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise RuntimeError('worker CVD must exactly match eight admitted GPUs')
    if not Path(cfg['cwd']).is_absolute():
        raise RuntimeError('absolute worker cwd required')
    for key in ('smoke_argv', 'resume_smoke_argv', 'full_argv'):
        if not cfg[key] or not all(isinstance(v, str) and v for v in cfg[key]):
            raise RuntimeError('explicit worker argv required: ' + key)
        if cfg['producer']['path'] not in cfg[key]:
            raise RuntimeError('argv does not name frozen producer: ' + key)
    queue, out = Path(cfg['queue']).resolve(), Path(cfg['out']).resolve()
    if (not queue.is_relative_to(EXPERIMENT / 'queues') or not out.is_relative_to(EXPERIMENT)
            or queue == out or queue.is_relative_to(out) or out.is_relative_to(queue)):
        raise RuntimeError('queue/output scope or overlap violation')
    for key in ('smoke_result', 'resume_smoke_result', 'full_result'):
        if not Path(cfg[key]).resolve().is_relative_to(out):
            raise RuntimeError('result outside owned new training output')
    if Path(cfg['smoke_result']).resolve() == Path(cfg['full_result']).resolve():
        raise RuntimeError('smoke/full outputs must differ')
    for split, count in (('train', 10713), ('val', 490)):
        if cfg['dataset'][split].get('count') != count:
            raise RuntimeError('fixed2s dataset exact count mismatch: ' + split)
    if len({Path(cfg[key]).parent for key in ('smoke_result', 'resume_smoke_result', 'full_result')}) != 1:
        raise RuntimeError('all three phases must use the same admitted run output directory')
    if Path(cfg['smoke_result']).parent == out:
        raise RuntimeError('trainer directory must be a new child of manager output root')
    for key, updates in (('smoke_result', 2), ('resume_smoke_result', 3), ('full_result', 670)):
        if Path(cfg[key]).name != 'outputs-u%07d.json' % updates:
            raise RuntimeError('phase must use immutable update-specific trainer output')
    for phase in ('smoke', 'resume_smoke', 'full'):
        validate_training_argv(cfg, phase)


def check_sources(cfg, config_path, config_sha, manager_sha):
    if sha(__file__) != manager_sha or sha(config_path) != config_sha:
        raise RuntimeError('manager or launch config SHA mismatch')
    bindings = dict(cfg['source_sha256'])
    producer = cfg['producer']
    if bindings.get(producer['path'], producer['sha256']) != producer['sha256']:
        raise RuntimeError('conflicting producer source binding')
    bindings[producer['path']] = producer['sha256']
    if str(Path(stage.__file__).resolve()) not in bindings:
        raise RuntimeError('stage_models.py must be in frozen source manifest')
    for split in ('train', 'val'):
        ref = cfg['dataset'][split]
        if bindings.get(ref['path'], ref['sha256']) != ref['sha256']:
            raise RuntimeError('conflicting dataset source binding')
        bindings[ref['path']] = ref['sha256']
    for name, digest in bindings.items():
        if not Path(name).is_absolute() or sha(name) != digest:
            raise RuntimeError('source/input changed: ' + name)
    for split in ('train', 'val'):
        ref = cfg['dataset'][split]
        with Path(ref['path']).open(encoding='utf-8-sig') as stream:
            count = sum(1 for line in stream if line.strip())
        if count != ref['count']:
            raise RuntimeError('dataset actual count changed: ' + split)


def check_ready(cfg):
    ref = cfg['code_ready']
    path = Path(ref['path']).resolve()
    if not path.is_relative_to(EXPERIMENT / 'runtime'):
        raise RuntimeError('code approval outside runtime')
    if not path.exists():
        return False
    if sha(path) != ref['sha256']:
        raise RuntimeError('code approval SHA changed')
    receipt = json.loads(path.read_text())
    if receipt.get('approved_for_gpu') is not True:
        return False
    for name, digest in cfg['source_sha256'].items():
        if receipt.get('source_sha256', {}).get(name) != digest:
            raise RuntimeError('coordinator approval missing frozen source: ' + name)
    return True


def progress(queue, phase, **value):
    payload = dict(phase=phase, utc=time.strftime('%FT%TZ', time.gmtime()), **value)
    temporary = queue / '.progress.tmp'
    with temporary.open('w') as stream:
        json.dump(payload, stream, indent=2)
    temporary.replace(queue / 'progress.json')
    try:
        print(json.dumps(payload), flush=True)
    except OSError:
        pass


def wait_dependencies(cfg, queue, check):
    deadline = time.monotonic() + cfg.get('wait_seconds', 7200)
    while True:
        check()
        code_ready = check_ready(cfg)
        try:
            stage.read_download(cfg['modelref'], cfg['family'])
            weights_ready = True
            reason = None
        except stage.WaitingWeights as error:
            weights_ready = False
            reason = str(error)
        if code_ready and weights_ready:
            progress(queue, 'ready_to_stage', no_holder_released=True, training_started=False)
            return
        progress(queue, 'waiting_weights' if not weights_ready else 'waiting_code',
                 reason=reason, code_ready=code_ready, weights_ready=weights_ready,
                 no_holder_released=True, training_started=False)
        if time.monotonic() >= deadline:
            raise RuntimeError('bounded dependency wait expired; no GPU holder released')
        time.sleep(cfg.get('poll_seconds', 10))


def check_model(cfg, marker):
    _, digest, _, _, files = stage.read_download(cfg['modelref'], cfg['family'])
    return stage.check_staged(marker, digest, cfg['family'], files, stage.MODEL_BASE / cfg['family'])


def capture_current(node, family):
    stage.validate_scope(node, family)
    running = apps()
    expected = []
    for gpu in range(8):
        uuid = uuid_for(gpu)
        selected = [row for row in running if row['uuid'] == uuid]
        if len(selected) != 1:
            raise RuntimeError('GPU%d lacks exactly one project holder; no signals sent' % gpu)
        current = identity(selected[0]['pid'])
        item = dict(current, uuid=uuid, gpu=gpu)
        check_holder(item, current, gpu, uuid, running)
        lease = lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))
        lock_path = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
        if lock_path.exists() or lock_path.is_symlink():
            st = lock_path.lstat()
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size:
                raise RuntimeError('unsafe GPU lock during capture')
            fd = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        item['previous_lease_mtime'] = lease.st_mtime if lease else None
        expected.append(item)
    # Fresh second pass closes the read-only sampling interval.
    for item in expected:
        check_holder(item, identity(item['pid']), item['gpu'], uuid_for(item['gpu']), apps())
    return dict(node=node, family=family, host=stage.AUTHORIZED[node][0], gpus=list(range(8)),
                holders=expected, captured_unix=time.time(), signals_sent=False,
                manager_identity=identity(os.getpid()), budget=headroom())


def run_child(cfg, argv, phase, entries, queue, save, owned):
    env = os.environ.copy()
    for name in ('PYTHONHOME', 'PYTHONPATH', 'CONDA_PREFIX'):
        env.pop(name, None)
    env.update(cfg['env'])
    runtime_log = queue / (phase + '.runtime.log')
    peak, gpu_peak = headroom()['current'], {g: 0 for g in cfg['gpus']}
    with runtime_log.open('xb') as log, (queue / (phase + '.resources.jsonl')).open('x') as samples:
        child = subprocess.Popen(argv, env=env, cwd=cfg['cwd'], stdout=log,
                                 stderr=subprocess.STDOUT, preexec_fn=reset_holder_signals)
        owned['child'] = child
        owned['started'] = True
        save(phase + '.live.json', dict(argv=argv, pid=child.pid, runtime_log=str(runtime_log),
                                       child_identity=identity(child.pid) if child.poll() is None else None))
        while child.poll() is None:
            refresh_leases(entries)
            memory = headroom()
            peak = max(peak, memory['current'])
            if memory['headroom'] < SAFETY and not (queue / 'low_memory_warning.json').exists():
                save('low_memory_warning.json', dict(memory=memory, no_other_training_should_start=True))
            snapshots = []
            for gpu in cfg['gpus']:
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
    owned['child'] = None
    save(phase + '.exit.json', dict(exit_code=code, cgroup_observed_peak=peak, gpu_peak_MiB=gpu_peak))
    if code != 0:
        raise RuntimeError(phase + ' worker exit ' + str(code))
    return code


def verified_ref(cfg, row):
    path = Path(row['path']).resolve()
    if not path.is_relative_to(Path(cfg['out']).resolve()) or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError('training artifact missing or outside owned output')
    if sha(path) != row['sha256'] or path.stat().st_size != row['size_bytes']:
        raise RuntimeError('training artifact SHA/size mismatch')
    return path


def check_training_result(cfg, phase, previous_result=None):
    path = Path(cfg[phase + '_result'])
    value = json.loads(path.read_text())
    if (value.get('schema') != 'music-planner-outputs-v1' or value.get('family') != cfg['family']
            or value.get('mode') != phase or value.get('world_size') != 8
            or value.get('test_selection_performed') is not False):
        raise RuntimeError('training result schema/family/mode/distributed/heldout mismatch')
    initial = 'official_base_then_matching_checkpoint' if phase != 'smoke' else 'official_base'
    if value.get('initialization') != initial:
        raise RuntimeError('training official-base initialization mismatch')
    expected = (10713, 490)
    if (value.get('train_count'), value.get('val_count')) != expected:
        raise RuntimeError('training result actual split counts mismatch')
    updates = value.get('optimizer_updates')
    start = 0 if phase == 'smoke' else 2 if phase == 'resume_smoke' else 3
    wanted_updates = 2 if phase == 'smoke' else 3 if phase == 'resume_smoke' else 670
    if (type(updates) is not int or updates != wanted_updates or value.get('starting_update') != start
            or value.get('invocation_optimizer_updates') != updates - start):
        raise RuntimeError('real optimizer updates or resume cursor mismatch')
    if (value.get('global_batch') != 32 or value.get('epoch_budget') != 2
            or value.get('source_exposure_budget') != 21426
            or value.get('source_exposures') != (21426 if phase == 'full' else wanted_updates * 32)):
        raise RuntimeError('same-baseline fair exposure budget mismatch')
    options = cfg[phase + '_argv'][cfg[phase + '_argv'].index(cfg['producer']['path']) + 1:]
    for split in ('train', 'val'):
        source = argv_option(options, '--' + split)
        if value.get('source_sha256', {}).get(split) != cfg['source_sha256'][source]:
            raise RuntimeError('training actual source SHA differs from admitted manifest')
    groups = value.get('gradient_groups', {})
    if (set(groups) != {'language_lora', 'audio_bridge'}
            or any(row.get('finite') is not True or row.get('nonzero') is not True for row in groups.values())):
        raise RuntimeError('nonzero finite language/audio gradients not demonstrated')
    if value.get('checkpoint_complete') is not True:
        raise RuntimeError('complete state save evidence missing')
    gradient_refs = value.get('gradient_evidence', [])
    if len(gradient_refs) != 8:
        raise RuntimeError('all eight rank backward evidence required')
    counts = dict(language_lora=0, audio_bridge=0)
    ranks = set()
    for ref in gradient_refs:
        evidence_path = verified_ref(cfg, ref)
        if (evidence_path.parent != path.parent or evidence_path.name not in
                {'backward-evidence-u%07d-rank%d.json' % (start, rank) for rank in range(8)}):
            raise RuntimeError('backward evidence outside exact invocation/rank')
        rank = evidence_path.name.removeprefix('backward-evidence-u%07d-rank' % start).removesuffix('.json')
        if rank not in {str(i) for i in range(8)} or rank in ranks:
            raise RuntimeError('backward rank inventory mismatch')
        ranks.add(rank)
        evidence = json.loads(evidence_path.read_text())
        if (evidence.get('starting_update') != start or evidence.get('actual_backward_observed') is not True
                or evidence.get('gradient_checkpointing_enabled') is not True
                or not sum(evidence.get('actual_checkpoint_calls', {}).values())):
            raise RuntimeError('real backward/gradient checkpointing evidence missing')
        norms = evidence.get('gradient_norms', {})
        if not norms:
            raise RuntimeError('empty actual gradient norms')
        for name, norm in norms.items():
            if isinstance(norm, bool) or not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm < 0:
                raise RuntimeError('nonfinite actual gradient evidence')
            audio = 'lora_' not in name or any(tag in name.lower() for tag in ('audio', 'acoustic', 'sound'))
            counts['audio_bridge' if audio else 'language_lora'] += int(norm > 0)
    if not all(counts.values()) or any(groups[k].get('observed_parameters') != counts[k] for k in counts):
        raise RuntimeError('reported nonzero gradients do not match raw backward evidence')
    checkpoint_path = verified_ref(cfg, value['checkpoint'])
    checkpoint = json.loads(checkpoint_path.read_text())
    if (checkpoint_path.name != 'trainer_state.json' or checkpoint.get('schema') != 'music-planner-train-v1'
            or checkpoint.get('complete') is not True or checkpoint.get('world_size') != 8
            or checkpoint.get('zero3') is not True or checkpoint.get('fingerprint') != value.get('fingerprint')
            or checkpoint.get('cursor', {}).get('update') != updates):
        raise RuntimeError('checkpoint fingerprint/ZeRO3/cursor mismatch')
    state_files = value.get('checkpoint_files', [])
    if not state_files or state_files != checkpoint.get('files'):
        raise RuntimeError('checkpoint whole-state manifest differs from actual marker')
    files = [verified_ref(cfg, row) for row in state_files]
    if len(set(files)) != len(files) or any(not file.is_relative_to(checkpoint_path.parent) for file in files):
        raise RuntimeError('checkpoint state file inventory mismatch')
    names = {file.name for file in files}
    if (not {'trainable.pt', 'scheduler.pt'}.issubset(names)
            or {file.relative_to(checkpoint_path.parent).as_posix() for file in files
                if file.name.endswith(('optim_states.pt', 'model_states.pt'))} !=
               {'distributed/zero3/zero_pp_rank_%d_mp_rank_00_%s_states.pt' % (rank, kind)
                for rank in range(8) for kind in ('optim', 'model')}
            or any('rng-rank%d.pt' % rank not in names or 'writer-rank%d.json' % rank not in names for rank in range(8))):
        raise RuntimeError('checkpoint lacks model/scheduler/eight optimizer partitions/RNG/writer state')
    trainable = verified_ref(cfg, value['trainable'])
    if trainable != checkpoint_path.parent / 'trainable.pt' or value['trainable']['sha256'] != checkpoint['trainable_sha256']:
        raise RuntimeError('portable trainable checkpoint digest mismatch')
    run_config_path = verified_ref(cfg, value['run_config'])
    run_config = json.loads(run_config_path.read_text())
    if run_config.get('fingerprint') != value.get('fingerprint') or run_config_path.parent != path.parent:
        raise RuntimeError('run config fingerprint or directory mismatch')
    if value.get('cursor') != checkpoint.get('cursor'):
        raise RuntimeError('outputs sampler cursor differs from actual checkpoint')
    if phase != 'smoke':
        if (value.get('resume_verified') is not True or value.get('verified_new_process') is not True
                or value.get('resumed_next_update_performed') is not True or previous_result is None
                or value.get('fingerprint') != previous_result['fingerprint']
                or value.get('resume_source_checkpoint') != str(Path(previous_result['checkpoint']['path']).parent)):
            raise RuntimeError('strict new-process resume and next real update not verified')
        refs = value.get('resume_evidence', [])
        if len(refs) != 8:
            raise RuntimeError('all eight rank new-process restore evidence required')
        ranks = set()
        for ref in refs:
            evidence_path = verified_ref(cfg, ref)
            if (evidence_path.parent != path.parent or evidence_path.name not in
                    {'resume-from-%07d-rank%d.json' % (start, rank) for rank in range(8)}):
                raise RuntimeError('resume evidence outside exact invocation/rank')
            rank = evidence_path.name.removeprefix('resume-from-%07d-rank' % start).removesuffix('.json')
            if rank not in {str(i) for i in range(8)} or rank in ranks:
                raise RuntimeError('resume rank inventory mismatch')
            ranks.add(rank)
            evidence = json.loads(evidence_path.read_text())
            optimizer = evidence.get('optimizer', {})
            writer, reader = evidence['writer'], evidence['reader']
            if (evidence.get('new_process') is not True or writer['nonce'] == reader['nonce']
                    or evidence.get('strict_trainable_verified') is not True
                    or evidence.get('scheduler_restored') is not True or evidence.get('rng_restored') is not True
                    or evidence.get('restored_cursor') != previous_result['cursor']
                    or optimizer.get('optimizer_state_sha256') != writer.get('optimizer', {}).get('optimizer_state_sha256')
                    or not optimizer.get('optimizer_state_sha256')
                    or evidence.get('source_checkpoint') != value['resume_source_checkpoint']
                    or not all(optimizer.get(k) is True for k in ('optimizer_state_initialized', 'moments_fp32', 'moments_finite'))):
                raise RuntimeError('raw strict optimizer/model/scheduler/RNG new-process restore evidence invalid')
    elif value.get('resume_verified') is not False or value.get('resume_source_checkpoint') is not None:
        raise RuntimeError('fresh official-base run unexpectedly resumed checkpoint')
    if phase == 'full':
        if (value.get('epochs_completed') != 2 or value.get('full_budget_completed') is not True
                or value.get('source_exposures') != 21426):
            raise RuntimeError('full two epochs/exposure budget not completed')
    return dict(result=value, result_path=str(path), result_sha256=sha(path))


def wait_gpu_cleanup(entries, queue, phase):
    uuids = {e['uuid'] for e in entries}
    deadline = time.monotonic() + 300
    while [r for r in apps() if r['uuid'] in uuids]:
        refresh_leases(entries)
        progress(queue, phase, locks_and_all_leases_retained=True, unknown_jobs_not_signaled=True)
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU cleanup pending; no next worker or unknown signal')
        time.sleep(5)


def finish_resources(entries, locks, queue, save, holders, owned):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cleanup_errors = set()
    def retained_error(e, error):
        key = e['gpu']
        if key not in cleanup_errors:
            save('lease_cleanup_failure_gpu%d.json' % key, dict(error=repr(error),
                 lock_retained=True, unknown_lease_not_modified=True))
            cleanup_errors.add(key)
    child = owned['child']
    if child is not None and child.poll() is None:
        child.terminate()  # Only this manager's direct torchrun child, never unknown process groups.
        deadline = time.monotonic() + 120
        while child.poll() is None:
            refresh_leases(entries, retained_error)
            if time.monotonic() > deadline and not (queue / 'child_cleanup_blocked.json').exists():
                save('child_cleanup_blocked.json', dict(pid=child.pid, unknown_jobs_not_signaled=True,
                     all_node_leases_heartbeat=True, lock_and_lease_retained=True))
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
    # Recheck every exact original/restored identity and sole compute PID immediately before unlock.
    for e in entries:
        if not restore_step(e, save, holders):
            while True:
                refresh_leases(entries, retained_error)
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
    save('resources_safe.json', dict(all_holders_safe=True, locks_released=True, leases_restored=True,
         managed_parent_wait=True, unknown_jobs_not_signaled=True, admitted_gpu_count=len(entries)))
    progress(queue, 'resources_safe', locks_released=True, leases_restored=True,
             managed_parent_wait=bool(holders), training_complete=(queue / 'full.complete.json').exists(),
             training_failed=(queue / 'failure.json').exists())
    for h, log in holders:
        h.wait()
        log.close()


def manage(cfg, config_path, config_sha, manager_sha):
    validate_config(cfg)
    queue, out = Path(cfg['queue']).resolve(), Path(cfg['out']).resolve()
    queue.mkdir(parents=True, exist_ok=False)
    if out.exists():
        raise RuntimeError('training output already exists; never overwrite or reuse smoke state')
    # Safety progress must never depend on a caller's pipe lifetime.
    logfd = os.open(queue / 'manager.stdout.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)
    os.close(logfd)
    def save(name, value):
        save_receipt(queue, name, value, 'planner_training')
    def check():
        check_sources(cfg, config_path, config_sha, manager_sha)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    locks, entries, holders = [], [], []
    owned = dict(child=None, started=False)
    try:
        wait_dependencies(cfg, queue, check)
        check()
        stage_deadline = time.monotonic() + cfg.get('wait_seconds', 7200)
        while True:
            try:
                marker = stage.stage_model(cfg['node'], cfg['family'], cfg['modelref'],
                                           lambda phase, value: progress(queue, phase, **value))
                break
            except BlockingIOError:
                progress(queue, 'waiting_local_stage', no_holder_released=True, training_started=False)
                if time.monotonic() >= stage_deadline:
                    raise RuntimeError('bounded local stage wait expired; no holder released')
                time.sleep(cfg.get('poll_seconds', 10))
                check()
        save('model_staged.json', marker)
        check_model(cfg, marker)
        check()
        needed = required_budget(cfg)
        budget = headroom()
        if budget['headroom'] < needed:
            save('pending_budget.json', dict(budget=budget, required_bytes=needed, no_holder_released=True))
            return
        for gpu, expected in zip(cfg['gpus'], cfg['holders']):
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
            e = dict(gpu=gpu, uuid=expected['uuid'], fd=fd, expected=expected, reserve=reserve,
                     old=old, leasefd=None, released=False, original_env=None, original_cwd=None)
            entries.append(e)
            if select.select([fd], [], [], 0)[0]:
                raise RuntimeError('expected holder already exited')
            current = identity(expected['pid'])
            check_holder(expected, current, gpu, uuid_for(gpu), apps())
            rawenv = (Path('/proc') / str(expected['pid']) / 'environ').read_bytes()
            e['original_env'] = dict(x.decode().split('=', 1) for x in rawenv.split(b'\0') if x)
            e['original_cwd'] = os.readlink(Path('/proc') / str(expected['pid']) / 'cwd')
            e['identity'] = current
            configure_local_restore(e, cfg)
        check()
        check_model(cfg, marker)
        budget = headroom()
        if budget['headroom'] < needed:
            raise RuntimeError('fresh headroom below admission; no holder released')
        save('admission.json', dict(node=cfg['node'], family=cfg['family'], gpus=cfg['gpus'],
             config_sha256=config_sha, manager_sha256=manager_sha, source_sha256=cfg['source_sha256'],
             manager_identity=identity(os.getpid()), budget=budget, required_bytes=needed,
             model_download_sha256=marker['download_receipt_sha256'],
             holders=[dict(gpu=e['gpu'], uuid=e['uuid'], identity=e['identity']) for e in entries]))
        for e in entries:
            acquire_lease(e)
        # Own all eight leases before signaling any exact holder.
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
            raise RuntimeError('driver/GPU remains busy; locks retained, unknown jobs not signaled')
        out.mkdir(parents=True, exist_ok=False)
        progress(queue, 'gpu_smoke', training_started=True, full_training_started=False)
        check()
        run_child(cfg, cfg['smoke_argv'], 'smoke', entries, queue, save, owned)
        check()
        first = check_training_result(cfg, 'smoke')
        save('smoke.saved.json', first)
        wait_gpu_cleanup(entries, queue, 'waiting_initial_smoke_gpu_cleanup')
        check()
        progress(queue, 'gpu_resume_smoke', initial_update_saved=True, full_training_started=False)
        run_child(cfg, cfg['resume_smoke_argv'], 'resume_smoke', entries, queue, save, owned)
        check()
        resumed = check_training_result(cfg, 'resume_smoke', first['result'])
        save('smoke.approved.json', resumed)
        wait_gpu_cleanup(entries, queue, 'waiting_resume_smoke_gpu_cleanup')
        progress(queue, 'full_training', smoke_true_updates_verified=True, training_started=True, full_training_started=True)
        check_model(cfg, marker)
        check()
        run_child(cfg, cfg['full_argv'], 'full', entries, queue, save, owned)
        check()
        save('full.complete.json', check_training_result(cfg, 'full', resumed['result']))
        progress(queue, 'restoring_holders', training_complete=True)
    except BaseException as error:
        save('failure.json', dict(error=repr(error), partial_outputs_preserved=True,
                                 worker_started=owned['started'], unknown_jobs_not_signaled=True))
        progress(queue, 'failed_restoring', error=repr(error), full_training_approved=False)
    finally:
        finish_resources(entries, locks, queue, save, holders, owned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--config-sha')
    ap.add_argument('--manager-sha')
    ap.add_argument('--capture-current', action='store_true')
    ap.add_argument('--node', choices=stage.AUTHORIZED)
    ap.add_argument('--family', choices=stage.MODELS)
    ap.add_argument('--out')
    args = ap.parse_args()
    if args.capture_current:
        if not args.node or not args.family or not args.out:
            ap.error('capture requires --node --family --out')
        path = Path(args.out).resolve()
        if not path.is_relative_to(EXPERIMENT):
            raise RuntimeError('capture output outside current experiment')
        capture = capture_current(args.node, args.family)
        with path.open('x') as stream:
            json.dump(capture, stream, indent=2)
        print(json.dumps(dict(path=str(path), sha256=sha(path), signals_sent=False)), flush=True)
        return
    if not args.config or not args.config_sha or not args.manager_sha:
        ap.error('run requires --config --config-sha --manager-sha')
    if sha(__file__) != args.manager_sha or sha(args.config) != args.config_sha:
        raise RuntimeError('manager or launch config SHA mismatch')
    cfg = json.loads(Path(args.config).read_text())
    manage(cfg, args.config, args.config_sha, args.manager_sha)


if __name__ == '__main__':
    main()
