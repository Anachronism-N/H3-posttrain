"""Managed foreground N2 single-GPU launch, exact holder-only release."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
from music_direct_compare_manager_v3 import identity, snapshot, sha, HOLDER, MS, PYTHON

ENTRY = Path(__file__).with_name('music_direct_token_full6k.py')
UUID = 'GPU-d89362d8-8b22-575c-2be3-86eba53ac29c'
PROTOCOL = 'direct_token_full6000_val500_epochs3_v1'


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def no_existing_mapping():
    conflicts = []
    names = {'music_direct_token_compare.py', 'music_direct_condition_diag.py',
             'music_direct_token_full6k.py', 'music_direct_compare_manager_v3.py'}
    for base in Path('/proc').iterdir():
        if not base.name.isdigit() or int(base.name)==os.getpid():
            continue
        try:
            process = identity(int(base.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        is_own_cpu = (len(process['argv']) > 2 and process['argv'][0]==PYTHON
            and Path(process['argv'][1]).name=='music_direct_token_full6k.py'
            and '--cpu_test' in process['argv'] and process['uids']==[0,0,0,0]
            and process['cuda'] in (None, '') and any(argument in (str(MS/'music_direct_full6k_cpu_acceptance_v1.json'),
                str(MS/'music_direct_full6k_cpu_acceptance_v2.json')) for argument in process['argv']))
        if any(Path(argument).name in names for argument in process['argv']) and not is_own_cpu:
            conflicts.append(process)
    if conflicts:
        raise RuntimeError('Existing mapping/diagnostic processes; no signal: '+json.dumps(conflicts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manager_sha', required=True)
    parser.add_argument('--cpu_report', type=Path, required=True)
    parser.add_argument('--queue_out', type=Path, required=True)
    parser.add_argument('--train_out', type=Path, required=True)
    parser.add_argument('--holder_pid', type=int, required=True)
    parser.add_argument('--holder_start_ticks', required=True)
    args = parser.parse_args()
    lock = Path('/tmp/music_direct_full6k_n2_gpu0.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if sha(__file__) != args.manager_sha:
        raise ValueError('Manager source identity changed')
    tested = json.loads(args.cpu_report.read_text())
    if (tested.get('complete') is not True or tested.get('protocol') != PROTOCOL
            or tested.get('updates') != 2250 or tested.get('moment_weight') != 1.
            or tested['source_fingerprints']['sources'][str(ENTRY.resolve())] != sha(ENTRY)):
        raise ValueError('Actual new CPU feature acceptance missing/mismatched')
    for source, expected in tested['source_fingerprints']['sources'].items():
        if sha(source) != expected:
            raise ValueError('Accepted source changed: '+source)
    if args.queue_out.exists() or args.train_out.exists():
        raise FileExistsError('Fresh dedicated full6k manager/train output required')
    args.queue_out.mkdir()
    save(args.queue_out/'plan.json',dict(protocol=PROTOCOL,manager_pid=os.getpid(),manager_sha256=args.manager_sha,
        entry_sha256=sha(ENTRY),cpu_report=str(args.cpu_report),node='N2',gpu_index=0,gpu_uuid=UUID,
        train_out=str(args.train_out),updates=2250,train=6000,val=500,epochs=3,global_batch=8,
        lifecycle='managed_Bash_SSH_foreground_no_detach',video='N4_only_not_here'))
    no_existing_mapping()
    resource = snapshot()
    gpu = next(row for row in resource['gpus'] if row[0]=='0')
    if (gpu[1]!=UUID or resource['memavailable_bytes']<64*1024**3
            or resource['cgroup_headroom_bytes']<64*1024**3):
        raise RuntimeError('Fresh N2 GPU UUID or cgroup/host memory outside authorized budget')
    reserve = Path('/tmp/od_reserve_0')
    assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink()
        and reserve.stat().st_uid==os.getuid() and reserve.stat().st_size==0)
    reserve.touch()
    active = [row for row in resource['apps'] if row[1]==UUID]
    if active:
        if len(active)!=1:
            raise RuntimeError('Unknown/multiple GPU0 jobs; no signal')
        pid = int(active[0][0]); holder = identity(pid)
        if (pid!=args.holder_pid or holder['start_ticks']!=args.holder_start_ticks
                or holder['uids']!=[0,0,0,0] or len(holder['argv'])!=2
                or not holder['argv'][0].endswith('/envs/longlive/bin/python')
                or holder['argv'][1]!=HOLDER or holder['cuda']!='0'
                or holder['cgroup']!=resource['cgroup'] or int(active[0][2])>1200):
            raise RuntimeError('Not exact freshly authorized project holder; no signal')
        fd = os.pidfd_open(pid)
        try:
            if identity(pid)!=holder:
                raise RuntimeError('Holder identity changed after pidfd_open; no signal')
            save(args.queue_out/'resource_release.json',dict(snapshot=resource,holder=holder,
                signal='pidfd_SIGTERM_only_exact_confirmed_project_GPU0_holder'))
            signal.pidfd_send_signal(fd,signal.SIGTERM)
            if not select.select([fd],[],[],30)[0]:
                raise RuntimeError('Holder did not exit; no escalation')
        finally:
            os.close(fd)
        time.sleep(3)
    fresh = snapshot()
    if any(row[1]==UUID for row in fresh['apps']):
        raise RuntimeError('GPU0 still occupied; no unknown process operation')
    no_existing_mapping()
    save(args.queue_out/'fresh_resource.json',fresh)
    if sha(__file__) != args.manager_sha or sha(ENTRY)!=tested['source_fingerprints']['sources'][str(ENTRY.resolve())]:
        raise ValueError('Source changed before actual GPU launch')
    command = [PYTHON,str(ENTRY),'--init',str(MS/'music_direct_public_init_v1.pt'),
        '--manifest',str(MS/'music_direct6k_v2_training_manifest_v2.json'),
        '--out',str(args.train_out),'--moment_weight','1','--cpu_report',str(args.cpu_report)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    save(args.queue_out/'launch.json',dict(command=command,time=time.time(),cuda='0'))
    with (args.queue_out/'train.log').open('x') as stream:
        child = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save(args.queue_out/'child.json',dict(pid=child.pid,identity=identity(child.pid)))
        while True:
            try:
                code = child.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                reserve.touch()
    terminal = args.train_out/'finished.json'
    complete = code==0 and terminal.is_file() and json.loads(terminal.read_text()).get('complete') is True
    save(args.queue_out/'finished.json',dict(exit_code=code,complete=complete,training_finished=str(terminal),
        inference_started=False))
    if not complete:
        raise RuntimeError('Full6k training partial/failed; outputs preserved')


if __name__=='__main__':
    main()
