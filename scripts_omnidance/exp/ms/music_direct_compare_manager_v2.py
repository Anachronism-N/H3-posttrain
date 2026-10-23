"""Foreground queue manager; never detach, kill unknown jobs, or run inference."""
import argparse
import hashlib
import json
import os
import signal
import select
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
ENTRY = Path(__file__).with_name('music_direct_token_compare.py')
HOLDER = str(ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py')
PYTHON = '/opt/venv_music/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    temporary = path.with_name('.'+path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
    os.replace(temporary,path)


def identity(pid):
    base = Path('/proc') / str(pid)
    status = base.joinpath('status').read_text()
    uids = [int(x) for line in status.splitlines() if line.startswith('Uid:') for x in line.split()[1:]]
    env = base.joinpath('environ').read_bytes().split(b'\0')
    cuda = next((v.decode().split('=',1)[1] for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
    return dict(pid=pid, uids=uids, argv=[v.decode() for v in base.joinpath('cmdline').read_bytes().split(b'\0') if v],
                start_ticks=base.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],
                cuda=cuda,cgroup=base.joinpath('cgroup').read_text())


def smi(query):
    return [line.split(', ') for line in subprocess.check_output(
        ['nvidia-smi',query,'--format=csv,noheader,nounits'],text=True).splitlines() if line.strip()]


def snapshot():
    gpus = smi('--query-gpu=index,uuid,memory.total,memory.used')
    apps = smi('--query-compute-apps=pid,gpu_uuid,used_memory')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))*1024
    cg = Path('/sys/fs/cgroup/memory')
    limit, used = [int((cg / name).read_text()) for name in ('memory.limit_in_bytes', 'memory.usage_in_bytes')]
    return dict(gpus=gpus,apps=apps,memavailable_bytes=available,hostname=socket.gethostname(),time=time.time(),
                cgroup=Path('/proc/self/cgroup').read_text(), cgroup_headroom_bytes=limit-used)


def cache_ready(queue):
    plan = json.loads((queue / 'plan.json').read_text())
    command = plan['command']
    source = command[command.index('--output_path') + 1]
    live = json.loads((queue / 'live.json').read_text())
    manager = live['manager_pid']
    if Path('/proc', str(manager)).exists():
        value = identity(manager)
        if str(MS / 'h3_ms_style_cache_manager_v1.py') not in value['argv']:
            raise RuntimeError('Old cache PID identity changed; investigate, no signal')
        return False, dict(reason='old_cache_manager_still_alive', pid=manager)
    for base in Path('/proc').iterdir():
        if not base.name.isdigit():
            continue
        try:
            argv = [v.decode() for v in base.joinpath('cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if source in argv and 'sft:data_process' in argv:
            return False, dict(reason='old_cache_rank_or_launcher_still_alive', pid=int(base.name))
    finished_path = queue / 'finished.json'
    previous = json.loads(finished_path.read_text()) if finished_path.is_file() else None
    return True, dict(reason='old_cache_processes_exited_no_data_dependency', previous_terminal=previous,
                      cache_success_required=False, teacher_source='existing_direct_raw_Y_manifest')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manager_sha', required=True)
    p.add_argument('--node',choices=['N2','N3'],required=True)
    p.add_argument('--arm',choices=['mean','token'],required=True)
    p.add_argument('--queue_out',type=Path,required=True)
    p.add_argument('--train_out',type=Path,required=True)
    p.add_argument('--cpu_report',type=Path,required=True)
    p.add_argument('--moment_weight',type=float,required=True)
    p.add_argument('--moment_source',type=Path,required=True)
    p.add_argument('--cache_queue',type=Path)
    p.add_argument('--expected_holder_pid',type=int)
    p.add_argument('--expected_holder_start_ticks')
    p.add_argument('--expected_gpu_uuid',required=True)
    args = p.parse_args()
    if sha(__file__) != args.manager_sha:
        raise ValueError('Manager source identity mismatch')
    if args.queue_out.exists() or args.train_out.exists():
        raise FileExistsError('Fresh dedicated queue/train output required')
    args.queue_out.mkdir()
    plan = dict(node=args.node,arm=args.arm,gpu_index=0,expected_uuid=args.expected_gpu_uuid,
        train_out=str(args.train_out),cpu_report=str(args.cpu_report),manager_pid=os.getpid(),
        manager_source_sha256=sha(__file__),entry_source_sha256=sha(ENTRY),
        moment_weight=args.moment_weight,moment_original_plan=str(args.moment_source),
        moment_original_plan_sha256=sha(args.moment_source),
        expected_holder_pid=args.expected_holder_pid,expected_holder_start_ticks=args.expected_holder_start_ticks,
        cache_queue=str(args.cache_queue) if args.cache_queue else None,
        lifecycle='managed_Bash_SSH_foreground_full_queue_no_detach',inference='N4_only_not_here')
    original = json.loads(args.moment_source.read_text())
    oldweight = original.get('configuration',{}).get('moment_weight',original.get('moment_weight'))
    if oldweight != args.moment_weight:
        raise ValueError('Original direct corresponding moment weight differs')
    save(args.queue_out/'plan.json',plan)
    if args.node=='N3':
        if not args.cache_queue:
            raise ValueError('N3 must wait for preexisting cache queue')
        while True:
            ready,evidence = cache_ready(args.cache_queue)
            save(args.queue_out/'waiting.json',dict(waiting=not ready,time=time.time(),**evidence))
            print(json.dumps(dict(event='cache_dependency_wait',ready=ready,**evidence)),flush=True)
            if ready:
                break
            time.sleep(60)
    while not args.cpu_report.is_file():
        save(args.queue_out/'waiting_cpu.json',dict(waiting=True,time=time.time(),
             reason='actual_new_CPU_feature_report_pending_no_GPU_release',cpu_report=str(args.cpu_report)))
        print(json.dumps(dict(event='cpu_acceptance_wait',path=str(args.cpu_report))),flush=True)
        time.sleep(15)
    save(args.queue_out/'waiting_cpu.json',dict(waiting=False,time=time.time(),cpu_report=str(args.cpu_report)))
    tested = json.loads(args.cpu_report.read_text())
    if (tested.get('complete') is not True or tested.get('protocol') != 'direct_mean_vs_token_soft_chamfer_100updates_v1'
            or tested.get('moment_weight') != args.moment_weight
            or tested.get('source_fingerprints',{}).get('sources',{}).get(str(ENTRY.resolve())) != sha(ENTRY)):
        raise RuntimeError('Actual new entry CPU acceptance absent or source changed; no holder release')
    resource = snapshot()
    gpu0 = next(row for row in resource['gpus'] if row[0]=='0')
    if gpu0[1]!=args.expected_gpu_uuid or resource['memavailable_bytes']<64*1024**3 or resource['cgroup_headroom_bytes']<64*1024**3:
        raise RuntimeError('Fresh GPU UUID/host memory differs from allowed single GPU')
    active = [row for row in resource['apps'] if row[1]==args.expected_gpu_uuid]
    if active:
        if len(active)!=1:
            raise RuntimeError('Unknown/multiple GPU0 workloads; do not terminate')
        pid = int(active[0][0])
        holder = identity(pid)
        if (holder['uids']!=[0,0,0,0] or len(holder['argv'])!=2
                or not holder['argv'][0].endswith('/envs/longlive/bin/python')
                or holder['argv'][1]!=HOLDER or holder['cuda']!='0' or holder['cgroup']!=resource['cgroup']
                or int(active[0][2])>1200):
            raise RuntimeError('GPU0 workload is not exact allowed project holder')
        if args.node=='N2' and (pid!=args.expected_holder_pid or holder['start_ticks']!=args.expected_holder_start_ticks):
            raise RuntimeError('N2 authorized exact holder PID/starttick changed')
        descriptor = os.pidfd_open(pid)
        try:
            if identity(pid)!=holder:
                raise RuntimeError('Holder identity changed after pidfd_open')
            save(args.queue_out/'resource_release.json',dict(snapshot=resource,holder=holder,
                released_uuid=args.expected_gpu_uuid,signal='pidfd_SIGTERM_only_exact_project_GPU0_holder'))
            signal.pidfd_send_signal(descriptor,signal.SIGTERM)
            if not select.select([descriptor], [], [], 30)[0]:
                raise RuntimeError('Exact holder did not exit; no escalation')
        finally:
            os.close(descriptor)
        time.sleep(3)
    final_resource = snapshot()
    if any(row[1]==args.expected_gpu_uuid for row in final_resource['apps']):
        raise RuntimeError('GPU0 not free after allowed holder release; no unknown kills')
    if sha(__file__) != args.manager_sha:
        raise ValueError('Manager source changed before launch')
    save(args.queue_out/'fresh_resource.json',final_resource)
    reserve = Path('/tmp/od_reserve_0')
    assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == os.getuid() and reserve.stat().st_size == 0)
    reserve.touch()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    command = [PYTHON,str(ENTRY),'--arm',args.arm,'--init',str(MS/'music_direct_public_init_v1.pt'),
        '--manifest',str(MS/'music_direct6k_v2_training_manifest_v2.json'),'--out',str(args.train_out),
        '--moment_weight',str(args.moment_weight),'--cpu_report',str(args.cpu_report)]
    save(args.queue_out/'launch.json',dict(command=command,time=time.time()))
    with (args.queue_out/'train.log').open('x') as stream:
        child = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save(args.queue_out/'child.json',dict(pid=child.pid,identity=identity(child.pid)))
        while True:
            try:
                code = child.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                reserve.touch()
    finish = args.train_out/'finished.json'
    complete = code==0 and finish.is_file() and json.loads(finish.read_text()).get('complete') is True
    save(args.queue_out/'finished.json',dict(exit_code=code,complete=complete,
        training_finished=str(finish),inference_started=False))
    if not complete:
        raise RuntimeError('Mapping training incomplete; partial outputs preserved')


if __name__=='__main__':
    main()
