"""Read-only external GPU/process evidence. Never claims, allocates, stops or resumes a GPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

NODES = {'a100-extra-%02d' % i for i in range(1, 7)} | {'N2', 'N3', 'N6'}
MARKERS = ('ms_unified', 'ms_a100', 'fullframe_pilot', 'unified_resume', 'run_pilot', 'motionscript', 'model_extract_reused')

def identity(pid, argv=False):
    root=Path('/proc')/str(pid)
    fields=(root/'stat').read_text().rsplit(') ',1)[1].split()
    env=(root/'environ').read_bytes().split(bytes([0]))
    value=dict(pid=pid,ppid=int(fields[1]),start_ticks=int(fields[19]),state=fields[0],uid=root.stat().st_uid,
               exe=os.readlink(root/'exe'),cgroup=(root/'cgroup').read_text().strip(),
               CUDA_VISIBLE_DEVICES=[x.decode(errors='replace') for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES=')],
               wchan=(root/'wchan').read_text().strip())
    if argv:value['argv']=[x.decode(errors='replace') for x in (root/'cmdline').read_bytes().split(bytes([0])) if x]
    return value

def gpu_rows(command):
    result=subprocess.run(command,capture_output=True,text=True,timeout=20)
    if result.returncode:raise RuntimeError('GPU query failed: '+result.stderr[-500:])
    return [tuple(x.strip() for x in line.split(',')) for line in result.stdout.splitlines() if line.strip()]

def sample(gpus):
    rows=gpu_rows(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
    devices=[dict(index=int(i),uuid=u,used_MiB=int(m),utilization_pct=int(t)) for i,u,m,t in rows if int(i) in gpus]
    if {r['index'] for r in devices}!=set(gpus):raise ValueError('Requested physical GPU missing')
    uuids={r['uuid'] for r in devices}
    compute=gpu_rows(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
    pipelines={}
    for root in Path('/proc').iterdir():
        if not root.name.isdigit() or int(root.name)==os.getpid():continue
        try:
            args=[x.decode(errors='replace') for x in (root/'cmdline').read_bytes().split(bytes([0])) if x]
            if '-c' in args or not any(any(marker in x for marker in MARKERS) for x in args):continue
            if not any('python' in Path(x).name.lower() or x.endswith('.py') for x in args):continue
            pipelines[int(root.name)]=identity(int(root.name),True)
        except OSError:continue
    apps=[]
    for uuid,pid,memory in compute:
        if uuid not in uuids:continue
        pid=int(pid);entry=dict(uuid=uuid,pid=pid,used_MiB=memory,verified_pipeline_ancestor=None)
        current=pid;chain=[]
        try:
            for _ in range(20):
                item=identity(current,current in pipelines);chain.append(item)
                if current in pipelines:entry['verified_pipeline_ancestor']=current;break
                if item['ppid']<=1:break
                current=item['ppid']
            entry['ancestry']=chain
        except OSError as error:entry['identity_error']=repr(error)
        apps.append(entry)
    locks=[]
    for row in Path('/proc/locks').read_text().splitlines():
        if 'FLOCK' in row:
            owner=int(row.split()[4])
            if owner in pipelines:locks.append(row)
    return dict(observed_UTC=datetime.now(timezone.utc).isoformat(),observed_unix=time.time(),
                devices=devices,compute_apps=apps,pipeline_processes=list(pipelines.values()),pipeline_kernel_FLOCKs=locks,
                resource_start_or_release_authorization=False)

def publish(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True)+'\n').encode()
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.live-',delete=False) as stream:
        temporary=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    return hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--node',required=True,choices=sorted(NODES));parser.add_argument('--gpus',nargs='+',type=int,required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--interval',type=int,default=60);parser.add_argument('--samples',type=int,default=2)
    a=parser.parse_args()
    if len(set(a.gpus))!=len(a.gpus) or any(x<0 or x>7 for x in a.gpus) or not 1<=a.samples<=10 or a.interval<1:parser.error('Invalid finite observation scope')
    value=dict(node_label_is_caller_supplied=True,schema='ms_external_live_snapshot_v1',node=a.node,observer_pid=os.getpid(),hostname=os.uname().nodename,samples=[],
               scope='live physical-GPU query plus PID/start-ticks/ancestry; not input-data or quality approval',no_model_calls=True,no_process_signals=True)
    for i in range(a.samples):
        value['samples'].append(sample(a.gpus));sha=publish(a.out,value)
        print(json.dumps(dict(node=a.node,sample=i+1,output=a.out,sha256=sha,compute_processes=len(value['samples'][-1]['compute_apps']))),flush=True)
        if i+1<a.samples:time.sleep(a.interval)

if __name__=='__main__':main()
