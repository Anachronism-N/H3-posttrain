"""External TP4 fresh-empty native manager with PIDFD/FLOCK/lease lifecycle."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import stat
import subprocess
import sys
import time
from stage2_local_publication_v1 import write_once,checked,require

NODES=tuple('a100-extra-0'+str(i) for i in range(3,7))
IDENTITY=('pid','start_ticks','uid','cgroup','exe','argv')


def identity(pid):
    p=Path('/proc',str(pid));a=(p/'stat').read_text().rsplit(') ',1)[1].split()
    return dict(pid=pid,ppid=int(a[1]),start_ticks=int(a[19]),state=a[0],uid=p.stat().st_uid,cgroup=(p/'cgroup').read_text(),exe=os.readlink(p/'exe'),argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x])


def same(expected):
    try:i=identity(expected['pid']);return all(i[k]==expected[k] for k in IDENTITY) and i['state'] not in ('Z','X','x')
    except FileNotFoundError:return False


def apps():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);return [dict(uuid=a.split(',')[0].strip(),pid=int(a.split(',')[1])) for a in out.splitlines() if a.strip()]


def memory():
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    p=Path('/sys/fs/cgroup/memory')
    if (p/'memory.usage_in_bytes').exists():current=int((p/'memory.usage_in_bytes').read_text());limit=int((p/'memory.limit_in_bytes').read_text())
    else:
        p=Path('/sys/fs/cgroup');current=int((p/'memory.current').read_text());v=(p/'memory.max').read_text().strip();limit=int(v) if v!='max' else current+available
    return dict(current=current,headroom=min(available,limit-current))


def verify(cfg,path):
    root=Path(cfg['local_root']);require(cfg['node'] in NODES and cfg['owner']==cfg['node']+'-g0123-external50000' and cfg['gpus']==[0,1,2,3],'Exact external TP4 ownership required')
    require(root.is_absolute() and str(root).startswith('/root/stage2_external50000_') and root.resolve()==root and root.stat().st_uid==os.getuid(),'Owned stable node-local root required')
    require(cfg['expected_hostname']==subprocess.check_output(['hostname'],text=True).strip() and cfg['new_stage_authorized'] is True,'Bound explicit physical external node required')
    require(Path(cfg['out'])==root/'out' and Path(cfg['queue'])==root/'queue' and str(Path(path))==cfg['config_path'],'Exclusive local lifecycle paths required')
    for name,digest in cfg['code_sha256'].items():require(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'Frozen external source changed')
    require(cfg['source_count']==50000 and cfg['source_manifest_sha256']=='5a25521c6908ed3852ebba54242b79e718f73a01444ade4260ebfaebf39f2ce1','Exact external source reservation required')
    model=Path(cfg['model_path']);index=json.loads((model/'model.safetensors.index.json').read_bytes());require(model.is_absolute() and model.resolve()==model and (model/'config.json').is_file(),'Complete original local model required')
    require(all((model/n).is_file() for n in set(index['weight_map'].values())),'Original32B model shard missing')
    mc=json.loads((model/'config.json').read_bytes());require(mc.get('model_type')=='qwen3_vl','Original Qwen3-VL model required, no substitute')
    require(cfg['dtype']=='bfloat16' and cfg['quantization'] is None and cfg['tp']==4 and cfg['max_num_seqs']==32,'Original unquantized TP4 recipe required')


def gate(cfg):
    q=Path(cfg['queue']);ad=json.loads((q/'admission.json').read_bytes());mine=identity(os.getpid());parent=identity(os.getppid());require(all(parent[k]==ad['manager_identity'][k] for k in IDENTITY) and mine['uid']==parent['uid'] and mine['cgroup']==parent['cgroup'],'Exact admitted native parent required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3' and os.getcwd()=='/tmp','Actual external physical device/cwd binding differs')
    live=json.loads((q/'live.json').read_bytes());require(all(mine[k]==live['child_identity'][k] for k in IDENTITY),'Exact worker live identity required')
    require(hashlib.sha256(Path(cfg['config_path']).read_bytes()).hexdigest()==ad['config_sha256'],'Frozen admitted config differs')
    for gpu in cfg['gpus']:
        p=Path('/tmp/od_reserve_'+str(gpu));st=p.lstat();require(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_size==0 and time.time()-st.st_mtime<30,'Owned native lease heartbeat missing')
        lock=Path('/tmp/ms12k_gpu_'+str(gpu)+'.lock');ls=lock.stat();owned=False
        for line in Path('/proc/locks').read_text().splitlines():
            a=line.split()
            if len(a)<6 or a[1]!='FLOCK' or a[4]!=str(parent['pid']):continue
            x,y,z=a[5].split(':')
            if (int(x,16),int(y,16),int(z))==(os.major(ls.st_dev),os.minor(ls.st_dev),ls.st_ino):owned=True
        require(owned,'Exact parent kernel FLOCK missing')
    for a in apps():
        if a['uuid'] not in ad['gpu_uuids']:continue
        pid=a['pid']
        for _ in range(16):
            if pid==mine['pid']:break
            i=identity(pid);require(i['uid']==mine['uid'] and i['cgroup']==mine['cgroup'],'Foreign GPU process owner');pid=i['ppid']
        require(pid==mine['pid'],'Foreign GPU process ancestry')
    require(memory()['headroom']>=80*2**30,'Original active80GiB safety reserve required')


def worker(path):
    import stage2_local_model_worker_v1 as w
    import stage2_model_codec_v1 as codec
    cfg=json.loads(Path(path).read_bytes());verify(cfg,path);q=Path(cfg['queue']);deadline=time.monotonic()+20
    while not (q/'live.json').exists():require(time.monotonic()<deadline,'Native live receipt absent');time.sleep(.1)
    gate(cfg);codec.MODEL=cfg['model_path'];llm,sampler=codec.runtime();gate(cfg);engine=llm.llm_engine.vllm_config;require(str(engine.model_config.dtype)=='torch.bfloat16' and engine.model_config.quantization is None and engine.parallel_config.tensor_parallel_size==4 and engine.scheduler_config.max_num_seqs==32,'Actual original external TP4 recipe differs')
    root=Path(cfg['out']);state=Path(cfg['local_root'])/'worker_state'/cfg['owner'];write_once(state/'load.receipt.json',dict(model_init_count=1,loaded_unix=time.time(),dtype='bfloat16',quantization=None,tp=4,max_num_seqs=32,model_path=cfg['model_path']))
    outcomes=[];used=set();fq=root/'ready'/cfg['owner'];begin=time.time();idle=0
    while True:
        gate(cfg);refs=[]
        for p in sorted(fq.glob('*.ready.json')):
            if p.name in used or (root/'consumed_ready'/cfg['owner']/p.name).exists():continue
            refs.append(checked(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))['packet_ref'])
            if len(refs)==4:break
        failed=(fq/'FEEDER_FAILED').exists();drained=(fq/'FEEDER_DRAINED').exists()
        if not refs and (drained or failed):break
        if len(refs)<4 and not (drained or failed):time.sleep(.5);idle+=.5;continue
        outcomes+=w.run_packets(refs,cfg['out'],cfg['owner'],llm,sampler,lambda:gate(cfg))
        for r in refs:
            name=Path(r['path']).name+'.ready.json';used.add(name);write_once(root/'consumed_ready'/cfg['owner']/name,dict(packet_ref=r,source_outputs_published_locally=True,models_already_called=True))
        write_once(state/('progress%06d.json'%len(outcomes)),dict(processed=len(outcomes),outcomes=outcomes,idle_seconds=idle,observed_unix=time.time(),quality_approved=False,training_eligible=False))
    write_once(root/'generation_drained'/cfg['owner'],dict(owner=cfg['owner'],processed=len(outcomes),feeder_failed=failed,models_never_retried=True))
    write_once(state/'summary.json',dict(processed=len(outcomes),all_outcomes=outcomes,model_init_count=1,begin_unix=begin,end_unix=time.time(),idle_seconds=idle,feeder_failed=failed,quality_approved=False,training_eligible=False))


def start(path):
    cfg=json.loads(Path(path).read_bytes());verify(cfg,path);q=Path(cfg['queue']);require(not q.exists(),'Existing manager queue cannot be restarted');require(memory()['headroom']>=160*2**30 and shutil.disk_usage('/tmp').free>=8*2**30,'Fresh160GiB/tmp8GiB admission required');q.mkdir();fds=[];leases=[];child=None;known={};uuids=[];stop=[False];exitcode=None
    def stopping(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    def heartbeat():
        for p,fd,old,devino in leases:
            st=p.lstat();require((st.st_dev,st.st_ino)==devino,'Owned lease replaced; retain locks');os.utime(fd,None)
    def descendants():
        if child is None:return
        pending=[child.pid];seen=set()
        while pending:
            pid=pending.pop()
            if pid in seen:continue
            seen.add(pid)
            try:
                i=identity(pid)
                if i['uid']!=os.getuid() or i['cgroup']!=identity(os.getpid())['cgroup']:continue
                known[(pid,i['start_ticks'])]=i
                # Linux per-thread children cover multiprocessing engine workers.
                for t in Path('/proc',str(pid),'task').iterdir():pending.extend(int(x) for x in (t/'children').read_text().split())
            except (FileNotFoundError,ProcessLookupError):continue
    try:
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True);mapping={int(l.split(',')[0]):l.split(',')[1].strip() for l in inventory.splitlines()};uuids=[mapping[g] for g in cfg['gpus']]
        require(not [a for a in apps() if a['uuid'] in uuids],'Fresh external GPUs must be empty; unknown processes never signaled')
        for g in cfg['gpus']:
            p=Path('/tmp/ms12k_gpu_'+str(g)+'.lock');fd=os.open(p,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fds.append(fd);st=os.fstat(fd);require(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_size==0,'Owned kernel GPU lock required');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for g in cfg['gpus']:
            p=Path('/tmp/od_reserve_'+str(g));old=p.lstat() if p.exists() else None
            if old:require(stat.S_ISREG(old.st_mode) and old.st_uid==os.getuid() and old.st_size==0 and time.time()-old.st_mtime>=60,'Stale owned lease or absence required')
            fd=os.open(p,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);st=os.fstat(fd);leases.append((p,fd,old,(st.st_dev,st.st_ino)))
        heartbeat();require(memory()['headroom']>=160*2**30 and not [a for a in apps() if a['uuid'] in uuids],'Fresh atomic TP4 admission changed')
        write_once(q/'admission.json',dict(manager_identity=identity(os.getpid()),gpus=cfg['gpus'],gpu_uuids=uuids,config_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),node=cfg['node'],native_fresh_empty_admission=True,source_count=50000,observed_unix=time.time()))
        env=os.environ.copy();env.update(cfg['env']);argv=[cfg['python'],'-B',str(Path(__file__).resolve()),'worker','--config',path]
        with (q/'worker.runtime.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT);time.sleep(.1);write_once(q/'live.json',dict(manager_identity=identity(os.getpid()),child_identity=identity(child.pid),observed_unix=time.time()))
            while child.poll() is None and not stop[0]:
                heartbeat();descendants();write_once(q/('tree%09d.json'%int(time.monotonic()*100)),dict(known=list(known.values()),observed_unix=time.time()))
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            exitcode=child.poll()
            if stop[0] and not (Path(cfg['out'])/'FEEDER_STOP').exists():write_once(Path(cfg['out'])/'FEEDER_STOP',dict(native_manager_stop=True))
        write_once(q/'worker_exit.json',dict(exit_code=exitcode,stopped=stop[0],partial_outputs_preserved=True,models_not_retried=True))
    except BaseException as e:write_once(q/'failure.json',dict(error=repr(e),partial_outputs_preserved=True,unknown_jobs_not_signaled=True))
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);descendants();signaled=set();cleanup_blocked=False
        while True:
            try:heartbeat()
            except BaseException as e:
                cleanup_blocked=True
                if not (q/'lease_cleanup_blocked.json').exists():write_once(q/'lease_cleanup_blocked.json',dict(error=repr(e),locks_retained=True,unknown_lease_not_modified=True))
            descendants();alive=[]
            for k,i in list(known.items()):
                if not same(i):continue
                alive.append(i)
                if k not in signaled:
                    fd=os.pidfd_open(i['pid'])
                    try:
                        if same(i):signal.pidfd_send_signal(fd,signal.SIGTERM);signaled.add(k)
                    finally:os.close(fd)
            active=[a for a in apps() if a['uuid'] in uuids]
            if not alive and not active and not cleanup_blocked:break
            if active and not (q/'restore_blocked.json').exists():write_once(q/'restore_blocked.json',dict(apps=active,unknown_jobs_not_signaled=True,locks_and_leases_retained=True))
            time.sleep(5)
        for p,fd,old,devino in leases:
            try:
                require((p.lstat().st_dev,p.lstat().st_ino)==devino,'Lease replaced during cleanup, locks retained')
                if old is None:p.unlink()
                else:os.utime(fd,ns=(old.st_atime_ns,old.st_mtime_ns))
            except BaseException as e:
                write_once(q/('lease_restore_blocked_'+p.name+'.json'),dict(error=repr(e),locks_retained=True,unknown_lease_not_modified=True))
                while True:time.sleep(5)
            os.close(fd)
        for fd in fds:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
        write_once(q/'resources_safe.json',dict(all_holders_safe=True,locks_released=True,leases_restored=True,previous_empty_GPUs_restored_empty=True,unknown_jobs_not_signaled=True,source_claims_deleted=0,observed_unix=time.time()))

if __name__=='__main__':
    mode='manager'
    if len(sys.argv)>1 and sys.argv[1]=='worker':mode='worker';sys.argv.pop(1)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    if mode=='worker':worker(a.config)
    else:start(a.config)
