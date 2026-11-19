"""Fresh empty-pair ownership handoff; never signals an existing process."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import ms_fullframe_vlm_postprocess_batch_v6 as v

SELF=Path(__file__).resolve()
RT,ROOT=v.RT,v.ROOT
TMP=Path('/tmp')
HOST='28.216.19.69'
PAIRS=((0,1),(2,3),(4,5),(6,7))

def need(condition,message):
    v.need(condition,message)

def check(job,job_ref):
    need(job.get('schema')=='ms_vlm_fresh_empty_pair_job_v1' and job.get('node')=='N1' and
         tuple(job.get('gpus',[])) in PAIRS and all(type(g) is int for g in job['gpus']), 'Explicit N1 empty pair required')
    need(job.get('fresh_pair_authorized') is True and job.get('may_signal_existing_processes') is False,
         'Fresh allocation only, no existing process signals')
    need(isinstance(job.get('phase'),str) and re.fullmatch('ms_vlm_fresh_seed_n1g[0-7][0-7]_[A-Za-z0-9_-]+',job['phase']), 'Independent ownership phase required')
    need(job['phase'].startswith('ms_vlm_fresh_seed_n1g'+''.join(map(str,job['gpus']))+'_'), 'Ownership phase and GPU pair differ')
    need(v.base.sha(v.SELF)=='2fe2ea66caebb7a8f195c173eb34209f85b7aa33dc589c74ec0df95a35dca23b', 'Frozen helper dependency changed')
    need(job.get('producer_ref')==v.ref(SELF) and job_ref['path']==str(RT/(job['phase']+'_job.json')), 'Frozen ownership job/source binding required')
    need(v.checked(job_ref)==job and job.get('holder_ref')==v.ref(v.base.HOLDER), 'Frozen job and existing holder source required')
    need(HOST in subprocess.check_output(['hostname','-I'],text=True).split(), 'Wrong physical allocation node')
    need(v.base.sha(v.base.__file__)==v.BASE_SHA, 'Original native lifecycle changed')
    need(v.actual_controller_headroom()['headroom']>=v.base.required_budget({'gpus':job['gpus']}), 'Original 160GiB node admission required')
    uuids=[v.base.uuid_for(g) for g in job['gpus']]
    need(len(set(uuids))==2, 'Distinct physical devices required')
    need(not [a for a in v.base.apps() if a['uuid'] in uuids], 'Empty pair is currently occupied')
    return uuids

def provision(job_path,digest):
    reference=dict(path=str(Path(job_path)),sha256=digest)
    job=v.checked(reference);uuids=check(job,reference)
    queue=ROOT/'queues'/job['phase'];queue.mkdir(parents=True,exist_ok=False)
    locks,entries,children=[],[],[]
    succeeded=False
    try:
        for gpu,uuid in zip(job['gpus'],uuids):
            lockfd=os.open(TMP/('ms12k_gpu_%d.lock'%gpu),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
            locks.append(lockfd);st=os.fstat(lockfd)
            need(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_size==0,'Unsafe existing GPU lock')
            fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            reserve=TMP/('od_reserve_%d'%gpu);old=v.base.lease_snapshot(reserve)
            entries.append(dict(gpu=gpu,uuid=uuid,reserve=reserve,old=old,leasefd=None))
        need(check(job,reference)==uuids,'Physical GPU identity changed during admission')
        for entry in entries:v.base.acquire_lease(entry)
        v.write_once(queue/'fresh_admission.json',dict(job_ref=reference,phase=job['phase'],node='N1',gpus=job['gpus'],uuids=uuids,
            manager_identity=v.base.identity(os.getpid()),initial_gpu_apps_empty=True,original_budget_bytes=160*2**30,
            all_pair_kernel_locks_owned=True,all_pair_leases_owned=True,source_claims_created=0,vlm_calls=0,unknown_process_signals=0))
        for gpu in job['gpus']:
            env=os.environ.copy()
            for key in ('PYTHONHOME','PYTHONPATH','CONDA_PREFIX'):env.pop(key,None)
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),OCCUPY_GPU='0',OMP_NUM_THREADS='2')
            log=(queue/('fresh_holder_gpu%d.log'%gpu)).open('xb')
            child=subprocess.Popen(v.base.RESTORE_ARGV,env=env,cwd='/tmp',stdout=log,stderr=subprocess.STDOUT,preexec_fn=v.base.reset_holder_signals)
            children.append((gpu,child,log))
        deadline=time.monotonic()+180
        while True:
            v.base.refresh_leases(entries)
            running=v.base.apps()
            owned=[a for a in running if a['uuid'] in uuids]
            expected={uuid:child.pid for uuid,(gpu,child,log) in zip(uuids,children)}
            need(all(a['pid']==expected[a['uuid']] for a in owned),'Unexpected process appeared; no unknown signals')
            need(all(child.poll() is None for gpu,child,log in children),'Owned holder exited, never restart')
            if len(owned)==2:break
            need(time.monotonic()<deadline,'Owned holder initialization incomplete; no automatic retry')
            time.sleep(2)
        for uuid,(gpu,child,log) in zip(uuids,children):
            identity=v.base.identity(child.pid);holder=dict(identity,gpu=gpu,uuid=uuid)
            v.base.check_holder(holder,v.base.identity(child.pid),gpu,v.base.uuid_for(gpu),v.base.apps())
            v.write_once(queue/('holder_restored_gpu%d.json'%gpu),dict(schema='ms_vlm_fresh_owned_holder_handoff_v1',phase=job['phase'],gpu=gpu,uuid=uuid,
                holder_identity=identity,created_from_verified_empty_pair=True,previous_task_holder_restored=False,unknown_jobs_not_signaled=True))
        for entry in entries:
            if entry['leasefd'] is not None:
                v.base.refresh_leases([entry])
                if entry['old'] is None:entry['reserve'].unlink()
                else:os.utime(entry['leasefd'],ns=(entry['old'].st_atime_ns,entry['old'].st_mtime_ns))
                os.close(entry['leasefd']);entry['leasefd']=None
        for fd in locks:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
        locks=[]
        v.write_once(queue/'resources_safe.json',dict(phase=job['phase'],all_holders_safe=True,locks_released=True,leases_restored=True,
            managed_parent_wait=True,unknown_jobs_not_signaled=True,fresh_owned_pair_handoff=True,no_existing_task_interrupted=True,
            source_claims_created=0,vlm_calls=0))
        succeeded=True
        print(v.h.bench.encoded(dict(event='fresh_empty_pair_owned_handoff',phase=job['phase'],queue=str(queue),gpu_count=2,source_claims=0,vlm_calls=0)).decode(),flush=True)
        for gpu,child,log in children:child.wait();log.close()
    finally:
        if not succeeded:
            for gpu,child,log in children:
                if child.poll() is None:child.terminate()
            deadline=time.monotonic()+180
            while any(child.poll() is None for gpu,child,log in children) or [a for a in v.base.apps() if a['uuid'] in uuids]:
                v.base.refresh_leases(entries)
                if time.monotonic()>deadline and not (queue/'cleanup_blocked.json').exists():
                    v.write_once(queue/'cleanup_blocked.json',dict(lock_retained=True,lease_heartbeat=True,unknown_jobs_not_signaled=True))
                time.sleep(2)
            for entry in entries:
                if entry.get('leasefd') is not None:
                    v.base.refresh_leases([entry])
                    if entry['old'] is None:entry['reserve'].unlink()
                    else:os.utime(entry['leasefd'],ns=(entry['old'].st_atime_ns,entry['old'].st_mtime_ns))
                    os.close(entry['leasefd']);entry['leasefd']=None
            for fd in locks:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
            for gpu,child,log in children:log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--job-sha',required=True);args=p.parse_args();provision(args.job,args.job_sha)
