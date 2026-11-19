"""Finite tail dispatcher: waits for native-safe predecessor, never interrupts it."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tempfile

RT=Path(__file__).resolve().parent
B=RT.parent
sys.path.insert(0,str(RT))
import ms_unified_native_continuous_manager_v2 as manager

def read(p):
    with Path(p).open('rb',buffering=0) as f:
        value=json.loads(f.read(8*1024*1024))
    return value

def write_new(p,value):
    p=Path(p)
    raw=(json.dumps(value,sort_keys=True,indent=2)+chr(10)).encode()
    with p.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())

def update_registry(job):
    path=RT/'ms_novlm_continuous_jobs_registry_v1.json'
    with path.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=read(path)
        matching=[i for i,item in enumerate(value['jobs']) if item['node']==job['node'] and item['gpu']==job['gpu']]
        if len(matching)!=1:raise RuntimeError('Registry does not contain exactly one owned slot')
        original=value['jobs'][matching[0]]
        if original['config_path']!=job['previous_config']:raise RuntimeError('Slot already handed over by another dispatcher')
        value['jobs'][matching[0]]=dict(node=job['node'],gpu=job['gpu'],host=job['host'],phase=job['phase'],
                                      config_path=str(RT/(job['phase']+'_config.json')),
                                      task_handle=read(job['job_path'])['dispatcher_task_handle'],expected_running=True)
        with tempfile.NamedTemporaryFile(dir=RT,prefix='.registry-',delete=False) as f:
            temp=Path(f.name);f.write((json.dumps(value,indent=2)+chr(10)).encode());f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        directory=os.open(RT,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    jp=Path(a.job)
    if not jp.is_absolute() or jp.parent!=RT:raise RuntimeError('Only root-owned refill job is permitted')
    job=read(jp);node,gpu=job['node'],job['gpu']
    if manager.HOSTS.get(node)!=job['host'] or gpu not in manager.ALLOWED_GPUS.get(node,()):raise RuntimeError('Unowned slot')
    with (Path('/tmp')/('ms-refill-'+node+'-'+str(gpu)+'.lock')).open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read(job['previous_config']);q=Path(previous['queue'])
        deadline=time.monotonic()+7200
        while not (q/'resources_safe.json').is_file():
            if (q/'failure.json').exists():raise RuntimeError('Previous worker failed; refill is not an automatic retry')
            live=read(q/'live.json');current=manager.base.identity(live['manager_identity']['pid'])
            if any(current[k]!=live['manager_identity'][k] for k in manager.base.IDENTITY_KEYS):raise RuntimeError('Previous manager changed')
            if time.monotonic()>deadline:raise RuntimeError('Native safety wait exceeded bound; no new model started')
            print(json.dumps(dict(event='waiting_predecessor',phase=previous['phase'],new_phase=job['phase'])),flush=True)
            time.sleep(15)
        safe=read(q/'resources_safe.json');exit=read(q/'worker_exit.json');summary=read(Path(previous['out'])/'summary.json')
        if not all(safe.get(k) is True for k in ['all_holders_safe','locks_released','leases_restored']):raise RuntimeError('Predecessor not safe')
        if exit.get('exit_code')!=0 or summary.get('status')!='completed' or summary.get('processed')!=previous['targetcount'] or summary.get('allprefetch_drained') is not True or summary.get('uncommitted_prefetch')!=0:raise RuntimeError('Previous finite assignment did not finish cleanly')
        if summary['selected_source_ids']!=previous['selected_source_ids'] or summary.get('vlm_called') is not False or summary.get('quality_approved') is not False:raise RuntimeError('Predecessor outcome scope differs')
        restored=read(q/('holder_restored_gpu%d.json'%gpu));holder=dict(restored['holder_identity'],gpu=gpu,uuid=restored['uuid'])
        manager.base.check_holder(holder,manager.base.identity(holder['pid']),gpu,manager.base.uuid_for(gpu),manager.base.apps())
        if manager.h.lock_snapshot(gpu)['held']:raise RuntimeError('GPU slot is now owned by another job')
        plan=Path(job['plan']);raw=plan.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=job['plan_sha256']:raise RuntimeError('Refill plan changed')
        rows=[json.loads(l) for l in raw.splitlines() if l.strip()]
        if len(rows)!=job['source_count'] or len({r['id'] for r in rows})!=len(rows):raise RuntimeError('Refill source count differs')
        phase=job['phase'];cp=RT/(phase+'_config.json');tp=RT/(phase+'_ticket.json')
        if cp.exists() or tp.exists() or (B/'queues'/phase).exists():raise RuntimeError('This refill already claimed; never restart')
        ticket=read(previous['ticket']['path']);ticket.update(edition=phase,ticket_id=phase,
            output_root=str(B/phase),sources=rows,stop_file=str(B/phase/phase/'STOP'))
        write_new(tp,ticket)
        cfg=dict(previous,phase=phase,config_path=str(cp),out=str(B/phase),queue=str(B/'queues'/phase),holders=[holder],
            targetcount=len(rows),input_count=len(rows),argv=[manager.PYTHON,'-B',str(manager.SELF),'worker','--config',str(cp)],
            producer=manager.ref(manager.SELF),resource_manager_ref=manager.ref(manager.SELF),core_ref=manager.ref(manager.CORE),
            ticket=manager.ref(tp),plan=manager.ref(plan),selected_source_ids=[r['id'] for r in rows],
            resource_previous_safe=manager.ref(q/'resources_safe.json'),holder_restored_refs=[manager.ref(q/('holder_restored_gpu%d.json'%gpu))])
        refs=[cfg[k] for k in ['ticket','plan','core_ref','modelref','resource_previous_safe']]+cfg['holder_restored_refs']
        cfg['source_sha256']={p:manager.base.sha(p) for p in sorted(manager.dependencies()|{r['path'] for r in refs})}
        write_new(cp,cfg);manager.validate(cfg);manager.check_model(cfg)
        sha=manager.base.sha(cp);args=[manager.PYTHON,'-B',str(manager.SELF),'--config',str(cp),'--config-sha',sha,'--manager-sha',manager.BASE_SHA]
        update_registry(job)
        print(json.dumps(dict(event='refill_launch',node=node,gpu=gpu,phase=phase,sources=len(rows),config_sha=sha)),flush=True)
        os.execv(manager.PYTHON,args)

if __name__=='__main__':main()
