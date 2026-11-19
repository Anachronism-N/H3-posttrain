"""Finite rebalance dispatcher: only transfers a retired, drained unclaimed suffix."""
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

def validate_release(job):
    release=manager.checked(job['source_release_ref'])
    if release.get('schema')!='ms_native_unclaimed_tail_release_v1':raise RuntimeError('Wrong source release schema')
    if not all(release.get(k) is True for k in ['original_task_retired_no_resume','no_claim_reset','allprefetch_drained']):raise RuntimeError('Original task is not permanently retired and drained')
    if any(type(release.get(k)) is not int or release[k]!=0 for k in ['claimed_suffix_intersection','uncommitted_prefetch']):raise RuntimeError('Release contains claimed or uncommitted sources')
    original=manager.checked(release['original_config_ref'])
    ticket=manager.checked(release['original_ticket_ref'])
    summary=manager.checked(release['original_summary_ref'])
    safe=manager.checked(release['original_resources_safe_ref'])
    exit=manager.checked(release['original_worker_exit_ref'])
    stop_path=Path(release['persistent_STOP_ref']['path'])
    if not stop_path.is_file() or stop_path.is_symlink():raise RuntimeError('Persistent original STOP is missing or symlinked')
    manager.checked(release['persistent_STOP_ref'])
    if original.get('config_path')!=release['original_config_ref']['path'] or original.get('ticket')!=release['original_ticket_ref']:raise RuntimeError('Release ticket does not belong to exact original config')
    phase=original['phase'];out=Path(original['out']);q=Path(original['queue'])
    if ticket.get('edition')!=phase or ticket.get('ticket_id')!=phase or ticket.get('output_root')!=str(out) or ticket.get('stop_file')!=str(stop_path) or stop_path!=out/phase/'STOP':raise RuntimeError('Release original ticket scope differs')
    if Path(release['original_summary_ref']['path'])!=out/'summary.json' or Path(release['original_resources_safe_ref']['path'])!=q/'resources_safe.json' or Path(release['original_worker_exit_ref']['path'])!=q/'worker_exit.json':raise RuntimeError('Release proofs do not belong to original task')
    if not all(safe.get(k) is True for k in ['all_holders_safe','locks_released','leases_restored','managed_parent_wait','unknown_jobs_not_signaled']):raise RuntimeError('Released original resources are not native-safe')
    if exit.get('exit_code')!=0 or exit.get('phase')!=phase:raise RuntimeError('Released original worker did not exit cleanly')
    sources=ticket['sources'];ids=[r['id'] for r in sources];processed=summary.get('processed')
    if len(set(ids))!=len(ids) or original.get('selected_source_ids')!=ids or original.get('targetcount')!=len(ids) or original.get('input_count')!=len(ids):raise RuntimeError('Original config selected source scope differs')
    if summary.get('status')!='drained_not_complete' or summary.get('selected_source_ids')!=ids or summary.get('edition')!=phase or summary.get('ticket_id')!=ticket['ticket_id']:raise RuntimeError('Released original summary scope differs')
    if type(processed) is not int or not 0<=processed<len(ids):raise RuntimeError('Original processed prefix count differs')
    if summary.get('allprefetch_drained') is not True or summary.get('uncommitted_prefetch')!=0 or summary.get('vlm_called') is not False or summary.get('quality_approved') is not False:raise RuntimeError('Original summary is not drained and archive-only')
    prefix=ids[:processed];suffix=ids[processed:];prefix_set=set(prefix)
    if [r['id'] for r in summary.get('all_outcomes',[])]!=prefix:raise RuntimeError('Original outcomes do not exactly cover processed prefix')
    if summary.get('unprocessed_source_ids')!=suffix or release.get('unclaimed_ids')!=suffix:raise RuntimeError('Release is not the exact original unprocessed suffix')
    if any(type(release.get(k)) is not int or release[k]!=count for k,count in [('unclaimed_source_count',len(suffix)),('original_processed_source_count',processed),('original_claim_count',processed)]):raise RuntimeError('Release source counts differ')
    claims=out/ticket['ticket_id']/'claims'/'sources'
    if not claims.is_dir() or claims.is_symlink() or claims.resolve(strict=True)!=claims:raise RuntimeError('Original own claims path is not stable')
    claimed=set()
    with os.scandir(claims) as entries:
        for index,entry in enumerate(entries):
            if index>=3000:raise RuntimeError('Original own claim listing exceeds 3000')
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False) or not entry.name.endswith('.json'):raise RuntimeError('Unexpected original own claim entry')
            claimed.add(entry.name[:-5])
    if claimed&set(suffix):raise RuntimeError('Released suffix already has an original own claim')
    if claimed!=prefix_set:raise RuntimeError('Original own claims do not exactly equal processed prefix')
    plan=Path(job['plan'])
    if not plan.is_absolute() or plan.resolve(strict=True)!=plan:raise RuntimeError('Stable absolute rebalance plan required')
    raw=plan.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=job['plan_sha256']:raise RuntimeError('Rebalance plan changed')
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if type(job.get('source_count')) is not int or len(rows)!=job['source_count'] or not rows or len({r['id'] for r in rows})!=len(rows):raise RuntimeError('Rebalance source count or unique IDs differ')
    suffix_by_id={r['id']:r for r in sources[processed:]}
    if any(r['id'] in prefix_set or r!=suffix_by_id.get(r['id']) for r in rows):raise RuntimeError('Rebalance plan is not an exact-row subset of released suffix')
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    jp=Path(a.job)
    if not jp.is_absolute() or jp.parent!=RT:raise RuntimeError('Only root-owned refill job is permitted')
    job=read(jp)
    rows=validate_release(job)
    node,gpu=job['node'],job['gpu']
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
        plan=Path(job['plan'])
        phase=job['phase'];cp=RT/(phase+'_config.json');tp=RT/(phase+'_ticket.json')
        if cp.exists() or tp.exists() or (B/'queues'/phase).exists() or (B/phase).exists():raise RuntimeError('This refill already claimed; never restart')
        ticket=read(previous['ticket']['path']);ticket.pop('source_budget',None)
        ticket.update(edition=phase,ticket_id=phase,output_root=str(B/phase),sources=rows,
            stop_file=str(B/phase/phase/'STOP'),source_transfer_ref=job['source_release_ref'])
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
