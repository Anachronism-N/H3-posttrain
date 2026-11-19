"""Launch a new frozen N1 pipeline only through a real fresh empty-pair handoff."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import time
import ms_fullframe_vlm_postprocess_batch_v8 as v
import ms_vlm_fresh_empty_pair_handoff_v1 as fresh

SELF=Path(__file__).resolve()
REGISTRY=v.RT/'ms_vlm_postprocess_jobs_registry_v1.json'


def validate_template(job):
    v.need(job.get('schema')=='ms_vlm_n1_fresh_pipeline_launch_job_v1' and job.get('producer_ref')==v.ref(SELF), 'Frozen N1 launcher required')
    cfg=v.checked(job['template_ref'])
    v.need(cfg['node']=='N1' and tuple(cfg['gpus']) in v.PAIRS['N1'] and cfg['schema']==v.SCHEMA and
           cfg['phase'].startswith(v.PREFIX) and cfg['new_stage_authorized'] is True and cfg['release_authorized'] is True, 'Exact authorized N1 pipeline scope required')
    v.need(cfg['producer']==cfg['resource_manager_ref']==v.ref(v.SELF) and
           cfg['holders']==cfg['resource_safe_refs']==cfg['holder_restored_refs']==[], 'Fresh template cannot invent predecessor holder identities')
    v.need(cfg['config_path']==str(v.RT/(cfg['phase']+'_config.json')) and cfg['out']==str(v.ROOT/cfg['phase']) and
           cfg['queue']==str(v.ROOT/'queues'/cfg['phase']) and cfg['argv']==[v.PYTHON,'-B',str(v.SELF),'worker','--config',cfg['config_path']] and
           cfg['cwd']=='/tmp' and cfg['env']==v.expected_env(cfg), 'Stable exact pipeline paths/env/argv required')
    rows=v.rows_for(cfg)
    expected=v.dependencies()|{r['path'] for r in v.source_refs(cfg,rows)}
    v.need(set(cfg['source_sha256'])==expected, 'Exact prevalidated source dependency set required')
    for path,sha in cfg['source_sha256'].items():v.need(v.base.sha(path)==sha,'Frozen pipeline source changed')
    cpu=v.checked(job['cpu_ref'])
    v.need(cpu['tests_run']==73 and cpu['source_unchanged'] is True and cpu['failures']==cpu['errors']==cpu['skips']==0 and
           cpu['before_sha256']['ms_fullframe_vlm_postprocess_batch_v8.py']==cfg['producer']['sha256'] and
           cpu['before_sha256']['ms_vlm_fresh_empty_pair_handoff_v1.py']==v.ref(fresh.SELF)['sha256'], 'Frozen fresh-admission CPU proof required')
    seed=v.checked(job['provision_job_ref'])
    v.need(seed['node']==cfg['node'] and seed['gpus']==cfg['gpus'] and seed['producer_ref']==v.ref(fresh.SELF), 'Fresh resource job scope differs')
    v.base.check_model(cfg)
    return cfg,rows,seed


def registry_promote(job,job_ref,cfg,rows,promote=False):
    fd=os.open(REGISTRY.with_suffix('.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);reg=v.load(REGISTRY)
        matches=[x for x in reg['jobs'] if x.get('phase')==cfg['phase']]
        v.need(len(matches)==1,'Exactly one new pending source allocation required')
        item=matches[0]
        v.need(item.get('expected_running') is False and item.get('launcher_job_ref')==job_ref and item.get('task_handle')==job['task_handle'] and
               item.get('plan_ref')==cfg['plan'] and item.get('selected_source_ids')==cfg['selected_source_ids'], 'Pending N1 allocation differs')
        v.need(not [x for x in reg['jobs'] if x is not item and x.get('node')=='N1' and x.get('gpus')==cfg['gpus'] and x.get('expected_running') is True], 'N1 slot has another registered owner')
        ids={r['id'].casefold() for r in rows}
        for entry in reg['jobs']:
            if entry is item:continue
            other=list(entry.get('selected_source_ids',[]));cp=Path(entry['config_path'])
            if cp.exists():other+=v.load(cp)['selected_source_ids']
            else:
                v.need(entry.get('expected_running') is False and 'plan_ref' in entry,'Unbound missing foreign config')
                plan=entry['plan_ref'];v.need(v.base.sha(plan['path'])==plan['sha256'],'Foreign pending plan changed')
                other += [v.h.bench.strict(line)['id'] for line in Path(plan['path']).read_bytes().splitlines() if line.strip()]
            v.need(not ids.intersection(i.casefold() for i in other),'New N1 plan overlaps another allocation')
        if promote:
            item.update(expected_running=True,worker_status='fresh_pair_validated_before_native_main')
            with tempfile.NamedTemporaryFile(dir=v.RT,prefix='.vlm-registry-',delete=False) as f:
                p=Path(f.name);f.write(v.h.bench.encoded(reg));f.flush();os.fsync(f.fileno())
            os.replace(p,REGISTRY);fd=os.open(v.RT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)


def launch(path,digest):
    ref=dict(path=str(Path(path)),sha256=digest);job=v.checked(ref);cfg,rows,seed=validate_template(job)
    v.need(Path(path)==v.RT/(cfg['phase']+'_launch_job.json'),'Stable scoped new launch job required')
    registry_promote(job,ref,cfg,rows)
    for p in (Path(cfg['config_path']),Path(cfg['queue']),Path(cfg['out'])):v.need(not p.exists(),'Existing phase must not be restarted')
    v.write_once(v.RT/(cfg['phase']+'_fresh_launch_started.json'),dict(job_ref=ref,launcher_identity=v.base.identity(os.getpid()),automatic_retry=False))
    seedq=v.ROOT/'queues'/seed['phase'];log=(v.RT/(cfg['phase']+'_fresh_seed.log')).open('xb');child=None;safe=False
    try:
        child=subprocess.Popen(['/opt/venv_eval/bin/python','-B',str(fresh.SELF),'--job',job['provision_job_ref']['path'],'--job-sha',job['provision_job_ref']['sha256']],cwd='/tmp',stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+240
        while not (seedq/'resources_safe.json').exists():
            v.need(child.poll() is None,'Fresh resource initialization failed, never restart')
            v.need(time.monotonic()<deadline,'Fresh ownership proof not ready; no model started')
            time.sleep(2)
        safe=True
        sr=v.ref(seedq/'resources_safe.json');rr=[v.ref(seedq/('holder_restored_gpu%d.json'%g)) for g in cfg['gpus']]
        cfg['holders']=[dict(v.checked(r)['holder_identity'],gpu=g,uuid=v.checked(r)['uuid']) for g,r in zip(cfg['gpus'],rr)]
        cfg['resource_safe_refs']=[sr,sr];cfg['holder_restored_refs']=rr
        refs=v.source_refs(cfg,rows);cfg['source_sha256']={p:v.base.sha(p) for p in sorted(v.dependencies()|{r['path'] for r in refs})}
        v.base.headroom=v.actual_controller_headroom;v.base.selected_sources=v.selected
        v.validate(cfg);v.write_once(cfg['config_path'],cfg);sha=v.base.sha(cfg['config_path'])
        v.base.check_sources(cfg,cfg['config_path'],sha,v.BASE_SHA);v.base.check_model(cfg);v.need(v.checked(ref)==job,'Frozen launch job changed')
        registry_promote(job,ref,cfg,rows,True)
        os.execv(v.PYTHON,[v.PYTHON,'-B',str(v.SELF),'--config',cfg['config_path'],'--config-sha',sha,'--manager-sha',v.BASE_SHA])
    except BaseException:
        if safe:
            for gpu in cfg['gpus']:
                proof=v.load(seedq/('holder_restored_gpu%d.json'%gpu));owner=dict(proof['holder_identity'],gpu=gpu,uuid=proof['uuid'])
                try:
                    fd=os.pidfd_open(owner['pid'])
                    try:
                        v.base.check_holder(owner,v.base.identity(owner['pid']),gpu,v.base.uuid_for(gpu),v.base.apps())
                        v.base.signal.pidfd_send_signal(fd,v.base.signal.SIGTERM)
                    finally:os.close(fd)
                except (FileNotFoundError,ProcessLookupError):pass
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--job-sha',required=True);a=p.parse_args();launch(a.job,a.job_sha)
