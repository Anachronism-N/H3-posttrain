"""Fresh native manager config consumes only zero-model-budget original owner packets."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime')


def run(node,pair):
    root=Path('/root/stage2_dynamic_native37610v3_'+node+'_20261028_v1');sys.path.insert(0,str(root/'code'))
    from stage2_local_publication_v1 import write_once,checked
    lock=os.open(root/'kernel_repair_start.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);dep=json.loads((root/'deployment.json').read_bytes());e=next(x for x in dep['configs'] if x['gpus']==pair);old=checked(e['config_ref']);oldbinding=root/'runtime_bindings'/(old['owner']+'.json')
    if oldbinding.exists():old=checked(json.loads(oldbinding.read_bytes())['config_ref'])
    q=Path(old['queue']);safe=json.loads((q/'resources_safe.json').read_bytes());assert all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored')) and (q/'failure.json').exists(),'Only exact failed-safe predecessor requires this repair'
    p=root/'code/stage2_local_dynamic_native_manager_v4.py';raw=(RT/p.name).read_bytes()
    if p.exists():assert p.read_bytes()==raw,'New repair code changed'
    else:
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    spec=importlib.util.spec_from_file_location('repair_native',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    live=json.loads((q/'live.json').read_bytes())['child_identity']
    try:current=m.b.identity(live['pid']);assert current['start_ticks']!=live['start_ticks'] or current['state'] in ('Z','X','x'),'Old worker still alive'
    except FileNotFoundError:pass
    for prior in sorted((root/'runtime_bindings').glob('*.json')):
        r=checked(json.loads(prior.read_bytes())['config_ref']);state=root/'worker_state'/r['owner']/r['phase'];deadline=time.monotonic()+360
        while not (state/'load.receipt.json').exists():
            assert not (Path(r['queue'])/'failure.json').exists(),'Earlier repair failed, do not expand';assert time.monotonic()<deadline,'Earlier repair load still pending';time.sleep(1)
    holders=[]
    for g in pair:
        uuid=m.b.uuid_for(g);apps=[x for x in m.b.apps() if x['uuid']==uuid];assert len(apps)==1,'Exact current restored holder required';i=m.b.identity(apps[0]['pid']);i.update(gpu=g,uuid=uuid);m.b.check_holder(i,m.b.identity(i['pid']),g,uuid,m.b.apps());holders.append(i)
    phase=old['phase']+'_memory_cachefix1';cp=root/(phase+'_config.json');cfg=dict(old);cfg.update(phase=phase,config_path=str(cp),queue=str(root/'queues'/phase),holders=holders,producer=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),resource_manager_ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),source_sha256=dict(old['source_sha256'],**{str(p):hashlib.sha256(raw).hexdigest()}),argv=['/opt/venv_vllm/bin/python','-B',str(p),'worker','--config',str(cp)],worker_state_layout='by_phase',recover_only_original_uncalled_claims=True,source_claims_reassigned=False,previous_failed_queue=old['queue']);cr=write_once(cp,cfg);m.check_sources(cfg,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(cfg);m.validate(cfg)
    # Do not spawn a duplicate feeder. Its original FIFO and source claims remain live.
    feeder_script=str(root/'code/stage2_cpu_feeder_v3.py');feeder_config=e['feeder_ref']['path'];feeders=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
        except (OSError,UnicodeDecodeError):continue
        if argv==['/opt/venv_eval/bin/python','-B',feeder_script,'--config',feeder_config]:feeders.append(int(proc.name))
    assert len(feeders)==1,'Exactly one existing original CPU feeder required'
    previous_binding=json.loads(oldbinding.read_bytes());write_once(root/'runtime_binding_history'/(cfg['owner']+'_before_memory_cachefix.json'),previous_binding)
    binding=write_once(root/'runtime_binding_history'/(cfg['owner']+'_memory_cachefix.json'),dict(config_ref=cr,previous_config_ref=previous_binding['config_ref'],original_owner=cfg['owner'],state_layout='by_phase',feeder_pid=feeders[0],old_failure_preserved=True,new_source_claims_created_by_repair=0,started_unix=time.time()))
    temp=root/'runtime_bindings'/(cfg['owner']+'.next.json');write_once(temp,json.loads(Path(binding['path']).read_bytes()));os.replace(temp,oldbinding);df=os.open(oldbinding.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
    logs=root/'logs';cmd=shlex.join(['/opt/venv_vllm/bin/python','-B',str(p),'--config',str(cp),'--config-sha',cr['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA])+' > '+shlex.quote(str(logs/(phase+'_manager.log')))+' 2>&1';subprocess.run(['tmux','-S',str(root/'tmux.sock'),'new-window','-d','-t','main3936','-n','repair-g'+''.join(map(str,pair)),'-c','/tmp',cmd],check=True);deadline=time.monotonic()+30
    while not (Path(cfg['queue'])/'live.json').exists():
        assert not (Path(cfg['queue'])/'failure.json').exists(),'New repair admission failed';assert time.monotonic()<deadline,'New repair live receipt pending';time.sleep(.2)
    now=json.loads((Path(cfg['queue'])/'live.json').read_bytes());print(json.dumps(dict(event='kernel_lock_parser_repair_admitted',node=node,gpus=pair,owner=cfg['owner'],new_worker_pid=now['child_identity']['pid'],original_feeder_pid=feeders[0],new_source_claims_created_by_repair=0,old_result_and_budgets_preserved=True,binding_ref=binding)),flush=True);os.close(lock)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=('N1','N2','N8'));p.add_argument('--pair',required=True,choices=('01','23','45','67'));a=p.parse_args();run(a.node,list(map(int,a.pair)))
