"""After all queued node work, consume56 original prepared claims with no model budgets."""
import argparse
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
ROOT=RT.parent


def put(p,raw):
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    df=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def run(node):
    if node not in ('N1','N8'):raise ValueError('Only nodes with original uncalled prepared claims allowed')
    old=Path('/root/stage2_dynamic_native37610_'+node+'_20261028_v1');last=Path('/root/stage2_dynamic_legacy15973_'+node+'_20261028_v1');sys.path.insert(0,str(old/'code'))
    from stage2_local_publication_v1 import checked,write_once
    ref=dict(path=str(old/'uncalled_prepared_original_claims.json'),sha256=hashlib.sha256((old/'uncalled_prepared_original_claims.json').read_bytes()).hexdigest());recovery=checked(ref);assert recovery['node']==node and recovery['recovery_requires_same_original_owner'] is True;deadline=time.monotonic()+86400
    print(json.dumps(dict(event='original_uncalled_tail_waiting',node=node,sources=recovery['source_count'],source_reclaims=0)),flush=True)
    while not (last/'deployment.json').exists():assert time.monotonic()<deadline;time.sleep(15)
    dep=json.loads((last/'deployment.json').read_bytes())
    while True:
        ready=True
        for e in dep['configs']:
            c=checked(e['config_ref']);q=Path(c['queue'])
            if (q/'failure.json').exists():raise RuntimeError('Final queued stage failed; do not consume recovery sources on occupied GPUs')
            if not (q/'resources_safe.json').exists():ready=False;continue
            s=json.loads((q/'resources_safe.json').read_bytes());assert all(s.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored'));assert json.loads((q/'worker_exit.json').read_bytes())['exit_code']==0
        if ready:
            source_keys={tuple(p.relative_to(last/'out'/'sources').parts[:2]) for p in (last/'out'/'sources').glob('*/*/commit.json')};acks=set()
            for p in (last/'out'/'outbox').iterdir():
                if not (p/'sealed.json').exists() or not (p/'upload_receipt.json').exists():ready=False;break
                seal=json.loads((p/'sealed.json').read_bytes());ack=json.loads((p/'upload_receipt.json').read_bytes());assert ack['batch_id']==seal['batch_id'] and ack['archive_sha256']==seal['archive_sha256'];acks.update(map(tuple,seal['source_keys']))
            ready=ready and acks==source_keys
        if ready:break
        assert time.monotonic()<deadline;time.sleep(10)
    write_once(old/'all_future_work_safe_for_original_recovery.json',dict(previous_root=str(last),native_safe=True,all_final_node_sources_shared_acked=True,recovery_manifest_ref=ref,original_source_claims_preserved=True,observed_unix=time.time()))
    if not (last/'out'/'UPLOAD_STOP').exists():write_once(last/'out'/'UPLOAD_STOP',dict(reason='all final node results ACKed; transfer single uploader to original claim recovery',models_called=0))
    def uploader_alive(root):
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:args=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
            except (OSError,UnicodeDecodeError):continue
            if len(args)>2 and args[2]==str(root/'code/stage2_node_uploader_v1.py'):return True
        return False
    until=time.monotonic()+20
    while uploader_alive(last):assert time.monotonic()<until;time.sleep(.5)
    assert not uploader_alive(old),'Original old uploader should already be stopped'
    for name in ('stage2_original_uncalled_recovery_manager.py','stage2_original_recovery_uploader.py'):
        put(old/'code'/name,(RT/name).read_bytes())
    manager_path=old/'code/stage2_original_uncalled_recovery_manager.py';spec=importlib.util.spec_from_file_location('original_recovery_manager',manager_path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);original_dep=json.loads((old/'deployment.json').read_bytes());groups={}
    for e in recovery['entries']:groups.setdefault(e['owner'],[]).append(e)
    configs=[]
    for owner,entries in groups.items():
        original=checked(next(e['config_ref'] for e in original_dep['configs'] if e['owner']==owner));pair=original['gpus'];rows=[]
        for e in entries:
            assert e['gpus']==pair and e['original_local_out']==str(old/'out');packet=checked(e['original_packet_ref']);assert packet['owner']==owner and not (old/'out'/'consumed_ready'/owner/Path(e['original_ready_ref']['path']).name).exists()
            for row in packet['claim']['rows']:assert not (old/'out'/'budgets'/row['split']/row['id']/'source.claim.json').exists();rows.append(row)
        holders=[]
        for g in pair:
            uuid=m.b.uuid_for(g);apps=[x for x in m.b.apps() if x['uuid']==uuid];assert len(apps)==1;identity=m.b.identity(apps[0]['pid']);identity.update(gpu=g,uuid=uuid);m.b.check_holder(identity,m.b.identity(identity['pid']),g,uuid,m.b.apps());holders.append(identity)
        phase=original['phase']+'_original_uncalled_tail';cp=old/(phase+'_config.json');plan=put(old/(phase+'_plan.jsonl'),b''.join((json.dumps(r,sort_keys=True,separators=(',',':'))+chr(10)).encode() for r in rows));cfg=dict(original);digest=hashlib.sha256(manager_path.read_bytes()).hexdigest();hashes=dict(original['source_sha256'],**{str(manager_path):digest,plan['path']:plan['sha256'],ref['path']:ref['sha256']});cfg.update(phase=phase,config_path=str(cp),queue=str(old/'queues'/phase),targetcount=len(rows),input_count=len(rows),plan=dict(path=plan['path'],sha256=plan['sha256']),holders=holders,source_sha256=hashes,producer=dict(path=str(manager_path),sha256=digest),resource_manager_ref=dict(path=str(manager_path),sha256=digest),argv=['/opt/venv_vllm/bin/python','-B',str(manager_path),'worker','--config',str(cp)],recovery_manifest_ref=ref,worker_state_layout='by_phase');cr=write_once(cp,cfg);m.check_sources(cfg,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(cfg);m.validate(cfg);write_once(old/'runtime_bindings'/(owner+'.json'),dict(config_ref=cr,original_uncalled_claim_recovery=True,state_layout='by_phase',original_failure_preserved=True,models_retried=0));configs.append((cfg,cr))
    sock=str(old/'original_recovery_tmux.sock');session='original-tail';up=shlex.join(['/opt/venv_eval/bin/python','-B',str(old/'code/stage2_original_recovery_uploader.py'),'--local-root',str(old/'out'),'--shared-root',str(ROOT/'stage2_dynamic_local_shared_results_20261028_v1'),'--node',node,'--seconds','21600'])+' > '+shlex.quote(str(old/'original_tail_uploader.log'))+' 2>&1';subprocess.run(['tmux','-S',sock,'new-session','-d','-s',session,'-n','uploader','-c','/tmp',up],check=True)
    previous=None
    for cfg,cr in configs:
        if previous:
            p=old/'worker_state'/previous['owner']/previous['phase']/'load.receipt.json';until=time.monotonic()+360
            while not p.exists():assert time.monotonic()<until and not (Path(previous['queue'])/'failure.json').exists();time.sleep(1)
        m.validate(cfg);cmd=shlex.join(['/opt/venv_vllm/bin/python','-B',str(manager_path),'--config',cfg['config_path'],'--config-sha',cr['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA])+' > '+shlex.quote(str(old/(cfg['phase']+'_manager.log')))+' 2>&1';subprocess.run(['tmux','-S',sock,'new-window','-d','-t',session,'-n',cfg['owner'],'-c','/tmp',cmd],check=True);previous=cfg
    write_once(old/'original_uncalled_recovery_launched.json',dict(source_count=recovery['source_count'],configs=[r for c,r in configs],new_source_claims_created=0,original_owners_preserved=True,observed_unix=time.time()))
    print(json.dumps(dict(event='original_uncalled_recovery_started',node=node,sources=recovery['source_count'],source_reclaims=0)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True);a=p.parse_args();run(a.node)
