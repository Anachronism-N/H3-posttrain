"""After native safe and all ACKs, start a fresh source group, never resume old claims."""
import argparse
import hashlib
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
    p.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    df=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def run(node):
    old=Path('/root/stage2_dynamic_main3936_'+node+'_20261028_v1');root=Path('/root/stage2_dynamic_remaining25965_'+node+'_20261028_v1')
    if node not in ('N1','N2','N3','N6'):raise ValueError('Explicit predecessor nodes only')
    olddep=json.loads((old/'deployment.json').read_bytes());root.mkdir(exist_ok=False);code=root/'code';code.mkdir();hashes={}
    for name in ('stage2_local_dynamic_native_manager_v2.py','stage2_local_model_worker_v1.py','stage2_local_packet_v1.py','stage2_local_publication_v1.py','stage2_model_codec_v1.py','stage2_cpu_feeder_v1.py','stage2_feeder_media_v1.py','stage2_node_uploader_v1.py','resource_manager.py','stage2_start_staged.py'):
        raw=(RT/name).read_bytes();compile(raw,str(code/name),'exec');ref=put(code/name,raw);hashes[ref['path']]=ref['sha256']
    sys.path.insert(0,str(code));import stage2_local_dynamic_native_manager_v2 as m
    from stage2_local_publication_v1 import write_once,checked
    allocation=ROOT/'stage2_main_remaining_direct_20261028_v1/allocation.json';a=json.loads(allocation.read_bytes());raw=Path(a['manifest_ref']['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a['manifest_ref']['sha256']:raise ValueError('Reserved next input plan changed')
    plan=put(root/'pool_manifest.jsonl',raw);hashes[plan['path']]=plan['sha256']
    raw=(old/'model_reference.json').read_bytes();model=put(root/'model_reference.json',raw);hashes[model['path']]=model['sha256'];model.pop('bytes');model['model_identity_sha256']=json.loads(raw)['model_identity']['sha256']
    token_code="import pathlib,sys;sys.stdout.buffer.write(pathlib.Path('/root/stage2_pool_remaining25965_20261028_v1/token').read_bytes())"
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','28.216.17.70','/opt/venv_eval/bin/python','-B','-c',shlex.quote(token_code)],capture_output=True,timeout=15)
    if r.returncode:raise RuntimeError('Next pool token transport failed')
    put(root/'token',r.stdout);deadline=time.monotonic()+21600
    write_once(root/'transition_wait.json',dict(node=node,previous_root=str(old),new_plan_ref=plan,source_claims_created=0,models_started=0,wait_started_unix=time.time(),no_old_claim_reassignment=True))
    print(json.dumps(dict(event='next_source_group_waiting_native_safe',node=node,source_count=25965)),flush=True)
    while True:
        if time.monotonic()>deadline:raise RuntimeError('Bounded native-safe wait ended, no sources claimed')
        ready=True
        for e in olddep['configs']:
            c=checked(e['config_ref']);q=Path(c['queue'])
            if (q/'failure.json').exists():raise RuntimeError('Predecessor failed; no automatic retry or new admission')
            if not (q/'resources_safe.json').exists() or not (old/'out'/'generation_drained'/e['owner']).exists():ready=False;continue
            safe=json.loads((q/'resources_safe.json').read_bytes())
            if not all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored')):ready=False;continue
            exit_=json.loads((q/'worker_exit.json').read_bytes())
            if exit_['exit_code']!=0 or exit_['source_unchanged'] is not True:raise RuntimeError('Predecessor exit closure differs')
            live=json.loads((q/'live.json').read_bytes());i=live['child_identity']
            try:
                current=m.b.identity(i['pid'])
                if current['start_ticks']==i['start_ticks'] and current['state'] not in ('Z','X','x'):raise RuntimeError('Previous worker still alive despite safe')
            except FileNotFoundError:pass
        if ready:
            commits={tuple(p.relative_to(old/'out'/'sources').parts[:2]) for p in (old/'out'/'sources').glob('*/*/commit.json')};acked=set()
            for d in (old/'out'/'outbox').iterdir():
                if not (d/'sealed.json').exists() or not (d/'upload_receipt.json').exists():ready=False;break
                seal=json.loads((d/'sealed.json').read_bytes());ack=json.loads((d/'upload_receipt.json').read_bytes())
                if ack['batch_id']!=seal['batch_id'] or ack['archive_sha256']!=seal['archive_sha256']:raise RuntimeError('Previous ACK differs')
                acked.update(map(tuple,seal['source_keys']))
            ready=ready and commits==acked
        if ready:break
        time.sleep(5)
    write_once(root/'previous_native_ack_closed.json',dict(previous_root=str(old),all_previous_node_pairs_native_safe=True,all_previous_local_results_shared_acked=True,source_count=len(commits),previous_source_keys=sorted(map(list,commits)),observed_unix=time.time(),models_called=0))
    stop=old/'out'/'UPLOAD_STOP'
    if not stop.exists():write_once(stop,dict(reason='node uploader handed off after all old ACKs',models_called=0))
    def old_uploader_alive():
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
            except (OSError,UnicodeDecodeError):continue
            if len(argv)>2 and argv[2]==str(old/'code/stage2_node_uploader_v1.py'):return True
        return False
    until=time.monotonic()+20
    while old_uploader_alive():
        if time.monotonic()>until:raise RuntimeError('Previous owned uploader has not drained')
        time.sleep(.5)
    entries=[]
    for previous in olddep['configs']:
        pair=previous['gpus'];holders=[]
        for gpu in pair:
            uuid=m.b.uuid_for(gpu);apps=[x for x in m.b.apps() if x['uuid']==uuid]
            if len(apps)!=1:raise RuntimeError('Fresh exact restored holder unavailable')
            owner=m.b.identity(apps[0]['pid']);owner.update(gpu=gpu,uuid=uuid);m.b.check_holder(owner,m.b.identity(owner['pid']),gpu,uuid,m.b.apps());holders.append(owner)
        label='g'+''.join(map(str,pair));phase='stage2_dynamic_remaining25965_'+node+'_'+label;owner=node+'-'+label+'-remaining25965';cp=root/(phase+'_config.json');producer=dict(path=str(code/'stage2_local_dynamic_native_manager_v2.py'),sha256=hashes[str(code/'stage2_local_dynamic_native_manager_v2.py')])
        c=dict(schema='stage2_local_dynamic_native_config_v1',phase=phase,node=node,gpus=pair,owner=owner,local_root=str(root),out=str(root/'out'),queue=str(root/'queues'/phase),config_path=str(cp),holders=holders,new_stage_authorized=True,release_authorized=True,targetcount=25965,input_count=25965,plan={k:plan[k] for k in ('path','sha256')},producer=producer,resource_manager_ref=producer,source_sha256=hashes,modelref=model,reuse_existing_results=True,cwd='/tmp',env=dict(CUDA_VISIBLE_DEVICES=','.join(map(str,pair)),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',PYTHONPATH='/opt/venv_eval/lib/python3.11/site-packages'),argv=['/opt/venv_vllm/bin/python','-B',str(code/'stage2_local_dynamic_native_manager_v2.py'),'worker','--config',str(cp)])
        cr=put(cp,json.dumps(c,sort_keys=True).encode());m.check_sources(c,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(c);m.validate(c)
        oldfeed=checked(previous['feeder_ref']);feed=dict(out=c['out'],owner=owner,token_path=str(root/'token'),claim_endpoint='http://28.216.17.70:18995/claim',ffmpeg_path=oldfeed['ffmpeg_path'],prefetch_packets=8);fr=put(root/(phase+'_feeder.json'),json.dumps(feed,sort_keys=True).encode());entries.append(dict(config_ref=cr,feeder_ref=fr,gpus=pair,owner=owner,phase=phase))
    write_once(root/'deployment.json',dict(node=node,configs=entries,source_sha256=hashes,global_pool_namespace='remaining25965',global_pool_count=25965,source_claims_created=0,gpu_hotpaths_all_local=True));logs=root/'logs';logs.mkdir()
    for e in entries:(root/'out'/'ready'/e['owner']).mkdir(parents=True,exist_ok=True)
    write_once(logs/'uploader_launch.json',dict(one_uploader_per_node=True,start_unix=time.time()))
    cmd=shlex.join(['/opt/venv_eval/bin/python','-B',str(code/'stage2_node_uploader_v1.py'),'--local-root',str(root/'out'),'--shared-root',str(ROOT/'stage2_dynamic_local_shared_results_20261028_v1'),'--node',node,'--seconds','86400'])+' > '+shlex.quote(str(logs/'uploader.log'))+' 2>&1'
    subprocess.run(['tmux','-S',str(root/'tmux.sock'),'new-session','-d','-s','main3936','-n','uploader','-c','/tmp',cmd],check=True)
    import stage2_start_staged as start
    for e in entries:start.start(root,e['gpus'])
    write_once(root/'transition_complete.json',dict(node=node,new_group='remaining25965',replicas=len(entries),previous_sources_never_reissued=True,quality_approved=False,training_eligible=False,observed_unix=time.time()))
    print(json.dumps(dict(event='next_group_native_started',node=node,replicas=len(entries))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True);a=p.parse_args();run(a.node)
