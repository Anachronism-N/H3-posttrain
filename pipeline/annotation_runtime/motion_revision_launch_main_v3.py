"""Deploy corrected full-source revision only after three real-window outputs pass protocol."""
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
ROOT=RT.parent;REVISION='english_merged_edges_motion_revision_v3'
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225','N8':'28.216.18.136'}


def put(p,raw):
    p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    d=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def old_closed(old):
    d=json.loads((old/'deployment.json').read_bytes())
    for e in d['configs']:
        c=json.loads(Path(e['config_ref']['path']).read_bytes());q=Path(c['queue'])
        if not q.exists():
            assert not (old/'out'/'feeder_claims'/c['owner']).exists(),'No queue but feeder claims exist; investigate, never assume unused'
            continue
        s=json.loads((q/'resources_safe.json').read_bytes());assert all(s.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored')),'Old node manager safe required';assert json.loads((q/'worker_exit.json').read_bytes())['exit_code']==0
    keys={tuple(p.relative_to(old/'out'/'sources').parts[:2]) for p in (old/'out'/'sources').glob('*/*/commit.json')};ack=set()
    for p in (old/'out'/'outbox').iterdir():
        assert (p/'sealed.json').exists() and (p/'upload_receipt.json').exists();value=json.loads((p/'sealed.json').read_bytes());receipt=json.loads((p/'upload_receipt.json').read_bytes());assert receipt['batch_id']==value['batch_id'] and receipt['archive_sha256']==value['archive_sha256'];ack.update(map(tuple,value['source_keys']))
    assert keys==ack,'Original node result set must all be shared ACKed';return d


def run(node):
    root=Path('/root/stage2_motion_revision_final_main_'+node+'_20261028_v1');root.mkdir(exist_ok=False);code=root/'code';code.mkdir();hashes={}
    for n in ('motion_revision_native_manager_v1.py','motion_revision_model_worker_v1.py','motion_revision_codec_v1.py','motion_revision_feeder_v1.py','stage2_local_publication_v1.py','stage2_node_uploader_v1.py','resource_manager.py'):
        raw=(RT/('motion_revision_codec_v3.py' if n=='motion_revision_codec_v1.py' else n)).read_bytes();compile(raw,str(code/n),'exec');r=put(code/n,raw);hashes[r['path']]=r['sha256']
    sys.path.insert(0,str(code));import motion_revision_native_manager_v1 as m
    from stage2_local_publication_v1 import checked,write_once
    deadline=time.monotonic()+3600;proof=ROOT/'motion_revision_english_merged_edges_final_20261028_v1/first3_correct_protocol_acceptance.json'
    write_once(root/'waiting_for_correct_first3.json',dict(revision=REVISION,old_wrong_outputs_not_used=True,new_source_claims=0,observed_unix=time.time()))
    while not proof.exists():assert time.monotonic()<deadline,'Three real corrected outputs still pending; no model or new source claim started';time.sleep(3)
    acceptance=json.loads(proof.read_bytes());assert acceptance['passed'] is True and acceptance['source_count']==3 and acceptance['revision']==REVISION;old=Path('/root/stage2_motion_revision_corrected_main_'+node+'_20261028_v1');
    while True:
        try:d=old_closed(old);break
        except (FileNotFoundError,AssertionError):
            assert time.monotonic()<deadline,'Prior candidate still draining; preserve claims and wait';time.sleep(3)
    # N2 must wait for its pilot native-safe and uploader ACK before reusing pair.
    if node=='N2':
        pilot=Path('/root/stage2_motion_revision_pilot3_corrected_N2_20261028_v1');old_closed(pilot);stops=[(old,'stage2_node_uploader_v1.py'),(pilot,'stage2_node_uploader_v1.py')]
    else:stops=[(old,'stage2_node_uploader_v1.py')]
    for base,name in stops:
        if not (base/'out'/'UPLOAD_STOP').exists():write_once(base/'out'/'UPLOAD_STOP',dict(reason='all old results ACKed; correct revision one-node uploader handoff',models_called=0))
        until=time.monotonic()+20
        while True:
            live=False
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:a=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
                except (OSError,UnicodeDecodeError):continue
                if len(a)>2 and a[2]==str(base/'code'/name):live=True
            if not live:break
            assert time.monotonic()<until,'Old single uploader still live';time.sleep(.5)
    base=checked(d['configs'][0]['config_ref']);model=put(root/'model_reference.json',Path(base['modelref']['path']).read_bytes());model.pop('bytes');model['model_identity_sha256']=base['modelref']['model_identity_sha256'];hashes[model['path']]=model['sha256'];contract=ROOT/'motion_revision_english_merged_edges_final_20261028_v1/revision_contract.json';plan=put(root/'source_revision_plan.json',contract.read_bytes());hashes[plan['path']]=plan['sha256'];info=json.loads(contract.read_bytes());assert info['revision']==REVISION and info['source_count']==176771
    query="import pathlib,sys;sys.stdout.buffer.write(pathlib.Path('/root/stage2_motion_revision_claim_service_20261028_v3/token').read_bytes())";r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HOSTS['N2'],'/opt/venv_eval/bin/python','-B','-c',shlex.quote(query)],capture_output=True,timeout=15);assert r.returncode==0;put(root/'token',r.stdout);entries=[]
    for oldentry in d['configs']:
        pair=oldentry['gpus'];holders=[]
        for g in pair:
            uuid=m.b.uuid_for(g);a=[x for x in m.b.apps() if x['uuid']==uuid];assert len(a)==1,'Exact restored authorized holder only';i=m.b.identity(a[0]['pid']);i.update(gpu=g,uuid=uuid);m.b.check_holder(i,m.b.identity(i['pid']),g,uuid,m.b.apps());holders.append(i)
        phase='motion_revision_main_'+node+'_g'+''.join(map(str,pair));owner=node+'-g'+''.join(map(str,pair))+'-rv3main';cp=root/(phase+'_config.json');producer=dict(path=str(code/'motion_revision_native_manager_v1.py'),sha256=hashes[str(code/'motion_revision_native_manager_v1.py')]);c=dict(base);c.update(phase=phase,owner=owner,gpus=pair,local_root=str(root),out=str(root/'out'),queue=str(root/'queues'/phase),config_path=str(cp),holders=holders,targetcount=176771,input_count=176771,plan=dict(path=plan['path'],sha256=plan['sha256']),source_sha256=hashes,modelref=model,producer=producer,resource_manager_ref=producer,argv=['/opt/venv_vllm/bin/python','-B',producer['path'],'worker','--config',str(cp)],env=dict(base['env'],CUDA_VISIBLE_DEVICES=','.join(map(str,pair))),worker_state_layout='by_phase',revision=REVISION);cr=write_once(cp,c);m.check_sources(c,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(c);feed=write_once(root/(phase+'_feeder.json'),dict(out=c['out'],owner=owner,token_path=str(root/'token'),claim_endpoint='http://28.216.17.70:19112/claim',ffmpeg_path='/usr/local/bin/ffmpeg',claim_size=8));entries.append(dict(config_ref=cr,feeder_ref=feed,owner=owner,phase=phase,gpus=pair));(root/'out'/'ready'/owner).mkdir(parents=True,exist_ok=True)
    write_once(root/'deployment.json',dict(node=node,revision=REVISION,configs=entries,global_source_count=176771,new_source_claims_created=0,original_source_metadata_used=True,rich_role='dance_style_only',window_policy='merge_short_edges'));logs=root/'logs';logs.mkdir();sock=str(root/'tmux.sock');session='correct-motion-revision'
    def launch(name,args,first=False):
        cmd=shlex.join(args)+' > '+shlex.quote(str(logs/(name+'.log')))+' 2>&1';a=['tmux','-S',sock,'new-session','-d','-s',session,'-n',name,'-c','/tmp',cmd] if first else ['tmux','-S',sock,'new-window','-d','-t',session,'-n',name,'-c','/tmp',cmd];subprocess.run(a,check=True)
    launch('uploader',['/opt/venv_eval/bin/python','-B',str(code/'stage2_node_uploader_v1.py'),'--local-root',str(root/'out'),'--shared-root',str(ROOT/'motion_revision_english_merged_edges_final_20261028_v1/production_results'),'--node',node,'--seconds','86400'],True);previous=None
    for e in entries:
        if previous:
            c=checked(previous['config_ref']);state=root/'worker_state'/c['owner']/c['phase'];until=time.monotonic()+360
            while not (state/'load.receipt.json').exists():assert time.monotonic()<until and not (Path(c['queue'])/'failure.json').exists(),'Previous node model failed or still loading, no extra admission';time.sleep(1)
        c=checked(e['config_ref'])
        # Free only loaded original weight disk cache; never lower memory safety limits.
        for shard in Path('/opt/models/Qwen3-VL-32B-Instruct').glob('*.safetensors'):
            fd=os.open(shard,os.O_RDONLY|os.O_NOFOLLOW)
            try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally:os.close(fd)
        m.validate(c);launch(e['phase'],['/opt/venv_vllm/bin/python','-B',c['producer']['path'],'--config',c['config_path'],'--config-sha',e['config_ref']['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA]);until=time.monotonic()+30
        while not (Path(c['queue'])/'live.json').exists():assert time.monotonic()<until and not (Path(c['queue'])/'failure.json').exists();time.sleep(.2)
        launch(e['phase']+'-feeder',['/opt/venv_eval/bin/python','-B',str(code/'motion_revision_feeder_v1.py'),'--config',e['feeder_ref']['path']]);previous=e;print(json.dumps(dict(event='correct_full_revision_native_admitted',node=node,owner=c['owner'],gpus=c['gpus'],revision=REVISION,old_wrong_outputs_not_used=True)),flush=True)
    write_once(root/'node_launch_complete.json',dict(revision=REVISION,replicas=len(entries),source_count=176771,first3_protocol_proof=str(proof),old_artifacts_preserved=True,observed_unix=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=tuple(HOSTS));a=p.parse_args();run(a.node)
