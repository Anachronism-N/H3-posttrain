"""Resume ONLY N2 zero-model deployment after its old uploader evidence-generation stop."""
import hashlib,importlib.util,json,os,shlex,subprocess,sys,time
from pathlib import Path
RT=Path(__file__).parent
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225','N8':'28.216.18.136'}
REVISION='english_merged_edges_MS_LLM32B_v4'
SHARED=RT.parent/'MS_LLM32_english_merged_edges_v4'
def put(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())

def run(node):
    root=Path('/root/stage2_MS_LLM32_main_'+node+'_v4');code=root/'code';oldroot=Path('/root/stage2_motion_revision_final_main_'+node+'_20261028_v1');assert not (root/'deployment.json').exists() and not (root/'queues').exists(),'Never resume a model launch or existing claim'
    sys.path.insert(0,str(code));from stage2_local_publication_v1 import write_once,checked
    entries=json.loads((root/'before_old_drain.json').read_bytes())['old_configs'];proof=json.loads((root/'old_native_safe_ACK_closed.json').read_bytes());assert proof['all_old_results_ACKed'] is True
    p=oldroot/'out/UPLOAD_STOP_EVIDENCE1'
    if not p.exists():write_once(p,dict(reason='All old sources ACKed; completed evidence-stage uploader must naturally stop',models_called=0))
    until=time.monotonic()+30
    while True:
        live=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:a=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
            except (OSError,UnicodeError):continue
            if len(a)>2 and a[2].startswith(str(oldroot)+'/') and 'uploader' in Path(a[2]).name:live.append(proc.name)
        if not live:break
        assert time.monotonic()<until,'Old evidence uploader not stopped';time.sleep(.5)
    codec_path=code/'ms_llm32_codec_v4.py';raw=(RT/'ms_llm32_codec_textp1_v4.py').read_bytes();compile(raw,str(codec_path),'exec');write_once(root/'unused_initial_codec_preserved.json',dict(original_codec_sha256=hashlib.sha256(codec_path.read_bytes()).hexdigest(),original_text=codec_path.read_text(),new_model_calls=0,never_admitted_any_model=True));codec_path.write_bytes(raw)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')};spec=importlib.util.spec_from_file_location('text_native_manager',code/'ms_llm32_native_manager_v4.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    base=entries[0];model=put(root/'model_reference.json',Path(base['modelref']['path']).read_bytes());model['model_identity_sha256']=base['modelref']['model_identity_sha256'];hashes[model['path']]=model['sha256'];plan=put(root/'text_revision_plan.json',(SHARED/'revision_contract.json').read_bytes());hashes[plan['path']]=plan['sha256'];contract=json.loads(Path(plan['path']).read_bytes());count=contract['original_inputs'];assert contract['revision']==REVISION
    if node=='N2':token=Path('/root/stage2_MS_LLM32_claim_service_v4/token').read_bytes()
    else:
        program="import pathlib,sys;sys.stdout.buffer.write(pathlib.Path('/root/stage2_MS_LLM32_claim_service_v4/token').read_bytes())";r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HOSTS['N2'],'/opt/venv_eval/bin/python','-B','-c',shlex.quote(program)],capture_output=True,timeout=15);assert r.returncode==0;token=r.stdout
    put(root/'token',token);out=root/'out';out.mkdir(exist_ok=True);configs=[];pairs=[[2,3],[4,5],[6,7]] if node=='N2' else [c['gpus'] for c in entries]
    for pair in pairs:
        holders=[]
        for gpu in pair:
            uuid=m.b.uuid_for(gpu);apps=[a for a in m.b.apps() if a['uuid']==uuid];assert len(apps)==1,'Only verified restored holder may be reused';identity=m.b.identity(apps[0]['pid']);identity.update(gpu=gpu,uuid=uuid);m.b.check_holder(identity,m.b.identity(identity['pid']),gpu,uuid,m.b.apps());holders.append(identity)
        phase='MS_LLM32_'+node+'_g'+''.join(map(str,pair))+'_v4';owner=node+'-g'+''.join(map(str,pair))+'-msllm4';cp=root/(phase+'_config.json');producer=dict(path=str(code/'ms_llm32_native_manager_v4.py'),sha256=hashes[str(code/'ms_llm32_native_manager_v4.py')]);c=dict(base);c.update(phase=phase,owner=owner,gpus=pair,local_root=str(root),out=str(out),queue=str(root/'queues'/phase),config_path=str(cp),holders=holders,producer=producer,resource_manager_ref=producer,source_sha256=dict(hashes),modelref=model,plan=plan,targetcount=count,input_count=count,argv=['/opt/venv_vllm/bin/python','-B',producer['path'],'worker','--config',str(cp)],env=dict(base['env'],CUDA_VISIBLE_DEVICES=','.join(map(str,pair))),revision=REVISION,max_active_conversations=32,worker_state_layout='by_phase',release_authorized=True,new_stage_authorized=True)
        cr=write_once(cp,c);m.check_sources(c,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(c);fr=write_once(root/(phase+'_feeder.json'),dict(out=str(out),owner=owner,token_path=str(root/'token'),claim_endpoint='http://28.216.17.70:19116/claim'));configs.append(dict(config_ref=cr,feeder_ref=fr,gpus=pair,owner=owner,phase=phase));(out/'ready'/owner).mkdir(parents=True,exist_ok=True)
    write_once(out/'EXPECTED_OWNERS.json',dict(owners=[e['owner'] for e in configs]));write_once(root/'deployment.json',dict(node=node,revision=REVISION,configs=configs,source_count=count,model='32B language-only',image_inputs=0,video_inputs=0,created_unix=time.time()));uploader=write_once(root/'uploader.json',dict(out=str(out),node=node,shared_root=str(SHARED/'production_results'),relay_url='http://28.216.17.70:19116',token_path=str(root/'token')));logs=root/'logs';logs.mkdir();sock=root/'tmux.sock';session='MS-LLM32-v4'
    def win(name,args,first=False):
        log=logs/(name+'.log');assert not log.exists();command=shlex.join(args)+' > '+shlex.quote(str(log))+' 2>&1';subprocess.run(['tmux','-S',str(sock),'new-session','-d','-s',session,'-n',name,'-c','/tmp',command] if first else ['tmux','-S',str(sock),'new-window','-d','-t',session,'-n',name,'-c','/tmp',command],check=True)
    win('uploader',['/opt/venv_eval/bin/python','-B',str(code/'motion_revision_external_uploader_v1.py'),'--config',uploader['path']],True);previous=None
    for entry in configs:
        if previous:
            c=checked(previous['config_ref']);state=root/'worker_state'/c['owner']/c['phase'];until=time.monotonic()+420
            while not (state/'load.receipt.json').exists():assert time.monotonic()<until and not (Path(c['queue'])/'failure.json').exists();time.sleep(1)
        c=checked(entry['config_ref']);m.validate(c);win(c['phase'],['/opt/venv_vllm/bin/python','-B',c['producer']['path'],'--config',c['config_path'],'--config-sha',entry['config_ref']['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA]);until=time.monotonic()+40
        while not (Path(c['queue'])/'live.json').exists():assert time.monotonic()<until and not (Path(c['queue'])/'failure.json').exists();time.sleep(.2)
        win(c['phase']+'-feeder',['/opt/venv_eval/bin/python','-B',str(code/'ms_llm32_feeder_v4.py'),'--config',entry['feeder_ref']['path']]);previous=entry;print(json.dumps(dict(event='MS_LLM32_native_admitted',node=node,owner=c['owner'],gpus=c['gpus'],batch_size=32,model_loading=True,image_inputs=0)),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=('N3',));run(p.parse_args().node)
