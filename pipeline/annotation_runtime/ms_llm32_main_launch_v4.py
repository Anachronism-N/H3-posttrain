"""Safely replace old visual wording with32B text-only single-stage production."""
import argparse,hashlib,importlib.util,json,os,shlex,subprocess,sys,time
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
    assert HOSTS[node] in subprocess.check_output(['hostname','-I'],text=True).split();oldroot=Path('/root/stage2_motion_revision_final_main_'+node+'_20261028_v1');olddep=json.loads((oldroot/'deployment.json').read_bytes());entries=[]
    for e in olddep['configs']:
        bp=oldroot/'runtime_bindings'/(e['owner']+'.json');ref=json.loads(bp.read_bytes())['config_ref'] if bp.exists() else e['config_ref'];raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];entries.append(json.loads(raw))
    root=Path('/root/stage2_MS_LLM32_main_'+node+'_v4')
    if node!='N2':root.mkdir(exist_ok=False)
    else:assert root.exists() and not (root/'deployment.json').exists(),'N2seed root exists but model deployment must be fresh'
    code=root/'code';code.mkdir();hashes={}
    for name in ('ms_llm32_native_manager_v4.py','ms_llm32_worker_v4.py','ms_llm32_codec_v4.py','ms_llm32_feeder_v4.py','motion_revision_external_uploader_v1.py','stage2_local_publication_v1.py','resource_manager.py'):
        raw=(RT/name).read_bytes();compile(raw,str(code/name),'exec');ref=put(code/name,raw);hashes[ref['path']]=ref['sha256']
    sys.path.insert(0,str(code));from stage2_local_publication_v1 import write_once,checked
    spec=importlib.util.spec_from_file_location('text_native_manager',code/'ms_llm32_native_manager_v4.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    write_once(root/'before_old_drain.json',dict(node=node,old_configs=entries,revision=REVISION,old_claims_budgets_results_preserved=True,no_N4_operation=True,observed_unix=time.time()))
    for name in {('FEEDER_STOP_BATCH16' if c.get('max_active_conversations',8)>8 else 'FEEDER_STOP') for c in entries}:
        p=oldroot/'out'/name
        if not p.exists():write_once(p,dict(reason='User approved MS+32B text-only wording, drain existing visual claims without retry',new_revision=REVISION))
    until=time.monotonic()+3600
    while True:
        complete=True
        for c in entries:
            q=Path(c['queue']);p=q/'resources_safe.json'
            if (q/'failure.json').exists() and not p.exists():raise RuntimeError('Old manager failure; preserve and investigate '+str(q))
            if not p.exists():complete=False;continue
            safe=json.loads(p.read_bytes());assert all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored'))
        if complete:break
        assert time.monotonic()<until,'Old node still draining, no new model budget';time.sleep(3)
    if not (oldroot/'out/GENERATION_DRAINED').exists():write_once(oldroot/'out/GENERATION_DRAINED',dict(all_current_native_managers_safe=True,reason='truthful final tail flush; old result evidence preserved'))
    while True:
        keys={tuple(p.relative_to(oldroot/'out/sources').parts[:2]) for p in (oldroot/'out/sources').glob('*/*/commit.json')};acks=set();pending=False
        for directory in (oldroot/'out/outbox').iterdir():
            p=directory/'sealed.json'
            if not p.exists():continue
            seal=json.loads(p.read_bytes());a=directory/'upload_receipt.json'
            if not a.exists():pending=True;continue
            receipt=json.loads(a.read_bytes());assert seal['batch_id']==receipt['batch_id'] and seal['archive_sha256']==receipt['archive_sha256'];acks.update(map(tuple,seal['source_keys']))
        if not pending and keys==acks:break
        assert time.monotonic()<until,'Old results not yet reliable ACK; no rerun';time.sleep(3)
    write_once(root/'old_native_safe_ACK_closed.json',dict(old_configs=entries,old_source_count=len(keys),all_old_results_ACKed=True,observed_unix=time.time()))
    for name in ('UPLOAD_STOP','UPLOAD_STOP_BATCH16','UPLOAD_STOP_EVIDENCE1'):
        p=oldroot/'out'/name
        if not p.exists():write_once(p,dict(reason='Old node all safe/ACK; handoff single uploader to text stage',models_called=0))
    # Wait for only old-node upload scripts; do not signal or stop unknown tasks.
    until=time.monotonic()+30
    while True:
        live=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:a=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
            except (OSError,UnicodeError):continue
            if len(a)>2 and str(oldroot)+'/' in a[2] and ('uploader' in Path(a[2]).name):live.append(proc.name)
        if not live:break
        assert time.monotonic()<until,'Old uploader still live; do not duplicate';time.sleep(.5)
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
    p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=tuple(HOSTS));run(p.parse_args().node)
