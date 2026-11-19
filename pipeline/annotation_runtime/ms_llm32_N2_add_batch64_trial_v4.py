"""Add exactly one authorized N2g01 replica, alternate batch32/64, then wait for measured decision."""
import hashlib,importlib.util,json,os,shlex,subprocess,sys,time
from pathlib import Path
RT=Path(__file__).parent

def put(path,value):
 raw=(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n').encode();fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())

def run():
 root=Path('/root/stage2_MS_LLM32_main_N2_v4');sys.path.insert(0,str(root/'code'));from stage2_local_publication_v1 import checked,write_once
 deployment=json.loads((root/'deployment.json').read_bytes());assert all(e['owner']!='N2-g01-msllm4' for e in deployment['configs']);old_binding=json.loads((root/'runtime_bindings/N2-g23-msllm4.json').read_bytes());base=checked(old_binding['config_ref']);assert base['revision']=='english_merged_edges_MS_LLM32B_v4';pair=[0,1]
 for name in ['caption_producer_cached_n2p0_v1','curated_collaborative3_n2p1_v1']:
  safe=json.loads((RT.parent/'queues'/name/'resources_safe.json').read_bytes());assert all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored'))
 code=root/'batch64trialcode';code.mkdir(exist_ok=False);mapping={'ms_llm32_native_manager_batch64_v4.py':'ms_llm32_native_manager_batch64_v4.py','ms_llm32_codec_v4.py':'ms_llm32_codec_batch64_v4.py','ms_llm32_worker_batch64_v4.py':'ms_llm32_worker_batch64_v4.py','ms_llm32_feeder_textp2_v4.py':'ms_llm32_feeder_textp2_v4.py','stage2_local_publication_v1.py':'stage2_local_publication_v1.py','resource_manager.py':'resource_manager.py'};hashes={}
 for target,source in mapping.items():
  raw=(RT/source).read_bytes();compile(raw,str(code/target),'exec');path=code/target;fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  hashes[str(path)]=hashlib.sha256(raw).hexdigest()
 path=code/'ms_llm32_native_manager_batch64_v4.py';spec=importlib.util.spec_from_file_location('batch64trial_native',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);holders=[]
 for g in pair:
  uuid=m.b.uuid_for(g);apps=[x for x in m.b.apps() if x['uuid']==uuid];assert len(apps)==1;h=m.b.identity(apps[0]['pid']);h.update(gpu=g,uuid=uuid);m.b.check_holder(h,m.b.identity(h['pid']),g,uuid,m.b.apps());holders.append(h)
 owner='N2-g01-msllm4';phase='MS_LLM32_N2_g01_batch64trial_v4';cp=root/(phase+'_config.json');producer=dict(path=str(path),sha256=hashes[str(path)]);cfg=dict(base);cfg.update(phase=phase,owner=owner,gpus=pair,holders=holders,queue=str(root/'queues'/phase),config_path=str(cp),producer=producer,resource_manager_ref=producer,source_sha256=dict(base['source_sha256'],**hashes),argv=['/opt/venv_vllm/bin/python','-B',str(path),'worker','--config',str(cp)],env=dict(base['env'],CUDA_VISIBLE_DEVICES='0,1'),max_active_conversations=64,batch_decision_path=str(root/'batch64_measured_decision.json'))
 cr=write_once(cp,cfg);m.check_sources(cfg,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(cfg);m.validate(cfg);fr=write_once(root/(phase+'_feeder.json'),dict(out=str(root/'out'),owner=owner,token_path=str(root/'token'),claim_endpoint='http://28.216.17.70:19116/claim'));entry=dict(config_ref=cr,feeder_ref=fr,gpus=pair,owner=owner,phase=phase)
 write_once(root/'batch64_before_admission.json',dict(config_ref=cr,pattern=[32,64,32,64,32,64],finite_trial_source_count=288,other_17_replicas_untouched=True,new_source_stage_budget_once=True,quality_approved=False,observed_unix=time.time()))
 old=root/'out/EXPECTED_OWNERS.json';previous=json.loads(old.read_bytes());write_once(root/'expected_owners_before_batch64.json',previous);tmp=old.with_suffix('.batch64.tmp');write_once(tmp,dict(owners=previous['owners']+[owner]));os.replace(tmp,old)
 write_once(root/'deployment_before_batch64.json',deployment);deployment['configs'].append(entry);tmp=root/'deployment.batch64.tmp';write_once(tmp,deployment);os.replace(tmp,root/'deployment.json');write_once(root/'runtime_bindings'/(owner+'.json'),dict(config_ref=cr,trial_pattern=[32,64,32,64,32,64],await_measured_decision=True))
 log=root/'batch64triallogs';log.mkdir();sock=root/'batch64trial.sock';session='batch64-trial'
 def win(name,args,first=False):
  command=shlex.join(args)+' > '+shlex.quote(str(log/(name+'.log')))+' 2>&1';subprocess.run(['tmux','-S',str(sock),'new-session','-d','-s',session,'-n',name,'-c','/tmp',command] if first else ['tmux','-S',str(sock),'new-window','-d','-t',session,'-n',name,'-c','/tmp',command],check=True)
 win('manager',['/opt/venv_vllm/bin/python','-B',str(path),'--config',str(cp),'--config-sha',cr['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA],True);until=time.monotonic()+40
 while not (Path(cfg['queue'])/'live.json').exists():assert time.monotonic()<until and not (Path(cfg['queue'])/'failure.json').exists();time.sleep(.2)
 win('feeder',['/opt/venv_eval/bin/python','-B',str(code/'ms_llm32_feeder_textp2_v4.py'),'--config',fr['path']]);print(json.dumps(dict(event='N2_added_replica_batch64_trial',gpus=[0,1],model32B_BF16=True,patterns=[32,64,32,64,32,64],other_replicas_untouched=True,quality_approval_not_inferred=True)),flush=True)
if __name__=='__main__':run()
