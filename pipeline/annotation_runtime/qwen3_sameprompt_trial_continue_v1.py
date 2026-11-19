"""Continue only unstarted fixed-three diagnostic after CPU handoff; production remains live."""
import fcntl,hashlib,importlib.util,json,os,shlex,subprocess,sys,time
from pathlib import Path
RT=Path(__file__).parent
OLD=Path('/root/stage2_MS_LLM32_main_N2_v4')
DIAG=Path('/root/stage2_Qwen3_sameprompt_N2_v1')
SHARED=RT.parent/'MS_LLM32_english_merged_edges_v4/Qwen3_sameprompt_three_v1'

def run():
 sys.path.insert(0,str(OLD/'code'));from stage2_local_publication_v1 import checked,write_once
 assert not DIAG.exists();fd=os.open(OLD/'qwen3_pair_controller.lock',os.O_RDWR|os.O_NOFOLLOW);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);entries=json.loads((OLD/'qwenpair_resume_deployment.json').read_bytes())['configs'];before=json.loads((OLD/'qwen3_pair_before_drain.json').read_bytes());configs=[checked(ref) for ref in before['config_refs'].values()];base=next(c for c in configs if c['gpus']==[2,3]);code=OLD/'qwenpair_resume_code';hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')};manager=code/'ms_llm32_native_manager_qwenpair_resume.py';spec=importlib.util.spec_from_file_location('prod_qwenpair_resume',manager);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);logs=OLD/'qwenpair_resume_logs';sock=OLD/'qwenpair_resume_tmux.sock';session='qwenpair'
 def window(name,args,first=False):
  log=logs/(name+'.log');assert not log.exists();cmd=shlex.join(args)+' > '+shlex.quote(str(log))+' 2>&1';subprocess.run(['tmux','-S',str(sock),'new-window','-d','-t',session,'-n',name,'-c','/tmp',cmd],check=True)
 def bind(entry):
  path=OLD/'runtime_bindings'/(entry['owner']+'.json');write_once(OLD/'qwenpair_binding_history'/(entry['owner']+'.json'),json.loads(path.read_bytes()));tmp=path.with_suffix('.qwenpair.tmp');write_once(tmp,dict(config_ref=entry['config_ref']));os.replace(tmp,path)
 def start_production(entry):
  cfg=checked(entry['config_ref']);m.validate(cfg);bind(entry);window(cfg['phase'],['/opt/venv_vllm/bin/python','-B',str(manager),'--config',cfg['config_path'],'--config-sha',entry['config_ref']['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA]);until=time.monotonic()+40
  while not (Path(cfg['queue'])/'live.json').exists():assert time.monotonic()<until and not (Path(cfg['queue'])/'failure.json').exists();time.sleep(.2)
  window(cfg['phase']+'-feeder',['/opt/venv_eval/bin/python','-B',str(code/'ms_llm32_feeder_qwenpair_resume.py'),'--config',entry['feeder_ref']['path']]);return cfg
 for entry in [e for e in entries if e['gpus']!=[0,1]]:
  c=checked(entry['config_ref']);q=Path(c['queue'])
  if not q.exists():start_production(entry)
  until=time.monotonic()+420
  while not (OLD/'worker_state'/c['owner']/c['phase']/'load.receipt.json').exists():assert time.monotonic()<until and not (q/'failure.json').exists();time.sleep(1)
 DIAG.mkdir(exist_ok=False);dcode=DIAG/'code';dcode.mkdir();dhash={}
 for target,source in {'qwen3_sameprompt_native_manager_v1.py':RT/'qwen3_sameprompt_native_manager_v1.py','ms_llm32_codec_v4.py':code/'ms_llm32_codec_v4.py','stage2_local_publication_v1.py':RT/'stage2_local_publication_v1.py','resource_manager.py':RT/'resource_manager.py'}.items():
  raw=source.read_bytes();compile(raw,str(dcode/target),'exec');p=dcode/target;f=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(f,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
  dhash[str(p)]=hashlib.sha256(raw).hexdigest()
 input_manifest=json.loads((SHARED/'inputs_ready.json').read_bytes());local_inputs=[]
 for ref in input_manifest['entries']:
  value=checked(ref);local_inputs.append(write_once(DIAG/(value['id']+'.input.json'),value))
 ir=write_once(DIAG/'inputs.json',dict(source_count=3,entries=local_inputs,new_source_claims=0));model=Path('/opt/models/Qwen3-32B');files={str(p.relative_to(model)):p.stat().st_size for p in model.iterdir() if p.is_file()};need_files=25;assert len(files)==need_files;identity=dict(root=str(model),files=files,config_sha256=m.b.sha(model/'config.json'),index_sha256=m.b.sha(model/'model.safetensors.index.json'));identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();modelref=write_once(DIAG/'model_reference.json',dict(identity,model_identity_sha256=identity_sha));modelref['model_identity_sha256']=identity_sha;planref=write_once(DIAG/'source_plan.json',dict(source_count=3,source_ids=[checked(ref)['id'] for ref in local_inputs],same_prompt=True));dhash[modelref['path']]=modelref['sha256'];dhash[planref['path']]=planref['sha256']
 path=dcode/'qwen3_sameprompt_native_manager_v1.py';spec=importlib.util.spec_from_file_location('pure_qwen_native',path);dm=importlib.util.module_from_spec(spec);spec.loader.exec_module(dm);holders=[]
 for g in [0,1]:
  uuid=dm.b.uuid_for(g);apps=[a for a in dm.b.apps() if a['uuid']==uuid];assert len(apps)==1;h=dm.b.identity(apps[0]['pid']);h.update(gpu=g,uuid=uuid);dm.b.check_holder(h,dm.b.identity(h['pid']),g,uuid,dm.b.apps());holders.append(h)
 phase='Qwen3_sameprompt_three_N2g01_v1';cp=DIAG/(phase+'_config.json');producer=dict(path=str(path),sha256=dhash[str(path)]);cfg=dict(base);cfg.update(phase=phase,owner='N2-g01-qwen3diag',gpus=[0,1],local_root=str(DIAG),out=str(DIAG/'out'),queue=str(DIAG/'queues'/phase),config_path=str(cp),holders=holders,producer=producer,resource_manager_ref=producer,source_sha256=dhash,modelref=modelref,plan=planref,targetcount=3,input_count=3,diagnostic_manifest_ref=ir,argv=['/opt/venv_vllm/bin/python','-B',str(path),'worker','--config',str(cp)],env=dict(base['env'],CUDA_VISIBLE_DEVICES='0,1'),max_active_conversations=32,diagnostic_only=True)
 cr=write_once(cp,cfg);dm.check_sources(cfg,cp,cr['sha256'],dm.ORIGINAL_BASE_SHA);dm.pure_model_check(cfg);dm.validate(cfg);write_once(DIAG/'deployment.json',dict(config_ref=cr,same_prompt=True,cases=3,no_pool_claims=True));print(json.dumps(dict(event='Qwen3_sameprompt_native_admitted',gpus=[0,1],other17_production_restored=True)),flush=True)
 window('pure_qwen3_diagnostic',['/opt/venv_vllm/bin/python','-B',str(path),'--config',str(cp),'--config-sha',cr['sha256'],'--manager-sha',dm.ORIGINAL_BASE_SHA])
 safe_path=Path(cfg['queue'])/'resources_safe.json';until=time.monotonic()+1200
 while not safe_path.exists():
  assert time.monotonic()<until,'Diagnostic native cleanup still pending'
  if (Path(cfg['queue'])/'failure.json').exists():raise RuntimeError((Path(cfg['queue'])/'worker.runtime.log').read_text()[-2000:])
  time.sleep(2)
 safe=json.loads(safe_path.read_bytes());assert all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored'));result=DIAG/'comparison_results.json';assert result.exists(),'Diagnostic exited without results; original evidence preserved';raw=result.read_bytes();v=json.loads(raw);(SHARED/'comparison_results.json').write_bytes(raw)
 for row in v['rows']:
  r=checked(row['result_ref']);(SHARED/(row['id']+'.qwen3_result.json')).write_text(json.dumps(r,ensure_ascii=False))
 first=next(e for e in entries if e['gpus']==[0,1]);c=checked(first['config_ref']);updated=[]
 for g in [0,1]:
  uuid=m.b.uuid_for(g);apps=[a for a in m.b.apps() if a['uuid']==uuid];assert len(apps)==1;h=m.b.identity(apps[0]['pid']);h.update(gpu=g,uuid=uuid);updated.append(h)
 # Only the never-started production config is superseded after diagnostic native restoration.
 c['holders']=updated;c['phase']+='_afterdiag';c['config_path']=str(OLD/(c['phase']+'_config.json'));c['queue']=str(OLD/'queues'/c['phase']);c['argv']=['/opt/venv_vllm/bin/python','-B',str(manager),'worker','--config',c['config_path']];cr=write_once(c['config_path'],c);entry=dict(first,config_ref=cr,phase=c['phase']);start_production(entry);print(json.dumps(dict(event='Qwen3_sameprompt_complete_and_all_production_resumed',results=str(SHARED/'comparison_results.json'),full_quality_approval=False)),flush=True);os.close(fd)
if __name__=='__main__':run()
