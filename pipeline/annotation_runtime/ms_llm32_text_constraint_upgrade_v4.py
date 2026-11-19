"""Amend wording constraints without touching original claims or replaying model calls."""
import argparse,fcntl,hashlib,importlib.util,json,os,shlex,subprocess,sys,time
from pathlib import Path
RT=Path(__file__).parent

def run(node):
 root=Path('/root/stage2_MS_LLM32_main_'+node+'_v4');deadline=time.monotonic()+3600
 while not (root/'deployment.json').exists():assert time.monotonic()<deadline,'Initial node still awaiting old native-safe';time.sleep(2)
 sys.path.insert(0,str(root/'code'));from stage2_local_publication_v1 import checked,write_once
 fd=os.open(root/'textp1_upgrade.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);deployment=json.loads((root/'deployment.json').read_bytes());configs=[checked(e['config_ref']) for e in deployment['configs']]
 # Original launch controller must finish sequential loading before old feeders stop.
 zero_stage=all(not Path(c['queue']).exists() for c in configs) and not list((root/'out/feeder_claims').glob('*/*.json'))
 if zero_stage:assert json.loads((root/'old_native_safe_ACK_closed.json').read_bytes())['all_old_results_ACKed']
 while not zero_stage and not all((Path(c['queue'])/'live.json').exists() for c in configs):assert time.monotonic()<deadline and not any((Path(c['queue'])/'failure.json').exists() for c in configs);time.sleep(1)
 write_once(root/'textp1_before_drain.json',dict(old_configs=deployment['configs'],reason='First batch hard sentence/length gate produced format rejection; preserve raw and use softer wording, no semantic-error claim',no_model_retry=True,observed_unix=time.time()))
 stop=root/'out/FEEDER_STOP'
 if not stop.exists():write_once(stop,dict(reason='Constraint amendment; finish existing source packets, no new source claims',models_called=0))
 while True:
  ready=True
  for c in configs:
   q=Path(c['queue']);p=q/'resources_safe.json'
   if zero_stage and not q.exists():continue
   if (q/'failure.json').exists() and not p.exists():raise RuntimeError('Original text worker failed; preserve and investigate '+str(q))
   if not p.exists():ready=False;continue
   s=json.loads(p.read_bytes());assert all(s.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored')) and json.loads((q/'worker_exit.json').read_bytes())['exit_code']==0
  if ready:break
  assert time.monotonic()<deadline,'Current packets still draining';time.sleep(2)
 while True:
  keys={tuple(p.relative_to(root/'out/sources').parts[:2]) for p in (root/'out/sources').glob('*/*/commit.json')};acked=set();pending=False
  for d in (root/'out/outbox').iterdir():
   if not (d/'sealed.json').exists():continue
   seal=json.loads((d/'sealed.json').read_bytes())
   if not (d/'upload_receipt.json').exists():pending=True;continue
   a=json.loads((d/'upload_receipt.json').read_bytes());assert a['batch_id']==seal['batch_id'] and a['archive_sha256']==seal['archive_sha256'];acked.update(map(tuple,seal['source_keys']))
  if not pending and keys==acked:break
  assert time.monotonic()<deadline,'Current source results still awaiting shared ACK';time.sleep(2)
 write_once(root/'textp1_old_safe_ACK.json',dict(source_count=len(keys),all_old_source_outputs_ACKed=True,new_model_calls=0))
 p=root/'out/UPLOAD_STOP'
 if not p.exists():write_once(p,dict(reason='Original stage all safe and ACKed; single uploader handoff'))
 until=time.monotonic()+30
 while True:
  found=False
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:a=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
   except (OSError,UnicodeError):continue
   if '--config' in a and a[a.index('--config')+1]==str(root/'uploader.json'):found=True
  if not found:break
  assert time.monotonic()<until,'Old node uploader still live';time.sleep(.5)
 code=root/'textp1code';code.mkdir(exist_ok=False);mapping={'ms_llm32_native_manager_textp1_v4.py':'ms_llm32_native_manager_textp1_v4.py','ms_llm32_codec_v4.py':'ms_llm32_codec_textp1_v4.py','ms_llm32_worker_v4.py':'ms_llm32_worker_v4.py','ms_llm32_feeder_textp1_v4.py':'ms_llm32_feeder_textp1_v4.py','ms_llm32_uploader_textp1_v4.py':'ms_llm32_uploader_textp1_v4.py','stage2_local_publication_v1.py':'stage2_local_publication_v1.py','resource_manager.py':'resource_manager.py'};hashes={}
 for target,source in mapping.items():
  raw=(RT/source).read_bytes();compile(raw,str(code/target),'exec');p=code/target;h=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(h,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  hashes[str(p)]=hashlib.sha256(raw).hexdigest()
 path=code/'ms_llm32_native_manager_textp1_v4.py';spec=importlib.util.spec_from_file_location('native_textp1',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);entries=[]
 for e,old in zip(deployment['configs'],configs):
  holders=[]
  for g in old['gpus']:
   uuid=m.b.uuid_for(g);apps=[a for a in m.b.apps() if a['uuid']==uuid];assert len(apps)==1;h=m.b.identity(apps[0]['pid']);h.update(gpu=g,uuid=uuid);m.b.check_holder(h,m.b.identity(h['pid']),g,uuid,m.b.apps());holders.append(h)
  phase=old['phase']+'_textp1';cp=root/(phase+'_config.json');producer=dict(path=str(path),sha256=hashes[str(path)]);c=dict(old);c.update(phase=phase,queue=str(root/'queues'/phase),config_path=str(cp),holders=holders,producer=producer,resource_manager_ref=producer,source_sha256=dict(old['source_sha256'],**hashes),argv=['/opt/venv_vllm/bin/python','-B',str(path),'worker','--config',str(cp)],text_constraint_amendment='short literal original-event wording; no hard sentence count; complete sentence schema; no semantic re-audit')
  cr=write_once(cp,c);m.check_sources(c,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(c);fr=write_once(root/(phase+'_feeder.json'),checked(e['feeder_ref']));entries.append(dict(config_ref=cr,feeder_ref=fr,owner=c['owner'],gpus=c['gpus'],phase=phase))
 write_once(root/'textp1_deployment.json',dict(configs=entries,old_budgets_preserved=True,automatic_retry=False,observed_unix=time.time()));up=json.loads((root/'uploader.json').read_bytes());ur=write_once(root/'textp1_uploader.json',up);logs=root/'textp1logs';logs.mkdir();sock=root/'textp1_tmux.sock';session='textp1'
 def win(name,args,first=False):
  cmd=shlex.join(args)+' > '+shlex.quote(str(logs/(name+'.log')))+' 2>&1';subprocess.run(['tmux','-S',str(sock),'new-session','-d','-s',session,'-n',name,'-c','/tmp',cmd] if first else ['tmux','-S',str(sock),'new-window','-d','-t',session,'-n',name,'-c','/tmp',cmd],check=True)
 win('uploader',['/opt/venv_eval/bin/python','-B',str(code/'ms_llm32_uploader_textp1_v4.py'),'--config',ur['path']],True);previous=None
 for e in entries:
  if previous:
   c=checked(previous['config_ref']);until=time.monotonic()+420
   while not (root/'worker_state'/c['owner']/c['phase']/'load.receipt.json').exists():assert time.monotonic()<until and not (Path(c['queue'])/'failure.json').exists();time.sleep(1)
  c=checked(e['config_ref']);until=time.monotonic()+1200
  while m.headroom()['headroom']<160*2**30:assert time.monotonic()<until,'Memory admission still below160GiB; no holder released';time.sleep(5)
  m.validate(c);write_once(root/'runtime_bindings'/(c['owner']+'.json'),dict(config_ref=e['config_ref'],previous_config_ref=next(x['config_ref'] for x in deployment['configs'] if x['owner']==c['owner'])));win(c['phase'],['/opt/venv_vllm/bin/python','-B',str(path),'--config',c['config_path'],'--config-sha',e['config_ref']['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA]);until=time.monotonic()+40
  while not (Path(c['queue'])/'live.json').exists():assert time.monotonic()<until and not (Path(c['queue'])/'failure.json').exists();time.sleep(.2)
  win(c['phase']+'-feeder',['/opt/venv_eval/bin/python','-B',str(code/'ms_llm32_feeder_textp1_v4.py'),'--config',e['feeder_ref']['path']]);previous=e;print(json.dumps(dict(event='text_constraint_fixed_replica_admitted',node=node,owner=c['owner'],gpus=c['gpus'])),flush=True)
 os.close(fd)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=('N1','N2','N3','N6','N8'));run(p.parse_args().node)
