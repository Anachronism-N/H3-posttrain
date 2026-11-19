"""One node safe upgrade to16 original conversations; old evidence and budgets immutable."""
import argparse
import fcntl,hashlib,importlib.util,json,os,shlex,subprocess,sys,time
from pathlib import Path
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime')
ROOT=RT.parent

def run(node):
 root=Path('/root/stage2_motion_revision_final_main_'+node+'_20261028_v1');sys.path.insert(0,str(root/'code'));from stage2_local_publication_v1 import checked,write_once
 fd=os.open(root/'batch16_upgrade.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);d=json.loads((root/'deployment.json').read_bytes());deadline=time.monotonic()+3600
 write_once(root/'batch16_waiting_original_safe.json',dict(node=node,old_configs=d['configs'],revision_unchanged=True,prompt_frames_windows_unchanged=True,source_claims_reassigned=False,observed_unix=time.time()));print(json.dumps(dict(event='batch16_upgrade_waiting_native_safe',node='N2',old_replicas=2)),flush=True)
 while True:
  ready=True
  for e in d['configs']:
   c=checked(e['config_ref']);q=Path(c['queue'])
   if (q/'failure.json').exists():raise RuntimeError('Original model failed; investigate before upgrade')
   if not (q/'resources_safe.json').exists():ready=False;continue
   s=json.loads((q/'resources_safe.json').read_bytes());assert all(s.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored')) and json.loads((q/'worker_exit.json').read_bytes())['exit_code']==0
  if ready:
   keys={tuple(p.relative_to(root/'out/sources').parts[:2]) for p in (root/'out/sources').glob('*/*/commit.json')};acks=set()
   for p in (root/'out/outbox').iterdir():
    if not (p/'sealed.json').exists() or not (p/'upload_receipt.json').exists():ready=False;break
    seal=json.loads((p/'sealed.json').read_bytes());ack=json.loads((p/'upload_receipt.json').read_bytes());assert seal['batch_id']==ack['batch_id'] and seal['archive_sha256']==ack['archive_sha256'];acks.update(map(tuple,seal['source_keys']))
   ready=ready and keys==acks
  if ready:break
  assert time.monotonic()<deadline,'Original node still draining; no new model budget';time.sleep(3)
 write_once(root/'batch16_original_safe_ACK_closed.json',dict(all_original_managers_safe=True,source_count=len(keys),all_original_source_commits_ACKed=True,source_claims_preserved=True,observed_unix=time.time()))
 if not (root/'out/UPLOAD_STOP').exists():write_once(root/'out/UPLOAD_STOP',dict(reason='Old8source models safe and all results ACKed; one node uploader handoff to16batch generation markers',model_calls=0))
 until=time.monotonic()+20
 while True:
  live=False
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:a=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
   except (OSError,UnicodeDecodeError):continue
   if len(a)>2 and a[2]==str(root/'code/stage2_node_uploader_v1.py'):live=True
  if not live:break
  assert time.monotonic()<until,'Old CPU uploader still running';time.sleep(.5)
 code=root/'batch16code';code.mkdir(exist_ok=False);mapping={'motion_revision_native_manager_batch16_v1.py':'motion_revision_native_manager_batch16_v1.py','motion_revision_model_worker_batch16_v1.py':'motion_revision_model_worker_batch16_v1.py','motion_revision_codec_v1.py':'motion_revision_codec_batch16_v1.py','motion_revision_feeder_batch16_v1.py':'motion_revision_feeder_batch16_v1.py','motion_revision_uploader_batch16_v1.py':'motion_revision_uploader_batch16_v1.py','stage2_local_publication_v1.py':'stage2_local_publication_v1.py','resource_manager.py':'resource_manager.py'};hashes={}
 for target,source in mapping.items():
  raw=(RT/source).read_bytes();compile(raw,str(code/target),'exec');p=code/target;fdout=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fdout,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  hashes[str(p)]=hashlib.sha256(raw).hexdigest()
 spec=importlib.util.spec_from_file_location('native_batch16',code/'motion_revision_native_manager_batch16_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);entries=[]
 for e in d['configs']:
  old=checked(e['config_ref']);pair=old['gpus'];holders=[]
  for g in pair:
   uuid=m.b.uuid_for(g);apps=[x for x in m.b.apps() if x['uuid']==uuid];assert len(apps)==1;holder=m.b.identity(apps[0]['pid']);holder.update(gpu=g,uuid=uuid);m.b.check_holder(holder,m.b.identity(holder['pid']),g,uuid,m.b.apps());holders.append(holder)
  phase=old['phase']+'_batch16trial1';cp=root/(phase+'_config.json');producer=dict(path=str(code/'motion_revision_native_manager_batch16_v1.py'),sha256=hashes[str(code/'motion_revision_native_manager_batch16_v1.py')]);c=dict(old);c.update(phase=phase,queue=str(root/'queues'/phase),config_path=str(cp),holders=holders,producer=producer,resource_manager_ref=producer,source_sha256=dict(old['source_sha256'],**hashes),argv=['/opt/venv_vllm/bin/python','-B',producer['path'],'worker','--config',str(cp)],worker_state_layout='by_phase',max_active_conversations=16,model_microclaim_count=2);cr=write_once(cp,c);m.check_sources(c,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(c);f=checked(e['feeder_ref']);f['prefetch_packets']=8;fr=write_once(root/(phase+'_feeder.json'),f);entry=dict(config_ref=cr,feeder_ref=fr,gpus=pair,owner=c['owner'],phase=phase);entries.append(entry)
 write_once(root/'batch16_trial_deployment.json',dict(configs=entries,revision=d['revision'],same_original_node_out_and_owners=True,same_original_model_database=True,old_outputs_budgets_preserved=True,created_unix=time.time()));sock=root/'batch16_tmux.sock';session='batch16-trial';logs=root/'batch16logs';logs.mkdir()
 def win(name,args,first=False):
  command=shlex.join(args)+' > '+shlex.quote(str(logs/(name+'.log')))+' 2>&1';subprocess.run(['tmux','-S',str(sock),'new-session','-d','-s',session,'-n',name,'-c','/tmp',command] if first else ['tmux','-S',str(sock),'new-window','-d','-t',session,'-n',name,'-c','/tmp',command],check=True)
 win('uploader',['/opt/venv_eval/bin/python','-B',str(code/'motion_revision_uploader_batch16_v1.py'),'--local-root',str(root/'out'),'--shared-root',str(ROOT/'motion_revision_english_merged_edges_final_20261028_v1/production_results'),'--node',node,'--seconds','86400'],True);previous=None
 for e in entries:
  if previous:
   c=checked(previous['config_ref']);until=time.monotonic()+360;state=root/'worker_state'/c['owner']/c['phase']
   while not (state/'load.receipt.json').exists():assert time.monotonic()<until and not (Path(c['queue'])/'failure.json').exists();time.sleep(1)
  c=checked(e['config_ref']);m.validate(c);write_once(root/'runtime_bindings'/(c['owner']+'.json'),dict(config_ref=e['config_ref'],previous_config_ref=next(x['config_ref'] for x in d['configs'] if x['owner']==c['owner']),state_layout='by_phase',prompt_and_frames_unchanged=True,batch_size=16,source_claim_reassignment=False));win(c['phase'],['/opt/venv_vllm/bin/python','-B',c['producer']['path'],'--config',c['config_path'],'--config-sha',e['config_ref']['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA]);until=time.monotonic()+40
  while not (Path(c['queue'])/'live.json').exists():assert time.monotonic()<until and not (Path(c['queue'])/'failure.json').exists();time.sleep(.2)
  win(c['phase']+'-feeder',['/opt/venv_eval/bin/python','-B',str(code/'motion_revision_feeder_batch16_v1.py'),'--config',e['feeder_ref']['path']]);previous=e;print(json.dumps(dict(event='original_revision_batch16_trial_admitted',owner=c['owner'],gpus=c['gpus'],worker_identity=json.loads((Path(c['queue'])/'live.json').read_bytes())['child_identity'],prompt_and_window_unchanged=True,source_reclaims=0)),flush=True)
 os.close(fd)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=('N1','N3','N6','N8'));run(p.parse_args().node)
