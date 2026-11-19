"""Finish fresh corrected three-case provisioning; never restart an existing model queue."""
import pathlib,json,os,hashlib,sys,subprocess,shlex,time
rt=pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime');root=pathlib.Path('/root/stage2_motion_revision_pilot3_corrected_N2_20261028_v1');root.mkdir(exist_ok=True);code=root/'code';code.mkdir(exist_ok=True);hashes={}
def put(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw;return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
for n in ('motion_revision_native_manager_v1.py','motion_revision_model_worker_v1.py','motion_revision_codec_v1.py','motion_revision_feeder_v1.py','stage2_local_publication_v1.py','stage2_node_uploader_v1.py','resource_manager.py'):
 raw=(rt/('motion_revision_codec_v2.py' if n=='motion_revision_codec_v1.py' else n)).read_bytes();compile(raw,str(code/n),'exec');r=put(code/n,raw);hashes[r['path']]=r['sha256']
sys.path.insert(0,str(code));import motion_revision_native_manager_v1 as m;from stage2_local_publication_v1 import write_once
old=pathlib.Path('/root/stage2_motion_revision_pilot3_N2_20261028_v1');template=json.loads((old/'manager.json').read_bytes());s=json.loads((pathlib.Path(template['queue'])/'resources_safe.json').read_bytes());assert all(s[k] is True for k in ('all_holders_safe','locks_released','leases_restored'))
for p in (old/'out'/'outbox').iterdir():assert (p/'upload_receipt.json').exists()
if not (old/'out'/'UPLOAD_STOP').exists():write_once(old/'out'/'UPLOAD_STOP',dict(reason='First3 protocol rejected; corrected natural-motion candidate independent outputs',models_called=0))
deadline=time.monotonic()+15
while True:
 running=False
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:a=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  except (OSError,UnicodeDecodeError):continue
  if len(a)>2 and a[2]==str(old/'code/stage2_node_uploader_v1.py'):running=True
 if not running:break
 assert time.monotonic()<deadline;time.sleep(.5)
model=put(root/'model_reference.json',pathlib.Path(template['modelref']['path']).read_bytes());model.pop('bytes');model['model_identity_sha256']=template['modelref']['model_identity_sha256'];hashes[model['path']]=model['sha256'];plan=put(root/'pilot_plan.json',(rt.parent/'motion_revision_english_merged_edges_corrected_20261028_v1/revision_contract.json').read_bytes());hashes[plan['path']]=plan['sha256'];put(root/'token',pathlib.Path('/root/stage2_motion_revision_claim_service_20261028_v2/token').read_bytes());holders=[]
for g in (2,3):
 uuid=m.b.uuid_for(g);apps=[x for x in m.b.apps() if x['uuid']==uuid];assert len(apps)==1;i=m.b.identity(apps[0]['pid']);i.update(gpu=g,uuid=uuid);m.b.check_holder(i,m.b.identity(i['pid']),g,uuid,m.b.apps());holders.append(i)
phase='motion_revision_corrected_pilot3_N2_g23';cp=root/'manager.json';producer=dict(path=str(code/'motion_revision_native_manager_v1.py'),sha256=hashes[str(code/'motion_revision_native_manager_v1.py')]);cfg=dict(template);cfg.update(phase=phase,local_root=str(root),out=str(root/'out'),queue=str(root/'queues'/phase),config_path=str(cp),holders=holders,plan=dict(path=plan['path'],sha256=plan['sha256']),source_sha256=hashes,modelref=model,producer=producer,resource_manager_ref=producer,argv=['/opt/venv_vllm/bin/python','-B',producer['path'],'worker','--config',str(cp)],revision='english_merged_edges_motion_revision_v2');cr=write_once(cp,cfg);m.check_sources(cfg,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(cfg);m.validate(cfg);fr=write_once(root/'feeder.json',dict(out=cfg['out'],owner=cfg['owner'],token_path=str(root/'token'),claim_endpoint='http://28.216.17.70:19111/claim',ffmpeg_path='/usr/local/bin/ffmpeg',claim_size=3,one_packet_only=True));write_once(root/'deployment.json',dict(node='N2',configs=[dict(config_ref=cr,feeder_ref=fr,owner=cfg['owner'],phase=phase,gpus=[2,3])],revision=cfg['revision'],source_count=3));(root/'out'/'ready'/cfg['owner']).mkdir(parents=True);sock=str(root/'tmux.sock');session='correct-pilot3v2'
def win(name,args,first=False):
 cmd=shlex.join(args)+' > '+shlex.quote(str(root/(name+'.log')))+' 2>&1';subprocess.run(['tmux','-S',sock,'new-session','-d','-s',session,'-n',name,'-c','/tmp',cmd] if first else ['tmux','-S',sock,'new-window','-d','-t',session,'-n',name,'-c','/tmp',cmd],check=True)
win('manager',['/opt/venv_vllm/bin/python','-B',producer['path'],'--config',str(cp),'--config-sha',cr['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA],True);deadline=time.monotonic()+30
while not (pathlib.Path(cfg['queue'])/'live.json').exists():assert time.monotonic()<deadline and not (pathlib.Path(cfg['queue'])/'failure.json').exists();time.sleep(.2)
win('uploader',['/opt/venv_eval/bin/python','-B',str(code/'stage2_node_uploader_v1.py'),'--local-root',cfg['out'],'--shared-root',str(rt.parent/'motion_revision_english_merged_edges_corrected_20261028_v1/results'),'--node','N2','--seconds','21600']);win('feeder',['/opt/venv_eval/bin/python','-B',str(code/'motion_revision_feeder_v1.py'),'--config',fr['path']]);print(json.dumps(dict(corrected_first3_admitted=True,revision=cfg['revision'],source_count=3,full_production_not_yet_admitted=True,old_outputs_preserved=True)))
