import argparse, datetime, hashlib, importlib.util, json, os, pathlib, select, subprocess, sys, traceback
P=pathlib.Path
MS=P('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS')
OUT=MS/'n7_omni6k_u3000_eval_n6_v1';QUEUE=OUT/'queue_retry_projector_fp32_v2'
STAGE=MS/'n7_omni6k_u3000_stage_n6_v1.py'
INFER=MS/'n7_omni6k_u3000_infer_n6_v1.py'
PY='/opt/venv_h3/bin/python'
H3=P('/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/scripts_omnidance/exp/ms/h3_video_infer.py')
H3SHA='77bafd9f6165f80328d4e272c901b21fbf45da3898161bcc039a572e1c86a2f8'
UUIDS={0:'GPU-bba35f4c-6cf5-7b0b-2791-fb6db663bde5',1:'GPU-17d5f199-b350-b969-e6a4-0e6d500912fd'}
IDS=['wJPptuhclZmKAEWs','wJPrpFsThWAcBQIw','wJQTclrlOLEYYfTB']
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with P(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def read(p):return json.loads(P(p).read_text())
def event(k,**v):
 x=dict(event=k,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manager_pid=os.getpid(),**v);print(json.dumps(x,ensure_ascii=False),flush=True)
 with (QUEUE/'events.jsonl').open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush()
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
RESERVES={}
def reserve(gpu):
 p=P('/tmp/od_reserve_'+str(gpu))
 if gpu not in RESERVES:
  assert not p.is_symlink()
  if p.exists():
   st=p.stat();assert p.is_file() and st.st_uid==os.getuid();RESERVES[gpu]=(st.st_dev,st.st_ino,st.st_atime_ns,st.st_mtime_ns)
  else:p.touch(exist_ok=False);st=p.stat();RESERVES[gpu]=(st.st_dev,st.st_ino,None,None)
 st=p.stat();assert (st.st_dev,st.st_ino)==RESERVES[gpu][:2] and st.st_uid==os.getuid();p.touch()
def unreserve(gpu):
 if gpu not in RESERVES:return
 p=P('/tmp/od_reserve_'+str(gpu));st=p.stat();identity=RESERVES.pop(gpu)
 assert (st.st_dev,st.st_ino)==identity[:2] and not p.is_symlink() and st.st_uid==os.getuid()
 if identity[2] is None:p.unlink()
 else:os.utime(p,ns=identity[2:])
def ram():
 mem={l.split()[0][:-1]:int(l.split()[1]) for l in P('/proc/meminfo').read_text().splitlines()};cg=P('/sys/fs/cgroup/memory');return dict(host_available_bytes=mem['MemAvailable']*1024,cgroup_headroom_bytes=int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text()),manager_cgroup=P('/proc/self/cgroup').read_text())
def main(stage_pid,expected_sha):
 assert os.getuid()==0 and sha(__file__)==expected_sha and sha(H3)==H3SHA
 assert OUT.is_dir();QUEUE.mkdir(exist_ok=False)
 event('manager_started',phase='wait_source_stage',stage_pid=stage_pid)
 stageproc=P('/proc')/str(stage_pid)
 if stageproc.exists():
  fd=os.pidfd_open(stage_pid)
  try:
   argv=[x.decode() for x in (stageproc/'cmdline').read_bytes().split(b'\0') if x]
   assert argv==[PY,'-B','-u',str(STAGE)] and stageproc.stat().st_uid==0
   event('waiting_exact_source_stage',pid=stage_pid,argv=argv);select.select([fd],[],[])
  finally:os.close(fd)
 assert read(OUT/'stage_complete.json')['complete']
 receipt=read('/opt/omni_n7_u3000_deploy_v1/deployment_receipt.json')
 admission=imp('admission_n6',MS/'h3_video_dimensions_case1_manager_v8.py');admission.QUEUE=QUEUE
 original_identity=admission.process_identity
 def identity(pid):
  value=original_identity(pid);value['cgroup']=P('/proc')/str(pid)/'cgroup';value['cgroup']=value['cgroup'].read_text()
  if value['cgroup']!=P('/proc/self/cgroup').read_text():raise RuntimeError('Unknown holder cgroup')
  return value
 admission.process_identity=identity
 frozen_infer_sha=sha(INFER)
 def launch(name,gpu,argv):
  assert sha(INFER)==frozen_infer_sha and sha(H3)==H3SHA
  log=(QUEUE/(name+'.log')).open('xb');child=subprocess.Popen(['env','CUDA_VISIBLE_DEVICES='+str(gpu),'OMP_NUM_THREADS=2','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','DIFFSYNTH_SKIP_DOWNLOAD=True',PY,'-B','-u']+argv,stdout=log,stderr=subprocess.STDOUT)
  event('actual_child_started',stage=name,gpu=gpu,child_pid=child.pid,argv=argv,log=str(log.name));return child,log,name,gpu
 def wait(children):
  pending=list(children);failed=[]
  while pending:
   for c in pending[:]:
    child,log,name,gpu=c;status=child.poll()
    if status is not None:
     log.close();event('actual_child_exited',stage=name,gpu=gpu,child_pid=child.pid,exit_code=status);pending.remove(c)
     if status:failed.append((name,status))
    else:reserve(gpu)
   if pending:
    import time;time.sleep(10)
  if failed:raise RuntimeError('Child failed; partial retained '+str(failed))
 def admit(name,gpus):
  capacity=ram();need=(180*len(gpus)+40)*2**30
  assert min(capacity['host_available_bytes'],capacity['cgroup_headroom_bytes'])>=need,(name,capacity,need)
  event('fresh_ram_before_stage',stage=name,required_bytes=need,**capacity)
  for gpu in gpus:reserve(gpu)
  admission.UUIDS={gpu:UUIDS[gpu] for gpu in gpus};admission.admit(name)
 # CPU functionality uses no CUDA allocation and no model weights loading.
 wait([launch('cpu_functional',0,[str(INFER),'--cpu-test'])]);unreserve(0)
 assert read(OUT/'cpu_functional_verified_v2.json')['complete']
 admit('omni',[0]);wait([launch('omni',0,[str(INFER)])]);unreserve(0)
 predictions=read(OUT/'omni_prediction3_raw.json');assert [r['id'] for r in predictions['predictions']]==IDS
 assert predictions['checkpoint_adapter_sha256']==receipt['deployment_files_sha256']['adapter/adapter_model.safetensors']
 event('omni_unloaded_before_H3',all_three_actual_captions=True,fresh_ram=ram())
 gt=read(MS/'omni_h3_ms_style_eval3_v1/h3_gt_caption3.json')['records'];assert [r['id'] for r in gt]==IDS
 configs=[]
 for row,pred in zip(gt,predictions['predictions']):
  r=dict(row);r['motion_caption']=pred['caption'];r['caption_version']='omni-n7-MS6k-u3000-raw:'+sha(OUT/'omni_prediction3_raw.json')
  case=OUT/r['id'];case.mkdir(exist_ok=False);cap=case/'caption_raw.json';save(cap,r)
  configs.append(dict(id=r['id'],caption=str(cap),cache=str(case/'cache'),video=str(case/'video'),image_sha256=sha(r['image_path'])))
 save(OUT/'h3_conditions_plan.json',dict(records=configs,source_checkpoint=receipt['source_checkpoint'],source_checkpoint_marker_sha256=receipt['source_checkpoint_marker_sha256'],adapter_sha256=predictions['checkpoint_adapter_sha256'],audio_input=False,audio_vae=False,audio_scheduler=False,seed=42,cfg=1,steps=50,flow_shift=12,geometry=dict(height=1280,width=704,num_frames=124,fps=24),human_review='pending'))
 # Same caption file binds cache and generation. At most two pipelines on this node.
 for stage in ('cache','video'):
  capacity=ram();parallel=min(capacity['host_available_bytes'],capacity['cgroup_headroom_bytes'])>=(2*180+40)*2**30
  batches=[[(0,configs[0]),(1,configs[1])],[(0,configs[2])]] if parallel else [[(0,c)] for c in configs]
  event('H3_stage_schedule',stage=stage,max_parallel=2 if parallel else 1,actual_ram=capacity)
  for batch in batches:
   admit('h3_'+stage+'_'+batch[0][1]['id'],[gpu for gpu,c in batch])
   children=[]
   for gpu,c in batch:
    argv=[str(H3),'--caption-json',c['caption'],'--weights-root','/dev/shm/h3/Ref2VA','--out',c[stage],'--limit','1']
    argv+=['--cache-only'] if stage=='cache' else ['--condition-cache',c['cache']]
    children.append(launch('h3_'+stage+'_'+c['id'],gpu,argv))
   wait(children)
   for gpu,c in batch:
    unreserve(gpu);report=read(P(c[stage])/'run_complete.json');audit=read(P(c[stage])/(c['id']+'.audit.json'))
    assert report['complete'] and report['completed_ids']==[c['id']] and audit['complete']
    assert report['source_code']['sha256']==H3SHA and report['lora_source'] is None and not report['old_y_reused']
    assert report['geometry']==dict(height=1280,width=704,num_frames=124,fps=24) and (report['steps'],report['seed'],report['cfg'],report['flow_shift'])==(50,42,1,12)
    assert audit['motion_caption']==read(c['caption'])['motion_caption'] and audit['sources']['image']['sha256']==c['image_sha256']
    guard=audit['audio_guard'];assert not any(guard['calls'].values()) and guard['audio_vae_is_none'] and guard['scheduler_audio_is_none'] and not audit['packed']['audio_pos_numel'] and not audit['packed']['tag2_count']
    if stage=='video':
     assert audit['sampling']['video_scheduler_updates']==50 and not audit['sampling']['target_audio_generated'] and sha(audit['video']['path'])==audit['video']['sha256']
     save(OUT/(c['id']+'.video_ready.json'),dict(complete=True,id=c['id'],video=audit['video'],media=audit['media'],source_checkpoint=receipt['source_checkpoint'],adapter_sha256=predictions['checkpoint_adapter_sha256'],human_review='pending'));event('video_ready_for_review',id=c['id'],video=audit['video']['path'])
 save(OUT/'generation_complete.json',dict(complete=True,actual_caption_cases=3,actual_video_cases=3,ids=IDS,records=configs,source_checkpoint=receipt['source_checkpoint'],source_checkpoint_marker_sha256=receipt['source_checkpoint_marker_sha256'],adapter_sha256=predictions['checkpoint_adapter_sha256'],full_training_completed=False,human_review='pending',review_http_verified=False))
 event('generation_complete_not_human_acceptance',actual_video_cases=3)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage-pid',type=int,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
 try:main(a.stage_pid,a.source_sha)
 except BaseException as e:
  if QUEUE.is_dir():event('manager_failed',error=repr(e),traceback=traceback.format_exc())
  raise
 finally:
  for gpu in list(RESERVES):unreserve(gpu)
