import datetime,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
ROOT=pathlib.Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher');CACHE=ROOT/'model_cache';LF=chr(10)
TICKET=pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_a100_validation_20261026_v1/a100-extra-03.json')
sys.path.insert(0,'/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/scripts_omnidance/exp/ms')
import numpy as np
import ms_pose2d_draft as rules
import ms_hand_roi as pose_api
OUT=None
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def emit(state,**data):
 r={'time':now(),'job':'assigned_03_three_case_validation','state':state,'pid':os.getpid(),**data}
 with (ROOT/'logs/preparation_events.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+LF);f.flush();os.fsync(f.fileno())
 print(json.dumps(r,allow_nan=False),flush=True)
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def write(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write(LF);f.flush();os.fsync(f.fileno())
started=time.monotonic();emit('started',sources=3,physical_gpu=0,diagnostic_not_production=True)
try:
 with TICKET.open() as f:ticket=json.load(f)
 if ticket['worker_id']!='a100-extra-03' or ticket['allowed_gpu_indices']!=[0] or ticket['source_count']!=3 or not ticket['diagnostic_not_production']:raise ValueError('ticket scope changed')
 if ticket['validation_execution']!='sequential_three_sources_one_process':raise ValueError('unexpected execution contract')
 namespace=pathlib.PurePosixPath(ticket['output_namespace'])
 if namespace.is_absolute() or '..' in namespace.parts:raise ValueError('unsafe namespace')
 OUT=ROOT/'results_unreviewed'/namespace
 OUT.mkdir(parents=True,exist_ok=False)
 stat=os.statvfs(ROOT)
 if stat.f_bavail*stat.f_frsize<2*2**30:raise RuntimeError('local storage below independent validation contract')
 limit=int(pathlib.Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text());usage=int(pathlib.Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
 if limit-usage<16*2**30:raise RuntimeError('insufficient container headroom for independent small-memory validation')
 snapshot=subprocess.run(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
 if snapshot.returncode:raise RuntimeError('GPU snapshot failed')
 uuid,used,free=[s.strip() for s in snapshot.stdout.strip().split(',')]
 compute=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
 if compute.returncode or any(s.strip().startswith(uuid) for s in compute.stdout.splitlines()) or int(used)!=0 or int(free)<4096:raise RuntimeError('GPU0 occupied;leave other tasks unchanged')
 identity_path=ROOT/'logs/gpu0_environment_receipt.json'
 with identity_path.open() as f:verified=json.load(f)
 identity=verified['pose']['local_model_identity']
 initialization=time.monotonic();model=pose_api.load_cuda_model(identity,{'num_threads':4,'device':0})
 from beat_this.inference import File2Beats
 with (CACHE/'BeatThis/local_source_manifest.json').open() as f:beat_binding=json.load(f)
 if sha(beat_binding['checkpoint_path'])!=beat_binding['checkpoint_sha256']:raise ValueError('BeatThis binding mismatch')
 beat=File2Beats(checkpoint_path=beat_binding['checkpoint_path'],device='cuda:0',float16=False,dbn=False)
 import torch
 torch.set_num_threads(4)
 model_init_seconds=time.monotonic()-initialization
 write(OUT/'run_identity.json',{'ticket_path':str(TICKET),'ticket_sha256':sha(TICKET),'entry_path':rules.__file__,'entry_sha256':sha(rules.__file__),'pose_model_identity':identity,'beat_local_source':beat_binding,'GPU_uuid':uuid,'pid':os.getpid(),'started':now(),'model_init_seconds':model_init_seconds,'memory_gate':{'scope':'independent_small_memory_diagnostic_only','cgroup_limit_bytes':limit,'cgroup_usage_bytes':usage,'required_headroom_bytes':16*2**30},'diagnostic_not_production':True,'formal_production_sources':0})
 cases=[]
 for source in ticket['sources']:
  row={**source,'diagnostic_not_production':True};case_start=time.monotonic();sid=row['id'];case_dir=OUT/row['split']/sid;receipt={'id':sid,'split':row['split'],'diagnostic_not_production':True,'formal_production_sources':0,'started':now(),'semantic_approval':False,'quality_approved':False,'training_eligible':False,'needs_visual_revision':True}
  emit('progress',source_id=sid,stage='input_verification')
  try:
   for key in ['video_path','audio_path','roi_manifest_path','npz_path','bar_plan_path','source_manifest_path']:
    path=pathlib.Path(row[key])
    if not path.is_file() or not os.access(path,os.R_OK):raise ValueError('unreadable '+key)
   roi,data,mask=rules.load_observations(row);rules.load_bar_windows(row,roi)
   if pathlib.Path(row['video_path']).stem!=sid or pathlib.Path(row['audio_path']).stem!=sid:raise ValueError('media source ID mismatch')
   if pathlib.Path(roi['source']['path']).resolve()!=pathlib.Path(row['video_path']).resolve():raise ValueError('ticket video differs from original ROI')
   if sha(row['video_path'])!=roi['video_sha256']:raise ValueError('ticket video SHA mismatch')
   receipt['original_inputs']={key:{'path':row[key],'bytes':pathlib.Path(row[key]).stat().st_size,'sha256':sha(row[key])} for key in ['video_path','audio_path','roi_manifest_path','npz_path','bar_plan_path','source_manifest_path']}
   receipt['observation']={'frame_count':len(data['times']),'first_time':float(data['times'][0]),'last_time':float(data['times'][-1]),'frame_indices':data['frame_indices'].tolist(),'source_timestamps':data['source_timestamps'].tolist(),'joint_layout_preserved':True,'effective_valid_count':int(mask.sum()),'score_above_one_count':int((data['score'][mask]>1).sum()),'no_3D_padding':True,'no_interpolation':True}
   tracked=np.flatnonzero(data['tracked'])
   if not len(tracked):raise ValueError('no tracked frame for actual-video model probe')
   k=int(tracked[0]);frame_index=int(data['frame_indices'][k]);probe_start=time.monotonic()
   info=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',row['video_path']],capture_output=True,timeout=30)
   if info.returncode:raise RuntimeError('video probe failed')
   dims=json.loads(info.stdout)['streams'][0];w,h=dims['width'],dims['height']
   if w*h>4096*4096:raise ValueError('frame exceeds small-memory diagnostic contract')
   filt='select=eq(n'+chr(92)+','+str(frame_index)+')'
   decode=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',row['video_path'],'-vf',filt,'-frames:v','1','-f','rawvideo','-pix_fmt','bgr24','pipe:1'],capture_output=True,timeout=60)
   if decode.returncode or len(decode.stdout)!=w*h*3:raise RuntimeError('native frame decode failed')
   frame=np.frombuffer(decode.stdout,dtype=np.uint8).reshape(h,w,3)
   compute_start=time.monotonic();points,scores,boxes=model(frame);torch.cuda.synchronize();compute_seconds=time.monotonic()-compute_start
   receipt['actual_video_pose_probe']={'status':'pass','scope':'single_original_observed_frame_model_probe_only_not_new_pose_cache','not_used_for_draft':True,'frame_index':frame_index,'time':float(data['times'][k]),'source_timestamp':float(data['source_timestamps'][k]),'output_xy_shape':list(np.asarray(points).shape),'output_score_shape':list(np.asarray(scores).shape),'box_shape':list(np.asarray(boxes).shape),'decode_plus_compute_seconds':time.monotonic()-probe_start,'model_compute_seconds':compute_seconds,'CUDA_providers':{'detector':model.session_det.get_providers(),'pose':model.session_pose.get_providers()}}
   emit('progress',source_id=sid,stage='actual_pose_probe_pass')
   audio_info=pathlib.Path(row['audio_path'])
   if audio_info.stat().st_size>512*2**20:raise ValueError('audio exceeds bounded diagnostic contract')
   start_copy=time.monotonic();local_audio=ROOT/'cache/current'/('validation_03_'+sid+'.wav')
   with audio_info.open('rb') as inp,local_audio.open('xb') as out:shutil.copyfileobj(inp,out,1024*1024);out.flush();os.fsync(out.fileno())
   if sha(local_audio)!=receipt['original_inputs']['audio_path']['sha256']:raise ValueError('audio local copy mismatch')
   copy_seconds=time.monotonic()-start_copy
   beat_start=time.monotonic()
   with torch.inference_mode():beats,downbeats=beat(str(local_audio))
   torch.cuda.synchronize();beat_seconds=time.monotonic()-beat_start
   beats=np.asarray(beats,dtype=float);downbeats=np.asarray(downbeats,dtype=float)
   if not np.isfinite(beats).all() or not np.isfinite(downbeats).all() or np.any(np.diff(beats)<=0) or np.any(np.diff(downbeats)<=0):raise ValueError('invalid BeatThis times')
   receipt['actual_audio_beat_probe']={'status':'pass','source_audio_path':row['audio_path'],'source_audio_sha256':receipt['original_inputs']['audio_path']['sha256'],'local_copy_path':str(local_audio),'copy_seconds':copy_seconds,'compute_seconds':beat_seconds,'beats':beats.tolist(),'downbeats':downbeats.tolist(),'checkpoint_sha256':beat_binding['checkpoint_sha256'],'library':'official beat_this.inference.File2Beats','dbn':False,'float16':False,'device':'cuda:0','new_probe_not_substituted_for_bound_bar_cache':True,'beat_accuracy_verified':False,'audio_video_alignment_verified':False}
   emit('progress',source_id=sid,stage='actual_beat_probe_pass')
   result=rules.process_row(row,OUT)
   if result['status']=='existing_not_overwritten':raise ValueError('existing draft;do not count or rerun')
   with pathlib.Path(result['path']).open() as f:draft=json.load(f)
   if draft['id']!=sid:raise ValueError('draft source mismatch')
   if any(draft.get(k)!=v for k,v in rules.FLAGS.items()):raise ValueError('unreviewed flags invalid')
   events=draft['events']
   if not events or any(not e.get('text') or not e.get('support',{}).get('frame_indices') for e in events):raise ValueError('no real action event/text/frame support')
   receipt['draft_result']=result;receipt['status']=draft['status'];receipt['action_event_count']=len(events);receipt['window_count']=len(draft['windows']);receipt['source_cache_reused_unchanged']=True
  except Exception as e:receipt['status']='failed';receipt['error']=type(e).__name__+': '+str(e);receipt['automatic_retry']=False;traceback.print_exc()
  receipt['finished']=now();receipt['case_total_seconds']=time.monotonic()-case_start;write(case_dir/'validation_receipt.json',receipt);cases.append(receipt)
  emit('progress',source_id=sid,case_status=receipt['status'],action_events=receipt.get('action_event_count',0),windows=receipt.get('window_count',0))
 summary={'ticket_path':str(TICKET),'ticket_sha256':sha(TICKET),'worker_id':'a100-extra-03','diagnostic_not_production':True,'formal_production_sources':0,'cases_attempted':len(cases),'status_counts':{s:sum(c['status']==s for c in cases) for s in ['draft_generated','partial','needs_regeneration','failed']},'action_events':sum(c.get('action_event_count',0) for c in cases),'windows':sum(c.get('window_count',0) for c in cases),'output_root':str(OUT),'wall_seconds_including_init_read_copy_models_rules_local_write':time.monotonic()-started,'shared_writeback':'pending protocol;local results durable','fifteen_minute_production_throughput':'not_measured','quality_approved':False,'training_eligible':False,'semantic_approval':False,'needs_visual_revision':True}
 write(OUT/'summary.json',summary)
 emit('completed' if not summary['status_counts']['failed'] else 'failed',**summary)
 sys.exit(0 if not summary['status_counts']['failed'] else 1)
except Exception as e:
 emit('failed',error=type(e).__name__+': '+str(e),automatic_retry=False);traceback.print_exc();sys.exit(1)
