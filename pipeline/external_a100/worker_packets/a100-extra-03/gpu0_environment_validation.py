import datetime,gc,hashlib,json,os,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher');CACHE=ROOT/'model_cache';LF=chr(10)
PROFILE=ROOT/'logs/gpu0_profile_v1';PROFILE.mkdir(exist_ok=False)
def emit(state,**data):
 r={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'gpu0_environment_validation','state':state,'pid':os.getpid(),**data}
 with (ROOT/'logs/preparation_events.jsonl').open('a') as f:f.write(json.dumps(r)+LF);f.flush();os.fsync(f.fileno())
 print(json.dumps(r),flush=True)
result={'scope':'synthetic_model_environment_test_not_real_cases','server':'a100-extra-03','physical_gpu':0,'models_loaded':True,'formal_sources':0,'motionscript_drafts':0,'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid()}
emit('started',physical_gpu=0)
try:
 p=subprocess.run(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 if p.returncode:raise RuntimeError('GPU query failed')
 uuid,used=[s.strip() for s in p.stdout.strip().split(',')]
 p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 if p.returncode or any(line.strip().startswith(uuid) for line in p.stdout.splitlines()) or int(used)!=0:raise RuntimeError('GPU0 now occupied;do not start or stop other processes')
 result['gpu_uuid']=uuid
 import numpy as np,torch
 torch.set_num_threads(4)
 if not torch.cuda.is_available():raise RuntimeError('torch CUDA unavailable')
 with torch.inference_mode():
  x=torch.ones((256,256),device='cuda:0');y=x@x;torch.cuda.synchronize();assert float(y[0,0])==256
 del x,y;torch.cuda.empty_cache();result['torch_cuda_tensor']='pass';emit('progress',step='torch_cuda_tensor_pass')
 sys.path.insert(0,'/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/scripts_omnidance/exp/ms')
 import ms_hand_roi as api
 scail=pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/pose-preprocess/third_party/scail_pose');code=scail/'DWPoseProcess/dwpose'
 files={'detector':api.file_identity(CACHE/'DWPose/yolox_l.onnx'),'pose':api.file_identity(CACHE/'DWPose/dw-ll_ucoco_384.onnx'),'wholebody':api.file_identity(code/'wholebody.py'),'onnxdet':api.file_identity(code/'onnxdet.py'),'onnxpose':api.file_identity(code/'onnxpose.py'),'dwpose_init':api.file_identity(code/'__init__.py'),'layout_source':api.file_identity(pathlib.Path(api.__file__).with_name('dwpose_full_extract.py')),'roi_source':api.file_identity(api.__file__)}
 identity={'scail_root':str(scail),'files':files,'sha256':api.digest(files),'joint_layout':api.joint_layout(),'full_hash_once_per_process':True,'upstream_package_init_executed':False}
 before=time.monotonic();model=api.load_cuda_model(identity,{'num_threads':4,'device':0},PROFILE);load_seconds=time.monotonic()-before;session_results={}
 for name,session in [('detector',model.session_det),('pose',model.session_pose)]:
  feeds={};spec=[]
  for inp in session.get_inputs():
   shape=[v if isinstance(v,int) and v>0 else 1 for v in inp.shape]
   if inp.type!='tensor(float)':raise RuntimeError('unexpected model input dtype '+inp.type)
   feeds[inp.name]=np.zeros(shape,np.float32);spec.append({'name':inp.name,'shape':shape})
  before=time.monotonic();outputs=session.run(None,feeds);torch.cuda.synchronize();seconds=time.monotonic()-before
  finite=all(np.isfinite(x).all() for x in outputs)
  profile=pathlib.Path(session.end_profiling())
  with profile.open() as f:events=json.load(f)
  cuda_events=[e for e in events if e.get('args',{}).get('provider')=='CUDAExecutionProvider']
  if not finite or not cuda_events:raise RuntimeError(name+':no finite CUDA inference evidence')
  session_results[name]={'providers':session.get_providers(),'inputs':spec,'output_shapes':[list(x.shape) for x in outputs],'all_finite':finite,'cuda_profile_node_events':len(cuda_events),'profile_path':str(profile),'synthetic_forward_seconds':seconds}
 result['pose']={'status':'pass','load_seconds':load_seconds,'sessions':session_results,'joint_layout':identity['joint_layout'],'local_model_identity':identity};emit('progress',step='pose_cuda_profile_verified')
 del model,session,outputs,feeds;gc.collect();torch.cuda.empty_cache()
 from beat_this.inference import Audio2Beats
 source=CACHE/'BeatThis/local_source_manifest.json'
 with source.open() as f:binding=json.load(f)
 def sha(path):
  h=hashlib.sha256()
  with pathlib.Path(path).open('rb') as f:
   while b:=f.read(4*1024*1024):h.update(b)
  return h.hexdigest()
 if sha(binding['checkpoint_path'])!=binding['checkpoint_sha256']:raise RuntimeError('BeatThis local model binding invalid')
 before=time.monotonic();beat=Audio2Beats(checkpoint_path=binding['checkpoint_path'],device='cuda:0',float16=False,dbn=False);load_seconds=time.monotonic()-before
 before=time.monotonic()
 with torch.inference_mode():beats,downbeats=beat(np.zeros(22050,np.float32),22050)
 torch.cuda.synchronize();seconds=time.monotonic()-before
 if not np.isfinite(beats).all() or not np.isfinite(downbeats).all():raise RuntimeError('nonfinite synthetic BeatThis output')
 result['beatthis']={'status':'pass','load_seconds':load_seconds,'synthetic_forward_seconds':seconds,'synthetic_audio_seconds':1,'checkpoint_sha256':binding['checkpoint_sha256'],'device':'cuda:0','float16':False,'dbn':False,'synthetic_only_no_beat_accuracy_claim':True}
 del beat;gc.collect();torch.cuda.empty_cache();emit('progress',step='beatthis_cuda_forward_pass')
 result['status']='pass';result['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (ROOT/'logs/gpu0_environment_receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write(LF)
 emit('completed',models=['pose_detector','wholebody_pose','BeatThis'],formal_sources=0,motionscript_drafts=0)
except Exception as e:
 result['status']='failed';result['error']=type(e).__name__+': '+str(e);result['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (ROOT/'logs/gpu0_environment_receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write(LF)
 emit('failed',error=result['error'],no_automatic_retry=True);traceback.print_exc();sys.exit(1)
