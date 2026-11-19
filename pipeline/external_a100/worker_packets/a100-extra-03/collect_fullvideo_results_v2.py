import concurrent.futures,datetime,hashlib,io,json,os,pathlib,shlex,shutil,subprocess,sys,tarfile,time
import numpy as np
BASE=pathlib.Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher';T=pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_a100_validation_20261026_v1');TAG='fullvideo_gpu_v2_20261026T1522Z';LF=chr(10)
NODES=[('a100-extra-03','launcher',None),('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]
FLAGS={'diagnostic_not_production':True,'semantic_approval':False,'quality_approved':False,'training_eligible':False,'needs_visual_revision':True}
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def write(path,obj):
 with pathlib.Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write(LF);f.flush();os.fsync(f.fileno())
def ssh(host,args):return ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
PACK='''
import pathlib,tarfile,json,hashlib,os,sys
root=pathlib.Path(sys.argv[1]);worker=sys.argv[2];tag=sys.argv[3]
source=root/'results_unreviewed'/worker/'validation_20261026_v1/fullvideo_gpu_v2'
with (source/'summary.json').open() as f:summary=json.load(f)
if summary['status'] not in ['success','partial','failed']:raise ValueError('not terminal')
package=root/'pending_writeback'/(tag+'.tar.gz')
with package.open('xb') as raw:
 with tarfile.open(fileobj=raw,mode='w:gz') as tar:
  for p in sorted(source.rglob('*')):
   if p.is_symlink():raise ValueError('unexpected output symlink')
   if p.is_file():tar.add(p,arcname=str(p.relative_to(source)),recursive=False)
  wrapper=root/'fullvideo_ticket_validation_v2.py';tar.add(wrapper,arcname='audit/current_wrapper.py',recursive=False)
  plan=root/'cache/current/assigned_validation_packet_v2/plan.json';tar.add(plan,arcname='audit/assigned_plan.json',recursive=False)
 raw.flush();os.fsync(raw.fileno())
h=hashlib.sha256()
with package.open('rb') as f:
 while b:=f.read(4*1024*1024):h.update(b)
print(json.dumps({'remote_package':str(package),'bytes':package.stat().st_size,'sha256':h.hexdigest(),'summary_status':summary['status']}))
'''
def validate_package(package,worker):
 results=[]
 with tarfile.open(package,mode='r:gz') as tar:
  members=tar.getmembers()
  if any(m.issym() or m.islnk() or pathlib.PurePosixPath(m.name).is_absolute() or '..' in pathlib.PurePosixPath(m.name).parts for m in members):raise ValueError('unsafe package member')
  if sum(m.size for m in members)>2*2**30:raise ValueError('result package expands beyond limit')
  def getjson(name):
   f=tar.extractfile(name);raw=f.read()
   if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('bad or partial JSON;package retained')
   return json.loads(raw)
  summary=getjson('summary.json');plan=getjson('audit/assigned_plan.json')
  if summary['worker_id']!=worker or summary['formal_production_sources']!=0:raise ValueError('wrong worker/production scope')
  for case in summary['cases']:
   name='train/'+case['id'];receipt=getjson(name+'/validation_receipt.json')
   if any(receipt.get(k)!=v for k,v in FLAGS.items()):raise ValueError('approval flag mismatch')
   if receipt['status']=='failed':results.append({'id':case['id'],'status':'failed','reason':receipt.get('error')});continue
   full=receipt['fullvideo'];stats=full['coverage'];audio=receipt['audio'];draft=getjson(name+'/draft.json')
   if full['status']!='pass' or full['decoded_frame_count']!=full['probed_frame_count'] or not stats['real_native_pts_full_coverage']:raise ValueError('unproven fullvideo coverage')
   npz_name=name+'/new_fullvideo_wholebody134.npz';raw=tar.extractfile(npz_name).read()
   if hashlib.sha256(raw).hexdigest()!=full['npz_sha256']:raise ValueError('newNPZ SHA mismatch')
   with np.load(io.BytesIO(raw),allow_pickle=False) as z:
    n=len(z['times']);assert z['xy'].shape==(n,134,2) and z['score'].shape==(n,134)
    assert n==full['probed_frame_count'] and np.array_equal(z['frame_indices'],np.arange(n))
    assert np.all(np.diff(z['times'])>0) and np.all(np.diff(z['source_timestamps'])>0)
    assert z['schema'].item()=='ms_fullvideo_pose2d_diagnostic_v1'
    assert z['id'].item()==case['id'];assert not ('binding_sha256' in z.files)
    valid=z['valid']&z['tracked'][:,None];assert np.isfinite(z['xy'][valid]).all() and np.isfinite(z['score'][valid]).all()
   profile=receipt['actual_video_cuda_profiles']
   assert all(x['cuda_node_events']>0 and 'CUDAExecutionProvider' in x['providers'] for x in profile.values())
   assert audio['status']=='pass' and audio['device']=='cuda:0' and audio['float16'] is False and audio['dbn'] is False
   for key in ['beats','downbeats']:
    a=np.array(audio[key]);assert np.isfinite(a).all() and np.all(a>=0) and np.all(np.diff(a)>0) and np.all(a<=audio['duration_seconds']+.02)
   assert all(draft.get(k)==v for k,v in FLAGS.items()) and draft['events']
   assert all(e.get('text') and e.get('support',{}).get('frame_indices') for e in draft['events'])
   assert receipt['draft']['new_npz_passed_to_rules'] is False and receipt['original_cache_coverage']['frame_count']==32
   results.append({'id':case['id'],'status':receipt['status'],'video_frames':full['decoded_frame_count'],'probed_frames':full['probed_frame_count'],'PTS_first_last':[stats['first_source_timestamp'],stats['last_source_timestamp']],'tracked_frames':stats['tracked_frames'],'valid_fraction':stats['valid_point_fraction_score_threshold_not_probability'],'new_NPZ_SHA':full['npz_sha256'],'both_sessions_CUDA':True,'CUDA_profile_events':{k:v['cuda_node_events'] for k,v in profile.items()},'beats':len(audio['beats']),'downbeats':len(audio['downbeats']),'audio_seconds':audio['duration_seconds'],'events':len(draft['events']),'windows':len(draft['windows']),'draft_status':draft['status'],'stage_seconds':receipt['stage_seconds'],'case_total_seconds':receipt['case_total_seconds'],'original_cache_frames':32,'unknown_intervals':receipt['draft']['unknown_intervals']})
 return summary,results
def collect(node):
 worker,role,host=node;root=BASE/role;start=time.monotonic();args=[str(root/'venv/bin/python'),'-c',PACK,str(root),worker,TAG]
 p=subprocess.run(ssh(host,args) if host else args,capture_output=True,timeout=120)
 if p.returncode:raise RuntimeError(worker+' result pack failed;no retry')
 info=json.loads(p.stdout)
 if info['bytes']>2*2**30:raise ValueError('output package over2GiB')
 if host:
  local=ROOT/'pending_writeback'/(worker+'_'+TAG+'.tar.gz')
  copycode='import sys,shutil;f=open(sys.argv[1],"rb");shutil.copyfileobj(f,sys.stdout.buffer,4*1024*1024)'
  cmd=ssh(host,['python3','-c',copycode,info['remote_package']])
  with local.open('xb') as out:
   p=subprocess.run(cmd,stdout=out,stderr=subprocess.PIPE,timeout=120);out.flush();os.fsync(out.fileno())
  if p.returncode:raise RuntimeError(worker+' download failed;retain partial')
 else:local=pathlib.Path(info['remote_package'])
 if local.stat().st_size!=info['bytes'] or sha(local)!=info['sha256']:raise ValueError('output copySHA mismatch')
 summary,cases=validate_package(local,worker)
 dest=T/'results'/worker;dest.mkdir(parents=True,exist_ok=True);target=dest/(TAG+'.tar.gz')
 with local.open('rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,4*1024*1024);out.flush();os.fsync(out.fileno())
 if sha(target)!=info['sha256']:raise ValueError('durable shared package SHA mismatch;no retry')
 receipt={**FLAGS,'worker_id':worker,'package_path':str(target),'package_bytes':info['bytes'],'package_sha256':info['sha256'],'verified_writeback':True,'formal_production_sources':0,'cases':cases,'status_counts':summary['status_counts'],'model_wall_seconds':summary['wall_seconds'],'relay_input_seconds':None,'writeback_seconds':time.monotonic()-start,'completed':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 relay=ROOT/'logs'/('relay_'+role+'_receipt_v2.json')
 if relay.exists():
  with relay.open() as f:rr=json.load(f)
  receipt['relay_input_seconds']=rr['relay_seconds'];receipt['input_packet_bytes']=rr['packet_bytes']
 write(dest/(TAG+'_receipt.json'),receipt);write(ROOT/'logs'/(worker+'_'+TAG+'_writeback.json'),receipt)
 return receipt
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(collect,NODES))
unique=sorted({c['id'] for r in results for c in r['cases']})
aggregate={**FLAGS,'formal_production_sources':0,'validation_attempts':sum(len(r['cases']) for r in results),'unique_source_ids':unique,'note':'03three sources duplicate04-06;do not sum as unique production','results':results,'verified_shared_writeback':True,'fifteen_minute_production_throughput':'not_measured'}
write(ROOT/'logs/fullvideo_all_nodes_verified_summary_v2.json',aggregate)
write(T/'results'/(TAG+'_all_nodes_summary.json'),aggregate)
print(json.dumps({'verified_nodes':4,'validation_attempts':aggregate['validation_attempts'],'unique_sources':len(unique),'failed':sum(r['status_counts']['failed'] for r in results),'partial':sum(r['status_counts']['partial'] for r in results),'packages':[{'worker':r['worker_id'],'path':r['package_path'],'bytes':r['package_bytes']} for r in results]},allow_nan=False),flush=True)
