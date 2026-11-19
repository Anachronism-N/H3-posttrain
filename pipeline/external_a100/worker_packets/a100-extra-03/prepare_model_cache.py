import datetime,hashlib,json,os,pathlib,shlex,subprocess,sys,time
BASE=pathlib.Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher';LF=chr(10)
NODES=[('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]
POSE=pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/pose-preprocess/third_party/scail_pose/pretrained_weights/DWPose')
BEAT=pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/models/BeatThis')
FILES=[(POSE/'yolox_l.onnx','DWPose/yolox_l.onnx'),(POSE/'dw-ll_ucoco_384.onnx','DWPose/dw-ll_ucoco_384.onnx'),(BEAT/'final0.verified.ckpt','BeatThis/final0.verified.ckpt'),(BEAT/'LICENSE','BeatThis/LICENSE'),(BEAT/'source_manifest.json','BeatThis/source_manifest.json')]
def emit(state,**data):
 r={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'model_cache_prepare','state':state,'pid':os.getpid(),**data}
 with (ROOT/'logs/preparation_events.jsonl').open('a') as f:f.write(json.dumps(r)+LF);f.flush();os.fsync(f.fileno())
 print(json.dumps(r),flush=True)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
REMOTE='''
import hashlib,json,pathlib,sys,os
path=pathlib.Path(sys.argv[1]);expected=sys.argv[2];size=int(sys.argv[3]);base=pathlib.Path(sys.argv[4])
if not base.is_dir() or base not in path.parents:raise ValueError('outside owned cache')
path.parent.mkdir(parents=True,exist_ok=True)
h=hashlib.sha256();count=0
with path.open('xb') as f:
 while b:=sys.stdin.buffer.read(4*1024*1024):f.write(b);h.update(b);count+=len(b)
 f.flush();os.fsync(f.fileno())
if h.hexdigest()!=expected or count!=size:raise ValueError('copied bytes mismatch;partial retained')
print(json.dumps({'path':str(path),'sha256':h.hexdigest(),'bytes':count,'verified':True}))
'''
emit('started')
try:
 with (BEAT/'source_manifest.json').open() as f:source_manifest=json.load(f)
 entries=[]
 for source,relative in FILES:
  target=ROOT/'model_cache'/relative;target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
  with source.open('rb') as inp,target.open('xb') as out:
   while b:=inp.read(4*1024*1024):out.write(b);h.update(b);count+=len(b)
   out.flush();os.fsync(out.fileno())
  actual=h.hexdigest()
  if digest(target)!=actual:raise ValueError('local cache verification failed')
  if relative=='BeatThis/final0.verified.ckpt' and actual!=source_manifest['checkpoint_sha256']:raise ValueError('BeatThis checkpoint differs from official source identity')
  if relative=='BeatThis/LICENSE' and actual!=source_manifest['license_sha256']:raise ValueError('BeatThis license mismatch')
  entry={'source_path':str(source),'relative_path':relative,'sha256':actual,'bytes':count,'local_path':str(target),'verified':True};entries.append(entry);emit('progress',server='a100-extra-03',file=relative,bytes=count)
 localindex=ROOT/'model_cache/cache_identity.json'
 with localindex.open('x') as f:json.dump({'schema':'local_model_cache_identity_v1','models_loaded':False,'source_manifest_preserved':True,'note':'BeatThis copied source manifest retains original shared paths;formal interface must explicitly resolve local cache with hash binding','files':entries},f,indent=2);f.write(LF)
 entries.append({'relative_path':'cache_identity.json','sha256':digest(localindex),'bytes':localindex.stat().st_size,'local_path':str(localindex),'verified':True})
 for server,role,host in NODES:
  cache=BASE/role/'model_cache';receipts=[]
  for entry in entries:
   source=pathlib.Path(entry['local_path']);target=cache/entry['relative_path']
   args=['python3','-c',REMOTE,str(target),entry['sha256'],str(entry['bytes']),str(cache)]
   cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
   with source.open('rb') as f:p=subprocess.run(cmd,stdin=f,capture_output=True,timeout=180)
   if p.returncode:raise RuntimeError(server+':copy failed rc='+str(p.returncode)+';partial retained;no automatic retry')
   if bytes([0]) in p.stdout or not p.stdout.endswith(bytes([10])):raise RuntimeError(server+':invalid receipt;retain output and stop')
   receipt=json.loads(p.stdout);receipts.append(receipt);emit('progress',server=server,file=entry['relative_path'],bytes=receipt['bytes'])
  with (ROOT/'logs'/('model_cache_'+role+'_receipt.json')).open('x') as f:json.dump({'server':server,'files':receipts,'model_loaded':False},f,indent=2);f.write(LF)
 emit('completed',servers=4,files_per_node=len(entries),models_loaded=False)
except Exception as e:
 emit('failed',error=type(e).__name__+': '+str(e));raise
