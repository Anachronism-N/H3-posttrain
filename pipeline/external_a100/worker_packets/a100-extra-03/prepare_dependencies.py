import datetime,hashlib,json,os,pathlib,shlex,subprocess,sys
BASE=pathlib.Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher';LF=chr(10)
NODES=[('a100-extra-03','launcher',None),('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]
PINS={'onnxruntime-gpu':'1.23.2','beat-this':'1.1.0','rotary-embedding-torch':'0.8.9','coloredlogs':'15.0.1','flatbuffers':'25.12.19','humanfriendly':'10.0'}
def emit(state,**data):
 r={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'dependencies_prepare','state':state,'pid':os.getpid(),**data}
 with (ROOT/'logs/preparation_events.jsonl').open('a') as f:f.write(json.dumps(r)+LF);f.flush();os.fsync(f.fileno())
 print(json.dumps(r),flush=True)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
TRANSFER='''
import hashlib,json,pathlib,sys,os
path=pathlib.Path(sys.argv[1]);expected=sys.argv[2];size=int(sys.argv[3]);base=pathlib.Path(sys.argv[4])
if not base.is_dir() or base not in path.parents:raise ValueError('outside owned wheelhouse')
h=hashlib.sha256();count=0
with path.open('xb') as f:
 while b:=sys.stdin.buffer.read(4*1024*1024):f.write(b);h.update(b);count+=len(b)
 f.flush();os.fsync(f.fileno())
if h.hexdigest()!=expected or count!=size:raise ValueError('wheel transfer mismatch;partial retained')
print(json.dumps({'sha256':h.hexdigest(),'bytes':count,'verified':True}))
'''
INSTALL='''
import datetime,importlib,importlib.metadata,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);pins=json.loads(sys.argv[2]);expected=str(root/'venv')
if sys.prefix!=expected:raise ValueError('not owned local venv')
def versions(names):
 r={}
 for name in names:
  try:r[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:r[name]=None
 return r
protected=['torch','torchaudio','numpy','opencv-python'];before=versions(protected)
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
args=[sys.executable,'-m','pip','--isolated','install','--disable-pip-version-check','--no-index','--no-deps','--find-links',str(root/'wheelhouse')]+[name+'=='+version for name,version in pins.items()]
with (root/'logs/dependency_install.log').open('x') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=300)
if p.returncode:raise RuntimeError('offline pip failed rc='+str(p.returncode)+';do not automatically retry')
after=versions(protected)
if before!=after:raise RuntimeError('protected base package version changed')
observed=versions(pins)
if observed!=pins:raise RuntimeError('installed versions differ from pins')
imports={}
for name in ['torch','torchaudio','numpy','cv2','soundfile','soxr','einops','onnxruntime','beat_this.inference','rotary_embedding_torch']:
 try:module=importlib.import_module(name);imports[name]={'ok':True,'version':str(getattr(module,'__version__','unknown'))}
 except Exception as e:imports[name]={'ok':False,'error':type(e).__name__+': '+str(e)[:500]}
missing=[];providers=[]
if imports['onnxruntime']['ok']:
 import onnxruntime as ort
 providers=ort.get_available_providers();lib=pathlib.Path(ort.__file__).parent/'capi/libonnxruntime_providers_cuda.so'
 if lib.is_file():
  p=subprocess.run(['ldd',str(lib)],text=True,capture_output=True,timeout=10);missing=[s.strip() for s in p.stdout.splitlines() if 'not found' in s]
 else:missing=['ORT CUDA provider library absent']
receipt={'pid':os.getpid(),'started':started,'finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prefix':sys.prefix,'pins':observed,'protected_before':before,'protected_after':after,'imports':imports,'ort_declared_providers':providers,'ort_cuda_missing_shared_libraries':missing,'models_loaded':False,'gpu_started':False,'ort_version_status':'locally_pinned_candidate_not_original_production_lock','beat_rotary_basis':'existing recorded beat shard receipt;other local base versions differ'}
with (root/'logs/dependency_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10))
print(json.dumps(receipt),flush=True)
if any(not x['ok'] for x in imports.values()):sys.exit(2)
'''
emit('started')
try:
 wheels=[]
 for name,version in PINS.items():
  matches=list((ROOT/'wheelhouse').glob(name.replace('-','_')+'-'+version+'-*.whl'))
  if len(matches)!=1:raise ValueError('nonunique wheel for '+name)
  p=matches[0];wheels.append({'path':str(p),'name':p.name,'sha256':digest(p),'bytes':p.stat().st_size})
 with (ROOT/'logs/dependency_final_pins.json').open('x') as f:json.dump({'pins':PINS,'wheels':wheels,'onnxruntime_basis':'local preparation candidate;upstream pose requirements unpinned','rotary_basis':'existing BeatThis receipt 0.8.9'},f,indent=2);f.write(LF)
 for server,role,host in NODES:
  root=BASE/role
  if host:
   for wheel in wheels:
    cmd=['python3','-c',TRANSFER,str(root/'wheelhouse'/wheel['name']),wheel['sha256'],str(wheel['bytes']),str(root/'wheelhouse')]
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(cmd)]
    with pathlib.Path(wheel['path']).open('rb') as f:p=subprocess.run(ssh,stdin=f,capture_output=True,timeout=180)
    if p.returncode:raise RuntimeError(server+':wheel transfer failed;no automatic retry')
    receipt=json.loads(p.stdout)
    if not receipt.get('verified'):raise ValueError('unverified transfer')
   args=[str(root/'venv/bin/python'),'-c',INSTALL,str(root),json.dumps(PINS)]
   cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,'PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES='+shlex.quote('')+' '+shlex.join(args)]
  else:cmd=[str(root/'venv/bin/python'),'-c',INSTALL,str(root),json.dumps(PINS)]
  p=subprocess.run(cmd,capture_output=True,timeout=360,env={**os.environ,'PYTHONWARNINGS':'ignore','CUDA_VISIBLE_DEVICES':''})
  with (ROOT/'logs'/('install_'+role+'_raw.log')).open('xb') as f:f.write(p.stdout);f.write(p.stderr)
  if p.returncode:raise RuntimeError(server+':dependency install/import failed rc='+str(p.returncode)+';log retained;no automatic retry')
  if bytes([0]) in p.stdout or not p.stdout.endswith(bytes([10])):raise RuntimeError('incomplete dependency receipt;raw retained')
  receipt=json.loads(p.stdout)
  with (ROOT/'logs'/('install_'+role+'_receipt.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write(LF)
  emit('progress',server=server,imports_ok=True,ort_cuda_missing_shared_libraries=receipt['ort_cuda_missing_shared_libraries'])
 emit('completed',servers=4,pins=PINS,models_loaded=False)
except Exception as e:emit('failed',error=type(e).__name__+': '+str(e));raise
