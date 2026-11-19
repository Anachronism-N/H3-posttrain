import datetime,hashlib,json,os,pathlib,shlex,subprocess,sys
BASE=pathlib.Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher';LF=chr(10)
NODES=[('launcher',None),('worker-0','11.254.248.52'),('worker-1','11.239.106.11'),('worker-2','11.239.84.43')]
CODE='''
import hashlib,json,pathlib,sys,os,datetime
root=pathlib.Path(sys.argv[1]);cache=root/'model_cache';index=cache/'cache_identity.json'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
with index.open() as f:identity=json.load(f)
for entry in identity['files']:
 p=cache/entry['relative_path']
 if digest(p)!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise ValueError('cache bytes mismatch')
 entry['local_path']=str(p)
identity['node_role']=root.name;identity['local_paths_revalidated']=True
identity['note']='Original BeatThis source_manifest.json retained as provenance;local_source_manifest.json explicitly binds identical bytes to local paths.'
temp=cache/'cache_identity.relocated.tmp'
with temp.open('x') as f:json.dump(identity,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
temp.replace(index)
original=cache/'BeatThis/source_manifest.json'
with original.open() as f:source=json.load(f)
source['upstream_source_manifest_path']=str(original);source['upstream_source_manifest_sha256']=digest(original)
source['upstream_checkpoint_path']=source['checkpoint_path'];source['upstream_license_path']=source['license_path']
source['checkpoint_path']=str(cache/'BeatThis/final0.verified.ckpt');source['license_path']=str(cache/'BeatThis/LICENSE')
source['local_relocation_only']=True;source['model_bytes_unchanged']=True
if digest(pathlib.Path(source['checkpoint_path']))!=source['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
if digest(pathlib.Path(source['license_path']))!=source['license_sha256']:raise ValueError('license mismatch')
local=cache/'BeatThis/local_source_manifest.json'
with local.open('x') as f:json.dump(source,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
print(json.dumps({'role':root.name,'local_paths_validated':True,'files_verified':len(identity['files']),'local_beat_source_manifest':str(local),'models_loaded':False}))
'''
def event(state,**data):
 record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'cache_local_identity','state':state,'pid':os.getpid(),**data}
 with (ROOT/'logs/preparation_events.jsonl').open('a') as f:f.write(json.dumps(record)+LF);f.flush();os.fsync(f.fileno())
 print(json.dumps(record),flush=True)
event('started')
try:
 for role,host in NODES:
  root=BASE/role;args=[str(root/'venv/bin/python'),'-c',CODE,str(root)]
  cmd=args if host is None else ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
  p=subprocess.run(cmd,capture_output=True,timeout=180)
  with (ROOT/'logs'/('model_cache_'+role+'_local_identity.log')).open('xb') as f:f.write(p.stdout);f.write(p.stderr)
  if p.returncode:raise RuntimeError(role+':identity failed;preserve cache and do not retry')
  receipt=json.loads(p.stdout)
  with (ROOT/'logs'/('model_cache_'+role+'_local_identity.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write(LF)
  event('progress',**receipt)
 event('completed',nodes=4)
except Exception as e:event('failed',error=type(e).__name__+': '+str(e));raise
