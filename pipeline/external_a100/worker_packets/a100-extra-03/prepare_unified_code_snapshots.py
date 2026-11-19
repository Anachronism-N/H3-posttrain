from pathlib import Path
import datetime,hashlib,json,os,shlex,subprocess,shutil
BASE=Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher'
P=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_run_20261027_v1')
T=json.loads((P/'assignments/a100-extra-03.ticket.json').read_bytes())
SNAP=ROOT/'unified_code_snapshot_v1';SNAP.mkdir(exist_ok=False)
files=[]
for ref in T['code_refs']:
 source=Path(ref['path']);b=source.read_bytes();h=hashlib.sha256(b).hexdigest()
 if h!=ref['sha256']:raise ValueError('unified algorithm source SHA does not match frozen ticket')
 path=SNAP/source.name
 with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.chmod(path,0o444);files.append({'path':str(path),'relative':source.name,'sha256':h,'bytes':len(b),'original_path':str(source)})
code=ROOT/'cache/current/assigned_validation_packet_v2/code/dwpose'
scail=ROOT/'unified_scail_local_v1';dst=scail/'DWPoseProcess/dwpose';dst.mkdir(parents=True,exist_ok=False)
for name in ['wholebody.py','onnxdet.py','onnxpose.py','__init__.py']:
 path=dst/name;b=(code/name).read_bytes();h=hashlib.sha256(b).hexdigest()
 with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.chmod(path,0o444);files.append({'path':str(path),'relative':'scail/DWPoseProcess/dwpose/'+name,'sha256':h,'bytes':len(b),'original_path':str(code/name)})
weights=scail/'pretrained_weights';weights.mkdir();(weights/'DWPose').symlink_to(ROOT/'model_cache/DWPose',target_is_directory=True)
RECV='''
from pathlib import Path
import json,hashlib,os,sys
root=Path(sys.argv[1]);relative=sys.argv[2];expected=sys.argv[3];b=sys.stdin.buffer.read()
path=root/('unified_scail_local_v1' if relative.startswith('scail/') else 'unified_code_snapshot_v1')/(relative[6:] if relative.startswith('scail/') else relative)
if not root.is_dir() or root not in path.parents or hashlib.sha256(b).hexdigest()!=expected:raise ValueError('unsafe transfer/hash')
path.parent.mkdir(parents=True,exist_ok=True)
with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
os.chmod(path,0o444)
print(json.dumps({'path':str(path),'sha256':expected,'bytes':len(b)}))
'''
LINK='''
from pathlib import Path
import sys,json
r=Path(sys.argv[1]);parent=r/'unified_scail_local_v1/pretrained_weights';parent.mkdir();p=parent/'DWPose';p.symlink_to(r/'model_cache/DWPose',target_is_directory=True)
print(json.dumps({'existing_local_weights_linked':True,'models_loaded':False,'weight_copy_bytes':0}))
'''
results=[]
for worker,role,host in [('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]:
 local=BASE/role;receipts=[]
 for item in files:
  args=['python3','-c',RECV,str(local),item['relative'],item['sha256']]
  cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
  with Path(item['path']).open('rb') as f:p=subprocess.run(cmd,stdin=f,capture_output=True,timeout=30)
  if p.returncode:raise RuntimeError(worker+' snapshot transfer failed; do not repeat partial blindly')
  receipts.append(json.loads(p.stdout))
 cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026',host,shlex.join(['python3','-c',LINK,str(local)])]
 p=subprocess.run(cmd,capture_output=True,timeout=20)
 if p.returncode:raise RuntimeError(worker+' weight symlink setup failed')
 results.append({'worker':worker,'files':receipts,'weight_copy_bytes':0,'models_loaded':False})
record={'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_refs':T['code_refs'],'03_files':files,'other_nodes':results,'no_manifest_or_media_copy':True}
with (ROOT/'logs/unified_code_snapshot_preparation.json').open('x') as f:json.dump(record,f,indent=2);f.write(chr(10))
print(json.dumps({'snapshot_files':len(files),'nodes':4,'core_sources_unchanged':True,'models_loaded':False,'weight_copy_bytes':0}))
