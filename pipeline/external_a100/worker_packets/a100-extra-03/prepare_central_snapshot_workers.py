from pathlib import Path
import hashlib,json,subprocess,shlex
BASE=Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher'
files=['run_central_two_samples.py','ms_external_live_snapshot_v1_exact.py']
receiver='''
from pathlib import Path
import sys,hashlib,os,json
p=Path(sys.argv[1]);expected=sys.argv[2];raw=sys.stdin.buffer.read()
if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('observer transfer hash mismatch')
compile(raw,str(p),'exec')
if p.exists():
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('existing observer differs;do not overwrite')
else:
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'file':str(p),'sha256':expected}))
'''
result=[]
for worker,role,host in [('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]:
 for name in files:
  src=ROOT/name;expected=hashlib.sha256(src.read_bytes()).hexdigest();target=BASE/role/name
  cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(['python3','-c',receiver,str(target),expected])]
  with src.open('rb') as f:p=subprocess.run(cmd,stdin=f,capture_output=True,timeout=25)
  if p.returncode:raise RuntimeError('snapshot observer preparation failed '+worker)
  result.append({'worker':worker,**json.loads(p.stdout)})
print(json.dumps({'copied_observers':result,'GPU_workers_unchanged':True,'no_model_weight_or_media_copy':True}))
