from pathlib import Path
import hashlib,json,subprocess,shlex,os
BASE=Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher'
FILES={'unified_lane_rollover_v1.py':'95eff3b6af6104d090b318647bb6442aa95f6b536c6beb4349ff33f8138af1b9','unified_access_transport_rollover_v1.py':'c8d36e48e930b5dbd235bb21541fe05adc4b6307060bf2bb67658f5f7ff35406'}
REC='''
from pathlib import Path
import sys,json,os,hashlib
p=Path(sys.argv[1]);expected=sys.argv[2];b=sys.stdin.buffer.read()
if hashlib.sha256(b).hexdigest()!=expected:raise ValueError('code SHA mismatch')
compile(b,str(p),'exec')
if p.exists():
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('existing code differs;no overwrite')
else:
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
os.chmod(p,0o444)
print(json.dumps({'path':str(p),'sha256':expected}))
'''
result=[]
for worker,role,host in [('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]:
 for name,h in FILES.items():
  command=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026',host,shlex.join(['python3','-c',REC,str(BASE/role/name),h])]
  with (ROOT/name).open('rb') as f:r=subprocess.run(command,stdin=f,capture_output=True,timeout=30)
  if r.returncode:raise RuntimeError(worker+' file preparation failed;do not overwrite')
  result.append({'worker':worker,**json.loads(r.stdout)})
print(json.dumps({'prepared_files':result,'current_GPUworkers_untouched':True,'model_copy_bytes':0,'media_copy_bytes':0}))
