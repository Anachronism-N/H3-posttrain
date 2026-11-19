from pathlib import Path
import hashlib,json,os,shlex,subprocess
BASE=Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher'
refs={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['unified_lane_worker_v4.py','unified_access_transport_v3.py']}
receiver='''
from pathlib import Path
import sys,os,hashlib,json
p=Path(sys.argv[1]);h=sys.argv[2];b=sys.stdin.buffer.read()
if hashlib.sha256(b).hexdigest()!=h:raise ValueError('snapshot hash differs')
compile(b,str(p),'exec')
if p.exists():
 if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('existing snapshot differs;do not overwrite')
else:
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
os.chmod(p,0o444)
print(json.dumps({'path':str(p),'sha256':h}))
'''
for name,h in refs.items():
 compile((ROOT/name).read_bytes(),str(ROOT/name),'exec');os.chmod(ROOT/name,0o444)
for role,host in [('worker-0','11.254.248.52'),('worker-1','11.239.106.11'),('worker-2','11.239.84.43')]:
 for name,h in refs.items():
  args=['python3','-c',receiver,str(BASE/role/name),h];cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026',host,shlex.join(args)]
  with (ROOT/name).open('rb') as f:r=subprocess.run(cmd,stdin=f,capture_output=True,timeout=30)
  if r.returncode:raise RuntimeError('snapshot transfer failed;retain state')
print(json.dumps({'refs':refs,'prepared_nodes':4,'models_reloaded_by_prepare':0}))
