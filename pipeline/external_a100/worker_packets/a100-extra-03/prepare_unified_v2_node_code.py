from pathlib import Path
import json,hashlib,subprocess,shlex
BASE=Path('/tmp/ms_pose2d_draft_20261026T1427Z');ROOT=BASE/'launcher'
refs={'unified_lane_worker_v2.py':'13d998589864f95833664c675819e7ef83d622c09534573f0bae5250808aca2a','unified_access_transport_v2.py':'2589592349d831054eb433001682d84df6a5495a60f89404650e5ef69986ec33'}
receiver='''
from pathlib import Path
import hashlib,sys,os,json
p=Path(sys.argv[1]);expected=sys.argv[2];b=sys.stdin.buffer.read()
if hashlib.sha256(b).hexdigest()!=expected:raise ValueError('transfer SHA mismatch')
if p.exists():
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('existing code differs;no overwrite')
else:
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.chmod(p,0o444)
print(json.dumps({'file':str(p),'SHA256':expected,'bytes':len(b)}))
'''
results=[]
for worker,role,host in [('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]:
 for name,sha in refs.items():
  target=BASE/role/name;args=['python3','-c',receiver,str(target),sha]
  cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
  with (ROOT/name).open('rb') as f:p=subprocess.run(cmd,stdin=f,capture_output=True,timeout=30)
  if p.returncode:raise RuntimeError(worker+' transfer failed;preserve state')
  results.append({'worker':worker,**json.loads(p.stdout)})
print(json.dumps(results))
