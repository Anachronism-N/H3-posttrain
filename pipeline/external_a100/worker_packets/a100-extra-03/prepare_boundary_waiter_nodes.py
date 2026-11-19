from pathlib import Path
import hashlib,shlex,subprocess,json
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher');source=ROOT/'rollover_boundary_waiter_v1.py';data=source.read_bytes();h=hashlib.sha256(data).hexdigest()
receiver='''
from pathlib import Path
import sys,os,hashlib,json
p=Path(sys.argv[1]);h=sys.argv[2];b=sys.stdin.buffer.read()
if hashlib.sha256(b).hexdigest()!=h:raise ValueError('waiter copy hash mismatch')
compile(b,str(p),'exec')
if p.exists():
 if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('existing waiter differs;no overwrite')
else:
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
os.chmod(p,0o444);print(json.dumps({'path':str(p),'sha256':h}))
'''
for role,host in [('worker-0','11.254.248.52'),('worker-1','11.239.106.11'),('worker-2','11.239.84.43')]:
 target=ROOT.parent/role/source.name;cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(['python3','-c',receiver,str(target),h])]
 p=subprocess.run(cmd,input=data,capture_output=True,timeout=25)
 if p.returncode:raise RuntimeError('safe waiter preparation failed;do not overwrite')
print(json.dumps({'prepared_waiter_workers':3,'sha256':h,'GPU_tasks_unchanged':True}))
