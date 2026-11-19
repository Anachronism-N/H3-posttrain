from pathlib import Path
import hashlib,shlex,subprocess,json
base=Path('/tmp/ms_pose2d_draft_20261026T1427Z')
source=base/'launcher/fullvideo_validated_helpers_ed283d684.py'
expected=hashlib.sha256(source.read_bytes()).hexdigest()
code='''
import pathlib,sys,hashlib,json,os
path=pathlib.Path(sys.argv[1]);b=sys.stdin.buffer.read();h=hashlib.sha256(b).hexdigest()
assert h==sys.argv[2]
if path.exists():
 assert hashlib.sha256(path.read_bytes()).hexdigest()==h
else:
 with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(path),'sha256':h,'verified':True}))
'''
results=[]
for worker,role,host in [('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]:
 target=base/role/source.name
 args=['python3','-c',code,str(target),expected]
 cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
 with source.open('rb') as f:p=subprocess.run(cmd,stdin=f,capture_output=True,timeout=20)
 if p.returncode:results.append({'worker':worker,'error':'transfer_rc_'+str(p.returncode)})
 else:results.append({'worker':worker,**json.loads(p.stdout)})
print(json.dumps({'model_copy_bytes':0,'results':results}))
