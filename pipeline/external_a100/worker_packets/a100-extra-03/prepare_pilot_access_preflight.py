from pathlib import Path
import hashlib,json,shlex,subprocess,os
base=Path('/tmp/ms_pose2d_draft_20261026T1427Z');source=base/'launcher/pilot_fullframe_access_v2.py';b=source.read_bytes();h=hashlib.sha256(b).hexdigest()
compile(b,str(source),'exec')
name='pilot_access_preflight_'+h[:12]+'.py'
receiver='''
import pathlib,sys,hashlib,os,json
p=pathlib.Path(sys.argv[1]);b=sys.stdin.buffer.read();h=hashlib.sha256(b).hexdigest()
if h!=sys.argv[2]:raise ValueError('transfer hash mismatch')
with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
os.chmod(p,0o444)
print(json.dumps({'path':str(p),'sha256':h,'scope':'CPU check-only candidate;not model run'}))
'''
result=[]
for w,role,host in [('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]:
 target=base/role/name
 args=['python3','-c',receiver,str(target),h]
 cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
 p=subprocess.run(cmd,input=b,capture_output=True,timeout=25)
 if p.returncode:raise RuntimeError(w+' preflightcopy failed')
 result.append({'worker':w,**json.loads(p.stdout)})
print(json.dumps({'candidate_sha256':h,'copies':result}))
