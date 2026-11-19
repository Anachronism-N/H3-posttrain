from pathlib import Path
import concurrent.futures,datetime,hashlib,io,json,os,shlex,subprocess,sys,tarfile,tempfile,time
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher');BASE=ROOT.parent
SHARED=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
NODES=[('a100-extra-03','launcher',None),('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]
PACK='''
from pathlib import Path
import sys,io,tarfile,json,time
root=Path(sys.argv[1]);node=sys.argv[2];side=root/'logs'/('central_live_snapshot_'+node+'_progress.json');deadline=time.monotonic()+180
while not side.exists() and time.monotonic()<deadline:time.sleep(2)
if not side.exists():raise RuntimeError('two local samples not completed;logs preserved')
snapshot=root/'logs'/('central_live_snapshot_'+node+'.json')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for p,name in [(snapshot,'snapshot.json'),(side,'progress.json')]:
  raw=p.read_bytes()
  if len(raw)>8*2**20:raise ValueError('observation payload exceeds bounded limit')
  member=tarfile.TarInfo(name);member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
'''
def publish(path,raw):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.observer-',delete=False) as f:
  temp=Path(f.name);f.write(raw);f.flush();os.fsync(f.fileno())
 temp.replace(path)
 if hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(raw).digest():raise ValueError('snapshot publication hash differs')
 fd=os.open(path.parent,os.O_DIRECTORY|os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def collect(n):
 node,role,host=n;root=BASE/role
 if host is None:
  side=root/'logs'/('central_live_snapshot_'+node+'_progress.json');deadline=time.monotonic()+180
  while not side.exists() and time.monotonic()<deadline:time.sleep(2)
  if not side.exists():raise RuntimeError('03 two samples not complete')
  snapshot_raw=(SHARED/'external_live_status'/(node+'.json')).read_bytes();side_raw=side.read_bytes()
 else:
  args=['python3','-c',PACK,str(root),node];cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
  p=subprocess.run(cmd,capture_output=True,timeout=200)
  if p.returncode:raise RuntimeError(node+':observation relay collection failed;rc='+str(p.returncode))
  if len(p.stdout)>20*2**20:raise ValueError('observation package exceeds bounded limit')
  with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:') as t:
   names=[m.name for m in t.getmembers()]
   if set(names)!={'snapshot.json','progress.json'} or len(names)!=2:raise ValueError('unsafe observation package members')
   snapshot_raw=t.extractfile('snapshot.json').read();side_raw=t.extractfile('progress.json').read()
 for raw in [snapshot_raw,side_raw]:
  if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('partial/NUL observation JSON;preserve and reread')
 snap=json.loads(snapshot_raw);side=json.loads(side_raw)
 if snap['node']!=node or len(snap['samples'])!=2 or side['node']!=node or len(side['samples'])!=2 or side['snapshot_script_sha256']!='377fc29671edd59cdf39ce593919ef67dc7cac452fe4eacf195418edb9882103':raise ValueError('node/sample/source version mismatch')
 if side['new_reliably_committed_sources_between_samples']<0:raise ValueError('publication counters decreased')
 publish(SHARED/'external_live_status'/(node+'.json'),snapshot_raw)
 publish(SHARED/'external_live_status'/(node+'.commit_progress.json'),side_raw)
 row={'node':node,'hostname':snap['hostname'],'snapshot_observer_pid':snap['observer_pid'],'snapshot_path':str(SHARED/'external_live_status'/(node+'.json')),'snapshot_SHA256':hashlib.sha256(snapshot_raw).hexdigest(),'two_samples':len(snap['samples']),'sample_interval_seconds':side['interval_seconds'],'new_reliable_commit_sources':side['new_reliably_committed_sources_between_samples'],'current_ticket':side['current_ticket'],'remaining_uncommitted':side['remaining_authorized_uncommitted'],'failed':side['failed_source_outcomes'],'GPU_samples':[],'lease_identity_verified':all(l.get('lease_identity_matches') for s in side['samples'] for l in s['lanes'] if l.get('pid')),'actual_GPU_workers_not_signalled':snap['no_process_signals']}
 for sample in snap['samples']:
  row['GPU_samples'].append({'time':sample['observed_UTC'],'devices':sample['devices'],'apps':[{'PID':a['pid'],'MiB':a['used_MiB'],'verified_pipeline_ancestor':a['verified_pipeline_ancestor']} for a in sample['compute_apps']],'actual_kernel_FLOCKs':len(sample['pipeline_kernel_FLOCKs'])})
 row['counter_basis']=side['counter_basis'];row['memory_headroom_bytes']=side['samples'][-1]['memory'].get('headroom_bytes');row['local_disk_free_bytes']=side['samples'][-1]['local_disk_free_bytes'];row['buffered_unpublished']=side['samples'][-1]['buffered_unpublished']
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(collect,NODES))
report={'schema':'03to06_live_two_sample_acceptance_v1','created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':rows,'GPU_models_restarted':False,'source_claims_modified':False,'no_quality_approval':True}
path=ROOT/'logs/central_live_two_samples_report.json'
with path.open('x') as f:json.dump(report,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
print(json.dumps(report))
