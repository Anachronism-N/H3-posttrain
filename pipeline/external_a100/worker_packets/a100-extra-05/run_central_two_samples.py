"""Two local snapshot observations; no GPU or claim operations."""
import argparse,datetime,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
BASE=Path('/tmp/ms_pose2d_draft_20261026T1427Z')
SHARED=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
NEXT=SHARED/'external_ms_unified_clusters80k_20261027_v1'
a=argparse.ArgumentParser();a.add_argument('--node',required=True);a.add_argument('--role',required=True);x=a.parse_args()
root=BASE/x.role;script=SHARED/'runtime/ms_external_live_snapshot_v1.py' if x.role=='launcher' else root/'ms_external_live_snapshot_v1_exact.py';out=SHARED/'external_live_status'/(x.node+'.json') if x.role=='launcher' else root/'logs'/('central_live_snapshot_'+x.node+'.json');side=root/'logs'/('central_live_snapshot_'+x.node+'_progress.json')
def get(path):
 b=path.read_bytes()
 if bytes([0]) in b or not b.endswith(bytes([10])):raise ValueError('NUL/partial observation;preserve and reread')
 return json.loads(b)
def source_read(path):
 if x.role=='launcher':return path.read_bytes()
 code='from pathlib import Path;import sys;sys.stdout.buffer.write(Path(sys.argv[1]).read_bytes())'
 cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pilot_relay_known_hosts','-o','ConnectTimeout=8','11.239.36.34',shlex.join(['python3','-c',code,str(path)])]
 r=subprocess.run(cmd,capture_output=True,timeout=30)
 if r.returncode:raise RuntimeError('ticket relay verification unavailable;rc='+str(r.returncode))
 return r.stdout
ticket_path=NEXT/'assignments'/(x.node+'.ticket.json');queue_path=NEXT/(x.node+'.queue.json');ticket_raw=source_read(ticket_path);ticket=json.loads(ticket_raw);queue=json.loads(source_read(queue_path));ref=next(v for v in queue['tickets'] if v['path']==str(ticket_path))
if hashlib.sha256(ticket_raw).hexdigest()!=ref['sha256'] or ticket['worker_id']!=x.node:raise ValueError('own current queue/ticket identity mismatch')
observations=[]
def capture(sample):
 lanes=[]
 for gpu in range(4):
  p=root/'logs'/('rollover_gpu'+str(gpu)+'_progress.json')
  if not p.exists():lanes.append({'gpu':gpu,'state':'not_started'});continue
  v=get(p);pid=v['pid'];lpath=root/'gpu_locks'/('physical_gpu'+str(gpu)+'.newepoch.lease.json');lease=get(lpath) if lpath.exists() else {};proc=Path('/proc/'+str(pid))
  lanes.append({'gpu':gpu,'pid':pid,'status':v['status'],'alive':proc.exists(),'start_ticks':v['start_ticks'],'heartbeat_unix':v['heartbeat_unix'],'processed':v['unique_processed'],'reliable_published_after_commit':v['published'],'published_with_events':v['published_generated_with_events'],'partial_processed':v['partial'],'failed':v['failed'],'needs_regeneration':v['needs_regeneration'],'stage':v.get('stage'),'cachebytes':v['cache_actual_bytes'],'pending_sources':v['pending_sources'],'lease_path':str(lpath),'lease_identity_matches':lease.get('pid')==pid and lease.get('start_ticks')==v['start_ticks'] and lease.get('assignment_sha256')==v['assignment_sha256'],'gpu_memory_peak_mib':v['gpu_memory_peak_mib'],'rss_peak_bytes':v['rss_peak_bytes']})
 mem_limit=Path('/sys/fs/cgroup/memory/memory.limit_in_bytes');mem_usage=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes');mem={'headroom_bytes':int(mem_limit.read_text())-int(mem_usage.read_text())} if mem_limit.exists() else {}
 snapshot=get(out);s=snapshot['samples'][-1]
 observations.append({'sample':sample,'observed_UTC':s['observed_UTC'],'observed_unix':s['observed_unix'],'counter_read_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lanes':lanes,'reliable_published':sum(l.get('reliable_published_after_commit',0) for l in lanes),'failed':sum(l.get('failed',0) for l in lanes),'buffered_unpublished':sum(l.get('processed',0)-l.get('reliable_published_after_commit',0) for l in lanes),'memory':mem,'local_disk_free_bytes':__import__('shutil').disk_usage(root).free})
cmd=[sys.executable,'-B',str(script),'--node',x.node,'--gpus','0','1','2','3','--samples','2','--interval','60','--out',str(out)]
log=root/'logs'/('central_live_snapshot_'+x.node+'.stderr.log')
with log.open('x') as errors:
 proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=errors,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 for line in proc.stdout:
  msg=json.loads(line);capture(msg['sample']);print(json.dumps({'node':x.node,'sample':msg['sample'],'local_out':str(out),'published':observations[-1]['reliable_published']}),flush=True)
 rc=proc.wait()
 if rc:raise RuntimeError('central snapshot command failed rc='+str(rc)+';log retained')
if len(observations)!=2:raise ValueError('two completed samples required')
first,last=observations;delta=last['reliable_published']-first['reliable_published'];result={'schema':'external_two_sample_commit_progress_v1','node':x.node,'snapshot_script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'snapshot_source_original':str(SHARED/'runtime/ms_external_live_snapshot_v1.py'),'output_original':str(SHARED/'external_live_status'/(x.node+'.json')),'output_accessible_local':str(out),'output_mapping':'local observations reliably republished through authorized03relay' if x.role!='launcher' else 'same machine direct sharedpublish','current_ticket':str(ticket_path),'ticket_sha256':hashlib.sha256(ticket_raw).hexdigest(),'assignment_sha256':ticket['assignment_sha256'],'source_count':ticket['source_count'],'samples':observations,'new_reliably_committed_sources_between_samples':delta,'interval_seconds':last['observed_unix']-first['observed_unix'],'remaining_authorized_uncommitted':ticket['source_count']-last['reliable_published'],'failed_source_outcomes':last['failed'],'counter_basis':'frozen runner published increments only after batch commit readback and file publication verification;not a full payload rescan','model_calls':False,'process_signals':False,'source_claims_changed':False}
with side.open('x') as f:json.dump(result,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
print(json.dumps({'node':x.node,'two_samples_completed':True,'new_committed_sources':delta,'current_remaining':result['remaining_authorized_uncommitted'],'failed':last['failed'],'local_snapshot':str(out),'sidecar':str(side)}),flush=True)
