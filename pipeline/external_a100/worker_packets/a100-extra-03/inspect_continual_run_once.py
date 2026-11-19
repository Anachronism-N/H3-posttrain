"""Bounded local monitoring snapshot; does not read media or alter workers."""
from pathlib import Path
from collections import deque
import datetime,json,os
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
def load(path):
 raw=path.read_bytes()
 if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('NUL/partial JSON;preserve:'+str(path))
 return json.loads(raw)
log=ROOT/'logs/monitor_samples_v6.jsonl';samples=deque(maxlen=40);alarms=[]
cursor_path=ROOT/'logs/continual_queue_monitor_cursor.json'
previous_cursor=load(cursor_path) if cursor_path.exists() else {}
blocked_offset=previous_cursor.get('blocked_offset')
with log.open('rb') as stream:
 size=os.fstat(stream.fileno()).st_size;start=max(0,size-8*1024*1024)
 if previous_cursor.get('path',str(log))!=str(log):raise ValueError('monitor log path changed;cursor retained')
 if previous_cursor.get('offset',0)>size:raise ValueError('monitor log truncated;cursor retained')
 if blocked_offset is not None and blocked_offset<start:raise ValueError('unresolved bad line at '+str(blocked_offset)+' outside bounded tail;cursor retained,not skipped')
 stream.seek(start)
 if start and start!=blocked_offset:stream.readline()
 blocked_offset=None
 cursor=stream.tell()
 while True:
  at=stream.tell();line=stream.readline()
  if not line:break
  if bytes([0]) in line or not line.endswith(bytes([10])):blocked_offset=at;alarms.append('NUL/partial line at '+str(at)+';cursor retained');break
  try:samples.append(json.loads(line))
  except (ValueError,UnicodeDecodeError):blocked_offset=at;alarms.append('bad JSON at '+str(at)+';cursor retained');break
  cursor=stream.tell()
cursor_temp=cursor_path.with_suffix('.tmp')
with cursor_temp.open('w') as stream:
 json.dump({'path':str(log),'offset':cursor,'blocked_offset':blocked_offset},stream,indent=2);stream.write(chr(10));stream.flush();os.fsync(stream.fileno())
cursor_temp.replace(cursor_path)
if not samples:raise ValueError('no complete monitoring samples;cursor retained')
last=samples[-1];end=datetime.datetime.fromisoformat(last['time']);older=[s for s in samples if datetime.datetime.fromisoformat(s['time'])<=end-datetime.timedelta(minutes=15)];first=older[-1] if older else samples[0];duration=(end-datetime.datetime.fromisoformat(first['time'])).total_seconds()
def counts(sample):return {node['server']:sum(lane['published'] for lane in node.get('rollover_lanes',[])) for node in sample['nodes']}
before,after=counts(first),counts(last);now=datetime.datetime.now(datetime.timezone.utc);workers=[]
supervisor=load(ROOT/'logs/rollover_goal_supervisor_latest.json');audit=load(ROOT/'logs/rollover_payload_audit_latest.json')
committed={entry['worker']:entry['published_unique'] for entry in supervisor['workers']}
for node in last['nodes']:
 lanes=node.get('rollover_lanes',[]);worker=node['server'];alive=[l for l in lanes if l['status']=='running' and l.get('observed_process_identity',{}).get('alive')]
 for lane in lanes:
  if lane['status'] in ['running','models_loading'] and (not lane.get('observed_process_identity',{}).get('alive') or now.timestamp()-lane['heartbeat_unix']>180):alarms.append(worker+':GPU'+str(lane['physical_gpu_index'])+':process/heartbeat anomaly')
 workers.append({'worker':worker,'committed_outcomes':committed[worker],'remaining':20000-committed[worker],'monitored_published_outcomes':after[worker],'new_window_outcomes':after[worker]-before[worker],'failed_processed':sum(l['failed'] for l in lanes),'buffered_unpublished':sum(l['unique_processed']-l['published'] for l in lanes),'active_GPU':[l['physical_gpu_index'] for l in alive],'PID':[l['pid'] for l in lanes],'cache_MiB':round(sum(l['cache_actual_bytes'] for l in lanes)/2**20,1),'disk_free_GiB':round(node['local_disk_available_bytes']/2**30,2)})
totals=supervisor['totals']
if sum(w['committed_outcomes'] for w in workers)!=totals['published_unique']:alarms.append('supervisor worker and total commit counts disagree;do not claim consistent totals')
delta=sum(w['new_window_outcomes'] for w in workers);report={'checked_at':now.isoformat(),'date':now.date().isoformat(),'commit_snapshot_time':supervisor['time'],'audit_snapshot_time':audit['time'],'monitor_snapshot_time':last['time'],'commit_count_basis':'one incrementally validated supervisor snapshot for worker totals and remaining','window_count_basis':'resident runner published-after-commit counters at the two monitor snapshots;not additional sources to sum with totals','window_start':first['time'],'window_end':last['time'],'seconds':duration,'full15min':bool(older),'new_window_outcomes':delta,'outcomes_per_minute':delta*60/duration,'workers':workers,'commit_totals':totals,'deep_verified_outcomes':audit['verified_unique_sources'],'audit_backlog':totals['published_unique']-audit['verified_unique_sources'],'remaining_authorized':80000-totals['published_unique'],'monitor_alive':Path('/proc/'+str(last['monitor_pid'])).exists(),'monitor_age_seconds':round((now-end).total_seconds()),'alarms':alarms+last['alarms']+supervisor['alarms'],'goal_complete':False}
for name,value in [('continual_queue_15min_latest.json',report)]:
 path=ROOT/'logs'/name;temp=path.with_suffix('.tmp')
 with temp.open('w') as stream:json.dump(value,stream,indent=2);stream.write(chr(10));stream.flush();os.fsync(stream.fileno())
 temp.replace(path)
print(json.dumps(report),flush=True)
