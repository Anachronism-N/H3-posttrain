"""Bounded authorized-run observation: no model/source/lease operations."""
from pathlib import Path
import json,datetime,time,os
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
start=time.monotonic();first=None;history=[];result=None
while time.monotonic()-start<580:
 now=datetime.datetime.now(datetime.timezone.utc);raw=(ROOT/'logs/monitor_latest.json').read_bytes()
 if bytes([0]) in raw or not raw.endswith(bytes([10])):
  result={'event':'ACTION_REQUIRED','reason':'NUL/half snapshot; retained for reread','time':now.isoformat()};break
 m=json.loads(raw);alarms=list(m.get('alarms',[]));total=0;failed=0;active=0;workers=[]
 for node in m['nodes']:
  lanes=node.get('rollover_lanes',[]);published=sum(l['published'] for l in lanes);errors=sum(l['failed'] for l in lanes);total+=published;failed+=errors
  live=sum(l['status']=='running' and l.get('observed_process_identity',{}).get('alive',False) for l in lanes);active+=live
  for lane in lanes:
   if lane['status']=='running' and now.timestamp()-lane['heartbeat_unix']>180:alarms.append(node['server']+':GPU'+str(lane['physical_gpu_index'])+':stale heartbeat')
  workers.append({'worker':node['server'],'published_outcomes':published,'remaining':20000-published,'failures':errors,'live_models':live})
 if first is None:first={'published':total,'failed':failed,'time':now.isoformat()}
 record={'time':now.isoformat(),'published':total,'failures':failed,'live_models':active}
 if not history or record['published']!=history[-1]['published']:history.append(record)
 if failed>first['failed'] or alarms or active!=16 or total==80000:
  result={'event':'ACTION_REQUIRED' if failed>first['failed'] or alarms or active!=16 else 'FINAL_QUEUE_AUDIT_REQUIRED','time':now.isoformat(),'workers':workers,'alarms':alarms,'no_auto_source_retry':True};break
 time.sleep(15)
if result is None:result={'event':'OBSERVATION_INTERVAL_FINISHED','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'published_outcomes':total,'new_outcomes_observed':total-first['published'],'remaining':80000-total,'live_models':active,'sealed_failures':failed,'workers':workers,'alarms':alarms,'scope_goal_complete':False}
result['progress_history']=history
path=ROOT/'logs/bounded_authorized_observation_latest.json';tmp=path.with_suffix('.tmp')
with tmp.open('w') as f:json.dump(result,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
tmp.replace(path)
print(json.dumps(result),flush=True)
