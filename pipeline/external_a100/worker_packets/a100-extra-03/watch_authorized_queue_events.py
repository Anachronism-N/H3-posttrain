from pathlib import Path
import datetime,json,time
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
last_alert=None;next_status=time.monotonic();cursor=0
while True:
 try:
  now=datetime.datetime.now(datetime.timezone.utc)
  raw=(ROOT/'logs/monitor_latest.json').read_bytes()
  if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('NUL/partial monitoring JSON;retain for reread')
  value=json.loads(raw);alerts=list(value.get('alarms',[]));age=(now-datetime.datetime.fromisoformat(value['time'])).total_seconds()
  if age>180:alerts.append('central monitor heartbeat stale')
  queues=[]
  for node in value['nodes']:
   old=node.get('unified_lanes',[]);new=node.get('rollover_lanes',[])
   for lane in new:
    if lane['status']=='failed_sealed_no_retry':alerts.append(node['server']+':GPU'+str(lane['physical_gpu_index'])+':'+str(lane.get('error')))
    if lane['status']=='running' and not lane.get('observed_process_identity',{}).get('alive'):alerts.append(node['server']+':GPU'+str(lane['physical_gpu_index'])+':PID exited')
   for waiter in node.get('rollover_waiters',[]):
    if waiter['status'].startswith('blocked') or waiter['status']=='next_lane_failed_sealed_no_retry':alerts.append(node['server']+':'+waiter['status'])
   queues.append({'worker':node['server'],'old_published':sum(l['published'] for l in old)+(3 if node['server']=='a100-extra-03' else 0),'new_published':sum(l['published'] for l in new),'new_partial_processed':sum(l['partial'] for l in new),'new_failures':sum(l['failed'] for l in new),'active_GPUs':[l['physical_gpu_index'] for l in new if l['status']=='running' and l.get('observed_process_identity',{}).get('alive')],'remaining_new':20000-sum(l['published'] for l in new)})
  for name in ['rollover_goal_supervisor_alarm.json','rollover_payload_audit_alarm.json']:
   p=ROOT/'logs'/name
   if p.exists():alerts.append(name+':'+p.read_text()[:2000])
  signature=json.dumps(alerts,sort_keys=True)
  if alerts and signature!=last_alert:
   print(json.dumps({'event':'ACTION_REQUIRED','time':now.isoformat(),'alarms':alerts,'queues':queues,'no_auto_source_retry':True}),flush=True);last_alert=signature
  if time.monotonic()>=next_status:
   print(json.dumps({'event':'FIFTEEN_MINUTE_EXECUTION_STATUS','time':now.isoformat(),'queues':queues,'goal_complete':False,'source_processing_unchanged':True}),flush=True);next_status=time.monotonic()+900
  acceptance=ROOT/'logs/rollover80000_artifact_acceptance.json'
  if acceptance.exists():
   a=json.loads(acceptance.read_bytes())
   if a.get('unique_sources_verified')==80000 and not any(q['active_GPUs'] for q in queues):
    print(json.dumps({'event':'ALL_PREAUTHORIZED_OUTPUTS_DEEP_VERIFIED','time':now.isoformat(),'acceptance':str(acceptance),'final_authoritative_check_required':True}),flush=True);break
 except Exception as error:
  alert=type(error).__name__+':'+str(error)
  if alert!=last_alert:print(json.dumps({'event':'ACTION_REQUIRED','error':alert,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'retain_cursor_no_retry':True}),flush=True);last_alert=alert
 time.sleep(5)
