"""Read-only finite-run supervisor; never launches, retries, or stops GPU workers."""
from pathlib import Path
import datetime,hashlib,json,os,time
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
P=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_clusters80k_20261027_v1')
WORKERS=['a100-extra-03','a100-extra-04','a100-extra-05','a100-extra-06']
FLAGS={'diagnostic_not_production':True,'semantic_approval':False,'quality_approved':False,'training_eligible':False,'needs_visual_revision':True}
LF=chr(10)
def now():return datetime.datetime.now(datetime.timezone.utc)
def atomic(path,value):
 tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(value,f,allow_nan=False);f.write(LF);f.flush();os.fsync(f.fileno())
 tmp.replace(path)
def read_json(path):
 raw=path.read_bytes()
 if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('partial/NUL JSON;file retained for reread: '+str(path))
 return json.loads(raw),raw
assignments={};tickets={}
for worker in WORKERS:
 ticket,traw=read_json(P/'assignments'/(worker+'.ticket.json'))
 path=Path(ticket['assignment_path']);raw=path.read_bytes()
 if bytes([0]) in raw or not raw.endswith(bytes([10])) or hashlib.sha256(raw).hexdigest()!=ticket['assignment_sha256']:raise ValueError('invalid original assignment: '+worker)
 rows=[json.loads(line) for line in raw.split(bytes([10]))[:-1]]
 if len(rows)!=20000 or len({(r['split'],r['id']) for r in rows})!=20000 or ticket['worker_id']!=worker:raise ValueError('assignment scope mismatch')
 assignments[worker]=rows;tickets[worker]=ticket
checkpoint=ROOT/'logs/rollover_verified_commit_cursor.json'
if checkpoint.exists():state,_=read_json(checkpoint)
else:state={'commits':{},'sources':{w:{} for w in WORKERS},'latest_changes':{},'invalid_commit_alarms':{}}
started=time.monotonic();sample_cursor=0;iteration=0
with (ROOT/'logs/rollover_goal_supervisor_identity.json').open('x') as f:json.dump({'pid':os.getpid(),'started':now().isoformat(),'interval_seconds':60,'scope':'read_only_current80000assignment','retries':False},f);f.write(LF)
print(json.dumps({'event':'supervisor_started','pid':os.getpid(),'time':now().isoformat(),'scope_sources':80000,'GPU_workers_untouched':True}),flush=True)
while time.monotonic()-started<3*86400:
 iteration+=1;begin=time.monotonic();timepoint=now();alarms=[];new_commits=0
 try:
  snapshot,_=read_json(ROOT/'logs/monitor_latest.json')
  age=(timepoint-datetime.datetime.fromisoformat(snapshot['time'])).total_seconds()
  monitor_alive=Path('/proc/'+str(snapshot['monitor_pid'])).exists()
  if not monitor_alive or age>150:alarms.append('central monitor exited or stale;no automatic restart')
  alarms+=snapshot.get('alarms',[])
  all_terminal=True;lanes_total=0;active_lanes=0;failed_total=0;details=[]
  for node in snapshot['nodes']:
   worker=node['server']
   if worker not in WORKERS:continue
   lanes=node.get('rollover_lanes',[])
   if len(lanes)!=4:
    all_terminal=False
    waiters=node.get('rollover_waiters',[])
    old_alive=any(l.get('status')=='running' and l.get('observed_process_identity',{}).get('alive') for l in node.get('unified_lanes',[]))
    if not old_alive and not waiters:alarms.append(worker+':next queue has no running lane or managed boundary waiter')
    for waiter in waiters:
     if waiter.get('status') in ['blocked_partial_or_NUL_progress','blocked_old_exit_without_terminal','next_lane_failed_sealed_no_retry','waiter_safety_cap_expired']:alarms.append(worker+':GPU'+str(waiter['gpu'])+':'+waiter['status'])
   for lane in lanes:
    lanes_total+=1;gpu=lane['physical_gpu_index'];pid=lane['pid'];status=lane['status'];identity=lane.get('observed_process_identity',{});alive=identity.get('alive',False)
    failed_total+=lane.get('failed',0)
    if status in ['running','models_loading']:
     all_terminal=False;active_lanes+=int(alive)
     if not alive or timepoint.timestamp()-lane.get('heartbeat_unix',0)>180:alarms.append(worker+':GPU'+str(gpu)+':process/heartbeat anomaly;no retry')
    elif status=='failed_sealed_no_retry':alarms.append(worker+':GPU'+str(gpu)+':sealed worker failure:'+str(lane.get('error')))
    cache=lane.get('cache_actual_bytes',0)+lane.get('cache_reserved_bytes',0)
    if cache>8*2**30:alarms.append(worker+':GPU'+str(gpu)+':cache reservation over8GiB')
    signature=[lane.get('unique_processed'),lane.get('published'),lane.get('current_source'),lane.get('current_total_frames'),status,lane.get('cache_actual_bytes'),lane.get('cache_reserved_bytes')]
    key=worker+'/gpu'+str(gpu);change=state['latest_changes'].get(key,{})
    if change.get('signature')!=signature:change={'signature':signature,'time':timepoint.isoformat(),'alarm_reported':False}
    unchanged=(timepoint-datetime.datetime.fromisoformat(change['time'])).total_seconds()
    if status=='running' and unchanged>900 and not change.get('alarm_reported'):
     alarms.append(key+':no processed or buffered-cache growth15min;verify publication channel before declaring stall;inspect only affected lane');change['alarm_reported']=True
    state['latest_changes'][key]=change
    details.append({'worker':worker,'gpu':gpu,'pid':pid,'status':status,'alive':alive,'processed':lane.get('unique_processed'),'published_counter':lane.get('published'),'unpublished_processed':lane.get('unique_processed',0)-lane.get('published',0),'failed':lane.get('failed',0),'cachebytes':cache,'file_or_source_unchanged_seconds':round(unchanged)})
  for worker in WORKERS:
   output=P/'results'/worker
   for commit_path in (output/'batches').glob('gpu*/*/commit.json'):
    key=str(commit_path)
    if key in state['commits']:continue
    try:
     commit,raw=read_json(commit_path)
     if commit['edition']!=P.name or commit['worker']!=worker or commit['assignment_sha256']!=tickets[worker]['assignment_sha256'] or commit['gpu'] not in range(4):raise ValueError('commit worker/edition/assignment/GPU mismatch')
     staged={};frames=0;counts={'partial':0,'draft_generated':0,'needs_regeneration':0,'failed':0};events=0
     for receipt in commit['receipts']:
      index=receipt.get('assignment_index');row=receipt.get('source_row')
      if type(index) is not int or not 0<=index<20000 or index%4!=commit['gpu'] or row!=assignments[worker][index]:raise ValueError('source assignment binding mismatch')
      if receipt['status']=='existing_claim_not_reassigned':continue
      sourcekey=row['split']+'/'+row['id']
      if sourcekey in staged or sourcekey in state['sources'][worker]:raise ValueError('duplicate committed source')
      if any(receipt.get(k)!=v for k,v in FLAGS.items()):raise ValueError('receipt approval/diagnostic flags mismatch')
      status='failed' if receipt['status']=='failed_sealed_no_retry' else receipt['draft_status']
      if status not in counts:raise ValueError('unknown draft terminal status')
      counts[status]+=1;frames+=receipt.get('frame_count',0);events+=receipt.get('event_count',0)
      case=commit_path.parent/'cases'/row['split']/row['id']
      if status!='failed':
       if status!='needs_regeneration' and (not receipt.get('generated_with_action_events') or not receipt.get('event_count')):raise ValueError('published event draft without evidenced action events')
       expected_files={str(case/name) for name in ['pose.npz','fullframe_manifest.json','beats.json','draft.json','receipt.json']}
       if not expected_files.issubset({item['path'] for item in commit['publications']}):raise ValueError('committed source lacks published five-file records')
      staged[sourcekey]={'assignment_index':index,'status':status,'frames':receipt.get('frame_count',0),'events':receipt.get('event_count',0),'commit_path':key}
     publications=commit['publications']
     for item in publications:
      path=Path(item['path'])
      if not path.is_relative_to(commit_path.parent):raise ValueError('publication path outside commit scope')
      if type(item['receipt'].get('bytes')) is not int or item['receipt']['bytes']<0 or item['sha256']!=item['receipt']['sha256'] or item['receipt'].get('path')!=item['path']:raise ValueError('verified publication receipt size/SHA/path binding mismatch')
     if len(staged)!=commit['unique_sources_attempted'] or sum(bool(s['events']) for s in staged.values())!=commit['unique_sources_with_action_events']:raise ValueError('commit source totals mismatch')
     state['sources'][worker].update(staged);state['commits'][key]={'sha256':hashlib.sha256(raw).hexdigest(),'verified_at':timepoint.isoformat(),'sources':len(staged),'frames':frames,'counts':counts,'events':events,'deep_payload_SHA_checked':False,'verification_scope':'commit publication receipts;independent payload audit only at completion'};new_commits+=1
     state['invalid_commit_alarms'].pop(key,None)
    except (KeyError,ValueError,OSError) as e:
     alarms.append('commit validation failed;retain for reread:'+key+':'+str(e));state['invalid_commit_alarms'][key]=str(e)
  log=ROOT/'logs/monitor_samples_v6.jsonl'
  if log.stat().st_size<sample_cursor:alarms.append('sample log truncated;cursor retained')
  else:
   with log.open('rb') as f:
    f.seek(sample_cursor)
    while True:
     offset=f.tell();line=f.readline()
     if not line:break
     if bytes([0]) in line or not line.endswith(bytes([10])):alarms.append('NUL/half monitoring line at'+str(offset)+';cursor retained');break
     try:json.loads(line)
     except (ValueError,UnicodeDecodeError):alarms.append('bad monitoring JSON at'+str(offset)+';cursor retained');break
     sample_cursor=f.tell()
  totals={'published_unique':sum(len(v) for v in state['sources'].values()),'partial':sum(s['status']=='partial' for v in state['sources'].values() for s in v.values()),'failed_published':sum(s['status']=='failed' for v in state['sources'].values() for s in v.values()),'frames_published':sum(s['frames'] for v in state['sources'].values() for s in v.values()),'events_published':sum(s['events'] for v in state['sources'].values() for s in v.values())}
  report={'time':timepoint.isoformat(),'iteration':iteration,'supervisor_pid':os.getpid(),'monitor_pid':snapshot['monitor_pid'],'active_model_lanes':active_lanes,'total_lanes':lanes_total,'new_commits_verified':new_commits,'workers':[{'worker':w,'published_unique':len(state['sources'][w]),'remaining':20000-len(state['sources'][w])} for w in WORKERS],'totals':totals,'source_failures_reported':failed_total,'alarms':alarms,'sample_cursor':sample_cursor,'lanes':details,'status':'completion_pending_final_payload_audit' if totals['published_unique']==80000 and all_terminal else 'running','caption_approval':False}
  atomic(checkpoint,state);atomic(ROOT/'logs/rollover_goal_supervisor_latest.json',report)
  print(json.dumps({k:report[k] for k in ['time','iteration','status','active_model_lanes','new_commits_verified','workers','totals','alarms']},allow_nan=False),flush=True)
  if totals['published_unique']==80000 and all_terminal and not alarms:
   print(json.dumps({'event':'all80000_committed_and_workers_terminal','final_audit_required':True,'checkpoint':str(checkpoint)}),flush=True);break
  if alarms:
   print(json.dumps({'event':'ACTION_REQUIRED','no_automatic_retry':True,'alarms':alarms}),flush=True)
 except Exception as e:
  message={'event':'ACTION_REQUIRED','time':now().isoformat(),'reason':type(e).__name__+':'+str(e),'no_automatic_retry':True}
  atomic(ROOT/'logs/rollover_goal_supervisor_alarm.json',message);print(json.dumps(message),flush=True)
 time.sleep(max(1,60-(time.monotonic()-begin)))
else:print(json.dumps({'event':'supervisor_safety_cap_expired','GPU_workers_not_stopped':True}),flush=True)
