from pathlib import Path
import concurrent.futures,datetime,hashlib,json,os,shlex,subprocess
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
BASE=ROOT.parent
P=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_a100_fullframe_pilot_20261026_v1')
NOW=datetime.datetime.now(datetime.timezone.utc)
def load(p):
 b=Path(p).read_bytes()
 if bytes([0]) in b or not b.endswith(bytes([10])):raise ValueError('NUL or half JSON; preserve: '+str(p))
 return json.loads(b)
cursor_path=ROOT/'logs/relay_pilot_event_cursors.json';cursors=load(cursor_path)
REMOTE='''
from pathlib import Path
import sys,json,hashlib
root=Path(sys.argv[1]);cursor=int(sys.argv[2]);raw=(root/'logs/pilot_fullframe_progress.json').read_bytes()
if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('partial/NUL progress;retain')
p=json.loads(raw);result={'worker':p['worker_id'],'runid':p['runid'],'pid':p['pid'],'process_alive':Path('/proc/'+str(p['pid'])).exists(),'progress':{k:p.get(k) for k in ['status','processed','success','partial','failed','skipped','remaining','totalframes','model_init_counts','cachebytes','queuedwriteback','inflight']},'alarms':[]}
if p.get('summary_path'):
 s=Path(p['summary_path']);raw=s.read_bytes()
 if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('partial/NUL summary;retain')
 result['local_summary_sha256']=hashlib.sha256(raw).hexdigest();result['local_summary_path']=str(s)
file=root/'logs'/('pilot_fullframe_events_'+p['runid']+'.jsonl');result['event_path']=str(file);result['event_cursor']=cursor
if file.stat().st_size<cursor:result['alarms'].append('event truncated;cursor retained')
else:
 with file.open('rb') as f:
  f.seek(cursor)
  while True:
   at=f.tell();line=f.readline()
   if not line:break
   if bytes([0]) in line or not line.endswith(bytes([10])):result['alarms'].append('NUL/half event at '+str(at)+';cursor retained');break
   try:json.loads(line)
   except (ValueError,UnicodeDecodeError):result['alarms'].append('bad JSON at '+str(at)+';cursor retained');break
   result['event_cursor']=f.tell()
print(json.dumps(result))
'''
nodes=[('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]
def inspect(n):
 worker,role,host=n
 args=['python3','-c',REMOTE,str(BASE/role),str(cursors.get(worker,{}).get('cursor',0))]
 cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
 p=subprocess.run(cmd,capture_output=True,timeout=30)
 if p.returncode:return {'worker':worker,'alarms':['remote inspection failed;retain cursors'],'rc':p.returncode}
 if bytes([0]) in p.stdout or not p.stdout.endswith(bytes([10])):return {'worker':worker,'alarms':['partial/NUL inspection;retain cursors']}
 r=json.loads(p.stdout);r['host']=host
 if r['progress']['status'] not in ['success','partial','failed','stopped']:return r
 path=P/'results'/worker/'pilot_runs'/r['runid']/'summary.json';raw=path.read_bytes();s=load(path)
 same=hashlib.sha256(raw).hexdigest()==r.get('local_summary_sha256')
 if not same:r['alarms'].append('local/shared summary differs;no retry')
 required=['new_fullvideo_wholebody134.npz','actual_audio_beats.json','fullframe_manifest.json','case_receipt.json']
 missing=[];bundles=0;committed=0
 for c in s['cases']:
  case=P/'results'/worker/c['split']/c['id']
  absent=[name for name in required if not (case/name).is_file()]
  if absent:missing.append({'id':c['id'],'files':absent})
  committed+=int(c.get('publication_committed',False))
  bundles+=int(bool(c.get('bundle')) and Path(c['bundle']['path']).is_file())
  if c.get('status') not in ['success','partial']:continue
  m=load(case/'fullframe_manifest.json');receipt=load(case/'case_receipt.json')
  if m.get('schema')!='ms_fullframe_pose2d_manifest_v1' or receipt.get('new_npz_schema')!='ms_fullvideo_pose2d_cache_v1':r['alarms'].append('output schema mismatch:'+c['id'])
  if m['duration_seconds']!=m['metadata_provenance']['probe']['duration']:r['alarms'].append('video duration binding mismatch:'+c['id'])
  if any(m.get(k)!=v or receipt.get(k)!=v for k,v in {'semantic_approval':False,'quality_approved':False,'training_eligible':False,'needs_visual_revision':True}.items()):r['alarms'].append('unreviewed flags mismatch:'+c['id'])
 r['terminal']={'status':s['status'],'processed':s['actual_processed'],'success':s['success'],'partial':s['partial'],'failed':s['failed'],'skipped':s['skipped'],'remaining':s['remaining'],'totalframes':s['totalframes'],'e2e_seconds':s['e2e_seconds'],'model_init_counts':s['model_init_counts'],'cachebytes':s['cachebytes'],'summary_path':str(path),'summary_SHA_match':same,'committed_cases':committed,'bundles_present':bundles,'missing_outputs':missing,'caption_generated':False,'quality_approved':False,'training_eligible':False}
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(inspect,nodes))
for r in results:
 if 'event_cursor' in r:cursors[r['worker']]={'path':r['event_path'],'cursor':r['event_cursor']}
with cursor_path.open('w') as f:json.dump(cursors,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
m=load(ROOT/'logs/monitor_latest.json')
report={'checked_at':NOW.isoformat(),'date':NOW.date().isoformat(),'workers':results,'monitor':{'pid':m['monitor_pid'],'alive':Path('/proc/'+str(m['monitor_pid'])).exists(),'age_seconds':round((NOW-datetime.datetime.fromisoformat(m['time'])).total_seconds()),'alarms':m['alarms']},'03_reexecuted':False}
with (ROOT/'logs/last_pilot_terminal_inspection.json').open('w') as f:json.dump(report,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
print(json.dumps(report))
