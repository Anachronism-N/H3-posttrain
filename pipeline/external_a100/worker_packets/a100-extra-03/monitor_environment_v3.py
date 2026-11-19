import concurrent.futures, datetime, hashlib, json, os, pathlib, subprocess, time
BASE=pathlib.Path('/tmp/ms_pose2d_draft_20261026T1427Z')
ROOT=BASE/'launcher'
LF=chr(10)
NODES=[('a100-extra-03','launcher',None),('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]
PROBE='''
import datetime,os,json,pathlib,subprocess,importlib.metadata,pwd,xml.etree.ElementTree as ET
role=os.environ.get('MS_MONITOR_ROLE','launcher');root=pathlib.Path('/tmp/ms_pose2d_draft_20261026T1427Z')/role
r={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'role':role,'probe_pid':os.getpid(),'gpus':[],'gpu_processes':[],'packages':{}}
p=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
r['nvidia_smi_rc']=p.returncode
if p.returncode==0:
 for line in p.stdout.splitlines():
  v=[s.strip() for s in line.split(',')];r['gpus'].append({'index':int(v[0]),'name':v[1],'total_mib':int(v[2]),'used_mib':int(v[3]),'utilization_percent':int(v[4])})
else:r['nvidia_smi_error']=p.stderr[:300]
p=subprocess.run(['nvidia-smi','-q','-x'],text=True,capture_output=True,timeout=10)
if p.returncode==0:
 try:
  for gpu in ET.fromstring(p.stdout).findall('gpu'):
   for proc in gpu.findall('processes/process_info'):
    entry={x.tag:x.text for x in proc};entry['gpu_id']=gpu.attrib.get('id');pid=entry.get('pid')
    if pid and pid.isdigit():
     try:
      uid=os.stat('/proc/'+pid).st_uid;entry['owner']=pwd.getpwuid(uid).pw_name;entry['uid']=uid
      ps=subprocess.run(['ps','-p',pid,'-o','lstart=,comm='],capture_output=True,text=True,timeout=3);entry['start_and_comm']=ps.stdout.strip()
     except OSError:entry['owner']='not_resolvable_in_container_pid_namespace'
    r['gpu_processes'].append(entry)
 except ET.ParseError:r['gpu_processes_alarm']='invalid_nvidia_xml'
for name in ['torch','onnxruntime-gpu','beat-this','rotary-embedding-torch']:
 try:r['packages'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:r['packages'][name]=None
for label,path in [('memory_limit','/sys/fs/cgroup/memory/memory.limit_in_bytes'),('memory_usage','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
 try:r[label]=int(pathlib.Path(path).read_text().strip())
 except (OSError,ValueError):r[label]=None
s=os.statvfs(root);r['local_disk_available_bytes']=s.f_bavail*s.f_frsize
r['shared_paths']={p:os.access(p,os.R_OK) and os.path.isdir(p) for p in ['/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain','/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1']}
r['model_cache_file_count']=sum(len(files) for _,_,files in os.walk(root/'model_cache'))
r['local_draft_files']=sum('draft.json' in files for _,_,files in os.walk(root/'results_unreviewed'))
r['pending_writeback_files']=sum(len(files) for _,_,files in os.walk(root/'pending_writeback'))
progress=root/'logs/fullvideo_validation_progress.json'
if progress.exists():
 try:
  raw=progress.read_bytes()
  if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('NUL_or_partial')
  r['validation_progress']=json.loads(raw)
  pid=r['validation_progress'].get('pid')
  if pid:
   p=subprocess.run(['ps','-p',str(pid),'-o','lstart=,comm='],capture_output=True,text=True,timeout=3);r['validation_process_identity']={'pid':pid,'start_and_comm':p.stdout.strip(),'alive':bool(p.stdout.strip())}
 except (ValueError,OSError,UnicodeDecodeError):r['validation_progress_alarm']='bad_or_partial_progress_JSON;retained;recheck_next_cycle'
source_ids=set()
for folder,dirs,files in os.walk(root/'results_unreviewed'):
 if 'draft.json' in files:source_ids.add(pathlib.Path(folder).name)
r['unique_local_draft_source_ids']=sorted(source_ids)
print(json.dumps(r,allow_nan=False),flush=True)
'''
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def probe(node):
 server,role,host=node;py=str(BASE/role/'venv/bin/python')
 if host:cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,'MS_MONITOR_ROLE='+role+' '+py+' -']
 else:cmd=[py,'-']
 try:
  p=subprocess.run(cmd,input=PROBE.encode(),capture_output=True,timeout=40,env={**os.environ,'MS_MONITOR_ROLE':role,'CUDA_VISIBLE_DEVICES':'','PYTHONWARNINGS':'ignore'})
  if p.returncode:return {'server':server,'role':role,'alarm':'probe_exit','returncode':p.returncode}
  raw=p.stdout;(ROOT/'logs'/('monitor_raw_'+role+'.txt')).write_bytes(raw)
  if bytes([0]) in raw or not raw.endswith(bytes([10])):return {'server':server,'role':role,'alarm':'NUL_or_partial_JSON;raw_retained;recheck_next_cycle'}
  try:r=json.loads(raw)
  except (ValueError,UnicodeDecodeError):return {'server':server,'role':role,'alarm':'bad_JSON;raw_retained;recheck_next_cycle'}
  r['server']=server;return r
 except subprocess.TimeoutExpired:return {'server':server,'role':role,'alarm':'probe_timeout'}
cursor=0;active={}
def read_events():
 global cursor
 p=ROOT/'logs/preparation_events.jsonl';events=[];alarms=[]
 if not p.exists():return events,alarms
 if p.stat().st_size<cursor:return [],['preparation_events_truncated;cursor_retained']
 with p.open('rb') as f:
  f.seek(cursor)
  while True:
   start=f.tell();line=f.readline()
   if not line:break
   if not line.endswith(bytes([10])) or bytes([0]) in line:alarms.append('partial_or_NUL_event_at_'+str(start)+';cursor_retained');break
   try:event=json.loads(line)
   except (ValueError,UnicodeDecodeError):alarms.append('bad_JSON_event_at_'+str(start)+';cursor_retained');break
   cursor=f.tell();events.append(event)
   if event.get('job') and event.get('state')=='started':active[event['job']]=event
   elif event.get('job') and event.get('state') in ['completed','failed','blocked']:active.pop(event['job'],None)
 return events,alarms
with (ROOT/'logs/monitor_identity_v3.json').open('x') as f:json.dump({'pid':os.getpid(),'started':now(),'interval_seconds':60,'script':__file__},f);f.write(LF)
log=(ROOT/'logs/monitor_samples_v3.jsonl').open('x',buffering=1)
started=time.monotonic();last_signature=None;last_change=started;stall_notified=False;cycle=0
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  while time.monotonic()-started<3*86400:
   cycle+=1;begin=time.monotonic();results=list(pool.map(probe,NODES));events,alarms=read_events()
   progress=[]
   for p in sorted((ROOT/'logs').glob('*')):
    if p.name.startswith(('dependency_','install_','model_cache_')) and p.is_file():
     s=p.stat();progress.append((p.name,s.st_size,s.st_mtime_ns))
   signature=hashlib.sha256(json.dumps({'progress':progress,'packages':[r.get('packages') for r in results],'cache_files':[r.get('model_cache_file_count') for r in results],'cursor':cursor},sort_keys=True).encode()).hexdigest()
   if signature!=last_signature:last_signature=signature;last_change=time.monotonic();stall_notified=False
   unchanged=time.monotonic()-last_change
   for r in results:
    if r.get('alarm'):alarms.append(r['server']+':'+r['alarm'])
    if r.get('validation_progress_alarm'):alarms.append(r['server']+':'+r['validation_progress_alarm'])
    vp=r.get('validation_progress',{});process=r.get('validation_process_identity',{})
    if vp.get('status') in ['running','started','processing'] and process and not process.get('alive'):alarms.append(r['server']+':validation_exited_without_terminal;do_not_auto_retry')
    if r.get('local_disk_available_bytes',2**63)<10*2**30:alarms.append(r['server']+':local_disk_below_10GiB')
   for key,event in active.items():
    pid=event.get('pid')
    if pid and not pathlib.Path('/proc/'+str(pid)).exists():alarms.append(key+':process_exited_without_completion_record;do_not_auto_retry')
   if unchanged>=300 and active and not stall_notified:alarms.append('preparation_no_progress_5min;do_not_auto_retry');stall_notified=True
   snapshot={'time':now(),'cycle':cycle,'monitor_pid':os.getpid(),'nodes':results,'new_preparation_events':events,'active_preparation_processes':list(active.values()),'alarms':alarms,'event_cursor':cursor,'unchanged_seconds':round(unchanged),'production':{'manifest_received':False,'gpu_authorized':True,'started':False,'success':0,'partial':0,'failure':0,'unique_sources':0,'windows':0,'source_throughput':'not_measured','window_throughput':'not_measured','feeding_wait':'waiting_for_formal_manifest_and_data_access','writeback_backlog_files':sum(r.get('pending_writeback_files',0) for r in results)}}
   line=json.dumps(snapshot,allow_nan=False);log.write(line+LF);log.flush();os.fsync(log.fileno())
   temp=ROOT/'logs/monitor_latest.json.tmp';temp.write_text(line+LF);temp.replace(ROOT/'logs/monitor_latest.json')
   print(json.dumps({'time':snapshot['time'],'cycle':cycle,'alarms':alarms,'gpu_process_count':sum(len(r.get('gpu_processes',[])) for r in results),'drafts_this_namespace':sum(r.get('local_draft_files',0) for r in results),'new_events':events},allow_nan=False),flush=True)
   time.sleep(max(1,60-(time.monotonic()-begin)))
finally:log.close()
