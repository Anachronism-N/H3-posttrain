"""User-authorized stop of owned annotation jobs: preserve every result/claim and restore then release GPUs."""
import argparse,fcntl,hashlib,json,os,re,select,signal,subprocess,time
from pathlib import Path
RT=Path(__file__).resolve().parent
REPORT=RT.parent/'user_stop_preserve_all_20261029_v1'
IDENTITY_KEYS=('pid','start_ticks','uid','argv','cgroup','exe')
HOLDER_SUFFIX='/diff/scripts_omnidance/exp/hold_gpu.py'

def identity(pid):
 p=Path('/proc',str(pid));s=(p/'stat').read_text().rsplit(') ',1)[1].split();return dict(pid=pid,ppid=int(s[1]),start_ticks=int(s[19]),state=s[0],uid=p.stat().st_uid,argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x],cgroup=(p/'cgroup').read_text(),exe=os.readlink(p/'exe'))

def same(i):
 try:r=identity(i['pid']);return all(r[k]==i[k] for k in IDENTITY_KEYS) and r['state'] not in ('Z','X','x')
 except (FileNotFoundError,ProcessLookupError):return False

def owned(i):
 scripts=[a for a in i['argv'] if a.endswith('.py') and a.startswith('/')]
 return next((s for s in scripts if s.startswith(('/root/stage2_',str(RT.parent)+'/')) or s.endswith(HOLDER_SUFFIX)),None)

def inventory():
 out={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:out[int(p.name)]=identity(int(p.name))
  except (OSError,UnicodeError):pass
 return out

def apps():
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);return [dict(uuid=a[0].strip(),pid=int(a[1]),mib=int(a[2])) for line in text.splitlines() if len(a:=line.split(','))==3]

def write_once(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(value,ensure_ascii=False)+'\n').encode();fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def run(node):
 local=Path('/root/stage2_user_stop_preserve_'+node+'_v1');local.mkdir(exist_ok=False);report=REPORT/node;report.mkdir(exist_ok=False);lock=os.open(local/'stop.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 inv=inventory();mine={p:i for p,i in inv.items() if owned(i)};cfgs={};managers={};blocked=set();outroots=set();events=[];unknown=[]
 for pid,i in mine.items():
  argv=i['argv'];script=owned(i)
  if '--config' not in argv:continue
  path=Path(argv[argv.index('--config')+1])
  try:c=json.loads(path.read_bytes())
  except (OSError,ValueError):continue
  cfgs[pid]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config=c)
  if c.get('out') and str(c['out']).startswith('/root/stage2_'):outroots.add(c['out'])
  if '--manager-sha' in argv and c.get('queue'):
   q=Path(c['queue']);managers[pid]=dict(identity=i,config_ref=dict(path=str(path),sha256=cfgs[pid]['sha256']),queue=str(q),out=c.get('out'),gpus=c.get('gpus',[]),natural_drain=Path(script).name.startswith(('ms_llm32_native_manager','qwen3_sameprompt_native_manager')))
   for p in q.glob('restore_blocked_gpu*.json'):
    try:v=json.loads(p.read_bytes());blocked.update(a['pid'] for a in v.get('apps',[]))
    except (OSError,ValueError):pass
 known_gpu=set();gpuapps=apps();gpu_paths={}
 for app in gpuapps:
  pid=app['pid'];chain=[];cur=pid;ok=pid in blocked
  for _ in range(20):
   i=inv.get(cur)
   if not i:break
   chain.append(i)
   if owned(i):ok=True
   if cur in blocked:ok=True
   cur=i['ppid']
   if cur<=1:break
  if ok and all(i['uid']==os.getuid() for i in chain):
   for i in chain:
    if owned(i) or i['argv'] and i['argv'][0].startswith('VLLM::'):known_gpu.add(i['pid'])
   gpu_paths[pid]=chain
  else:unknown.append(app)
 frozen={p:inv[p] for p in set(mine)|known_gpu if p in inv};fds={}
 for pid,i in frozen.items():
  try:fd=os.pidfd_open(pid)
  except ProcessLookupError:continue
  if same(i):fds[pid]=fd
  else:os.close(fd)
 before=dict(node=node,observed_unix=time.time(),latest_user_authorized_stop=True,processes=list(mine.values()),config_bindings=cfgs,native_managers=managers,owned_GPU_applications=[a for a in gpuapps if a['pid'] in gpu_paths],unknown_GPU_applications_not_signaled=unknown,known_model_descendants=[frozen[p] for p in known_gpu],raw_data_results_claims_budgets_preserved=True)
 write_once(local/'before.json',before);write_once(report/'before.json',before)
 signaled=set()
 def term(pid,why):
  if pid in signaled:return
  i=frozen.get(pid);fd=fds.get(pid)
  if not i or fd is None or select.select([fd],[],[],0)[0] or not same(i):return
  signal.pidfd_send_signal(fd,signal.SIGTERM);signaled.add(pid);event=dict(pid=pid,start_ticks=i['start_ticks'],argv=i['argv'],reason=why,observed_unix=time.time());events.append(event);write_once(local/('TERM_%d.json'%pid),event);print(json.dumps(dict(event='exact_owned_TERM',node=node,pid=pid,reason=why)),flush=True)
 # All actual feeder generations receive their own STOP marker; inputs already claimed are preserved.
 for pid,i in mine.items():
  script=owned(i);name=Path(script).name
  if 'feeder' not in name or pid not in cfgs:continue
  cfg=cfgs[pid]['config'];out=Path(cfg.get('out',''))
  if not out.is_absolute() or not str(out).startswith('/root/stage2_'):continue
  try:source=Path(script).read_text()
  except OSError:continue
  markers=set(re.findall(r"['\"](FEEDER_STOP[A-Z0-9_]*)['\"]",source)) or {'FEEDER_STOP'}
  for marker in markers:
   p=out/marker
   if not p.exists():write_once(p,dict(reason='User requested stop; preserve assigned packets and results',node=node,new_claims_authorized=False))
 # Prevent old polling controllers from admitting replacement workers during cleanup.
 for pid,i in mine.items():
  script=owned(i);name=Path(script).name
  if pid in managers or any(i['ppid']==m for m in managers) or name.endswith('hold_gpu.py') or 'feeder' in name or 'uploader' in name:continue
  if name in ('coarse3_review_server_wrapper_v1.py',):continue
  if name.startswith(('coarse6when_safe','stage2_owned_model_readcache_guard','caption_live_counts','caption_two_stage_live_monitor','update_salvage_error_review_web','motion_revision_latest','motion_revision_preserve','stage2_ack','motion_revision_ACK')):term(pid,'Stop owned annotation support/waiter; no new job admission')
 # Legacy native managers stop their exact direct children and perform their unchanged restore protocol.
 for pid,m in managers.items():
  q=Path(m['queue'])
  if not m['natural_drain'] and not (q/'resources_safe.json').exists():term(pid,'Stop legacy annotation native manager, preserve incomplete outputs')
 # Nonnative annotation workers have no restoration protocol; their known model ancestry is recorded first.
 for pid in known_gpu:
  if pid in managers:continue
  i=frozen[pid];script=owned(i)
  if script and not script.endswith('hold_gpu.py') and i['ppid'] not in managers:term(pid,'Stop recorded nonnative annotation worker')
 # Existing project holders blocking native cleanup are released; managers restore correct holders themselves.
 for pid,i in mine.items():
  if owned(i).endswith('hold_gpu.py'):term(pid,'Release project holder under explicit all-annotation-resources stop')
 # Orphan engines from prior stopped managers, and newly orphaned engines, are exact-known processes only.
 begun=time.monotonic();deadline=begun+900;last_cleanup=0
 while True:
  now=inventory()
  for pid in known_gpu:
   i=frozen[pid]
   if not i['argv'] or not i['argv'][0].startswith('VLLM::') or not same(i):continue
   lineage=gpu_paths.get(pid)
   chain=[];cur=pid
   for _ in range(20):
    x=now.get(cur)
    if not x:break
    chain.append(x);cur=x['ppid']
    if cur<=1:break
   active_parent=any(owned(x) and not owned(x).endswith('hold_gpu.py') and x['pid'] not in managers and x['state'] not in ('Z','X','x') for x in chain[1:])
   if not active_parent:term(pid,'Recorded vLLM survivor after annotation worker exited; preserve old files')
  safe=True
  for pid,m in managers.items():
   q=Path(m['queue']);p=q/'resources_safe.json'
   if not p.exists():safe=False;continue
   s=json.loads(p.read_bytes());assert all(s.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored'))
  if safe:break
  assert time.monotonic()<deadline,'Native restoration still blocked; do not force-release locks'
  time.sleep(3)
 # Restore completed. Stop only project holders now present, including fresh restored ones.
 fresh=inventory();released=[]
 for pid,i in fresh.items():
  script=owned(i)
  if not script or not script.endswith('hold_gpu.py'):continue
  if i['uid']!=os.getuid():continue
  fd=os.pidfd_open(pid)
  if same(i):signal.pidfd_send_signal(fd,signal.SIGTERM);released.append(dict(identity=i,reason='Native restoration complete; user requests bare free GPU, no reservation'))
  os.close(fd)
 until=time.monotonic()+40
 while time.monotonic()<until:
  current=apps();remain=[a for a in current if a['pid'] in known_gpu or any(a['pid']==x['identity']['pid'] for x in released)]
  if not remain:break
  time.sleep(2)
 current=apps();owned_remaining=[]
 for a in current:
  try:i=identity(a['pid'])
  except OSError:continue
  if a['pid'] in known_gpu or owned(i):owned_remaining.append(a)
 # Flush completed outputs now, before stopping the only node uploader.
 for out in map(Path,outroots):
  if not out.exists():continue
  if not (out/'GENERATION_DRAINED').exists():write_once(out/'GENERATION_DRAINED',dict(user_stopped_all_models=True,no_new_claims=True,observed_unix=time.time()))
 preservation=[]
 for out in map(Path,outroots):
  if not (out/'sources').exists():continue
  keys={tuple(p.relative_to(out/'sources').parts[:2]) for p in (out/'sources').glob('*/*/commit.json')};acked=set();pending=[]
  for directory in (out/'outbox').iterdir() if (out/'outbox').exists() else []:
   if not (directory/'sealed.json').exists():continue
   seal=json.loads((directory/'sealed.json').read_bytes());ack=directory/'upload_receipt.json'
   if ack.exists():
    a=json.loads(ack.read_bytes());assert a['batch_id']==seal['batch_id'] and a['archive_sha256']==seal['archive_sha256'];acked.update(map(tuple,seal['source_keys']))
   else:pending.append(str(directory))
  preservation.append(dict(local_out=str(out),local_source_commits=len(keys),reliable_ACKed_sources=len(acked),unacked_source_count=len(keys-acked),pending_archives=pending,all_original_files_preserved=True))
 report_value=dict(node=node,stopped_new_feeders=True,native_managers=len(managers),all_native_safe=True,owned_GPU_applications_remaining=owned_remaining,all_GPU_applications_now=current,unknown_GPU_apps_not_signaled=unknown,restored_project_holders_released=released,TERM_events=events,source_preservation=preservation,GPU_inventory_now=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),data_deleted=False,claims_deleted=False,budgets_reset=False,observed_unix=time.time())
 write_once(local/'gpu_release_report.json',report_value);write_once(report/'gpu_release_report.json',report_value);print(json.dumps(dict(event='annotation_GPUs_released',node=node,owned_GPU_apps_remaining=len(owned_remaining),all_gpu_apps_remaining=len(current),native_safe=True,preservation=preservation)),flush=True)
 for fd in fds.values():os.close(fd)
 os.close(lock)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=('N1','N2','N3','N4','N5','N6','N7','N8'));run(p.parse_args().node)
