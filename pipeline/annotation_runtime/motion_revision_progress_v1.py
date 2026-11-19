"""Only final English merged-window revision progress; obsolete captions are never counted."""
import argparse,json,time,sqlite3,subprocess,hashlib
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os,fcntl

ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
OUT=ROOT/'motion_revision_english_merged_edges_final_20261028_v1'
LOCAL=Path('/root/stage2_motion_revision_progress_20261028_v1')
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225','N8':'28.216.18.136'}
REVISION='english_merged_edges_motion_revision_v3'
EXTERNAL=tuple('a100-extra-0'+str(i) for i in range(1,7))


def atomic(p,v):
 raw=(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n').encode();tmp=p.with_name(p.name+'.tmp');fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)


def collect(node):
 code='''import pathlib,json,time,hashlib
root=pathlib.Path('/root/stage2_motion_revision_final_main_'+NODE+'_20261028_v1');r=[]
if (root/'deployment.json').exists():
 for e in json.loads((root/'deployment.json').read_bytes())['configs']:
  c=json.loads(pathlib.Path(e['config_ref']['path']).read_bytes());base=root/'worker_state'/c['owner'];binding=root/'runtime_bindings'/(c['owner']+'.json')
  if binding.exists():
   ref=json.loads(binding.read_bytes())['config_ref'];raw=pathlib.Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];c=json.loads(raw)
  q=pathlib.Path(c['queue']);s=base/c['phase'];ps=sorted(s.glob('progress*.json'));i=json.loads((q/'live.json').read_bytes())['child_identity'] if (q/'live.json').exists() else None;live=False
  if i:
   proc=pathlib.Path('/proc',str(i['pid']))
   try:a=(proc/'stat').read_text().rsplit(') ',1)[1].split();live=int(a[19])==i['start_ticks'] and a[0] not in ('Z','X','x')
   except FileNotFoundError:pass
  processed=0
  if base.exists():
   for prior in base.iterdir():
    if not prior.is_dir():continue
    records=sorted(prior.glob('progress*.json'));summary=prior/'summary.json';p=summary if summary.exists() else records[-1] if records else None
    if p:processed+=json.loads(p.read_bytes())['processed']
  queue=root/'out/ready'/c['owner'];ready=sum(json.loads(p.read_bytes())['source_count'] for p in queue.glob('*.ready.json') if not (root/'out/consumed_ready'/c['owner']/p.name).exists());r.append(dict(owner=c['owner'],gpus=c['gpus'],worker_pid=i['pid'] if i else None,live=live,loaded=(s/'load.receipt.json').exists(),local_processed=processed,batch_size=c.get('max_active_conversations',8),phase=c['phase'],ready_sources=ready,failure=json.loads((q/'failure.json').read_bytes()).get('error') if (q/'failure.json').exists() else None))
print(json.dumps(dict(node=NODE,slots=r)))
'''.replace('NODE',repr(node))
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HOSTS[node],'/opt/venv_eval/bin/python','-B','-'],input=code,text=True,capture_output=True,timeout=20)
 if p.returncode:raise ValueError(node+': '+p.stderr[-500:])
 return json.loads(p.stdout)


def run():
 LOCAL.mkdir(exist_ok=True);fd=os.open(LOCAL/'collector.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);cache={};history=deque()
 previous_logs=[Path('/root/stage2_motion_revision_support_services_20261028_v1/progress.log'),Path('/root/stage2_motion_revision_six_node_progress_service_20261028_v1/service.log')]
 for previous in previous_logs:
  if not previous.exists():continue
  for line in previous.read_text().splitlines():
   try:v=json.loads(line);history.append((v['observed_unix'],v['reliable_English_window_caption_sources'],not v['errors']))
   except (ValueError,KeyError):pass
 while not (LOCAL/'STOP_V3').exists():
  errors=[];completed=set();status=Counter();nodes=[];external_rows=[]
  for node in tuple(HOSTS)+EXTERNAL:
   node_keys=set();node_states=Counter()
   directory=OUT/'production_results'/node
   for batch in directory.iterdir() if directory.exists() else []:
    p=batch/'commit.json'
    if not p.exists():continue
    try:
     if str(p) not in cache:
      c=json.loads(p.read_bytes());assert c['schema']=='stage2_shared_metadata_commit_v1' and c['node']==node and c['source_count']==len(c['source_keys']) and sum(c['status_counts'].values())==c['source_count'];assert Path(c['archive']['path']).stat().st_size==c['archive']['bytes'];cache[str(p)]=c
     c=cache[str(p)];keys={k[1].casefold() for k in c['source_keys']};assert not completed&keys,'Duplicate source key in revision final result';completed.update(keys);status.update(c['status_counts']);node_keys.update(keys);node_states.update(c['status_counts'])
    except Exception as e:errors.append(repr(e))
   if node in EXTERNAL:external_rows.append(dict(node=node,reliable_terminal_sources=len(node_keys),reliable_caption_sources=len(node_keys)-node_states.get('failed_sealed_no_retry',0),status_counts=dict(node_states),live_model_verified=False))
  with ThreadPoolExecutor(max_workers=5) as pool:
   futures={n:pool.submit(collect,n) for n in HOSTS}
   for n,p in futures.items():
    try:nodes.append(p.result())
    except Exception as e:errors.append(repr(e))
  inputdb=sqlite3.connect('file:/root/stage2_motion_revision_inputs_20261028_v2/catalog.db?mode=ro',uri=True);input_count=inputdb.execute('SELECT count(*) FROM sources').fetchone()[0];blocked=inputdb.execute('SELECT count(*) FROM errors').fetchone()[0];latest_add=inputdb.execute('SELECT COALESCE(sum(source_count),0) FROM incremental_commits WHERE status="ready"').fetchone()[0] if inputdb.execute("SELECT 1 FROM sqlite_master WHERE name='incremental_commits'").fetchone() else 0;inputdb.close();model=sqlite3.connect('file:/root/stage2_motion_revision_claim_service_20261028_v3/model_pool.db?mode=ro',uri=True);states=dict(model.execute('SELECT state,count(*) FROM sources GROUP BY state'));allocations={}
  if model.execute("SELECT 1 FROM sqlite_master WHERE name='external_allocations'").fetchone():
   for node in EXTERNAL:allocations[node]=dict(model.execute('SELECT s.state,count(*) FROM sources s JOIN external_allocations a ON a.source_key=s.source_key WHERE a.node=? GROUP BY s.state',(node,)).fetchall())
  model.close();now=time.time();success=len(completed)-status.get('failed_sealed_no_retry',0);history.append((now,success,not errors))
  while len(history)>1 and history[1][0]<now-1800:history.popleft()
  elapsed=now-history[0][0];valid=elapsed>=1800 and all(x[2] for x in history);active=sum(x['live'] and x['loaded'] for n in nodes for x in n['slots']);value=dict(revision=REVISION,observed_unix=now,initial_original_source_count=176771,expected_latest_MS_sources=45185,latest_MS_added_unique=latest_add,prepared_original_inputs=input_count,blocked_original_inputs=blocked,model_pool_states=states,reliable_final_terminal_sources=len(completed),reliable_English_window_caption_sources=success,failed_sealed_sources=status.get('failed_sealed_no_retry',0),status_counts=dict(status),loaded_active_replicas=active,loaded_active_replicas_scope='main_only',external_allocation_states=allocations,external_results=external_rows,external_allocated_total=72000,external_models_not_yet_independently_verified=True,window_elapsed_seconds=elapsed,sources_per_hour_30min=(success-history[0][1])*3600/elapsed if valid else None,full_corpus_complete=False,old_wrong_caption_outputs_excluded=True,language='English',rich_role='dance_style_only',window_policy='merge_short_edges',errors=errors,nodes=nodes,quality_approved=False,training_eligible=False);atomic(LOCAL/'latest.json',value);print(json.dumps({k:value[k] for k in ('observed_unix','prepared_original_inputs','reliable_English_window_caption_sources','loaded_active_replicas','errors')}),flush=True);time.sleep(30)


def show(watch,raw):
 while True:
  p=LOCAL/'latest.json'
  if p.exists():v=json.loads(p.read_bytes())
  else:
   code='import pathlib;print(pathlib.Path('+repr(str(p))+').read_text())';r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HOSTS['N2'],'/opt/venv_eval/bin/python','-B','-'],input=code,text=True,capture_output=True,timeout=15)
   if r.returncode:raise RuntimeError('New-revision progress unavailable: '+r.stderr[-400:])
   v=json.loads(r.stdout)
  if raw:print(json.dumps(v,ensure_ascii=False))
  else:
   print('新协议: 英文 / 短边缘合并 / rich仅舞种 / 跨窗动作修订')
   print('可靠新正文%s | 失败封存%s | 主群已加载活副本%s'%(v['reliable_English_window_caption_sources'],v['failed_sealed_sources'],v['loaded_active_replicas']))
   print('原始输入已准备%s | 引用待修%s | 01–02最后一票已并入%s'%(v['prepared_original_inputs'],v['blocked_original_inputs'],v['latest_MS_added_unique']))
   print('30分钟可靠吞吐: '+('%.1f条/h'%v['sources_per_hour_30min'] if v['sources_per_hour_30min'] is not None else '新协议完整窗口尚未形成'))
   if v.get('external_allocation_states'):
    for node,states in sorted(v['external_allocation_states'].items()):print('外部 '+node+': '+json.dumps(states,ensure_ascii=False))
   for n in v['nodes']:print(n['node']+' '+ '; '.join('%s:%s 本地%s ready%s%s'%(x['owner'],'运行' if x['live'] and x['loaded'] else '加载' if x['live'] else '未准入/结束',x['local_processed'],x['ready_sources'],' 失败='+str(x['failure']) if x['failure'] else '') for x in n['slots']))
   if v['errors']:print('统计缺口: '+json.dumps(v['errors'],ensure_ascii=False))
   print('旧错误协议产量不计入；未人审，不可训练。')
  if not watch:break
  time.sleep(30)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--service',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args()
 if a.service:run()
 else:show(a.watch,a.json)
