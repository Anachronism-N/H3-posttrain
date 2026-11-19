"""Read cached global VLM progress; the service reads only small commit metadata."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

ROOT = Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1")
SERVICE = Path('/root/stage2_progress_service_20261028_v1')
HOSTS = {'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225','N8':'28.216.18.136'}
SHARED = ROOT / 'stage2_dynamic_local_shared_results_20261028_v1'
EXTERNAL_NODES = tuple('a100-extra-0'+str(i) for i in range(3,7))
SUCCESS = {'postprocessed_unapproved','postprocessed_needs_visual_revision','insufficient_visible_evidence','published_unapproved','needs_visual_revision'}


def key(row):
    return row['id'].casefold()


def atomic(path, value):
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode()
    tmp = path.with_name(path.name + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def collect_node(node):
    program = '''import json,pathlib,time,hashlib
roots=[p for p in pathlib.Path('/root').glob('stage2_dynamic_*_'+NODE+'_20261028_v1') if (p/'deployment.json').exists()]
jobs=[];local_keys=set();ready_sources=0;upload=[]
for root in sorted(roots):
 d=json.loads((root/'deployment.json').read_bytes());d['configs']+=[json.loads(p.read_bytes()) for p in sorted((root/'extra_deployments').glob('*.json'))]
 for e in d['configs']:
  c=json.loads(pathlib.Path(e['config_ref']['path']).read_bytes());previous=c;binding=root/'runtime_bindings'/(c['owner']+'.json')
  if binding.exists():
   b=json.loads(binding.read_bytes());ref=b['config_ref'];raw=pathlib.Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];c=json.loads(raw)
  q=pathlib.Path(c['queue']);oldstate=root/'worker_state'/c['owner'];state=oldstate/c['phase'] if c.get('worker_state_layout')=='by_phase' else oldstate;r=dict(owner=c['owner'],pool=d['global_pool_namespace'],gpus=c['gpus'],admitted=(q/'live.json').exists(),loaded=(state/'load.receipt.json').exists(),worker_alive=False,processed_local=0,runtime_config_unix=pathlib.Path(c['config_path']).stat().st_mtime,previous_failure_preserved=binding.exists())
  if (q/'live.json').exists():
   live=json.loads((q/'live.json').read_bytes());i=live['child_identity'];p=pathlib.Path('/proc',str(i['pid']));r['worker_pid']=i['pid']
   try:
    st=(p/'stat').read_text().rsplit(') ',1)[1].split();r['worker_alive']=int(st[19])==i['start_ticks'] and st[0] not in ('Z','X','x');r['wchan']=(p/'wchan').read_text()
   except FileNotFoundError:pass
  r['failure']=json.loads((q/'failure.json').read_bytes()).get('error') if (q/'failure.json').exists() else None
  fqueue=root/'out'/'ready'/c['owner'];r['feeder_failed']=(fqueue/'FEEDER_FAILED').exists()
  pending=[p for p in fqueue.glob('*.ready.json') if not (root/'out'/'consumed_ready'/c['owner']/p.name).exists()];r['ready_sources']=sum(json.loads(p.read_bytes())['source_count'] for p in pending);ready_sources+=r['ready_sources']
  owned_keys=set()
  for current_state in [oldstate]+[p for p in oldstate.iterdir() if p.is_dir()] if oldstate.exists() else []:
   ps=sorted(current_state.glob('progress*.json'));p=current_state/'summary.json';p=p if p.exists() else ps[-1] if ps else None
   if p:
    v=json.loads(p.read_bytes());outcomes=v.get('all_outcomes',v.get('outcomes',[]));owned_keys.update(x['id'].casefold() for x in outcomes);r['idle_seconds']=v.get('idle_seconds',0)
  r['processed_local']=len(owned_keys);local_keys.update(owned_keys)
  jobs.append(r)
 for p in (root/'out'/'upload_metrics').glob('*.json'):
  v=json.loads(p.read_bytes());upload.append(dict(committed=v['committed'],wall_seconds=v['wall_seconds'],end_unix=v['end_unix']))
print(json.dumps(dict(node=NODE,jobs=jobs,local_keys=sorted(local_keys),ready_sources=ready_sources,upload_attempts=upload)))
'''.replace('NODE', repr(node))
    r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HOSTS[node],'/opt/venv_eval/bin/python','-B','-'], input=program, text=True, capture_output=True, timeout=20)
    if r.returncode:
        raise RuntimeError(node + ': ' + r.stderr[-400:])
    return json.loads(r.stdout)


def service():
    SERVICE.mkdir(parents=True, exist_ok=True)
    lock = os.open(SERVICE / 'collector.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_cache = {}; commit_cache = {}; history = deque(); initial = None; began = time.time(); baseline_written = (SERVICE/'baseline.json').exists()
    external_manifest = ROOT/'external_stage2_vlm_50000_20261028_v1/source_keys.jsonl'
    raw=external_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='9e5dad34ec7c7e10ec40b477b39ee9592b88c52c46b42e6535dab5b402e7a9fc':raise ValueError('Frozen external source-key allocation changed')
    external_allowed={tuple(json.loads(line)) for line in raw.splitlines() if line.strip()}
    for previous in (SERVICE/'service.log',SERVICE/'service_v2.log',SERVICE/'service_v3.log',SERVICE/'service_v4.log',SERVICE/'service_v5.log',SERVICE/'service_v6.log'):
        if previous.exists():
            for line in previous.read_text().splitlines():
                try:
                    v=json.loads(line); history.append((v['observed_unix'],v['vlm_processed_unique'],not v['errors']))
                except (ValueError,KeyError): continue
    if history: began=history[0][0]
    while not (SERVICE / 'STOP_V7').exists():
        started = time.monotonic(); errors = []; old_success = set(); old_failed = set(); reserved = set(); missing = []
        try:
            registry = json.loads((ROOT/'runtime/ms_vlm_postprocess_jobs_registry_v1.json').read_bytes())
            for j in registry['jobs']:
                reserved.update(x.casefold() for x in j.get('selected_source_ids', []))
                if not j['phase'].startswith('ms_fullframe_vlm_postprocess_'): continue
                name = j['config_path']
                if name not in old_cache:
                    cp = Path(name)
                    if not cp.exists(): missing.append(name); continue
                    c = json.loads(cp.read_bytes()); old_cache[name] = {'config':c,'closed':False,'outcomes':[]}
                item = old_cache[name]; c = item['config']; reserved.update(x.casefold() for x in c.get('selected_source_ids', []))
                if not item['closed']:
                    out = Path(c['out']); p = out/'summary.json'; closed = p.exists()
                    if not closed:
                        ps = sorted((out/'progress').glob('*.json')); p = ps[-1] if ps else None
                    if p:
                        v = json.loads(p.read_bytes()); item['outcomes'] = v.get('all_outcomes',v.get('outcomes',[])); item['closed'] = closed
                for x in item['outcomes']:
                    if x['status'] in SUCCESS and x.get('commit_ref'): old_success.add(key(x))
                    elif x['status'] == 'failed_sealed_no_retry' and x.get('commit_ref'): old_failed.add(key(x))
        except Exception as e: errors.append('old_registry: '+repr(e))
        dynamic = set(); dynamic_failed = set(); status_counts = Counter(); duplicates = 0; newest = None; published_nodes = {}
        try:
            for node_dir in SHARED.iterdir() if SHARED.exists() else []:
                if not node_dir.is_dir(): continue
                for batch in node_dir.iterdir():
                    p = batch/'commit.json'
                    if not p.is_file(): continue
                    name = str(p)
                    if name not in commit_cache:
                        raw = p.read_bytes(); v = json.loads(raw)
                        if v.get('schema') != 'stage2_shared_metadata_commit_v1': continue
                        if v['quality_approved'] is not False or v['training_eligible'] is not False: raise ValueError('Unexpected quality approval')
                        if v['node']!=node_dir.name or v['batch_id']!=batch.name:raise ValueError('Shared commit physical node/batch identity differs')
                        if node_dir.name in EXTERNAL_NODES and not set(map(tuple,v['source_keys']))<=external_allowed:raise ValueError('External result outside reserved50000 source group')
                        if len(v['source_keys']) != v['source_count'] or sum(v['status_counts'].values()) != v['source_count']: raise ValueError('Commit source count differs')
                        archive = Path(v['archive']['path'])
                        if archive.parent != batch or archive.stat().st_size != v['archive']['bytes']: raise ValueError('Committed archive absent or size differs')
                        commit_cache[name] = (v, p.stat().st_mtime, hashlib.sha256(raw).hexdigest())
                    v, stamp, digest = commit_cache[name]; keys = [k[1].casefold() for k in v['source_keys']]
                    if len(set(keys)) != len(keys): raise ValueError('Duplicate source in shared commit')
                    duplicates += len(dynamic.intersection(keys)); dynamic.update(keys); status_counts.update(v['status_counts']); newest = max(newest or stamp, stamp)
                    part=published_nodes.setdefault(node_dir.name,dict(keys=set(),batches=0,status_counts=Counter(),latest_shared_commit_unix=None))
                    part['keys'].update(keys);part['batches']+=1;part['status_counts'].update(v['status_counts']);part['latest_shared_commit_unix']=max(part['latest_shared_commit_unix'] or stamp,stamp)
        except Exception as e: errors.append('shared_commits: '+repr(e))
        external_keys=set(); main_keys=set(old_success); external_results=[]
        for name,part in published_nodes.items():
            if name in EXTERNAL_NODES:
                external_keys.update(part['keys']);counts=part['status_counts'];external_results.append(dict(node=name,shared_terminal_unique=len(part['keys']),processed_shared_unique=len(part['keys'])-counts.get('failed_sealed_no_retry',0),status_counts=dict(counts),batches=part['batches'],latest_shared_commit_unix=part['latest_shared_commit_unix'],live_model_verified=False,local_upload_ack_independently_verified=False))
            else:main_keys.update(part['keys'])
        if duplicates:errors.append('Duplicate source across shared batches')
        nodes = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {n:pool.submit(collect_node,n) for n in HOSTS}
            for node, future in futures.items():
                try: nodes.append(future.result())
                except Exception as e: errors.append(repr(e))
        local = set(); ready = 0; active = 0; admitted = 0; pauses = []
        for n in nodes:
            local.update(n.pop('local_keys')); ready += n['ready_sources']
            for j in n['jobs']:
                active += bool(j['loaded'] and j['worker_alive']); admitted += bool(j['admitted'] and j['worker_alive'])
            pauses += [u['wall_seconds'] for u in n.pop('upload_attempts') if u['wall_seconds'] >= 5]
        pool_states = Counter(); pools = {}
        for group in ('main3936','remaining25965','native37610','externalrem34803','legacy15973','archivegap452'):
            try:
                db = sqlite3.connect('file:/root/stage2_pool_'+group+'_20261028_v1/pool.db?mode=ro', uri=True, timeout=5)
                try: pools[group] = dict(db.execute('SELECT state,count(*) FROM sources GROUP BY state').fetchall())
                finally: db.close()
                pool_states.update(pools[group])
            except Exception as e: errors.append('pool '+group+': '+repr(e))
        all_done = old_success | dynamic
        # New shared batches carry aggregate statuses; do not guess per-source failed identities.
        dynamic_failed_count = status_counts.get('failed_sealed_no_retry',0)
        now = time.time()
        if initial is None: initial = set(all_done)
        history.append((now,len(all_done)-dynamic_failed_count,not errors))
        while len(history)>1 and history[1][0] <= now-1800: history.popleft()
        span = now-history[0][0]; valid_window = span>=1800 and all(x[2] for x in history)
        rate = (len(all_done)-dynamic_failed_count-history[0][1])*3600/span if valid_window else None
        value = dict(schema='stage2_realtime_progress_v1',observed_unix=now,collector_pid=os.getpid(),collector_started_unix=began,
                     historical_draft_count_estimate=175276,total_exact_known=False,registered_reserved_unique=len(reserved),
                     reliable_terminal_unique=len(all_done | old_failed),vlm_processed_unique=len(all_done)-dynamic_failed_count,
                     old_protocol_completed_unique=len(old_success),new_shared_terminal_unique=len(dynamic),new_shared_status_counts=dict(status_counts),
                     external_shared_terminal_unique=len(external_keys),external_shared_results=external_results,external_not_shared=50000-len(external_keys),external_main_overlap=len(external_keys & main_keys),main_shared_terminal_unique=len(main_keys),
                     external_model_live_status='not_independently_verified',loaded_active_replicas_scope='main_N1_N2_N3_N6_N8_only',
                     failed_sealed_unique=len(old_failed)+dynamic_failed_count,local_complete_pending_upload=len(local-dynamic),
                     approximate_remaining_not_vlm=175276-(len(all_done)-dynamic_failed_count),
                     dynamic_pool_states=dict(pool_states),dynamic_pools=pools,monitored_nodes=list(HOSTS),prepared_ready_sources=ready,loaded_active_replicas=active,admitted_live_replicas=admitted,
                     window_elapsed_seconds=span,global_30min_sources_per_hour=rate,global_30min_verified=valid_window,
                     includes_loading_idle_and_pauses=True,upload_stalls_over_5s=len(pauses),longest_upload_seconds=max(pauses,default=0),
                     latest_shared_commit_unix=newest,shared_duplicate_sources=duplicates,missing_old_configs=missing,errors=errors,nodes=nodes,
                     quality_approved=False,training_eligible=False)
        atomic(SERVICE/'latest.json',value)
        if not baseline_written and not errors:
            atomic(SERVICE/'baseline.json',dict(start_unix=now,completed_keys=sorted(all_done),reliable_terminal_unique=value['reliable_terminal_unique']))
            baseline_written=True
        if valid_window and not (SERVICE/'first_30min_report.json').exists(): atomic(SERVICE/'first_30min_report.json',value)
        print(json.dumps({k:value[k] for k in ('observed_unix','vlm_processed_unique','new_shared_terminal_unique','loaded_active_replicas','errors')}),flush=True)
        time.sleep(max(1,30-(time.monotonic()-started)))


def display(v, all_jobs=False):
    age=time.time()-v['observed_unix']; rate=v['global_30min_sources_per_hour']
    print(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(v['observed_unix']))+'  快照年龄 %.0fs'%age)
    print('VLM处理后可靠提交: %s | 失败封存: %s | 本地完成待上传: %s'%(v['vlm_processed_unique'],v['failed_sealed_unique'],v['local_complete_pending_upload']))
    print('新版共享终态: %s | 主群已加载活副本: %s | 主群已准入存活: %s | 主群已准备来源: %s'%(v['new_shared_terminal_unique'],v['loaded_active_replicas'],v['admitted_live_replicas'],v['prepared_ready_sources']))
    if 'external_shared_results' in v:
        print('外部03–06共享终态: %s/50000 | 未共享: %s（不等于未领）'%(v['external_shared_terminal_unique'],v['external_not_shared']))
        for node in sorted(v['external_shared_results'],key=lambda x:x['node']):print('  %s: 可靠提交%s，批数%s，模型活体未独立核实'%(node['node'],node['shared_terminal_unique'],node['batches']))
    print('全部已接入动态池: '+json.dumps(v['dynamic_pool_states'],ensure_ascii=False))
    for group,states in v['dynamic_pools'].items():
        if states.get('unused',0):print('  '+group+': 未领%s，在途%s'%(states.get('unused',0),states.get('assigned',0)))
    print('全局30分钟可靠吞吐: '+('%.1f 条/h'%rate if rate is not None else '尚未形成无缺口的完整30分钟窗口（已观察 %.0fs）'%v['window_elapsed_seconds']))
    print('历史草稿估算约175276；尚未VLM处理约%s（非当前全量exact分母）'%v['approximate_remaining_not_vlm'])
    print('未人审，不可训练。')
    if age>90: print('警告: 进度采集快照已过期，不能据此判断当前运行。')
    for n in v['nodes']:
        latest={}
        for j in n['jobs']:
            k=tuple(j['gpus'])
            if k not in latest or j.get('runtime_config_unix',0)>latest[k].get('runtime_config_unix',0):latest[k]=j
        jobs=n['jobs'] if all_jobs else list(latest.values())
        print(n['node']+'  '+'; '.join('%s: %s 本地%s ready%s%s'%(j['owner'],'运行' if j['loaded'] and j['worker_alive'] else '加载/待启动' if j['worker_alive'] else '未运行',j['processed_local'],j['ready_sources'],' 失败='+str(j['failure']) if j['failure'] else ' 供料失败' if j['feeder_failed'] else '') for j in jobs))
    if v['errors']: print('统计缺口: '+json.dumps(v['errors'],ensure_ascii=False))


def show(watch, as_json, all_jobs):
    while True:
        if (SERVICE/'latest.json').is_file(): raw=(SERVICE/'latest.json').read_text()
        else:
            code='import pathlib;print(pathlib.Path('+repr(str(SERVICE/'latest.json'))+').read_text())'
            r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HOSTS['N2'],'/opt/venv_eval/bin/python','-B','-'],input=code,text=True,capture_output=True,timeout=12)
            if r.returncode: raise RuntimeError('进度服务不可读: '+r.stderr[-400:])
            raw=r.stdout
        v=json.loads(raw)
        if watch and not as_json and os.isatty(1): print('\033[2J\033[H',end='')
        if as_json: print(json.dumps(v,ensure_ascii=False))
        else: display(v,all_jobs)
        if not watch: return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--service',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--json',action='store_true');p.add_argument('--all',action='store_true');a=p.parse_args()
    if a.service: service()
    else: show(a.watch,a.json,a.all)
