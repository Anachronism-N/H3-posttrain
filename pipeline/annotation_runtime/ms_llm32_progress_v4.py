"""Reliable text-stage progress only; old visual outputs are never counted."""
import argparse,fcntl,hashlib,json,os,sqlite3,subprocess,time
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
RT=Path(__file__).parent
SHARED=RT.parent/'MS_LLM32_english_merged_edges_v4'
LOCAL=Path('/root/stage2_MS_LLM32_progress_v4')
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225','N8':'28.216.18.136'}
EXTERNAL=tuple('a100-extra-0'+str(i) for i in range(1,7))

def collect(node):
    program='''import pathlib,json,hashlib,time
r=pathlib.Path('/root/stage2_MS_LLM32_main_'+NODE+'_v4');out=[]
if (r/'deployment.json').exists():
 for entry in json.loads((r/'deployment.json').read_bytes())['configs']:
  c=json.loads(pathlib.Path(entry['config_ref']['path']).read_bytes());q=pathlib.Path(c['queue']);state=r/'worker_state'/c['owner']/c['phase'];i=json.loads((q/'live.json').read_bytes())['child_identity'] if (q/'live.json').exists() else None;live=False
  if i:
   try:s=(pathlib.Path('/proc',str(i['pid']))/'stat').read_text().rsplit(') ',1)[1].split();live=int(s[19])==i['start_ticks'] and s[0] not in ('Z','X','x')
   except FileNotFoundError:pass
  progress=sorted(state.glob('progress*.json'));p=json.loads(progress[-1].read_bytes()) if progress else {};out.append(dict(owner=c['owner'],gpus=c['gpus'],live=live,loaded=(state/'load.receipt.json').exists(),processed=p.get('processed',0),idle_seconds=p.get('idle_seconds',0),worker_pid=i['pid'] if i else None,failure=json.loads((q/'failure.json').read_bytes())['error'] if (q/'failure.json').exists() else None))
print(json.dumps(dict(node=NODE,slots=out)))
'''.replace('NODE',repr(node))
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HOSTS[node],'/opt/venv_eval/bin/python','-B','-'],input=program,text=True,capture_output=True,timeout=20)
    if r.returncode:raise RuntimeError(node+': '+r.stderr[-300:])
    return json.loads(r.stdout)

def run():
    LOCAL.mkdir(exist_ok=True);fd=os.open(LOCAL/'service.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);cache={};history=deque();contract=json.loads((SHARED/'revision_contract.json').read_bytes())
    while not (LOCAL/'STOP').exists():
        counts=Counter();completed=set();errors=[];external=[]
        for node in tuple(HOSTS)+EXTERNAL:
            localcount=Counter();d=SHARED/'production_results'/node
            for batch in d.iterdir() if d.exists() else []:
                p=batch/'commit.json'
                if not p.exists():continue
                try:
                    if str(p) not in cache:
                        c=json.loads(p.read_bytes());assert c['node']==node and c['source_count']==len(c['source_keys']) and Path(c['archive']['path']).stat().st_size==c['archive']['bytes'];cache[str(p)]=c
                    c=cache[str(p)];keys={id_.casefold() for split,id_ in c['source_keys']};assert len(keys)==c['source_count'] and not completed&keys;completed.update(keys);counts.update(c['status_counts']);localcount.update(c['status_counts'])
                except Exception as e:errors.append(repr(e))
            if node in EXTERNAL:external.append(dict(node=node,shared_caption_sources=localcount.get('postprocessed_unapproved',0),failed_sealed=localcount.get('failed_sealed_no_retry',0),live_models_independently_verified=False))
        with ThreadPoolExecutor(max_workers=5) as executor:
            fs={n:executor.submit(collect,n) for n in HOSTS};nodes=[]
            for node,future in fs.items():
                try:nodes.append(future.result())
                except Exception as e:errors.append(repr(e))
        db=sqlite3.connect('file:/root/stage2_MS_LLM32_claim_service_v4/model_pool.db?mode=ro',uri=True);states=dict(db.execute('SELECT state,count(*) FROM sources GROUP BY state'));db.close();now=time.time();success=counts.get('postprocessed_unapproved',0);history.append((now,success,not errors))
        while len(history)>1 and history[1][0]<now-1800:history.popleft()
        elapsed=now-history[0][0];rate=(success-history[0][1])*3600/elapsed if elapsed>=1800 and all(x[2] for x in history) else None
        v=dict(revision=contract['revision'],observed_unix=now,original_inputs=contract['original_inputs'],blocked_original_inputs=contract['blocked_original_inputs'],reliable_caption_sources=success,failed_sealed_sources=counts.get('failed_sealed_no_retry',0),reliable_terminal_sources=len(completed),model_pool_states=states,loaded_active_main_replicas=sum(s['live'] and s['loaded'] for n in nodes for s in n['slots']),nodes=nodes,external=external,sources_per_hour_30min=rate,observation_elapsed_seconds=elapsed,errors=errors,full_corpus_complete=False,old_VLM_outputs_excluded=True,image_inputs=0,video_inputs=0,quality_approved=False,training_eligible=False)
        tmp=LOCAL/'latest.tmp';raw=json.dumps(v,ensure_ascii=False).encode();f=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(f,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,LOCAL/'latest.json');print(json.dumps({k:v[k] for k in ('observed_unix','reliable_caption_sources','failed_sealed_sources','loaded_active_main_replicas','errors')}),flush=True);time.sleep(30)

def show(watch=False,raw=False):
    while True:
        p=LOCAL/'latest.json'
        if p.exists():v=json.loads(p.read_bytes())
        else:
            code="import pathlib;print(pathlib.Path('/root/stage2_MS_LLM32_progress_v4/latest.json').read_text())";r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HOSTS['N2'],'/opt/venv_eval/bin/python','-B','-c',__import__('shlex').quote(code)],capture_output=True,text=True,timeout=15)
            if r.returncode:raise RuntimeError('Text-stage progress unavailable: '+r.stderr[-300:])
            v=json.loads(r.stdout)
        if raw:print(json.dumps(v,ensure_ascii=False))
        else:
            print('新主线：原MS＋32B纯文本轻润色 / 英文 / 短边缘合并 / rich仅舞种 / 0图像')
            print('可靠新正文%s | 失败封存%s | 已准备原输入%s | 待修原引用%s'%(v['reliable_caption_sources'],v['failed_sealed_sources'],v['original_inputs'],v['blocked_original_inputs']))
            print('主群已加载活副本%s | 30分钟可靠吞吐%s'%(v['loaded_active_main_replicas'],'%.1f条/h'%v['sources_per_hour_30min'] if v['sources_per_hour_30min'] is not None else '完整窗口尚未形成'))
            for n in v['nodes']:print(n['node']+' '+ '; '.join('%s:%s 本地%s%s'%(s['owner'],'运行' if s['loaded'] and s['live'] else '加载' if s['live'] else '排空/待准入',s['processed'],' 错误='+str(s['failure']) if s['failure'] else '') for s in n['slots']))
            for e in v['external']:print(e['node']+' 可靠新正文'+str(e['shared_caption_sources']))
            if v['errors']:print('统计缺口 '+str(v['errors']))
            print('旧VLM正文不计入；未人审，不可训练。')
        if not watch:break
        time.sleep(30)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--service',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args();run() if a.service else show(a.watch,a.json)
