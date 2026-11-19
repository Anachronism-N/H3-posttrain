"""Read-only progress/commit monitor for finite continuous MotionScript workers."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

HOSTS = {"N2":"28.216.17.70", "N3":"28.216.19.213", "N6":"28.216.19.225"}
REMOTE = r'''
import pathlib,json,hashlib,sys,time
jobs=json.loads(sys.argv[1]);out=[]
def read(p):
 p=pathlib.Path(p)
 with p.open('rb',buffering=0) as f:
  if p.stat().st_size>8*1024*1024:raise ValueError('Oversize metadata')
  data=f.read(8*1024*1024+1)
 if b'\x00' in data:raise ValueError('NUL metadata')
 return json.loads(data,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON'))),data
for job in jobs:
 result={'job':job,'batches':[],'warnings':[]}
 try:
  cfg,_=read(job['config_path']);q=pathlib.Path(cfg['queue']);ticket,_=read(cfg['ticket']['path']);rows=ticket['sources']
  if cfg['phase']!=job['phase'] or len(rows)!=cfg['targetcount'] or [r['id'] for r in rows]!=cfg['selected_source_ids']:raise ValueError('Config/ticket source scope changed')
  live,_=read(q/'live.json');x=live['child_identity'];proc=pathlib.Path('/proc')/str(x['pid']);alive=False
  if proc.exists():
   st=(proc/'stat').read_text().split(') ',1)[1].split();args=(proc/'cmdline').read_bytes().rstrip(bytes([0])).decode().split(chr(0))
   alive=int(st[19])==x['start_ticks'] and int(st[1])==x['ppid'] and args==cfg['argv'] and st[0] not in ['Z','X']
   result['process']={'pid':x['pid'],'alive':alive,'start_ticks':int(st[19]),'state':st[0],'wchan':(proc/'wchan').read_text().strip()}
  result['worker_alive']=alive
  if (q/'failure.json').exists():result['failure']=read(q/'failure.json')[0]
  result['resources_safe']=False
  if (q/'resources_safe.json').exists():
   safe,_=read(q/'resources_safe.json');result['resources_safe']=all(safe.get(k) is True for k in ['all_holders_safe','locks_released','leases_restored'])
  p=pathlib.Path(cfg['out'])/ticket['edition']/'progress.json'
  if not p.exists():result['status']='starting';out.append(result);continue
  progress,_=read(p);result['progress']=progress;result['progress_age_s']=time.time()-p.stat().st_mtime
  if progress['source_count']!=len(rows) or progress['vlm_called'] is not False or progress['quality_approved'] is not False:raise ValueError('Progress scope/flags mismatch')
  n=progress['completed_batch_count'];last=job.get('last_verified',0);batchsize=ticket['batch_sources']
  for i in range(last+1,min(n,last+10)+1):
   p=pathlib.Path(cfg['out'])/ticket['edition']/'batches'/(ticket['ticket_id']+'-b%06d'%i)/'commit.json';commit,raw=read(p)
   start=(i-1)*batchsize;expected=rows[start:start+batchsize];receipts=commit['receipts']
   if commit['parent_ticket_id']!=ticket['ticket_id'] or commit['batch_index']!=i or commit['schema']!='ms_unified_novlm_batch_commit_v3' or commit['prefetch_drained'] is not True or commit['vlm_called'] is not False:raise ValueError('Batch identity/flags mismatch')
   if [r['source_row'] for r in receipts]!=expected or commit['selected_range']!={'start':start,'end':start+len(expected)}:raise ValueError('Actual batch sources differ from assignment')
   records=[]
   for r in receipts:
    if r['vlm_called'] is not False or any(r.get(k) is not False for k in ['semantic_approval','quality_approved','training_eligible']) or r.get('needs_visual_revision') is not True:raise ValueError('Unreviewed source flags differ')
    records.append({'id':r['source_row']['id'],'split':r['source_row']['split'],'status':r['status'],'draft_status':r.get('draft_status')})
   result['batches'].append({'index':i,'commit':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'records':records})
  result['status']=progress['status']
 except Exception as e:result['warnings'].append(type(e).__name__+': '+str(e));result['measurement_incomplete']=True
 out.append(result)
print(json.dumps(out))
'''

class State:
    def __init__(self, clock=time.monotonic):
        self.clock=clock; self.started=clock(); self.cursors={}; self.baseline=set(); self.seen=set(); self.events=deque(); self.counts=Counter(); self.gaps=deque(); self.last_progress={}
    def sample(self, observations):
        now=self.clock(); reports=[];warnings=[]
        for obs in observations:
            job=obs['job'];phase=job['phase'];first=phase not in self.cursors
            if obs.get('measurement_incomplete') or obs.get('warnings'):
                warnings.extend([phase+': '+x for x in obs.get('warnings',[])]); self.gaps.append(now)
            cursor=self.cursors.get(phase,0)
            for batch in obs.get('batches',[]):
                if batch['index']!=cursor+1:
                    warnings.append(phase+': batch index gap');self.gaps.append(now);break
                for row in batch['records']:
                    key=(row['split'],row['id'])
                    if first:self.baseline.add(key)
                    elif key not in self.seen and key not in self.baseline:
                        self.seen.add(key);self.events.append((now,key,row['status'],row.get('draft_status')));self.counts[row['status']]+=1
                cursor=batch['index']
            if not obs.get('measurement_incomplete') or obs.get('batches'):self.cursors[phase]=cursor
            progress=obs.get('progress',{});count=progress.get('processed')
            previous=self.last_progress.get(phase)
            if previous is None or previous[0]!=count:self.last_progress[phase]=(count,now)
            idle=now-self.last_progress.get(phase,(None,now))[1]
            expected=obs.get('worker_alive') is True and obs.get('status') not in ['completed','drained_not_complete']
            alarms=[]
            if obs.get('failure'):alarms.append('worker_failure')
            if expected and obs.get('progress_age_s',0)>180:alarms.append('progress_stale_over_180s')
            if expected and idle>=600:alarms.append('no_new_batch_over_600s')
            reports.append({'node':job['node'],'gpu':job['gpu'],'phase':phase,'status':obs.get('status','unknown'),'process':obs.get('process'),'processed':count,'source_count':progress.get('source_count'),'counts':progress.get('counts'),'model_init_count':progress.get('model_init_count'),'latest_verified_batch':cursor,'resource_safe':obs.get('resources_safe',False),'progress_age_s':obs.get('progress_age_s'),'alarms':alarms,'warnings':obs.get('warnings',[])})
        while self.events and self.events[0][0]<=now-900:self.events.popleft()
        while self.gaps and self.gaps[0]<=now-900:self.gaps.popleft()
        duration=min(900,max(0,now-self.started));successful=sum(s=='extracted_rule_draft_unapproved' for _,_,s,_ in self.events)
        return {'schema':'ms_continuous_patrol_v1','observed_unix':time.time(),'monitor_pid':os.getpid(),'scope':'observed_committed_unique_IDs_since_start; no archive-content or semantic approval','baseline_sources':len(self.baseline),'new_unique_sources':len(self.seen),'source_status_counts':dict(self.counts),'jobs':reports,'warnings':warnings,'rolling_15min':{'seconds':duration,'full_window':duration==900,'success_sources':successful,'measurement_incomplete':bool(self.gaps),'sources_per_hour':successful*3600/duration if duration and not self.gaps else None,'observed_lower_bound_per_hour':successful*3600/duration if duration else None},'quality_approved':False,'production_complete_claimed':False}

def collect(registry, state):
    groups={}
    for job in registry['jobs']:
        if HOSTS.get(job['node'])!=job['host']:raise ValueError('Unapproved monitor host')
        j=dict(job,last_verified=state.cursors.get(job['phase'],0));groups.setdefault(job['host'],[]).append(j)
    def host_sample(item):
        host,jobs=item
        try:
            payload='python3 -B -c '+shlex.quote(REMOTE)+' '+shlex.quote(json.dumps(jobs))
            r=subprocess.run(['ssh','-n','-o','BatchMode=yes','-o','ConnectTimeout=10',host,payload],capture_output=True,text=True,timeout=25)
            if r.returncode:raise RuntimeError(r.stderr[:500] or 'SSH exit '+str(r.returncode))
            return json.loads(r.stdout)
        except Exception as e:return [{'job':j,'measurement_incomplete':True,'warnings':[type(e).__name__+': '+str(e)]} for j in jobs]
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        return [v for group in pool.map(host_sample,groups.items()) for v in group]

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--out',required=True);p.add_argument('--interval',type=float,default=60);p.add_argument('--once',action='store_true');a=p.parse_args()
    if a.interval<=0:raise ValueError('Positive interval required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);lock=Path(tempfile.gettempdir())/('ms-patrol-'+__import__('hashlib').sha256(str(out.resolve()).encode()).hexdigest()+'.lock')
    with lock.open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);state=State()
        while not(out/'STOP').exists():
            print(json.dumps({'phase':'sampling','time':time.time(),'pid':os.getpid()}),flush=True)
            registry=json.loads(Path(a.registry).read_bytes());summary=state.sample(collect(registry,state));raw=(json.dumps(summary,allow_nan=False)+'\n').encode()
            tmp=out/'latest_summary.tmp';tmp.write_bytes(raw);os.replace(tmp,out/'latest_summary.json')
            with (out/'monitor.jsonl').open('ab') as f:f.write(raw)
            print(json.dumps({'phase':'sampled','committed_new':summary['new_unique_sources'],'warnings':len(summary['warnings']),'jobs':[{'phase':j['phase'],'processed':j['processed'],'alarms':j['alarms']} for j in summary['jobs']]}),flush=True)
            if a.once:return
            until=time.monotonic()+a.interval
            while time.monotonic()<until and not(out/'STOP').exists():time.sleep(min(1,max(0,until-time.monotonic())))

if __name__=='__main__':main()
