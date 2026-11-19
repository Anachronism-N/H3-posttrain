"""Finite metadata-only cleaning patrol for explicitly authorized N1/N2/N3/N6."""
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

ROOT=Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1")
RT=ROOT/'runtime'
OUT=ROOT/'ms_vlm_live_cleaning_patrol_20261028_v3'
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225'}

def collect(node,jobs):
    assert node in HOSTS
    paths=[x['config_path'] for x in jobs]
    for p in paths:assert Path(p).parent==RT and Path(p).suffix=='.json'
    program='''import json,pathlib,time,hashlib
paths=json.loads(PATHS)
root=pathlib.Path(ROOTPATH);result=[]
for name in paths:
 p=pathlib.Path(name);r=dict(config_path=name)
 if not p.exists():r['pending_config']=True;result.append(r);continue
 cfg=json.loads(p.read_bytes());assert pathlib.Path(cfg['out']).parent==root and pathlib.Path(cfg['queue']).parent==root/'queues'
 q=pathlib.Path(cfg['queue']);out=pathlib.Path(cfg['out']);r.update(phase=cfg['phase'],source_count=cfg['targetcount'],loaded=(out/'load.receipt.json').exists(),failure=None,processed=0)
 if (q/'failure.json').exists():r['failure']=json.loads((q/'failure.json').read_bytes())['error']
 if (q/'live.json').exists():
  live=json.loads((q/'live.json').read_bytes());identity=live.get('child_identity');pid=identity['pid'] if identity else 0;proc=pathlib.Path('/proc',str(pid));r['worker_pid']=pid;r['worker_alive']=proc.exists()
  if proc.exists():
   stat=(proc/'stat').read_text().rsplit(') ',1)[1].split();r['worker_alive']=int(stat[19])==identity['start_ticks'] and stat[0] not in ('Z','X','x');r['wchan']=(proc/'wchan').read_text()
 summary=out/'summary.json';progress=[]
 if (out/'progress').exists():progress=list((out/'progress').iterdir());assert len(progress)<=10000
 latest=summary if summary.exists() else max(progress,key=lambda x:x.name) if progress else None
 if latest:
  value=json.loads(latest.read_bytes());r.update(processed=value['processed'],model_requests=value['new_model_calls'],status_counts={k:sum(x['status']==k for x in value['all_outcomes']) for k in set(x['status'] for x in value['all_outcomes'])},last_commit_unix=latest.stat().st_mtime,source_keys=[[x['split'],x['id']] for x in value['all_outcomes']])
  if value['all_outcomes']:
   commit=value['all_outcomes'][-1]['commit_ref'];blob=pathlib.Path(commit['path']).read_bytes();r['latest_commit_sha_verified']=hashlib.sha256(blob).hexdigest()==commit['sha256']
 r['complete_summary']=summary.exists();r['native_safe']=False
 if (q/'resources_safe.json').exists():
  safe=json.loads((q/'resources_safe.json').read_bytes());r['native_safe']=all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored','managed_parent_wait','unknown_jobs_not_signaled'))
 result.append(r)
print(json.dumps(dict(node=NODELABEL,observed_unix=time.time(),jobs=result)))
'''.replace('PATHS',repr(json.dumps(paths))).replace('ROOTPATH',repr(str(ROOT))).replace('NODELABEL',repr(node))
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOSTS[node],'/opt/venv_eval/bin/python','-B','-'],input=program,text=True,capture_output=True,timeout=35)
    if r.returncode:raise RuntimeError(node+' metadata collection failed: '+r.stderr[-1000:])
    return json.loads(r.stdout)

def save(value):
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n').encode();temp=OUT/'latest_summary.json.tmp';fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(temp,OUT/'latest_summary.json');fd=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)

if __name__=='__main__':
    assert not OUT.exists(),'Existing monitor output preserved; do not duplicate restart'
    OUT.mkdir();deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline and not (OUT/'STOP').exists():
        value=json.loads((RT/'ms_vlm_postprocess_jobs_registry_v1.json').read_bytes());jobs=[j for j in value['jobs'] if j.get('expected_running') is True and j.get('user_stage_authorized') is True];assert all(j['node'] in HOSTS for j in jobs)
        grouped={node:[j for j in jobs if j['node']==node] for node in HOSTS};results=[];errors=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={node:pool.submit(collect,node,js) for node,js in grouped.items() if js}
            for node,future in futures.items():
                try:results.append(future.result())
                except Exception as e:errors.append(dict(node=node,error=str(e)))
        keys=set();requests=0
        for node in results:
            for j in node['jobs']:keys.update(tuple(k) for k in j.pop('source_keys',[]));requests+=j.get('model_requests',0)
        snapshot=dict(schema='ms_vlm_authorized_cleaning_patrol_v3',monitor_pid=os.getpid(),observed_unix=time.time(),nodes=results,errors=errors,current_registered_stage_outcomes=len(keys),current_registered_model_requests=requests,not_global_corpus_count=True,quality_approved=False,training_eligible=False)
        save(snapshot);print(json.dumps(dict(event='cleaning_snapshot',outcomes=len(keys),model_requests=requests,errors=errors)),flush=True)
        time.sleep(60)
