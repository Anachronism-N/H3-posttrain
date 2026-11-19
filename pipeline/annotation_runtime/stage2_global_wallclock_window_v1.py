"""One fixed wall-clock global source window; loading and idle time are included."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1")
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N4':'28.216.19.143','N5':'28.216.19.137','N6':'28.216.19.225','N7':'28.216.18.144','N8':'28.216.18.136'}

def write(path,value):
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True)+chr(10)).encode();fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    d=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)


def collect(node,paths):
    code='''import pathlib,json,time
root=pathlib.Path(ROOTVALUE);paths=json.loads(PATHVALUE);jobs=[]
for name in paths:
 p=pathlib.Path(name);r=dict(config=name,processed=0,outcomes=[])
 if not p.exists():r['pending']=True;jobs.append(r);continue
 c=json.loads(p.read_bytes());assert pathlib.Path(c['out']).parent==root;r.update(phase=c['phase'],gpus=c['gpus']);o=pathlib.Path(c['out']);q=pathlib.Path(c['queue']);r['loaded']=(o/'load.receipt.json').exists()
 if (q/'live.json').exists():
  l=json.loads((q/'live.json').read_bytes());i=l.get('child_identity');p=pathlib.Path('/proc',str(i['pid'])) if i else None;r['worker_alive']=False
  if p and p.exists():s=(p/'stat').read_text().rsplit(') ',1)[1].split();r['worker_alive']=int(s[19])==i['start_ticks'] and s[0] not in ('Z','X','x')
 s=o/'summary.json'
 if not s.exists():
  ps=list((o/'progress').iterdir()) if (o/'progress').exists() else [];s=max(ps,key=lambda p:p.name) if ps else None
 if s:
  v=json.loads(s.read_bytes());r['processed']=v['processed'];r['outcomes']=[dict(id=x['id'],split=x['split'],status=x['status']) for x in v['all_outcomes']]
 r['failure']=(q/'failure.json').exists();jobs.append(r)
print(json.dumps(dict(node=NODEVALUE,observed_unix=time.time(),jobs=jobs)))
'''.replace('ROOTVALUE',repr(str(ROOT))).replace('PATHVALUE',repr(json.dumps(paths))).replace('NODEVALUE',repr(node))
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOSTS[node],'/opt/venv_eval/bin/python','-B','-'],input=code,text=True,capture_output=True,timeout=40)
    if r.returncode:raise RuntimeError(node+': '+r.stderr[-1000:])
    return json.loads(r.stdout)


def snapshot():
    registry=json.loads((ROOT/'runtime/ms_vlm_postprocess_jobs_registry_v1.json').read_bytes());groups={}
    for j in registry['jobs']:
        if not j['phase'].startswith('ms_fullframe_vlm_postprocess_batch_v'):continue
        node=j['node'];assert node in HOSTS;groups.setdefault(node,[]).append(j['config_path'])
    nodes=[];errors=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures={n:ex.submit(collect,n,paths) for n,paths in groups.items()}
        for node,f in futures.items():
            try:nodes.append(f.result())
            except Exception as e:errors.append(dict(node=node,error=str(e)))
    completed=set();failed=set();active=0
    for n in nodes:
        for j in n['jobs']:
            active+=bool(j.get('worker_alive') and j.get('loaded'))
            for x in j['outcomes']:
                key=(x['split'],x['id'])
                if x['status'] in ('postprocessed_unapproved','postprocessed_needs_visual_revision','insufficient_visible_evidence'):completed.add(key)
                elif x['status']=='failed_sealed_no_retry':failed.add(key)
    return dict(observed_unix=time.time(),completed_keys=sorted(map(list,completed)),failed_keys=sorted(map(list,failed)),loaded_active_replicas=active,errors=errors,nodes=nodes)


def run(out,seconds=1800):
    out=Path(out);assert out.is_absolute() and str(out).startswith('/root/') and not out.exists();assert type(seconds) is int and seconds>=1800;out.mkdir(parents=True)
    first=snapshot();write(out/'000000.json',first);deadline=time.monotonic()+seconds;seq=0;gaps=bool(first['errors'])
    while time.monotonic()<deadline:
        time.sleep(min(60,max(0,deadline-time.monotonic())));seq+=1;current=snapshot();gaps|=bool(current['errors']);write(out/('%06d.json'%seq),current)
    elapsed=current['observed_unix']-first['observed_unix'];initial=set(map(tuple,first['completed_keys']));final=set(map(tuple,current['completed_keys']));failed=set(map(tuple,current['failed_keys']))-set(map(tuple,first['failed_keys']))
    report=dict(schema='stage2_global_wall_clock_window_v1',start_unix=first['observed_unix'],end_unix=current['observed_unix'],elapsed_seconds=elapsed,new_completed_unique_sources=len(final-initial),new_failed_sealed=len(failed),sources_per_hour=len(final-initial)*3600/elapsed if not gaps else None,measurement_incomplete=gaps,includes_all_loading_idle_and_pauses=True,observed_existing_pipeline_not_new_local_dynamic_acceptance=True,initial_active_replicas=first['loaded_active_replicas'],final_active_replicas=current['loaded_active_replicas'],quality_approved=False,training_eligible=False);write(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seconds',type=int,default=1800);a=p.parse_args();run(a.out,a.seconds)
