"""Execute registered24 whole songs across four frozen adaptations, no best-of."""
import argparse,json,time,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from choreobench.composition_workflows import run_composition


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--provider',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--streaming-catalog',action='store_true');a=p.parse_args()
 r=a.root;r.mkdir(exist_ok=False);dataset=json.loads(a.dataset.read_text());provider=json.loads(a.provider.read_text());methods=('workflow','self_refine','react','choreography');views={p.stem:p for p in a.catalog.glob('shard*/views/*.json')};songs=dataset['songs']
 (r/'registration.json').write_text(json.dumps(dict(songs=[s['song_id'] for s in songs],methods=methods,cells=96,selected_expert_catalog=str(a.catalog),same_K=2,same_budget=True,formal=True,new_training_updates=0),indent=2))
 def one(cell):
  song,method=cell;sid=song['song_id'];out=r/'runs'/sid/method
  if a.streaming_catalog:
   deadline=time.monotonic()+86400
   while sid not in views:
    candidates=list(a.catalog.glob('shard*/views/'+sid+'.json'))
    if len(candidates)>1:raise ValueError('Duplicate sealed song view')
    if candidates:views[sid]=candidates[0];break
    failures=list(a.catalog.glob('shard*/views/'+sid+'.failed.json'))
    if failures or all((a.catalog/('shard'+str(i))/'complete.json').exists() for i in range(4)):
     return dict(song_id=sid,method=method,status='local_proposals_failed',full_song_succeeded=False,calls_used=0)
    if time.monotonic()>deadline:return dict(song_id=sid,method=method,status='source_wait_timeout',full_song_succeeded=False,calls_used=0)
    time.sleep(10)
  if sid not in views:return dict(song_id=sid,method=method,status='local_proposals_failed',full_song_succeeded=False,calls_used=0)
  started=time.monotonic()
  result=run_composition(views[sid],sid,provider,out,method=method,seed=42,max_request_characters=300000)
  row=dict(song_id=sid,method=method,status='returned' if result['complete'] else 'infrastructure_failed',full_song_succeeded=result['full_song_succeeded'],result=result,run_dir=str(out),wallclock_seconds=time.monotonic()-started)
  print(json.dumps(dict(song=sid,method=method,complete=result['complete'],success=result['full_song_succeeded'],calls=result['calls_used'],error=result['error'])),flush=True);return row
 rows=[]
 with (r/'cells.jsonl').open('x') as stream,ThreadPoolExecutor(max_workers=16) as pool:
  for row in pool.map(one,[(s,m) for s in songs for m in methods]):rows.append(row);stream.write(json.dumps(row)+chr(10));stream.flush()
 if len(rows)!=96:raise ValueError('Formal cell inventory differs')
 (r/'execution_results.json').write_text(json.dumps(dict(registered=96,rows=rows,quality_verified=False,origin='real model formal generation; failures preserved'),indent=2)+chr(10))


if __name__=='__main__':main()
