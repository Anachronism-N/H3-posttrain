"""Own never-dispatched E3 cells; do not replace or retry any original call."""
import json,time,shutil,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from choreobench.composition_workflows import run_composition
from choreobench.catalog_song import export_song
from choreobench.baseline import _verify

root=Path('/opt/music_action_runs/20261113_e3_continuation_v2')
formal=Path('/opt/music_action_runs/20261113_e34_formal_v1/formal')
catalog=Path('/opt/music_action_runs/20261113_e3_moss_catalog_v1')
registration=json.loads((root/'partition.json').read_text())
provider=json.loads((root/'provider.json').read_text())
dataset=json.loads((root/'dataset.json').read_text());songs={s['song_id']:s for s in dataset['songs']}
# Reservations were recorded before starting this process. Original executor
# refuses existing output directories before dispatch, so it cannot duplicate them.
def one(cell):
 sid=cell['song_id'];method=cell['method'];target=formal/'runs'/sid/method
 owner=json.loads((target/'continuation_owner.json').read_text())
 if owner!=cell:raise ValueError('Cell ownership changed')
 out=root/'runs'/sid/method;started=time.monotonic();view=None
 deadline=started+86400
 while view is None:
  matches=list(catalog.glob('shard*/views/'+sid+'.json'))
  if len(matches)>1:raise ValueError('Duplicated song view')
  if matches:view=matches[0];break
  candidates=[p for p in catalog.glob('shard*/catalog/contract.json') if any(s['song_id']==sid for s in json.loads(p.read_text())['dataset_snapshot']['songs'])]
  if len(candidates)!=1:raise ValueError('Source catalogue differs')
  cat=candidates[0].parent
  if (cat/'results.json').exists():
   try:
    view=cat.parent/'views'/(sid+'.json');export_song(cat,sid,view)
   except (ValueError,TypeError,KeyError) as error:
    row=dict(song_id=sid,method=method,status='local_proposals_failed',full_song_succeeded=False,calls_used=0,error=str(error))
    (target/'cell_failure.json').write_text(json.dumps(row));return row
  elif time.monotonic()>deadline:
   row=dict(song_id=sid,method=method,status='source_wait_timeout',full_song_succeeded=False,calls_used=0)
   (target/'cell_failure.json').write_text(json.dumps(row));return row
  else:time.sleep(10)
 result=run_composition(view,sid,provider,out,method=method,seed=42,max_request_characters=300000)
 _verify(result)
 if any(p.name!='continuation_owner.json' for p in target.iterdir()):raise ValueError('Unexpected original dispatch in reserved cell')
 shutil.copytree(out,target,dirs_exist_ok=True)
 row=dict(song_id=sid,method=method,status='returned' if result['complete'] else 'infrastructure_failed',full_song_succeeded=result['full_song_succeeded'],result=result,run_dir=str(target),wallclock_seconds=time.monotonic()-started,routing_epoch='balanced_disjoint_continuation_v2')
 (target/'continuation_cell.json').write_text(json.dumps(row));print(json.dumps(dict(song=sid,method=method,success=result['full_song_succeeded'],calls=result['calls_used'],stop=result['stop_reason'])),flush=True);return row
with (root/'cells.jsonl').open('x') as f,ThreadPoolExecutor(max_workers=12) as pool:
 for row in pool.map(one,registration['cells']):f.write(json.dumps(row)+chr(10));f.flush()
(root/'complete.json').write_text(json.dumps(dict(completed=True,registered=len(registration['cells']),never_dispatched_cells_only=True,new_training_updates=0)))
# Gather original and continuation terminal artifacts once, keeping one slot/cell.
start=time.monotonic()
while True:
 paths=list(formal.glob('runs/*/*/results.json'));failures=list(formal.glob('runs/*/*/cell_failure.json'))
 if len(paths)+len(failures)==96:break
 if time.monotonic()-start>172800:raise TimeoutError('Original formal cells not terminal; no resampling')
 time.sleep(30)
rows=[]
for sid in songs:
 for method in ('workflow','self_refine','react','choreography'):
  d=formal/'runs'/sid/method
  if (d/'cell_failure.json').exists():rows.append(json.loads((d/'cell_failure.json').read_text()));continue
  result=_verify(json.loads((d/'results.json').read_text()));row=dict(song_id=sid,method=method,status='returned' if result['complete'] else 'infrastructure_failed',full_song_succeeded=result['full_song_succeeded'],result=result,run_dir=str(d))
  if (d/'continuation_cell.json').exists():row=json.loads((d/'continuation_cell.json').read_text())
  else:
   calls=[json.loads(p.read_text()) for p in (d/'calls').glob('*.raw.json')]
   starts=[json.loads(p.read_text())['started_unix'] for p in (d/'calls').glob('*.started.json')]
   ends=[c['completed_unix'] for c in calls]
   row['wallclock_seconds']=max(ends)-min(starts) if starts and ends else None;row['routing_epoch']='original_frozen_v1'
  rows.append(row)
if len(rows)!=96 or len({(r['song_id'],r['method']) for r in rows})!=96:raise ValueError('Final registration differs')
with (formal/'execution_results.json').open('x') as f:json.dump(dict(registered=96,rows=rows,quality_verified=False,origin='TX A800 original+disjoint balanced continuation; same source/expert/model/tools/budget, routing fix only; no resampling'),f,indent=2)
print('Registered96 terminal cells collected without duplicate generation',flush=True)
