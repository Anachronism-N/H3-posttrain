"""Run original48 single-turn edit slots as their fixed source plans become terminal."""
import argparse,json,time,shutil,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from choreobench.baseline import _sealed,_verify
from choreobench.io import file_sha,object_sha,write_json,read_json
from choreobench.edit_study import run_edit_study


def main():
 p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--formal',type=Path,required=True);p.add_argument('--provider',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;root.mkdir(exist_ok=False)
 registry=_verify(read_json(a.registry));original=list(registry['slots']);assert len(original)==48
 rebound={k:v for k,v in registry.items() if k!='record_sha256'};rebound['slots']=[]
 for slot in original:
  changed=dict(slot);changed['source_plan_path']=str(a.formal/'runs'/slot['song_id']/slot['source_method']/'plan.json');rebound['slots'].append(changed)
 rebound['path_rebinding_only']=dict(original_registry_path=str(a.registry),sha256=file_sha(a.registry),reason='Same original source-method final artifacts read node-locally; no different cases, methods or categories selected')
 rebound=_sealed(rebound);write_json(root/'registry.json',rebound);provider=read_json(a.provider)
 state={s['slot_id']:'waiting_original_source' for s in rebound['slots']};waiting={s['slot_id']:s for s in rebound['slots']};futures={};completed={};fatal=threading.Event();start=time.monotonic()
 def run(slot):
  one={k:v for k,v in rebound.items() if k!='record_sha256'};one['slots']=[slot];one['expected_tasks']=1;one['parent_registry_sha256']=object_sha(rebound);path=root/'registries'/(slot['slot_id']+'.json');write_json(path,_sealed(one))
  result=run_edit_study(path,a.provider,provider,root/'slots'/slot['slot_id'],seed=42,max_calls=6,max_output_tokens=2048)
  for cell in result['cells']:
   info=cell.get('result') or {}
   if cell['status'] in ('execution_error',) or info.get('unknown_outcomes_count',0) or info.get('unknown_usage_count',0):fatal.set()
  print(json.dumps(dict(slot=slot['slot_id'],category=slot['registered_category'],source_method=slot['source_method'],editor_status=result['editor_status_counts'],editor_calls=result['actual_editor_calls'])),flush=True)
  return result
 with ThreadPoolExecutor(max_workers=8) as pool:
  while waiting or futures:
   for sid,future in list(futures.items()):
    if not future.done():continue
    try:completed[sid]=future.result();state[sid]='terminal'
    except Exception as error:completed[sid]=dict(error=str(error));state[sid]='infrastructure_error';fatal.set()
    del futures[sid]
   for sid,slot in list(waiting.items()):
    path=Path(slot['source_plan_path']);result=path.parent/'results.json';failure=path.parent/'cell_failure.json'
    if not result.exists() and not failure.exists():continue
    if fatal.is_set():completed[sid]=dict(error='not_attempted_after_infrastructure_failure');state[sid]='not_attempted';del waiting[sid];continue
    if result.exists():
     outcome=_verify(read_json(result))
     if not outcome['full_song_succeeded']:
      completed[sid]=dict(error='original_registered_source_generation_failed',source_outcome=outcome['stop_reason']);state[sid]='source_failed';del waiting[sid];continue
     if not path.exists() or file_sha(path)!=outcome['artifacts']['plan.json']:raise ValueError('Terminal original source plan differs from generation result')
     futures[sid]=pool.submit(run,slot);state[sid]='authoring_and_editing';del waiting[sid]
    else:completed[sid]=dict(error='original_source_proposals_failed');state[sid]='source_failed';del waiting[sid]
   (root/'progress.json').write_text(json.dumps(dict(registered=48,status_counts={s:sum(v==s for v in state.values()) for s in set(state.values())},states=state),indent=2))
   if time.monotonic()-start>172800:raise TimeoutError('Original registered E3 sources not terminal; no substitute tasks')
   if waiting or futures:time.sleep(10)
 # Merge only actual sealed per-slot attempts. No new authoring or editor requests.
 authoring=root/'authoring';authoring.mkdir();(authoring/'author').mkdir();(authoring/'tasks').mkdir();author_rows=[];cells=[];artifact_sources=[];author_calls=0;editor_calls=0
 for slot in rebound['slots']:
  sid=slot['slot_id'];result=completed[sid]
  if 'cells' not in result:
   status=state[sid];author_rows.append(dict(slot_id=sid,song_id=slot['song_id'],category=slot['registered_category'],status=status,task_path=None,error=result.get('error')))
   for method in ('generic','choreography'):cells.append(dict(slot_id=sid,song_id=slot['song_id'],registered_category=slot['registered_category'],method=method,source_method=slot['source_method'],status='author_'+status,result=None))
   continue
  base=root/'slots'/sid;author=_verify(read_json(base/'authoring/results.json'));author_calls+=author['calls_used'];editor_calls+=result['actual_editor_calls'];cells.extend(result['cells']);author_rows.extend(author['rows'])
  for folder in ('author','tasks'):
   for file in (base/'authoring'/folder).glob('*.json'):
    target=authoring/folder/file.name
    if target.exists():raise ValueError('Duplicated author task artifact')
    shutil.copyfile(file,target)
  artifact_sources.append(dict(slot_id=sid,study_result=str(base/'results.json'),study_sha256=file_sha(base/'results.json'),author_result=str(base/'authoring/results.json'),author_sha256=file_sha(base/'authoring/results.json')))
 if len(cells)!=96 or len(author_rows)!=48 or len({(x['slot_id'],x['method']) for x in cells})!=96:raise ValueError('Original48-slot denominator changed')
 write_json(authoring/'results.json',_sealed(dict(rows=author_rows,calls_used=author_calls,registered_tasks=48,per_slot_source_snapshots=artifact_sources,aggregated_actual_artifacts_only=True)))
 write_json(root/'results.json',_sealed(dict(cells=cells,actual_editor_calls=editor_calls,actual_author_calls=author_calls,expected_registered_tasks=48,expected_editor_cells=96,registry_sha256=object_sha(rebound),per_slot_source_snapshots=artifact_sources,editor_status_counts={s:sum(x['status']==s for x in cells) for s in {x['status'] for x in cells}},generation_called_once_per_slot=True,no_source_replacement=True,human_reviews=0,formal_E4_quality_verified=False)))
 print(json.dumps(dict(completed=True,registered_tasks=48,editor_cells=96,actual_author_calls=author_calls,actual_editor_calls=editor_calls)),flush=True)


if __name__=='__main__':main()
