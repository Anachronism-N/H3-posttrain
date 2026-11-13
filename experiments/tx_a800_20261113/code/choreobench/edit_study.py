"""Same-source paired edit execution over an independently frozen task registry."""
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .baseline import _verify,_sealed
from .edit_authoring import run_authoring
from .edit_agent import run_edit_agent
from .io import file_lock,file_sha,object_sha,read_json,read_json_snapshot,write_json


def run_edit_study(registry_path,author_provider_path,editor_provider_spec,out_dir,*,seed=42,max_calls=6,max_output_tokens=2048,
                   resume=False,author_runner=None,editor_runner=None):
 registry_path=Path(registry_path).resolve();author_provider_path=Path(author_provider_path).resolve();output=Path(out_dir).resolve()
 registry,sha=read_json_snapshot(registry_path);_verify(registry)
 if registry.get('schema')!='choreobench.e4_registered_slots.v1' or registry.get('expected_tasks')!=len(registry.get('slots',[])):
  raise ValueError('Edit study requires an independent registered task denominator')
 if registry.get('methods')!=['generic','choreography']:raise ValueError('Paired study requires same two declared editor policies')
 if type(resume) is not bool:raise ValueError('Resume must be bool')
 for name,value,lower,upper in [('seed',seed,0,2**63-1),('max_calls',max_calls,1,8),('max_output_tokens',max_output_tokens,1,8192)]:
  if type(value) is not int or not lower<=value<=upper:raise ValueError('Invalid paired edit '+name)
 author_spec,author_sha=read_json_snapshot(author_provider_path)
 if not isinstance(editor_provider_spec,dict) or editor_provider_spec.get('kind') not in ('chat_completions','replay'):
  raise ValueError('Explicit real/replay editor required')
 if output.exists() and not resume:raise FileExistsError(str(output))
 if resume and not (output/'contract.json').exists():raise ValueError('Edit study resume needs original contract')
 output.mkdir(parents=True,exist_ok=True)
 with file_lock(output/'.study.lock'):
  code={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}
  contract=dict(schema='choreobench.paired_edit_study_contract.v1',registry_sha256=sha,author_provider_sha256=author_sha,
   editor_provider=copy.deepcopy(editor_provider_spec),seed=seed,max_calls=max_calls,max_output_tokens=max_output_tokens,
   expected_tasks=len(registry['slots']),expected_editor_cells=2*len(registry['slots']),code_sha256=code)
  def save(path,value):
   if path.exists():
    if read_json(path)!=value:raise ValueError('Immutable paired edit evidence changed')
   else:write_json(path,value)
  save(output/'contract.json',contract)
  author=(author_runner or run_authoring)(registry_path,author_provider_path,output/'authoring',seed=seed,max_output_tokens=max_output_tokens,resume=resume)
  slots={s['slot_id']:s for s in registry['slots']}
  if len(slots)!=len(registry['slots']) or {r['slot_id'] for r in author['rows']}!=set(slots):raise ValueError('Author evidence omits or adds registered tasks')
  previous=_verify(read_json(output/'results.json')) if (output/'results.json').exists() else None
  if previous is not None:
   if previous['contract_sha256']!=object_sha(contract) or previous['author_result_sha256']!=object_sha(author):raise ValueError('Original author/contract binding changed')
   for name,digest in previous['artifacts'].items():
    if not (output/name).is_file() or file_sha(output/name)!=digest:raise ValueError('Completed paired edit artifact missing/changed; no replacement')
  previous_cells={(r['slot_id'],r['method']):r for r in previous['cells']} if previous else {}
  artifacts=copy.deepcopy(previous['artifacts']) if previous is not None else {};fatal=threading.Event()
  def run_cell(pair):
   row,method=pair;slot=slots[row['slot_id']]
   cell=dict(slot_id=slot['slot_id'],song_id=slot['song_id'],registered_category=slot['registered_category'],method=method,
    source_method=slot['source_method'],status='author_'+row['status'],result=None)
   local_artifacts={}
   if row['status']!='model_declared_valid':return cell,local_artifacts
   if fatal.is_set():cell['status']='not_attempted_after_execution_failure';return cell,local_artifacts
   task_path=output/'authoring'/row['task_path'];source_path=Path(slot['source_plan_path']);key=(slot['slot_id'],method)
   if previous is not None and previous_cells.get(key,{}).get('status') in ('execution_failed','execution_error','not_attempted_after_execution_failure'):
    prior=previous_cells[key];info=prior.get('result') or {}
    if prior['status'] in ('execution_error','not_attempted_after_execution_failure') or info.get('unknown_outcomes_count',0) or info.get('unknown_usage_count',0):fatal.set()
    return prior,local_artifacts
   run=output/'runs'/object_sha(slot['slot_id'])/method
   try:
    result=(editor_runner or run_edit_agent)(source_path,task_path,editor_provider_spec,run,policy=method,seed=seed,
     max_calls=max_calls,max_output_tokens=max_output_tokens,resume=resume)
    cell.update(status='returned' if result['complete'] else 'execution_failed',result=result,run_path=str(run))
    result_path=run/'results.json'
    if result_path.exists():local_artifacts[str(result_path.relative_to(output))]=file_sha(result_path)
    if not result['complete'] and (result.get('unknown_outcomes_count',0) or result.get('unknown_usage_count',0)):fatal.set()
   except (ValueError,TypeError,KeyError,OSError) as error:
    cell.update(status='execution_error',error=dict(type=type(error).__name__,message=str(error)));fatal.set()
   return cell,local_artifacts
  cells=[]
  with ThreadPoolExecutor(max_workers=16) as pool:
   for cell,local_artifacts in pool.map(run_cell,[(r,m) for r in author['rows'] for m in registry['methods']]):
    cells.append(cell);artifacts.update(local_artifacts)
  if len(cells)!=contract['expected_editor_cells']:raise ValueError('Paired edit output denominator changed')
  if file_sha(registry_path)!=sha or file_sha(author_provider_path)!=author_sha:raise ValueError('Registered edit inputs changed')
  if any(file_sha(Path(__file__).parent/p)!=digest for p,digest in code.items()):raise ValueError('Paired edit execution code changed')
  results=[r['result'] for r in cells if r['result'] is not None]
  report=_sealed(dict(schema='choreobench.paired_edit_study_result.v1',contract_sha256=object_sha(contract),cells=cells,
   author_result_sha256=object_sha(author),expected_registered_tasks=contract['expected_tasks'],expected_editor_cells=contract['expected_editor_cells'],artifacts=artifacts,
   actual_editor_calls=sum(r['calls_used'] for r in results),actual_author_calls=author['calls_used'],
   editor_status_counts={s:sum(c['status']==s for c in cells) for s in sorted({c['status'] for c in cells})},
   model_declared_valid_tasks=author['model_declared_valid'],origin='model_authored_experiment',human_reviews=0,
   semantic_success=None,formal_E4_completed=False,
   note='Every independently registered task produces both policy slots, including source/author unknown/failed outcomes. No-op, abstention and submissions are raw policy results, not semantic successes. Both editors read the same frozen original source, never another method final state. Private author evaluation is separate; public instruction includes protected facts. All cumulative edit proposals compared to original by shared tools.'))
  if previous is not None and report!=previous:raise ValueError('Completed paired edit study reconstructed differently')
  save(output/'results.json',report)
  return report


def main(argv=None):
 import argparse
 from .io import json_text
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('registry','author-provider','editor-provider','out-dir'):parser.add_argument('--'+name,required=True,type=Path)
 parser.add_argument('--seed',type=int,default=42)
 parser.add_argument('--max-calls',type=int,default=6)
 parser.add_argument('--max-output-tokens',type=int,default=2048)
 parser.add_argument('--resume',action='store_true')
 args=parser.parse_args(argv)
 result=run_edit_study(args.registry,args.author_provider,read_json(args.editor_provider),args.out_dir,
  seed=args.seed,max_calls=args.max_calls,max_output_tokens=args.max_output_tokens,resume=args.resume)
 print(json_text({k:result[k] for k in ('expected_registered_tasks','expected_editor_cells','actual_author_calls','actual_editor_calls','editor_status_counts','formal_E4_completed')}))


if __name__=='__main__':main()
