"""Synthetic same-source edit-study execution, not semantic quality."""
from pathlib import Path
import copy,json
import pytest
from choreobench.edit_study import run_edit_study
from choreobench.edit_authoring import run_authoring
from choreobench.io import read_json,write_json,object_sha
from choreobench.baseline import _sealed
from test_choreobench_edit_authoring import author_material


def material(tmp_path,failed=False):
 registry,provider,_,factory,ac=author_material(tmp_path);ec=[]
 original=read_json(registry);original['methods']=['generic','choreography']
 original.pop('record_sha256');write_json(registry,_sealed(original),overwrite=True)
 def author(*a,**k):return run_authoring(*a,**k,provider_factory=factory)
 def editor(plan,task,spec,out,**kwargs):
  assert read_json(task)['source_plan_sha256']==object_sha(read_json(plan))
  ec.append(dict(source=str(plan),task=str(task),kwargs=copy.deepcopy(kwargs)))
  result=_sealed(dict(complete=not failed,calls_used=1,source_plan_sha256=object_sha(read_json(plan)),policy=kwargs['policy'],
   final_plan_sha256=object_sha(read_json(plan)),submitted=False,unchanged_original=True,stop_reason='execution_failed' if failed else 'abstained',quality_verified=False))
  out=Path(out);out.mkdir(parents=True,exist_ok=True)
  if (out/'results.json').exists():assert read_json(out/'results.json')==result
  else:write_json(out/'results.json',result)
  return result
 return registry,provider,tmp_path/'study',author,editor,ac,ec


def test_two_policy_same_original_source_and_full_missing_denominator(tmp_path):
 registry,provider,out,author,editor,ac,ec=material(tmp_path)
 r=run_edit_study(registry,provider,dict(kind='replay'),out,author_runner=author,editor_runner=editor)
 assert r['expected_registered_tasks']==2 and r['expected_editor_cells']==4
 assert r['editor_status_counts']=={'author_missing_source':2,'returned':2}
 assert r['actual_editor_calls']==2 and r['actual_author_calls']==1
 assert ec[0]['source']==ec[1]['source'] and ec[0]['task']==ec[1]['task']
 assert r['semantic_success'] is None and not r['formal_E4_completed']
 assert run_edit_study(registry,provider,dict(kind='replay'),out,resume=True,author_runner=author,editor_runner=editor)==r
 assert len(ac)==1 and all(c['kwargs']['resume'] for c in ec[2:])


def test_failed_execution_other_policy_not_attempted_no_replacement(tmp_path):
 registry,provider,out,author,editor,ac,ec=material(tmp_path,failed=True)
 r=run_edit_study(registry,provider,dict(kind='replay'),out,author_runner=author,editor_runner=editor)
 assert len(ec)==1 and r['actual_editor_calls']==1
 assert r['editor_status_counts']=={'author_missing_source':2,'execution_failed':1,'not_attempted_after_execution_failure':1}
 assert run_edit_study(registry,provider,dict(kind='replay'),out,resume=True,author_runner=author,editor_runner=lambda *a,**k:pytest.fail('No replacement'))==r
 assert len(ac)==1


def test_completed_missing_editor_raw_result_rejected_before_call(tmp_path):
 registry,provider,out,author,editor,ac,ec=material(tmp_path)
 run_edit_study(registry,provider,dict(kind='replay'),out,author_runner=author,editor_runner=editor)
 next((out/'runs').rglob('results.json')).unlink()
 with pytest.raises(ValueError,match='artifact missing/changed'):
  run_edit_study(registry,provider,dict(kind='replay'),out,resume=True,author_runner=author,editor_runner=lambda *a,**k:pytest.fail('No replacement'))
 assert len(ac)==1 and len(ec)==2


def test_edit_budget_changed_cannot_resume_old_study(tmp_path):
 registry,provider,out,author,editor,ac,ec=material(tmp_path)
 run_edit_study(registry,provider,dict(kind='replay'),out,author_runner=author,editor_runner=editor)
 with pytest.raises(ValueError,match='Immutable'):
  run_edit_study(registry,provider,dict(kind='replay'),out,max_calls=7,resume=True,author_runner=author,editor_runner=editor)
 assert len(ec)==2


def test_study_runs_actual_shared_edit_executor_not_only_test_seam(tmp_path,monkeypatch):
 import json
 from types import SimpleNamespace
 from choreobench import edit_agent
 registry,provider,_,factory,ac=author_material(tmp_path)
 value=read_json(registry);value['methods']=['generic','choreography'];value.pop('record_sha256');write_json(registry,_sealed(value),overwrite=True)
 def author(*a,**k):return run_authoring(*a,**k,provider_factory=factory)
 calls=[];identity=dict(kind='replay',synthetic=True,fixture='actual-shared-edit-tool')
 def generate(request):
  calls.append(copy.deepcopy(request));data=json.loads(request['messages'][-1]['content'])
  assert 'semantic_goals' not in data['task'] and 'goals' not in data['task']
  event=data['task']['allowed_event_ids'][0]
  answer=dict(tool='submit_patch',operations=[dict(event_id=event,field='action',value='Step left while both arms sweep low.')],reason='Synthetic shared-executor edit')
  return dict(text=json.dumps(answer),raw={},identity=identity,synthetic=True,stop_reason='stop',costs=dict(input_tokens=10,output_tokens=20,latency_seconds=0))
 monkeypatch.setattr(edit_agent,'create_provider',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
 out=tmp_path/'study'
 result=run_edit_study(registry,provider,dict(kind='replay'),out,author_runner=author)
 assert result['actual_editor_calls']==2 and len(calls)==2
 submitted=[r for r in result['cells'] if r['status']=='returned']
 assert len(submitted)==2 and all(r['result']['submitted'] for r in submitted)
 assert submitted[0]['result']['source_plan_sha256']==submitted[1]['result']['source_plan_sha256']
 original=read_json(tmp_path/'original/plan.json');assert original['events'][0]['action']=='Step left while both wrists circle.'
 for row in submitted:
  plan=read_json(Path(row['run_path'])/'plan.json')
  assert plan['events'][0]['action']=='Step left while both arms sweep low.'
  assert all(plan['events'][0][k]==original['events'][0][k] for k in original['events'][0] if k!='action')
 assert run_edit_study(registry,provider,dict(kind='replay'),out,author_runner=author,resume=True)==result
 assert len(calls)==2 and len(ac)==1


@pytest.mark.parametrize('verdict',['supported','unsupported','unknown','invalid_quote'])
def test_independent_task_audit_gates_both_editors_without_replacing_slots(tmp_path,monkeypatch,verdict):
 import json
 from types import SimpleNamespace
 from choreobench import providers
 registry,provider,out,author,editor,ac,ec=material(tmp_path)
 audit_provider=tmp_path/'audit-provider.json';write_json(audit_provider,dict(kind='replay',path='independent-fixture'))
 calls=[];identity=dict(kind='replay',synthetic=True,model='independent-task-auditor')
 def generate(request):
  calls.append(copy.deepcopy(request));data=json.loads(request['messages'][-1]['content'])
  assert 'semantic_goals' not in data and 'author' not in data
  target=next(e for e in data['source_events'] if e['event_id']==data['target_event_id'])
  evidence=dict(status='unknown' if verdict=='unknown' else 'unsupported' if verdict=='unsupported' else 'supported',
   reason='Synthetic pipeline boundary test',instruction_quote='' if verdict=='unknown' else data['instruction'],
   source_quotes=[] if verdict=='unknown' else [dict(event_id=target['event_id'],quote=target['action'])])
  answer={k:copy.deepcopy(evidence) for k in ('category_fit','satisfiable','boundary_compatible')}
  if verdict=='invalid_quote':answer['boundary_compatible']['source_quotes'][0]['quote']='Never supplied source movement'
  return dict(text=json.dumps(answer),raw={},identity=identity,synthetic=True,stop_reason='stop',costs=dict(input_tokens=10,output_tokens=20))
 monkeypatch.setattr(providers,'create_provider',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
 r=run_edit_study(registry,provider,dict(kind='replay'),out,author_runner=author,editor_runner=editor,audit_provider_path=audit_provider)
 expected='returned' if verdict=='supported' else 'task_audit_failed' if verdict=='invalid_quote' else 'task_audit_'+verdict
 assert r['editor_status_counts']=={'author_missing_source':2,expected:2}
 assert r['expected_registered_tasks']==2 and r['expected_editor_cells']==4
 assert r['actual_task_audit_calls']==1 and len(calls)==1 and len(ac)==1
 assert r['tasks_independently_model_checked'] and not r['task_semantics_authenticated']
 assert r['semantic_success'] is None and not r['formal_E4_completed']
 assert len(ec)==(2 if verdict=='supported' else 0)
 assert run_edit_study(registry,provider,dict(kind='replay'),out,resume=True,author_runner=author,editor_runner=editor,audit_provider_path=audit_provider)==r
 assert len(calls)==1 and len(ac)==1


def test_audit_provider_identity_change_rejected_before_edit_dispatch(tmp_path):
 registry,provider,out,author,editor,ac,ec=material(tmp_path)
 audit_provider=tmp_path/'audit-provider.json';write_json(audit_provider,dict(kind='replay',path='original-auditor'))
 def audit(registry_path,author_results_path,provider_path,out_dir,**kwargs):
  rows=read_json(author_results_path)['rows']
  write_json(provider_path,dict(kind='replay',path='changed-auditor'),overwrite=True)
  return _sealed(dict(rows=[dict(slot_id=r['slot_id'],status='unknown') for r in rows],calls_used=1))
 with pytest.raises(ValueError,match='Task audit provider changed'):
  run_edit_study(registry,provider,dict(kind='replay'),out,author_runner=author,editor_runner=editor,audit_provider_path=audit_provider,audit_runner=audit)
 assert not ec


@pytest.mark.parametrize('status,expected',[('pass',True),('fail',False),('unknown',None)])
def test_final_edit_opinion_preserves_permissions_and_unknown(status,expected):
 import json
 from choreobench.edit_study import edit_outcome_request,bind_edit_outcome
 from choreobench.edit_authoring import bind_authoring
 from test_choreobench_edit_authoring import material as task_material
 slot,before,answer=task_material();task=bind_authoring(slot,before,answer,model_id='author')['task']
 after=copy.deepcopy(before);after['events'][0]['action']='Step left while both arms sweep low.'
 request=edit_outcome_request(task,before,after);payload=json.loads(request['messages'][-1]['content'])
 assert set(payload)=={'instruction','target_event_id','source_events','final_events','sample_rate','unchanged_original'}
 assert 'semantic_goals' not in payload and 'source_method' not in payload and 'author' not in payload
 event=before['events'][0]['event_id']
 verdict=dict(status=status,reason='Synthetic final-state evidence test',instruction_quote='' if status=='unknown' else answer['instruction'],
  source_quotes=[] if status=='unknown' else [dict(event_id=event,quote='Step left')],
  final_quotes=[] if status=='unknown' else [dict(event_id=event,quote='Step left')])
 response={k:copy.deepcopy(verdict) for k in ['target','protection','boundary']}
 result=bind_edit_outcome(task,before,after,response,model_id='independent-fixture')
 assert result['model_joint_success'] is expected and result['mechanical_permissions_valid']
 assert not result['semantic_truth_verified'] and not result['music_seen'] and result['human_reviews']==0
 if status=='unknown':
  response['protection'].update(status='fail',instruction_quote=answer['instruction'],source_quotes=[dict(event_id=event,quote='Step left')],final_quotes=[dict(event_id=event,quote='Step left')])
  assert bind_edit_outcome(task,before,after,response,model_id='fixture')['model_joint_success'] is False


@pytest.mark.parametrize('change',['event_id','time','evidence','metadata','other_action','extra_event'])
def test_final_outcome_rejects_unauthorized_changes_before_judging(change):
 from choreobench.edit_study import edit_outcome_request
 from choreobench.edit_authoring import bind_authoring
 from test_choreobench_edit_authoring import material as task_material
 slot,before,answer=task_material()
 if change=='other_action':
  before['events'][0]['end']//=2
  event=copy.deepcopy(before['events'][0]);event.update(event_id='untouched-neighbor',start=event['end'],end=before['num_samples'])
  before['events'].append(event)
 task=bind_authoring(slot,before,answer,model_id='author')['task'];after=copy.deepcopy(before)
 if change=='event_id':after['events'][0]['event_id']='changed-id'
 elif change=='time':after['events'][0]['start']+=1
 elif change=='evidence':after['events'][0]['evidence']=['unrequested-evidence']
 elif change=='metadata':after['provenance']={'invented':'revision'}
 elif change=='other_action':after['events'][1]['action']='Unrequested other action'
 else:
  event=copy.deepcopy(after['events'][0]);event['event_id']='new-id';after['events'].append(event)
 with pytest.raises(ValueError):edit_outcome_request(task,before,after)


@pytest.mark.parametrize('change',['instruction','final_quote','source_quote','missing_final','missing_source','bool_status'])
def test_final_outcome_known_verdict_requires_exact_original_and_final_evidence(change):
 from choreobench.edit_study import bind_edit_outcome
 from choreobench.edit_authoring import bind_authoring
 from test_choreobench_edit_authoring import material as task_material
 slot,before,answer=task_material();task=bind_authoring(slot,before,answer,model_id='author')['task'];after=copy.deepcopy(before)
 after['events'][0]['action']='Step left while both arms sweep low.';event=before['events'][0]['event_id']
 verdict=dict(status='pass',reason='Synthetic evidence',instruction_quote=answer['instruction'],source_quotes=[dict(event_id=event,quote='Step left')],final_quotes=[dict(event_id=event,quote='Step left')])
 response={k:copy.deepcopy(verdict) for k in ['target','protection','boundary']}
 if change=='instruction':response['target']['instruction_quote']='Never supplied instruction'
 elif change=='final_quote':response['target']['final_quotes'][0]['quote']='both wrists circle'
 elif change=='source_quote':response['protection']['source_quotes'][0]['quote']='both arms sweep low'
 elif change=='missing_final':response['target']['final_quotes']=[]
 elif change=='missing_source':response['boundary']['source_quotes']=[]
 else:response['target']['status']=True
 with pytest.raises(ValueError):bind_edit_outcome(task,before,after,response,model_id='fixture')


@pytest.mark.parametrize('status,joint',[('pass',True),('fail',False),('unknown',None)])
def test_final_outcome_known_unknown_and_fail_are_distinct(status,joint):
 from choreobench.edit_authoring import bind_authoring
 from choreobench.edit_study import edit_outcome_request,bind_edit_outcome
 from test_choreobench_edit_authoring import material as task_material
 import json
 slot,before,answer=task_material();task=bind_authoring(slot,before,answer,model_id='author')['task']
 after=copy.deepcopy(before);after['events'][0]['action']='Step left while both arms sweep low.'
 request=edit_outcome_request(task,before,after);data=json.loads(request['messages'][-1]['content'])
 assert set(data)=={'instruction','target_event_id','source_events','final_events','sample_rate','unchanged_original'}
 assert not data['unchanged_original'] and 'semantic_goals' not in data
 event=before['events'][0]['event_id']
 evidence=dict(status=status,reason='Synthetic final-state verdict only',instruction_quote='' if status=='unknown' else task['instruction'],
  source_quotes=[] if status=='unknown' else [dict(event_id=event,quote=before['events'][0]['action'])],
  final_quotes=[] if status=='unknown' else [dict(event_id=event,quote=after['events'][0]['action'])])
 value={k:copy.deepcopy(evidence) for k in ('target','protection','boundary')}
 result=bind_edit_outcome(task,before,after,value,model_id='independent-reviewer')
 assert result['model_joint_success'] is joint and not result['semantic_truth_verified']
 assert result['origin']=='model_judge' and result['human_reviews']==0 and not result['music_seen']
 assert result['final_plan_sha256']==object_sha(after) and result['source_plan_sha256']==object_sha(before)
 if status=='unknown':
  value['boundary']=dict(evidence,status='fail',instruction_quote=task['instruction'],source_quotes=[dict(event_id=event,quote=before['events'][0]['action'])],final_quotes=[dict(event_id=event,quote=after['events'][0]['action'])])
  assert bind_edit_outcome(task,before,after,value,model_id='independent-reviewer')['model_joint_success'] is False


@pytest.mark.parametrize('change',['final_quote','source_quote','instruction','foreign_event','empty_final','empty_source','extra_field','score'])
def test_final_outcome_cannot_use_invented_or_wrong_side_evidence(change):
 from choreobench.edit_authoring import bind_authoring
 from choreobench.edit_study import bind_edit_outcome
 from test_choreobench_edit_authoring import material as task_material
 slot,before,answer=task_material();task=bind_authoring(slot,before,answer,model_id='author')['task']
 after=copy.deepcopy(before);after['events'][0]['action']='Step left while both arms sweep low.';event=before['events'][0]['event_id']
 verdict=dict(status='pass',reason='Synthetic boundary',instruction_quote=task['instruction'],source_quotes=[dict(event_id=event,quote=before['events'][0]['action'])],final_quotes=[dict(event_id=event,quote=after['events'][0]['action'])])
 value={k:copy.deepcopy(verdict) for k in ('target','protection','boundary')}
 v=value['protection']
 if change=='final_quote':v['final_quotes'][0]['quote']=before['events'][0]['action']
 elif change=='source_quote':v['source_quotes'][0]['quote']=after['events'][0]['action']
 elif change=='instruction':v['instruction_quote']='Never supplied instruction'
 elif change=='foreign_event':v['final_quotes'][0]['event_id']='absent-event'
 elif change=='empty_final':v['final_quotes']=[]
 elif change=='empty_source':v['source_quotes']=[]
 elif change=='extra_field':v['truth']='authenticated-human'
 else:v['status']='rated';v['score']=5
 with pytest.raises(ValueError):bind_edit_outcome(task,before,after,value,model_id='reviewer')


@pytest.mark.parametrize('field,value',[('track','upper'),('end',1),('evidence',['unrequested-evidence']),('event_id','different-event')])
def test_final_outcome_rejects_non_action_edit_before_review_request(field,value):
 from choreobench.edit_authoring import bind_authoring
 from choreobench.edit_study import edit_outcome_request
 from test_choreobench_edit_authoring import material as task_material
 slot,before,answer=task_material();task=bind_authoring(slot,before,answer,model_id='author')['task']
 after=copy.deepcopy(before);after['events'][0][field]=value
 with pytest.raises(ValueError):edit_outcome_request(task,before,after)


def outcome_material(tmp_path,monkeypatch,mode='submitted'):
 import json
 from types import SimpleNamespace
 from choreobench import edit_agent
 from choreobench.edit_study import run_edit_outcomes
 registry,provider,_,factory,ac=author_material(tmp_path)
 value=read_json(registry);value['methods']=['generic','choreography'];value.pop('record_sha256');write_json(registry,_sealed(value),overwrite=True)
 def author(*a,**k):return run_authoring(*a,**k,provider_factory=factory)
 editor_identity=dict(kind='replay',synthetic=True,model='fixture-editor')
 def generate(request):
  data=json.loads(request['messages'][-1]['content']);event=data['task']['allowed_event_ids'][0]
  answer=dict(tool='abstain',operations=[],reason='Synthetic abstention') if mode=='abstained' else dict(tool='submit_patch',operations=[dict(event_id=event,field='action',value='Step left while both arms sweep low.')],reason='Synthetic edit')
  return dict(text=json.dumps(answer),raw={},identity=editor_identity,synthetic=True,stop_reason='stop',costs=dict(input_tokens=10,output_tokens=20))
 monkeypatch.setattr(edit_agent,'create_provider',lambda *a,**k:SimpleNamespace(identity=editor_identity,generate=generate))
 study=tmp_path/'study';r=run_edit_study(registry,provider,dict(kind='replay'),study,author_runner=author)
 reviewer=tmp_path/'reviewer.json';write_json(reviewer,dict(kind='replay',path='independent-reviewer'))
 calls=[];identity=dict(kind='replay',synthetic=True,model='fixture-outcome-reviewer')
 def review_generate(request):
  calls.append(copy.deepcopy(request));data=json.loads(request['messages'][-1]['content']);event=data['target_event_id']
  a=next(e for e in data['source_events'] if e['event_id']==event);b=next(e for e in data['final_events'] if e['event_id']==event)
  assert 'semantic_goals' not in data and 'author' not in data and 'method' not in data
  verdict=dict(status='pass',reason='Synthetic outcome',instruction_quote=data['instruction'],source_quotes=[dict(event_id=event,quote=a['action'])],final_quotes=[dict(event_id=event,quote=b['action'])])
  answer={k:copy.deepcopy(verdict) for k in ('target','protection','boundary')}
  return dict(text=json.dumps(answer),raw={},identity=identity,synthetic=True,stop_reason='stop',costs=dict(input_tokens=15,output_tokens=25))
 return study,reviewer,tmp_path/'outcomes',calls,lambda *a,**k:SimpleNamespace(identity=identity,generate=review_generate),r


@pytest.mark.parametrize('mode',['submitted','abstained'])
def test_outcome_runner_same_final_original_full_denominator_and_zero_call_resume(tmp_path,monkeypatch,mode):
 from choreobench.edit_study import run_edit_outcomes
 study,provider,out,calls,factory,original=outcome_material(tmp_path,monkeypatch,mode)
 result=run_edit_outcomes(study,provider,out,provider_factory=factory)
 assert result['expected_editor_cells']==4 and result['calls_used']==2 and len(calls)==2
 assert result['status_counts']=={'study_author_missing_source':2,'valid':2}
 assert len({c['call_id'] for c in calls})==2
 assert result['model_joint_successes']==2 and not result['formal_E4_completed'] and not result['semantic_truth_verified']
 assert all(json.loads(c['messages'][-1]['content'])['unchanged_original']==(mode=='abstained') for c in calls)
 assert run_edit_outcomes(study,provider,out,resume=True,provider_factory=lambda *a,**k:pytest.fail('No reviewer in complete audit'))==result
 assert len(calls)==2 and read_json(study/'results.json')==original


@pytest.mark.parametrize('artifact',['editor_raw','author_raw','before','task','review_raw'])
def test_outcome_runner_missing_original_or_review_evidence_no_replacement(tmp_path,monkeypatch,artifact):
 from choreobench.edit_study import run_edit_outcomes
 study,provider,out,calls,factory,_=outcome_material(tmp_path,monkeypatch)
 run_edit_outcomes(study,provider,out,provider_factory=factory)
 if artifact=='editor_raw':path=next((study/'runs').rglob('*.raw.json'))
 elif artifact=='author_raw':path=next((study/'authoring/calls').glob('*.raw.json'))
 elif artifact=='before':path=next((study/'runs').rglob('before.json'))
 elif artifact=='task':path=next((study/'authoring/tasks').glob('*.json'))
 else:path=next((out/'calls').glob('*.raw.json'))
 path.unlink()
 with pytest.raises(ValueError):run_edit_outcomes(study,provider,out,resume=True,provider_factory=lambda *a,**k:pytest.fail('No replacement reviewer'))
 assert len(calls)==2


def test_outcome_runner_invalid_review_retained_not_best_of(tmp_path,monkeypatch):
 from types import SimpleNamespace
 from choreobench.edit_study import run_edit_outcomes
 study,provider,out,calls,factory,_=outcome_material(tmp_path,monkeypatch)
 original=factory();identity=original.identity
 def generate(request):
  response=original.generate(request);value=json.loads(response['text']);value['target']['final_quotes'][0]['quote']='Invented final action';response['text']=json.dumps(value);return response
 result=run_edit_outcomes(study,provider,out,provider_factory=lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
 assert result['status_counts']=={'invalid':2,'study_author_missing_source':2} and result['model_joint_successes']==0
 assert run_edit_outcomes(study,provider,out,resume=True,provider_factory=lambda *a,**k:pytest.fail('No retry'))==result
 assert len(calls)==2


def test_outcome_runner_infrastructure_failure_stops_and_remains_original_on_resume(tmp_path,monkeypatch):
 from types import SimpleNamespace
 from choreobench.edit_study import run_edit_outcomes
 study,provider,out,calls,factory,_=outcome_material(tmp_path,monkeypatch)
 original=factory();identity=original.identity
 def generate(request):
  response=original.generate(request);response['stop_reason']='infrastructure_error';response['text']='';return response
 result=run_edit_outcomes(study,provider,out,provider_factory=lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
 assert result['calls_used']==1 and len(calls)==1
 assert result['status_counts']=={'not_attempted_after_infrastructure_failure':1,'study_author_missing_source':2,'unknown_infrastructure':1}
 assert run_edit_outcomes(study,provider,out,resume=True,provider_factory=lambda *a,**k:pytest.fail('No replacement'))==result
 assert len(calls)==1


@pytest.mark.parametrize('missing',['target','neighbor'])
def test_boundary_pass_requires_target_and_unchanged_neighbor_evidence(missing):
 from choreobench.edit_study import bind_edit_outcome
 from choreobench.edit_authoring import bind_authoring
 from test_choreobench_edit_authoring import material as task_material
 slot,before,answer=task_material()
 before['events'][0]['end']//=2
 neighbor=copy.deepcopy(before['events'][0]);neighbor.update(event_id='untouched-neighbor',start=neighbor['end'],end=before['num_samples'],action='Begin seated, then rise.')
 before['events'].append(neighbor)
 task=bind_authoring(slot,before,answer,model_id='author')['task'];after=copy.deepcopy(before)
 after['events'][0]['action']='Step left while both arms sweep low.'
 target=before['events'][0]['event_id']
 verdict=dict(status='pass',reason='Synthetic evidence boundary',instruction_quote=answer['instruction'],source_quotes=[dict(event_id=target,quote=before['events'][0]['action'])],final_quotes=[dict(event_id=target,quote=after['events'][0]['action'])])
 response={k:copy.deepcopy(verdict) for k in ['target','protection','boundary']}
 boundary=response['boundary']
 if missing=='target':
  for side in ['source_quotes','final_quotes']:
   boundary[side]=[dict(event_id='untouched-neighbor',quote=neighbor['action'])]
 with pytest.raises(ValueError,match='target|neighbor'):bind_edit_outcome(task,before,after,response,model_id='fixture')
 for side,plan in [('source_quotes',before),('final_quotes',after)]:
  boundary[side]=[dict(event_id=e['event_id'],quote=e['action']) for e in plan['events']]
 assert bind_edit_outcome(task,before,after,response,model_id='fixture')['model_joint_success'] is True


@pytest.mark.parametrize('omit',['target','neighbor','none'])
def test_known_boundary_verdict_requires_target_and_all_supplied_context(omit):
 from choreobench.edit_study import bind_edit_outcome,edit_outcome_request
 from choreobench.edit_authoring import bind_authoring
 from test_choreobench_edit_authoring import material as task_material
 slot,before,answer=task_material();before['events'][0]['end']//=2
 neighbor=copy.deepcopy(before['events'][0]);neighbor.update(event_id='untouched-neighbor',start=neighbor['end'],end=before['num_samples'])
 neighbor['action']='Begin seated, then rise slowly.';before['events'].append(neighbor)
 task=bind_authoring(slot,before,answer,model_id='author')['task'];after=copy.deepcopy(before)
 after['events'][0]['action']='Step left while both arms sweep low.'
 eid=before['events'][0]['event_id']
 evidence=dict(status='pass',reason='Synthetic full boundary comparison',instruction_quote=answer['instruction'],
  source_quotes=[dict(event_id=eid,quote=before['events'][0]['action'])],final_quotes=[dict(event_id=eid,quote=after['events'][0]['action'])])
 response={k:copy.deepcopy(evidence) for k in ['target','protection','boundary']}
 response['boundary']['source_quotes'].append(dict(event_id=neighbor['event_id'],quote=neighbor['action']))
 response['boundary']['final_quotes'].append(dict(event_id=neighbor['event_id'],quote=neighbor['action']))
 if omit!='none':
  removed=eid if omit=='target' else neighbor['event_id']
  for side in ['source_quotes','final_quotes']:response['boundary'][side]=[r for r in response['boundary'][side] if r['event_id']!=removed]
  with pytest.raises(ValueError,match='target action|neighboring/concurrent'):bind_edit_outcome(task,before,after,response,model_id='fixture')
 else:
  assert bind_edit_outcome(task,before,after,response,model_id='fixture')['model_joint_success'] is True
  data=__import__('json').loads(edit_outcome_request(task,before,after)['messages'][-1]['content'])
  assert {r['event_id'] for r in data['final_events']}=={eid,neighbor['event_id']}


def test_outcome_real_executor_provenance_is_bound_and_not_a_permission_bypass():
 from choreobench.edit_study import edit_outcome_request
 from choreobench.edit_authoring import bind_authoring
 from test_choreobench_edit_authoring import material as task_material
 slot,before,answer=task_material();task=bind_authoring(slot,before,answer,model_id='author')['task'];after=copy.deepcopy(before)
 after['events'][0]['action']='Step left while both arms sweep low.'
 after['provenance']['choreobench_edit']=dict(base_plan_sha256=object_sha(before),patch_sha256='a'*64)
 assert edit_outcome_request(task,before,after)['response_kind']=='edit_outcome_review'
 after['provenance']['choreobench_edit']['base_plan_sha256']='0'*64
 with pytest.raises(ValueError,match='provenance'):edit_outcome_request(task,before,after)


def test_final_outcome_categories_keep_controls_separate(tmp_path,monkeypatch):
 from choreobench.edit_study import run_edit_outcomes
 study,provider,out,calls,factory,_=outcome_material(tmp_path,monkeypatch)
 result=run_edit_outcomes(study,provider,out,provider_factory=factory)
 assert len(result['category_summaries'])==2
 assert all(s['registered_category']=='body_preservation' and s['registered_states']==2 and s['model_pass']==1 and s['formal_success_rate'] is None for s in result['category_summaries'])
 assert all(r['registered_category']=='body_preservation' for r in result['rows'])
