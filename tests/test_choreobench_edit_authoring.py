"""Model author declarations never substitute for human semantic validation."""
import copy,json
import pytest
from choreobench.edit_authoring import author_request,bind_authoring
from choreobench.io import object_sha
from test_choreobench_review_sources import fixture


def material(category='body_preservation'):
 plan=fixture()['plan']
 event=plan['events'][0];event['action']='Step left while both wrists circle.'
 slot=dict(slot_id='registered-1',song_id=plan['song_id'],registered_category=category)
 answer=dict(event_id=event['event_id'],instruction='Keep the leftward step and change the wrist circles to low arm sweeps.',
  protected_facts=[dict(description='Keep the leftward step.',source_quote='Step left')],before_status='unmet',source_quote=event['action'],reason='Current wrists circle, not sweep.')
 return slot,plan,answer


def test_authoring_uses_public_plan_no_identity_or_reference_leak():
 slot,plan,answer=material();request=author_request(slot,plan)
 payload=json.loads(request['messages'][-1]['content'])
 assert payload['source_plan']['provenance']=={}
 assert all(e['evidence']==[] for e in payload['source_plan']['events'])
 assert set(payload)=={'category','source_plan','edit_permissions'}
 assert request['call_id']=='registered-1/author'


def test_model_authored_task_bound_to_exact_source_and_preservation_contract():
 slot,plan,answer=material();before=copy.deepcopy(plan)
 record=bind_authoring(slot,plan,json.dumps(answer),model_id='synthetic-author')
 task=record['task']
 assert task['source_plan_sha256']==object_sha(plan) and task['allowed_event_ids']==[plan['events'][0]['event_id']]
 assert len(task['semantic_goals'])==3 and task['goals']==[] and task['repair_event_ids']==[]
 assert record['origin']=='model_judge' and record['status']=='model_declared_valid'
 assert not record['human_validated'] and not record['private_requirements_within_instruction_authenticated']
 assert plan==before and record['record_sha256']==object_sha({k:v for k,v in record.items() if k!='record_sha256'})


@pytest.mark.parametrize('category,status',[('already_satisfied','already_satisfied'),('ambiguous_or_conflicting','conflict')])
def test_control_categories_are_preserved_separate_from_change_tasks(category,status):
 slot,plan,answer=material(category);answer.update(before_status=status,instruction='Keep original and abstain under the public condition.')
 r=bind_authoring(slot,plan,answer,model_id='fixture')
 assert r['registered_category']==category and r['task']['instruction'].startswith(answer['instruction'])


def test_unknown_unconstructible_slot_no_fabricated_task():
 slot,plan,answer=material();answer.update(event_id=None,source_quote='',protected_facts=[],before_status='unknown',instruction='')
 r=bind_authoring(slot,plan,answer,model_id='fixture')
 assert r['status']=='unknown' and r['task'] is None


@pytest.mark.parametrize('bad',['event','quote','protected_quote','no_facts','relabel','unknown_facts','extra','empty_instruction','reserved_scope'])
def test_author_source_permissions_quotes_and_category_cannot_be_changed(bad):
 slot,plan,answer=material()
 if bad=='event':answer['event_id']='absent'
 elif bad=='quote':answer['source_quote']='Invented source'
 elif bad=='protected_quote':answer['protected_facts'][0]['source_quote']='Never present'
 elif bad=='no_facts':answer['protected_facts']=[]
 elif bad=='relabel':answer['before_status']='already_satisfied'
 elif bad=='unknown_facts':answer.update(before_status='unknown',event_id=None,source_quote='')
 elif bad=='extra':answer['human_pass']=True
 elif bad=='empty_instruction':answer['instruction']=' '
 else:slot['registered_category']='freely_rewrite_all_times'
 with pytest.raises(ValueError):bind_authoring(slot,plan,answer,model_id='fixture')


def test_protected_facts_are_public_even_when_not_repeated_in_initial_instruction():
 slot,plan,answer=material();answer['instruction']='Change wrist circles to low arm sweeps.'
 record=bind_authoring(slot,plan,answer,model_id='fixture')
 instruction=record['task']['instruction']
 assert 'Keep the leftward step.' in instruction and 'Step left' in instruction
 assert 'preserve every other event' in instruction
 assert instruction.startswith(answer['instruction'])
 assert answer['instruction']==record['decision']['instruction']


@pytest.mark.parametrize('quote',[[],{},None])
def test_unknown_author_quote_cannot_use_falsy_nontext(quote):
 slot,plan,answer=material();answer.update(event_id=None,source_quote=quote,protected_facts=[],before_status='unknown',instruction='')
 with pytest.raises(ValueError,match='bounded text'):bind_authoring(slot,plan,answer,model_id='fixture')


@pytest.mark.parametrize('seed,budget',[(2**63,2048),(42,8193),(True,2048),(42,0)])
def test_author_generation_budget_rejected_before_dispatch(seed,budget):
 slot,plan,_=material()
 with pytest.raises(ValueError):author_request(slot,plan,seed=seed,max_output_tokens=budget)


def author_material(tmp_path,mode=None):
 from pathlib import Path
 from types import SimpleNamespace
 from choreobench.io import write_json,file_sha
 from choreobench.baseline import _sealed
 slot,plan,answer=material()
 song=fixture()['song']
 dataset=tmp_path/'dataset.json'
 write_json(dataset,dict(schema='choreobench.dataset.v1',dataset_id='author-fixture',synthetic=True,grouping=dict(status='unverified',evidence=''),songs=[song]))
 source=tmp_path/'original';source.mkdir();write_json(source/'plan.json',plan)
 write_json(source/'results.json',_sealed(dict(complete=True,full_song_succeeded=True,method='react',artifacts={'plan.json':file_sha(source/'plan.json')})))
 slot.update(source_song_sha256=object_sha(song),source_method='react',source_plan_path=str(source/'plan.json'))
 missing=dict(slot,slot_id='registered-missing',source_plan_path=str(tmp_path/'missing/plan.json'))
 registry=tmp_path/'registry.json'
 write_json(registry,_sealed(dict(schema='choreobench.e4_registered_slots.v1',expected_tasks=2,slots=[slot,missing],source_dataset=dict(path=str(dataset),sha256=file_sha(dataset)))))
 provider=tmp_path/'provider.json';write_json(provider,dict(kind='replay',path='not-dispatched-directly'))
 calls=[];identity=dict(kind='replay',synthetic=True,model='fixture-author')
 def generate(request):
  calls.append(copy.deepcopy(request));value=copy.deepcopy(answer)
  if mode=='bad_quote':value['source_quote']='Invented source movement'
  return dict(text=json.dumps(value),identity=identity,synthetic=True,raw={},stop_reason='infrastructure_error' if mode=='infra' else 'stop',costs=dict(input_tokens=10,output_tokens=20))
 return registry,provider,tmp_path/'author',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate),calls


def test_author_runner_retains_missing_registered_tasks_no_fake_targets(tmp_path):
 from choreobench.edit_authoring import run_authoring
 from choreobench.io import read_json
 registry,provider,out,factory,calls=author_material(tmp_path)
 result=run_authoring(registry,provider,out,provider_factory=factory)
 assert result['expected_registered_tasks']==2 and result['calls_used']==1 and result['model_declared_valid']==1 and result['missing_source']==1
 assert result['rows'][1]['status']=='missing_source' and result['rows'][1]['task_path'] is None
 task=read_json(out/result['rows'][0]['task_path']);assert 'Keep the leftward step.' in task['instruction']
 assert not result['formal_E4_completed']
 assert run_authoring(registry,provider,out,resume=True,provider_factory=lambda *a,**k:pytest.fail('Model recreated in complete audit'))==result
 assert len(calls)==1


@pytest.mark.parametrize('mode',['bad_quote','infra'])
def test_author_failed_or_unknown_original_response_not_replaced(tmp_path,mode):
 from choreobench.edit_authoring import run_authoring
 registry,provider,out,factory,calls=author_material(tmp_path,mode)
 result=run_authoring(registry,provider,out,provider_factory=factory)
 assert result['failed']==1 and result['calls_used']==1 and result['rows'][0]['task_path'] is None
 assert run_authoring(registry,provider,out,resume=True,provider_factory=lambda *a,**k:pytest.fail('No replacement'))==result
 assert len(calls)==1


def test_author_completed_raw_missing_is_not_resampled(tmp_path):
 from choreobench.edit_authoring import run_authoring
 registry,provider,out,factory,calls=author_material(tmp_path)
 run_authoring(registry,provider,out,provider_factory=factory)
 next((out/'calls').glob('*.raw.json')).unlink()
 with pytest.raises(ValueError,match='missing/changed'):run_authoring(registry,provider,out,resume=True,provider_factory=lambda *a,**k:pytest.fail('No replacement'))
 assert len(calls)==1


def test_new_source_appearing_requires_new_author_freeze_not_silent_resume(tmp_path):
 from choreobench.edit_authoring import run_authoring
 from choreobench.io import read_json,write_json,file_sha
 from choreobench.baseline import _sealed
 registry,provider,out,factory,calls=author_material(tmp_path)
 run_authoring(registry,provider,out,provider_factory=factory)
 p=tmp_path/'missing';p.mkdir();write_json(p/'plan.json',read_json(tmp_path/'original/plan.json'))
 write_json(p/'results.json',_sealed(dict(complete=True,full_song_succeeded=True,method='react',artifacts={'plan.json':file_sha(p/'plan.json')})))
 with pytest.raises(ValueError,match='snapshot/code changed'):run_authoring(registry,provider,out,resume=True,provider_factory=lambda *a,**k:pytest.fail('New source dispatch'))
 assert len(calls)==1


@pytest.mark.parametrize('change',['quote','source','empty','status','extra'])
def test_separate_task_audit_requires_exact_instruction_and_source_evidence(change):
 from choreobench.edit_authoring import audit_task_request,bind_task_audit
 slot,plan,answer=material();record=bind_authoring(slot,plan,answer,model_id='author')
 task=record['task'];req=audit_task_request(task,plan,slot['registered_category'])
 public=json.loads(req['messages'][-1]['content'])
 assert 'author' not in public and 'semantic_goals' not in public
 evidence=dict(status='supported',reason='Software fixture only',instruction_quote=answer['instruction'],
  source_quotes=[dict(event_id=plan['events'][0]['event_id'],quote='Step left')])
 response={k:copy.deepcopy(evidence) for k in ('category_fit','satisfiable','boundary_compatible')}
 if change=='quote':response['satisfiable']['instruction_quote']='Never supplied instruction'
 elif change=='source':response['satisfiable']['source_quotes'][0]['quote']='Invented source evidence'
 elif change=='empty':response['satisfiable']['source_quotes']=[]
 elif change=='status':response['satisfiable']['status']='human-pass'
 else:response['satisfiable']['score']=5
 with pytest.raises(ValueError):bind_task_audit(task,plan,slot['registered_category'],response,model_id='judge')


def test_task_audit_unknown_and_unsupported_do_not_rewrite_original():
 from choreobench.edit_authoring import bind_task_audit
 slot,plan,answer=material();task=bind_authoring(slot,plan,answer,model_id='author')['task'];before=copy.deepcopy(task)
 response={k:dict(status='unknown',reason='Insufficient supported task evidence',instruction_quote='',source_quotes=[])
  for k in ('category_fit','satisfiable','boundary_compatible')}
 r=bind_task_audit(task,plan,slot['registered_category'],response,model_id='judge')
 assert r['status']=='unknown' and not r['human_validated'] and task==before
 response['satisfiable']=dict(status='unsupported',reason='Software constraint conflict',instruction_quote=answer['instruction'],
  source_quotes=[dict(event_id=plan['events'][0]['event_id'],quote='Step left')])
 r=bind_task_audit(task,plan,slot['registered_category'],response,model_id='judge')
 assert r['status']=='unsupported' and task==before and not r['task_rewritten']
