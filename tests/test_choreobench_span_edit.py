"""Source-text permissions are literal, never verified visual or music facts."""
import copy,json
from types import SimpleNamespace
import pytest
from choreobench import span_edit
from choreobench.fixtures import create_demo
from choreobench.io import read_json,write_json,file_sha
from choreobench.revision_protocol import action_task


@pytest.fixture
def setup(tmp_path,monkeypatch):
 made=create_demo(tmp_path/'demo');plan=read_json(made['source_plan'])
 event=plan['events'][0];source=event['action'];needle='small side steps'
 # demo order puts lower first; permit replacing an exact phrase only.
 assert needle in source
 start=source.index(needle);task=action_task(plan,event['event_id'],'Replace only the small side steps phrase with quiet forward steps; preserve all surrounding text.')
 doc=dict(schema='choreobench.span_edit_task.v1',task=task,event_id=event['event_id'],source_action=source,
   spans=[dict(span_id='motion',start=start,end=start+len(needle),source_text=needle)])
 path=tmp_path/'task.json';write_json(path,doc)
 inventory=tmp_path/'inventory.json';write_json(inventory,dict(schema='choreobench.span_pair_inventory.v1',seed=7,cases=[
   dict(plan_path=made['source_plan'],plan_sha256=file_sha(made['source_plan']),task_path=str(path),task_sha256=file_sha(path))]))
 calls=[];identity=dict(kind='replay',synthetic=True,fixture='not-real-model');behavior=dict(bad=None)
 def generate(request):
  calls.append(copy.deepcopy(request));sid=event['event_id']
  replacement='quiet forward steps'
  if behavior['bad']=='length':replacement='step '*25
  if request['call_id'].endswith('/whole_action'):
   value=dict(event_id=sid,action=source.replace(needle,replacement))
   if behavior['bad']=='outside':value['action']='Discard all original steps.'
  else:
   value=dict(event_id=sid,replacements=[dict(span_id='motion',text=replacement)])
   if behavior['bad']=='id':value['replacements'][0]['span_id']='wrong'
  if behavior['bad']=='identity':identity['drift']=True
  return dict(text=json.dumps(value),raw={},identity=copy.deepcopy(identity),synthetic=True,stop_reason='stop',costs=dict(input_tokens=10,output_tokens=10,latency_seconds=0))
 monkeypatch.setattr(span_edit,'create_provider',lambda *a,**kw:SimpleNamespace(identity=identity,generate=generate))
 return inventory,doc,plan,calls,behavior


def test_both_formats_compile_same_authorized_replacement_without_mutation(setup):
 _,doc,_,_,_=setup;old=copy.deepcopy(doc)
 action=doc['source_action'].replace('small side steps','quiet forward steps')
 a=span_edit.compile_span_response(doc,json.dumps(dict(event_id=doc['event_id'],action=action)),'whole_action')
 b=span_edit.compile_span_response(doc,json.dumps(dict(event_id=doc['event_id'],replacements=[dict(span_id='motion',text='quiet forward steps')])),'source_spans')
 assert a==b and doc==old


def test_pair_one_call_each_same_system_seed_scope_and_zero_resume_calls(setup,tmp_path):
 inv,doc,plan,calls,_=setup;out=tmp_path/'runs'
 r=span_edit.run_span_pair(inv,dict(kind='replay'),out)
 assert r['complete'] and r['calls_used']==2 and all(x['status']=='valid' for x in r['rows'])
 assert calls[0]['messages'][0]==calls[1]['messages'][0] and calls[0]['seed']==calls[1]['seed']
 a,b=[json.loads(x['messages'][-1]['content']) for x in calls]
 a.pop('output_contract');b.pop('output_contract');assert a==b
 assert span_edit.run_span_pair(inv,dict(kind='replay'),out,resume=True)==r and len(calls)==2
 assert r['human_reviews']==0 and not r['quality_verified']


@pytest.mark.parametrize('bad',['outside','id','length'])
def test_illegal_source_changes_ids_and_writing_limits_retained_no_resampling(setup,tmp_path,bad):
 inv,_,_,calls,behavior=setup;behavior['bad']=bad
 r=span_edit.run_span_pair(inv,dict(kind='replay'),tmp_path/'runs')
 assert r['complete'] and len(calls)==2
 assert any(x['status']=='failed' and x['retained_in_denominator'] for x in r['rows'])


def test_span_offsets_and_overlap_rejected(setup):
 _,doc,plan,_,_=setup
 bad=copy.deepcopy(doc);bad['spans'][0]['start']+=1
 with pytest.raises(ValueError,match='source text'):span_edit.validate_span_task(bad,plan)
 bad=copy.deepcopy(doc);bad['spans']*=2
 with pytest.raises(ValueError):span_edit.validate_span_task(bad,plan)


def test_completed_missing_call_or_artifact_refuses_before_dispatch(setup,tmp_path):
 inv,_,_,calls,_=setup;out=tmp_path/'runs';span_edit.run_span_pair(inv,dict(kind='replay'),out)
 p=next((out/'calls').glob('*.raw.json'));p.unlink()
 with pytest.raises(ValueError,match='inventory changed'):span_edit.run_span_pair(inv,dict(kind='replay'),out,resume=True)
 assert len(calls)==2


def test_provider_drift_stops_remaining_requests(setup,tmp_path):
 inv,_,_,calls,behavior=setup;behavior['bad']='identity'
 r=span_edit.run_span_pair(inv,dict(kind='replay'),tmp_path/'runs')
 assert not r['complete'] and len(calls)==1


def test_literal_permission_does_not_claim_semantic_contradiction_detection(setup):
 _,doc,_,_,_=setup
 action,replacements=span_edit.compile_span_response(doc,json.dumps(dict(event_id=doc['event_id'],
  replacements=[dict(span_id='motion',text='quiet forward steps while not moving feet')])),'source_spans')
 assert 'not moving feet' in action
 # This semantic conflict cannot be certified by retaining source bytes.
 assert replacements[0]['text'].endswith('not moving feet')
