"""Synthetic open-motion diagnostics verify binding, not metric truth."""
import copy,json
from types import SimpleNamespace
import pytest
from captionlab import motion_fact_eval as m
from choreobench.io import read_json,write_json


def material(tmp_path,monkeypatch,mode=None):
 src=tmp_path/'inventory.json';spec=tmp_path/'spec.json';out=tmp_path/'run';calls=[]
 write_json(src,dict(schema='captionlab.motion_fact_inventory.v1',seed=42,cases=[dict(case_id='c',candidate_text='An arm opens.',reference_text='One arm unfolds.',private_expected=dict(match=True),private_source='Software fixture, not human gold')]))
 write_json(spec,dict(kind='replay',path='unused'))
 identity=dict(kind='replay',synthetic=True,model='fixture')
 def generate(req):
  calls.append(copy.deepcopy(req));data=json.loads(req['messages'][-1]['content'])
  assert 'private_expected' not in data and 'private_source' not in data
  if 'source_text' in data:
   t=data['source_text'];v=dict(facts=[dict(id='f',action='open',body_part='arm',side='unspecified',side_frame='unspecified',start=0,end=len(t),quote=t)],relations=[],uncertainties=[])
   if mode=='bad_span':v['facts'][0]['quote']='invented'
  else:
   v=dict(judgments=[dict(candidate_id=p['candidate']['id'],reference_id=p['reference']['id'],status='match',reason='Synthetic equivalent phrases.',candidate_quote=p['candidate']['quote'],reference_quote=p['reference']['quote']) for p in data['pairs']])
  return dict(text=json.dumps(v),raw={},identity=identity,synthetic=True,stop_reason='stop',costs=dict(input_tokens=100,output_tokens=20))
 monkeypatch.setattr(m,'create_provider',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
 return src,spec,out,calls


def test_full_fact_eval_three_calls_and_no_resampling(tmp_path,monkeypatch):
 src,spec,out,calls=material(tmp_path,monkeypatch)
 r=m.run_diagnostics(src,spec,out)
 assert r['calls_used']==3 and r['rows'][0]['score']['known_fact_metrics']['f1']==1
 assert not r['extraction_or_matching_truth_verified'] and not r['formal_E5_completed']
 assert m.run_diagnostics(src,spec,out,resume=True)==r and len(calls)==3


def test_bad_extraction_retains_failed_case_without_invented_score(tmp_path,monkeypatch):
 src,spec,out,calls=material(tmp_path,monkeypatch,'bad_span')
 r=m.run_diagnostics(src,spec,out)
 assert r['calls_used']==1 and r['registered_cases']==1 and r['rows'][0]['status']=='failed'
 assert 'score' not in r['rows'][0]
 assert m.run_diagnostics(src,spec,out,resume=True)==r and len(calls)==1


def test_insufficient_matching_budget_and_resume_preserve_same_failure(tmp_path,monkeypatch):
 src,spec,out,calls=material(tmp_path,monkeypatch)
 r=m.run_diagnostics(src,spec,out,max_calls=2)
 assert r['calls_used']==2 and r['rows'][0]['status']=='failed' and 'score' not in r['rows'][0]
 assert 'fit remaining frozen budget' in r['rows'][0]['error']
 assert m.run_diagnostics(src,spec,out,max_calls=2,resume=True)==r and len(calls)==2


@pytest.mark.parametrize('artifact',['raw','extraction','match','score'])
def test_completed_missing_evidence_refuses_replacement(tmp_path,monkeypatch,artifact):
 src,spec,out,calls=material(tmp_path,monkeypatch);m.run_diagnostics(src,spec,out)
 if artifact=='raw':p=next((out/'calls').glob('*.raw.json'))
 elif artifact=='extraction':p=out/'c/candidate.json'
 elif artifact=='match':p=out/'c/matches.json'
 else:p=out/'c/score.json'
 p.unlink()
 with pytest.raises((ValueError,FileNotFoundError)):m.run_diagnostics(src,spec,out,resume=True)
 assert len(calls)==3


def test_changed_inventory_not_reused_or_replaced(tmp_path,monkeypatch):
 src,spec,out,calls=material(tmp_path,monkeypatch);m.run_diagnostics(src,spec,out)
 value=read_json(src);value['cases'][0]['reference_text']='Unrelated phrase';write_json(src,value,overwrite=True)
 with pytest.raises(ValueError,match='drift'):m.run_diagnostics(src,spec,out,resume=True)
 assert len(calls)==3



def test_quote_protocol_real_binding_and_mode_change_cannot_resume(tmp_path,monkeypatch):
 src,spec,out,calls=material(tmp_path,monkeypatch)
 original=m.create_provider({},'planner');generate=original.generate
 def quotes(req):
  result=generate(req);value=json.loads(result['text'])
  for field in ('facts','relations'):
   for row in value.get(field,[]):
    row.pop('start');row.pop('end');row['occurrence']=0
  result['text']=json.dumps(value);return result
 original.generate=quotes;monkeypatch.setattr(m,'create_provider',lambda *a,**k:original)
 result=m.run_diagnostics(src,spec,out,evidence_mode='quotes')
 assert result['successful_cases']==1 and result['calls_used']==3
 assert m.run_diagnostics(src,spec,out,evidence_mode='quotes',resume=True)==result and len(calls)==3
 with pytest.raises(ValueError,match='drift'):m.run_diagnostics(src,spec,out,evidence_mode='spans',resume=True)
 assert len(calls)==3


def test_provider_identity_drift_aborts_without_following_model_call(tmp_path,monkeypatch):
 src,spec,out,calls=material(tmp_path,monkeypatch)
 provider=m.create_provider({},role='planner');generate=provider.generate
 def drift(request):
  value=generate(request);provider.identity=dict(provider.identity,model='changed')
  return value
 provider.generate=drift;monkeypatch.setattr(m,'create_provider',lambda *a,**kw:provider)
 with pytest.raises(ValueError,match='identity changed'):
  m.run_diagnostics(src,spec,out)
 assert len(calls)==1 and len(list((out/'calls').glob('*.raw.json')))==1



def test_image_reference_provenance_entered_request_and_anatomy_rejected(tmp_path,monkeypatch):
 src,spec,out,calls=material(tmp_path,monkeypatch)
 inv=read_json(src);inv['cases'][0]['source_side_frames']={'reference':'image'};write_json(src,inv,overwrite=True)
 provider=m.create_provider({},'planner');generate=provider.generate
 def wrong(request):
  result=generate(request)
  if '/reference/' in request['call_id']:
   payload=json.loads(request['messages'][-1]['content'])
   assert payload['source_direction_frame']=='image'
   assert request['response_schema']['properties']['facts']['items']['properties']['side_frame']['enum']==['image','unspecified']
   value=json.loads(result['text']);value['facts'][0].update(side='left',side_frame='anatomical');result['text']=json.dumps(value)
  return result
 provider.generate=wrong;monkeypatch.setattr(m,'create_provider',lambda *a,**k:provider)
 result=m.run_diagnostics(src,spec,out)
 assert result['failed_cases']==1 and 'anatomical' in result['rows'][0]['error'] and result['calls_used']==2
 assert m.run_diagnostics(src,spec,out,resume=True)==result and len(calls)==2


def test_reference_score_reports_bounded_extraction_not_full_recall(tmp_path,monkeypatch):
 src,spec,out,calls=material(tmp_path,monkeypatch)
 result=m.run_diagnostics(src,spec,out)
 score=result['rows'][0]['score']
 assert score['extraction_fact_limit']==8 and not score['full_source_recall_verified']
 assert score['extraction_limit_reached']=={'candidate':False,'reference':False}
 assert score['metric_denominator_scope'].startswith('Only extracted bounded')
 assert m.run_diagnostics(src,spec,out,resume=True)==result and len(calls)==3
