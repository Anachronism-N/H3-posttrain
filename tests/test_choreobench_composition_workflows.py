"""Synthetic four-method trajectories do not establish choreographic quality."""
import copy,json
from types import SimpleNamespace
import pytest
from choreobench import composition_workflows as m
from choreobench.io import read_json
from test_choreobench_song_agent import source


def material(tmp_path,monkeypatch,method,mutate=None):
 path,sid=source(tmp_path,monkeypatch);calls=[];identity=dict(kind='replay',synthetic=True,fixture='four-method')
 def generate(req):
  calls.append(copy.deepcopy(req));data=json.loads(req['messages'][-1]['content'])
  if req['response_kind']=='coordination_outline':
   answer=dict(motifs=[dict(motif_id='m1',description='Side steps with coordinated arms')],sections=[dict(section_id=s['section_id'],intent='Develop a movement phrase',motif_ids=['m1']) for s in data['song']['music']['sections']])
  elif req['response_kind']=='composition_feedback':answer=dict(feedback='Keep one sustained motion; do not invent prior posture.')
  elif method in ('react','choreography') and data['committed_units']==data['total_units']:answer=dict(tool='finish',proposal=None,reason='All units committed')
  elif method in ('react','choreography') and data['current_draft'] is not None:answer=dict(tool='commit_block',proposal=None,reason='Commit draft')
  else:
   unit=data['planning_unit'];proposal=dict(unit_id=unit['unit_id'],events=[dict(start=unit['start'],end=unit['end'],track='body',action='A sustained side step with concurrent arms',motif_id='m1',source_call_ids=[data['local_proposals'][0]['call_id']])])
   answer=dict(tool='propose_block',proposal=proposal,reason='Draft from local evidence') if method in ('react','choreography') else proposal
  if mutate:mutate(answer,req,data,len(calls))
  return dict(text=json.dumps(answer),identity=identity,synthetic=True,raw={},stop_reason='stop',costs=dict(input_tokens=100,output_tokens=50))
 monkeypatch.setattr(m,'create_provider',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
 out=tmp_path/'run';return path,sid,out,calls


@pytest.mark.parametrize('method,calls_expected',[('workflow',3),('self_refine',7),('react',6),('choreography',6)])
def test_each_method_fresh_outline_empty_history_full_song_and_resume(tmp_path,monkeypatch,method,calls_expected):
 path,sid,out,calls=material(tmp_path,monkeypatch,method)
 result=m.run_composition(path,sid,{'kind':'replay'},out,method=method)
 assert result['full_song_succeeded'] and result['calls_used']==calls_expected and result['committed_units']==2
 first=json.loads(calls[1]['messages'][-1]['content']);assert first['committed_plan']['events']==[]
 assert read_json(out/'contract.json')['fresh_method_outline'] and not result['formal_E3_completed']
 assert m.run_composition(path,sid,{'kind':'replay'},out,method=method,resume=True)==result and len(calls)==calls_expected
 if method=='self_refine':assert [c['response_kind'] for c in calls]==['coordination_outline','composition_propose','composition_feedback','composition_revise','composition_propose','composition_feedback','composition_revise']


def state(tmp_path,monkeypatch):
 path,sid=source(tmp_path,monkeypatch);data=m.load_song_source(path,sid,{})
 outline=data['outline'];return m.CompositionState(data,outline)


def proposal(s,action='Original draft'):
 u=s.source['units']['units'][s.index]
 return dict(unit_id=u['unit_id'],events=[dict(start=u['start'],end=u['end'],track='body',action=action,motif_id='m1',source_call_ids=[m._candidates(s.source['options'],u)[0]['call_id']])])


def decision(tool,value=None):return dict(tool=tool,proposal=value,reason='Synthetic tool contract')


def test_draft_read_check_revision_never_commits_prematurely(tmp_path,monkeypatch):
 s=state(tmp_path,monkeypatch);s.execute(decision('propose_block',proposal(s)))
 assert s.plan['events']==[] and s.index==0
 assert s.execute(decision('check_block'))['status']=='structurally_valid' and s.plan['events']==[]
 assert s.execute(decision('read_history'))['committed_plan']['events']==[]
 s.execute(decision('revise_block',proposal(s,'Revised draft')));s.execute(decision('commit_block'))
 assert s.plan['events'][0]['action']=='Revised draft' and s.index==1 and s.draft is None
 before=copy.deepcopy(s.plan);bad=proposal(s);bad['unit_id']='pu0000'
 with pytest.raises(ValueError):s.execute(decision('propose_block',bad))
 assert s.plan==before


def test_invalid_draft_visible_for_repair_not_historical_state(tmp_path,monkeypatch):
 s=state(tmp_path,monkeypatch);p=proposal(s);p['events'][0]['source_call_ids']=[]
 s.execute(decision('propose_block',p))
 with pytest.raises(ValueError):s.execute(decision('commit_block'))
 assert s.plan['events']==[] and s.draft==p
 s.execute(decision('revise_block',proposal(s)));s.execute(decision('commit_block'))
 assert len(s.plan['events'])==1


@pytest.mark.parametrize('tool',['read_evidence','read_history','check_block','commit_block','finish','rewrite_history'])
def test_read_or_unsupported_tool_cannot_mutate_committed_plan(tmp_path,monkeypatch,tool):
 s=state(tmp_path,monkeypatch)
 with pytest.raises(ValueError):s.execute(decision(tool,proposal(s)))
 assert not s.plan['events'] and s.index==0


def test_early_finish_remains_partial_not_success(tmp_path,monkeypatch):
 def mutate(answer,req,data,n):
  if n==2:answer.clear();answer.update(decision('finish'))
 path,sid,out,calls=material(tmp_path,monkeypatch,'react',mutate)
 r=m.run_composition(path,sid,{'kind':'replay'},out,method='react')
 assert not r['full_song_succeeded'] and not (out/'plan.json').exists()
 assert r['stop_reason']=='incomplete_finish' and r['calls_used']==2


def test_total_token_budget_enforced_no_partial_accepted_as_full(tmp_path,monkeypatch):
 path,sid,out,calls=material(tmp_path,monkeypatch,'workflow')
 r=m.run_composition(path,sid,{'kind':'replay'},out,method='workflow',max_input_tokens=150)
 assert not r['full_song_succeeded'] and r['calls_used']==0
 assert not read_json(out/'partial-plan.not-success.json')['events']


def test_completed_missing_raw_cannot_dispatch_replacement(tmp_path,monkeypatch):
 path,sid,out,calls=material(tmp_path,monkeypatch,'workflow');m.run_composition(path,sid,{'kind':'replay'},out,method='workflow')
 next((out/'calls').glob('*.raw.json')).unlink()
 with pytest.raises(ValueError,match='raw missing'):m.run_composition(path,sid,{'kind':'replay'},out,method='workflow',resume=True)
 assert len(calls)==3


@pytest.mark.parametrize('method',['workflow','self_refine','react','choreography'])
def test_truncated_call_spends_budget_and_never_reuses_same_call_forever(tmp_path,monkeypatch,method):
 path,sid,out,calls=material(tmp_path,monkeypatch,method)
 provider=m.create_provider({},'planner');generate=provider.generate
 def truncated(request):
  value=generate(request)
  if len(calls)==2:value['stop_reason']='budget_truncation'
  return value
 provider.generate=truncated
 monkeypatch.setattr(m,'create_provider',lambda *a,**k:provider)
 result=m.run_composition(path,sid,{'kind':'replay'},out,method=method)
 assert result['calls_used']==len(calls)==len({c['call_id'] for c in calls})
 assert result['input_tokens']==100*len(calls) and result['output_tokens']==50*len(calls)
 if method in ('react','choreography'):
  assert result['full_song_succeeded'] and len(calls)==7
 else:
  assert not result['full_song_succeeded'] and len(calls)==2
 assert m.run_composition(path,sid,{'kind':'replay'},out,method=method,resume=True)==result


def test_infrastructure_response_aborts_following_model_calls(tmp_path,monkeypatch):
 path,sid,out,calls=material(tmp_path,monkeypatch,'react')
 provider=m.create_provider({},'planner');generate=provider.generate
 def failed(request):
  value=generate(request)
  if len(calls)==2:value['stop_reason']='infrastructure_error'
  return value
 provider.generate=failed;monkeypatch.setattr(m,'create_provider',lambda *a,**k:provider)
 result=m.run_composition(path,sid,{'kind':'replay'},out,method='react')
 assert result['calls_used']==len(calls)==2 and not result['full_song_succeeded']
 assert result['input_tokens']==200 and result['output_tokens']==100
 assert not result['complete']
 with pytest.raises(ValueError,match='infrastructure'):
  m.run_composition(path,sid,{'kind':'replay'},out,method='react',resume=True)
 assert len(calls)==2


def test_new_outline_validates_effective_public_sections_not_original_metadata(tmp_path,monkeypatch):
 path,sid,out,calls=material(tmp_path,monkeypatch,'workflow')
 original_source=m._source
 def with_sidecar(path,sid,cfg,sources):
  result=original_source(path,sid,cfg,sources)
  sections=result['song']['music']['sections']
  for section in sections:section['section_id']='public-'+section['section_id']
  for unit in result['units']['units']:
   unit['section_id']='public-'+unit['section_id']
  result['original']=copy.deepcopy(result['original'])
  result['original']['music']['sections']=[dict(section_id='original-one-section',start=0,end=result['original']['audio']['num_samples'],label='unknown',repeat_group=None)]
  return result
 monkeypatch.setattr(m,'_source',with_sidecar)
 result=m.run_composition(path,sid,{'kind':'replay'},out,method='workflow')
 assert result['full_song_succeeded'] and result['calls_used']==3
 assert all(s['section_id'].startswith('public-') for s in read_json(out/'outline.json')['sections'])


@pytest.mark.parametrize('method',['workflow','self_refine','react','choreography'])
def test_original_outline_reuse_matches_exact_request_and_no_new_outline_call(tmp_path,monkeypatch,method):
 path,sid,out,calls=material(tmp_path,monkeypatch,method)
 first=m.run_composition(path,sid,{'kind':'replay'},out,method=method);old_calls=len(calls)
 restored=tmp_path/'restored'
 result=m.run_composition(path,sid,{'kind':'replay'},restored,method=method,outline_reuse_run=out)
 assert result['full_song_succeeded'] and result['original_outline_reused']
 assert result['new_model_calls']==result['calls_used']-1
 assert result['reused_outline_calls']==1 and result['token_totals_include_reused_outline']
 assert result['reused_outline_source_costs']['input_tokens']==100
 assert len(calls)-old_calls==first['calls_used']-1
 before=len(calls)
 assert m.run_composition(path,sid,{'kind':'replay'},restored,method=method,outline_reuse_run=out,resume=True)==result
 assert len(calls)==before
 with pytest.raises(ValueError,match='request differs'):
  m.run_composition(path,sid,{'kind':'replay'},tmp_path/'wrong',method=method,seed=43,outline_reuse_run=out)
 assert len(calls)==before


def test_reused_outline_must_be_bound_by_original_results(tmp_path,monkeypatch):
 from choreobench.io import write_json,object_sha
 path,sid,out,calls=material(tmp_path,monkeypatch,'workflow')
 m.run_composition(path,sid,{'kind':'replay'},out,method='workflow')
 v=read_json(out/'results.json');v['call_records'][sid+'/four-method/0000']='0'*64
 v.pop('record_sha256');v['record_sha256']=object_sha(v);write_json(out/'results.json',v,overwrite=True)
 with pytest.raises(ValueError,match='does not bind'):
  m.run_composition(path,sid,{'kind':'replay'},tmp_path/'corrected',method='workflow',outline_reuse_run=out)
 assert len(calls)==3
