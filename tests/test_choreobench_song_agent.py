"""Synthetic tool trajectories do not establish whole-song quality."""
import copy
import json
from types import SimpleNamespace
import pytest

from choreobench import song_agent, harness
from choreobench.io import read_json, write_json, object_sha
from test_choreobench_coordination import config, fake_planner
from test_choreobench_harness import fixture


def source(tmp_path,monkeypatch):
    dataset,song,bank=fixture(tmp_path)
    fake_planner(monkeypatch,[])
    out=tmp_path/'source'
    result=harness.run_harness(dataset,bank,out,config(structured_unit_output=True))
    assert result['all_songs_succeeded']
    return out,song['song_id']


def fake_agent(monkeypatch,calls,tools=None,mutate=None,infrastructure=False):
    identity=dict(kind='replay',synthetic=True,fixture='song-agent')
    def generate(request):
        calls.append(copy.deepcopy(request))
        if infrastructure:raise OSError('Unknown network outcome')
        payload=json.loads(request['messages'][-1]['content'])
        tool=tools[len(calls)-1] if tools else 'finish' if payload.get('all_units_committed') else 'commit_block'
        proposal=None
        if tool in ('commit_block','check_block'):
            unit=payload['planning_unit']
            proposal=dict(unit_id=unit['unit_id'],events=[dict(start=unit['start'],end=unit['end'],track='body',
                action='A continuous phrase for '+unit['unit_id'],motif_id='m1',
                source_call_ids=[payload['local_proposals'][0]['call_id']])])
        answer=dict(tool=tool,proposal=proposal,reason='Synthetic tool fixture only.')
        if mutate:mutate(answer,payload,len(calls))
        return dict(text=json.dumps(answer),raw={},identity=identity,synthetic=True,stop_reason='stop',
            costs=dict(input_tokens=100,output_tokens=50,latency_seconds=0))
    monkeypatch.setattr(song_agent,'create_provider',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))


def run(tmp_path,monkeypatch,tools=None,mutate=None,**kwargs):
    root,sid=source(tmp_path,monkeypatch);calls=[]
    fake_agent(monkeypatch,calls,tools,mutate)
    out=tmp_path/'agent'
    result=song_agent.run_song_agent(root,sid,dict(kind='replay'),out,max_calls=kwargs.pop('max_calls',8),**kwargs)
    return result,calls,root,sid,out


def test_causal_empty_prefix_then_commits_and_finish(tmp_path,monkeypatch):
    result,calls,root,sid,out=run(tmp_path,monkeypatch)
    assert result['full_song_succeeded'] and result['committed_units']==2 and result['calls_used']==3
    first,second=[json.loads(r['messages'][-1]['content']) for r in calls[:2]]
    assert first['committed_plan']['events']==[]
    assert len(second['committed_plan']['events'])==1
    assert second['committed_plan']['events'][0]['action']=='A continuous phrase for pu0000'
    assert not read_json(out/'contract.json')['historical_prefix_reused']
    assert result['new_audio_calls']==0 and result['human_reviews']==0 and not result['quality_verified']
    assert song_agent.run_song_agent(root,sid,dict(kind='replay'),out,max_calls=8,resume=True)==result
    assert len(calls)==3


def test_inspect_and_check_do_not_commit_or_borrow_original_prefix(tmp_path,monkeypatch):
    result,calls,_,_,out=run(tmp_path,monkeypatch,['inspect_context','check_block','commit_block','commit_block','finish'])
    assert result['full_song_succeeded'] and result['calls_used']==5
    assert all(json.loads(r['messages'][-1]['content'])['committed_plan']['events']==[] for r in calls[:3])
    check=read_json(out/'steps/001.json')
    assert check['committed_units_before']==check['committed_units_after']==0
    assert 'all_local_proposals' in json.loads(calls[1]['messages'][-1]['content'])['tool_observations'][0]


def test_early_finish_is_incomplete_not_full_song_success(tmp_path,monkeypatch):
    result,_,_,_,out=run(tmp_path,monkeypatch,['finish'])
    assert result['complete'] and not result['full_song_succeeded'] and result['stop_reason']=='incomplete_finish'
    assert not (out/'plan.json').exists() and (out/'partial-plan.not-success.json').exists()


def test_budget_exhaustion_with_all_units_but_no_finish_is_not_success(tmp_path,monkeypatch):
    result,_,_,_,out=run(tmp_path,monkeypatch,['check_block','commit_block','commit_block'],max_calls=3)
    assert result['committed_units']==2 and result['stop_reason']=='budget_exhausted' and not result['full_song_succeeded']
    assert not (out/'plan.json').exists()


def test_rejected_proposal_is_visible_but_never_future_history(tmp_path,monkeypatch):
    def mutate(answer,payload,n):
        if n==1:answer['proposal']['events'][0]['end']=payload['num_samples']+1
    result,calls,_,_,out=run(tmp_path,monkeypatch,mutate=mutate)
    assert result['full_song_succeeded'] and result['calls_used']==4
    second=json.loads(calls[1]['messages'][-1]['content'])
    assert second['committed_plan']['events']==[] and second['tool_observations'][0]['status']=='rejected'
    assert read_json(out/'steps/000.json')['committed_units_after']==0


def test_missing_completed_raw_cannot_trigger_replacement_call(tmp_path,monkeypatch):
    result,calls,root,sid,out=run(tmp_path,monkeypatch)
    next((out/'calls').glob('*.raw.json')).unlink()
    with pytest.raises(ValueError,match='missing/changed'):
        song_agent.run_song_agent(root,sid,dict(kind='replay'),out,max_calls=8,resume=True)
    assert len(calls)==3


def test_infrastructure_failure_aborts_and_never_automatically_retries(tmp_path,monkeypatch):
    root,sid=source(tmp_path,monkeypatch);calls=[];fake_agent(monkeypatch,calls,infrastructure=True)
    out=tmp_path/'agent'
    result=song_agent.run_song_agent(root,sid,dict(kind='replay'),out,max_calls=8)
    assert not result['complete'] and not result['full_song_succeeded'] and result['calls_used']==1
    with pytest.raises(ValueError,match='Infrastructure-incomplete'):
        song_agent.run_song_agent(root,sid,dict(kind='replay'),out,max_calls=8,resume=True)
    assert len(calls)==1


def test_same_inputs_tools_and_schema_policy_only_difference(tmp_path,monkeypatch):
    root,sid=source(tmp_path,monkeypatch);sources={};data=song_agent.load_song_source(root,sid,sources)
    plan=song_agent._empty_plan(data['original']);plan['motifs']=copy.deepcopy(data['outline']['motifs'])
    cfg=dict(seed=7,max_calls=8,max_output_tokens=512,max_request_characters=200000,call_index=0)
    a=song_agent._request(data,plan,0,[],cfg,'generic');b=song_agent._request(data,plan,0,[],cfg,'choreography')
    assert a['messages'][1]==b['messages'][1] and a['response_schema']==b['response_schema']
    assert a['messages'][0]['content']!=b['messages'][0]['content']
    assert song_agent.TOOLS in a['messages'][0]['content'] and song_agent.TOOLS in b['messages'][0]['content']
    assert all(a[k]==b[k] for k in a if k!='messages')


@pytest.mark.parametrize('change',['unknown_tool','wrong_unit','missing_quote','read_with_proposal','foreign_motif'])
def test_tool_permissions_checked_before_commit(tmp_path,monkeypatch,change):
    def mutate(answer,payload,n):
        if n!=1:return
        if change=='unknown_tool':answer['tool']='rewrite_history'
        elif change=='wrong_unit':answer['proposal']['unit_id']='pu9999'
        elif change=='missing_quote':answer['proposal']['events'][0]['source_call_ids']=[]
        elif change=='foreign_motif':answer['proposal']['events'][0]['motif_id']='unseen'
        else:answer['tool']='inspect_context'
    result,calls,_,_,out=run(tmp_path,monkeypatch,mutate=mutate)
    assert result['full_song_succeeded'] and result['calls_used']==4
    assert read_json(out/'steps/000.json')['committed_units_after']==0
    assert json.loads(calls[1]['messages'][-1]['content'])['committed_plan']['events']==[]


def test_final_plan_tampering_blocks_completed_resume(tmp_path,monkeypatch):
    result,calls,root,sid,out=run(tmp_path,monkeypatch)
    p=out/'plan.json';p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='artifact changed'):
        song_agent.run_song_agent(root,sid,dict(kind='replay'),out,max_calls=8,resume=True)
    assert len(calls)==3


def test_metadata_only_hides_all_prior_action_paths_but_keeps_authoritative_plan(tmp_path,monkeypatch):
    result,calls,root,sid,out=run(tmp_path,monkeypatch,history_mode='metadata_only')
    assert result['full_song_succeeded'] and result['history_mode']=='metadata_only'
    payload=json.loads(calls[1]['messages'][-1]['content'])
    assert len(payload['committed_plan']['events'])==1
    assert all('action' not in e for e in payload['committed_plan']['events'])
    assert all('action' not in e for e in payload['recent_committed_events'])
    for part in payload['boundary_context']['body_parts'].values():
        if part['latest_event'] is not None:assert 'action' not in part['latest_event']
    for motif in payload['boundary_context']['motif_memory']:
        assert all('action' not in e for e in motif['recent_examples'])
    prefix=read_json(out/'prefixes/000.json')
    assert prefix['events'][0]['action']=='A continuous phrase for pu0000'
    assert read_json(out/'plan.json')['events'][0]['action']==prefix['events'][0]['action']
    assert 'prior action text withheld' in payload['history_representation']
    assert song_agent.run_song_agent(root,sid,dict(kind='replay'),out,max_calls=8,history_mode='metadata_only',resume=True)==result
    assert len(calls)==3


def test_metadata_projection_keeps_current_evidence_outline_clock_and_seed_exact(tmp_path,monkeypatch):
    root,sid=source(tmp_path,monkeypatch);sources={};data=song_agent.load_song_source(root,sid,sources)
    plan=song_agent._empty_plan(data['original']);plan['motifs']=copy.deepcopy(data['outline']['motifs'])
    cfg=dict(seed=7,max_calls=8,max_output_tokens=512,max_request_characters=200000,call_index=0)
    initial=song_agent._request(data,plan,0,[],cfg,'generic')
    public=json.loads(initial['messages'][-1]['content']);unit=public['planning_unit']
    decision=dict(tool='commit_block',reason='Software fixture',proposal=dict(unit_id=unit['unit_id'],events=[dict(
        start=unit['start'],end=unit['end'],track='body',action='PRIVATE_OLD_ACTION_NOT_A_SOURCE_PHRASE',motif_id='m1',
        source_call_ids=[public['local_proposals'][0]['call_id']])]))
    observation,plan,index,_=song_agent._execute(data,plan,0,decision)
    original=copy.deepcopy(plan)
    full=song_agent._request(data,plan,index,[observation],dict(cfg,call_index=1,history_mode='full'),'generic')
    meta=song_agent._request(data,plan,index,[observation],dict(cfg,call_index=1,history_mode='metadata_only'),'generic')
    a,b=[json.loads(r['messages'][-1]['content']) for r in (full,meta)]
    for field in ('local_proposals','whole_song_design','current_section_design','current_motif_definitions','planning_unit',
                  'music_context','event_clock','output_contract','tool_observations','remaining_model_calls'):
        assert a[field]==b[field]
    assert 'PRIVATE_OLD_ACTION_NOT_A_SOURCE_PHRASE' in full['messages'][-1]['content']
    assert 'PRIVATE_OLD_ACTION_NOT_A_SOURCE_PHRASE' not in meta['messages'][-1]['content']
    assert plan==original and full['response_schema']==meta['response_schema']
    assert all(full[k]==meta[k] for k in full if k!='messages')


def test_metadata_mode_change_cannot_resume_original_full_run(tmp_path,monkeypatch):
    result,calls,root,sid,out=run(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='artifact changed'):
        song_agent.run_song_agent(root,sid,dict(kind='replay'),out,max_calls=8,history_mode='metadata_only',resume=True)
    assert len(calls)==3


def test_main_cli_bound_run_and_resume(tmp_path,monkeypatch):
    from choreobench.cli import main
    root,sid=source(tmp_path,monkeypatch);calls=[];fake_agent(monkeypatch,calls)
    spec=tmp_path/'provider.json';write_json(spec,dict(kind='replay'))
    out=tmp_path/'agent'
    args=['run-song-agent','--source-run',str(root),'--song-id',sid,'--provider',str(spec),'--out-dir',str(out),'--max-calls','8']
    assert main(args)==0 and read_json(out/'results.json')['full_song_succeeded']
    assert main(args+['--resume'])==0 and len(calls)==3
    assert main(args)==2 and len(calls)==3


@pytest.mark.parametrize('change',['artifact','config','foreign_song'])
def test_source_or_contract_drift_rejected_without_dispatch(tmp_path,monkeypatch,change):
    result,calls,root,sid,out=run(tmp_path,monkeypatch)
    if change=='artifact':
        p=next((root/'options').glob('*.json'));p.write_bytes(p.read_bytes()+b' ')
    elif change=='config':
        p=out/'contract.json';doc=read_json(p);doc['config']['seed']=8;write_json(p,doc,overwrite=True)
    else:sid='unknown'
    with pytest.raises(ValueError):song_agent.run_song_agent(root,sid,dict(kind='replay'),out,max_calls=8,resume=True)
    assert len(calls)==3
