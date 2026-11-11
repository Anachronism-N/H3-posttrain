"""Shared-tool fixtures; no human or musical scores are generated."""
import copy
import json
from types import SimpleNamespace

import pytest

from choreobench import edit_agent
from choreobench.fixtures import create_demo
from choreobench.io import read_json,write_json,object_sha
from choreobench.revision_protocol import action_task


@pytest.fixture
def world(tmp_path,monkeypatch):
    made=create_demo(tmp_path/'demo')
    plan=read_json(made['source_plan']);target=plan['events'][0]['event_id']
    task=action_task(plan,target,'Change the target to a soft continuous motion, keep all other actions.')
    task['semantic_goals']=['PRIVATE_EVALUATION_MARKER']
    tp=tmp_path/'task.json';write_json(tp,task)
    state=dict(requests=[],tools=['check_patch','submit_patch'],value='Softly draw the arm through a single continuous arc.',bad=None)
    identity=dict(kind='replay',synthetic=True,fixture='not-actual-agent')
    def generate(request):
        state['requests'].append(copy.deepcopy(request))
        i=int(request['call_id'].split('/')[-1]);tool=state['tools'][min(i,len(state['tools'])-1)]
        decision=dict(tool=tool,operations=[] if tool in ['abstain','inspect_context'] else [
            dict(event_id=target,field='action',value=state['value'])],reason='Synthetic choice, not a quality rating.')
        if state['bad']=='outside':decision['operations'][0]['event_id']='missing'
        if state['bad']=='identity':identity['changed']=True
        if state['bad']=='network':raise OSError('Unknown synthetic transport outcome')
        if state['bad']=='private':decision['private_goals']=['bad']
        return dict(text=json.dumps(decision),raw={},identity=copy.deepcopy(identity),synthetic=True,stop_reason='stop',
                    costs=dict(input_tokens=10,output_tokens=10,latency_seconds=0))
    monkeypatch.setattr(edit_agent,'create_provider',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
    return made['source_plan'],tp,plan,task,state


def run(world,tmp_path,policy='generic',**kw):
    return edit_agent.run_edit_agent(world[0],world[1],dict(kind='replay'),tmp_path/policy,policy=policy,**kw)


def test_same_input_tools_permissions_budget_policy_only_differs(world,tmp_path):
    generic=run(world,tmp_path,'generic');harness=run(world,tmp_path,'choreography')
    requests=world[4]['requests'];assert len(requests)==4
    for first,second in zip(requests[:2],requests[2:]):
        assert first['messages'][-1]==second['messages'][-1]
        assert first['response_schema']==second['response_schema'] and first['seed']==second['seed']
        assert first['max_output_tokens']==second['max_output_tokens']
        assert first['messages'][0]!=second['messages'][0]
        assert 'PRIVATE_EVALUATION_MARKER' not in json.dumps(first)
    assert generic['submitted'] and harness['submitted']
    assert generic['changed_event_ids']==harness['changed_event_ids']==[world[3]['allowed_event_ids'][0]]
    assert generic['semantic_joint_success'] is None and not generic['quality_verified']


def test_check_does_not_commit_and_each_proposal_binds_initial(world,tmp_path):
    result=run(world,tmp_path)
    first=read_json(tmp_path/'generic/steps/00.json');second=read_json(tmp_path/'generic/steps/01.json')
    assert first['source_plan_sha256']==second['source_plan_sha256']==object_sha(world[2])
    assert first['observation']['status']=='structurally_valid'
    payload=json.loads(world[4]['requests'][1]['messages'][-1]['content'])
    assert payload['source_plan']['events']==edit_agent.public_plan(world[2])['events']
    after=read_json(tmp_path/'generic/plan.json')
    old={e['event_id']:e for e in world[2]['events']};new={e['event_id']:e for e in after['events']}
    assert all({k:v for k,v in e.items() if k!='action'}=={k:v for k,v in new[eid].items() if k!='action'} for eid,e in old.items())


def test_complete_resume_no_new_calls_and_missing_raw_refused(world,tmp_path):
    result=run(world,tmp_path)
    assert run(world,tmp_path,resume=True)==result and len(world[4]['requests'])==2
    raw=next((tmp_path/'generic/calls').glob('*.raw.json'));raw.unlink()
    with pytest.raises(ValueError,match='inventory changed'):run(world,tmp_path,resume=True)
    assert len(world[4]['requests'])==2


def test_abstain_and_read_until_exhaustion_do_not_claim_success(world,tmp_path):
    world[4]['tools']=['abstain'];a=run(world,tmp_path,'generic')
    assert a['stop_reason']=='abstained' and a['unchanged_original'] and a['calls_used']==1
    world[4]['tools']=['inspect_context'];b=run(world,tmp_path,'choreography',max_calls=2)
    assert b['stop_reason']=='budget_exhausted' and not b['submitted'] and b['unchanged_original']
    assert b['budget_exhaustion_is_not_success'] and b['semantic_joint_success'] is None


@pytest.mark.parametrize('bad',['outside','time'])
def test_rejected_tool_checks_do_not_expand_scope(world,tmp_path,bad):
    world[4]['tools']=['submit_patch']
    if bad=='outside':world[4]['bad']='outside'
    else:world[4]['value']='Start at 40000 samples then turn.'
    result=run(world,tmp_path,max_calls=2)
    assert result['calls_used']==2 and result['unchanged_original'] and not result['submitted']
    assert all(read_json(tmp_path/('generic/steps/%02d.json'%i))['observation']['status']=='rejected' for i in range(2))


@pytest.mark.parametrize('bad',['identity','network','private'])
def test_protocol_or_infrastructure_failure_stops_no_retry(world,tmp_path,bad):
    world[4]['bad']=bad
    result=run(world,tmp_path)
    assert not result['complete'] and result['stop_reason']=='execution_failed' and result['calls_used']==1
    assert result['unchanged_original'] and len(world[4]['requests'])==1


def test_final_artifact_tampering_blocks_resume_before_call(world,tmp_path):
    run(world,tmp_path);p=tmp_path/'generic/plan.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='artifact changed'):run(world,tmp_path,resume=True)
    assert len(world[4]['requests'])==2


def test_unknown_policy_and_invalid_budget_no_dispatch(world,tmp_path):
    with pytest.raises(ValueError):run(world,tmp_path,'unknown')
    with pytest.raises(ValueError):run(world,tmp_path,max_calls=0)
    assert world[4]['requests']==[]
