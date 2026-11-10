"""Synthetic bounded content probe; never physical or musical quality evidence."""
import copy
import json
from types import SimpleNamespace

import pytest

from choreobench import harness, memory_replay, duration_content
from choreobench.io import file_sha, read_json, write_json
from test_choreobench_coordination import fake_planner, config
from test_choreobench_harness import fixture


@pytest.fixture
def source(tmp_path, monkeypatch):
    dataset, song, bank = fixture(tmp_path)
    def boundaries(answer, request, payload):
        if request['response_kind'] == 'coordination_unit':
            event = answer['events'][0]; event.pop('start'); event.pop('end')
            event.update(payload['event_clock']['permitted_schedules'][0][0])
    fake_planner(monkeypatch, [], mutate=boundaries)
    run=tmp_path/'source'
    harness.run_harness(dataset,bank,run,config(event_time_format='boundary_ids',event_schedule_policy='balanced_full_unit',
        event_grid_seconds=.25,min_event_seconds=.5,structured_unit_output=True))
    inventory=tmp_path/'inventory.json'
    write_json(inventory,dict(schema='choreobench.memory_replay_inventory.v1',cases=[dict(run_dir=str(run),
        contract_sha256=file_sha(run/'contract.json'),results_sha256=file_sha(run/'results.json'),song_id=song['song_id'],unit_id='pu0001')]))
    requests=[]
    identity=dict(kind='replay',synthetic=True,fixture='content-test')
    behavior={'bad':None}
    def generate(request):
        requests.append(copy.deepcopy(request));payload=json.loads(request['messages'][-1]['content'])
        event=dict(payload['event_clock']['permitted_schedules'][0][0],track='body',
            action='Shift weight smoothly as both arms sweep together, then settle into a controlled stance.',
            motif_id=payload['current_section_design']['motif_ids'][0],source_call_ids=[payload['local_proposals'][0]['call_id']])
        if behavior['bad']=='words':event['action']='move '*33
        if behavior['bad']=='characters':event['action']='a'*241
        if behavior['bad']=='schedule':event['end_boundary']-=1
        if behavior['bad']=='time':event['action']='Wait for 1 second then raise the arm.'
        if behavior['bad']=='identity':identity['mutated']=True
        events=[dict(event, **slot) for slot in payload['event_clock']['permitted_schedules'][0]]
        if behavior['bad']=='schedule':events[0]['end_boundary']-=1
        return dict(text=json.dumps(dict(unit_id=payload['planning_unit']['unit_id'],events=events)),raw={},
            identity=copy.deepcopy(identity),synthetic=True,stop_reason='stop',costs=dict(input_tokens=20,output_tokens=10,latency_seconds=0))
    monkeypatch.setattr(memory_replay,'create_provider',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
    old=tmp_path/'old'
    result=memory_replay.run_memory_replay(inventory,dict(kind='replay'),old)
    assert all(r['status']=='valid' for r in result['rows'])
    requests.clear()
    return inventory,old,requests,behavior


def test_only_duration_instruction_differs_under_common_schema_payload_and_seed(source):
    inventory,old,_,_=source
    sources={};cases,requests=duration_content.prepare_duration_requests(inventory,sources,old)
    assert len(cases)==1 and len(requests)==2
    b,d=[r for _,_,r in requests]
    assert b['seed']==d['seed'] and b['response_schema']==d['response_schema']
    assert b['messages'][-1]==d['messages'][-1]
    assert d['messages'][0]['content']==b['messages'][0]['content']+duration_content.DURATION
    assert cases[0]['clock']['permitted_schedules']==[json.loads(b['messages'][-1]['content'])['content_contract']['fixed_schedule']]
    assert cases[0]['content_limits']==duration_content.LIMITS
    assert str(old/'results.json') in sources


def test_complete_resume_and_readonly_audit_never_resample(source,tmp_path):
    inventory,old,requests,_=source
    out=tmp_path/'new'
    result=memory_replay.run_memory_replay(inventory,dict(kind='replay'),out,duration_source_run=old)
    assert result['complete'] and result['calls_used']==2
    assert all(r['status']=='valid' and r['content_lengths'][0]['target_word_range_met'] for r in result['rows'])
    assert memory_replay.run_memory_replay(inventory,dict(kind='replay'),out,resume=True,duration_source_run=old)==result
    assert len(requests)==2
    assert memory_replay.audit_completed_replay(out)['valid_rows']==2
    assert len(requests)==2
    contract=read_json(out/'contract.json')
    assert contract['modes']==['brief','duration'] and contract['fixed_schedule_content_probe']


@pytest.mark.parametrize('bad',['words','characters','schedule','time'])
def test_shared_hard_limits_and_frozen_slots_do_not_truncate_or_retry(source,tmp_path,bad):
    inventory,old,requests,behavior=source;behavior['bad']=bad
    result=memory_replay.run_memory_replay(inventory,dict(kind='replay'),tmp_path/'new',duration_source_run=old)
    assert result['complete'] and len(requests)==2
    assert all(r['status']=='failed' for r in result['rows'])
    assert memory_replay.audit_completed_replay(tmp_path/'new')['failed_rows']==2


def test_changed_completed_content_artifact_blocks_before_dispatch(source,tmp_path):
    inventory,old,requests,_=source;out=tmp_path/'new'
    result=memory_replay.run_memory_replay(inventory,dict(kind='replay'),out,duration_source_run=old)
    path=out/result['rows'][0]['song_id']/'brief/events.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='artifact bytes changed'):
        memory_replay.run_memory_replay(inventory,dict(kind='replay'),out,resume=True,duration_source_run=old)
    assert len(requests)==2


def test_provider_identity_drift_stops_next_condition(source,tmp_path):
    inventory,old,requests,behavior=source;behavior['bad']='identity'
    result=memory_replay.run_memory_replay(inventory,dict(kind='replay'),tmp_path/'new',duration_source_run=old)
    assert not result['complete'] and len(requests)==1


def test_word_target_is_reported_not_fabricated_equal_length():
    events=[dict(event_id='e',action='Step softly together.')]
    statistics=duration_content.content_statistics(events,duration_content.LIMITS)
    assert statistics[0]['words']==3 and not statistics[0]['target_word_range_met']


def test_duration_cli(source,tmp_path,capsys):
    inventory,old,requests,_=source
    provider_path=tmp_path/'provider.json';write_json(provider_path,dict(kind='replay'))
    duration_content.main(['--inventory',str(inventory),'--source-run',str(old),'--provider',str(provider_path),'--out-dir',str(tmp_path/'new')])
    assert json.loads(capsys.readouterr().out)['calls_used']==2 and len(requests)==2



def test_adaptive_slots_and_boundary_history_single_event_copy_path(source,tmp_path):
    inventory,old,requests,_=source
    cases,paired=duration_content.prepare_duration_requests(inventory,{},old,adaptive_boundary=True)
    assert len(cases[0]['fixed_intervals'])==3
    for interval in cases[0]['fixed_intervals']:
        assert interval['end']-interval['start']>=32000
    metadata,boundary=[r for _,_,r in paired]
    assert metadata['messages'][0]==boundary['messages'][0] and metadata['response_schema']==boundary['response_schema']
    m,b=[json.loads(r['messages'][-1]['content']) for r in (metadata,boundary)]
    assert m['boundary_actions']==[] and len(b['boundary_actions'])==1
    assert 'action' in b['boundary_actions'][0] and b['boundary_actions'][0]['exit']=={}
    b['boundary_actions']=[]
    assert m==b
    result=memory_replay.run_memory_replay(inventory,dict(kind='replay'),tmp_path/'new',duration_source_run=old,adaptive_boundary=True)
    assert result['complete'] and all(r['status']=='valid' for r in result['rows'])
    assert memory_replay.audit_completed_replay(tmp_path/'new')['valid_rows']==2
    assert len(requests)==2
