"""Synthetic fixed-prefix probes; no model quality claims or GPU calls."""
import copy
import json
from types import SimpleNamespace

import pytest

from choreobench import harness, memory_replay
from choreobench.io import file_sha, object_sha, read_json, write_json
from test_choreobench_coordination import fake_planner, config
from test_choreobench_harness import fixture


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source, song, bank = fixture(tmp_path)
    old_requests = []
    fake_planner(monkeypatch, old_requests)
    run = tmp_path / "source-run"
    result = harness.run_harness(source, bank, run, config())
    inventory = tmp_path / "inventory.json"
    write_json(inventory, dict(schema="choreobench.memory_replay_inventory.v1", cases=[dict(
        run_dir=str(run), contract_sha256=file_sha(run / "contract.json"), results_sha256=file_sha(run / "results.json"),
        song_id=song["song_id"], unit_id="pu0001")]))
    return inventory, run, old_requests[-1]


def test_projection_removes_only_all_three_historical_action_paths(prepared):
    request = prepared[2]
    original = copy.deepcopy(request)
    assert memory_replay.project_history(request, "full") == request
    projected = memory_replay.project_history(request, "metadata_only")
    old, new = [json.loads(r["messages"][-1]["content"]) for r in (request, projected)]
    assert request == original
    assert request["messages"][0] == projected["messages"][0]
    assert request["seed"] == projected["seed"] and request["max_output_tokens"] == projected["max_output_tokens"]
    assert old["current_motif_definitions"] == new["current_motif_definitions"]
    assert old["local_proposals"] == new["local_proposals"]
    assert old["boundary_context"]["source_prefix_sha256"] == new["boundary_context"]["source_prefix_sha256"]
    assert all('action' not in e for e in new['recent_committed_events'])
    assert all(part['latest_event'] is None or 'action' not in part['latest_event'] for part in new['boundary_context']['body_parts'].values())
    assert all('action' not in e for m in new['boundary_context']['motif_memory'] for e in m['recent_examples'])
    for event, original_event in zip(new['recent_committed_events'], old['recent_committed_events']):
        event['action'] = original_event['action']
    for key, part in new['boundary_context']['body_parts'].items():
        if part['latest_event'] is not None:
            part['latest_event']['action'] = old['boundary_context']['body_parts'][key]['latest_event']['action']
    for motif, original_motif in zip(new['boundary_context']['motif_memory'], old['boundary_context']['motif_memory']):
        for event, original_event in zip(motif['recent_examples'], original_motif['recent_examples']):
            event['action'] = original_event['action']
    assert new == old


def provider(monkeypatch, behavior=None):
    requests = []
    identity = dict(kind='replay', synthetic=True, fixture='not-a-model')
    def generate(request):
        requests.append(copy.deepcopy(request))
        payload = json.loads(request['messages'][-1]['content']);unit = payload['planning_unit']
        answer = dict(unit_id=unit['unit_id'],events=[dict(start=unit['start'],end=unit['end'],track='body',
            action='Synthetic connected phrase.',motif_id=payload['current_section_design']['motif_ids'][0],
            source_call_ids=[payload['local_proposals'][0]['call_id']])])
        if behavior=='outside':answer['events'][0]['end']+=1
        if behavior=='network':raise OSError('Synthetic unknown infrastructure outcome')
        return dict(text=json.dumps(answer), raw={}, identity=identity,synthetic=True,stop_reason='stop',
                    costs=dict(input_tokens=10,output_tokens=5,latency_seconds=0))
    monkeypatch.setattr(memory_replay,'create_provider',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
    return requests


def test_replay_fixed_prefix_two_calls_and_resume_no_new_dispatch(prepared, tmp_path, monkeypatch):
    requests = provider(monkeypatch)
    output = tmp_path/'replay'
    result=memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output)
    assert result['complete'] and result['calls_used']==2
    assert all(r['status']=='valid' for r in result['rows'])
    assert not result['quality_verified'] and result['new_audio_calls']==result['new_outline_calls']==0
    assert requests[0]['seed']==requests[1]['seed']==prepared[2]['seed']
    assert requests[0]['messages'][0]==requests[1]['messages'][0]
    assert result['rows'][0]['historical_prefix_sha256']==result['rows'][1]['historical_prefix_sha256']
    assert memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output,resume=True)==result
    assert len(requests)==2
    with pytest.raises(FileExistsError):memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output)


@pytest.mark.parametrize('path', ['contract.json','results.json'])
def test_modified_source_rejected_before_provider(prepared,tmp_path,monkeypatch,path):
    source=prepared[1]/path;source.write_text(source.read_text()+' ')
    monkeypatch.setattr(memory_replay,'create_provider',lambda *a,**k:pytest.fail('provider before source validation'))
    with pytest.raises(ValueError,match='source bytes differ'):
        memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),tmp_path/'replay')


def test_first_block_and_duplicate_case_rejected(prepared,tmp_path,monkeypatch):
    inventory=read_json(prepared[0]);inventory['cases'][0]['unit_id']='pu0000'
    write_json(prepared[0],inventory,overwrite=True)
    monkeypatch.setattr(memory_replay,'create_provider',lambda *a,**k:pytest.fail('no provider'))
    with pytest.raises(ValueError,match='nonempty original prefix'):
        memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),tmp_path/'replay')
    inventory['cases'][0]['unit_id']='pu0001';inventory['cases']*=2
    write_json(prepared[0],inventory,overwrite=True)
    with pytest.raises(ValueError,match='one preselected probe per song'):
        memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),tmp_path/'replay')


def test_invalid_generation_retained_no_repair_or_retry(prepared,tmp_path,monkeypatch):
    requests=provider(monkeypatch,'outside')
    result=memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),tmp_path/'replay')
    assert result['complete'] and len(requests)==2
    assert all(r['status']=='failed' and r['retained_in_denominator'] for r in result['rows'])


def test_unknown_infrastructure_stops_subsequent_calls(prepared,tmp_path,monkeypatch):
    requests=provider(monkeypatch,'network')
    result=memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),tmp_path/'replay')
    assert not result['complete'] and len(requests)==1 and result['unknown_outcomes_count']==1
    assert result['rows'][0]['status']=='failed'


def test_projection_rejects_no_history_and_unknown_mode(prepared):
    with pytest.raises(ValueError):memory_replay.project_history(prepared[2],'none')
    request=copy.deepcopy(prepared[2]);payload=json.loads(request['messages'][-1]['content']);payload['history_available']=False
    request['messages'][-1]['content']=json.dumps(payload)
    with pytest.raises(ValueError,match='supplied history'):
        memory_replay.project_history(request,'metadata_only')
