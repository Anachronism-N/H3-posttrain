"""Synthetic fixed-prefix probes; no model quality claims or GPU calls."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from choreobench import harness, memory_replay
from choreobench.baseline import _sealed
from choreobench.io import file_sha, object_sha, read_json, write_json
from test_choreobench_coordination import fake_planner, config, sidecar
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
        if behavior in ('identity_first', 'identity_last') and len(requests)==(1 if behavior=='identity_first' else 2):
            identity['fixture']='changed-after-dispatch'
        returned_identity = dict(identity, fixture='different-reported-provider') if behavior=='reported_identity' else identity
        return dict(text=json.dumps(answer), raw={}, identity=returned_identity,synthetic=True,stop_reason='stop',
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


@pytest.mark.parametrize('behavior,expected_calls', [('identity_first',1),('identity_last',2),('reported_identity',1)])
def test_identity_drift_stops_dispatch_and_cannot_mutate_frozen_contract(prepared,tmp_path,monkeypatch,behavior,expected_calls):
    requests=provider(monkeypatch,behavior)
    output=tmp_path/'replay'
    result=memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output)
    assert not result['complete'] and len(requests)==expected_calls
    assert result['rows'][-1]['status']=='failed' and 'identity changed during dispatch' in result['rows'][-1]['error']
    contract=read_json(output/'contract.json')
    assert contract['provider_identity']['fixture']=='not-a-model'
    assert result['contract_sha256']==object_sha(contract)
    assert not (output/prepared[2]['call_id'].split('/')[0]/('full' if expected_calls==1 else 'metadata_only')/'plan.json').exists()


@pytest.mark.parametrize('damage',['call_pair','calls_directory','record','prefix','events','plan','artifact_bytes','statistics'])
def test_completed_resume_checks_inventory_and_artifacts_before_any_dispatch(prepared,tmp_path,monkeypatch,damage):
    requests=provider(monkeypatch);output=tmp_path/'replay'
    result=memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output)
    first=result['rows'][0];directory=output/first['song_id']/'full'
    if damage=='call_pair':
        call_id=next(key for key in result['call_records'] if key.endswith('/full'))
        name=hashlib.sha256(call_id.encode()).hexdigest()
        for suffix in ('.started.json','.raw.json'):(output/'calls'/(name+suffix)).unlink()
    elif damage=='calls_directory':
        for path in (output/'calls').iterdir():path.unlink()
        (output/'calls').rmdir()
    elif damage=='record':
        path=next((output/'calls').glob('*.raw.json'));value=read_json(path)
        value['result']['costs']['output_tokens']+=1;value.pop('record_sha256')
        write_json(path,_sealed(value),overwrite=True)
    elif damage in ('prefix','events','plan'):
        (directory/(damage+'.json')).unlink()
    elif damage=='artifact_bytes':
        path=directory/'historical_events.json';path.write_text(path.read_text()+' ')
    else:
        result['rows'][0]['action_characters']+=1;result.pop('record_sha256')
        write_json(output/'results.json',_sealed(result),overwrite=True)
    count=len(requests);before=file_sha(output/'results.json')
    with pytest.raises(ValueError):
        memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output,resume=True)
    assert len(requests)==count and file_sha(output/'results.json')==before
    if damage=='calls_directory':assert not (output/'calls').exists()
    if damage in ('prefix','events','plan'):assert not (directory/(damage+'.json')).exists()


def change_source_request(prepared,change):
    inventory=read_json(prepared[0]);entry=inventory['cases'][0]
    call_id=entry['song_id']+'/coordinate/'+entry['unit_id'];name=hashlib.sha256(call_id.encode()).hexdigest()
    start_path=prepared[1]/'calls'/(name+'.started.json');raw_path=prepared[1]/'calls'/(name+'.raw.json')
    start=read_json(start_path);change(start['request']);start['request_sha256']=object_sha(start['request'])
    start.pop('record_sha256');write_json(start_path,_sealed(start),overwrite=True)
    raw=read_json(raw_path);raw['started_sha256']=read_json(start_path)['record_sha256']
    raw.pop('record_sha256');write_json(raw_path,_sealed(raw),overwrite=True)
    result_path=prepared[1]/'results.json';result=read_json(result_path)
    result['call_records'][call_id]=read_json(raw_path)['record_sha256']
    result.pop('record_sha256');write_json(result_path,_sealed(result),overwrite=True)
    entry['results_sha256']=file_sha(result_path);write_json(prepared[0],inventory,overwrite=True)


@pytest.mark.parametrize('field',['semantic_goals','song_id','sample_rate','num_samples','brief','position_in_section',
                                 'unit_clock','music_context','history_available','outline_available','top_level','response_schema','boolean_type'])
def test_resealed_source_with_private_or_inconsistent_public_inputs_rejected_before_provider(prepared,tmp_path,monkeypatch,field):
    def change(request):
        if field=='top_level':request['semantic_goals']=['Private evaluator instruction'];return
        if field=='response_schema':request.setdefault('response_schema',{})['description']='Private evaluator instruction';return
        value=json.loads(request['messages'][-1]['content'])
        if field=='semantic_goals':value[field]=['Private evaluator instruction']
        elif field=='boolean_type':value['outline_available']=1
        elif field in ('sample_rate','num_samples'):value[field]+=1
        elif field in ('history_available','outline_available'):value[field]=False
        elif field=='position_in_section':value[field]['index']+=1
        elif field=='unit_clock':value[field]['duration_seconds']+=1
        elif field=='music_context':value[field]['beats']=[1]
        else:value[field]='Different public input'
        request['messages'][-1]['content']=json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)
    change_source_request(prepared,change)
    monkeypatch.setattr(memory_replay,'create_provider',lambda *a,**k:pytest.fail('provider created before source rejection'))
    with pytest.raises(ValueError,match='Original request does not match'):
        memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),tmp_path/'replay')
    assert not (tmp_path/'replay').exists()


def test_historical_system_prompt_bytes_are_preserved_without_current_prompt_equality(prepared,tmp_path,monkeypatch):
    change_source_request(prepared,lambda request:request['messages'][0].update(content='Historical frozen system instruction.'))
    requests=provider(monkeypatch)
    result=memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),tmp_path/'replay')
    assert result['complete'] and all(request['messages'][0]['content']=='Historical frozen system instruction.' for request in requests)


def test_audit_only_is_read_only_and_does_not_load_provider_or_current_code_hashes(prepared,tmp_path,monkeypatch,capsys):
    provider(monkeypatch);output=tmp_path/'replay'
    result=memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output)
    before={str(path.relative_to(output)):file_sha(path) for path in output.rglob('*') if path.is_file()}
    historical_code=read_json(output/'contract.json')['code_sha256']
    monkeypatch.setattr(memory_replay,'create_provider',lambda *a,**k:pytest.fail('read-only audit created provider'))
    original_sha=memory_replay.file_sha
    def changed_code_sha(path):
        assert not str(path).endswith('.py'),'read-only audit must not require current code hashes'
        return original_sha(path)
    monkeypatch.setattr(memory_replay,'file_sha',changed_code_sha)
    audit=memory_replay.audit_completed_replay(output)
    assert audit['audit_complete'] and audit['new_model_calls']==0 and audit['calls_used']==result['calls_used']
    assert audit['historical_code_sha256']==historical_code and audit['valid_rows']==2
    assert audit['result_file_sha256']==file_sha(output/'results.json')
    memory_replay.main(['--audit-only','--out-dir',str(output)])
    assert json.loads(capsys.readouterr().out)['audit_complete']
    assert before=={str(path.relative_to(output)):file_sha(path) for path in output.rglob('*') if path.is_file()}


def test_audit_missing_calls_does_not_create_directory(prepared,tmp_path,monkeypatch):
    provider(monkeypatch);output=tmp_path/'replay'
    memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output)
    for path in (output/'calls').iterdir():path.unlink()
    (output/'calls').rmdir()
    monkeypatch.setattr(memory_replay,'create_provider',lambda *a,**k:pytest.fail('audit created provider'))
    with pytest.raises(ValueError,match='journal'):
        memory_replay.audit_completed_replay(output)
    assert not (output/'calls').exists()


def test_completed_invalid_generation_remains_auditable_without_retry(prepared,tmp_path,monkeypatch):
    requests=provider(monkeypatch,'outside');output=tmp_path/'replay'
    result=memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output)
    assert result['complete'] and len(requests)==2
    audit=memory_replay.audit_completed_replay(output)
    assert audit['failed_rows']==2 and audit['valid_rows']==0
    assert memory_replay.run_memory_replay(prepared[0],dict(kind='replay'),output,resume=True)==result
    assert len(requests)==2


def test_public_structure_source_replay_and_read_only_audit_preserve_effective_music(tmp_path,monkeypatch):
    source,song,bank=fixture(tmp_path)
    path,structure=sidecar(tmp_path,source,song)
    old_requests=[];fake_planner(monkeypatch,old_requests)
    run=tmp_path/'source-run'
    harness.run_harness(source,bank,run,config(public_structure_path=str(path),unit_config={'fallback_seconds':6}))
    inventory=tmp_path/'inventory.json'
    write_json(inventory,dict(schema='choreobench.memory_replay_inventory.v1',cases=[dict(
        run_dir=str(run),contract_sha256=file_sha(run/'contract.json'),results_sha256=file_sha(run/'results.json'),
        song_id=song['song_id'],unit_id='pu0001')]))
    requests=provider(monkeypatch);output=tmp_path/'replay'
    result=memory_replay.run_memory_replay(inventory,dict(kind='replay'),output)
    assert result['complete'] and len(requests)==2
    for request in requests:
        payload=json.loads(request['messages'][-1]['content'])
        assert payload['music_context']['source']==structure['songs'][0]['music']['source']
        assert payload['planning_unit']['section_id']=='new'
    monkeypatch.setattr(memory_replay,'create_provider',lambda *a,**k:pytest.fail('audit created provider'))
    assert memory_replay.audit_completed_replay(output)['valid_rows']==2
