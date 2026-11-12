"""Synthetic catalog tests never imply audio or choreographic quality."""
import copy
from types import SimpleNamespace
import pytest
from choreobench import untimed_catalog as catalog
from choreobench.io import read_json,write_json,file_sha,object_sha
from test_choreobench_baseline import dataset_file


def material(tmp_path,monkeypatch,fail=None):
    dataset,song=dataset_file(tmp_path,seconds=12)
    provider=tmp_path/'provider.json';write_json(provider,dict(kind='native',family='omni_captioner',model='fake-model',task_prompt='One possible untimed phrase.'))
    identity=dict(kind='replay',synthetic=True,fixture='untimed-catalog');calls=[]
    def generate(request):
        calls.append(copy.deepcopy(request))
        return dict(text='A sustained side step with concurrent arm sweep.',identity=identity,synthetic=True,raw={},
            stop_reason='budget_truncation' if fail==len(calls) else 'eos',costs=dict(input_tokens=12,output_tokens=10,latency_seconds=0))
    factory=lambda *a,**k:SimpleNamespace(identity=identity,generate=generate)
    return dataset,song,provider,factory,calls


def test_whole_window_k2_inventory_raw_binding_and_no_timestamp_split(tmp_path,monkeypatch):
    dataset,song,provider,factory,calls=material(tmp_path,monkeypatch)
    out=tmp_path/'catalog';result=catalog.run_catalog(dataset,provider,out,max_calls=10,provider_factory=factory)
    assert result['expected_calls']==result['calls_used']==6 and result['complete_proposals']==6
    assert all(e['text']=='A sustained side step with concurrent arm sweep.' for e in result['entries'])
    assert all(r['audio']['num_samples']==81000 and r['audio']['parent_audio_sha256']==song['audio']['sha256'] for r in calls)
    assert all(set(r)=={'call_id','seed','max_output_tokens','response_kind','audio'} for r in calls)
    assert not result['quality_verified'] and not result['formal_E3_completed']
    result2=catalog.run_catalog(dataset,provider,out,max_calls=10,resume=True,provider_factory=lambda *a,**k:pytest.fail('Completed resume loaded a model'))
    assert result==result2 and len(calls)==6
    sources={};original,options,synthetic,digest=catalog.load_catalog_song(out/'results.json',song['song_id'],sources)
    assert original==song and len(options['windows'])==3 and all(len(w['candidates'])==2 for w in options['windows'])
    assert digest==result['record_sha256'] and synthetic and len(calls)==6


def test_failed_candidate_retained_without_replacement_or_fake_action(tmp_path,monkeypatch):
    dataset,song,provider,factory,calls=material(tmp_path,monkeypatch,fail=2)
    result=catalog.run_catalog(dataset,provider,tmp_path/'catalog',max_calls=10,provider_factory=factory)
    assert result['calls_used']==6 and result['failed_proposals']==1 and result['complete_proposals']==5
    assert result['entries'][1]['status']=='failed' and result['entries'][1]['text']
    with pytest.raises(ValueError,match='failed catalog'):
        catalog.load_catalog_song(tmp_path/'catalog/results.json',song['song_id'],{})
    assert len(calls)==6


@pytest.mark.parametrize('tamper',['raw_missing','audio_missing','audio_changed','text_changed','config'])
def test_completed_catalog_tampering_refuses_reconstruction_calls(tmp_path,monkeypatch,tamper):
    dataset,song,provider,factory,calls=material(tmp_path,monkeypatch);out=tmp_path/'catalog'
    catalog.run_catalog(dataset,provider,out,max_calls=10,provider_factory=factory)
    if tamper=='raw_missing':next((out/'calls').glob('*.raw.json')).unlink()
    elif tamper=='audio_missing':next((out/'audio').rglob('*.wav')).unlink()
    elif tamper=='audio_changed':
        p=next((out/'audio').rglob('*.wav'));p.write_bytes(p.read_bytes()+b' ')
    elif tamper=='text_changed':
        value=read_json(out/'results.json');value['entries'][0]['text']='Forged action';write_json(out/'results.json',value,overwrite=True)
    else:
        value=read_json(provider);value['task_prompt']='Changed task';write_json(provider,value,overwrite=True)
    with pytest.raises(ValueError):catalog.run_catalog(dataset,provider,out,max_calls=10,resume=True,provider_factory=lambda *a,**k:pytest.fail('Replacement model call'))
    assert len(calls)==6


def test_budget_or_checkpoint_override_rejected_before_provider(tmp_path,monkeypatch):
    dataset,song,provider,factory,calls=material(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='complete independent'):
        catalog.run_catalog(dataset,provider,tmp_path/'catalog',max_calls=5,provider_factory=lambda *a,**k:pytest.fail('Before budget'))
    spec=read_json(provider);spec['checkpoint']='not-authorized';write_json(provider,spec,overwrite=True)
    with pytest.raises(ValueError,match='base-only'):
        catalog.run_catalog(dataset,provider,tmp_path/'catalog2',max_calls=10,provider_factory=lambda *a,**k:pytest.fail('Before source validation'))


def test_catalog_infrastructure_failure_stops_following_calls_and_is_never_replaced(tmp_path,monkeypatch):
    dataset,song,provider,factory,calls=material(tmp_path,monkeypatch)
    identity=factory().identity
    def failed(request):
        calls.append(request)
        return dict(identity=identity,synthetic=True,text='',raw={},stop_reason='infrastructure_error',costs={})
    out=tmp_path/'catalog'
    with pytest.raises(RuntimeError,match='infrastructure outcome'):
        catalog.run_catalog(dataset,provider,out,max_calls=10,provider_factory=lambda *a,**k:SimpleNamespace(identity=identity,generate=failed))
    assert len(calls)==1 and not (out/'results.json').exists()
    with pytest.raises(RuntimeError,match='infrastructure outcome'):
        catalog.run_catalog(dataset,provider,out,max_calls=10,resume=True,provider_factory=lambda *a,**k:pytest.fail('No replay replacement'))
    assert len(calls)==1


def test_catalog_changed_returned_identity_cannot_be_counted_as_text_failure(tmp_path,monkeypatch):
    dataset,song,provider,factory,calls=material(tmp_path,monkeypatch)
    native=factory();generate=native.generate
    def changed(request):
        value=generate(request);value['identity']=dict(native.identity,version='different');return value
    with pytest.raises(ValueError,match='returned provider identity'):
        catalog.run_catalog(dataset,provider,tmp_path/'catalog',max_calls=10,
            provider_factory=lambda *a,**k:SimpleNamespace(identity=native.identity,generate=changed))
    assert len(calls)==1


@pytest.mark.parametrize('change',['missing_audio','changed_audio','duplicate_candidate'])
def test_catalog_consumption_requires_exact_waveform_and_candidate_inventory(tmp_path,monkeypatch,change):
    dataset,song,provider,factory,calls=material(tmp_path,monkeypatch);out=tmp_path/'catalog'
    catalog.run_catalog(dataset,provider,out,max_calls=10,provider_factory=factory)
    if change=='missing_audio':next((out/'audio').rglob('*.wav')).unlink()
    elif change=='changed_audio':
        p=next((out/'audio').rglob('*.wav'));p.write_bytes(p.read_bytes()+b' ')
    else:
        value=read_json(out/'results.json');value['entries'][1]=copy.deepcopy(value['entries'][0])
        value.pop('record_sha256');value['record_sha256']=object_sha(value);write_json(out/'results.json',value,overwrite=True)
    with pytest.raises(ValueError):catalog.load_catalog_song(out/'results.json',song['song_id'],{})
    assert len(calls)==6
