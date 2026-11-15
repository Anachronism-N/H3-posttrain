"""CPU contract tests; synthetic waveforms never establish model quality."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from choreobench import bar_windows as bw, untimed_catalog as catalog
from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.planning_units import build_bar_planning_units
from choreobench.providers import NativeProvider
from test_choreobench_baseline import dataset_file


def material(tmp_path, seconds=20, downbeats=None):
    dataset, song = dataset_file(tmp_path, seconds=seconds)
    value = read_json(dataset)
    # A single section isolates the bar grouping from optional public form cuts.
    song['music']['sections'] = [dict(section_id='whole', start=0, end=song['audio']['num_samples'],
                                    label='unknown', repeat_group=None)]
    value['songs'] = [song]
    write_json(dataset, value, overwrite=True)
    times = list(range(0, seconds, 2)) if downbeats is None else downbeats
    row = dict(schema='choreobench.beat_detections.v1', id=song['song_id'], status='ok',
        audio_sha256=song['audio']['sha256'], sample_rate=song['audio']['sample_rate'],
        num_samples=song['audio']['num_samples'], time_unit='seconds', synthetic=True,
        detector=dict(name='beat_this', version='test', model_sha256='a'*64, code_sha256='b'*64,
                      parameters=dict(dbn=False)), beats=times, downbeats=times)
    detections = tmp_path/'detections.jsonl'
    detections.write_text(json.dumps(row)+'\n', encoding='utf-8')
    return dataset, song, detections, row


def test_dynamic_closed_bars_cover_clock_and_keep_residuals(tmp_path):
    dataset, song, detections, row = material(tmp_path, 11, [0, 2, 4, 6, 8, 10])
    manifest = bw.build_manifest(dataset, detections, tmp_path/'windows.json')
    windows = manifest['songs'][0]['windows']
    complete = [w for w in windows if w['kind']=='complete_bars']
    assert sorted(w['complete_bar_count'] for w in complete) == [2, 3]
    assert sorted(w['actual_window_seconds'] for w in complete) == [4., 6.]
    assert windows[-1]['kind']=='suffix_fragment' and windows[-1]['actual_window_seconds']==1
    assert windows[0]['observation_start']==0 and windows[-1]['observation_end']==11*16000
    assert all(a['ownership_end']==b['ownership_start'] for a,b in zip(windows,windows[1:]))
    units = build_bar_planning_units(song, manifest['songs'][0])
    assert [(u['start'],u['end']) for u in units['units']] == [(w['observation_start'],w['observation_end']) for w in windows]
    assert not manifest['songs'][0]['semantic_phrase_verified']


def test_short_edges_remain_auditable_and_tempo_is_not_forced_to_five_seconds(tmp_path):
    _, song, _, row = material(tmp_path, 17, [.1, 4.1, 8.1, 12.1, 16.8])
    result = bw.make_song_windows(song,row)
    assert result['windows'][0]['observation_start']==0
    assert result['windows'][-1]['observation_end']==17*16000
    assert all(w['actual_window_seconds'] > 5.0625 for w in result['windows'])
    assert any('absorbed_short_fragment' in w['reasons'] for w in result['windows'])
    assert all(w['kind']=='partial_fragment' for w in result['windows'])


@pytest.mark.parametrize('damage', ['wrong_hash','librosa','dbn','no_bars','clock','order'])
def test_wrong_detector_and_source_fail_without_fallback(tmp_path,damage):
    _, song, _, row = material(tmp_path)
    if damage=='wrong_hash':row['audio_sha256']='c'*64
    elif damage=='librosa':row['detector']['name']='librosa'
    elif damage=='dbn':row['detector']['parameters']['dbn']=True
    elif damage=='no_bars':row['beats']=row['downbeats']=[0]
    elif damage=='clock':row['num_samples']+=1
    else:row['downbeats']=[2,0]
    with pytest.raises(ValueError):bw.make_song_windows(song,row)


def test_tampered_manifest_and_wrong_dataset_refused(tmp_path):
    dataset,song,detections,_=material(tmp_path)
    path=tmp_path/'windows.json';document=bw.build_manifest(dataset,detections,path)
    changed=copy.deepcopy(document);changed['songs'][0]['windows'][0]['observation_end']+=1
    with pytest.raises(ValueError,match='recomputation'):bw.validate_manifest(changed,read_json(dataset),file_sha(dataset))
    with pytest.raises(ValueError,match='different dataset'):bw.validate_manifest(document,read_json(dataset),'c'*64)


def test_variable_catalog_roundtrip_exact_audio_and_no_extra_resume_calls(tmp_path):
    dataset,song,detections,_=material(tmp_path)
    manifest=tmp_path/'windows.json';document=bw.build_manifest(dataset,detections,manifest)
    provider=tmp_path/'provider.json'
    write_json(provider,dict(kind='native',model='fixture',audio_contract='variable_duration_v1',task_prompt='Untimed choreography.'))
    identity=dict(kind='replay',synthetic=True,fixture='bar-v2');calls=[]
    def generate(request):
        calls.append(copy.deepcopy(request))
        return dict(text='Step sideways while raising the arms.',raw={},identity=identity,synthetic=True,
                    stop_reason='eos',costs=dict(input_tokens=8,output_tokens=9,latency_seconds=0))
    factory=lambda *a,**k:SimpleNamespace(identity=identity,generate=generate)
    output=tmp_path/'catalog'
    result=catalog.run_catalog(dataset,provider,output,windows_manifest=manifest,provider_factory=factory)
    assert result['schema']=='choreobench.untimed_catalog.v2'
    lengths=[w['observation_end']-w['observation_start'] for w in document['songs'][0]['windows']]
    assert [r['audio']['num_samples'] for r in calls]==[n for n in lengths for _ in range(2)]
    original,options,synthetic,_=catalog.load_catalog_song(output/'results.json',song['song_id'],{})
    assert options['bar_windows']==document['songs'][0]
    assert options['window_manifest_sha256']==file_sha(manifest)
    assert original==song and synthetic
    count=len(calls)
    assert catalog.run_catalog(dataset,provider,output,windows_manifest=manifest,resume=True,
                               provider_factory=lambda *a,**k:pytest.fail('No reload on complete replay'))==result
    assert len(calls)==count
    document['songs'][0]['windows'][0]['actual_window_seconds']+=1
    write_json(manifest,document,overwrite=True)
    with pytest.raises(ValueError):catalog.run_catalog(dataset,provider,output,windows_manifest=manifest,resume=True,
                                                     provider_factory=lambda *a,**k:pytest.fail('Invalid manifest must not call model'))


def test_fixed_provider_rejected_before_variable_catalog_dispatch(tmp_path):
    dataset,_,detections,_=material(tmp_path)
    manifest=tmp_path/'windows.json';bw.build_manifest(dataset,detections,manifest)
    provider=tmp_path/'provider.json';write_json(provider,dict(kind='native',model='fixture',task_prompt='Untimed.'))
    with pytest.raises(ValueError,match='variable_duration_v1'):
        catalog.run_catalog(dataset,provider,tmp_path/'catalog',windows_manifest=manifest,
                            provider_factory=lambda *a,**k:pytest.fail('No fixed model for variable window'))


def adapter_audio_function():
    # Execute the actual CPU method without importing the GPU/training module.
    path=Path(__file__).parents[1]/'scripts_omnidance/exp/ms/music_planner_adapters.py'
    module=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='PlannerAdapter')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_audio')
    namespace=dict(Path=Path,hashlib=hashlib,re=re,np=np,math=math)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_audio']


def test_native_audio_keeps_real_six_seconds_and_old_protocol_still_rejects(tmp_path):
    dataset,song=dataset_file(tmp_path,seconds=6)
    row=dict(id='six',audio_path=str(tmp_path/'song.wav'),audio_sha256=song['audio']['sha256'],
             audio_contract='variable_duration_v1',audio_window_seconds=[0.,6.],
             audio_num_samples=96000,audio_sample_rate=16000)
    adapter=SimpleNamespace(_audio_identities={});load=adapter_audio_function()
    audio=load(adapter,row)
    assert len(audio)==96000 and np.all(audio==1/32768)
    assert adapter._last_audio_contract_receipt['source_seconds']==6
    wrong=dict(row,audio_window_seconds=[0.,5.0625])
    with pytest.raises(ValueError,match='exact complete waveform'):load(adapter,wrong)
    legacy={k:v for k,v in row.items() if k not in ('audio_contract','audio_num_samples','audio_sample_rate')}
    legacy['audio_window_seconds']=[0.,5.0625]
    with pytest.raises(ValueError,match='5.0625-second'):load(adapter,legacy)


def test_native_provider_transmits_actual_duration_not_fixed_metadata(tmp_path,monkeypatch):
    _,song=dataset_file(tmp_path,seconds=6)
    prefix='scripts_omnidance.exp.ms.'
    identity=ModuleType('identity');identity.audit_model_artifacts=lambda *a:{}
    identity.verify_loaded_portable=lambda *a:{'verified':True}
    search=ModuleType('search');search.validate_sampling=lambda *a:None
    captured=[]
    def sample(adapter,row,**kwargs):
        captured.append(row);adapter_audio_function()(adapter,row)
        return dict(raw_text='Sway with an arm sweep.',new_token_ids=[1,2],stop_reason='eos',input_tokens=8)
    search.sample_native=sample
    adapters=ModuleType('adapters')
    adapters.load_planner_adapter=lambda *a,**k:SimpleNamespace(_audio_identities={},processor=SimpleNamespace(
        apply_chat_template=lambda *a,**k:'Untimed choreography.'))
    for name,value in [('music_planner_search_identity',identity),('music_planner_search',search),('music_planner_adapters',adapters)]:
        monkeypatch.setitem(sys.modules,prefix+name,value)
    provider=NativeProvider(dict(kind='native',model='fixture',task_prompt='Untimed choreography.',audio_contract='variable_duration_v1'),'local')
    result=provider.generate(dict(call_id='six',seed=42,max_output_tokens=10,response_kind='local_caption',
        audio=dict(song['audio'],path=str(tmp_path/'song.wav'))))
    assert captured[0]['audio_window_seconds']==[0.,6.] and captured[0]['audio_num_samples']==96000
    assert result['raw']['audio_contract_receipt']['complete_waveform_loaded']
    assert provider.identity['audio_contract']=='variable_duration_v1'
