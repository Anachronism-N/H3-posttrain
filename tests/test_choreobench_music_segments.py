"""Canonical synthetic audio clocks only; no trusted beat or motion facts."""
import copy
import pytest
from choreobench.fixtures import create_demo
from choreobench.io import read_json
from choreobench.audio import make_windows
from choreobench.music_segments import make_music_observations,paired_observation,UNTIMED_PROMPT


@pytest.fixture
def song(tmp_path):
    made=create_demo(tmp_path/'demo',seconds=30)
    s=read_json(made['dataset'])['songs'][0]
    s['music']['beats']=[0,64000,124800,192000,252800,316800,380800,444800]
    s['music']['downbeats']=s['music']['beats'][:]
    return s


def test_predicted_starts_full_windows_and_nonoverlapping_complete_ownership(song):
    original=copy.deepcopy(song)
    result=make_music_observations(song)
    windows=result['windows'];cursor=0
    for w in windows:
        assert w['observation_end']-w['observation_start']==81000
        assert w['ownership_start']==cursor and w['ownership_end']>cursor
        assert w['ownership_end']<=w['observation_end']
        cursor=w['ownership_end']
    assert cursor==song['audio']['num_samples'] and song==original
    assert any(w['boundary_source']=='predicted_downbeat_backward_anchor' for w in windows)
    assert not result['predicted_meter_verified'] and not result['trained_untimed_captioner_loaded']


def test_fixed_mode_preserves_existing_native_window_clock(song):
    result=make_music_observations(song,'fixed')
    actual=[(w['observation_start'],w['observation_end'],w['ownership_start'],w['ownership_end']) for w in result['windows']]
    expected=[(w['observation_start'],w['observation_end'],w['ownership_start'],w['ownership_end']) for w in make_windows(song['audio']['num_samples'],16000)]
    assert actual==expected


def test_missing_downbeats_explicit_time_fallback(song):
    song['music']['downbeats']=[]
    result=make_music_observations(song)
    assert all(w['boundary_source']!='predicted_downbeat_backward_anchor' for w in result['windows'])
    assert len(result['windows'])==len(make_windows(song['audio']['num_samples'],16000))


def test_variable_meter_times_are_used_not_bpm_extrapolation(song):
    r=make_music_observations(song)
    used=[w['observation_start'] for w in r['windows'] if w['predicted_downbeat_at_start']]
    assert set(used).issubset(set(song['music']['downbeats']))
    assert any(w['boundary_source']=='real_full_window_tail' for w in r['windows'])


def test_pair_geometry_is_same_length_and_bounded_shift(song):
    p=paired_observation(song,250000)
    assert p['fixed']['observation_end']-p['fixed']['observation_start']==81000
    assert p['predicted_downbeat']['observation_end']-p['predicted_downbeat']['observation_start']==81000
    assert abs(p['start_shift_samples'])<=24000 and p['changed']
    assert not p['meter_verified'] and not p['caption_output_times_inferred']


def test_edge_pair_and_no_candidates_do_not_pad(song):
    song['music']['downbeats']=[]
    for target in [0,song['audio']['num_samples']-1]:
        p=paired_observation(song,target)
        assert not p['changed'] and p['boundary_source']=='time_fallback'
        assert 0<=p['fixed']['observation_start']<p['fixed']['observation_end']<=song['audio']['num_samples']


@pytest.mark.parametrize('value',[-1,3.1])
def test_shift_bounds_rejected(song,value):
    with pytest.raises(ValueError):make_music_observations(song,max_shift_seconds=value)
    with pytest.raises(ValueError):paired_observation(song,0,value)


def test_untimed_prompt_does_not_request_observed_actions_or_music_cutoff_as_ending():
    assert 'possible' in UNTIMED_PROMPT and 'without headers' in UNTIMED_PROMPT
    assert 'cutoff' in UNTIMED_PROMPT and 'timestamps' in UNTIMED_PROMPT



def test_untimed_native_probe_six_wave_only_requests_no_caption_timestamps(song,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from choreobench import music_segments
    from choreobench.io import write_json,file_sha
    dataset=dict(schema='choreobench.dataset.v1',dataset_id='probe',synthetic=True,grouping=dict(status='unverified',evidence='fixture'),songs=[song])
    # Preserve the fixture's canonical relative audio location.
    source=tmp_path/'demo/dataset.probe.json';write_json(source,dataset)
    inventory=tmp_path/'probe.json';write_json(inventory,dict(schema='choreobench.untimed_segment_probe.v1',seed=7,cases=[
        dict(dataset_path=str(source),dataset_sha256=file_sha(source),song_id=song['song_id'],target_sample=250000)]))
    requests=[];identity=dict(kind='native',synthetic=True,task_prompt=UNTIMED_PROMPT)
    def generate(request):
        requests.append(request)
        assert set(request)=={'call_id','seed','max_output_tokens','response_kind','audio'}
        assert request['audio']['num_samples']==81000
        return dict(text='Travel softly with flowing arm sweeps and coordinated torso motion.',raw={},identity=identity,
                    synthetic=True,stop_reason='stop',costs=dict(input_tokens=10,output_tokens=12,latency_seconds=0))
    monkeypatch.setattr(music_segments,'create_provider',lambda *a,**k:SimpleNamespace(identity=identity,generate=generate))
    result=music_segments.run_untimed_probe(inventory,dict(kind='native'),tmp_path/'runs')
    assert result['complete'] and result['calls_used']==2 and len(requests)==2
    assert requests[0]['seed']==requests[1]['seed']==7
    assert result['trained_untimed_captioner_loaded'] is False
    assert all(r['untimed_format_verified'] for r in result['rows'])
    assert not result['quality_verified']
    with pytest.raises(FileExistsError):music_segments.run_untimed_probe(inventory,dict(kind='native'),tmp_path/'runs')
