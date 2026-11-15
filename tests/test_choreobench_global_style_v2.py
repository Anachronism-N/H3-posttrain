"""CPU synthetic contract tests; no GPU/model dispatch or quality claims."""
import copy
import json

import pytest

from choreobench import global_style as gs
from choreobench.composition_workflows import _global_condition, _shared_request
from choreobench.io import file_sha, write_json
from choreobench.public_style_prior import load_public_style_prior


def identity():
    labels = ['class%d' % i for i in range(22)]
    return dict(architecture='frozen MuQ + focal head', checkpoint_sha256='a'*64,
                encoder_weights_sha256='b'*64, class_order=labels, supported_training_classes=labels[:15],
                temperature=1.0, calibration_scope='synthetic test identity, no model claim')


def observation(start, end, winner, **kwargs):
    p = [0.] * 22; p[winner] = 1.
    return dict(start=start, end=end, audio_sha256='c'*64, probabilities=p, **kwargs)


def fixture(tmp_path, windows):
    song = dict(song_id='s1', audio=dict(sha256='d'*64, sample_rate=10, num_samples=100))
    row = dict(song_id='s1', audio_sha256='d'*64, sample_rate=10, num_samples=100, windows=windows)
    value = dict(schema=gs.SCHEMA, dataset_sha256='e'*64, model_identity=identity(), songs=[row])
    path = tmp_path/'style.json'; write_json(path,value)
    return load_public_style_prior(path,'e'*64,[song]), value, song, path


def test_overlap_partitions_time_and_global_condition_is_constant(tmp_path):
    prior, _, _, _ = fixture(tmp_path,[observation(0,80,0),observation(60,100,1)])
    signal = prior.signal('s1')
    assert signal['aggregation']['weight_sum'] == 100
    assert signal['aggregation']['coverage_samples'] == 100
    assert signal['probabilities'][:2] == pytest.approx([.7,.3])
    assert prior.signal('s1',dict(start=80,end=100)) == signal
    assert signal['header'] == 'Dance style is class0.'


def test_explicit_ownership_prevents_context_from_inflating_weight(tmp_path):
    prior, _, _, _ = fixture(tmp_path,[observation(0,80,0,ownership_start=0,ownership_end=40),
                                     observation(20,100,1,ownership_start=40,ownership_end=100)])
    assert prior.signal('s1')['probabilities'][:2] == pytest.approx([.4,.6])


def test_unsupported_argmax_is_not_masked_and_missing_is_unknown(tmp_path):
    prior, value, song, path = fixture(tmp_path,[observation(0,100,21)])
    signal=prior.signal('s1')
    assert signal['label']=='unspecified' and signal['status']=='unsupported_argmax'
    assert signal['probabilities'][21]==1 and signal['argmax_label']=='class21'
    value['songs'][0]['windows']=[];write_json(path,value,overwrite=True)
    signal=load_public_style_prior(path,'e'*64,[song]).signal('s1')
    assert signal['status']=='no_valid_coverage' and signal['probabilities'] is None
    assert _global_condition(prior,'s1','hip-hop')['status']=='explicit_user_override'


def test_invalid_identity_or_clock_rejected(tmp_path):
    _, value, song, path=fixture(tmp_path,[observation(0,100,0)])
    value['model_identity']['supported_training_classes'].pop()
    write_json(path,value,overwrite=True)
    with pytest.raises(ValueError,match='15'):
        load_public_style_prior(path,'e'*64,[song])


def test_shared_evidence_and_header_are_separate_from_action_schema(tmp_path):
    prior,_,_,_=fixture(tmp_path,[observation(0,100,0)])
    source=dict(global_style=_global_condition(prior,'s1'),music_summary=dict(blocks=[],relations=[],parameters={}))
    request=dict(messages=[dict(role='system',content='Generate JSON'),dict(role='user',content='{}')],
                 response_schema=dict(type='object',properties={'events':{'type':'array'}}))
    requests=[_shared_request(copy.deepcopy(request),source,dict(max_request_characters=30000)) for _ in range(4)]
    assert all(r==requests[0] for r in requests)
    assert requests[0]['response_schema']==request['response_schema']
    assert requests[0]['messages'][0]['content'].startswith('Dance style is class0.')
    assert 'music_summary' in json.loads(requests[0]['messages'][-1]['content'])
    plan=dict(sample_rate=10,events=[dict(start=0,end=100,action='Raise both arms.')])
    text=gs.presentation(plan,source['global_style'])
    assert text.count('Dance style is')==1 and plan['events'][0]['action']=='Raise both arms.'
