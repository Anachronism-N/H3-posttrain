"""Structured design and targeted revision: contracts and synthetic trajectories.

These fixtures prove control behaviour only; they say nothing about choreography
quality or musical fit.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from choreobench import composition_structured as s
from choreobench import composition_workflows as m
from choreobench.io import read_json


RATE = 16000


def _events(*pairs, motif='m1'):
    return [dict(event_id='u0000-e%02d' % i, start=start, end=end, track='body', action=action,
                 motif_id=motif, evidence=['c1']) for i, (start, end, action) in enumerate(pairs)]


def _requirement(roles=('m1',), development='introduce', transition='unknown'):
    return dict(required_motifs=list(roles), development=development,
                transition=dict(kind=transition, description=''), primary_motifs=['m1'])


def test_detects_missing_required_motif():
    issues = s.detect_issues(_events((0, 16000, 'A sustained step'), motif=None),
                             dict(unit_id='u0000', start=0, end=16000), _requirement(), [], RATE)
    assert [i['code'] for i in issues] == ['missing_required_motif']


def test_detects_action_timing_contradiction():
    issues = s.detect_issues(_events((0, 8000, 'Step left; turn right; raise both arms')),
                             dict(unit_id='u0000', start=0, end=8000), _requirement(roles=()), [], RATE)
    assert [i['code'] for i in issues] == ['action_timing_contradiction']


def test_detects_missing_transition_only_when_design_asks_for_continuity():
    previous = [dict(event_id='p-e00', start=0, end=16000, track='body', action='Hold', motif_id='m1')]
    unit = dict(unit_id='u0001', start=32000, end=48000)
    events = _events((32000, 48000, 'A new phrase'))
    assert 'missing_transition' in [i['code'] for i in
                                    s.detect_issues(events, unit, _requirement(transition='continue'), previous, RATE)]
    assert 'missing_transition' not in [i['code'] for i in
                                        s.detect_issues(events, unit, _requirement(transition='shift'), previous, RATE)]


def test_detects_verbatim_repetition_but_not_when_repeat_is_designed():
    previous = [dict(event_id='p-e00', start=0, end=16000, track='body', action='Side step left.', motif_id='m1')]
    events = _events((16000, 32000, 'Side step left.'))
    unit = dict(unit_id='u0001', start=16000, end=32000)
    assert 'mechanical_repetition' in [i['code'] for i in
                                       s.detect_issues(events, unit, _requirement(transition='shift'), previous, RATE)]
    assert 'mechanical_repetition' not in [i['code'] for i in
                                           s.detect_issues(events, unit, _requirement(development='repeat', transition='shift'),
                                                           previous, RATE)]


def test_clean_block_has_no_issues():
    assert s.detect_issues(_events((0, 32000, 'A sustained side step with concurrent arm swing')),
                           dict(unit_id='u0000', start=0, end=32000), _requirement(), [], RATE) == []


def _song():
    return dict(song_id='s1', brief='b', audio=dict(sample_rate=RATE, num_samples=192000),
                music=dict(source='public', beats=[0], downbeats=[0],
                           sections=[dict(section_id='a', start=0, end=96000, label='unknown', repeat_group=None),
                                     dict(section_id='b', start=96000, end=192000, label='unknown', repeat_group=None)]))


def _options():
    return dict(windows=[dict(window=dict(observation_start=0, observation_end=96000, index=0),
                              candidates=[dict(call_id='c1', raw_text='caption text')])])


def _outline_entry(section_id, **changes):
    entry = dict(section_id=section_id, primary_motifs=['m1'], development='introduce',
                 quantity_change='steady', transition_from_previous=dict(kind='unknown', description='none yet'),
                 evidence=[dict(source='public_section_span', note='duration only')], uncertainty='beat unknown')
    entry.update(changes)
    return entry


def outline_text(sections=None):
    return json.dumps(dict(motifs=[dict(motif_id='m1', description='Side step with arm swing')],
                           sections=sections or [_outline_entry('a'),
                                                 _outline_entry('b', development='vary',
                                                                transition_from_previous=dict(kind='continue',
                                                                                              description='keep stepping'))]))


def test_outline_accepts_honest_unknowns():
    value = s.read_outline(outline_text(), _song(), _options())
    assert [sec['section_id'] for sec in value['sections']] == ['a', 'b']
    assert value['sections'][0]['transition_from_previous']['kind'] == 'unknown'
    assert value['sections'][0]['motif_ids'] == ['m1']


@pytest.mark.parametrize('sections,fragment', [
    ([_outline_entry('a', transition_from_previous=dict(kind='continue', description='x')),
      _outline_entry('b')], 'first section'),
    ([_outline_entry('a'),
      _outline_entry('b', evidence=[dict(source='public_beat_density', note='tempo')])], 'not available'),
    ([_outline_entry('a', primary_motifs=[]), _outline_entry('b')], 'at least one primary motif'),
    ([_outline_entry('a', evidence=[]), _outline_entry('b')], 'at least one evidence entry'),
    ([_outline_entry('a'), _outline_entry('a')], 'rename, omit or reorder'),
])
def test_outline_refuses_unsupported_design_claims(sections, fragment):
    with pytest.raises(ValueError, match=fragment):
        s.read_outline(outline_text(sections), _song(), _options())


def _material(tmp_path, monkeypatch, draft_for_block):
    path, sid = __import__('test_choreobench_song_agent').source(tmp_path, monkeypatch)
    calls = []
    identity = dict(kind='replay', synthetic=True, fixture='structured')
    state = dict(block=0)

    def generate(request):
        calls.append(copy.deepcopy(request))
        data = json.loads(request['messages'][-1]['content'])
        if request['response_kind'] == 'structured_outline':
            answer = json.loads(outline_text())
        elif data['committed_units'] == data['total_units']:
            answer = dict(tool='finish', proposal=None, reason='All units committed')
        elif data['current_draft'] is None:
            answer = dict(tool='propose_block', proposal=draft_for_block(data), reason='Draft from public evidence')
        elif data.get('tool_observations') and any(
                o.get('issue_scan') for o in data['tool_observations'] if isinstance(o, dict)):
            answer = dict(tool='revise_block', proposal=draft_for_block(data, fixed=True),
                          reason='Fix only the listed issue')
        else:
            answer = dict(tool='commit_block', proposal=None, reason='Commit')
        return dict(text=json.dumps(answer), identity=identity, synthetic=True, raw={}, stop_reason='stop',
                    costs=dict(input_tokens=100, output_tokens=50))
    monkeypatch.setattr(m, 'create_provider', lambda *a, **k: SimpleNamespace(identity=identity, generate=generate))
    return path, sid, tmp_path / 'run', calls


def _clean(data, fixed=False):
    unit = data['planning_unit']
    return dict(unit_id=unit['unit_id'], events=[dict(
        start=unit['start'], end=unit['end'], track='body',
        action='A sustained phrase for ' + unit['unit_id'], motif_id='m1',
        source_call_ids=[data['local_proposals'][0]['call_id']])])


def _without_motif(data, fixed=False):
    draft = _clean(data)
    if not fixed:
        draft['events'][0]['motif_id'] = None
    return draft


def test_clean_run_commits_without_extra_revision_calls(tmp_path, monkeypatch):
    path, sid, out, calls = _material(tmp_path, monkeypatch, _clean)
    result = m.run_composition(path, sid, {'kind': 'replay'}, out, method='structured', controller_version='v2')
    assert result['full_song_succeeded'] and result['committed_units'] == 2
    assert result['calls_used'] == 3, [c['response_kind'] for c in calls]
    stats = result['structured_statistics']
    assert stats['blocks_with_detected_issues'] == 0 and stats['revision_calls'] == 0
    assert stats['budget_unchanged'] is True
    assert m.run_composition(path, sid, {'kind': 'replay'}, out,
                             method='structured', controller_version='v2', resume=True) == result


def test_missing_motif_triggers_exactly_one_targeted_revision(tmp_path, monkeypatch):
    path, sid, out, calls = _material(tmp_path, monkeypatch, _without_motif)
    result = m.run_composition(path, sid, {'kind': 'replay'}, out, method='structured', controller_version='v2')
    assert result['full_song_succeeded']
    stats = result['structured_statistics']
    assert stats['blocks_with_detected_issues'] >= 1 and stats['revision_calls'] >= 1
    outline_call, first_propose = calls[0], calls[1]
    assert outline_call['response_kind'] == 'structured_outline'
    payload = json.loads(first_propose['messages'][-1]['content'])
    assert payload['block_requirement']['required_motifs'] == ['m1']
    assert payload['section_requirement']['design']['section_id'] in ('a', 'b')
    steps = sorted((out / 'steps').glob('*.json'))
    scans = [read_json(p) for p in steps if read_json(p)['observation'].get('issue_scan')]
    assert scans and all(i['code'] in s.ISSUE_CODES for scan in scans for i in scan['observation']['issue_scan']['issues'])


def test_reasonable_repeat_block_is_not_penalised_end_to_end(tmp_path, monkeypatch):
    path, sid, out, calls = _material(tmp_path, monkeypatch, _clean)
    result = m.run_composition(path, sid, {'kind': 'replay'}, out, method='structured', controller_version='v2')
    assert result['structured_statistics']['revision_calls'] == 0
    # Identical action text across blocks is allowed by the clean fixture and must not loop.
    assert result['calls_used'] == 3

def test_structured_requires_v2():
    with pytest.raises(ValueError, match='requires the bounded v2 controller'):
        m.run_composition('x', 'y', {'kind': 'replay'}, 'z', method='structured', controller_version='v1')
