"""Typed organization tests, never a measured motion validator."""
import copy
import json
from types import SimpleNamespace

import pytest

from choreobench import memory_replay, motion_phases
from choreobench.io import read_json
from test_choreobench_duration_content import source


def answer(case):
    samples = case['clock']['boundary_samples']
    return dict(unit_id=case['unit_id'], events=[dict(
        start_boundary=samples.index(i['start']), end_boundary=samples.index(i['end']), track='body',
        phases=[dict(main='Continue a small torso rotation', concurrent=['arms circle softly'], finish='settle with controlled stillness')],
        motif_id=case['outline']['sections'][0]['motif_ids'][0], source_call_ids=[case['options']['windows'][-1]['candidates'][0]['call_id']])
        for i in case['fixed_intervals']])


def test_both_conditions_share_phase_budgets_inputs_seeds_and_schedule(source):
    inventory, old, _, _ = source
    _, requests = motion_phases.prepare_phase_requests(inventory, {}, old)
    a, b = [r for _, _, r in requests]
    assert a['seed'] == b['seed'] and a['max_output_tokens'] == b['max_output_tokens']
    assert b['messages'][0]['content'] == a['messages'][0]['content'] + motion_phases.FORM
    first, second = [json.loads(r['messages'][-1]['content']) for r in (a, b)]
    second['output_contract']['fields'][second['output_contract']['fields'].index('phases')] = 'action'
    assert first == second
    assert all(c['phase_budgets'] == [2] for c, _, _ in requests)
    assert requests[0][0]['fixed_intervals'] == requests[1][0]['fixed_intervals']


def test_compiler_keeps_modifiers_and_finish_literal_not_inventing_state(source):
    inventory, old, _, _ = source
    _, requests = motion_phases.prepare_phase_requests(inventory, {}, old)
    case = requests[1][0]
    value = answer(case); original = copy.deepcopy(value)
    output = json.loads(motion_phases.decode_phases(case, json.dumps(value)))
    assert value == original
    assert output['events'][0]['action'] == 'Continue a small torso rotation while arms circle softly; settle with controlled stillness.'
    assert {k: v for k, v in output['events'][0].items() if k != 'action'} == {
        k: v for k, v in original['events'][0].items() if k != 'phases'}


@pytest.mark.parametrize('bad', ['phase_count', 'main_words', 'modifier_words', 'finish_words', 'unknown', 'empty'])
def test_component_limits_and_unknown_fields_reject_not_rewrite(source, bad):
    inventory, old, _, _ = source
    _, requests = motion_phases.prepare_phase_requests(inventory, {}, old)
    case = requests[1][0]; value = answer(case); phase = value['events'][0]['phases'][0]
    if bad == 'phase_count': value['events'][0]['phases'] *= 3
    elif bad == 'main_words': phase['main'] = 'move ' * 9
    elif bad == 'modifier_words': phase['concurrent'] = ['arms ' * 5]
    elif bad == 'finish_words': phase['finish'] = 'hold ' * 7
    elif bad == 'unknown': phase['support_foot'] = 'left'
    elif bad == 'empty': value['events'][0]['phases'] = []
    with pytest.raises(ValueError): motion_phases.decode_phases(case, json.dumps(value))


def test_typed_raw_compiles_to_saved_plan_and_resume_audit_is_zero_call(source, tmp_path, monkeypatch):
    inventory, old, _, _ = source
    dispatched = []; identity = dict(kind='replay', synthetic=True, fixture='phase-test')
    def generate(request):
        dispatched.append(copy.deepcopy(request)); payload = json.loads(request['messages'][-1]['content'])
        events = []
        for interval in payload['content_contract']['fixed_schedule']:
            event = dict(interval, track='body', motif_id=payload['output_contract']['allowed_motif_ids'][0],
                         source_call_ids=[payload['local_proposals'][0]['call_id']])
            if request['call_id'].endswith('/phase_structured'):
                event['phases'] = [dict(main='Shift weight gradually', concurrent=['arms sweep together'], finish='settle softly')]
            else:
                event['action'] = 'Shift weight gradually while both arms sweep together, then settle softly.'
            events.append(event)
        return dict(text=json.dumps(dict(unit_id=payload['planning_unit']['unit_id'], events=events)), raw={},
            identity=identity, synthetic=True, stop_reason='stop', costs=dict(input_tokens=10, output_tokens=10, latency_seconds=0))
    monkeypatch.setattr(memory_replay, 'create_provider', lambda *a, **k: SimpleNamespace(identity=identity, generate=generate))
    out = tmp_path / 'phases'
    result = memory_replay.run_memory_replay(inventory, dict(kind='replay'), out, duration_source_run=old, motion_phase_probe=True)
    assert result['complete'] and all(r['status'] == 'valid' for r in result['rows'])
    assert result['calls_used'] == 2 and not result['quality_verified']
    assert read_json(out / 'contract.json')['motion_phase_probe']
    assert memory_replay.run_memory_replay(inventory, dict(kind='replay'), out, duration_source_run=old, motion_phase_probe=True, resume=True) == result
    assert memory_replay.audit_completed_replay(out)['valid_rows'] == 2
    assert len(dispatched) == 2
