"""CPU tests of stochastic native search and fail-closed reviewed data staging."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import wave

import pytest

from scripts_omnidance.exp.ms.music_planner_search import (
    DIMENSIONS, GROUPS, export_reviewed, file_sha, generate_candidates, object_sha,
    prepare_fair_inputs, read_json, review_request, sample_native, validate_isolation, validate_pool,
    validate_policy, validate_reviews, validate_staging, write_json,
)


def write_wave(path, frames=121500):
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((1, 2, 24000, frames, 'NONE', 'not compressed'))
        stream.writeframes(b'\0\0' * frames)


@pytest.fixture
def inputs(tmp_path):
    members, rows = [], []
    for index, split in enumerate(('train', 'dev', 'test')):
        audio = tmp_path / (split + '.wav')
        write_wave(audio)
        # Distinguish source bytes without changing its complete duration.
        data = bytearray(audio.read_bytes())
        data[-2:] = bytes((index + 1, 0))
        audio.write_bytes(data)
        member = dict(id=split, split=split, audio_sha256=file_sha(audio),
                      **{key: f'{key}-{split}' for key in GROUPS})
        members.append(member)
        rows.append(dict(member, audio_path=str(audio), audio_window_seconds=[0., 5.0625],
                         target_text='SECRET ground truth', genre='SECRET genre',
                         source_video='must-not-leak.mp4', reference_image='must-not-leak.jpg'))
    manifest = dict(schema='music_planner_isolation_v1', version='verified-fixture-v1',
                    verification={key: dict(status='verified', evidence_sha256='e' * 64) for key in GROUPS},
                    members=members)
    generator = dict(family='moss', model_id='planner/model', base_model_id='planner/base',
                     model_sha256='a' * 64, version='checkpoint-10')
    return rows, manifest, generator


def mock_sampler(row, *, seed, temperature, top_p, top_k, max_new_tokens):
    assert set(row) == {'id', 'source_group_id', 'audio_path', 'audio_sha256', 'audio_window_seconds'}
    assert temperature > 0 and top_p > 0
    return dict(raw_text=f'[0-2s] Steps left {seed}. [2-5.062s] Raises both hands {seed}.',
                raw_with_special_tokens=f'raw-{seed}<eos>',
                new_token_ids=[17, 2], eos_token_ids=[2], stop_reason='eos',
                budget_truncated=False, seed=seed,
                decoding=dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                              num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                              encoder_repetition_penalty=1.0),
                input_tokens=3, prefix_sha256='b' * 64, audio_input='full-waveform',
                silence_applied=False, processor_sample_rate=16000, n_samples=81000)


@pytest.fixture
def pool(inputs):
    rows, manifest, generator = inputs
    return generate_candidates(rows[:2], manifest, generator, count=3, sampler=mock_sampler)


@pytest.fixture
def policy():
    text = ('Assess music compatibility, plausible dance genre, and continuous action planning. '
            'Do not reward formatting, length, or matching an original video choreography.')
    return dict(schema='music_planner_review_policy_v1',
                rubric=dict(id='planner-music-text', version='1', text=text,
                            sha256=hashlib.sha256(text.encode()).hexdigest()),
                judge=dict(kind='human', id='reviewer-1', version='protocol-1',
                           model_id=None, base_model_id=None, model_sha256=None),
                thresholds=dict(min_dimension_score=3., min_confidence=.8, preference_margin=1.))


def reviews_for(pool, policy, *, equal=False):
    request = review_request(pool, policy)
    reviews = []
    for item in pool['candidates']:
        index, source = item['index'], item['source']
        score = 4. if equal else (5. if index == 0 else 2.)
        reviews.append(dict(candidate_sha256=item['candidate_sha256'], source_id=source['id'],
                            audio_sha256=source['audio_sha256'], original_audio_reviewed=True,
                            verdict='approve' if equal or index == 0 else 'reject',
                            scores={key: score for key in DIMENSIONS}, confidence=.95,
                            reason='Listened to the audio; this motion density follows its energy change.',
                            allow_preference_use=True))
    return dict(schema='music_planner_reviews_v1', request_sha256=request['request_sha256'],
                judge=copy.deepcopy(policy['judge']), rubric=copy.deepcopy(policy['rubric']),
                reviews=reviews)


def test_candidates_are_audio_only_stochastic_distinct_and_deterministic(inputs):
    rows, manifest, generator = inputs
    first = generate_candidates(rows[:2], manifest, generator, count=4, sampler=mock_sampler)
    second = generate_candidates(list(reversed(rows[:2])), manifest, generator, count=4, sampler=mock_sampler)
    assert first == second
    for sid in ('train', 'dev'):
        candidates = [c for c in first['candidates'] if c['source']['id'] == sid]
        assert len({c['generation']['raw_text'] for c in candidates}) == 4
        assert len({c['generation']['seed'] for c in candidates}) == 4
        assert all(c['generation']['decoding']['do_sample'] for c in candidates)
        assert all(c['diagnostics']['diagnostic_only_not_music_reward'] for c in candidates)
    assert 'SECRET' not in json.dumps(first)


@pytest.mark.parametrize('key', GROUPS + ('audio_sha256',))
def test_manifest_rejects_heldout_group_or_exact_audio_leakage(inputs, key):
    _, manifest, _ = inputs
    manifest['members'][1][key] = manifest['members'][0][key]
    with pytest.raises(ValueError, match='Cross-split leakage'):
        validate_isolation(manifest)


def test_manifest_requires_verified_song_near_duplicate_and_complete_inventory(inputs):
    _, manifest, _ = inputs
    manifest['verification']['song_group_id']['status'] = 'unknown'
    manifest['verification']['song_group_id']['evidence_sha256'] = None
    with pytest.raises(ValueError, match='must be verified'):
        validate_isolation(manifest, require_verified=True)
    manifest['verification']['song_group_id']['status'] = 'verified'
    manifest['verification']['song_group_id']['evidence_sha256'] = 'e' * 64
    manifest['members'].pop()
    with pytest.raises(ValueError, match='include train, dev, and test'):
        validate_isolation(manifest)


def test_unknown_song_groups_allow_search_review_but_never_train_export(inputs, policy):
    rows, manifest, generator = inputs
    for key in ('song_group_id', 'near_duplicate_group_id'):
        manifest['verification'][key] = dict(status='unknown', evidence_sha256=None)
        for member in manifest['members']:
            member[key] = None
        for row in rows:
            row[key] = None
    pool = generate_candidates(rows[:1], manifest, generator, sampler=mock_sampler)
    request = review_request(pool, policy)
    assert request['selection_status'] == 'pending_independent_review'
    result = export_reviewed(pool, policy, reviews_for(pool, policy))
    assert result['isolation_training_eligible'] is False
    assert result['pseudo_sft'] == result['preference_pairs'] == []
    assert result['decisions'][0]['status'] == 'blocked_unverified_isolation_no_training_export'


def test_prepare_current_fair_data_preserves_unknown_groups_and_hides_targets(inputs, tmp_path):
    rows, _, _ = inputs
    data = tmp_path / 'fair'
    data.mkdir()
    for row, filename in zip(rows, ('train', 'val', 'test')):
        original = {key: value for key, value in row.items() if key not in ('song_group_id', 'near_duplicate_group_id')}
        original['split'] = filename
        (data / (filename + '.jsonl')).write_text(json.dumps(original) + '\n', encoding='utf-8')
    selected, manifest = prepare_fair_inputs(data, 'dev', 1)
    assert selected[0]['split'] == 'dev'
    assert selected[0]['song_group_id'] is selected[0]['near_duplicate_group_id'] is None
    assert 'SECRET' not in json.dumps(selected)
    assert manifest['verification']['song_group_id']['status'] == 'unknown'
    validate_isolation(manifest)


def test_test_sources_never_reach_sampler(inputs):
    rows, manifest, generator = inputs
    with pytest.raises(ValueError, match='final-only'):
        generate_candidates(rows, manifest, generator,
                            sampler=lambda *a, **k: pytest.fail('test must fail before generation'))


def test_mismatch_declared_audio_or_physical_crop_fails_before_sampler(inputs):
    rows, manifest, generator = inputs
    rows[0]['audio_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='differs from manifest'):
        generate_candidates(rows[:1], manifest, generator, sampler=mock_sampler)
    rows[0]['audio_sha256'] = manifest['members'][0]['audio_sha256']
    rows[0]['audio_window_seconds'] = [0., 2.]
    with pytest.raises(ValueError, match='crop'):
        generate_candidates(rows[:1], manifest, generator, sampler=mock_sampler)


def test_greedy_adapter_with_only_changed_seed_is_rejected(inputs):
    rows, manifest, generator = inputs
    def greedy(*args, **kwargs):
        result = mock_sampler(*args, **kwargs)
        result['decoding']['do_sample'] = False
        return result
    with pytest.raises(ValueError, match='stochastic decoding'):
        generate_candidates(rows[:1], manifest, generator, sampler=greedy)


def test_no_reviews_produce_request_and_pending_not_best(pool, policy):
    request = review_request(pool, policy)
    assert request['selection_status'] == 'pending_independent_review'
    assert 'reference_target' not in json.dumps(request)
    assert request['scope'] == 'audio_and_text_plan_only'
    reviews = reviews_for(pool, policy)
    reviews['reviews'] = []
    result = export_reviewed(pool, policy, reviews)
    assert result['pseudo_sft'] == result['preference_pairs'] == []
    assert all(d['selected_candidate_sha256'] is None for d in result['decisions'])


def test_approved_export_is_train_only_and_generated_plan_not_observation(pool, policy):
    reviews = reviews_for(pool, policy)
    result = export_reviewed(pool, policy, reviews)
    assert len(result['pseudo_sft']) == 1
    assert len(result['preference_pairs']) == 2
    row = result['pseudo_sft'][0]
    assert row['split'] == 'train' and row['id'] == 'train'
    assert row['provenance']['pool_source'] == 'generated_plan'
    assert row['provenance']['observed_video_ground_truth'] is False
    assert row['fair_baseline_loader_compatible'] is False
    assert result['automatic_weight_update_implemented'] is False
    assert result['test_used'] is False
    assert result['decisions'][0]['status'] == 'dev_review_only_never_training_export'
    assert export_reviewed(pool, policy, reviews) == result
    assert validate_staging(pool, policy, reviews, result)['valid']


def test_ties_not_arbitrarily_broken_for_sft_or_preferences(pool, policy):
    result = export_reviewed(pool, policy, reviews_for(pool, policy, equal=True))
    assert result['pseudo_sft'] == result['preference_pairs'] == []
    assert any(d['status'] == 'tie_no_unique_selection' for d in result['decisions'])


def test_incomplete_review_does_not_select_from_partially_scored_candidates(pool, policy):
    reviews = reviews_for(pool, policy)
    reviews['reviews'] = [r for r in reviews['reviews'] if not (
        r['source_id'] == 'train' and r['candidate_sha256'] == pool['candidates'][-1]['candidate_sha256'])]
    result = export_reviewed(pool, policy, reviews)
    assert result['pseudo_sft'] == result['preference_pairs'] == []


@pytest.mark.parametrize('change', ('audio', 'source', 'candidate', 'version', 'rubric', 'request', 'text_only', 'extra'))
def test_strict_reviews_reject_cross_sample_stale_or_text_only_scores(pool, policy, change):
    reviews = reviews_for(pool, policy)
    row = reviews['reviews'][0]
    if change == 'audio':
        row['audio_sha256'] = 'f' * 64
    elif change == 'source':
        row['source_id'] = 'other-source'
    elif change == 'candidate':
        row['candidate_sha256'] = 'f' * 64
    elif change == 'version':
        reviews['judge']['version'] = 'old-judge'
    elif change == 'rubric':
        reviews['rubric']['version'] = 'old-rubric'
    elif change == 'request':
        reviews['request_sha256'] = 'f' * 64
    elif change == 'text_only':
        row['original_audio_reviewed'] = False
    else:
        row['unregistered_score'] = 5
    with pytest.raises(ValueError):
        validate_reviews(pool, policy, reviews)


@pytest.mark.parametrize('key', ('model_id', 'base_model_id', 'model_sha256'))
def test_same_model_self_scores_rejected_even_when_reviewer_alias_changes(pool, policy, key):
    policy['judge'] = dict(kind='audio_model', id='different-display-name', version='v1',
                           model_id='judge/model', base_model_id='judge/base', model_sha256='f' * 64)
    policy['judge'][key] = pool['generator'][key]
    with pytest.raises(ValueError, match='Same-model|identical'):
        validate_policy(policy, pool['generator'])


def test_text_model_is_not_accepted_as_audio_judge(pool, policy):
    policy['judge']['kind'] = 'text_model'
    with pytest.raises(ValueError, match='independent audio_model'):
        review_request(pool, policy)


@pytest.mark.parametrize('value', (float('nan'), float('inf'), True, -1, 6))
def test_nonfinite_boolean_and_out_of_range_review_scores_fail(pool, policy, value):
    reviews = reviews_for(pool, policy)
    reviews['reviews'][0]['scores']['music_fit'] = value
    with pytest.raises(ValueError):
        validate_reviews(pool, policy, reviews)


def test_low_confidence_uncertain_and_disallowed_preference_are_not_admitted(pool, policy):
    reviews = reviews_for(pool, policy)
    for row in reviews['reviews']:
        if row['source_id'] == 'train':
            row['verdict'] = 'uncertain'
    assert not export_reviewed(pool, policy, reviews)['pseudo_sft']
    reviews = reviews_for(pool, policy)
    for row in reviews['reviews']:
        row['confidence'] = .5
    assert not export_reviewed(pool, policy, reviews)['pseudo_sft']
    reviews = reviews_for(pool, policy)
    for row in reviews['reviews']:
        row['allow_preference_use'] = False
    result = export_reviewed(pool, policy, reviews)
    assert len(result['pseudo_sft']) == 1 and not result['preference_pairs']


def test_valid_reviews_never_rescue_truncated_or_bad_format_candidate(inputs, policy):
    rows, manifest, generator = inputs
    def truncated(row, **kwargs):
        result = mock_sampler(row, **kwargs)
        result.update(new_token_ids=[17] * kwargs['max_new_tokens'], stop_reason='budget_truncation',
                      budget_truncated=True)
        return result
    pool = generate_candidates(rows[:1], manifest, generator, sampler=truncated)
    result = export_reviewed(pool, policy, reviews_for(pool, policy))
    assert not result['pseudo_sft'] and not result['preference_pairs']


def test_empty_native_output_is_preserved_as_unknown_stop_not_a_candidate_success(inputs, policy):
    rows, manifest, generator = inputs
    def empty(row, **kwargs):
        result = mock_sampler(row, **kwargs)
        result.update(raw_text='', raw_with_special_tokens='', new_token_ids=[],
                      stop_reason='unknown_stop')
        return result
    pool = generate_candidates(rows[:1], manifest, generator, sampler=empty)
    assert all(not item['diagnostics']['format_eligible'] for item in pool['candidates'])
    assert not export_reviewed(pool, policy, reviews_for(pool, policy))['pseudo_sft']


def test_audio_replacement_and_candidate_tampering_are_detected(pool, policy):
    altered = copy.deepcopy(pool)
    altered['candidates'][0]['generation']['raw_text'] += ' tampered'
    with pytest.raises(ValueError, match='run hash'):
        validate_pool(altered)
    path = Path(pool['candidates'][0]['source']['audio_path'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='audio path/bytes'):
        review_request(pool, policy)


def test_export_validation_rejects_heldout_or_edited_rows(pool, policy):
    reviews = reviews_for(pool, policy)
    staging = export_reviewed(pool, policy, reviews)
    staging['pseudo_sft'][0]['split'] = 'dev'
    with pytest.raises(ValueError, match='deterministic reviewed train-only'):
        validate_staging(pool, policy, reviews, staging)


def test_json_io_rejects_duplicates_nonfinite_and_overwrite(tmp_path):
    path = tmp_path / 'artifact.json'
    path.write_text('{"x":1,"x":2}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        read_json(path)
    path.write_text('{"x":NaN}', encoding='utf-8')
    with pytest.raises(ValueError, match='Non-finite'):
        read_json(path)
    with pytest.raises(FileExistsError):
        write_json(path, {})


def test_cli_request_export_and_check_complete_round_trip(tmp_path, pool, policy):
    paths = {name: tmp_path / (name + '.json') for name in ('pool', 'policy', 'reviews', 'request', 'staging')}
    write_json(paths['pool'], pool)
    write_json(paths['policy'], policy)
    write_json(paths['reviews'], reviews_for(pool, policy))
    prefix = [sys.executable, '-m', 'scripts_omnidance.exp.ms.music_planner_search']
    common = ['--pool', str(paths['pool']), '--policy', str(paths['policy'])]
    for command in (
            ['review-request'] + common + ['--out', str(paths['request'])],
            ['export'] + common + ['--reviews', str(paths['reviews']), '--out', str(paths['staging'])],
            ['check'] + common + ['--reviews', str(paths['reviews']), '--staging', str(paths['staging'])]):
        actual = subprocess.run(prefix + command, capture_output=True, text=True, check=False)
        assert actual.returncode == 0, actual.stderr
    assert json.loads(actual.stdout)['pseudo_sft_rows'] == 1
    assert read_json(paths['request'])['selection_status'] == 'pending_independent_review'


class TinyTokenizer:
    def decode(self, ids, skip_special_tokens=True, **kwargs):
        body = f'[0-2s] Steps left {ids[0]}. [2-5.062s] Waves both hands {ids[0]}.'
        return body if skip_special_tokens else body + '<eos>'


class TinyProcessor:
    audio_token_id = 3
    tokenizer = TinyTokenizer()

    def __init__(self):
        self.calls = []
        self.messages = None

    def _build_default_prompt(self, instruction, has_audio):
        assert has_audio
        self.messages = instruction
        return 'MOSS|' + instruction

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'native-template|' + json.dumps(messages)

    def __call__(self, **kwargs):
        import torch
        self.calls.append(kwargs)
        return dict(input_ids=torch.tensor([[3, 7, 9]]), attention_mask=torch.ones(1, 3, dtype=torch.long))


@pytest.mark.parametrize('family', ('moss', 'music_flamingo', 'omni_captioner'))
def test_real_torch_sampling_reuses_each_native_prefix_audio_checks_and_restores_rng(inputs, family):
    torch = pytest.importorskip('torch')
    from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter, TASK_INSTRUCTION

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs['do_sample'] is True
            assert kwargs['temperature'] == .8 and kwargs['top_p'] == .9 and kwargs['top_k'] == 30
            token = torch.randint(10, 10000, (1,)).item()
            return torch.cat((kwargs['input_ids'], torch.tensor([[token, 2]])), dim=1)

    rows, _, _ = inputs
    row = {key: rows[0][key] for key in ('id', 'source_group_id', 'audio_path', 'audio_sha256', 'audio_window_seconds')}
    processor, model = TinyProcessor(), Model()
    adapter = PlannerAdapter(family, model, processor)
    state = torch.random.get_rng_state().clone()
    outputs = [sample_native(adapter, row, seed=seed, temperature=.8, top_p=.9, top_k=30,
                             max_new_tokens=16) for seed in range(4)]
    assert len({value['raw_text'] for value in outputs}) > 1
    assert torch.equal(state, torch.random.get_rng_state())
    assert sample_native(adapter, row, seed=0, temperature=.8, top_p=.9, top_k=30,
                         max_new_tokens=16) == outputs[0]
    assert all(value['n_samples'] == 81000 and value['stop_reason'] == 'eos' for value in outputs)
    if family == 'omni_captioner':
        assert processor.messages == [{'role': 'user', 'content': [{'type': 'audio', 'audio': 'native-waveform'}]}]
        assert TASK_INSTRUCTION not in str(processor.messages)
    elif family == 'music_flamingo':
        assert TASK_INSTRUCTION in str(processor.messages)
        assert processor.calls[-1]['audio_kwargs'] == {'padding': 'max_length'}
    else:
        assert processor.messages == TASK_INSTRUCTION
        assert 'audio_input_mask' in model.calls[-1]
    short = Path(row['audio_path']).with_name('short.wav')
    write_wave(short, frames=48000)
    short_row = dict(row, audio_path=str(short), audio_sha256=file_sha(short))
    with pytest.raises(ValueError, match='complete 5.0625-second'):
        sample_native(adapter, short_row, seed=0, max_new_tokens=16)


def test_native_sampling_does_not_accept_prefix_change_or_context_truncation(inputs):
    torch = pytest.importorskip('torch')
    from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter
    class ChangedPrefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.generation_config = SimpleNamespace(eos_token_id=2)
        def generate(self, **kwargs):
            return torch.tensor([[8, 7, 9, 17, 2]])
    row = inputs[0][0]
    adapter = PlannerAdapter('moss', ChangedPrefix(), TinyProcessor())
    with pytest.raises(ValueError, match='changed its input prefix'):
        sample_native(adapter, row, seed=0, max_new_tokens=16)
    adapter.max_length = 5
    with pytest.raises(ValueError, match='budget exceeds context'):
        sample_native(adapter, row, seed=0, max_new_tokens=16)
