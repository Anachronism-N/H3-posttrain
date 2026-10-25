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


def mock_runtime():
    source = dict(path='mock-runtime-fixture.py', sha256='d' * 64, size_bytes=1)
    return dict(schema='music_planner_sampling_runtime_v2', adapter_source_ref=source,
                moss_cache=dict(requested_ref=source, applied_ref=source, status='applied'),
                moss_position_buffer=dict(requested_ref=source, applied_ref=None, status='not_present',
                                          canonical_tensor_ref=None),
                eos_bridge=dict(requested_ref=source, applied_ref=None, status='native_eos_already_valid'))


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
                runtime_provenance=mock_runtime(),
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
    eos_token_id = 2

    def encode(self, text, **kwargs):
        assert text == '<|im_end|>'
        return [2]

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
            self.config = SimpleNamespace(model_type='moss_music')
            self._planner_native_forward = self.forward
            self.calls = []

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return dict(input_ids=input_ids, **kwargs)

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
    assert all(value['runtime_provenance']['moss_cache']['status'] ==
               ('applied' if family == 'moss' else 'not_moss') for value in outputs)
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
    overlong = short.with_name('overlong.wav')
    write_wave(overlong, frames=121505)
    with pytest.raises(ValueError, match='complete 5.0625-second'):
        sample_native(adapter, dict(row, audio_path=str(overlong), audio_sha256=file_sha(overlong)),
                      seed=0, max_new_tokens=16)


def test_native_sampling_does_not_accept_prefix_change_or_context_truncation(inputs):
    torch = pytest.importorskip('torch')
    from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter
    class ChangedPrefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.config = SimpleNamespace(model_type='moss_music')
            self._planner_native_forward = self.forward
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return dict(input_ids=input_ids, **kwargs)
        def generate(self, **kwargs):
            return torch.tensor([[8, 7, 9, 17, 2]])
    row = inputs[0][0]
    adapter = PlannerAdapter('moss', ChangedPrefix(), TinyProcessor())
    with pytest.raises(ValueError, match='changed its input prefix'):
        sample_native(adapter, row, seed=0, max_new_tokens=16)
    adapter.max_length = 5
    with pytest.raises(ValueError, match='budget exceeds context'):
        sample_native(adapter, row, seed=0, max_new_tokens=16)


def test_sampling_actually_executes_eos_bridge_without_mutating_config(inputs):
    torch = pytest.importorskip('torch')
    from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter
    class MissingEOS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.generation_config = SimpleNamespace(eos_token_id=None)
            self.calls = []
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            tokens = [17, 2] if kwargs.get('eos_token_id') == [2] else [17, 2, 9, 2]
            return torch.cat((kwargs['input_ids'], torch.tensor([tokens])), dim=1)
    model = MissingEOS()
    result = sample_native(PlannerAdapter('omni_captioner', model, TinyProcessor()),
                           inputs[0][0], seed=7, max_new_tokens=16)
    assert model.calls[0]['eos_token_id'] == [2]
    assert result['new_token_ids'] == [17, 2]
    assert model.generation_config.eos_token_id is None
    receipt = result['runtime_provenance']['eos_bridge']
    assert receipt['status'] == 'applied'
    assert receipt['applied_ref'] == receipt['requested_ref']
    assert receipt['applied_ref']['sha256'] == file_sha(receipt['applied_ref']['path'])


def test_sampling_actually_executes_moss_cached_decode_patch(inputs):
    torch = pytest.importorskip('torch')
    from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter
    class CachedMoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(model_type='moss_music')
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self._planner_native_forward = self.forward
            self.positions = []
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            self.positions.append(kwargs.get('cache_position'))
            return dict(input_ids=input_ids, **kwargs)
        def generate(self, **kwargs):
            ids = kwargs['input_ids']
            cached = SimpleNamespace(get_seq_length=lambda: ids.shape[1])
            self.prepare_inputs_for_generation(torch.cat((ids, torch.tensor([[17]])), dim=1),
                                               past_key_values=cached)
            return torch.cat((ids, torch.tensor([[17, 2]])), dim=1)
    model = CachedMoss()
    result = sample_native(PlannerAdapter('moss', model, TinyProcessor()),
                           inputs[0][0], seed=7, max_new_tokens=16)
    assert model.positions[0].tolist() == [3]
    receipt = result['runtime_provenance']['moss_cache']
    assert receipt['status'] == 'applied'
    assert receipt['applied_ref']['sha256'] == file_sha(receipt['applied_ref']['path'])


def test_tokens_generated_after_first_eos_never_become_valid_success(inputs, policy):
    def bad_eos(row, **kwargs):
        result = mock_sampler(row, **kwargs)
        result.update(new_token_ids=[17, 2, 9, 2], stop_reason='tokens_after_eos')
        return result
    pool = generate_candidates(inputs[0][:1], inputs[1], inputs[2], sampler=bad_eos)
    assert pool['candidates'][0]['generation']['new_token_ids'] == [17, 2, 9, 2]
    assert not pool['candidates'][0]['diagnostics']['format_eligible']
    assert not export_reviewed(pool, policy, reviews_for(pool, policy))['pseudo_sft']


def test_blind_package_hides_model_rank_scores_and_binds_real_audio(pool, policy):
    from scripts_omnidance.exp.ms.music_planner_search_harness import blind_package, validate_blind_package
    public, private = blind_package(pool, policy, secret='1' * 64)
    serialized = json.dumps(public)
    assert pool['generator']['model_id'] not in serialized
    for key in ('candidate_sha256', 'source_id', 'index', 'format_eligible', 'decoding', 'generator'):
        assert '"' + key + '"' not in serialized
    assert len({row['opaque_id'] for row in public['candidates']}) == len(pool['candidates'])
    assert all(file_sha(row['audio_ref']['path']) == row['audio_ref']['sha256'] for row in public['candidates'])
    assert validate_blind_package(pool, policy, public, private)
    assert blind_package(pool, policy, secret='1' * 64) == (public, private)


def blind_response_for(pool, policy, public, private):
    by_hash = {row['candidate_sha256']: row for row in reviews_for(pool, policy)['reviews']}
    response = dict(schema='music_planner_blind_reviews_v1', request_sha256=public['request_sha256'],
                    judge=policy['judge'], rubric=policy['rubric'], reviews=[])
    for binding in private['bindings']:
        row = by_hash[binding['candidate_sha256']]
        response['reviews'].append(dict({key: value for key, value in row.items()
                                         if key not in ('candidate_sha256', 'source_id')},
                                       opaque_id=binding['opaque_id']))
    return response


def test_blind_reviews_restore_strict_candidate_bindings_and_detect_tampering(pool, policy):
    from scripts_omnidance.exp.ms.music_planner_search_harness import blind_package, import_blind_reviews
    public, private = blind_package(pool, policy, secret='2' * 64)
    response = blind_response_for(pool, policy, public, private)
    restored, receipt = import_blind_reviews(pool, policy, public, private, response)
    assert restored == reviews_for(pool, policy)
    assert receipt['imported_reviews_sha256'] == object_sha(restored)
    altered = copy.deepcopy(public)
    altered['candidates'][0]['proposed_plan_text'] += 'tampered'
    with pytest.raises(ValueError, match='exact candidate/audio/policy'):
        import_blind_reviews(pool, policy, altered, private, response)
    response['reviews'][0]['audio_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='another audio'):
        import_blind_reviews(pool, policy, public, private, response)


@pytest.mark.parametrize('change', ('opaque', 'duplicate', 'judge', 'mapping'))
def test_blind_import_rejects_unknown_duplicate_or_stale_binding(pool, policy, change):
    from scripts_omnidance.exp.ms.music_planner_search_harness import blind_package, import_blind_reviews
    public, private = blind_package(pool, policy, secret='3' * 64)
    response = blind_response_for(pool, policy, public, private)
    if change == 'opaque':
        response['reviews'][0]['opaque_id'] = '0' * 32
    elif change == 'duplicate':
        response['reviews'].append(response['reviews'][0])
    elif change == 'judge':
        response['judge'] = dict(response['judge'], version='old')
    else:
        private['bindings'][0]['source_id'] = 'other'
    with pytest.raises(ValueError):
        import_blind_reviews(pool, policy, public, private, response)


def test_budget_report_separates_train_dev_without_inventing_quality_reward(pool, policy):
    from scripts_omnidance.exp.ms.music_planner_search_harness import search_report
    result = search_report(pool, policy)
    assert result['baseline_kind'] == 'single_stochastic_sample_index0_not_fair_greedy'
    assert set(result['splits']) == {'train', 'dev'}
    assert result['splits']['train']['sources'] == result['splits']['dev']['sources'] == 1
    assert result['splits']['train']['candidate_calls'] == 3
    assert result['splits']['train']['search_new_tokens'] == 6
    assert result['splits']['train']['baseline_new_tokens'] == 2
    assert result['format_and_length_are_diagnostics_not_music_rewards'] is True
    assert all(row['selected_candidate_sha256'] is None for row in result['sources'])
    assert result['splits']['dev']['independent_audit_source_bootstrap'] is None


def test_paired_independent_audit_can_reject_apparent_selection_gain(pool, policy):
    from scripts_omnidance.exp.ms.music_planner_search_harness import search_report
    selection = reviews_for(pool, policy)
    index = {item['candidate_sha256']: item['index'] for item in pool['candidates']}
    for row in selection['reviews']:
        top = index[row['candidate_sha256']] == 1
        row['verdict'] = 'approve' if top else 'reject'
        row['scores'] = {key: 5. if top else 2. for key in DIMENSIONS}
    audit_policy = copy.deepcopy(policy)
    audit_policy['judge']['id'] = 'independent-auditor-2'
    audit = reviews_for(pool, audit_policy)
    for row in audit['reviews']:
        row['scores'] = {key: 1. if index[row['candidate_sha256']] == 1 else 5. for key in DIMENSIONS}
    result = search_report(pool, policy, selection, audit_policy=audit_policy, audit_reviews=audit)
    for split in ('train', 'dev'):
        stats = result['splits'][split]
        assert stats['descriptive_same_review_mean_delta'] == 3.
        assert stats['same_review_generalization_ci'] is None
        assert stats['independently_audited_pairs'] == 1
        assert stats['independent_audit_source_bootstrap']['mean'] == -4.
        assert stats['independent_audit_source_bootstrap']['small_sample_warning'] is True
    assert result == search_report(pool, policy, selection, audit_policy=audit_policy, audit_reviews=audit)
    with pytest.raises(ValueError, match='independent of selection'):
        search_report(pool, policy, selection, audit_policy=policy, audit_reviews=selection)


def test_blind_cli_preflights_both_outputs_before_writing(tmp_path, pool, policy):
    pool_path, policy_path = tmp_path / 'pool.json', tmp_path / 'policy.json'
    write_json(pool_path, pool)
    write_json(policy_path, policy)
    private, public = tmp_path / 'private.json', tmp_path / 'public.json'
    private.write_text('already exists', encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'scripts_omnidance.exp.ms.music_planner_search_harness',
                             'blind-request', '--pool', str(pool_path), '--policy', str(policy_path),
                             '--public-out', str(public), '--private-out', str(private)],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0 and not public.exists()
    assert private.read_text(encoding='utf-8') == 'already exists'


def test_blind_request_import_report_cli_complete_round_trip(tmp_path, pool, policy):
    paths = {name: tmp_path / (name + '.json') for name in
             ('pool', 'policy', 'public', 'private', 'response', 'restored', 'receipt', 'report')}
    write_json(paths['pool'], pool)
    write_json(paths['policy'], policy)
    prefix = [sys.executable, '-m', 'scripts_omnidance.exp.ms.music_planner_search_harness']
    common = ['--pool', str(paths['pool']), '--policy', str(paths['policy'])]
    result = subprocess.run(prefix + ['blind-request'] + common +
                            ['--public-out', str(paths['public']), '--private-out', str(paths['private'])],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    public, private = read_json(paths['public']), read_json(paths['private'])
    write_json(paths['response'], blind_response_for(pool, policy, public, private))
    result = subprocess.run(prefix + ['import-blind'] + common +
                            ['--public', str(paths['public']), '--private', str(paths['private']),
                             '--response', str(paths['response']), '--out', str(paths['restored']),
                             '--receipt-out', str(paths['receipt'])], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(prefix + ['report'] + common + ['--reviews', str(paths['restored']),
                                                            '--out', str(paths['report'])],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = read_json(paths['report'])
    assert report['splits']['train']['selected_sources'] == 1
    assert report['splits']['dev']['same_review_generalization_ci'] is None
    assert read_json(paths['receipt'])['imported_reviews_sha256'] == object_sha(read_json(paths['restored']))


def collection_sources():
    runtime = mock_runtime()
    return dict(zip(('music_planner_adapters.py', 'music_planner_moss_generation_patch.py',
                     'music_planner_moss_position_buffer_patch.py', 'music_planner_generation_eos_patch.py'),
                    (runtime['adapter_source_ref'], runtime['moss_cache']['requested_ref'],
                     runtime['moss_position_buffer']['requested_ref'], runtime['eos_bridge']['requested_ref'])))


def test_interrupted_collection_reuses_completed_raw_and_publishes_only_when_complete(inputs, tmp_path):
    from scripts_omnidance.exp.ms.music_planner_search_collection import collect_candidates
    rows, isolation, generator = inputs
    journal, out = tmp_path / 'journal', tmp_path / 'pool.json'
    calls = []
    def interrupted(row, **kwargs):
        calls.append(kwargs['seed'])
        if len(calls) == 2:
            raise KeyboardInterrupt('intentional interrupted CPU collection')
        return mock_sampler(row, **kwargs)
    options = dict(journal_dir=journal, out=out, count=3, sources=collection_sources())
    with pytest.raises(KeyboardInterrupt):
        collect_candidates(rows[:1], isolation, generator, sampler=interrupted, **options)
    assert not out.exists() and read_json(journal / 'status.json')['state'] == 'incomplete'
    completed_file = next((journal / 'candidates').glob('*.json'))
    original = completed_file.read_bytes()
    resumed_calls = []
    def resumed(row, **kwargs):
        resumed_calls.append(kwargs['seed'])
        return mock_sampler(row, **kwargs)
    result = collect_candidates(rows[:1], isolation, generator, sampler=resumed, resume=True, **options)
    assert calls[0] not in resumed_calls and len(resumed_calls) == 2
    assert completed_file.read_bytes() == original
    assert result == read_json(out)
    assert len(list((journal / 'attempts').glob('*.failed.json'))) == 1
    before = out.read_bytes()
    collect_candidates(rows[:1], isolation, generator, sampler=lambda *a, **k: pytest.fail('complete resume must not sample'),
                       resume=True, **options)
    assert out.read_bytes() == before


def test_resume_promotes_durable_raw_after_crash_before_complete_marker(inputs, tmp_path, monkeypatch):
    from scripts_omnidance.exp.ms import music_planner_search_collection as collection
    rows, isolation, generator = inputs
    journal, out = tmp_path / 'journal', tmp_path / 'pool.json'
    create = collection.durable_create
    def crash_after_raw(path, value):
        if Path(path).parent.name == 'candidates':
            raise KeyboardInterrupt('crash after raw flush before complete marker')
        return create(path, value)
    monkeypatch.setattr(collection, 'durable_create', crash_after_raw)
    options = dict(journal_dir=journal, out=out, count=2, sources=collection_sources())
    with pytest.raises(KeyboardInterrupt):
        collection.collect_candidates(rows[:1], isolation, generator, sampler=mock_sampler, **options)
    raw = next((journal / 'attempts').glob('*.raw.json'))
    bytes_before, seed_done = raw.read_bytes(), read_json(raw)['seed']
    monkeypatch.setattr(collection, 'durable_create', create)
    calls = []
    def sampler(row, **kwargs):
        calls.append(kwargs['seed'])
        return mock_sampler(row, **kwargs)
    collection.collect_candidates(rows[:1], isolation, generator, sampler=sampler, resume=True, **options)
    assert seed_done not in calls and len(calls) == 1
    assert raw.read_bytes() == bytes_before


@pytest.mark.parametrize('change', ('params', 'seed', 'model', 'source', 'runtime', 'completed_raw'))
def test_resume_rejects_changed_contract_or_completed_evidence_before_new_sampling(inputs, tmp_path, change):
    from scripts_omnidance.exp.ms.music_planner_search_collection import collect_candidates
    rows, isolation, generator = inputs
    options = dict(journal_dir=tmp_path / 'journal', out=tmp_path / 'pool.json', count=2, sources=collection_sources())
    collect_candidates(rows[:1], isolation, generator, sampler=mock_sampler, **options)
    if change == 'params':
        options['sampling'] = dict(temperature=.5, top_p=.95, top_k=50, max_new_tokens=2048)
    elif change == 'seed':
        options['seed'] = 17
    elif change == 'model':
        generator = dict(generator, model_sha256='f' * 64)
    elif change == 'source':
        rows[0]['source_group_id'] = 'another source group'
    elif change == 'runtime':
        options['sources'] = copy.deepcopy(options['sources'])
        options['sources']['music_planner_adapters.py']['sha256'] = 'f' * 64
    else:
        path = next((options['journal_dir'] / 'candidates').glob('*.json'))
        data = read_json(path)
        data['candidate']['generation']['raw_text'] += 'tampered'
        path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        collect_candidates(rows[:1], isolation, generator,
                           sampler=lambda *a, **k: pytest.fail('identity mismatch must precede sampling'),
                           resume=True, **options)


def test_invalid_durable_raw_is_preserved_and_blocks_resampling(inputs, tmp_path, monkeypatch):
    from scripts_omnidance.exp.ms import music_planner_search_collection as collection
    rows, isolation, generator = inputs
    options = dict(journal_dir=tmp_path / 'journal', out=tmp_path / 'pool.json', count=2, sources=collection_sources())
    def invalid(row, **kwargs):
        result = mock_sampler(row, **kwargs)
        result['seed'] += 1
        return result
    with pytest.raises(ValueError, match='seed'):
        collection.collect_candidates(rows[:1], isolation, generator, sampler=invalid, **options)
    raw = next((options['journal_dir'] / 'attempts').glob('*.raw.json'))
    original = raw.read_bytes()
    with pytest.raises(ValueError):
        collection.collect_candidates(rows[:1], isolation, generator, resume=True,
            sampler=lambda *a, **k: pytest.fail('must not sample over invalid durable raw'), **options)
    assert raw.read_bytes() == original and not options['out'].exists()


def test_direct_protocol_does_not_accept_short_plan_xml_as_action_caption(inputs):
    def xml(row, **kwargs):
        result = mock_sampler(row, **kwargs)
        result['raw_text'] = '<music_evidence>Music.</music_evidence><caption>' + result['raw_text'] + '</caption>'
        return result
    pool = generate_candidates(inputs[0][:1], inputs[1], inputs[2], sampler=xml)
    assert all(not item['diagnostics']['format_eligible'] for item in pool['candidates'])


def test_captioner_real_torch_sampling_resumes_with_native_audio_only_and_eos(inputs, tmp_path):
    torch = pytest.importorskip('torch')
    from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter
    from scripts_omnidance.exp.ms.music_planner_search_collection import collect_candidates
    class Captioner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.generation_config = SimpleNamespace(eos_token_id=None)
        def generate(self, **kwargs):
            assert kwargs['do_sample'] is True and kwargs['eos_token_id'] == [2]
            token = int(torch.randint(10, 10000, (1,)).item())
            return torch.cat((kwargs['input_ids'], torch.tensor([[token, 2]])), dim=1)
    rows, isolation, generator = inputs
    generator = dict(generator, family='omni_captioner')
    first = PlannerAdapter('omni_captioner', Captioner(), TinyProcessor())
    calls = []
    def interrupt(row, **kwargs):
        calls.append(kwargs['seed'])
        if len(calls) == 2:
            raise KeyboardInterrupt('between actual CPU sampled candidates')
        return sample_native(first, row, **kwargs)
    options = dict(journal_dir=tmp_path / 'journal', out=tmp_path / 'pool.json', count=3)
    state = torch.random.get_rng_state().clone()
    with pytest.raises(KeyboardInterrupt):
        collect_candidates(rows[:1], isolation, generator, sampler=interrupt, **options)
    second = PlannerAdapter('omni_captioner', Captioner(), TinyProcessor())
    resumed = []
    def sampler(row, **kwargs):
        resumed.append(kwargs['seed'])
        return sample_native(second, row, **kwargs)
    result = collect_candidates(rows[:1], isolation, generator, sampler=sampler, resume=True, **options)
    assert len(resumed) == 2 and calls[0] not in resumed
    assert all(c['generation']['new_token_ids'][-1] == 2 for c in result['candidates'])
    assert second.processor.messages == [{'role': 'user', 'content': [{'type': 'audio', 'audio': 'native-waveform'}]}]
    assert torch.equal(state, torch.random.get_rng_state())


def test_moss_collection_restores_position_buffer_and_checks_real_forward_logits(inputs, tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter
    from scripts_omnidance.exp.ms import music_planner_moss_position_buffer_patch as position
    class Moss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(model_type='moss_music')
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self._planner_native_forward = self.forward
            self.restored, self.nonfinite, self.draws = False, False, 0
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return dict(input_ids=input_ids, **kwargs)
        def forward(self, input_ids):
            assert self.restored
            logits = torch.zeros(1, input_ids.shape[1], 8)
            if self.nonfinite:
                logits[0, -1, 0] = float('nan')
            return SimpleNamespace(logits=logits)
        def generate(self, **kwargs):
            self(kwargs['input_ids'])
            self.draws += 1
            token = int(torch.randint(10, 10000, (1,)).item())
            return torch.cat((kwargs['input_ids'], torch.tensor([[token, 2]])), dim=1)
    model = Moss()
    adapter = PlannerAdapter('moss', model, TinyProcessor())
    # Attach after constructing the boundary fixture; the native runtime itself
    # is not replaced or claimed to be a pretrained MOSS model.
    model.audio_encoder = torch.nn.Identity()
    with pytest.raises(ValueError, match='audited official model path'):
        sample_native(adapter, inputs[0][0], seed=1, max_new_tokens=16)
    adapter._sampling_official_model_path = tmp_path
    restores = []
    def restore(current, official):
        assert current is adapter and official == tmp_path
        restores.append(True)
        model.restored = True
        return dict(applied_position_buffer_patch_ref=position.position_buffer_patch_ref(),
                    canonical_tensor_ref=dict(shape=[640], dtype='torch.float32', sha256='c' * 64))
    monkeypatch.setattr(position, 'restore_moss_position_buffer', restore)
    for seed in (1, 2):
        result = sample_native(adapter, inputs[0][0], seed=seed, max_new_tokens=16)
        assert result['runtime_provenance']['moss_position_buffer']['status'] == 'restored'
        assert result['finite_native_logits_forward_count'] == 1
    assert len(restores) == 1 and not model._forward_hooks
    model.nonfinite = True
    with pytest.raises(FloatingPointError, match='non-finite last-token logits'):
        sample_native(adapter, inputs[0][0], seed=3, max_new_tokens=16)
    assert model.draws == 2 and not model._forward_hooks
