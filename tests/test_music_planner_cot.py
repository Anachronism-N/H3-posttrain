"""Real CPU autograd with mock native processors; no pretrained quality claim."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts_omnidance.exp.ms import music_planner_cot as cot


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def example(tmp_path, name='song', split='train', frequency=220):
    audio = tmp_path / (name + '.wav')
    samples = .2 * np.sin(np.arange(121500) * (2 * np.pi * frequency / 24000))
    sf.write(audio, samples, 24000, subtype='PCM_16')
    caption = '[0-2s] Steps left while lifting both arms. [2-5.062s] Lowers arms and turns right.'
    sha = hashlib.sha256(audio.read_bytes()).hexdigest()
    return dict(id=name, source_group_id=name, split=split, audio_path=str(audio), audio_sha256=sha,
        audio_window_seconds=[0., 5.0625], target_text=caption, caption_version='fixture-v1',
        target_windows=[dict(start=0., end=2., caption='Steps left while lifting both arms.'),
                        dict(start=2., end=5.0625, caption='Lowers arms and turns right.')],
        semantic_approval=False, human_review_completed=False,
        music_evidence=dict(text='Steady percussion with a louder ending.', provenance=dict(
            source='human_audio_review', human_approved=True, reviewer='fixture-only', version='v1',
            input_modalities=['audio'], audio_sha256=sha)),
        short_plan=dict(text='Use restrained steps first, then increase the movement range.', provenance=dict(
            source='human_choreography_plan', human_approved=True, reviewer='fixture-only', version='v1',
            audio_sha256=sha, target_caption_sha256=hashlib.sha256(caption.encode()).hexdigest())))


class BoundaryTokenizer:
    """Merge newline+'<' so concatenating separate segment encodings is wrong."""
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        ids, offsets, position = [], [], 0
        while position < len(text):
            if text[position:position + 2] == '\n<':
                ids.append(244)
                offsets.append((position, position + 2))
                position += 2
            else:
                ids.append(ord(text[position]) + 8)
                offsets.append((position, position + 1))
                position += 1
        return dict(input_ids=ids, offset_mapping=offsets)

    def encode(self, text, **kwargs):
        return [2] if text == '<|im_end|>' else self(text)['input_ids']

    def decode(self, ids, **kwargs):
        return ''.join('\n<' if value == 244 else chr(value - 8) for value in ids if value != 2)


class MockProcessor:
    tokenizer = BoundaryTokenizer()
    audio_token_id = 3

    def _build_default_prompt(self, instruction, has_audio):
        return 'native audio. ' + instruction + ' assistant: '

    def apply_chat_template(self, messages, **kwargs):
        parts = messages[0]['content']
        return 'native audio. ' + ''.join(p['text'] for p in parts if p['type'] == 'text') + ' assistant: '

    def __call__(self, text, **kwargs):
        text = text[0] if isinstance(text, list) else text
        ids = torch.tensor([self.tokenizer.encode(text)])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), input_features=torch.ones(1, 1))

    def save_pretrained(self, path):
        (Path(path) / 'mock_processor.json').write_text('{}')


class TinyPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(123)
        self.base_embedding = torch.nn.Parameter(torch.randn(256, 8, generator=generator), requires_grad=False)
        self.lora_A = torch.nn.Parameter(.05 * torch.randn(8, 256, generator=generator))
        self.audio_projector = torch.nn.Parameter(.05 * torch.randn(1, 8, generator=generator))
        self.config = SimpleNamespace(use_cache=True)
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.generation_text = ''
        self.calls = []

    def forward(self, input_ids, attention_mask=None, input_features=None):
        hidden = self.base_embedding[input_ids] + (input_features @ self.audio_projector).unsqueeze(1)
        return SimpleNamespace(logits=hidden @ self.lora_A)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        suffix = BoundaryTokenizer().encode(self.generation_text) + [2]
        return torch.cat((kwargs['input_ids'], torch.tensor([suffix])), dim=1)

    def save_pretrained(self, path, state_dict=None, **kwargs):
        torch.save(state_dict, Path(path) / 'mock_native.pt')


@pytest.fixture
def adapter():
    return cot.ShortPlanAdapter('omni_captioner', TinyPlanner(), MockProcessor(), max_length=8192)


@pytest.mark.parametrize('name', ['music_evidence', 'short_plan'])
def test_missing_annotations_never_synthesized(tmp_path, name):
    row = example(tmp_path)
    del row[name]
    with pytest.raises(ValueError, match='Missing reviewed'):
        cot.validate_annotations(row)


@pytest.mark.parametrize('change', ['video', 'approval', 'audio_binding', 'target_binding', 'nan', 'tags'])
def test_unreviewed_leaked_or_unbound_labels_fail(tmp_path, change):
    row = example(tmp_path)
    if change == 'video':
        row['music_evidence']['provenance']['input_modalities'] = ['audio', 'video']
    elif change == 'approval':
        row['music_evidence']['provenance']['human_approved'] = False
    elif change == 'audio_binding':
        row['music_evidence']['provenance']['audio_sha256'] = 'f' * 64
    elif change == 'target_binding':
        row['short_plan']['provenance']['target_caption_sha256'] = 'f' * 64
    elif change == 'nan':
        row['music_evidence']['confidence'] = float('nan')
    else:
        row['music_evidence']['text'] = '<caption>leak</caption>'
    with pytest.raises(ValueError):
        cot.validate_annotations(row)


def test_full_tokenization_boundary_ownership_and_eos(tmp_path, adapter):
    row = example(tmp_path)
    text, bounds = cot.serialize_target(row)
    tokenizer = adapter.tokenizer
    ids, ownership, metadata = cot.tokenized_target(tokenizer, row, [2])
    separately = [value for left, right in bounds for value in tokenizer.encode(text[left:right])]
    assert separately != ids[:-1]  # This fixture actually exercises cross-boundary merges.
    assert ids[:-1] == tokenizer.encode(text)
    assert metadata['boundary_crossing_tokens'] == 2
    assert ownership[-1] == 2 and ids[-1] == 2 and ids.count(2) == 1
    batch = adapter.collate([row])
    prefix = adapter.prefix(row)
    length = prefix['input_ids'].shape[1]
    assert torch.equal(batch['input_ids'][:, :length], prefix['input_ids'])
    assert batch['labels'][:, :length].eq(-100).all()
    assert batch['segment_ids'][:, :length].eq(-1).all()
    assert batch['labels'][0, length:].tolist() == ids
    assert batch['segment_ids'][0, length:].tolist() == ownership


def test_real_segment_gradients_eos_and_causal_prefix_mask():
    labels = torch.tensor([[-100, -100, 4, 5, 6, 2]])
    sections = torch.tensor([[-1, -1, 0, 1, 2, 2]])
    logits = torch.zeros(1, 6, 8, requires_grad=True)
    loss, sums, counts = cot.segment_nll(logits, labels, sections, (0., 0., 1.))
    assert counts.tolist() == [1, 1, 2]
    torch.testing.assert_close(loss, sums[2] / 2)
    loss.backward()
    assert logits.grad[:, :3].eq(0).all()  # Prefix and zero-weight music/plan do not contribute.
    assert logits.grad[0, 4, 2] < 0  # Native EOS is supervised from the preceding position.
    assert logits.grad[:, -1].eq(0).all()  # Final token has no next-token target.


def test_segment_loss_formula_and_invalid_masks():
    torch.manual_seed(1)
    logits = torch.randn(1, 6, 8, requires_grad=True)
    labels = torch.tensor([[-100, -100, 4, 5, 6, 2]])
    sections = torch.tensor([[-1, -1, 0, 1, 2, 2]])
    loss, sums, counts = cot.segment_nll(logits, labels, sections, (.2, .3, 1.))
    expected = (.2 * sums[0] + .3 * sums[1] + sums[2] / 2) / 1.5
    torch.testing.assert_close(loss, expected)
    with pytest.raises(ValueError, match='ownership'):
        cot.segment_nll(logits, labels, torch.zeros_like(labels))
    with pytest.raises(ValueError, match='weights'):
        cot.segment_nll(logits, labels, sections, (float('nan'), .3, 1.))


@pytest.mark.parametrize('family', cot.FAMILIES)
def test_prompt_policy_and_generation_never_reads_gold(tmp_path, family):
    row = example(tmp_path)
    adapter = cot.ShortPlanAdapter(family, TinyPlanner(), MockProcessor())
    target, _ = cot.serialize_target(row)
    adapter.model.generation_text = target
    prompt = adapter._prompt()
    assert (cot.INSTRUCTION in prompt) is (family != 'omni_captioner')
    assert 'system' not in prompt

    class AudioOnly(dict):
        def __getitem__(self, key):
            assert key not in ('target_text', 'target_windows', 'music_evidence', 'short_plan', 'genre')
            return super().__getitem__(key)

    result = adapter.generate_caption(AudioOnly(row), max_new_tokens=1024)
    assert result['caption_valid'] is True
    assert result['caption'] == row['target_text']
    assert result['temporal']['matches_fixed_windows'] is True
    actual = adapter.model.calls[-1]['input_ids']
    assert actual.tolist() == adapter.prefix(row)['input_ids'].tolist()
    assert 'segment_ids' not in adapter.model.calls[-1]
    assert 'labels' not in adapter.model.calls[-1]


@pytest.mark.parametrize('raw', [
    '<caption>[0-2s] step</caption>',
    '<music_evidence>x</music_evidence><short_plan>x</short_plan><caption>[0-1s] step</caption>',
    '<music_evidence>x</music_evidence><short_plan>x</short_plan><caption>[0-2s] step [2-5.062s]</caption>',
])
def test_incomplete_or_wrong_time_generation_is_invalid(tmp_path, adapter, raw):
    adapter.model.generation_text = raw
    result = adapter.generate_caption(example(tmp_path), max_new_tokens=1024)
    assert result['caption_valid'] is False and result['caption'] is None
    assert result['raw_text'] == raw  # No rewrite or hidden repair.


def test_context_overflow_and_slow_tokenizer_fail_explicitly(tmp_path, adapter):
    row = example(tmp_path)
    adapter.max_length = adapter.prefix(row)['input_ids'].shape[1] + 1
    with pytest.raises(ValueError, match='truncation forbidden'):
        adapter.collate([row])
    class NoOffsets(BoundaryTokenizer):
        def __call__(self, *args, **kwargs):
            raise NotImplementedError
    with pytest.raises(ValueError, match='offset mapping'):
        cot.tokenized_target(NoOffsets(), row, [2])


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


@pytest.mark.parametrize('mode', cot.MODES)
def test_training_entry_two_updates_checkpoint_reload_resume_and_generate(tmp_path, monkeypatch, mode):
    train = tmp_path / 'train.jsonl'
    val = tmp_path / 'val.jsonl'
    rows = [example(tmp_path, 'a', frequency=220), example(tmp_path, 'b', frequency=330)]
    validation = example(tmp_path, 'c', 'val', frequency=440)
    if mode == 'direct-caption':
        for row in rows + [validation]:
            del row['music_evidence']
            del row['short_plan']
    write_rows(train, rows)
    write_rows(val, [validation])
    local_model = tmp_path / 'model'
    local_model.mkdir()
    (local_model / 'config.json').write_text('{}')
    (local_model / 'fixture.bin').write_bytes(b'not-pretrained: loader injected by test')
    loaded = []

    def native_factory(family, model_path, checkpoint=None, **kwargs):
        model = TinyPlanner()
        if checkpoint:
            cot.base.restore_trainable(model, torch.load(Path(checkpoint) / 'trainable.pt', weights_only=True))
            model.generation_text = cot.serialize_target(rows[0], mode)[0]
        native = cot.PlannerAdapter(family, model, MockProcessor(), max_length=kwargs.get('max_length', 8192))
        loaded.append(native)
        return native

    monkeypatch.setattr(cot, 'load_planner_adapter', native_factory)
    shared = ['--family', 'omni_captioner', '--model', str(local_model), '--train', str(train),
              '--val', str(val), '--out', str(tmp_path / 'run'), '--device', 'cpu', '--dtype', 'float32',
              '--no-gradient-checkpointing', '--epochs', '2', '--verify-reload', '--mode', mode,
              '--loss-chunk-size', '32']
    before = copy.deepcopy(TinyPlanner().state_dict())
    first = cot.main(['train'] + shared + ['--max-updates', '2'])
    assert first['cursor']['update'] == 2 and first['reload_verified'] is True
    after = loaded[-1].model.state_dict()
    assert not torch.equal(before['lora_A'], after['lora_A'])
    assert not torch.equal(before['audio_projector'], after['audio_projector'])
    assert torch.equal(before['base_embedding'], after['base_embedding'])
    assert first['test_used'] is False and first['quality_accepted'] is False
    assert first['schema'] == cot.mode_schema(mode) and first['mode'] == mode
    assert first['invocation_optimizer_updates'] == 2
    assert first['supervised_tokens_total'] > 0
    if mode == 'direct-caption':
        assert first['supervised_segment_tokens']['music_evidence'] == 0
        assert first['supervised_segment_tokens']['short_plan'] == 0
    else:
        assert all(value > 0 for value in first['supervised_segment_tokens'].values())
    config = json.loads((tmp_path / 'run/run_config.json').read_text())
    assert set(config['code_sha256']) == {'music_planner_cot.py', 'music_planner_train.py',
                                        'music_planner_adapters.py', 'music_planner_eval.py',
                                        'music_planner_segment_loss.py'}
    assert (Path(first['checkpoint']) / 'adapter/short_plan_adapter.json').is_file()
    assert json.loads((tmp_path / 'run/best.json').read_text())['selection_metric'] == 'val_caption_segment_nll'
    second = cot.main(['train'] + shared + ['--max-updates', '3', '--resume', first['checkpoint']])
    assert second['cursor']['update'] == 3 and second['cursor']['source_exposures'] == 3
    assert second['cursor']['epoch'] == 1
    assert second['invocation_optimizer_updates'] == 1
    assert second['supervised_tokens_total'] > first['supervised_tokens_total']
    val_stats = json.loads((Path(first['checkpoint']) / 'validation.json').read_text())
    if mode == 'direct-caption':
        assert val_stats['segment_nll']['music_evidence'] is None
        assert val_stats['segment_nll']['short_plan'] is None
        assert val_stats['segment_tokens']['music_evidence'] == val_stats['segment_tokens']['short_plan'] == 0
    trained = cot.load_short_plan_adapter('omni_captioner', local_model, checkpoint=second['checkpoint'], mode=mode)
    assert isinstance(trained, cot.ShortPlanAdapter)
    trained.model.generation_text = cot.serialize_target(rows[0], mode)[0]
    assert trained.generate_caption(rows[0], max_new_tokens=1024)['caption_valid'] is True
    other = next(item for item in cot.MODES if item != mode)
    with pytest.raises(ValueError, match='config/fingerprint'):
        cot.load_short_plan_adapter('omni_captioner', local_model, checkpoint=second['checkpoint'], mode=other)
    if mode == 'direct-caption':
        with pytest.raises(ValueError, match='config/fingerprint'):
            cot.load_short_plan_adapter('omni_captioner', local_model, checkpoint=second['checkpoint'],
                                       mode=mode, require_shared_annotations=True)
        with pytest.raises(ValueError, match='resume config/data/source'):
            cot.main(['train'] + shared + ['--require-shared-annotations', '--max-updates', '4',
                                          '--resume', second['checkpoint']])
    with pytest.raises(ValueError, match='resume config/data/source'):
        cot.main(['train'] + shared + ['--mode', other, '--max-updates', '4', '--resume', second['checkpoint']])
    audio_input = tmp_path / 'audio_only.jsonl'
    write_rows(audio_input, [{key: validation[key] for key in
        ('id', 'split', 'audio_path', 'audio_sha256', 'audio_window_seconds')}])
    generated = cot.main(['generate', '--mode', mode, '--family', 'omni_captioner', '--model', str(local_model),
        '--checkpoint', second['checkpoint'], '--input', str(audio_input), '--out', str(tmp_path / 'generated.jsonl'),
        '--device', 'cpu', '--dtype', 'float32', '--max-new-tokens', '1024'])
    assert generated[0]['caption_valid'] is True and generated[0]['mode'] == mode
    from scripts_omnidance.exp.ms import music_planner_cot_audit as audit
    audited = audit.main(['audit', '--mode', mode, '--model', str(local_model),
        '--checkpoint', second['checkpoint'], '--input', str(audio_input), '--out', str(tmp_path / 'audit.json'),
        '--device', 'cpu', '--dtype', 'float32', '--max-new-tokens', '1024'])
    assert audited['summary']['valid_final_captions'] == 1
    assert audited['teacher_forced_nll_used'] is False
    with pytest.raises(ValueError, match='context'):
        cot.load_short_plan_adapter('omni_captioner', local_model, checkpoint=second['checkpoint'], mode=mode, max_length=4000)
    sidecar = Path(second['checkpoint']) / 'adapter/short_plan_adapter.json'
    original = sidecar.read_bytes()
    sidecar.write_text('{}')
    with pytest.raises(ValueError, match='export bytes'):
        cot.load_short_plan_adapter('omni_captioner', local_model, checkpoint=second['checkpoint'], mode=mode)
    sidecar.write_bytes(original)
    with monkeypatch.context() as changed_source:
        changed_source.setattr(cot, 'implementation_sources', lambda: {'music_planner_cot.py': 'f' * 64})
        with pytest.raises(ValueError, match='source version differs'):
            cot.load_short_plan_adapter('omni_captioner', local_model, checkpoint=second['checkpoint'], mode=mode)
    marker_path = Path(second['checkpoint']) / 'trainer_state.json'
    marker = json.loads(marker_path.read_text())
    marker['complete'] = False
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='completed portable'):
        cot.load_short_plan_adapter('omni_captioner', local_model, checkpoint=second['checkpoint'], mode=mode)
    wrong = dict(rows[0], split='test')
    write_rows(train, [wrong])
    with pytest.raises(ValueError, match='test/mixed'):
        cot.check_data(train, val)


def test_check_data_only_fails_missing_evidence_without_loading_model(tmp_path, monkeypatch):
    train = tmp_path / 'train.jsonl'
    val = tmp_path / 'val.jsonl'
    row = example(tmp_path)
    del row['music_evidence']
    write_rows(train, [row])
    write_rows(val, [example(tmp_path, 'v', 'val', frequency=330)])
    monkeypatch.setattr(cot, 'load_short_plan_adapter', lambda *a, **k: pytest.fail('No model loading'))
    with pytest.raises(ValueError, match='Missing reviewed'):
        cot.main(['train', '--family', 'moss', '--model', str(tmp_path), '--train', str(train),
                  '--val', str(val), '--out', str(tmp_path / 'unused'), '--check-data-only'])
    assert not (tmp_path / 'unused').exists()


def test_missing_full_checkpoint_marker_refused_before_native_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(cot, 'load_planner_adapter', lambda *a, **k: pytest.fail('Do not load invalid export'))
    with pytest.raises(ValueError, match='full experiment checkpoint'):
        cot.load_short_plan_adapter('moss', tmp_path, checkpoint=tmp_path)


@pytest.mark.parametrize('family', cot.FAMILIES)
def test_direct_caption_has_no_intermediate_tokens_or_loss(tmp_path, family):
    row = example(tmp_path)
    adapter = cot.ShortPlanAdapter(family, TinyPlanner(), MockProcessor(), mode='direct-caption',
                                   require_shared_annotations=True)
    text, _ = cot.serialize_target(row, 'direct-caption')
    assert text == row['target_text']
    assert all(tag not in text for tag in cot.SEGMENTS)
    prompt = adapter._prompt()
    assert (cot.DIRECT_INSTRUCTION in prompt) is (family != 'omni_captioner')
    assert cot.INSTRUCTION not in prompt
    batch = adapter.collate([row])
    assert set(batch['segment_ids'].flatten().tolist()) == {-1, 2}
    logits = adapter.model(**{key: value for key, value in batch.items()
                              if key not in ('labels', 'segment_ids', 'audio_input_mask')}).logits
    loss, totals, counts = cot.segment_nll(logits, batch['labels'], batch['segment_ids'], mode='direct-caption')
    assert counts[0] == counts[1] == 0
    torch.testing.assert_close(loss, totals[2] / counts[2])
    loss.backward()
    assert adapter.model.lora_A.grad.norm() > 0
    adapter.model.generation_text = text
    generated = adapter.generate_caption(row, max_new_tokens=1024)
    assert generated['caption_valid'] is True and generated['caption'] == text
    assert 'music_evidence' not in generated and 'short_plan' not in generated
    adapter.model.generation_text = cot.serialize_target(row, 'short-plan')[0]
    assert adapter.generate_caption(row, max_new_tokens=1024)['caption_valid'] is False
    del row['short_plan']
    with pytest.raises(ValueError, match='Missing reviewed'):
        adapter.collate([row])  # Same reviewed data is mandatory in both ablation modes.


def test_direct_mode_rejects_inactive_segment_weight_and_membership():
    with pytest.raises(ValueError, match='requires weights'):
        cot.effective_weights('direct-caption', (.2, .3, 1.))
    labels = torch.tensor([[-100, 4, 5, 2]])
    sections = torch.tensor([[-1, 0, 1, 2]])
    with pytest.raises(ValueError, match='membership'):
        cot.segment_nll(torch.randn(1, 4, 8), labels, sections, mode='direct-caption')


@pytest.mark.parametrize('mode', cot.MODES)
@pytest.mark.parametrize('family', ['moss', 'omni_captioner'])
def test_upstream_native_generation_bridges_execute_and_receipts_are_eval_only(tmp_path, mode, family, monkeypatch):
    class NativeCacheFixture(TinyPlanner):
        def __init__(self):
            super().__init__()
            self.config.model_type = 'moss_music'
            self._planner_native_forward = self.forward
            self.generation_config.eos_token_id = None
            self.cached_calls = 0

        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, cache_position=None, **kwargs):
            if past_key_values is not None:
                assert int(cache_position[0]) == past_key_values.get_seq_length()
                self.cached_calls += 1
            return dict(input_ids=input_ids, cache_position=cache_position)

        def generate(self, **kwargs):
            assert kwargs['eos_token_id'] == [2]  # Supplied by the actual upstream EOS wrapper.
            if family == 'moss':
                ids = kwargs['input_ids']
                self.prepare_inputs_for_generation(ids, audio_data=torch.ones(1, 2),
                    audio_input_mask=ids.eq(3))
                cache = SimpleNamespace(get_seq_length=lambda: ids.shape[1])
                self.prepare_inputs_for_generation(torch.cat((ids, torch.ones(1, 1, dtype=torch.long)), 1),
                    past_key_values=cache, attention_mask=torch.ones(1, ids.shape[1] + 1))
            return super().generate(**kwargs)

    row = example(tmp_path)
    model = NativeCacheFixture()
    adapter = cot.ShortPlanAdapter(family, model, MockProcessor(), mode=mode)
    model.generation_text = cot.serialize_target(row, mode)[0]
    before = copy.deepcopy(model.state_dict())
    sources = cot.implementation_sources()
    position_calls = []
    def restore_position(loaded_adapter, model_path):
        position_calls.append((loaded_adapter, model_path))
        return dict(applied_position_buffer_patch_ref=cot.position_patch.position_buffer_patch_ref())
    monkeypatch.setattr(cot.position_patch, 'restore_moss_position_buffer', restore_position)
    receipt = cot.apply_generation_compatibility(adapter, 'unchanged-training-fingerprint', tmp_path)
    result = adapter.generate_caption(row, max_new_tokens=1024)
    assert result['caption_valid'] is True
    assert result['applied_generation_eos_patch_ref'] == cot.eos_patch.generation_eos_patch_ref()
    assert receipt['training_fingerprint'] == 'unchanged-training-fingerprint'
    assert receipt['training_fingerprint_mutated'] is False
    assert 'music_planner_generation_eos_patch.py' not in sources
    assert 'music_planner_moss_generation_patch.py' not in sources
    assert cot.implementation_sources() == sources
    assert model.generation_config.eos_token_id is None
    for name, tensor in before.items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
    if family == 'moss':
        assert position_calls == [(adapter, tmp_path)]
        assert receipt['applied_sources']['moss_position'] == cot.position_patch.position_buffer_patch_ref()
        assert model.cached_calls == 1
        assert result['applied_generation_patch_ref'] == cot.moss_patch.generation_patch_ref()


@pytest.mark.parametrize('was_training', [True, False])
def test_validation_restores_model_mode_when_annotation_processing_fails(was_training):
    model = TinyPlanner()
    model.train(was_training)

    def broken_collate(rows):
        assert model.training is False
        raise ValueError('fixture annotation failure')

    adapter = SimpleNamespace(mode='short-plan', collate=broken_collate)
    cache = SimpleNamespace(verify=lambda row: None)
    with pytest.raises(ValueError, match='fixture annotation failure'):
        cot.evaluate(model, adapter, [{}], cache, torch.device('cpu'), torch.float32, (.2, .3, 1.))
    assert model.training is was_training


@pytest.mark.parametrize('mode', cot.MODES)
@pytest.mark.parametrize('new_ids,budget', [([10, 2, 11, 2], 8), ([10, 11, 12, 2], 3), ([10, 11], 8)])
def test_generated_raw_token_suffix_and_budget_invalid_without_cleaning(tmp_path, mode, new_ids, budget):
    row = example(tmp_path)
    adapter = cot.ShortPlanAdapter('omni_captioner', TinyPlanner(), MockProcessor(), mode=mode)
    raw = cot.serialize_target(row, mode)[0]
    adapter.generate = lambda *a, **k: dict(raw_text=raw, new_token_ids=new_ids,
        eos_token_ids=[2], stop_reason='eos')
    result = adapter.generate_caption(row, max_new_tokens=budget)
    assert result['caption_valid'] is False
    assert result['new_token_ids'] == new_ids and result['raw_text'] == raw


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('mode', cot.MODES)
def test_checkpointed_assistant_chunks_match_dense_reference_loss_and_grad(dtype, mode):
    torch.manual_seed(2)
    labels = torch.tensor([[-100, -100, 4, 5, 6, 2], [-100, 3, 4, 7, 6, 2]])
    sections = torch.tensor([[-1, -1, 0, 1, 2, 2], [-1, 0, 0, 1, 2, 2]])
    if mode == 'direct-caption':
        sections = torch.where(labels.ne(-100), 2, -1)
    first = torch.randn(2, 6, 11, dtype=dtype, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    weights = cot.effective_weights(mode)
    dense_ce = torch.nn.functional.cross_entropy(first[:, :-1].float().reshape(-1, 11),
        labels[:, 1:].reshape(-1), ignore_index=-100, reduction='none').reshape(2, 5)
    active = (0, 1, 2) if mode == 'short-plan' else (2,)
    expected = sum(weights[index] * dense_ce[sections[:, 1:].eq(index)].mean()
                   for index in active) / sum(weights)
    actual, sums, counts = cot.segment_nll(second, labels, sections, mode=mode, chunk_size=2)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(second.grad, first.grad, rtol=.01 if dtype == torch.bfloat16 else 2e-6,
                               atol=2e-4 if dtype == torch.bfloat16 else 2e-7)
    assert second.grad[:, -1].eq(0).all()


def test_checkpointed_ce_does_not_save_all_fp32_vocabulary_chunks():
    labels = torch.tensor([[-100, -100, 4, 5, 6, 2]])
    sections = torch.tensor([[-1, -1, 0, 1, 2, 2]])
    def saved_shapes(chunk):
        saved = []
        def pack(value):
            saved.append((value.dtype, tuple(value.shape)))
            return value
        logits = torch.randn(1, 6, 101, dtype=torch.bfloat16, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            loss, _, _ = cot.segment_nll(logits, labels, sections, chunk_size=chunk)
        loss.backward()
        return saved
    plain, chunked = saved_shapes(0), saved_shapes(2)
    def fp32_vocabulary(entries):
        return [shape for dtype, shape in entries if dtype == torch.float32 and len(shape) == 2 and shape[-1] == 101]
    assert fp32_vocabulary(plain)
    assert not fp32_vocabulary(chunked)


def test_shared_annotation_protocol_remains_explicit(tmp_path):
    row = example(tmp_path)
    del row['music_evidence']
    del row['short_plan']
    assert cot.serialize_target(row, 'direct-caption')[0] == row['target_text']
    with pytest.raises(ValueError, match='Missing reviewed'):
        cot.serialize_target(row, 'direct-caption', require_shared_annotations=True)
    with pytest.raises(ValueError, match='Missing reviewed'):
        cot.serialize_target(row, 'short-plan')
    assert cot.mode_schema('direct-caption') != cot.mode_schema('direct-caption', True)


def test_teacher_free_audit_comparison_requires_same_audio_and_budget(tmp_path):
    from scripts_omnidance.exp.ms import music_planner_cot_audit as audit
    source = example(tmp_path, split='val')
    audio = {key: source[key] for key in ('id', 'split', 'audio_path', 'audio_sha256', 'audio_window_seconds')}
    data = {split: dict(sha256=split + '-fixed-manifest') for split in ('train', 'val')}
    reports = []
    for mode in cot.MODES:
        adapter = cot.ShortPlanAdapter('omni_captioner', TinyPlanner(), MockProcessor(),
                                      mode=mode, require_shared_annotations=True)
        adapter.model.generation_text = cot.serialize_target(source, mode, True)[0]
        reports.append(audit.generate_audit(adapter, [audio], max_new_tokens=1024,
            checkpoint_identity=dict(fingerprint='fixture-' + mode), training_data=data,
            input_ref=dict(sha256='fixture-audio-only'), base_identity=dict(model='same-fixture-base'),
            comparison_contract=dict(settings={key: 1 for key in audit.COMPARISON_KEYS},
                initialization='fresh-local-base-and-LoRA-no-warm-start'), source_exposures=100))
    result = audit.compare_audits(*reports, require_shared_annotations=True)
    assert result['shared_reviewed_training_manifests'] is True
    assert result['quality_winner'] is None and result['teacher_forced_nll_used'] is False
    assert result['controlled_ablation'] is True
    left_path, right_path = tmp_path / 'left.json', tmp_path / 'right.json'
    left_path.write_text(json.dumps(reports[0]))
    right_path.write_text(json.dumps(reports[1]))
    assert audit.main(['compare', '--left', str(left_path), '--right', str(right_path),
        '--out', str(tmp_path / 'comparison.json'), '--require-shared-annotations']) == result
    edited = copy.deepcopy(reports[1])
    edited['max_new_tokens'] = 2048
    edited['audit_sha256'] = cot.base.object_sha({k: v for k, v in edited.items() if k != 'audit_sha256'})
    with pytest.raises(ValueError, match='budgets'):
        audit.compare_audits(reports[0], edited)
    unmatched = copy.deepcopy(reports[1])
    unmatched['source_exposures'] = 200
    unmatched['audit_sha256'] = cot.base.object_sha({k: v for k, v in unmatched.items() if k != 'audit_sha256'})
    assert audit.compare_audits(reports[0], unmatched)['controlled_ablation'] is False
    with pytest.raises(ValueError, match='actual source exposures'):
        audit.compare_audits(reports[0], unmatched, require_shared_annotations=True)
    tampered = copy.deepcopy(reports[0])
    tampered['summary']['valid_final_captions'] = 50
    with pytest.raises(ValueError, match='changed teacher-free'):
        audit.compare_audits(tampered, reports[1])
