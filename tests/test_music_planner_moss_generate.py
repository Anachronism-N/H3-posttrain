"""CPU native-shaped fixtures, not real MOSS/PEFT GPU generation acceptance."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from scripts_omnidance.exp.ms import music_planner_moss_generate as generate
from scripts_omnidance.exp.ms import music_planner_jev as jev
from scripts_omnidance.exp.ms import music_planner_train as trainer


class TinyTokenizer:
    eos_token_id = 1

    def __init__(self):
        self.to_id, self.to_text = {}, {}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == '<|im_end|>':
            return [1]
        result = []
        for char in text:
            if char not in self.to_id:
                token = len(self.to_id) + 10
                self.to_id[char] = token
                self.to_text[token] = char
            result.append(self.to_id[char])
        # Deliberately demonstrate that concat tokenization is not compositional.
        if text.startswith('舞种：') and '[' in text:
            result = [500] + result
        return result

    def decode(self, tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return ''.join('<|im_end|>' if token == 1 else self.to_text.get(token, '<unknown>') for token in tokens)


class TinyCache:
    def __init__(self, length=0):
        self.length = length

    def get_seq_length(self):
        return self.length


class TinyMoss(torch.nn.Module):
    """Native-shaped audio mask/cache forwarding plus actual JEV and torch forwards."""
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = torch.nn.Embedding(512, 4)
        self.language_model = torch.nn.Module()
        self.language_model.norm = torch.nn.LayerNorm(4)
        self.language_model.config = SimpleNamespace(hidden_size=4)
        self.lm_head = torch.nn.Linear(4, 512)
        self.audio_encoder = torch.nn.Module()
        self.audio_encoder.embed_positions = torch.nn.Module()
        self.audio_encoder.embed_positions.register_buffer('inv_timescales', torch.linspace(.001, 1., 640), persistent=False)
        self.config = SimpleNamespace(model_type='moss_music', language_config=SimpleNamespace(hidden_size=4))
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self._planner_native_forward = self.forward
        self.seen_generate, self.forward_inputs = [], []
        self.continuation = None
        self.over_budget = False
        self.bad_prefix = False
        self.nonfinite = False

    def forward(self, input_ids=None, attention_mask=None, audio_input_mask=None, **kwargs):
        self.forward_inputs.append(dict(ids=input_ids.detach().clone(), audio_mask=audio_input_mask))
        hidden = self.language_model.norm(self.embedding(input_ids))
        logits = self.lm_head(hidden)
        if self.nonfinite:
            logits = logits * float('nan')
        return SimpleNamespace(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, cache_position=None, **kwargs):
        cached = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cached and cache_position is not None:
            return dict(input_ids=input_ids[:, -1:], attention_mask=attention_mask,
                        audio_data=None, audio_input_mask=None, past_key_values=past_key_values)
        return dict(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.seen_generate.append(dict(input_ids=input_ids.clone(), attention_mask=attention_mask.clone(), **kwargs))
        text = self.tokenizer.decode(input_ids[0].tolist())
        actions = '[0-1s] 抬起右臂。[1-2s] 向右侧踏。' if 'Style-B' in text else '[0-1s] 抬起左臂。[1-2s] 向左侧踏。'
        tokens = self.tokenizer.encode(actions) + [1] if self.continuation is None else list(self.continuation)
        if not self.over_budget:
            tokens = tokens[:kwargs['max_new_tokens']]
        ids = input_ids.clone()
        mask = attention_mask.clone()
        cache = TinyCache()
        for token in tokens:
            prepared = self.prepare_inputs_for_generation(ids, past_key_values=cache,
                attention_mask=mask, audio_data=kwargs.get('audio_data'),
                audio_input_mask=kwargs.get('audio_input_mask'), use_cache=True)
            self(**prepared)
            cache.length = ids.shape[1]
            ids = torch.cat((ids, ids.new_tensor([[token]])), 1)
            mask = torch.cat((mask, mask.new_ones((1, 1))), 1)
        if self.bad_prefix:
            ids[0, 0] = 9
        return ids


class TinyAdapter:
    family = 'moss'

    def __init__(self):
        self.tokenizer = TinyTokenizer()
        self.processor = SimpleNamespace(audio_token_id=3)
        self.model = TinyMoss(self.tokenizer)
        self.max_length = 1024
        self.received_sources = []
        self.contract = dict(schema=generate.CONTRACT_SCHEMA, caption_version='tiny-v2', task_instruction='fixture',
            audio_min_seconds=1., audio_max_seconds=30., required_markers=[], require_target_windows=False,
            dance_caption_format=dict(schema='music-planner-dance-caption-v1', genre_header_template='舞种：{genre}\n',
                                      time_tolerance_seconds=.001, require_full_coverage=True))
        self.dance_format = self.contract['dance_caption_format']
        self.genre_classes = ['Style-A', 'Style-B']
        self.header_token_ids = {genre: self.tokenizer.encode('舞种：' + genre + '\n') for genre in self.genre_classes}
        self.token_protocol = generate.TOKEN_PROTOCOL
        self.dance_caption_protocol = dict(schema=generate.PROTOCOL_SCHEMA, family='moss',
            classes=self.genre_classes, classes_sha256=jev.classes_identity(self.genre_classes),
            dance_format=self.dance_format, header_token_ids=self.header_token_ids, token_protocol=self.token_protocol)

    def prefix(self, source):
        assert set(source) == set(generate.SOURCE_KEYS)
        self.received_sources.append(dict(source))
        return dict(input_ids=torch.tensor([[2, 3, 4]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                    audio_input_mask=torch.tensor([[False, True, False]]), audio_data=torch.ones(1, 8),
                    token_type_ids=torch.zeros(1, 3, dtype=torch.long))

    def _move(self, batch):
        return batch

    def eos_token_ids(self):
        return [1]

    def generate(self, *args, **kwargs):
        raise AssertionError('The old adapter.generate must not be used for conditioned prefill')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def bind_checkpoint(adapter, root, exercised=True):
    checkpoint = root / 'checkpoint-0000002'
    dump(checkpoint / 'adapter/dance_caption_protocol.json', adapter.dance_caption_protocol)
    dump(checkpoint / 'adapter/caption_contract.json', adapter.contract)
    (checkpoint / 'trainable.pt').write_bytes(b'tiny portable weights fixture')
    for name in ('scheduler.pt', 'rng-rank0.pt', 'optimizer-rank0.pt', 'writer-rank0.json'):
        (checkpoint / name).write_bytes(b'complete checkpoint member fixture')
    config = dict(family='moss', max_length=adapter.max_length, model={'fixture': True},
                  data={'contract': adapter.contract, 'test_used': False}, world_size=1)
    config['fingerprint'] = trainer.object_sha(config)
    dump(root / 'run_config.json', config)
    marker = dict(schema='music-planner-train-v1', complete=True, fingerprint=config['fingerprint'], world_size=1, zero3=False,
                  cursor={'update': 2, 'jev_head_exercised': exercised},
                  trainable_sha256=generate._sha(checkpoint / 'trainable.pt'),
                  files=[generate._ref(path) for path in sorted(checkpoint.rglob('*')) if path.is_file() and path.name != 'trainer_state.json'])
    dump(checkpoint / 'trainer_state.json', marker)
    adapter.training_protocol_identity = dict(checkpoint_path=str(checkpoint), marker_sha256=generate._sha(checkpoint / 'trainer_state.json'),
        cursor=copy.deepcopy(marker['cursor']), fingerprint=marker['fingerprint'], family='moss',
        trainable_sha256=marker['trainable_sha256'])
    return checkpoint


def position_receipt(adapter, path=None):
    tensor = adapter.model.audio_encoder.embed_positions.inv_timescales
    return dict(applied_position_buffer_patch_ref=generate.position_patch.position_buffer_patch_ref(),
                parameters_changed=False, persistent=False, training_identity_preserved=True,
                canonical_tensor_ref=generate.position_patch._tensor_ref(tensor))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    adapter = TinyAdapter()
    jev.install_jev(adapter, adapter.genre_classes, head_dim=4)
    # Make an unmistakable real head argmax without claiming a learned data result.
    with torch.no_grad():
        adapter.model.jev_head.network[-1].weight.zero_()
        adapter.model.jev_head.network[-1].bias.copy_(torch.tensor([-2., 2.]))
    checkpoint = bind_checkpoint(adapter, tmp_path / 'run')
    monkeypatch.setattr(generate.position_patch, 'restore_moss_position_buffer', position_receipt)
    generate.install_runtime_patches(adapter, tmp_path / 'official-fixture')
    source = dict(id='audio-1', source_group_id='song-1', audio_path=str(tmp_path / 'input.wav'),
                  audio_sha256='a' * 64, audio_duration_seconds=2., audio_window_seconds=[0., 2.])
    return adapter, source, checkpoint


def test_default_real_head_selects_header_before_native_generation(fixture):
    adapter, source, _ = fixture
    adapter.model.train()
    output = generate.generate_conditioned(adapter, source, max_new_tokens=128)
    assert output['caption_valid'], output['validation_errors']
    assert output['selected_genre'] == 'Style-B'
    assert '右臂' in output['caption'] and '向右' in output['caption']
    assert output['actual_style_forward_count'] == 1
    assert output['actual_generation_forward_count'] > 0
    assert output['expected_prefills'] == 2
    assert output['kv_cache_shared_between_selection_and_generation'] is False
    assert adapter.model.training is True
    actual = adapter.model.seen_generate[-1]
    expected_header = adapter.header_token_ids['Style-B']
    assert actual['input_ids'].tolist() == [[2, 3, 4] + expected_header]
    assert actual['audio_input_mask'].tolist() == [[False, True, False] + [False] * len(expected_header)]
    assert actual['attention_mask'].shape == actual['input_ids'].shape
    assert actual['token_type_ids'].shape == actual['input_ids'].shape
    assert actual['use_cache'] is True and actual['eos_token_id'] == [1]
    assert output['raw_new_token_ids'][-1] == 1
    assert output['new_tokens'] == len(output['raw_new_token_ids'])
    assert output['forced_header_tokens'] == len(expected_header)
    assert output['raw_text'].endswith('<|im_end|>')
    assert not output['action_text'].endswith('<|im_end|>')


def test_explicit_intervention_changes_actual_prefill_and_actions(fixture):
    adapter, source, checkpoint = fixture
    bind_checkpoint(adapter, checkpoint.parent, exercised=False)
    first = generate.generate_conditioned(adapter, source, 128, selected_genre='Style-A')
    second = generate.generate_conditioned(adapter, source, 128, selected_genre='Style-B')
    assert first['caption_valid'] and second['caption_valid']
    assert first['caption'] != second['caption']
    assert '左臂' in first['action_text'] and '右臂' in second['action_text']
    assert first['actual_style_forward_count'] == 0 and first['expected_prefills'] == 1
    assert first['probabilities'] is None
    with pytest.raises(ValueError, match='head was exercised'):
        generate.generate_conditioned(adapter, source, 128)


def test_gold_annotations_are_never_read_or_forwarded(fixture):
    adapter, source, _ = fixture
    class Poison(dict):
        def __getitem__(self, key):
            if key not in generate.SOURCE_KEYS:
                raise AssertionError('Gold label read: ' + key)
            return super().__getitem__(key)
        def get(self, key, *args):
            raise AssertionError('Do not get any row labels or defaults')
        def items(self):
            raise AssertionError('Do not enumerate full annotated row')
    row = Poison(source, target_text='GOLD MUST NOT APPEAR', genre='Style-A', target_windows='GOLD')
    output = generate.generate_conditioned(adapter, row, 128)
    assert output['caption_valid'] and output['target_consumed'] is False
    assert all(set(value) == set(generate.SOURCE_KEYS) for value in adapter.received_sources)
    assert all('GOLD' not in str(value) for value in adapter.model.seen_generate)


@pytest.mark.parametrize('fault,match', [
    ('legacy_contract', 'v2'), ('header_ids', 'header IDs'), ('token_protocol', 'token protocol'),
    ('checkpoint_identity', 'Restore'), ('marker_change', 'marker'), ('protocol_export', 'export'),
    ('wrong_family', 'MOSS'), ('cache_patch', 'cached-decode'), ('position_buffer', 'position buffer'),
])
def test_metadata_and_patch_preconditions_reject_before_forward(fixture, fault, match):
    adapter, source, checkpoint = fixture
    if fault == 'legacy_contract':
        adapter.contract = dict(adapter.contract, schema='music-planner-caption-contract-v1')
    elif fault == 'header_ids':
        adapter.header_token_ids['Style-A'] = [44]
    elif fault == 'token_protocol':
        adapter.token_protocol = 'unknown'
    elif fault == 'checkpoint_identity':
        del adapter.training_protocol_identity
    elif fault == 'marker_change':
        path = checkpoint / 'trainer_state.json'
        path.write_bytes(path.read_bytes() + b'\n')
    elif fault == 'protocol_export':
        dump(checkpoint / 'adapter/dance_caption_protocol.json', {'unknown': True})
    elif fault == 'wrong_family':
        adapter.family = 'omni_captioner'
    elif fault == 'cache_patch':
        adapter.applied_generation_patch_ref = None
    else:
        adapter.model.audio_encoder.embed_positions.inv_timescales[0] = 0.
    with pytest.raises(ValueError, match=match):
        generate.generate_conditioned(adapter, source, 128, selected_genre='Style-A')
    assert not adapter.model.forward_inputs and not adapter.model.seen_generate


def test_training_header_segmentation_is_not_guessed_from_concat_tokenization(fixture):
    adapter, source, _ = fixture
    header = '舞种：Style-A\n'
    actions = '[0-2s] 抬起双臂。'
    concatenated = adapter.tokenizer.encode(header + actions)
    segmented = adapter.tokenizer.encode(header) + adapter.tokenizer.encode(actions)
    assert concatenated != segmented
    output = generate.generate_conditioned(adapter, source, 128, selected_genre='Style-A')
    assert output['caption_valid']
    assert adapter.model.seen_generate[-1]['input_ids'][0, 3:].tolist() == adapter.tokenizer.encode(header)


@pytest.mark.parametrize('kind', ['early_eos', 'double_eos', 'no_eos', 'over_budget', 'empty_actions',
                                  'bad_timestamp', 'missing_coverage', 'bad_prefix'])
def test_invalid_outputs_preserve_raw_and_never_produce_caption(fixture, kind):
    adapter, source, _ = fixture
    correct = adapter.tokenizer.encode('[0-2s] 抬起双臂。')
    budget = 128
    if kind == 'early_eos':
        tokens = correct + [1] + adapter.tokenizer.encode('ignored tail')
    elif kind == 'double_eos':
        tokens = correct + [1, 1]
    elif kind == 'no_eos':
        tokens = correct
    elif kind == 'over_budget':
        tokens = correct + [1]
        budget = len(tokens) - 1
        adapter.model.over_budget = True
    elif kind == 'empty_actions':
        tokens = [1]
    elif kind == 'bad_timestamp':
        tokens = adapter.tokenizer.encode('[0-3s] 抬起双臂。') + [1]
    elif kind == 'missing_coverage':
        tokens = adapter.tokenizer.encode('[0-1s] 抬起双臂。') + [1]
    else:
        tokens = correct + [1]
        adapter.model.bad_prefix = True
    adapter.model.continuation = tokens
    output = generate.generate_conditioned(adapter, source, budget, selected_genre='Style-A')
    assert output['caption'] is None and not output['caption_valid']
    assert output['validation_errors']
    assert output['raw_new_token_ids'] == tokens
    assert output['raw_text'] == adapter.tokenizer.decode(tokens)
    assert output['raw_sequence_token_ids'][0][-len(tokens):] == tokens


def test_context_budget_no_truncation_and_model_mode_restored_on_error(fixture):
    adapter, source, checkpoint = fixture
    adapter.max_length = 20
    bind_checkpoint(adapter, checkpoint.parent)
    adapter.model.train()
    output = generate.generate_conditioned(adapter, source, 128, selected_genre='Style-A')
    assert not output['caption_valid'] and 'context' in output['validation_errors'][0]
    assert not adapter.model.seen_generate
    assert adapter.model.training


def test_nonfinite_native_logits_fail_before_success(fixture):
    adapter, source, _ = fixture
    adapter.model.nonfinite = True
    adapter.model.train()
    output = generate.generate_conditioned(adapter, source, 128, selected_genre='Style-A')
    assert output['caption'] is None and not output['caption_valid']
    assert 'non-finite' in output['validation_errors'][0]
    assert adapter.model.training


def test_checkpoint_verification_rejects_incomplete_or_changed_artifacts(fixture):
    adapter, _, checkpoint = fixture
    verified = generate.verify_checkpoint(checkpoint)
    assert verified['all_recorded_checkpoint_bytes_verified'] is True
    marker = generate._read(checkpoint / 'trainer_state.json')
    marker['complete'] = False
    dump(checkpoint / 'trainer_state.json', marker)
    with pytest.raises(ValueError, match='complete'):
        generate.verify_checkpoint(checkpoint)
    checkpoint = bind_checkpoint(adapter, checkpoint.parent)
    (checkpoint / 'trainable.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        generate.verify_checkpoint(checkpoint)


def test_chinese_training_fingerprint_uses_exact_trainer_serialization(fixture):
    _, _, checkpoint = fixture
    config = generate._read(checkpoint.parent / 'run_config.json')
    canonical = {key: value for key, value in config.items() if key != 'fingerprint'}
    assert '舞种' in json.dumps(canonical, ensure_ascii=False)
    utf8_alternative = generate.hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()
    assert utf8_alternative != config['fingerprint']
    assert generate.verify_checkpoint(checkpoint)['marker']['fingerprint'] == trainer.object_sha(canonical)


def test_renamed_checkpoint_consumes_its_own_bytes_not_the_recorded_writer_path(fixture):
    adapter, source, checkpoint = fixture
    relocated = checkpoint.with_name('renamed-checkpoint-copy')
    assert relocated.resolve().parent == checkpoint.resolve().parent
    shutil.copytree(checkpoint, relocated)
    (checkpoint / 'trainable.pt').write_bytes(b'old writer bytes no longer valid')
    assert generate.verify_checkpoint(relocated)['all_recorded_checkpoint_bytes_verified']
    adapter.training_protocol_identity['checkpoint_path'] = str(relocated)
    output = generate.generate_conditioned(adapter, source, 128, selected_genre='Style-A')
    assert output['caption_valid'], output['validation_errors']


def test_load_entry_restores_protocol_and_patches_before_any_forward(fixture, monkeypatch, tmp_path):
    adapter, source, checkpoint = fixture
    from scripts_omnidance.exp.ms import music_planner_adapters as adapters
    from scripts_omnidance.exp.ms import music_planner_schema as schema
    from scripts_omnidance.exp.ms.music_planner_cluster import model_inventory
    model = tmp_path / 'official'
    dump(model / 'config.json', {'fixture': True})
    (model / 'model.safetensors').write_bytes(b'official fixture model bytes')
    config = generate._read(checkpoint.parent / 'run_config.json')
    config['model'] = dict(content_inventory_sha256=model_inventory(model)['inventory_sha256'])
    config['fingerprint'] = trainer.object_sha({k: v for k, v in config.items() if k != 'fingerprint'})
    dump(checkpoint.parent / 'run_config.json', config)
    marker = generate._read(checkpoint / 'trainer_state.json')
    marker['fingerprint'] = config['fingerprint']
    dump(checkpoint / 'trainer_state.json', marker)
    adapter.training_protocol_identity.update(fingerprint=marker['fingerprint'], marker_sha256=generate._sha(checkpoint / 'trainer_state.json'))
    called = []
    def loader(family, model_path, checkpoint=None, **kwargs):
        called.append(('native-load', family, str(checkpoint)))
        assert not adapter.model.forward_inputs
        return adapter
    def restore(native, checkpoint):
        called.append(('restore-protocol', str(checkpoint)))
        return native
    monkeypatch.setattr(adapters, 'load_planner_adapter', loader)
    monkeypatch.setattr(schema, 'restore_adapter_protocol', restore)
    loaded = generate.load_conditioned_adapter(model, checkpoint, dtype=torch.float32)
    assert [event[0] for event in called] == ['native-load', 'restore-protocol']
    assert loaded.inference_base_identity['content_verified'] is True
    result = generate.generate_conditioned(loaded, source, 128)
    assert result['caption_valid']
    (model / 'model.safetensors').write_bytes(b'other model bytes')
    with pytest.raises(ValueError, match='base model contents'):
        generate.load_conditioned_adapter(model, checkpoint, dtype=torch.float32)


def test_legacy_base_identity_is_explicitly_not_content_verified(fixture, monkeypatch, tmp_path):
    adapter, _, checkpoint = fixture
    from scripts_omnidance.exp.ms import music_planner_adapters as adapters
    from scripts_omnidance.exp.ms import music_planner_schema as schema
    model = tmp_path / 'legacy-base'
    dump(model / 'config.json', {'fixture': True})
    (model / 'model.safetensors').write_bytes(b'legacy byte identity is not proven')
    config = generate._read(checkpoint.parent / 'run_config.json')
    config['model'] = dict(config_sha256=generate._sha(model / 'config.json'),
                          weights=[dict(name='model.safetensors', size=(model / 'model.safetensors').stat().st_size)])
    config['fingerprint'] = trainer.object_sha({k: v for k, v in config.items() if k != 'fingerprint'})
    dump(checkpoint.parent / 'run_config.json', config)
    marker = generate._read(checkpoint / 'trainer_state.json')
    marker['fingerprint'] = config['fingerprint']
    dump(checkpoint / 'trainer_state.json', marker)
    adapter.training_protocol_identity.update(fingerprint=marker['fingerprint'], marker_sha256=generate._sha(checkpoint / 'trainer_state.json'))
    monkeypatch.setattr(adapters, 'load_planner_adapter', lambda *args, **kwargs: adapter)
    monkeypatch.setattr(schema, 'restore_adapter_protocol', lambda native, checkpoint: native)
    loaded = generate.load_conditioned_adapter(model, checkpoint, dtype=torch.float32)
    assert loaded.inference_base_identity['content_verified'] is False
    assert 'not proven' in loaded.inference_base_identity['limitation']


def test_cli_writes_raw_failure_without_overwriting_previous_output(fixture, monkeypatch, tmp_path, capsys):
    adapter, source, checkpoint = fixture
    adapter.conditioned_checkpoint_verification = {'fixture': True}
    monkeypatch.setattr(generate, 'load_conditioned_adapter', lambda *args, **kwargs: adapter)
    adapter.model.continuation = adapter.tokenizer.encode('[0-2s] 动作。') + [1, 1]
    row_path, output = tmp_path / 'source.json', tmp_path / 'result.json'
    dump(row_path, dict(source, target_text='NOT AN INFERENCE INPUT'))
    argv = ['generate', '--model', str(tmp_path), '--checkpoint', str(checkpoint), '--row', str(row_path),
            '--output', str(output), '--max-new-tokens', '128', '--selected-genre', 'Style-A']
    assert generate.main(argv) == 2
    saved = generate._read(output)
    assert saved['caption'] is None and saved['raw_new_token_ids'][-2:] == [1, 1]
    assert saved['raw_text'].endswith('<|im_end|><|im_end|>')
    assert json.loads(capsys.readouterr().out)['actual_training_started'] is False
    previous = output.read_bytes()
    with pytest.raises(ValueError, match='Output exists'):
        generate.main(argv)
    assert output.read_bytes() == previous


# This is a format/conditioning draft, not a claim that the user's actual dance
# actions or music alignment have been independently approved.
ENGLISH_ACTION_DRAFT = (
    '[0.000-1.960s] The dancer shifts weight left, opens both arms, and turns the torso.\n'
    '[1.960-5.062s] The dancer steps right, bends the knees, and sweeps both arms upward.'
)


@pytest.fixture
def formal_profile(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    adapter = TinyAdapter()
    adapter.contract = generate._read(root / 'examples/planner_training/caption_contract_moss_v2.example.json')
    adapter.dance_format = adapter.contract['dance_caption_format']
    adapter.genre_supervision = adapter.contract['genre_supervision']
    adapter.genre_classes = generate._read(root / 'examples/planner_training/dance_styles_23.json')
    adapter.header_token_ids = {genre: adapter.tokenizer.encode(generate.caption_format.format_header(
        genre, adapter.genre_classes, adapter.dance_format)) for genre in adapter.genre_classes}
    adapter.dance_caption_protocol = dict(schema=generate.PROTOCOL_SCHEMA, family='moss',
        classes=adapter.genre_classes, classes_sha256=jev.classes_identity(adapter.genre_classes),
        dance_format=adapter.dance_format, header_token_ids=adapter.header_token_ids,
        token_protocol=adapter.token_protocol, genre_supervision=adapter.genre_supervision)
    semantic_classes = [name for name in adapter.genre_classes if name != adapter.genre_supervision['unknown_label']]
    assert len(adapter.genre_classes) == 23 and len(semantic_classes) == 22
    jev.install_jev(adapter, semantic_classes, head_dim=4)
    with torch.no_grad():
        adapter.model.jev_head.network[-1].weight.zero_()
        adapter.model.jev_head.network[-1].bias.zero_()
        adapter.model.jev_head.network[-1].bias[semantic_classes.index('contemporary')] = 4.
    checkpoint = bind_checkpoint(adapter, tmp_path / 'formal-profile-run')
    monkeypatch.setattr(generate.position_patch, 'restore_moss_position_buffer', position_receipt)
    generate.install_runtime_patches(adapter, tmp_path / 'official-fixture')
    adapter.model.continuation = adapter.tokenizer.encode(ENGLISH_ACTION_DRAFT) + [1]
    source = dict(id='format-draft', source_group_id='test-only-format-draft', audio_path=str(tmp_path / 'source.wav'),
                  audio_sha256='b' * 64, audio_duration_seconds=5.0625, audio_window_seconds=[0., 5.0625])
    return adapter, source, checkpoint, semantic_classes


def test_formal_english_contemporary_profile_has_22_probabilities_and_23_headers(formal_profile):
    adapter, source, _, semantic_classes = formal_profile
    result = generate.generate_conditioned(adapter, source, max_new_tokens=256)
    assert result['caption_valid'], result['validation_errors']
    assert result['caption'] == 'Dance style: contemporary\n' + ENGLISH_ACTION_DRAFT
    assert result['selected_genre'] == 'contemporary'
    assert result['selected_genre_is_unknown'] is False
    assert result['probability_classes'] == semantic_classes
    assert len(result['probabilities']) == 22
    assert 'unknown' not in result['probability_classes']
    assert result['min_style_confidence'] is None and result['calibrated'] is False
    assert result['parsed']['windows'][-1]['end'] == 5.062  # No snapping to 5.0625.
    assert adapter.model.seen_generate[-1]['input_ids'][0, 3:].tolist() == adapter.header_token_ids['contemporary']


def test_explicit_unknown_is_an_abstention_header_not_freestyle(formal_profile):
    adapter, source, checkpoint, _ = formal_profile
    bind_checkpoint(adapter, checkpoint.parent, exercised=False)
    result = generate.generate_conditioned(adapter, source, 256, selected_genre='unknown')
    assert result['caption_valid'], result['validation_errors']
    assert result['selected_genre_is_unknown'] is True
    assert result['style_selection_reason'] == 'explicit-unknown-abstention'
    assert result['caption'].startswith('Dance style: unknown\n')
    assert result['probabilities'] is None and result['probability_classes'] is None
    assert result['actual_style_forward_count'] == 0
    assert adapter.model.seen_generate[-1]['input_ids'][0, 3:].tolist() == adapter.header_token_ids['unknown']


def test_explicit_uncalibrated_threshold_routes_22way_confidence_to_unknown(formal_profile):
    adapter, source, _, semantic_classes = formal_profile
    result = generate.generate_conditioned(adapter, source, 256, min_style_confidence=.9)
    assert result['caption_valid'], result['validation_errors']
    assert result['selected_genre'] == 'unknown' and result['selected_genre_is_unknown']
    assert result['selection_mode'] == 'confidence-threshold-abstention'
    assert result['style_selection_reason'] == 'below-explicit-confidence-threshold-abstention'
    assert result['confidence_threshold_source'] == 'explicit-user-parameter'
    assert result['min_style_confidence'] == .9 and result['calibrated'] is False
    assert result['style_confidence'] < .9
    assert result['probability_classes'] == semantic_classes and len(result['probabilities']) == 22
    assert 'unknown' not in result['probability_classes']
    assert abs(sum(result['probabilities']) - 1) < 1e-6
    boundary = generate.generate_conditioned(adapter, source, 256, min_style_confidence=result['style_confidence'])
    assert boundary['caption_valid'] and boundary['selected_genre'] == 'contemporary'
    assert boundary['style_selection_reason'] == 'explicit-confidence-threshold-met'


@pytest.mark.parametrize('threshold', [-.1, 1.1, float('nan'), True])
def test_unknown_confidence_threshold_requires_explicit_valid_policy(formal_profile, threshold):
    adapter, source, _, _ = formal_profile
    with pytest.raises(ValueError, match='confidence threshold'):
        generate.generate_conditioned(adapter, source, 256, min_style_confidence=threshold)
    assert not adapter.model.forward_inputs


def test_threshold_cannot_override_explicit_selection_or_generic_profile(formal_profile, fixture):
    adapter, source, _, _ = formal_profile
    with pytest.raises(ValueError, match='head selection'):
        generate.generate_conditioned(adapter, source, 256, selected_genre='unknown', min_style_confidence=.9)
    generic, generic_source, _ = fixture
    with pytest.raises(ValueError, match='unknown policy'):
        generate.generate_conditioned(generic, generic_source, 128, min_style_confidence=.9)


def test_formal_head_must_not_treat_unknown_as_a_23rd_semantic_class(formal_profile):
    adapter, source, _, _ = formal_profile
    adapter.model._planner_jev_capture.metadata['classes'] = list(adapter.genre_classes)
    with pytest.raises(ValueError, match='semantic classes'):
        generate.generate_conditioned(adapter, source, 256)
    assert not adapter.model.forward_inputs


@pytest.mark.parametrize('bad_actions', [
    ENGLISH_ACTION_DRAFT.replace('[0.000-1.960s]', '[0-1.960s]'),
    ENGLISH_ACTION_DRAFT.replace('The dancer shifts', 'At frame 12, the dancer shifts'),
])
def test_formal_formatter_rejects_precision_or_frame_metadata_without_cleaning(formal_profile, bad_actions):
    adapter, source, _, _ = formal_profile
    adapter.model.continuation = adapter.tokenizer.encode(bad_actions) + [1]
    result = generate.generate_conditioned(adapter, source, 256, selected_genre='contemporary')
    assert result['caption'] is None and not result['caption_valid']
    assert result['raw_text'] == bad_actions + '<|im_end|>'
    assert result['raw_new_token_ids'] == adapter.model.continuation


def test_cli_exposes_optional_min_style_confidence():
    parsed = generate.parser().parse_args(['generate', '--model', 'm', '--checkpoint', 'c',
        '--row', 'audio.json', '--output', 'output.json', '--min-style-confidence', '0.8'])
    assert parsed.min_style_confidence == .8 and parsed.selected_genre is None
