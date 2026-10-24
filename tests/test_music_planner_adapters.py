"""CPU native-processor tests; random tiny architectures are not pretrained training."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts_omnidance.exp.ms.music_planner_adapters import (
    FAMILIES, TASK_INSTRUCTION, PlannerAdapter, _local_moss_class,
    attach_lora, load_native_processor, load_planner_adapter, _cast_parameter_dtype,
)

METADATA = Path(os.environ.get('MUSIC_PLANNER_METADATA_ROOT',
    str(Path(__file__).resolve().parents[2] / 'models' / 'music_planner_metadata')))


@pytest.fixture(scope='session', autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def row(tmp_path):
    path = tmp_path / 'source.wav'
    time = np.arange(121500, dtype=np.float32) / 24000
    sf.write(path, .2 * np.sin(2 * np.pi * 220 * time), 24000, subtype='PCM_16')
    return dict(id='source', source_group_id='song', audio_path=str(path),
        audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        audio_window_seconds=[0., 5.0625],
        target_text='[0-2s] Steps left and sweeps both arms. [2-5.062s] Turns right with a low bounce.',
        target_windows=[dict(start=0., end=2.), dict(start=2., end=5.0625)])


class GenerationRecorder(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.tokenizer = tokenizer
        self.generation_config = SimpleNamespace(eos_token_id=tokenizer.eos_token_id)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        ids = self.tokenizer.encode('Repeated step. Repeated step.', add_special_tokens=False)
        ids.append(self.tokenizer.eos_token_id)
        return torch.cat((kwargs['input_ids'], torch.tensor([ids])), dim=1)


@pytest.fixture(params=FAMILIES)
def adapter(request):
    path = METADATA / request.param
    if not path.is_dir():
        pytest.skip('Local native processor metadata is not staged; no download performed')
    processor = load_native_processor(request.param, path)
    return PlannerAdapter(request.param, GenerationRecorder(processor.tokenizer), processor)


def test_native_prefix_assistant_eos_exact_mask(adapter, row):
    prefix = adapter.prefix(row)
    batch = adapter.collate([row])
    count = prefix['input_ids'].shape[1]
    assert torch.equal(batch['input_ids'][:, :count], prefix['input_ids'])
    assert batch['labels'][:, :count].eq(-100).all()
    expected = adapter.tokenizer.encode(row['target_text'], add_special_tokens=False)
    expected += adapter.tokenizer.encode('<|im_end|>', add_special_tokens=False)
    assert batch['labels'][0, count:].tolist() == expected
    assert expected[-1] in adapter.eos_token_ids()
    assert expected.count(expected[-1]) == 1
    assert batch['labels'].shape == batch['input_ids'].shape == batch['attention_mask'].shape
    feature_key = 'audio_data' if adapter.family == 'moss' else 'input_features'
    assert torch.equal(prefix[feature_key], batch[feature_key])
    assert prefix[feature_key].shape[-1] == (3000 if adapter.family == 'music_flamingo' else 506)
    if adapter.family == 'music_flamingo':
        mask = prefix['input_features_mask']
        assert mask.shape == (1, 3000)
        assert mask[:, :507].eq(1).all() and mask[:, 507:].eq(0).all()
        assert prefix['input_ids'].eq(adapter.processor.audio_token_id).sum() == 127
    if adapter.family == 'moss':
        assert adapter.processor.enable_time_marker is True
        assert batch['labels'][batch['audio_input_mask']].eq(-100).all()
        assert not batch['audio_input_mask'][:, count:].any()
        assert prefix['audio_input_mask'].sum() == 64
    else:
        audio_id = adapter.tokenizer.convert_tokens_to_ids(adapter.processor.audio_token)
        assert batch['input_ids'].eq(audio_id).sum() > 1
        assert batch['labels'][batch['input_ids'].eq(audio_id)].eq(-100).all()


def test_native_template_has_no_reference_or_captioner_instruction(adapter, row):
    prompt = adapter._prompt()
    assert row['target_text'] not in prompt
    if adapter.family == 'omni_captioner':
        assert TASK_INSTRUCTION not in prompt
        assert '<|im_start|>system' not in prompt
        assert prompt == '<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n'
    else:
        assert TASK_INSTRUCTION in prompt


def test_generate_never_reads_target_preserves_repetition_and_tokens(adapter, row):
    class AudioOnly(dict):
        def __getitem__(self, key):
            assert key not in ('target_text', 'target_windows', 'genre', 'caption', 'reference')
            return super().__getitem__(key)
        def get(self, key, *default):
            assert key not in ('target_text', 'target_windows', 'genre', 'caption', 'reference')
            return super().get(key, *default)
    audio_only = AudioOnly({k: v for k, v in row.items() if not k.startswith('target')})
    generated = adapter.generate(audio_only, max_new_tokens=128, seed=13)
    assert generated['raw_text'] == 'Repeated step. Repeated step.'
    assert generated['new_token_ids'][-1] in generated['eos_token_ids']
    assert generated['stop_reason'] == 'eos'
    assert adapter.model.calls[-1]['do_sample'] is False
    assert adapter.model.calls[-1]['num_beams'] == 1
    assert adapter.model.calls[-1]['repetition_penalty'] == 1.0
    assert adapter.model.calls[-1]['encoder_repetition_penalty'] == 1.0
    assert adapter.model.calls[-1]['no_repeat_ngram_size'] == 0
    assert generated['processor_sample_rate'] == 16000
    assert generated['n_samples'] == 81000
    assert 'max_length' not in adapter.model.calls[-1]
    assert generated['silence_applied'] is False
    assert 'labels' not in adapter.model.calls[-1]


def test_short_non_eos_output_is_unknown_not_native_stop(adapter, row, monkeypatch):
    monkeypatch.setattr(adapter.model, 'generate', lambda **kwargs: torch.cat(
        (kwargs['input_ids'], torch.tensor([[17]])), dim=1))
    generated = adapter.generate(row, max_new_tokens=128)
    assert generated['stop_reason'] == 'unknown_stop'
    assert generated['native_stop'] is False
    silent = adapter.generate(row, max_new_tokens=128, silence=True)
    assert silent['n_samples'] == 81000
    assert silent['silence_applied'] is True


@pytest.mark.parametrize('dtype, partition_dtype, trainable', [
    (torch.float32, torch.bfloat16, True),
    (torch.bfloat16, torch.float32, False),
])
def test_zero3_cast_preserves_partition_identity_shape_and_attributes(dtype, partition_dtype, trainable):
    parameter = torch.nn.Parameter(torch.zeros(0, dtype=torch.bfloat16), requires_grad=trainable)
    parameter.ds_shape = torch.Size([8, 4])
    parameter.ds_tensor = torch.tensor([.12345, -.6789, 1.2345, -2.3456], dtype=partition_dtype)
    parameter.ds_tensor.ds_numel = 4
    parameter.ds_tensor.status = 'fixture-available'
    parameter.ds_secondary_tensor = torch.tensor([.34567, -.45678], dtype=partition_dtype)
    parameter.ds_secondary_tensor.ds_numel = 2
    parameter.ds_secondary_tensor.status = 'fixture-secondary'
    primary, secondary = parameter.ds_tensor, parameter.ds_secondary_tensor
    expected_primary, expected_secondary = primary.to(dtype), secondary.to(dtype)
    _cast_parameter_dtype(parameter, dtype)
    assert parameter.dtype == primary.dtype == secondary.dtype == dtype
    assert parameter.ds_tensor is primary and parameter.ds_secondary_tensor is secondary
    assert torch.equal(primary, expected_primary) and torch.equal(secondary, expected_secondary)
    assert primary.ds_numel == 4 and primary.status == 'fixture-available'
    assert secondary.ds_numel == 2 and secondary.status == 'fixture-secondary'
    assert parameter.ds_shape == torch.Size([8, 4]) and parameter.numel() == 0
    assert primary.numel() == 4 and secondary.numel() == 2
    assert parameter.requires_grad is trainable


def test_zero3_frozen_bf16_parameter_fp32_storage_rejected_until_storage_cast():
    from scripts_omnidance.exp.ms.music_planner_train import optimizer_groups
    model = torch.nn.Module()
    model.base_weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
    model.base_weight.ds_tensor = torch.ones(4, dtype=torch.float32)
    model.lora_weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    with pytest.raises(ValueError, match='ZeRO partition dtype disagrees'):
        optimizer_groups(model, 1e-4, 2e-4)
    _cast_parameter_dtype(model.base_weight, torch.bfloat16)
    groups, names = optimizer_groups(model, 1e-4, 2e-4)
    assert len(groups) == 1 and names == {'lora': ['lora_weight'], 'projector': []}


def test_full_audio_sha_stat_cache_and_silence(adapter, row, monkeypatch):
    audio = adapter._audio(row)
    assert audio.shape == (81000,)
    assert audio.dtype == np.float32
    assert np.count_nonzero(audio) > 0
    with monkeypatch.context() as scoped:
        scoped.setattr(hashlib, 'sha256', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Repeated SHA')))
        assert np.array_equal(audio, adapter._audio(row))
    assert not np.count_nonzero(adapter._audio(row, silence=True))
    changed = dict(row, audio_sha256='0' * 64)
    with pytest.raises(ValueError, match='Audio bytes'):
        adapter._audio(changed)
    with pytest.raises(ValueError, match='physical window'):
        adapter._audio(dict(row, audio_window_seconds=[0., 2.]))
    sf.write(row['audio_path'], np.zeros(121500), 24000, subtype='PCM_16')
    with pytest.raises(ValueError, match='Audio bytes'):
        adapter._audio(row)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('rate,extra_frames,accepted', [
    (24000, 0, True), (24000, 1, True), (24000, 5, True),
    (24000, 6, False), (24000, -2, False),
    (16000, 5, False), (48000, 5, False),
])
def test_complete_source_positive_frame_boundary_without_crop_or_pad(tmp_path, family, rate, extra_frames, accepted):
    from scipy.signal import resample_poly
    path = tmp_path / 'boundary.wav'
    frames = int(5.0625 * rate) + extra_frames
    sf.write(path, np.linspace(-.2, .2, frames, dtype=np.float32), rate, subtype='PCM_16')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = dict(audio_path=str(path), audio_sha256=digest, audio_window_seconds=[0., 5.0625])
    native = PlannerAdapter(family, None, None)
    if not accepted:
        with pytest.raises(ValueError, match='complete 5.0625-second source'):
            native._audio(source)
        return
    waveform, actual_rate = sf.read(path, dtype='float32')
    divisor = math.gcd(actual_rate, 16000)
    expected = resample_poly(waveform, 16000 // divisor, actual_rate // divisor).astype(np.float32)
    actual = native._audio(source)
    assert np.array_equal(actual, expected)
    assert len(actual) == math.ceil(frames * 16000 / rate)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    silence = native._audio(source, silence=True)
    assert silence.shape == actual.shape and not np.count_nonzero(silence)
    with pytest.raises(ValueError, match='Audio bytes'):
        native._audio(dict(source, audio_sha256='0' * 64))


def test_invalid_context_and_microbatch_fail_without_truncation(adapter, row):
    with pytest.raises(ValueError, match='microbatch=1'):
        adapter.collate([row, row])
    with pytest.raises(ValueError, match='reserved chat'):
        adapter.collate([dict(row, target_text='<|im_end|>Injected turn')])
    length = adapter.prefix(row)['input_ids'].shape[1]
    adapter.max_length = length
    with pytest.raises(ValueError, match='assistant target exceeds'):
        adapter.collate([row])
    with pytest.raises(ValueError, match='Generation budget exceeds'):
        adapter.generate(row, max_new_tokens=1)
    adapter.max_length = length - 1
    with pytest.raises(ValueError, match='Expanded audio prefix'):
        adapter.prefix(row)


def tiny_model(family, dtype=torch.bfloat16):
    from transformers import (WhisperConfig, Qwen3Config, Qwen3OmniMoeThinkerConfig,
        Qwen3OmniMoeThinkerForConditionalGeneration, MusicFlamingoConfig,
        MusicFlamingoForConditionalGeneration)
    audio = dict(num_mel_bins=8, encoder_layers=5, encoder_attention_heads=2,
        encoder_ffn_dim=32, d_model=16, output_dim=32, max_source_positions=100,
        n_window=50, downsample_hidden_size=8, n_window_infer=100, conv_chunksize=100)
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, pad_token_id=0, eos_token_id=2,
        rope_parameters=dict(rope_type='default', rope_theta=10000., mrope_section=[1, 1, 2]),
        tie_word_embeddings=False, use_cache=False)
    if family == 'moss':
        config_cls = _local_moss_class(METADATA / family, 'configuration_moss_music', 'MossMusicConfig')
        model_cls = _local_moss_class(METADATA / family, 'modeling_moss_music', 'MossMusicModel')
        audio.update(output_dim=16, num_mel_bins=128)
        audio.update(deepstack_encoder_layer_indexes=[1, 3], layer_norm_eps=1e-5,
                     encoder_attention_window_size=100, dropout=0., attention_dropout=0.)
        config = config_cls(audio_config=WhisperConfig(**audio), language_config=Qwen3Config(**text),
            adapter_hidden_size=48, deepstack_num_inject_layers=2, audio_token_id=3)
    elif family == 'music_flamingo':
        # Native Flamingo uses AudioFlamingo3Encoder, not the full Whisper encoder/decoder.
        audio = dict(model_type='audioflamingo3_encoder', num_mel_bins=8, hidden_size=16,
                     num_hidden_layers=5, num_attention_heads=2, intermediate_size=32,
                     max_source_positions=16, dropout=0., attention_dropout=0.)
        text['model_type'] = 'qwen3'
        config = MusicFlamingoConfig(audio_config=audio, text_config=text, audio_token_id=3)
        model_cls = MusicFlamingoForConditionalGeneration
    else:
        vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
            out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[])
        text.update(num_experts=2, num_experts_per_tok=2, moe_intermediate_size=16,
            rope_parameters=dict(rope_type='default', rope_theta=10000.,
                                 mrope_section=[1, 1, 2], mrope_interleaved=True))
        config = Qwen3OmniMoeThinkerConfig(audio_config=audio, text_config=text,
            vision_config=vision, audio_token_id=3, audio_start_token_id=9,
            image_token_id=10, video_token_id=11)
        config.vision_start_token_id = 10
        model_cls = Qwen3OmniMoeThinkerForConditionalGeneration
    config._attn_implementation = 'eager'
    return model_cls(config).to(dtype)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_real_tiny_loader_syncs_requested_parameter_and_partition_dtype(family, dtype, monkeypatch):
    if family == 'moss' and not (METADATA / family).is_dir():
        pytest.skip('Local MOSS modeling source not staged')
    model = tiny_model(family).float()
    partitions = []
    for parameter in model.parameters():
        parameter.ds_tensor = parameter.detach().float().clone()
        parameter.ds_secondary_tensor = parameter.detach().float().flatten()[:2].clone()
        parameter.ds_tensor.status = 'fixture-loaded'
        partitions.append((parameter, parameter.ds_tensor, parameter.ds_secondary_tensor))
    def from_pretrained(*args, **kwargs):
        assert kwargs['dtype'] == dtype and kwargs['local_files_only'] is True
        return model
    monkeypatch.setattr(type(model), 'from_pretrained', from_pretrained)
    adapter = load_planner_adapter(family, METADATA / family, dtype=dtype)
    assert adapter.model is model
    for parameter, primary, secondary in partitions:
        assert parameter.dtype == primary.dtype == secondary.dtype == dtype
        assert parameter.ds_tensor is primary and parameter.ds_secondary_tensor is secondary
        assert primary.status == 'fixture-loaded'
        assert torch.equal(parameter.detach(), primary)
        assert torch.equal(parameter.detach().flatten()[:2], secondary)


@pytest.mark.parametrize('family', FAMILIES)
def test_real_tiny_loaded_zero3_full_scope_base_bf16_trainables_fp32(family, monkeypatch):
    if family == 'moss' and not (METADATA / family).is_dir():
        pytest.skip('Local MOSS modeling source not staged')
    import peft
    from scripts_omnidance.exp.ms.music_planner_train import optimizer_groups
    base = tiny_model(family)
    for parameter in base.parameters():
        parameter.ds_tensor = parameter.detach().float().clone()
        parameter.ds_secondary_tensor = parameter.detach().float().flatten()[:2].clone()
    monkeypatch.setattr(type(base), 'from_pretrained', lambda *args, **kwargs: base)
    native_get_peft_model = peft.get_peft_model
    partitions = []
    def get_peft_model(model, config):
        assert all(parameter.dtype == parameter.ds_tensor.dtype ==
                   parameter.ds_secondary_tensor.dtype == torch.bfloat16
                   for parameter in model.parameters())
        model = native_get_peft_model(model, config)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.ds_tensor = parameter.detach().to(torch.bfloat16).clone()
                parameter.ds_secondary_tensor = parameter.detach().flatten()[:2].to(torch.bfloat16).clone()
            parameter.ds_tensor.status = 'fixture-partitioned'
            partitions.append((parameter, parameter.ds_tensor, parameter.ds_secondary_tensor))
        return model
    monkeypatch.setattr(peft, 'get_peft_model', get_peft_model)
    adapter = load_planner_adapter(family, METADATA / family, lora_rank=2, lora_alpha=4)
    assert {item['scope'] for item in adapter.trainable_info['trainable']} == {
        'language_lora', 'audio_lora', 'projector'}
    for parameter, primary, secondary in partitions:
        expected_dtype = torch.float32 if parameter.requires_grad else torch.bfloat16
        assert parameter.dtype == primary.dtype == secondary.dtype == expected_dtype
        assert parameter.ds_tensor is primary and parameter.ds_secondary_tensor is secondary
        assert primary.status == 'fixture-partitioned'
    groups, names = optimizer_groups(adapter.model, 1e-4, 2e-4)
    assert {group['name'] for group in groups} == {'lora', 'projector'}
    assert set(names['lora'] + names['projector']) == {
        item['name'] for item in adapter.trainable_info['trainable']}


def test_real_tiny_omni_zero3_empty_fused_experts_keep_complete_lora_scope():
    from peft.tuners.lora.layer import ParamWrapper
    base = tiny_model('omni_captioner')
    expected = {}
    for name, parameter in base.named_parameters():
        if name.endswith(('.mlp.experts.gate_up_proj', '.mlp.experts.down_proj')):
            expected[name] = parameter.shape
            parameter.ds_shape = parameter.shape
            parameter.ds_tensor = parameter.detach().clone().flatten()[:4]
            parameter.data = torch.empty(0, dtype=parameter.dtype)
            assert parameter.ndim == 1 and len(parameter.ds_shape) == 3
    assert len(expected) == 6
    model, info = attach_lora(base, 'omni_captioner', rank=2, alpha=4)
    assert set(info['target_parameters']) == set(expected)
    assert set(model.base_model.targeted_parameter_names) == set(expected)
    wrappers = {module.parameter_name: [] for module in model.modules() if isinstance(module, ParamWrapper)}
    for name, module in model.named_modules():
        if not isinstance(module, ParamWrapper):
            continue
        parameter = module.get_param()
        assert parameter.numel() == 0 and parameter.ndim == 1
        assert module._param_ndim == 3 and module.num_experts == parameter.ds_shape[0]
        assert module.lora_A['default'].weight.requires_grad
        assert module.lora_B['default'].weight.requires_grad
        assert module.lora_A['default'].weight.dtype == module.lora_B['default'].weight.dtype == torch.float32
        wrappers[module.parameter_name].append(module)
    assert {name: len(items) for name, items in wrappers.items()} == {'gate_up_proj': 3, 'down_proj': 3}


@pytest.mark.parametrize('family', FAMILIES)
def test_real_tiny_lora_scope_dtype_and_portable_state(family, tmp_path, monkeypatch):
    if family == 'moss' and not (METADATA / family).is_dir():
        pytest.skip('Local MOSS modeling source not staged')
    from peft import PeftModel
    from peft.utils import get_peft_model_state_dict
    torch.manual_seed(19)
    base = tiny_model(family)
    pristine = copy.deepcopy(base.state_dict())
    model, info = attach_lora(base, family, rank=2, alpha=4)
    assert info['audio_last_layers'] == [1, 2, 3, 4]
    assert {r['scope'] for r in info['trainable']} == {'language_lora', 'audio_lora', 'projector'}
    assert bool(info['target_parameters']) == (family == 'omni_captioner')
    for name, value in model.named_parameters():
        assert value.dtype == (torch.float32 if value.requires_grad else torch.bfloat16)
        if value.requires_grad:
            assert not any(s in name for s in ('visual', 'lm_head', 'talker', 'code2wav'))
    # Full portable trainable dict mirrors trainer's ZeRO3 export, not partition tensors.
    with torch.no_grad():
        for value in model.parameters():
            if value.requires_grad:
                value.add_(.01)
    state = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    processor = load_native_processor(family, METADATA / family)
    adapter = PlannerAdapter(family, model, processor)
    adapter.trainable_info = info
    adapter.save_pretrained(tmp_path / 'adapter', state_dict=state)
    assert (tmp_path / 'adapter' / 'adapter_model.safetensors').is_file()
    restored = tiny_model(family)
    restored.load_state_dict(pristine)
    # Only heavyweight pretrained weight loading is substituted; native processor,
    # PEFT creation, FP32 allocation, state restoration and export reload are real.
    monkeypatch.setattr(type(restored), 'from_pretrained', lambda *args, **kwargs: restored)
    reloaded = load_planner_adapter(family, METADATA / family, checkpoint=tmp_path / 'adapter')
    restored = reloaded.model
    assert type(reloaded.processor).__name__ == type(processor).__name__
    expected, actual = get_peft_model_state_dict(model), get_peft_model_state_dict(restored)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert any('modules_to_save' in name for name in state)


@pytest.fixture
def zero3_cpu_module_hooks():
    # Use DeepSpeed's real registration/autograd hooks; only partition fetch/release
    # callbacks are CPU recorders, since these tiny parameters are not sharded.
    import types
    from deepspeed.runtime.zero import parameter_offload as offload
    original_stack = offload.FWD_MODULE_STACK[:]
    installed = []

    def install(model):
        events = {'pre_backward': [], 'post_backward': []}
        manager = SimpleNamespace(forward_hooks=[], backward_hooks=[])
        manager._register_deepspeed_module = types.MethodType(
            offload.DeepSpeedZeRoOffload._register_deepspeed_module, manager)
        manager.pre_sub_module_forward_function = offload.FWD_MODULE_STACK.append
        manager.post_sub_module_forward_function = lambda module: None
        manager.pre_sub_module_backward_function = events['pre_backward'].append
        manager.post_sub_module_backward_function = events['post_backward'].append
        for module in model.modules():
            module._external_params = {}
        manager._register_deepspeed_module(model, count=[0])
        offload.FWD_MODULE_STACK[:] = [model]
        installed.append(manager)
        return events

    yield install
    for manager in installed:
        for handle in manager.forward_hooks + manager.backward_hooks:
            handle.remove()
    offload.FWD_MODULE_STACK[:] = original_stack


@pytest.mark.parametrize('family, target, projector, width', [
    ('moss', 'audio_encoder.layers.4.self_attn.q_proj', 'audio_adapter', 16),
    ('music_flamingo', 'model.audio_tower.layers.4.self_attn.q_proj',
     'model.multi_modal_projector', 16),
    ('omni_captioner', 'audio_tower.layers.4.self_attn.q_proj', 'audio_tower.proj1', 16),
])
def test_real_peft_zero3_attribute_proxy_reproduces_wrong_hook_owner(
        family, target, projector, width, zero3_cpu_module_hooks):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(tiny_model(family).float(), LoraConfig(
        r=2, lora_alpha=4, target_modules=[target], modules_to_save=[projector]))
    wrapper = model.get_base_model().get_submodule(projector)
    active = wrapper.modules_to_save['default']
    assert not any('post_bwd_fn' in module.__dict__ for module in model.modules())
    zero3_cpu_module_hooks(model)
    assert 'post_bwd_fn' not in wrapper.__dict__
    assert wrapper.post_bwd_fn is active.post_bwd_fn
    with pytest.raises(AttributeError, match='ds_grads_remaining'):
        wrapper(torch.randn(1, 4, width, requires_grad=True))
    assert wrapper.ds_grads_remaining == 0
    assert 'ds_grads_remaining' not in active.__dict__


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('zero3_hooks', [False, True])
@pytest.mark.parametrize('reentrant', [False, True])
@pytest.mark.parametrize('all_fp32_storage, amp', [(False, True), (True, True), (True, False)])
def test_real_tiny_backward_checkpoint_recompute_and_all_scopes(
        family, zero3_hooks, reentrant, all_fp32_storage, amp, zero3_cpu_module_hooks):
    if family == 'moss' and not (METADATA / family).is_dir():
        pytest.skip('Local MOSS modeling source not staged')
    torch.manual_seed(23)
    base = tiny_model(family)
    if all_fp32_storage:
        base.float()
    model, info = attach_lora(base, family, rank=2, alpha=4)
    model.train()
    if zero3_hooks:
        events = zero3_cpu_module_hooks(model)
        for module in model.modules():
            for name in ('pre_bwd_fn', 'post_bwd_fn'):
                assert name in module.__dict__
                closure = getattr(module, name).forward.__closure__
                owners = [cell.cell_contents for cell in closure
                          if isinstance(cell.cell_contents, torch.nn.Module)]
                assert owners == [module]
    audio_tokens = 8 if family == 'music_flamingo' else 4
    ids = torch.tensor([[8, 9] + [3] * audio_tokens + [12, 7, 4, 5, 2]])
    features = torch.randn(1, 128 if family == 'moss' else 8, 32,
                           dtype=torch.bfloat16 if amp else torch.float32)
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    if family == 'moss':
        batch.update(audio_data=features, audio_data_seqlens=torch.tensor([32]), audio_input_mask=ids.eq(3))
    else:
        batch.update(input_features=features)
        key = 'input_features_mask' if family == 'music_flamingo' else 'feature_attention_mask'
        batch[key] = torch.ones(1, 32, dtype=torch.long)
    if family == 'omni_captioner':
        model.get_base_model().visual.register_forward_pre_hook(
            lambda *args: (_ for _ in ()).throw(AssertionError('Visual tower executed')))
    def forward_backward():
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
            output = model(**batch)
            loss = torch.nn.functional.cross_entropy(output.logits[0, -4:-1].float(), ids[0, -3:])
        loss.backward()
        return loss.detach(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    baseline_loss, baseline_gradients = forward_backward()
    assert torch.isfinite(baseline_loss)
    named = dict(model.named_parameters())
    for scope in ('language_lora', 'audio_lora', 'projector'):
        assert any(p['scope'] == scope and p['name'] in baseline_gradients and
                   baseline_gradients[p['name']].abs().sum() > 0 for p in info['trainable'])
    for projector in info['projectors']:
        assert any(projector + '.modules_to_save.' in name and grad.abs().sum() > 0
                   for name, grad in baseline_gradients.items())
    assert all(torch.isfinite(grad).all() for grad in baseline_gradients.values())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if zero3_hooks:
        called = {id(module) for module in events['pre_backward']}
        assert id(model) in called
        for projector in info['projectors']:
            wrapper = model.get_base_model().get_submodule(projector)
            assert id(wrapper) in called
            assert id(wrapper.modules_to_save['default']) in called
        assert events['post_backward']
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': reentrant})
    checkpoint_loss, checkpoint_gradients = forward_backward()
    assert torch.equal(baseline_loss, checkpoint_loss)
    assert baseline_gradients.keys() == checkpoint_gradients.keys()
    for name, grad in baseline_gradients.items():
        if amp and reentrant and family == 'moss' and not zero3_hooks:
            # Reentrant DeepStack branches accumulate BF16 gradients in a different
            # order; pure FP32 and DS-hook cases below retain the strict comparison.
            assert torch.allclose(grad, checkpoint_gradients[name], atol=2e-5, rtol=2e-2), name
        else:
            assert torch.allclose(grad, checkpoint_gradients[name], atol=1e-6, rtol=1e-5), name
    if family == 'moss':
        assert all(not any(getattr(hook, '_planner_deepstack', False) for hook in layer._forward_hooks.values())
                   for layer in model.get_base_model().language_model.layers)


def test_moss_native_fp32_storage_amp_fusion_failure_and_runtime_fix():
    model, info = attach_lora(tiny_model('moss', dtype=torch.float32), 'moss', rank=2, alpha=4)
    base = model.get_base_model()
    ids = torch.tensor([[8, 9, 3, 3, 3, 3, 12, 7, 4, 5, 2]])
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                 audio_data=torch.randn(1, 128, 32), audio_data_seqlens=torch.tensor([32]),
                 audio_input_mask=ids.eq(3))
    storage = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match='masked_scatter'):
            base._planner_native_forward(**batch)
        output = model(**batch)
        loss = torch.nn.functional.cross_entropy(output.logits[0, -4:-1].float(), ids[0, -3:])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.dtype == torch.float32 and torch.equal(parameter, storage[name])
               for name, parameter in model.named_parameters())
    for scope in ('audio_lora', 'language_lora', 'projector'):
        names = {item['name'] for item in info['trainable'] if item['scope'] == scope}
        assert any(name in names and parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for name, parameter in model.named_parameters())


@pytest.mark.parametrize('reentrant, early_stop', [(False, True), (False, False), (True, True)])
def test_real_ds_linear_released_parameter_checkpoint_metadata(
        reentrant, early_stop, zero3_cpu_module_hooks):
    from deepspeed.runtime.zero import parameter_offload
    from deepspeed.runtime.zero.linear import LinearFunctionForZeroStage3
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    class Linear(torch.nn.Linear):
        def forward(self, inputs):
            return LinearFunctionForZeroStage3.apply(inputs, self.weight, self.bias)
    model = torch.nn.Sequential(Linear(4, 4), torch.nn.Tanh(), Linear(4, 4)).requires_grad_(False)
    stored = {parameter: parameter.detach().clone() for parameter in model.parameters()}
    zero3_cpu_module_hooks(model)
    # The actual DS callbacks are module-bound closures, so substitute only the
    # recorder object's fetch/release methods with real Parameter.data transitions.
    first_hook = next(iter(model._forward_pre_hooks.values()))
    closure = getattr(first_hook, '__wrapped__', first_hook).__closure__
    manager = next(cell.cell_contents for cell in closure if isinstance(cell.cell_contents, SimpleNamespace))
    def fetch(module):
        for parameter in module.parameters(recurse=False):
            parameter.data = stored[parameter].clone()
    def release(module):
        for parameter in module.parameters(recurse=False):
            parameter.data = torch.empty(0, dtype=parameter.dtype)
    def forward_fetch(module):
        fetch(module)
        parameter_offload.FWD_MODULE_STACK.append(module)
    manager.pre_sub_module_forward_function = forward_fetch
    manager.post_sub_module_forward_function = release
    manager.pre_sub_module_backward_function = fetch
    manager.post_sub_module_backward_function = release
    inputs = torch.randn(2, 4, requires_grad=True)
    with set_checkpoint_early_stop(early_stop):
        outputs = checkpoint(model, inputs, use_reentrant=reentrant)
    if reentrant:
        outputs.sum().backward()
        assert inputs.grad.abs().sum() > 0
    else:
        with pytest.raises(torch.utils.checkpoint.CheckpointError, match='different metadata'):
            outputs.sum().backward()


@pytest.mark.skipif(not os.environ.get('MUSIC_PLANNER_ZERO3_DEVICE'),
                    reason='Explicit torchrun ZeRO3 diagnostic only; no implicit GPU allocation')
@pytest.mark.parametrize('family', FAMILIES)
def test_real_zero3_partitioned_native_reentrant_all_scopes(family):
    import deepspeed
    from deepspeed.utils import safe_get_full_grad
    from scripts_omnidance.exp.ms.music_planner_train import optimizer_groups
    device_type = os.environ['MUSIC_PLANNER_ZERO3_DEVICE']
    local_rank = int(os.environ['LOCAL_RANK'])
    if device_type == 'cuda':
        torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend='nccl' if device_type == 'cuda' else 'gloo',
                              auto_mpi_discovery=False)
    world = torch.distributed.get_world_size()
    assert world == 2, 'Only the explicitly allocated two-rank diagnostic is allowed'
    config = dict(train_batch_size=world, train_micro_batch_size_per_gpu=1,
                  gradient_accumulation_steps=1, bf16={'enabled': False},
                  fp16={'enabled': False},
                  torch_autocast={'enabled': True, 'dtype': 'bfloat16'},
                  zero_allow_untested_optimizer=True,
                  zero_force_ds_cpu_optimizer=False,
                  zero_optimization={'stage': 3, 'stage3_param_persistence_threshold': 0,
                                     'reduce_bucket_size': 1024,
                                     'stage3_prefetch_bucket_size': 0})
    torch.manual_seed(23)
    base = tiny_model(family, dtype=torch.float32)
    model, info = attach_lora(base, family, rank=2, alpha=4)
    # Official conversion of an already initialized tiny architecture avoids
    # pretending that random construction is a pretrained-weight load.
    with deepspeed.zero.Init(module=model, config_dict_or_path=config, dtype=torch.float32):
        pass
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    groups, _ = optimizer_groups(model, 1e-4, 2e-4, base_dtype=torch.float32)
    optimizer = torch.optim.AdamW(groups)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,
                                          config=config, dist_init_required=False)
    engine.train()
    audio_tokens = 8 if family == 'music_flamingo' else 4
    ids = torch.tensor([[8, 9] + [3] * audio_tokens + [12, 7, 4, 5, 2]], device=engine.device)
    features = torch.randn(1, 128 if family == 'moss' else 8, 32,
                           device=engine.device, dtype=torch.float32)
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    if family == 'moss':
        batch.update(audio_data=features, audio_data_seqlens=torch.tensor([32], device=engine.device),
                     audio_input_mask=ids.eq(3))
    else:
        key = 'input_features_mask' if family == 'music_flamingo' else 'feature_attention_mask'
        batch.update(input_features=features, **{key: torch.ones(1, 32, dtype=torch.long, device=engine.device)})
    loss = torch.nn.functional.cross_entropy(engine(**batch).logits[0, -4:-1].float(), ids[0, -3:])
    assert torch.isfinite(loss)
    assert any(parameter.numel() == 0 for parameter in engine.module.parameters())
    assert all(parameter.dtype == parameter.ds_tensor.dtype == torch.float32
               for parameter in engine.module.parameters())
    engine.backward(loss)
    gradients = {}
    for name, parameter in engine.module.named_parameters():
        if parameter.requires_grad:
            gradient = safe_get_full_grad(parameter)
            assert gradient is not None and torch.isfinite(gradient).all(), name
            gradients[name] = float(gradient.abs().sum())
        else:
            assert parameter.grad is None
    for scope in ('language_lora', 'audio_lora', 'projector'):
        assert any(item['scope'] == scope and gradients[item['name']] > 0 for item in info['trainable'])
    for projector in info['projectors']:
        assert any(projector + '.modules_to_save.' in name and norm > 0 for name, norm in gradients.items())
    engine.step()
    torch.distributed.barrier()
    engine.destroy()


def test_flamingo_complete_native_processor_fixed_positions_forward_backward(row):
    from transformers import MusicFlamingoForConditionalGeneration
    processor = load_native_processor('music_flamingo', METADATA / 'music_flamingo')
    config = copy.deepcopy(tiny_model('music_flamingo').config)
    config.audio_config.num_mel_bins = processor.feature_extractor.feature_size
    config.audio_config.max_source_positions = processor.feature_extractor.nb_max_frames // 2
    config.audio_token_id = processor.audio_token_id
    config.text_config.vocab_size = len(processor.tokenizer)
    config.text_config.pad_token_id = processor.tokenizer.pad_token_id
    config.text_config.eos_token_id = processor.tokenizer.eos_token_id
    config.text_config.max_position_embeddings = 1024
    model, info = attach_lora(MusicFlamingoForConditionalGeneration(config).float(),
                              'music_flamingo', rank=2, alpha=4)
    adapter = PlannerAdapter('music_flamingo', model, processor)
    waveform = adapter._audio(row)
    assert waveform.shape == (81000,)
    native = processor(text=[adapter._prompt()], audio=[waveform], sampling_rate=16000,
                       return_tensors='pt', text_kwargs={'padding': False})
    prefix = adapter.prefix(row)
    assert prefix.keys() == native.keys()
    assert all(torch.equal(prefix[key], native[key]) for key in prefix)
    assert prefix['input_features'].shape == (1, 128, 3000)
    mask = prefix['input_features_mask']
    assert mask[:, :507].eq(1).all() and mask[:, 507:].eq(0).all()
    assert prefix['input_ids'].eq(processor.audio_token_id).sum() == 127
    silent = adapter.prefix(row, silence=True)
    assert torch.equal(silent['input_features_mask'], mask)
    assert not torch.equal(silent['input_features'], prefix['input_features'])
    batch = adapter.collate([row])
    tower = model.get_base_model().model.audio_tower
    assert tower.embed_positions.weight.shape[0] == 1500
    positions = tower.embed_positions.weight.detach().clone()
    observed = []
    handle = tower.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(args[0].shape),
        with_kwargs=True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(**adapter._move(batch), logits_to_keep=4)
        loss = torch.nn.functional.cross_entropy(
            output.logits[0, :-1].float(), batch['labels'][0, -3:])
    loss.backward()
    handle.remove()
    assert observed == [torch.Size([1, 128, 3000])]
    assert torch.equal(tower.embed_positions.weight, positions)
    assert torch.isfinite(loss)
    for scope in ('language_lora', 'audio_lora', 'projector'):
        names = {item['name'] for item in info['trainable'] if item['scope'] == scope}
        assert any(name in names and parameter.grad is not None and
                   torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
                   for name, parameter in model.named_parameters())
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)


def test_teacher_forced_uses_shifted_assistant_tokens_only():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, **kwargs):
            assert 'labels' not in kwargs
            logits = torch.zeros(1, 5, 8)
            logits[0, 1, 3] = 2
            logits[0, 2, 4] = 3
            logits[0, 3, 2] = 4
            return SimpleNamespace(logits=logits)
    adapter = PlannerAdapter('moss', Model(), SimpleNamespace())
    labels = torch.tensor([[-100, -100, 3, 4, 2]])
    adapter.collate = lambda rows: dict(input_ids=torch.tensor([[7, 7, 3, 4, 2]]), labels=labels)
    actual = adapter.teacher_forced({'target_text': 'unused by model'})
    assert actual['token_count'] == 3
    expected = sum(math.log(math.exp(v) + 7) - v for v in (2, 3, 4))
    assert actual['sum'] == pytest.approx(expected)


def test_factory_rejects_nonlocal_and_invalid_family():
    with pytest.raises(ValueError, match='Unknown planner'):
        load_planner_adapter('generic', 'unused')
    with pytest.raises(ValueError, match='existing local'):
        load_planner_adapter('moss', '/nonexistent-native-model-path')
