"""CPU native-shaped fusion protocols; not pretrained/ZeRO3/GPU acceptance."""
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from scripts_omnidance.exp.ms import music_planner_encoder_fusion as fusion
from scripts_omnidance.exp.ms import music_planner_jev as jev
from scripts_omnidance.exp.ms import music_planner_moss_generation_patch as cache_patch


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 2, 1)
        self.conv2 = nn.Conv2d(2, 2, 3, 2, 1)
        self.conv3 = nn.Conv2d(2, 2, 3, 2, 1)
        self.layer = nn.Linear(2, 4)
        self.checkpointing = False
        self.reentrant = True

    def forward(self, audio, lengths):
        x = self.conv3(self.conv2(self.conv1(audio[:, None])))
        x = x.mean(2).transpose(1, 2)[:, :int((lengths[0] + 7) // 8)]
        def run(values):
            return self.layer(values).tanh()
        return checkpoint(run, x, use_reentrant=self.reentrant) if self.checkpointing else run(x)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.layer = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.checkpointing, self.reentrant = False, True

    def forward(self, x, mask=None, deep=None):
        def run(values, ds):
            hidden = self.layer(values.cumsum(1)).tanh()
            if mask is not None:
                hidden = hidden.clone()
                hidden[mask] = hidden[mask] + ds.reshape(-1, 8)
            return hidden
        if deep is None:
            deep = x.new_empty(0, 8)
        x = checkpoint(run, x, deep, use_reentrant=self.reentrant) if self.checkpointing else run(x, deep)
        return self.norm(x)


class TinyNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type='moss_music', language_config=SimpleNamespace(hidden_size=8))
        self.embed = nn.Embedding(31, 8)
        self.audio_encoder = TinyEncoder()
        self.audio_adapter = nn.Linear(4, 8)
        self.deepstack_audio_merger_list = nn.ModuleList([nn.Linear(4, 8)])
        self.language_model = TinyDecoder()
        self.lm_head = nn.Linear(8, 31)
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, audio_data=None, audio_data_seqlens=None,
                audio_input_mask=None, past_key_values=None, inputs_embeds=None, **kwargs):
        # No side inputs may reach the released native model or language model.
        assert not (set(kwargs) & set(fusion.SIDE_KEYS))
        self.calls.append(audio_data is not None)
        values = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        deep = None
        if audio_data is not None:
            encoded = self.audio_encoder(audio_data, audio_data_seqlens)
            audio = self.audio_adapter(encoded)
            values = values.clone()
            values[audio_input_mask] = audio.reshape(-1, 8)
            deep = self.deepstack_audio_merger_list[0](encoded)
        hidden = self.language_model(values, audio_input_mask, deep)
        return SimpleNamespace(logits=self.lm_head(hidden))

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        cached = past_key_values is not None
        return dict(input_ids=input_ids[:, -1:] if cached else input_ids,
            past_key_values=past_key_values,
            audio_data=None if cached else kwargs.get('audio_data'),
            audio_data_seqlens=None if cached else kwargs.get('audio_data_seqlens'),
            audio_input_mask=None if cached else kwargs.get('audio_input_mask'))


class TinyPeft(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.base_model = native

    def get_base_model(self):
        return self.base_model

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


def adapter(peft=False):
    native = TinyNative()
    processor = SimpleNamespace(config=SimpleNamespace(mel_sr=16000, mel_hop_length=160, mel_dim=128))
    return SimpleNamespace(family='moss', native=native, model=TinyPeft(native) if peft else native,
                           processor=processor)


def batch(length=21, padded=None, feature_dim=6):
    tokens = (length + 7) // 8
    frames = (length + 3) // 4
    result = dict(input_ids=torch.arange(tokens + 3)[None],
        attention_mask=torch.ones(1, tokens + 3, dtype=torch.long),
        audio_data=torch.randn(1, 128, padded or length),
        audio_data_seqlens=torch.tensor([length]),
        audio_input_mask=torch.tensor([[False] + [True] * tokens + [False, False]]),
        musicfm_features=torch.randn(1, frames, feature_dim),
        musicfm_frame_times_seconds=torch.arange(frames, dtype=torch.float32)[None] * .04,
        musicfm_duration_seconds=torch.tensor([length * .01]))
    return result


@pytest.mark.parametrize('peft', [False, True])
def test_real_gradient_reaches_fusion_native_adapter_and_encoder(peft):
    torch.manual_seed(7)
    a = adapter(peft)
    metadata = fusion.install_fusion(a, feature_dim=6, hidden_dim=5)
    values = batch()
    values['musicfm_features'].requires_grad_(True)
    loss = a.model(**values).logits.square().mean()
    loss.backward()
    for module in (a.model.planner_fusion, a.native.audio_adapter, a.native.audio_encoder,
                   a.native.deepstack_audio_merger_list):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())
    assert values['musicfm_features'].grad is None  # frozen cached feature source
    assert all(p.dtype == torch.float32 for p in a.model.planner_fusion.parameters())
    assert all(type(m) is not nn.Linear for m in a.model.planner_fusion.modules() if isinstance(m, nn.Linear))
    assert metadata['deepstack'] == 'unchanged'
    assert metadata['chunk_packing_supported'] is False


@pytest.mark.parametrize('reentrant', [False, True])
def test_native_layer_checkpoint_replay_two_outstanding_batches_matches_uncheckpointed(reentrant):
    torch.manual_seed(42)
    plain, replay = adapter(), adapter()
    fusion.install_fusion(plain, 6, 5)
    fusion.install_fusion(replay, 6, 5)
    replay.model.load_state_dict(plain.model.state_dict())
    for layer in (replay.native.audio_encoder, replay.native.language_model):
        layer.checkpointing, layer.reentrant = True, reentrant
    batches = [batch(21), batch(29)]
    losses = []
    for a in (plain, replay):
        outputs = [a.model(**values).logits.square().sum() for values in batches]
        # Both forwards finish before backward; no "last side features" global.
        assert a.model._planner_fusion_capture.stack() == []
        total = sum(outputs)
        total.backward()
        losses.append(total.detach())
    torch.testing.assert_close(losses[0], losses[1])
    for (name, p), (other_name, q) in zip(plain.model.named_parameters(), replay.model.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize('reentrant', [False, True])
def test_explicit_whole_forward_checkpoint_closure_replays_side_inputs(reentrant):
    a = adapter()
    fusion.install_fusion(a, 6, 5)
    values = batch()
    audio = values.pop('audio_data').requires_grad_(True)
    def run(mel):
        return a.model(audio_data=mel, **values).logits
    checkpoint(run, audio, use_reentrant=reentrant).square().mean().backward()
    assert a.model.planner_fusion.network[1].weight.grad.abs().sum() > 0
    assert a.model._planner_fusion_capture.stack() == []


def test_absolute_grid_not_uniform_stretch_or_padded_native_length():
    a = adapter()
    metadata = fusion.install_fusion(a, 6, 5)
    request = fusion._request(batch(21, padded=120), {key: batch()[key] for key in fusion.SIDE_KEYS}, metadata)
    torch.testing.assert_close(request['native_times'], torch.tensor([[0., .08, .16]]))
    layer = fusion.DenseMusicFusion(2, 2, 2)
    layer.network = nn.Identity()
    layer.gate_logit.data.zero_()
    features = torch.arange(6.).view(1, 6, 1).expand(1, 6, 2)
    result = layer(torch.zeros(1, 3, 2), features, torch.arange(6.)[None] * .04,
                   request['native_times'])
    # .16 sec = side frame 4; normalized-length interpolation would read frame 5.
    torch.testing.assert_close(result, torch.tensor([[[0., 0.], [1., 1.], [2., 2.]]]))


def test_real_five_second_plus_clip_uses_full_valid_grid_without_chunk_stretch():
    a = adapter()
    metadata = fusion.install_fusion(a, 6, 5)
    values = batch(506)
    values['musicfm_duration_seconds'] = torch.tensor([5.0625])
    record = fusion._request(values, {key: values[key] for key in fusion.SIDE_KEYS}, metadata)
    assert record['tokens'] == 64
    assert values['musicfm_features'].shape[1] == 127
    torch.testing.assert_close(record['native_times'][0, -1], torch.tensor(5.04))
    torch.testing.assert_close(values['musicfm_frame_times_seconds'][0, -1], torch.tensor(5.04))
    longer = batch(850)
    record = fusion._request(longer, {key: longer[key] for key in fusion.SIDE_KEYS}, metadata)
    assert record['tokens'] == 107  # not split or packed into padded 5s pieces


def test_bf16_native_embedding_keeps_fp32_fusion_parameters_and_gradients():
    module = fusion.DenseMusicFusion(6, 5, 8)
    native = torch.randn(1, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    side = torch.randn(1, 6, 6, dtype=torch.bfloat16)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = module(native, side, torch.arange(6.)[None] * .04, torch.arange(3.)[None] * .08)
    assert result.dtype == torch.bfloat16
    result.float().square().mean().backward()
    assert native.grad is not None
    for parameter in module.parameters():
        assert parameter.dtype == parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize('peft', [False, True])
def test_generation_prefill_and_cached_decode_without_reencoding(peft):
    a = adapter(peft)
    fusion.install_fusion(a, 6, 5)
    values = batch()
    signature = inspect.signature(a.native.prepare_inputs_for_generation)
    assert all(key in signature.parameters for key in fusion.SIDE_KEYS)
    prefill = a.native.prepare_inputs_for_generation(**values)
    assert all(key in prefill for key in fusion.SIDE_KEYS)
    a.native(**prefill)
    cached = a.native.prepare_inputs_for_generation(**values, past_key_values=object())
    assert not (set(cached) & set(fusion.SIDE_KEYS))
    a.native(**cached)
    assert a.native.calls == [True, False]
    assert cached['input_ids'].shape == (1, 1)


@pytest.mark.parametrize('peft', [False, True])
def test_real_cache_patch_preserves_fusion_prefill_and_removes_decode_side(peft):
    a = adapter(peft)
    a.native._planner_native_forward = a.native.forward
    a.processor.audio_token_id = 30
    a.generate = lambda *args, **kwargs: {}
    fusion.install_fusion(a, 6, 5)
    fusion_prepare = a.native.prepare_inputs_for_generation
    cache_patch.apply_moss_generation_patch(a)
    assert a.native._planner_generation_native_prepare == fusion_prepare
    assert a.native.prepare_inputs_for_generation.__func__ is cache_patch._prepare_inputs_for_generation
    # HF checks generation kwargs against explicit prepare/forward signatures.
    assert all(key in inspect.signature(a.native.forward).parameters for key in fusion.SIDE_KEYS)
    values = batch()
    values['input_ids'][values['audio_input_mask']] = a.processor.audio_token_id
    prefill = a.native.prepare_inputs_for_generation(**values)
    assert all(key in prefill for key in fusion.SIDE_KEYS)
    a.native(**prefill)
    cache = SimpleNamespace(get_seq_length=lambda: values['input_ids'].shape[1] - 1)
    decoded = a.native.prepare_inputs_for_generation(**values, past_key_values=cache)
    assert not (set(decoded) & set(fusion.SIDE_KEYS))
    a.native(**decoded)
    assert a.native.calls == [True, False]


@pytest.mark.parametrize('order', ['jev-first', 'fusion-first'])
@pytest.mark.parametrize('peft', [False, True])
def test_jev_install_order_and_both_losses(order, peft):
    a = adapter(peft)
    if order == 'jev-first':
        jev.install_jev(a, ['ballet', 'hip-hop'], head_dim=4)
        fusion.install_fusion(a, 6, 5)
    else:
        fusion.install_fusion(a, 6, 5)
        jev.install_jev(a, ['ballet', 'hip-hop'], head_dim=4)
    values = batch()
    values['jev_prefix_end'] = torch.tensor([values['input_ids'].shape[1]])
    output = a.model(**values)
    loss = output.logits.square().mean() + output.jev_logits.square().mean()
    loss.backward()
    assert a.model.planner_fusion.network[1].weight.grad.abs().sum() > 0
    assert a.model.jev_head.network[1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('change,match', [
    (lambda b: b.pop('musicfm_features'), 'musicfm_features'),
    (lambda b: b.pop('musicfm_frame_times_seconds'), 'frame_times'),
    (lambda b: b.pop('musicfm_duration_seconds'), 'duration'),
    (lambda b: b.update(musicfm_features=torch.randn(1, 6, 7)), 'musicfm_features'),
    (lambda b: b['musicfm_features'].fill_(float('nan')), 'musicfm_features'),
    (lambda b: b.update(musicfm_frame_times_seconds=b['musicfm_frame_times_seconds'].bfloat16()), 'FP32'),
    (lambda b: b.update(musicfm_duration_seconds=b['musicfm_duration_seconds'].bfloat16()), 'FP32'),
    (lambda b: b['musicfm_frame_times_seconds'].add_(.01), 'grid'),
    (lambda b: b.update(musicfm_duration_seconds=torch.tensor([.4])), 'duration'),
    (lambda b: b.update(audio_data_seqlens=torch.tensor([0])), 'valid length'),
    (lambda b: b.update(audio_data_seqlens=torch.tensor([22])), 'valid length'),
    (lambda b: b.update(audio_data_seqlens=torch.tensor([21.])), 'seqlens'),
    (lambda b: b.update(audio_data=b['audio_data'].expand(2, -1, -1)), 'packed chunks'),
    (lambda b: b['audio_input_mask'].fill_(False), 'placeholders'),
    (lambda b: b.update(audio_input_mask=b['audio_input_mask'].long()), 'placeholders'),
    (lambda b: b.update(input_ids=torch.ones(1, 2, dtype=torch.long)), 'positions'),
    (lambda b: b.update(musicfm_features=torch.randn(1, 20, 6),
                       musicfm_frame_times_seconds=torch.arange(20.)[None] * .04), 'length'),
])
def test_invalid_shapes_lengths_grid_and_packing_fail_before_native_forward(change, match):
    a = adapter()
    fusion.install_fusion(a, 6, 5)
    values = batch()
    change(values)
    with pytest.raises(ValueError, match=match):
        a.model(**values)
    assert a.native.calls == []
    assert a.model._planner_fusion_capture.stack() == []


def test_missing_side_generation_and_direct_adapter_are_not_silent_fallbacks():
    a = adapter()
    fusion.install_fusion(a, 6, 5)
    values = batch()
    values.pop('musicfm_features')
    with pytest.raises(ValueError, match='all MusicFM'):
        a.native.prepare_inputs_for_generation(**values)
    with pytest.raises(ValueError, match='without explicit'):
        a.native.audio_adapter(torch.randn(1, 3, 4))


def test_native_padding_uses_valid_token_count_and_preserves_prefix_positions():
    a = adapter()
    fusion.install_fusion(a, 6, 5)
    values = batch()
    padded = dict(values, audio_data=torch.nn.functional.pad(values['audio_data'], (0, 100)))
    observed = []
    handle = a.native.audio_adapter.register_forward_hook(lambda m, i, o: observed.append(o.shape))
    a.model(**padded)
    handle.remove()
    assert observed == [(1, 3, 8)]
    assert torch.equal(values['audio_input_mask'], padded['audio_input_mask'])


@pytest.mark.parametrize('peft', [False, True])
def test_portable_roundtrip_trainable_and_identity(peft, tmp_path):
    a = adapter(peft)
    fusion.install_fusion(a, 6, 5)
    identity = {'run_fingerprint': 'abc', 'source': '音乐'}
    state = a.model.state_dict()
    saved = fusion.export_fusion(a, tmp_path / 'export', identity=identity, state_dict=state)
    b = adapter(peft)
    loaded = fusion.load_fusion(b, tmp_path / 'export', expected_identity=identity, trainable=True)
    assert saved == loaded
    assert all(p.requires_grad for p in b.model.planner_fusion.parameters())
    for name, value in a.model.planner_fusion.state_dict().items():
        torch.testing.assert_close(value, b.model.planner_fusion.state_dict()[name])
    with pytest.raises(ValueError, match='identity'):
        fusion.load_fusion(adapter(peft), tmp_path / 'export', expected_identity={'different': True})


def test_portable_eval_load_and_tamper_fail(tmp_path):
    a = adapter()
    fusion.install_fusion(a, 6, 5)
    fusion.export_fusion(a, tmp_path / 'export')
    b = adapter()
    fusion.load_fusion(b, tmp_path / 'export')
    assert not any(p.requires_grad for p in b.model.planner_fusion.parameters())
    path = tmp_path / 'export' / 'fusion.pt'
    path.write_bytes(path.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='content'):
        fusion.load_fusion(adapter(), tmp_path / 'export')


def test_export_requires_complete_fp32_gathered_state(tmp_path):
    a = adapter()
    fusion.install_fusion(a, 6, 5)
    for p in a.model.planner_fusion.parameters():
        p.ds_id = 1
    with pytest.raises(ValueError, match='gathered'):
        fusion.export_fusion(a, tmp_path / 'no-gather')
    state = a.model.state_dict()
    state.pop('planner_fusion.gate_logit')
    with pytest.raises(ValueError, match='every'):
        fusion.export_fusion(a, tmp_path / 'incomplete', state_dict=state)
    state = a.model.state_dict()
    state['planner_fusion.gate_logit'] = state['planner_fusion.gate_logit'].bfloat16()
    with pytest.raises(ValueError, match='FP32'):
        fusion.export_fusion(a, tmp_path / 'bf16', state_dict=state)


def test_install_idempotence_and_native_grid_change_rejected():
    a = adapter()
    metadata = fusion.install_fusion(a, 6, 5)
    assert fusion.install_fusion(a, 6, 5) == metadata
    with pytest.raises(ValueError, match='different widths'):
        fusion.install_fusion(a, 6, 7)
    b = adapter()
    b.native.audio_encoder.conv1.padding = (0, 0)
    with pytest.raises(ValueError, match='convolution grid'):
        fusion.install_fusion(b, 6, 5)
    b = adapter()
    b.processor.config.mel_hop_length = 320
    with pytest.raises(ValueError, match='preprocessing'):
        fusion.install_fusion(b, 6, 5)


def test_native_failure_clears_context():
    a = adapter()
    fusion.install_fusion(a, 6, 5)
    original = a.native.language_model.forward
    def fail(*args, **kwargs):
        raise RuntimeError('native failure')
    a.native.language_model.forward = fail
    with pytest.raises(RuntimeError, match='native failure'):
        a.model(**batch())
    assert a.model._planner_fusion_capture.stack() == []
    a.native.language_model.forward = original
    assert torch.isfinite(a.model(**batch()).logits).all()
