"""Dense stereo music to H3 raw text hidden states; no generator changes."""
import math
from contextlib import nullcontext
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


INTERFACE_TYPE = 'h3_text_raw_hidden'
PREPROCESSING = {
    'input_sample_rate': 32000,
    'backbone_sample_rate': 24000,
    'channels': 2,
    'channel_order': 'left_right',
    'resample': 'torchaudio.functional.resample_default',
    'waveform_normalization': 'none',
    'dense_padding': 'after_backbone_with_explicit_valid_mask',
    'dense_quantization': 'float16_roundtrip_for_cached_and_online_features',
    'position_encoding': 'absolute_feature_index_sinusoidal_plus_stereo',
}
BACKENDS = {'muq', 'musicfm', 'mert'}
ONLINE_PREPROCESSING = dict(PREPROCESSING, dense_quantization='none_fp32_online_no_feature_cache')
DENSE_PARAMETER_SCOPE = ('model.conv', 'model.conformer.layer_norm', 'model.conformer.layers')


def _musicfm_dense_modules(backbone):
    if backbone.backend != 'musicfm':
        raise ValueError('Online training only supports MusicFM')
    modules = dict(backbone.named_modules())
    if any(name not in modules or next(modules[name].parameters(), None) is None for name in DENSE_PARAMETER_SCOPE):
        raise ValueError('MusicFM requires conv and Conformer layer_norm/layers dense modules')
    return tuple(modules[name] for name in DENSE_PARAMETER_SCOPE)


def _finite(value, name):
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f'{name} must be a finite tensor')


def _lengths(value, batch, maximum, device, name):
    if not isinstance(value, torch.Tensor) or value.shape != (batch,):
        raise ValueError(f'{name} must have shape [B]')
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f'{name} must be integer lengths')
    value = value.to(device=device)
    if torch.any((value < 1) | (value > maximum)):
        raise ValueError(f'{name} must be in [1,{maximum}]')
    return value


def _feature_mask(features, feature_lengths):
    batch, channels, frames, _ = features.shape
    if feature_lengths is None:
        raise ValueError('Explicit feature_lengths/valid mask is required; padding is never inferred')
    if not isinstance(feature_lengths, torch.Tensor):
        raise ValueError('feature_lengths must be a tensor')
    value = feature_lengths.to(device=features.device)
    if value.dtype == torch.bool:
        if value.shape == (batch, frames):
            value = value[:, None, :].expand(-1, channels, -1)
        if value.shape != (batch, channels, frames) or not value.any(-1).all():
            raise ValueError('Valid mask must be bool [B,2,T] (or [B,T]) with true = valid')
        return value
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError('feature_lengths must be integer lengths or a boolean valid mask')
    if value.shape == (batch,):
        value = value[:, None].expand(-1, channels)
    if value.shape != (batch, channels) or torch.any((value < 1) | (value > frames)):
        raise ValueError('feature_lengths must be [B] or [B,2], each in [1,T]')
    return torch.arange(frames, device=features.device)[None, None, :] < value[..., None]


def _position_encoding(count, width, device, dtype):
    positions = torch.arange(count, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32)
                            * (-math.log(10000.0) / width))
    angles = positions * frequencies[None, :]
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2).to(dtype=dtype)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, width=256, heads=8):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, queries, memory, valid_mask):
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(self.query_norm(queries), normalized_memory,
                                           normalized_memory, key_padding_mask=~valid_mask,
                                           need_weights=False)
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class MusicTextAdapter(nn.Module):
    """qmax comes from training token lengths, never a deployment ground-truth length."""
    def __init__(self, qmax, feature_dim=1024, raw_dim=5120, width=256, heads=8,
                 context_policy='tc_prefix_context', head_normalization='layer_norm'):
        super().__init__()
        if type(qmax) is not int or qmax < 1:
            raise ValueError('qmax must be a positive integer from training data')
        if (feature_dim, raw_dim) != (1024, 5120):
            raise ValueError('This interface requires feature_dim=1024, raw_dim=5120')
        if (type(width) is not int or width < 2 or width % 2 or type(heads) is not int
                or heads < 1 or width % heads):
            raise ValueError('width must be a positive even integer divisible by positive integer heads')
        if type(context_policy) is not str or context_policy not in ('tc_prefix_context', 'music_only_global_bias'):
            raise ValueError('Unsupported adapter context_policy')
        self.qmax = qmax
        self.feature_dim, self.raw_dim, self.width, self.heads = feature_dim, raw_dim, width, heads
        if type(head_normalization) is not str or head_normalization not in ('layer_norm', 'identity'):
            raise ValueError('Unsupported adapter head_normalization')
        self.context_policy = context_policy
        self.head_normalization = head_normalization
        self.memory_projection = nn.Linear(feature_dim, width)
        self.stereo_positions = nn.Parameter(torch.empty(2, width))
        self.query_positions = nn.Embedding(qmax, width)
        if context_policy == 'tc_prefix_context':
            self.context_projection = nn.Linear(raw_dim, width)
        else:
            self.global_context_bias = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList([_CrossAttentionBlock(width, heads), _CrossAttentionBlock(width, heads)])
        self.output_norm = nn.LayerNorm(width) if head_normalization == 'layer_norm' else nn.Identity()
        self.raw_output = nn.Linear(width, raw_dim)
        self.length_head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.normal_(self.stereo_positions, std=0.02)
        nn.init.normal_(self.query_positions.weight, std=0.02)
        self.register_buffer('output_center', None, persistent=True)
        self.register_buffer('output_basis', None, persistent=True)
        self.output_space_source_sha256 = None

    def _apply(self, fn, recurse=True):
        # Module.half()/to(dtype=...) must not quantize the fixed FP32 affine space.
        fixed = {name: self._buffers[name] for name in ('output_center', 'output_basis')}
        result = super()._apply(fn, recurse=recurse)
        for name, original in fixed.items():
            if original is not None:
                self._buffers[name] = original.to(device=self._buffers[name].device, dtype=torch.float32)
        return result

    def set_output_space(self, center, basis, source_sha256):
        for name, value in (('output_center', center), ('output_basis', basis)):
            _finite(value, name)
            if not value.is_floating_point():
                raise ValueError(name + ' must be floating-point')
        if (center.shape != (self.raw_dim,) or basis.ndim != 2 or basis.shape[0] != self.raw_dim
                or not 1 <= basis.shape[1] <= self.raw_dim):
            raise ValueError('Affine output space requires center [5120] and basis [5120,K], 1<=K<=5120')
        if (type(source_sha256) is not str or len(source_sha256) != 64
                or any(char not in '0123456789abcdef' for char in source_sha256)):
            raise ValueError('Output space requires its lowercase artifact SHA256')
        device = self.raw_output.weight.device
        center = center.detach().to(device=device, dtype=torch.float32).clone()
        basis = basis.detach().to(device=device, dtype=torch.float32).clone()
        _finite(center, 'FP32 output_center')
        _finite(basis, 'FP32 output_basis')
        with torch.autocast(device_type=device.type, enabled=False):
            gram = basis.T @ basis
        if not torch.allclose(gram, torch.eye(basis.shape[1], device=device), rtol=0, atol=1e-5):
            raise ValueError('Output basis must have orthonormal columns within absolute tolerance 1e-5')
        self.output_center, self.output_basis = center, basis
        self.output_space_source_sha256 = source_sha256
        return self

    def output_space_metadata(self):
        if self.output_basis is None:
            return {'mode': 'none'}
        return dict(mode='fixed_affine_projection', rank=self.output_basis.shape[1],
                    source_sha256=self.output_space_source_sha256,
                    projection='center+((raw_output(output_norm(queries))-center)@basis)@basis.T',
                    buffer_dtype='float32', trainable=False)

    def encode_queries(self, features, context=None, feature_lengths=None, query_count=None):
        """Differentiable internal queries before the trainable raw-output head."""
        if isinstance(features, Mapping):
            record = features
            metadata = record.get('metadata')
            if not isinstance(metadata, Mapping) or metadata.get('hidden_size') != self.feature_dim:
                raise ValueError('Feature records require metadata.hidden_size=1024')
            if metadata.get('backend') not in BACKENDS:
                raise ValueError('Feature records require the music backend in metadata')
            features = record.get('features')
            if feature_lengths is None:
                feature_lengths = record.get('valid_mask', record.get('feature_lengths'))
                if feature_lengths is None:
                    feature_lengths = metadata.get('valid_mask', metadata.get('feature_lengths'))
                if feature_lengths is None and metadata.get('all_frames_valid') is True:
                    if not isinstance(features, torch.Tensor) or features.ndim != 4:
                        raise ValueError('Feature records require [B,2,T,1024] features')
                    feature_lengths = torch.full((features.shape[0],), features.shape[2],
                                                 dtype=torch.long, device=features.device)
        _finite(features, 'features')
        if features.ndim != 4 or features.shape[1] != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError('Expected complete dense features [B,2,T,1024]')
        batch, _, frames, _ = features.shape
        if batch < 1 or frames < 1 or not features.is_floating_point():
            raise ValueError('Expected nonempty floating-point dense features')
        if features.device != self.memory_projection.weight.device:
            raise ValueError('Features and adapter must use the same device')
        if self.context_policy == 'music_only_global_bias':
            if context is not None:
                raise ValueError('music_only_global_bias forbids external context, including zero or cached TC context')
        else:
            _finite(context, 'context')
            if context.shape != (batch, self.raw_dim) or not context.is_floating_point():
                raise ValueError('Expected prefix last-row floating-point context [B,5120]')
            if features.device != context.device:
                raise ValueError('Features and context must use the same device')
        count = self.qmax if query_count is None else query_count
        if type(count) is not int or not 1 <= count <= self.qmax:
            raise ValueError('query_count must be an integer in [1,qmax]')
        valid_mask = _feature_mask(features, feature_lengths)
        dtype = self.memory_projection.weight.dtype
        features = features.to(dtype=dtype).masked_fill(~valid_mask[..., None], 0)
        memory = self.memory_projection(features)
        memory = memory + self.stereo_positions[None, :, None, :]
        memory = memory + _position_encoding(frames, self.width, memory.device, memory.dtype)[None, None]
        memory = memory.flatten(1, 2)
        flat_mask = valid_mask.flatten(1, 2)
        projected_context = (self.global_context_bias[None].expand(batch, -1)
                             if self.context_policy == 'music_only_global_bias'
                             else self.context_projection(context.to(dtype=dtype)))
        # No query self-attention or length-relative positions: short-query prefixes stay stable.
        queries = self.query_positions.weight[:count][None] + projected_context[:, None]
        for block in self.blocks:
            queries = block(queries, memory, flat_mask)
        summary = memory.masked_fill(~flat_mask[..., None], 0).sum(1) / flat_mask.sum(1)[:, None]
        log_length = self.length_head(torch.cat((summary, projected_context), dim=-1)).squeeze(-1)
        _finite(log_length, 'log_length')
        return queries, log_length

    def forward(self, features, context=None, feature_lengths=None, query_count=None):
        queries, log_length = self.encode_queries(features, context, feature_lengths, query_count)
        raw_tokens = self.raw_output(self.output_norm(queries))
        if self.output_basis is not None:
            with torch.autocast(device_type=raw_tokens.device.type, enabled=False):
                raw_tokens = self.output_center + ((raw_tokens.float() - self.output_center)
                                                   @ self.output_basis) @ self.output_basis.T
        _finite(raw_tokens, 'raw_tokens')
        return raw_tokens, log_length

    @torch.no_grad()
    def generate(self, features, context=None, feature_lengths=None):
        if self.training:
            raise RuntimeError('Call eval() before generation')
        raw_tokens, log_length = self(features, context, feature_lengths=feature_lengths)
        return truncate_predicted_tokens(raw_tokens, log_length, self.qmax)


def resample_audio32k_to24k(waveform):
    """Shared post-load_audio32k path for training and deployment, without normalization."""
    _finite(waveform, 'waveform')
    if waveform.ndim not in (2, 3) or waveform.shape[-2] != 2 or waveform.shape[-1] < 1:
        raise ValueError('Expected stereo [2,L] or [B,2,L] at 32kHz')
    import torchaudio
    result = torchaudio.functional.resample(waveform.float(), 32000, 24000)
    _finite(result, 'resampled waveform')
    return result


class MusicToH3Text(nn.Module):
    """Same dense backbone and raw-token adapter for training and deployment."""
    sample_rate = 32000

    def __init__(self, backbone, adapter, freeze_backbone=True):
        super().__init__()
        if not isinstance(adapter, MusicTextAdapter):
            raise ValueError('adapter must be MusicTextAdapter')
        if backbone.hidden_size != 1024 or backbone.sample_rate != 24000 or backbone.backend not in BACKENDS:
            raise ValueError('Expected a 24kHz, 1024-dimensional DenseMusicEncoder')
        self.backbone = backbone
        self.adapter = adapter
        self.freeze_backbone = bool(freeze_backbone)
        self.online_training = False
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def configure_online_training(self, train_backbone):
        """Opt into the FP32 A/B protocol; eval-mode MusicFM can still receive gradients."""
        if type(train_backbone) is not bool:
            raise ValueError('train_backbone must be an explicit boolean')
        dense_modules = _musicfm_dense_modules(self.backbone)
        self.backbone.requires_grad_(False)
        # Clear stale gradients when switching between A/B on an existing wrapper.
        for parameter in self.backbone.parameters():
            parameter.grad = None
        if train_backbone:
            for module in dense_modules:
                module.requires_grad_(True)
        self.adapter.requires_grad_(True)
        self.freeze_backbone = not train_backbone
        self.online_training = True
        self.float()
        self.backbone.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone or self.online_training:
            self.backbone.eval()
        return self

    def _online_precision_context(self, device):
        if not self.online_training:
            return nullcontext()
        if any(value.dtype != torch.float32 for value in self.parameters()):
            raise ValueError('Online music/text parameters must remain FP32')
        if any(value.is_floating_point() and value.dtype != torch.float32 for value in self.buffers()):
            raise ValueError('Online floating-point music/text buffers must remain FP32')
        return torch.autocast(device_type=device.type, enabled=False)

    def dense_features(self, waveform, waveform_lengths=None):
        _finite(waveform, 'waveform')
        if waveform.ndim != 3 or waveform.shape[1] != 2 or waveform.shape[0] < 1 or waveform.shape[-1] < 1:
            raise ValueError('Expected nonempty stereo [B,2,L] at 32kHz')
        batch, _, samples = waveform.shape
        if waveform_lengths is None:
            waveform_lengths = torch.full((batch,), samples, dtype=torch.long, device=waveform.device)
        lengths = _lengths(waveform_lengths, batch, samples, waveform.device, 'waveform_lengths')
        # Crop before encoding so waveform padding cannot contaminate valid backbone frames.
        encoded = [None] * batch
        for length in lengths.unique().tolist():
            indices = torch.nonzero(lengths == length, as_tuple=False).flatten()
            grad_context = torch.no_grad() if self.freeze_backbone else nullcontext()
            with self._online_precision_context(waveform.device), grad_context:
                audio = resample_audio32k_to24k(waveform[indices, :, :length])
                hidden = self.backbone(audio.flatten(0, 1))
            _finite(hidden, 'backbone features')
            if hidden.ndim != 3 or hidden.shape[0] != 2 * len(indices) or hidden.shape[-1] != 1024:
                raise ValueError('Invalid DenseMusicEncoder output')
            hidden = hidden.reshape(len(indices), 2, -1, 1024)
            for index, value in zip(indices.tolist(), hidden):
                encoded[index] = value
        feature_lengths = torch.tensor([value.shape[1] for value in encoded], device=waveform.device,
                                       dtype=torch.long)
        maximum = int(feature_lengths.max())
        features = torch.stack([F.pad(value, (0, 0, 0, maximum - value.shape[1])) for value in encoded])
        if self.freeze_backbone and not self.online_training:
            # Preserve the original FP16 feature-cache protocol for legacy checkpoints.
            features = features.half().float()
        return features, feature_lengths

    def forward(self, waveform, context=None, waveform_lengths=None, query_count=None):
        if self.adapter.context_policy == 'music_only_global_bias' and context is not None:
            raise ValueError('music_only_global_bias forbids external context before backbone encoding')
        features, feature_lengths = self.dense_features(waveform, waveform_lengths)
        with self._online_precision_context(features.device):
            return self.adapter(features, context, feature_lengths, query_count)

    def student_condition(self, waveform, context=None, waveform_lengths=None):
        """Differentiable full-qmax student condition; no caption or teacher length input."""
        raw_tokens, log_length = self(waveform, context, waveform_lengths)
        tails, lengths = truncate_predicted_tokens(raw_tokens, log_length, self.adapter.qmax)
        valid_mask = torch.arange(self.adapter.qmax, device=raw_tokens.device)[None] < lengths[:, None]
        return dict(raw_tokens=raw_tokens, log_length=log_length, lengths=lengths,
                    valid_mask=valid_mask, tails=tails)

    @torch.no_grad()
    def generate(self, waveform, context=None, waveform_lengths=None):
        if self.training:
            raise RuntimeError('Call eval() before generation')
        condition = self.student_condition(waveform, context, waveform_lengths)
        return condition['tails'], condition['lengths']


def convert_music_text_to_no_text_encoder(model, features, feature_lengths):
    """Convert legacy weights without reading text/cache; verify old zero-context equivalence."""
    if not isinstance(model, MusicToH3Text) or model.adapter.context_policy != 'tc_prefix_context':
        raise ValueError('noTC conversion requires a complete legacy tc_prefix_context wrapper')
    old = model.adapter
    before = model.state_dict()
    adapter = MusicTextAdapter(old.qmax, width=old.width, heads=old.heads,
                               context_policy='music_only_global_bias', head_normalization=old.head_normalization)
    adapter = adapter.to(device=old.memory_projection.weight.device, dtype=old.memory_projection.weight.dtype)
    state = dict(old.state_dict())
    del state['context_projection.weight']
    state['global_context_bias'] = state.pop('context_projection.bias').detach().clone()
    if old.output_basis is not None:
        adapter.set_output_space(old.output_center, old.output_basis, old.output_space_source_sha256)
    adapter.load_state_dict(state, strict=True)
    adapter.requires_grad_(True)
    adapter.train(old.training)
    with torch.no_grad():
        zero_context = torch.zeros(features.shape[0], old.raw_dim, device=features.device,
                                   dtype=old.memory_projection.weight.dtype)
        expected = old(features, zero_context, feature_lengths)
        actual = adapter(features, None, feature_lengths)
    if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
        raise ValueError('noTC conversion changed the legacy zero-context raw tokens or log length')
    model.adapter = adapter
    after = model.state_dict()
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    common = sorted(set(before) & set(after))
    if (removed != ['adapter.context_projection.bias', 'adapter.context_projection.weight']
            or added != ['adapter.global_context_bias']
            or not all(torch.equal(before[name], after[name]) for name in common)
            or not torch.equal(before['adapter.context_projection.bias'], after['adapter.global_context_bias'])):
        raise ValueError('noTC conversion must preserve every common adapter/backbone state and rename only the context bias')
    return dict(context_policy='music_only_global_bias', removed_state=removed, added_state=added,
                removed_context_weight='adapter.context_projection.weight',
                bias_rename={'adapter.context_projection.bias': 'adapter.global_context_bias'},
                preserved_common_state=common, common_state_exact=True, backbone_state_exact=True,
                old_zero_context_raw_tokens_exact=True, old_zero_context_log_length_exact=True,
                cached_context_read=False, text_encoder_called=False, all_remaining_adapter_parameters_trainable=True,
                initialization_is_not_no_tc_training_acceptance=True)


def masked_token_loss(student, teacher, valid_mask, mse_weight=1.0, cosine_weight=1.0):
    _finite(student, 'student raw tokens')
    _finite(teacher, 'teacher raw tokens')
    if student.ndim != 3 or student.shape != teacher.shape or student.shape[-1] != 5120:
        raise ValueError('Student and teacher must have matching [B,Q,5120] raw hidden states')
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise ValueError('valid_mask must be boolean with true = valid')
    if valid_mask.shape != student.shape[:2] or not valid_mask.any(1).all():
        raise ValueError('valid_mask must be [B,Q] with at least one valid token per example')
    if student.device != teacher.device or valid_mask.device != student.device:
        raise ValueError('Loss tensors must share a device')
    if not math.isfinite(mse_weight) or not math.isfinite(cosine_weight) or min(mse_weight, cosine_weight) < 0:
        raise ValueError('Loss weights must be finite and nonnegative')
    student_valid, teacher_valid = student[valid_mask].float(), teacher[valid_mask].detach().float()
    raw_mse = F.mse_loss(student_valid, teacher_valid)
    cosine_loss = (1 - F.cosine_similarity(student_valid, teacher_valid, dim=-1)).mean()
    student_energy = student_valid.square().mean().detach()
    teacher_energy = teacher_valid.square().mean().detach()
    return {
        'loss': mse_weight * raw_mse + cosine_weight * cosine_loss,
        'raw_mse': raw_mse,
        'cosine_loss': cosine_loss,
        'student_rms': student_energy.sqrt(),
        'teacher_rms': teacher_energy.sqrt(),
        'student_teacher_energy_ratio': student_energy / teacher_energy.clamp_min(1e-12),
        'teacher_student_energy_ratio': teacher_energy / student_energy.clamp_min(1e-12),
    }


def length_loss(log_length, target_lengths, qmax):
    _finite(log_length, 'log_length')
    if log_length.ndim != 1 or type(qmax) is not int or qmax < 1:
        raise ValueError('Expected log_length [B] and positive integer qmax')
    targets = _lengths(target_lengths, len(log_length), qmax, log_length.device, 'target_lengths')
    return F.mse_loss(log_length.float(), targets.float().log())


def predicted_lengths(log_length, qmax):
    _finite(log_length, 'log_length')
    if log_length.ndim != 1 or type(qmax) is not int or qmax < 1:
        raise ValueError('Expected log_length [B] and positive integer qmax')
    # Clamp in log space first to avoid exp overflow; equivalent to exp/round/clamp.
    return log_length.float().clamp(0, math.log(qmax)).exp().round().clamp(1, qmax).long()


def truncate_predicted_tokens(raw_tokens, log_length, qmax):
    _finite(raw_tokens, 'raw_tokens')
    if raw_tokens.ndim != 3 or raw_tokens.shape[-1] != 5120 or raw_tokens.shape[1] != qmax:
        raise ValueError('Generation must produce [B,qmax,5120] before predicted-length truncation')
    lengths = predicted_lengths(log_length, qmax)
    if lengths.shape != (raw_tokens.shape[0],):
        raise ValueError('One predicted length is required per example')
    return [value[:int(length)] for value, length in zip(raw_tokens, lengths)], lengths


def _validate_metadata(actual, expected, name='metadata'):
    if type(actual) is not type(expected):
        raise ValueError('Incompatible raw-text checkpoint ' + name)
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Incompatible raw-text checkpoint keys: ' + name)
        for key, value in expected.items():
            _validate_metadata(actual[key], value, name + '.' + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Incompatible raw-text checkpoint ' + name)
        for index, value in enumerate(expected):
            _validate_metadata(actual[index], value, name + f'[{index}]')
    elif actual != expected:
        raise ValueError('Incompatible raw-text checkpoint ' + name)


def _musicfm_normalization(stat):
    keys = ('melspec_2048_mean', 'melspec_2048_std')
    if not isinstance(stat, dict) or any(type(stat.get(key)) not in (int, float)
                                       or not math.isfinite(stat[key]) for key in keys):
        raise ValueError('MusicFM dense mel normalization must be explicit and finite')
    if stat['melspec_2048_std'] <= 0:
        raise ValueError('MusicFM dense mel normalization std must be positive')
    return {key: float(stat[key]) for key in keys}


def _online_training_metadata(model):
    dense_ids = {id(parameter) for module in _musicfm_dense_modules(model.backbone)
                 for parameter in module.parameters()}
    expected_names = sorted(name for name, parameter in model.backbone.named_parameters()
                            if id(parameter) in dense_ids and not model.freeze_backbone)
    actual_names = sorted(name for name, parameter in model.backbone.named_parameters() if parameter.requires_grad)
    if actual_names != expected_names or not all(parameter.requires_grad for parameter in model.adapter.parameters()):
        raise ValueError('Online checkpoint trainable flags do not match the explicit dense/adapter scope')
    if any(module.training for module in model.backbone.modules()):
        raise ValueError('Online checkpoint backbone must use eval-mode BN and stochastic layers')
    model._online_precision_context(model.adapter.memory_projection.weight.device)
    state_names = set(model.state_dict())
    return {
        'protocol': 'musicfm_dense_online_v1',
        'train_backbone': not model.freeze_backbone,
        'dense_parameter_scope': list(DENSE_PARAMETER_SCOPE),
        'trainable_backbone_parameters': actual_names,
        'adapter_scope': 'all_parameters_including_raw_output_and_length_head',
        'backbone_mode': 'eval_with_gradients_when_trainable',
        'bn_running_statistics': 'fixed',
        'bn_affine': 'frozen' if model.freeze_backbone else 'trainable_in_dense_scope',
        'stochastic_layers': 'disabled_by_backbone_eval_including_dropout_and_layerdrop',
        'parameter_dtype': 'float32',
        'floating_buffer_dtype': 'float32',
        'dense_feature_dtype': 'float32',
        'autocast': 'disabled',
        'feature_cache': 'forbidden',
        'student_queries': 'full_qmax',
        'student_length': 'round_exp_log_length_clamp_1_qmax',
        'student_mask': 'predicted_length_only',
        'buffer_policy': 'all_persistent_state_dict_buffers',
        'persistent_buffers': sorted(name for name, _ in model.named_buffers() if name in state_names),
        # MusicFM's JSON scalars are not registered buffers, but affect the dense forward.
        'backbone_normalization': _musicfm_normalization(model.backbone.model.stat),
    }


def _checkpoint_metadata(model, backend):
    wrapper = isinstance(model, MusicToH3Text)
    adapter = model.adapter if wrapper else model
    if not isinstance(adapter, MusicTextAdapter):
        raise ValueError('Checkpoint model must be MusicTextAdapter or MusicToH3Text')
    if wrapper:
        if backend is not None and backend != model.backbone.backend:
            raise ValueError('Checkpoint backend disagrees with the shared backbone')
        backend = model.backbone.backend
    if backend not in BACKENDS:
        raise ValueError('Checkpoint requires an explicit supported music backend')
    metadata = {
        'format_version': 1,
        'interface_type': INTERFACE_TYPE,
        'feature_dim': 1024,
        'raw_dim': 5120,
        'query_max': adapter.qmax,
        'width': adapter.width,
        'heads': adapter.heads,
        'layers': 2,
        'query_self_attention': False,
        'output_normalization': 'none_raw_hidden',
        'backend': backend,
        'preprocessing': dict(PREPROCESSING),
        'state_scope': 'music_to_h3_text' if wrapper else 'adapter',
        'freeze_backbone': model.freeze_backbone if wrapper else True,
    }
    if wrapper:
        metadata['backbone_metadata'] = model.backbone.metadata()
        if model.online_training:
            metadata['format_version'] = 2
            metadata['preprocessing'] = dict(ONLINE_PREPROCESSING)
            metadata['online_training'] = _online_training_metadata(model)
    if ((adapter.width, adapter.heads) != (256, 8) or adapter.output_basis is not None
            or adapter.context_policy == 'music_only_global_bias' or adapter.head_normalization != 'layer_norm'):
        metadata['format_version'] = 3
        metadata['output_space'] = adapter.output_space_metadata()
        metadata['head_normalization'] = adapter.head_normalization
    if adapter.context_policy == 'music_only_global_bias':
        metadata['context_policy'] = adapter.context_policy
    return metadata


def save_music_text_checkpoint(path, model, backend=None):
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    metadata = _checkpoint_metadata(model, backend)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for name, value in state.items():
        _finite(value, 'checkpoint weight ' + name)
    # Exclusive creation refuses an existing checkpoint, including concurrent saves.
    with path.open('xb') as stream:
        torch.save({'metadata': metadata, 'model': state}, stream)


def load_music_text_checkpoint(path, backbone=None, expected_backend=None, expected_query_max=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {'metadata', 'model'}:
        raise ValueError('Unsupported raw-text checkpoint structure')
    metadata, state = checkpoint['metadata'], checkpoint['model']
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError('Checkpoint metadata and model must be dictionaries')
    version = metadata.get('format_version')
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError('Unsupported raw-text checkpoint format_version')
    is_online = version == 2 or (version == 3 and 'online_training' in metadata)
    width, heads = (metadata.get('width'), metadata.get('heads')) if version == 3 else (256, 8)
    required = {
        'format_version': version, 'interface_type': INTERFACE_TYPE, 'feature_dim': 1024,
        'raw_dim': 5120, 'width': width, 'heads': heads, 'layers': 2,
        'query_self_attention': False, 'output_normalization': 'none_raw_hidden',
        'preprocessing': ONLINE_PREPROCESSING if is_online else PREPROCESSING,
    }
    for key, value in required.items():
        _validate_metadata(metadata.get(key), value, 'metadata.' + key)
    backend, qmax = metadata.get('backend'), metadata.get('query_max')
    if type(backend) is not str or backend not in BACKENDS or type(qmax) is not int or qmax < 1:
        raise ValueError('Invalid checkpoint backend or query_max')
    if expected_backend is not None and backend != expected_backend:
        raise ValueError('Checkpoint backend does not match deployment')
    if expected_query_max is not None and qmax != expected_query_max:
        raise ValueError('Checkpoint query_max does not match deployment')
    if type(metadata.get('freeze_backbone')) is not bool:
        raise ValueError('Checkpoint freeze_backbone must be explicit')
    online = metadata.get('online_training')
    if is_online:
        if (backend != 'musicfm' or metadata.get('state_scope') != 'music_to_h3_text'
                or not isinstance(online, dict) or type(online.get('train_backbone')) is not bool
                or online['train_backbone'] == metadata['freeze_backbone']):
            raise ValueError('V2 requires a complete MusicFM wrapper and explicit consistent online scope')
        normalization = _musicfm_normalization(online.get('backbone_normalization'))
        _validate_metadata(online['backbone_normalization'], normalization, 'metadata.online_training.backbone_normalization')
    for name, value in state.items():
        if not isinstance(name, str):
            raise ValueError('Checkpoint state names must be strings')
        _finite(value, 'checkpoint weight ' + name)
    context_policy = metadata.get('context_policy', 'tc_prefix_context') if version == 3 else 'tc_prefix_context'
    if type(context_policy) is not str or context_policy not in ('tc_prefix_context', 'music_only_global_bias'):
        raise ValueError('Invalid checkpoint context_policy')
    head_normalization = metadata.get('head_normalization', 'layer_norm') if version == 3 else 'layer_norm'
    adapter = MusicTextAdapter(qmax, width=width, heads=heads, context_policy=context_policy,
                               head_normalization=head_normalization)
    scope = metadata.get('state_scope')
    if version == 3:
        output_space = metadata.get('output_space')
        if not isinstance(output_space, dict):
            raise ValueError('V3 requires explicit output_space metadata')
        if output_space.get('mode') == 'fixed_affine_projection':
            prefix = 'adapter.' if scope == 'music_to_h3_text' else ''
            center, basis = (state.get(prefix + name) for name in ('output_center', 'output_basis'))
            if (not isinstance(center, torch.Tensor) or not isinstance(basis, torch.Tensor)
                    or center.dtype != torch.float32 or basis.dtype != torch.float32):
                raise ValueError('V3 fixed-space buffers must be present as FP32 tensors')
            adapter.set_output_space(center, basis, output_space.get('source_sha256'))
        elif output_space != {'mode': 'none'}:
            raise ValueError('Invalid V3 output_space mode or metadata')
    if scope == 'music_to_h3_text':
        backbone_metadata = metadata.get('backbone_metadata')
        if (not isinstance(backbone_metadata, dict) or backbone_metadata.get('backend') != backend
                or backbone_metadata.get('sample_rate') != 24000 or backbone_metadata.get('hidden_size') != 1024
                or not isinstance(backbone_metadata.get('model_path'), str)):
            raise ValueError('Invalid shared-backbone checkpoint metadata')
        if backbone is None:
            if __package__:
                from .music_encoder_adapter import DenseMusicEncoder
            else:
                from music_encoder_adapter import DenseMusicEncoder
            backbone = DenseMusicEncoder(backend, backbone_metadata['model_path'])
        model = MusicToH3Text(backbone, adapter, metadata['freeze_backbone'])
        if is_online:
            # Apply the saved scope before comparing metadata's requires_grad-derived counts.
            model.configure_online_training(online['train_backbone'])
            _musicfm_normalization(backbone.model.stat)
            backbone.model.stat.update(normalization)
        _validate_metadata(backbone.metadata(), backbone_metadata, 'metadata.backbone_metadata')
    elif scope == 'adapter':
        if backbone is not None:
            raise ValueError('Adapter-only checkpoint does not contain backbone weights')
        model = adapter
    else:
        raise ValueError('Invalid checkpoint state_scope')
    expected_metadata = _checkpoint_metadata(model, backend)
    if version == 3 and 'head_normalization' not in metadata:
        # Existing V3 checkpoints predate the configurable head and always used LayerNorm.
        expected_metadata.pop('head_normalization')
    _validate_metadata(metadata, expected_metadata)
    if version in (2, 3):
        expected_state = model.state_dict()
        if set(state) != set(expected_state):
            raise ValueError('V2/V3 checkpoint must contain the complete model state, including buffers')
        for name, value in state.items():
            if value.dtype != expected_state[name].dtype or value.shape != expected_state[name].shape:
                raise ValueError('V2/V3 checkpoint state shape/dtype disagrees with the FP32 protocol: ' + name)
    model.load_state_dict(state, strict=True)
    model = model.float().eval()
    for name, value in model.state_dict().items():
        _finite(value, 'loaded weight ' + name)
    model.checkpoint_metadata = metadata
    return model
