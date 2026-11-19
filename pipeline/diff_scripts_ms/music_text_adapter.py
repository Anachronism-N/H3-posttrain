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
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(256)
        self.memory_norm = nn.LayerNorm(256)
        self.cross_attention = nn.MultiheadAttention(256, 8, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, queries, memory, valid_mask):
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(self.query_norm(queries), normalized_memory,
                                           normalized_memory, key_padding_mask=~valid_mask,
                                           need_weights=False)
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class MusicTextAdapter(nn.Module):
    """qmax comes from training token lengths, never a deployment ground-truth length."""
    def __init__(self, qmax, feature_dim=1024, raw_dim=5120, width=256):
        super().__init__()
        if type(qmax) is not int or qmax < 1:
            raise ValueError('qmax must be a positive integer from training data')
        if (feature_dim, raw_dim, width) != (1024, 5120, 256):
            raise ValueError('This interface requires feature_dim=1024, raw_dim=5120, width=256')
        self.qmax = qmax
        self.feature_dim, self.raw_dim, self.width = feature_dim, raw_dim, width
        self.memory_projection = nn.Linear(feature_dim, width)
        self.stereo_positions = nn.Parameter(torch.empty(2, width))
        self.query_positions = nn.Embedding(qmax, width)
        self.context_projection = nn.Linear(raw_dim, width)
        self.blocks = nn.ModuleList([_CrossAttentionBlock(), _CrossAttentionBlock()])
        self.output_norm = nn.LayerNorm(width)
        self.raw_output = nn.Linear(width, raw_dim)
        self.length_head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.normal_(self.stereo_positions, std=0.02)
        nn.init.normal_(self.query_positions.weight, std=0.02)

    def forward(self, features, context, feature_lengths=None, query_count=None):
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
        _finite(context, 'context')
        if features.ndim != 4 or features.shape[1] != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError('Expected complete dense features [B,2,T,1024]')
        batch, _, frames, _ = features.shape
        if batch < 1 or frames < 1 or context.shape != (batch, self.raw_dim):
            raise ValueError('Expected nonempty dense features and prefix last-row context [B,5120]')
        if not features.is_floating_point() or not context.is_floating_point():
            raise ValueError('Features and raw context must be floating-point tensors')
        if features.device != context.device or features.device != self.memory_projection.weight.device:
            raise ValueError('Features, context, and adapter must use the same device')
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
        projected_context = self.context_projection(context.to(dtype=dtype))
        # No query self-attention or length-relative positions: short-query prefixes stay stable.
        queries = self.query_positions.weight[:count][None] + projected_context[:, None]
        for block in self.blocks:
            queries = block(queries, memory, flat_mask)
        raw_tokens = self.raw_output(self.output_norm(queries))
        summary = memory.masked_fill(~flat_mask[..., None], 0).sum(1) / flat_mask.sum(1)[:, None]
        log_length = self.length_head(torch.cat((summary, projected_context), dim=-1)).squeeze(-1)
        _finite(raw_tokens, 'raw_tokens')
        _finite(log_length, 'log_length')
        return raw_tokens, log_length

    @torch.no_grad()
    def generate(self, features, context, feature_lengths=None):
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

    def forward(self, waveform, context, waveform_lengths=None, query_count=None):
        features, feature_lengths = self.dense_features(waveform, waveform_lengths)
        with self._online_precision_context(features.device):
            return self.adapter(features, context, feature_lengths, query_count)

    def student_condition(self, waveform, context, waveform_lengths=None):
        """Differentiable full-qmax student condition; no caption or teacher length input."""
        raw_tokens, log_length = self(waveform, context, waveform_lengths)
        tails, lengths = truncate_predicted_tokens(raw_tokens, log_length, self.adapter.qmax)
        valid_mask = torch.arange(self.adapter.qmax, device=raw_tokens.device)[None] < lengths[:, None]
        return dict(raw_tokens=raw_tokens, log_length=log_length, lengths=lengths,
                    valid_mask=valid_mask, tails=tails)

    @torch.no_grad()
    def generate(self, waveform, context, waveform_lengths=None):
        if self.training:
            raise RuntimeError('Call eval() before generation')
        condition = self.student_condition(waveform, context, waveform_lengths)
        return condition['tails'], condition['lengths']


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
        'width': 256,
        'heads': 8,
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
    if type(version) is not int or version not in (1, 2):
        raise ValueError('Unsupported raw-text checkpoint format_version')
    required = {
        'format_version': version, 'interface_type': INTERFACE_TYPE, 'feature_dim': 1024,
        'raw_dim': 5120, 'width': 256, 'heads': 8, 'layers': 2,
        'query_self_attention': False, 'output_normalization': 'none_raw_hidden',
        'preprocessing': PREPROCESSING if version == 1 else ONLINE_PREPROCESSING,
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
    if version == 2:
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
    adapter = MusicTextAdapter(qmax)
    scope = metadata.get('state_scope')
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
        if version == 2:
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
    _validate_metadata(metadata, _checkpoint_metadata(model, backend))
    if version == 2:
        expected_state = model.state_dict()
        if set(state) != set(expected_state):
            raise ValueError('V2 checkpoint must contain the complete model state, including buffers')
        for name, value in state.items():
            if value.dtype != expected_state[name].dtype or value.shape != expected_state[name].shape:
                raise ValueError('V2 checkpoint state shape/dtype disagrees with the FP32 protocol: ' + name)
    model.load_state_dict(state, strict=True)
    model = model.float().eval()
    for name, value in model.state_dict().items():
        _finite(value, 'loaded weight ' + name)
    model.checkpoint_metadata = metadata
    return model
