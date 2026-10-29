"""Frozen MusicFM dense features as a residual on MOSS's final audio adapter.

This module does not load a MusicFM backbone. Its cached frame grid is 25 Hz,
starting at zero, and MOSS's native grid is 12.5 Hz, starting at zero. Sampling
uses absolute seconds, never the padded tensor length or clip-normalized time.

The official processor encodes a whole clip, not 5-second chunks. Version 1
supports exactly one audio/clip per native forward; packed/chunk batches fail.
Only the final adapter is modified; native DeepStack merger outputs and audio
placeholder positions are unchanged. Native Whisper/Qwen layer checkpointing
is supported: the fused embedding is an ordinary autograd input to the decoder,
and backward needs no ephemeral side-feature context. Whole-forward checkpoint
closures must, as usual, replay the explicit side-feature inputs.

Sources inspected 2026-10-29:
https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct/tree/main
modeling_moss_music.py sha256 fd8ae2c79dbfa591ebf12e353627d4343ef3e14ca6fbc852cb5e38fb94291c13
processing_moss_music.py sha256 c46df20dff4c4e93490fd12fe44f5ec817b241713a1d53386c3a897ffa9262c2
"""
import hashlib
import inspect
import json
from pathlib import Path
import threading
import types
import weakref

import torch
from torch import nn


PROTOCOL = 'moss-musicfm-dense-final-adapter-residual-v1'
PORTABLE_FORMAT = 'planner-musicfm-fusion-v1'
SIDE_KEYS = ('musicfm_features', 'musicfm_frame_times_seconds', 'musicfm_duration_seconds')
MUSICFM_HOP_SECONDS = .04
NATIVE_MEL_HOP_SECONDS = .01
NATIVE_AUDIO_STRIDE = 8


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate fusion metadata key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


class FP32FusionLinear(nn.Linear):
    """Keep exact-class DeepSpeed nn.Linear BF16 communication off this module."""


class DenseMusicFusion(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (feature_dim, hidden_dim, output_dim)):
            raise ValueError('Fusion widths must be positive integers')
        self.feature_dim, self.hidden_dim, self.output_dim = feature_dim, hidden_dim, output_dim
        self.network = nn.Sequential(nn.LayerNorm(feature_dim), FP32FusionLinear(feature_dim, hidden_dim),
                                     nn.GELU(), FP32FusionLinear(hidden_dim, output_dim))
        # Small, nonzero start permits gradients to both projectors on step 1.
        self.gate_logit = nn.Parameter(torch.tensor(-4., dtype=torch.float32))
        self.float()

    def forward(self, native, dense, frame_times, native_times):
        if (native.ndim != 3 or native.shape[0] != 1 or native.shape[2] != self.output_dim or
                dense.ndim != 3 or dense.shape[0] != 1 or dense.shape[2] != self.feature_dim or
                frame_times.shape != dense.shape[:2] or native_times.shape != native.shape[:2]):
            raise ValueError('Fusion requires aligned single-clip [1,T,D] embeddings and times')
        with torch.autocast(device_type=native.device.type, enabled=False):
            times = frame_times[0].to(device=native.device, dtype=torch.float32)
            query = native_times[0].to(device=native.device, dtype=torch.float32)
            dense = dense.detach().to(device=native.device, dtype=torch.float32)[0]
            # Clamp convolution boundary centers only; no clip-length stretching.
            if len(times) == 1:
                aligned = dense.expand(len(query), -1)
            else:
                upper = torch.searchsorted(times.contiguous(), query).clamp(1, len(times) - 1)
                lower = upper - 1
                weight = ((query - times[lower]) / (times[upper] - times[lower])).clamp(0., 1.)
                aligned = dense[lower] + weight[:, None] * (dense[upper] - dense[lower])
            residual = self.network(aligned).unsqueeze(0)
            result = native.float() + self.gate_logit.sigmoid() * residual
        return result.to(native.dtype)


def _base_model(model):
    getter = getattr(model, 'get_base_model', None)
    return getter() if callable(getter) else model


def _request(inputs, side, metadata):
    audio, lengths, mask = (inputs.get(key) for key in ('audio_data', 'audio_data_seqlens', 'audio_input_mask'))
    if (not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[:2] != (1, 128) or
            not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long or lengths.shape != (1,)):
        raise ValueError('Fusion supports one full native audio [1,128,Tmel] with int64 seqlens[1]; no packed chunks')
    length = int(lengths[0])
    if not 1 <= length <= audio.shape[-1]:
        raise ValueError('Native mel valid length is outside its input tensor')
    tokens = (length + NATIVE_AUDIO_STRIDE - 1) // NATIVE_AUDIO_STRIDE
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 2 or
            mask.shape[0] != 1 or int(mask.sum()) != tokens):
        raise ValueError('Native audio mask must contain exactly ceil(valid_mel_length/8) audio placeholders')
    ids = inputs.get('input_ids')
    embeds = inputs.get('inputs_embeds')
    if isinstance(ids, torch.Tensor) and mask.shape != ids.shape:
        raise ValueError('Native audio mask and input token positions differ')
    if isinstance(embeds, torch.Tensor) and mask.shape != embeds.shape[:2]:
        raise ValueError('Native audio mask and input embedding positions differ')
    features, times, duration = (side.get(key) for key in SIDE_KEYS)
    if (not isinstance(features, torch.Tensor) or not features.is_floating_point() or features.ndim != 3 or
            features.shape[0] != 1 or features.shape[1] < 1 or features.shape[2] != metadata['feature_dim'] or
            not torch.isfinite(features).all()):
        raise ValueError('musicfm_features must be finite floating [1,T,feature_dim]')
    if (not isinstance(times, torch.Tensor) or times.dtype not in (torch.float32, torch.float64) or
            times.shape != features.shape[:2] or not torch.isfinite(times).all()):
        raise ValueError('musicfm_frame_times_seconds must be finite FP32/FP64 [1,T], not BF16')
    expected = torch.arange(features.shape[1], device=times.device, dtype=torch.float64)[None] * MUSICFM_HOP_SECONDS
    if not torch.allclose(times.double(), expected, rtol=1e-6, atol=1e-6):
        raise ValueError('MusicFM frame times must use verified 0 + frame_index * .040 seconds grid')
    if (not isinstance(duration, torch.Tensor) or duration.dtype not in (torch.float32, torch.float64) or
            duration.shape != (1,) or not torch.isfinite(duration).all() or float(duration[0]) <= 0):
        raise ValueError('musicfm_duration_seconds must be positive finite FP32/FP64 [1]')
    seconds = float(duration[0])
    # Mel STFT centers give floor(duration*100) frames after dropping last frame.
    # One mel-frame tolerance permits sample-rate conversion rounding only.
    if not -.0001 <= seconds - length * NATIVE_MEL_HOP_SECONDS <= .0101:
        raise ValueError('Native valid mel length disagrees with the cached source duration')
    expected_frames = (length + 3) // 4
    if abs(features.shape[1] - expected_frames) > 1:
        raise ValueError('MusicFM dense length disagrees with its verified 25 Hz full-clip grid')
    if float(times[0, -1]) >= seconds or seconds - float(times[0, -1]) > .0501:
        raise ValueError('MusicFM frame extent disagrees with source duration; padding must be removed')
    native_times = torch.arange(tokens, device=audio.device, dtype=torch.float32)[None] * .08
    return dict(features=features, times=times, native_times=native_times, tokens=tokens, consumed=0)


class _FusionCapture:
    def __init__(self, owner, target, metadata):
        self.owner = weakref.ref(owner)
        self.metadata = metadata
        self.local = threading.local()
        self.handle = target.register_forward_hook(self.capture)

    def stack(self):
        if not hasattr(self.local, 'requests'):
            self.local.requests = []
        return self.local.requests

    def capture(self, module, inputs, output):
        stack = self.stack()
        if not stack or stack[-1] is None:
            raise ValueError('Native audio adapter ran without explicit MusicFM fusion inputs')
        record = stack[-1]
        if record['consumed']:
            raise ValueError('Native audio adapter executed more than once in a fusion forward')
        if (not isinstance(output, torch.Tensor) or output.shape !=
                (1, record['tokens'], self.metadata['output_dim'])):
            raise ValueError('Native audio adapter output differs from verified valid-length grid')
        record['consumed'] += 1
        return self.owner().planner_fusion(output, record['features'], record['times'], record['native_times'])


def _with_side_signature(function, original):
    signature = inspect.signature(original)
    parameters = list(signature.parameters.values())
    # Bound method signature lacks self; MethodType removes our first self again.
    parameters.insert(0, inspect.Parameter('self', inspect.Parameter.POSITIONAL_ONLY))
    present = set(signature.parameters)
    position = next((i for i, p in enumerate(parameters) if p.kind == inspect.Parameter.VAR_KEYWORD), len(parameters))
    for name in SIDE_KEYS:
        if name not in present:
            parameters.insert(position, inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None))
            position += 1
    function.__signature__ = signature.replace(parameters=parameters)


def install_fusion(adapter, feature_dim=1024, hidden_dim=256):
    """Install after any PEFT wrapping and before distributed/DeepSpeed wrapping.

    Parameters are registered on adapter.model as planner_fusion.*. Forward and
    prepare wrappers live on the native MOSS instance so native generate also
    sees them. JEV may be installed before or after this function.
    """
    if getattr(adapter, 'family', None) != 'moss':
        raise ValueError('MusicFM fusion currently supports native MOSS only')
    owner, native = adapter.model, _base_model(adapter.model)
    if hasattr(owner, 'module') or hasattr(owner, 'optimizer'):
        raise ValueError('Install fusion before distributed wrapping')
    existing = vars(owner).get('_planner_fusion_capture')
    if existing is not None:
        if (feature_dim, hidden_dim) != (existing.metadata['feature_dim'], existing.metadata['hidden_dim']):
            raise ValueError('Fusion already installed with different widths')
        return dict(existing.metadata)
    if 'planner_fusion' in dict(owner.named_children()):
        raise ValueError('Unrecognized existing planner_fusion module')
    if getattr(getattr(native, 'config', None), 'model_type', None) != 'moss_music':
        raise ValueError('Expected native MOSS Music model')
    processor_config = getattr(adapter.processor, 'config', None)
    if (getattr(processor_config, 'mel_sr', None) != 16000 or
            getattr(processor_config, 'mel_hop_length', None) != 160 or
            getattr(processor_config, 'mel_dim', None) != 128):
        raise ValueError('Fusion requires verified native 16 kHz / hop160 / 128mel preprocessing')
    for name in ('conv1', 'conv2', 'conv3'):
        layer = getattr(native.audio_encoder, name, None)
        if (not isinstance(layer, nn.Conv2d) or layer.kernel_size != (3, 3) or layer.stride != (2, 2) or
                layer.padding != (1, 1) or layer.dilation != (1, 1)):
            raise ValueError('Native convolution grid changed at audio_encoder.' + name)
    output_dim = getattr(getattr(native.config, 'language_config', None), 'hidden_size', None)
    if type(output_dim) is not int or output_dim < 1:
        raise ValueError('Native language hidden width is missing')
    metadata = dict(protocol=PROTOCOL, family='moss', feature_dim=feature_dim, hidden_dim=hidden_dim,
        output_dim=output_dim, target='audio_adapter.output', batch_size=1, chunk_packing_supported=False,
        musicfm_frame_hop_seconds=.04, musicfm_first_frame_seconds=0.,
        native_mel_hop_seconds=.01, native_audio_stride=8, native_first_frame_seconds=0.,
        feature_source='frozen-offline-musicfm-dense-cache', deepstack='unchanged',
        alignment='absolute-seconds-linear-interpolation-boundary-clamp',
        precision='fp32-fusion-math-parameters-and-default-zero3-communication-v1', code_sha256=_file_sha(__file__))
    fusion = DenseMusicFusion(feature_dim, hidden_dim, output_dim)
    fusion.to(device=next(owner.parameters()).device, dtype=torch.float32)
    owner.add_module('planner_fusion', fusion)
    capture = _FusionCapture(owner, native.audio_adapter, metadata)
    original = native.forward
    signature = inspect.signature(original)

    def forward(self, *args, **kwargs):
        side = {key: kwargs.pop(key, None) for key in SIDE_KEYS}
        bound = signature.bind_partial(*args, **kwargs)
        inputs = dict(bound.arguments)
        var_kwargs = next((name for name, value in signature.parameters.items()
                           if value.kind == inspect.Parameter.VAR_KEYWORD), None)
        if var_kwargs is not None:
            inputs.update(inputs.pop(var_kwargs, {}))
        record = _request(inputs, side, metadata) if inputs.get('audio_data') is not None else None
        stack = capture.stack()
        stack.append(record)
        try:
            result = original(*args, **kwargs)
            if record is not None and record['consumed'] != 1:
                raise ValueError('Native audio adapter did not consume the MusicFM input')
            return result
        finally:
            stack.pop()

    _with_side_signature(forward, original)
    native.forward = types.MethodType(forward, native)
    original_prepare = native.prepare_inputs_for_generation

    def prepare(self, *args, **kwargs):
        side = {key: kwargs.pop(key, None) for key in SIDE_KEYS}
        result = original_prepare(*args, **kwargs)
        if result.get('audio_data') is not None:
            if any(value is None for value in side.values()):
                raise ValueError('MOSS generation prefill requires all MusicFM side inputs')
            result.update(side)
        # HF keeps original model_kwargs for subsequent steps; omit audio side
        # tensors from this forward when native preparation drops the audio.
        return result

    _with_side_signature(prepare, original_prepare)
    native.prepare_inputs_for_generation = types.MethodType(prepare, native)
    owner._planner_fusion_capture = capture
    return dict(metadata)


def export_fusion(adapter, path, identity=None, state_dict=None):
    """Export FP32 parameters; ZeRO3 callers must supply gathered root state."""
    capture = vars(adapter.model).get('_planner_fusion_capture')
    if capture is None:
        raise ValueError('No installed fusion module to export')
    _json_sha(identity)
    fusion = adapter.model.planner_fusion
    expected = fusion.state_dict()
    if state_dict is None:
        if any(hasattr(p, 'ds_id') for p in fusion.parameters()):
            raise ValueError('ZeRO3 fusion export requires a gathered portable state_dict')
        saved = expected
    else:
        saved = {name[len('planner_fusion.'):]: value for name, value in state_dict.items()
                 if name.startswith('planner_fusion.')}
    if set(saved) != set(expected):
        raise ValueError('Portable state must contain every planner_fusion parameter')
    result, parameters = {}, dict(fusion.named_parameters())
    for name, value in saved.items():
        shape = tuple(getattr(parameters[name], 'ds_shape', parameters[name].shape))
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32 or
                not torch.isfinite(value).all()):
            raise ValueError('Invalid portable finite FP32 fusion tensor: ' + name)
        result[name] = value.detach().cpu().clone()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    torch.save(result, path / 'fusion.pt')
    metadata = dict(format=PORTABLE_FORMAT, installation=capture.metadata, identity=identity,
                    state_sha256=_file_sha(path / 'fusion.pt'))
    (path / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return metadata


def load_fusion(adapter, path, expected_identity=None, trainable=False):
    path = Path(path)
    metadata = _read_json(path / 'metadata.json')
    if (metadata.get('format') != PORTABLE_FORMAT or
            metadata.get('state_sha256') != _file_sha(path / 'fusion.pt')):
        raise ValueError('Fusion portable format/content identity differs')
    if expected_identity is not None and metadata.get('identity') != expected_identity:
        raise ValueError('Fusion checkpoint/model identity differs')
    spec = metadata['installation']
    actual = install_fusion(adapter, feature_dim=spec['feature_dim'], hidden_dim=spec['hidden_dim'])
    if actual != spec:
        raise ValueError('Fusion installation/native-time-grid identity differs')
    saved = torch.load(path / 'fusion.pt', map_location='cpu', weights_only=True)
    expected = adapter.model.planner_fusion.state_dict()
    if set(saved) != set(expected) or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
            value.shape != expected[name].shape or not torch.isfinite(value).all() for name, value in saved.items()):
        raise ValueError('Fusion keys/shapes/finite FP32 values differ')
    adapter.model.planner_fusion.load_state_dict(saved, strict=True)
    adapter.model.planner_fusion.requires_grad_(trainable)
    adapter.model.planner_fusion.train(adapter.model.training if trainable else False)
    return metadata
