"""Local probability-decision auxiliary task for MOSS and Qwen Captioner.

This borrows Jev's decision-interface idea, not its private RLCD implementation.
One native causal forward supplies both caption logits and a categorical genre
decision.  The feature is the last INPUT prefix state, before any answer token.
The initial protocol supports hard/soft categorical labels, not multi-labels.

Primary reference: https://docs.typesafe.ai/introduction . The hosted Jev API
does not accept audio/video (https://docs.typesafe.ai/concepts/state); this
module is our local auxiliary classifier, not an integration of that service.
"""
import functools
import hashlib
import json
import math
from collections.abc import Mapping, MutableMapping
from pathlib import Path
import threading
import types
import weakref

import torch
from torch import nn
from torch.nn import functional as F


PROTOCOL = 'planner-jev-prefix-decision-v1'
FEATURE_PROTOCOL = 'native-causal-prefill-last-valid-token-v1'
HEAD_FORMAT = 'planner-jev-portable-head-v1'
HEAD_PRECISION_PROTOCOL = 'fp32-compute-and-default-zero3-communication-v1'
FAMILIES = ('moss', 'omni_captioner')


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate JSON object key: ' + name)
        result[name] = value
    return result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object)


def validate_classes(classes):
    if (not isinstance(classes, list) or len(classes) < 2 or
            any(not isinstance(name, str) or not name.strip() or name != name.strip() or
                len(name) > 80 or any(ord(ch) < 32 for ch in name) for name in classes) or
            len(set(classes)) != len(classes)):
        raise ValueError('Classes must be a JSON array of >=2 unique nonempty plain genre names')
    return list(classes)


def read_classes(path):
    return validate_classes(_read_json(path))


def classes_identity(classes):
    return _object_sha(validate_classes(classes))


def _base_model(model):
    getter = getattr(model, 'get_base_model', None)
    return getter() if callable(getter) else model


def _norm_and_width(adapter, hidden_dim):
    if adapter.family not in FAMILIES:
        raise ValueError('JEV branch supports moss and omni_captioner only')
    base = _base_model(adapter.model)
    path = 'language_model.norm' if adapter.family == 'moss' else 'model.norm'
    try:
        norm = base.get_submodule(path)
    except (AttributeError, KeyError) as error:
        raise ValueError('Native decoder final norm is missing: ' + path) from error
    candidates = []
    config = getattr(base, 'config', None)
    text_config = getattr(config, 'language_config' if adapter.family == 'moss' else 'text_config', None)
    for owner in (text_config, getattr(base.get_submodule(path.rsplit('.', 1)[0]), 'config', None)):
        width = getattr(owner, 'hidden_size', None)
        if width is not None:
            candidates.append(width)
    weight = getattr(norm, 'weight', None)
    if isinstance(weight, torch.Tensor):
        shape = tuple(getattr(weight, 'ds_shape', weight.shape))
        if len(shape) == 1 and shape[0] > 0:
            candidates.append(shape[0])
    if hidden_dim is not None:
        candidates.append(hidden_dim)
    if (not candidates or any(type(value) is not int or value <= 0 for value in candidates) or
            len(set(candidates)) != 1):
        raise ValueError('Native decoder hidden width is missing or conflicting')
    return norm, path, candidates[0]


class FP32DecisionLinear(nn.Linear):
    """Keep head parameters outside DeepSpeed's default AMP communication list.

    DeepSpeed matches exact nn.Linear classes for BF16 communication. Its gather
    handles restore the storage dtype, but the communication round-trip still
    rounds values. This dedicated subclass preserves FP32 communication without
    changing native-model Linear/AMP policy. Custom DS safe lists remain explicit.
    """


class JevHead(nn.Module):
    def __init__(self, hidden_dim, classes, head_dim=256):
        super().__init__()
        classes = validate_classes(classes)
        if type(hidden_dim) is not int or type(head_dim) is not int or min(hidden_dim, head_dim) < 1:
            raise ValueError('Positive integer JEV hidden/head dimensions required')
        self.network = nn.Sequential(nn.LayerNorm(hidden_dim), FP32DecisionLinear(hidden_dim, head_dim),
                                     nn.GELU(), FP32DecisionLinear(head_dim, len(classes)))
        self.float()

    def forward(self, features):
        # Parent native forward may be inside BF16 autocast. Keep head math FP32.
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.network(features.float())


class _ForwardCapture:
    def __init__(self, model, norm, metadata):
        self.model = weakref.ref(model)
        self.metadata = metadata
        self.local = threading.local()
        self.handle = norm.register_forward_hook(self.capture)

    def stack(self):
        if not hasattr(self.local, 'requests'):
            self.local.requests = []
        return self.local.requests

    def capture(self, module, inputs, output):
        requests = self.stack()
        # Gradient-checkpoint replay after the forward has no active request.
        if not requests or requests[-1] is None:
            return
        record = requests[-1]
        if record['features'] is not None:
            raise ValueError('Native decoder norm ran more than once in one JEV forward')
        if (not isinstance(output, torch.Tensor) or output.ndim != 3 or
                output.shape[0] != len(record['ends']) or
                output.shape[1] != record['sequence_length'] or
                output.shape[2] != self.metadata['hidden_dim']):
            raise ValueError('Native final norm must expose aligned [B,T,D] token states')
        ends = record['ends'].to(output.device)
        features = output[torch.arange(len(ends), device=output.device), ends - 1]
        if not torch.isfinite(features).all():
            raise FloatingPointError('Nonfinite input-prefix JEV feature')
        if (torch.is_grad_enabled() and self.model().training and
                any(p.requires_grad for n, p in self.model().named_parameters() if not n.startswith('jev_head.')) and
                not features.requires_grad):
            raise ValueError('JEV native prefix features lost the training graph')
        record['features'] = features


def _request(inputs, ends):
    ids, mask = inputs.get('input_ids'), inputs.get('attention_mask')
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or
            not isinstance(ends, torch.Tensor) or ends.dtype != torch.long or
            ends.ndim != 1 or len(ends) != ids.shape[0] or
            (ends <= 0).any() or (ends > ids.shape[1]).any()):
        raise ValueError('jev_prefix_end must be int64 [B] exclusive native-prefix lengths')
    if (not isinstance(mask, torch.Tensor) or mask.shape != ids.shape or
            not mask[torch.arange(len(ends), device=mask.device), ends.to(mask.device) - 1].bool().all()):
        raise ValueError('JEV prefix must end at an unpadded input token')
    labels = inputs.get('labels')
    if labels is not None:
        if not isinstance(labels, torch.Tensor) or labels.shape != ids.shape:
            raise ValueError('JEV native labels must align with input tokens')
        prefix = torch.arange(ids.shape[1], device=labels.device)[None] < ends.to(labels.device)[:, None]
        if labels[prefix].ne(-100).any():
            raise ValueError('JEV prefix may not include teacher-forced answer labels')
    return dict(ends=ends, sequence_length=ids.shape[1], features=None)


def install_jev(adapter, classes, hidden_dim=None, head_dim=256):
    """Install before DDP/ZeRO: native forward + ``jev_prefix_end`` -> jev_logits.

    No metadata means ordinary native generation, with no head call. The final
    prefix token includes the native instruction (MOSS) or audio-only template
    (Captioner), never assistant style/action targets. Do not reuse old audio-token
    pooled cache heads: this is a distinct feature protocol.
    """
    classes = validate_classes(classes)
    model = adapter.model
    if hasattr(model, 'module') or hasattr(model, 'optimizer'):
        raise ValueError('Install JEV before distributed wrapping')
    existing = vars(model).get('_planner_jev_capture')
    if existing is not None:
        expected = existing.metadata
        if (classes != expected['classes'] or head_dim != expected['head_dim'] or
                (hidden_dim is not None and hidden_dim != expected['hidden_dim'])):
            raise ValueError('JEV is already installed with another dictionary/config')
        return dict(expected)
    if 'jev_head' in dict(model.named_children()):
        raise ValueError('Model already contains an unrecognized jev_head')
    norm, path, width = _norm_and_width(adapter, hidden_dim)
    metadata = dict(protocol=PROTOCOL, feature_protocol=FEATURE_PROTOCOL, family=adapter.family,
                    classes=classes, classes_sha256=classes_identity(classes), hidden_dim=width,
                    head_dim=head_dim, norm_path=path, feature='last-input-prefix-token',
                    task='categorical', supports_multilabel=False, jev_private_algorithm_reproduced=False,
                    head_precision_protocol=HEAD_PRECISION_PROTOCOL, code_sha256=_file_sha(__file__))
    head = JevHead(width, classes, head_dim).to(device=next(model.parameters()).device, dtype=torch.float32)
    model.add_module('jev_head', head)
    capture = _ForwardCapture(model, norm, metadata)
    original = model.forward

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        ends = kwargs.pop('jev_prefix_end', None)
        native_inputs = dict(kwargs)
        if args:
            native_inputs.setdefault('input_ids', args[0])
        record = None if ends is None else _request(native_inputs, ends)
        stack = capture.stack()
        stack.append(record)
        try:
            result = original(*args, **kwargs)
            if record is None:
                return result
            if record['features'] is None:
                raise ValueError('Native decoder final norm was not executed during JEV forward')
            logits = self.jev_head(record['features'])
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite JEV logits')
            if isinstance(result, MutableMapping):
                result['jev_logits'] = logits
            elif hasattr(result, '__dict__'):
                result.jev_logits = logits
            else:
                raise ValueError('JEV requires native return_dict/ModelOutput, not tuple output')
            return result
        finally:
            stack.pop()

    model.forward = types.MethodType(forward, model)
    # Keep controller out of nn.Module's module tree and checkpoint tensors.
    model._planner_jev_capture = capture
    return dict(metadata)


def get_jev_logits(output):
    value = output.get('jev_logits') if isinstance(output, Mapping) else getattr(output, 'jev_logits', None)
    if not isinstance(value, torch.Tensor):
        raise ValueError('Native forward did not return JEV logits')
    return value


def _distribution(label, classes):
    if ('genre' in label) == ('genre_distribution' in label):
        raise ValueError('Specify exactly one hard genre or soft categorical genre_distribution')
    if 'genre' in label:
        if label['genre'] not in classes:
            raise ValueError('Unknown hard genre outside the frozen class dictionary')
        return [float(name == label['genre']) for name in classes]
    distribution = label['genre_distribution']
    if (not isinstance(distribution, dict) or not distribution or
            any(name not in classes for name in distribution) or
            any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in distribution.values())):
        raise ValueError('Soft categorical labels require finite probabilities for known classes')
    target = [float(distribution.get(name, 0.)) for name in classes]
    if not math.isclose(sum(target), 1., rel_tol=0., abs_tol=1e-6):
        raise ValueError('Categorical probabilities must sum to one; multi-label targets are unsupported')
    return target


def join_jev_labels(rows, path, classes, split=None):
    """Join independent labels; an absent manifest/row is unknown, never negative."""
    classes = validate_classes(classes)
    index = {}
    for row in rows:
        sid = row.get('id')
        if not isinstance(sid, str) or not sid or sid in index:
            raise ValueError('JEV source rows need unique explicit ids')
        if (any(not isinstance(row.get(key), str) or not row[key] for key in ('source_group_id', 'audio_sha256', 'split')) or
                len(row['audio_sha256']) != 64 or any(ch not in '0123456789abcdef' for ch in row['audio_sha256']) or
                (split is not None and row['split'] != split)):
            raise ValueError('JEV source group/audio SHA/split identity is invalid')
        index[sid] = row
    labels = {}
    if path is not None:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            label = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(label, dict) or label.get('id') not in index or label['id'] in labels:
                raise ValueError('Duplicate genre label or label without a matching source')
            row = index[label['id']]
            for key in ('source_group_id', 'audio_sha256', 'split'):
                if not isinstance(label.get(key), str) or not label[key] or label[key] != row.get(key):
                    raise ValueError('Genre label/source identity differs: ' + key)
            if split is not None and label['split'] != split:
                raise ValueError('Genre label belongs to another split')
            if any(not isinstance(label.get(key), str) or not label[key].strip()
                   for key in ('label_source', 'label_version')):
                raise ValueError('Independent genre labels require label_source and label_version')
            labels[label['id']] = _distribution(label, classes)
    joined = [dict(row, jev_target=labels.get(row['id'])) for row in rows]
    return joined, dict(protocol=PROTOCOL, task='categorical', supports_multilabel=False,
        classes_sha256=classes_identity(classes), known_labels=len(labels), unknown_labels=len(rows) - len(labels),
        source_label_manifest=None if path is None else dict(path=str(Path(path).resolve()), sha256=_file_sha(path)),
        labels_guessed_from_captions=False, semantic_label_quality_verified=False)


def batch_jev_targets(rows, classes, device=None):
    classes = validate_classes(classes)
    targets = torch.zeros(len(rows), len(classes), dtype=torch.float32, device=device)
    mask = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        value = row.get('jev_target')
        if value is not None:
            target = torch.as_tensor(value, dtype=torch.float32, device=device)
            if (target.shape != (len(classes),) or not torch.isfinite(target).all() or
                    (target < 0).any() or not torch.isclose(target.sum(), target.new_tensor(1.), atol=1e-5, rtol=0)):
                raise ValueError('jev_target must be a categorical distribution, or None for unknown')
            targets[i] = target
            mask[i] = True
    return targets, mask


def jev_loss(logits, targets, label_mask=None):
    """Return source-summed categorical CE and count, for global normalization."""
    if logits.ndim != 2 or logits.shape[-1] < 2 or not torch.isfinite(logits).all():
        raise ValueError('JEV logits must be finite [B,C]')
    if targets.ndim == 1:
        if targets.dtype != torch.long or targets.shape != logits.shape[:1]:
            raise ValueError('Hard JEV labels must be int64 [B], -100 for unknown')
        inferred = targets.ne(-100)
        if label_mask is None:
            label_mask = inferred
        elif not torch.equal(label_mask, inferred):
            raise ValueError('Hard label mask must agree with -100 unknown labels')
    elif targets.shape != logits.shape:
        raise ValueError('Soft JEV targets must have shape [B,C]')
    if label_mask is None:
        label_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    if label_mask.dtype != torch.bool or label_mask.shape != logits.shape[:1]:
        raise ValueError('JEV label mask must be boolean [B]')
    label_mask = label_mask.to(logits.device)
    targets = targets.to(logits.device)
    count = label_mask.sum()
    if not bool(count):
        # Keep every head parameter in DDP/ZeRO's graph even on unlabeled ranks.
        return logits.float().sum() * 0., count
    if targets.ndim == 1:
        if ((targets[label_mask] < 0) | (targets[label_mask] >= logits.shape[-1])).any():
            raise ValueError('Hard JEV target outside dictionary')
        total = F.cross_entropy(logits[label_mask].float(), targets[label_mask], reduction='sum')
    else:
        chosen = targets[label_mask]
        if (not chosen.is_floating_point() or not torch.isfinite(chosen).all() or (chosen < 0).any() or
                not torch.allclose(chosen.sum(-1), torch.ones_like(chosen.sum(-1)), atol=1e-5, rtol=0)):
            raise ValueError('Soft JEV targets must be categorical distributions summing to one')
        total = -(chosen * F.log_softmax(logits[label_mask].float(), -1)).sum()
    return total, count


@torch.no_grad()
def predict_jev(adapter, row):
    """Input-only inference through the same head path; never read target/genre."""
    capture = vars(adapter.model).get('_planner_jev_capture')
    if capture is None:
        raise ValueError('Install or load a JEV head before prediction')
    source = {key: row[key] for key in ('id', 'source_group_id', 'audio_path', 'audio_sha256', 'audio_window_seconds')}
    if 'audio_duration_seconds' in row:
        source['audio_duration_seconds'] = row['audio_duration_seconds']
    batch = adapter.prefix(source)
    if 'labels' in batch or any('pixel_values' in name or 'grid_thw' in name for name in batch):
        raise ValueError('JEV prediction requires the native audio-only prefix')
    mask = batch.get('attention_mask')
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
        raise ValueError('Native prefix needs a token attention mask')
    positions = torch.arange(mask.shape[1], device=mask.device)[None] + 1
    ends = positions.masked_fill(~mask.bool(), 0).max(-1).values.long()
    batch['jev_prefix_end'] = ends
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        parameter = next(adapter.model.parameters())
        with torch.autocast(parameter.device.type, dtype=torch.bfloat16, enabled=parameter.dtype == torch.bfloat16):
            output = adapter.model(**adapter._move(batch), return_dict=True, use_cache=False)
        logits = get_jev_logits(output).detach().float().cpu()
    finally:
        adapter.model.train(was_training)
    return dict(logits=logits, probabilities=logits.softmax(-1), classes=list(capture.metadata['classes']),
                feature_protocol=FEATURE_PROTOCOL, prefix_end=ends.cpu(), target_consumed=False,
                calibrated=False, probability_is_not_verified_genre_quality=True)


def export_jev(adapter, path, identity=None, state_dict=None):
    """Export the registered FP32 head; pass gathered portable state for ZeRO3."""
    capture = vars(adapter.model).get('_planner_jev_capture')
    if capture is None:
        raise ValueError('No installed JEV head to export')
    _object_sha(identity)  # Reject non-JSON identities and nonfinite numbers.
    expected = adapter.model.jev_head.state_dict()
    if state_dict is None:
        if any(hasattr(p, 'ds_id') for p in adapter.model.jev_head.parameters()):
            raise ValueError('ZeRO3 head export needs the gathered portable state_dict')
        saved = expected
    else:
        saved = {name[len('jev_head.'):]: value for name, value in state_dict.items() if name.startswith('jev_head.')}
    if set(saved) != set(expected):
        raise ValueError('Portable state must contain all and only registered jev_head keys')
    result = {}
    for name, value in saved.items():
        expected_shape = tuple(getattr(expected[name], 'ds_shape', expected[name].shape))
        # state_dict() detaches Parameters and may lose ds_shape; named params retain it.
        parameter = dict(adapter.model.jev_head.named_parameters()).get(name)
        if parameter is not None:
            expected_shape = tuple(getattr(parameter, 'ds_shape', parameter.shape))
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape or
                value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('Invalid portable FP32 JEV head tensor: ' + name)
        result[name] = value.detach().cpu().clone()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    torch.save(result, path / 'head.pt')
    metadata = dict(format=HEAD_FORMAT, installation=capture.metadata, identity=identity,
                    head_sha256=_file_sha(path / 'head.pt'), calibrated=False)
    (path / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return metadata


def load_jev(adapter, path, expected_identity=None, trainable=False):
    path = Path(path)
    metadata = _read_json(path / 'metadata.json')
    if (metadata.get('format') != HEAD_FORMAT or metadata.get('calibrated') is not False or
            metadata.get('head_sha256') != _file_sha(path / 'head.pt')):
        raise ValueError('JEV portable format/content identity differs')
    if expected_identity is not None and metadata.get('identity') != expected_identity:
        raise ValueError('JEV checkpoint/model identity differs')
    spec = metadata['installation']
    actual = install_jev(adapter, spec['classes'], hidden_dim=spec['hidden_dim'], head_dim=spec['head_dim'])
    if actual != spec:
        raise ValueError('JEV dictionary/family/feature protocol differs')
    saved = torch.load(path / 'head.pt', map_location='cpu', weights_only=True)
    expected = adapter.model.jev_head.state_dict()
    if set(saved) != set(expected) or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
            value.shape != expected[name].shape or not torch.isfinite(value).all() for name, value in saved.items()):
        raise ValueError('JEV head keys/shapes/finite FP32 values differ')
    adapter.model.jev_head.load_state_dict(saved, strict=True)
    adapter.model.jev_head.requires_grad_(trainable)
    adapter.model.jev_head.train(adapter.model.training if trainable else False)
    return metadata
