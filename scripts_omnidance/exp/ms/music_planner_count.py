"""Window-count auxiliary head on the native audio prefix.

Predicts log1p(number of caption windows) from the same last-input-prefix
feature used by JEV.  The loss is a small-weight SmoothL1 term; at inference
the predicted count can drive coverage-constrained decoding.
"""
import functools
import json
import math
import re
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from . import music_planner_jev as jev


PROTOCOL = 'planner-window-count-prefix-regression-v1'
HEAD_FORMAT = 'planner-window-count-portable-head-v1'
WINDOW_RE = re.compile(r'\[\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*s\s*\]')


class CountHead(nn.Module):
    def __init__(self, hidden_dim, head_dim=128):
        super().__init__()
        if type(hidden_dim) is not int or type(head_dim) is not int or min(hidden_dim, head_dim) < 1:
            raise ValueError('Positive integer count hidden/head dimensions required')
        self.network = nn.Sequential(nn.LayerNorm(hidden_dim), jev.FP32DecisionLinear(hidden_dim, head_dim),
                                     nn.GELU(), jev.FP32DecisionLinear(head_dim, 2))
        self.float()

    def forward(self, features):
        """Return [B, 2]: log1p(window_count) and last-window end time (seconds)."""
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.network(features.float())


def window_count(text):
    if not isinstance(text, str):
        return 1
    spans = WINDOW_RE.findall(text)
    return max(1, len(spans))


def last_end(text):
    spans = WINDOW_RE.findall(text or '')
    return float(spans[-1][1]) if spans else 0.0


def batch_count_targets(rows, device=None):
    values = torch.tensor([[math.log1p(window_count(row.get('target_text', ''))),
                            last_end(row.get('target_text', ''))] for row in rows],
                          dtype=torch.float32, device=device)
    mask = torch.ones(len(rows), dtype=torch.bool, device=device)
    return values, mask


def count_loss(pred, targets, mask=None, end_weight=0.5):
    """Summed SmoothL1 over count (log1p) and last-window end (seconds)."""
    if pred.ndim != 2 or pred.shape[1] != 2 or not torch.isfinite(pred).all():
        raise ValueError('Window-structure predictions must be finite [B,2]')
    if targets.shape != pred.shape:
        raise ValueError('Window-structure targets must align with predictions')
    if mask is None:
        mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    mask = mask.to(pred.device).bool()
    targets = targets.to(pred.device)
    count = mask.sum()
    if not bool(count):
        return pred.float().sum() * 0., count
    selected = pred[mask].float()
    truth = targets[mask]
    count_term = F.smooth_l1_loss(selected[:, 0], truth[:, 0], reduction='sum', beta=0.1)
    end_term = F.smooth_l1_loss(selected[:, 1] / 5.0625, truth[:, 1] / 5.0625, reduction='sum', beta=0.1)
    return count_term + end_weight * end_term, count


def install_count(adapter, hidden_dim=None, head_dim=128):
    model = adapter.model
    if hasattr(model, 'module') or hasattr(model, 'optimizer'):
        raise ValueError('Install window-count head before distributed wrapping')
    existing = vars(model).get('_planner_count_capture')
    if existing is not None:
        return dict(existing.metadata)
    if 'count_head' in dict(model.named_children()):
        raise ValueError('Model already contains a count_head')
    norm, path, width = jev._norm_and_width(adapter, hidden_dim)
    metadata = dict(protocol=PROTOCOL, feature_protocol=jev.FEATURE_PROTOCOL, family=adapter.family,
                    hidden_dim=width, head_dim=head_dim, norm_path=path, head_prefix='count_head',
                    feature='last-input-prefix-token', task='regression',
                    target='log1p(window_count)+last_window_end_seconds')
    head = CountHead(width, head_dim).to(device=next(model.parameters()).device, dtype=torch.float32)
    model.add_module('count_head', head)
    capture = jev._ForwardCapture(model, norm, metadata)
    original = model.forward

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        ends = kwargs.pop('count_prefix_end', None)
        native_inputs = dict(kwargs)
        if args:
            native_inputs.setdefault('input_ids', args[0])
        record = None if ends is None else jev._request(native_inputs, ends)
        stack = capture.stack()
        stack.append(record)
        try:
            result = original(*args, **kwargs)
            if record is None:
                return result
            if record['features'] is None:
                raise ValueError('Native decoder final norm was not executed during count forward')
            head_device = self.count_head.network[0].weight.device
            pred = self.count_head(record['features'].to(head_device))
            if isinstance(result, dict):
                result['count_pred'] = pred
            elif hasattr(result, '__dict__'):
                result.count_pred = pred
            else:
                raise ValueError('Window-count head requires native return_dict/ModelOutput output')
            return result
        finally:
            stack.pop()

    model.forward = types.MethodType(forward, model)
    model._planner_count_capture = capture
    return dict(metadata)


def get_count_pred(output):
    value = None
    if isinstance(output, dict):
        value = output.get('count_pred')
    elif hasattr(output, 'count_pred'):
        value = output.count_pred
    if not isinstance(value, torch.Tensor):
        raise ValueError('Native forward did not return window-count predictions')
    return value


@torch.no_grad()
def predict_count(adapter, row):
    capture = vars(adapter.model).get('_planner_count_capture')
    if capture is None:
        raise ValueError('Install or load a window-count head before prediction')
    source = {key: row[key] for key in ('id', 'source_group_id', 'audio_path', 'audio_sha256', 'audio_window_seconds')}
    if 'audio_duration_seconds' in row:
        source['audio_duration_seconds'] = row['audio_duration_seconds']
    batch = adapter.prefix(source)
    mask = batch.get('attention_mask')
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
        raise ValueError('Native prefix needs a token attention mask')
    positions = torch.arange(mask.shape[1], device=mask.device)[None] + 1
    ends = positions.masked_fill(~mask.bool(), 0).max(-1).values.long()
    batch['count_prefix_end'] = ends
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        parameter = next(adapter.model.parameters())
        with torch.autocast(parameter.device.type, dtype=torch.bfloat16, enabled=parameter.dtype == torch.bfloat16):
            output = adapter.model(**adapter._move(batch), return_dict=True, use_cache=False)
        pred = get_count_pred(output).detach().float().cpu()
    finally:
        adapter.model.train(was_training)
    values = pred.tolist()
    return dict(log1p_count=[v[0] for v in values], count=[max(1, round(math.expm1(v[0]))) for v in values],
                last_end=[max(0., v[1]) for v in values])


def export_count(adapter, path, identity=None, state_dict=None):
    capture = vars(adapter.model).get('_planner_count_capture')
    if capture is None:
        raise ValueError('No installed window-count head to export')
    expected = adapter.model.count_head.state_dict()
    if state_dict is None:
        saved = expected
    else:
        saved = {name[len('count_head.'):]: value for name, value in state_dict.items()
                 if name.startswith('count_head.')}
    if set(saved) != set(expected):
        raise ValueError('Portable state must contain all and only count_head keys')
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.detach().float().cpu() for name, value in saved.items()}, path / 'count_head.pt')
    (path / 'count.json').write_text(json.dumps(dict(format=HEAD_FORMAT, identity=identity,
        metadata=capture.metadata), ensure_ascii=False, indent=2) + '\n')
    return dict(path=str(path), keys=sorted(saved))


def load_count(adapter, path):
    path = Path(path)
    metadata = json.loads((path / 'count.json').read_text(encoding='utf-8'))
    install = install_count(adapter, hidden_dim=metadata['metadata'].get('hidden_dim'),
                            head_dim=metadata['metadata']['head_dim'])
    state = torch.load(path / 'count_head.pt', map_location='cpu', weights_only=True)
    adapter.model.count_head.load_state_dict(state, strict=True)
    return dict(installation=install, state=state)
