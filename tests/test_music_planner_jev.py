"""CPU protocol tests; tiny native-shaped models, not pretrained model approval."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from scripts_omnidance.exp.ms import music_planner_jev as jev


CLASSES = ['hip-hop', '手势舞']


class TinyDecoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width)
        self.proj = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.checkpointing = False

    def forward(self, values):
        def causal(x):
            return torch.tanh(self.proj(x.cumsum(1)))
        values = checkpoint(causal, values, use_reentrant=True) if self.checkpointing else causal(values)
        return self.norm(values)


class TinyNative(nn.Module):
    def __init__(self, family, mapping_output=False):
        super().__init__()
        self.embed = nn.Embedding(31, 8)
        self.lm_head = nn.Linear(8, 31)
        config = SimpleNamespace(hidden_size=8)
        self.config = SimpleNamespace(**{'language_config' if family == 'moss' else 'text_config': config})
        self.add_module('language_model' if family == 'moss' else 'model', TinyDecoder(8))
        self.family, self.mapping_output = family, mapping_output
        self.native_calls, self.fail = 0, False

    @property
    def decoder(self):
        return self.language_model if self.family == 'moss' else self.model

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True, use_cache=False):
        self.native_calls += 1
        hidden = self.decoder(self.embed(input_ids))
        if self.fail:
            raise RuntimeError('deliberate native failure after norm')
        logits = self.lm_head(hidden)
        if not return_dict:
            return (logits,)
        return {'logits': logits} if self.mapping_output else SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask=None):
        # Native generation calls forward without any JEV metadata.
        output = self.forward(input_ids, attention_mask)
        return output['logits'] if self.mapping_output else output.logits


class TinyPeft(nn.Module):
    """Only the structural wrapper behavior needed by install_jev is mocked."""
    def __init__(self, native):
        super().__init__()
        self.base_model = native

    def get_base_model(self):
        return self.base_model

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


class TinyAdapter:
    def __init__(self, family='moss', mapping_output=False, peft=False):
        self.family = family
        self.model = TinyNative(family, mapping_output)
        self.native = self.model
        if peft:
            self.model = TinyPeft(self.native)
        self.seen = None

    def prefix(self, row):
        assert set(row) - {'audio_duration_seconds'} == {'id', 'source_group_id', 'audio_path', 'audio_sha256', 'audio_window_seconds'}
        self.seen = dict(row)
        return batch([1, 2, 3, 4], prefix=None)

    def _move(self, values):
        device = next(self.model.parameters()).device
        return {name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in values.items() if name != 'labels'}


def batch(tokens, prefix=4):
    values = dict(input_ids=torch.tensor([tokens]), attention_mask=torch.ones(1, len(tokens), dtype=torch.long))
    if prefix is not None:
        values['jev_prefix_end'] = torch.tensor([prefix], dtype=torch.long)
    return values


def source(sid='a', split='train'):
    return dict(id=sid, source_group_id='song-' + sid, audio_path='unused.wav',
                audio_sha256=sid * 64, audio_window_seconds=[0., 5.0625], split=split)


def label(row, **target):
    return dict({name: row[name] for name in ('id', 'source_group_id', 'audio_sha256', 'split')},
                label_source='independent-video-style-review', label_version='v1', **target)


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    return path


@pytest.mark.parametrize('family', ['moss', 'omni_captioner'])
@pytest.mark.parametrize('peft', [False, True])
def test_install_native_shape_head_registration_and_gradients(family, peft):
    adapter = TinyAdapter(family, peft=peft)
    metadata = jev.install_jev(adapter, CLASSES, head_dim=5)
    assert metadata['norm_path'] == ('language_model.norm' if family == 'moss' else 'model.norm')
    assert metadata['supports_multilabel'] is False
    assert 'jev_head' in dict(adapter.model.named_children())
    assert any(name.startswith('jev_head.') for name, _ in adapter.model.named_parameters())
    output = adapter.model(**batch([1, 2, 3, 4, 5, 6]))
    logits = jev.get_jev_logits(output)
    assert logits.shape == (1, 2) and logits.dtype == torch.float32
    loss, count = jev.jev_loss(logits, torch.tensor([1]))
    loss.backward()
    assert count == 1
    assert adapter.native.decoder.proj.weight.grad.norm() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.model.jev_head.parameters())


@pytest.mark.parametrize('family', ['moss', 'omni_captioner'])
def test_answer_changes_cannot_change_input_prefix_decision(family):
    adapter = TinyAdapter(family)
    jev.install_jev(adapter, CLASSES, head_dim=4)
    adapter.model.eval()
    first = jev.get_jev_logits(adapter.model(**batch([1, 2, 3, 4, 5, 6])))
    second = jev.get_jev_logits(adapter.model(**batch([1, 2, 3, 4, 29, 30, 28])))
    prefix = jev.get_jev_logits(adapter.model(**batch([1, 2, 3, 4])))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, prefix, rtol=0, atol=0)


def test_no_metadata_native_generation_does_not_call_head():
    adapter = TinyAdapter()
    expected = adapter.model.generate(**batch([1, 2, 3, 4], None))
    jev.install_jev(adapter, CLASSES, head_dim=4)
    calls = []
    handle = adapter.model.jev_head.register_forward_hook(lambda *unused: calls.append(1))
    actual = adapter.model.generate(**batch([1, 2, 3, 4], None))
    handle.remove()
    assert not calls
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mapping_model_output_supported():
    adapter = TinyAdapter(mapping_output=True)
    jev.install_jev(adapter, CLASSES, head_dim=4)
    output = adapter.model(**batch([1, 2, 3, 4]))
    assert isinstance(output, dict) and output['jev_logits'].shape == (1, 2)


def test_checkpoint_backward_replay_and_failed_forward_leave_no_stale_hook():
    adapter = TinyAdapter()
    adapter.native.decoder.checkpointing = True
    jev.install_jev(adapter, CLASSES, head_dim=4)
    logits = jev.get_jev_logits(adapter.model(**batch([1, 2, 3, 4, 5])))
    logits.square().sum().backward()
    assert adapter.native.decoder.proj.weight.grad.norm() > 0
    capture = vars(adapter.model)['_planner_jev_capture']
    assert capture.stack() == []
    # A norm invocation with no active forward request must not save a graph.
    adapter.native.decoder.norm(torch.randn(1, 3, 8))
    adapter.native.fail = True
    with pytest.raises(RuntimeError, match='deliberate'):
        adapter.model(**batch([1, 2, 3, 4]))
    assert capture.stack() == []
    adapter.native.fail = False
    assert jev.get_jev_logits(adapter.model(**batch([2, 3, 4, 5]))).shape == (1, 2)


def test_fp32_head_inside_native_bfloat16_autocast():
    adapter = TinyAdapter()
    jev.install_jev(adapter, CLASSES, head_dim=4)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits = jev.get_jev_logits(adapter.model(**batch([1, 2, 3, 4])))
    assert logits.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in adapter.model.jev_head.parameters())


def test_default_zero3_amp_registry_preserves_fp32_head_and_native_policy():
    adapter = TinyAdapter()
    metadata = jev.install_jev(adapter, CLASSES, head_dim=4)
    # Faithful CPU exercise of the documented DeepSpeed exact-class marking,
    # not an assertion that a real ZeRO3 engine ran in this test environment.
    # deepspeed/runtime/torch_autocast.py:49-52 (default registry at lines13-18).
    safe_classes = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]
    for module in adapter.model.modules():
        if module.__class__ in safe_classes:
            for parameter in module.parameters(recurse=False):
                parameter.comm_dtype = torch.bfloat16
    assert all(getattr(p, 'comm_dtype', None) is None for p in adapter.model.jev_head.parameters())
    assert adapter.native.decoder.proj.weight.comm_dtype == torch.bfloat16
    assert adapter.native.lm_head.weight.comm_dtype == torch.bfloat16
    assert all(type(layer) is jev.FP32DecisionLinear for layer in adapter.model.jev_head.modules()
               if isinstance(layer, nn.Linear))
    assert metadata['head_precision_protocol'] == jev.HEAD_PRECISION_PROTOCOL
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = adapter.model(**batch([1, 2, 3, 4, 5]))
        logits = jev.get_jev_logits(output)
        loss, _ = jev.jev_loss(logits, torch.tensor([1]))
    loss.backward()
    assert logits.dtype == torch.float32
    assert all(p.dtype == torch.float32 and p.grad is not None and p.grad.dtype == torch.float32 and
               torch.isfinite(p.grad).all() for p in adapter.model.jev_head.parameters())
    assert adapter.native.decoder.proj.weight.grad.norm() > 0


@pytest.mark.parametrize('ends', [torch.tensor([0]), torch.tensor([7]), torch.tensor([4.]), torch.tensor([[4]])])
def test_invalid_prefix_boundaries_rejected(ends):
    adapter = TinyAdapter()
    jev.install_jev(adapter, CLASSES, head_dim=4)
    values = batch([1, 2, 3, 4])
    values['jev_prefix_end'] = ends
    with pytest.raises(ValueError, match='prefix_end'):
        adapter.model(**values)


def test_answer_label_and_padding_cannot_be_selected_as_prefix_end():
    adapter = TinyAdapter()
    jev.install_jev(adapter, CLASSES, head_dim=4)
    values = batch([1, 2, 3, 4, 5], 5)
    values['labels'] = torch.tensor([[-100, -100, -100, -100, 5]])
    with pytest.raises(ValueError, match='answer labels'):
        adapter.model(**values)
    values = batch([1, 2, 3, 4])
    values['attention_mask'][0, -1] = 0
    with pytest.raises(ValueError, match='unpadded'):
        adapter.model(**values)


def test_actual_per_row_exclusive_prefix_end():
    adapter = TinyAdapter()
    jev.install_jev(adapter, CLASSES, head_dim=4)
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 8]])
    result = jev.get_jev_logits(adapter.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                             jev_prefix_end=torch.tensor([4, 2])))
    one = jev.get_jev_logits(adapter.model(**batch([1, 2, 3, 4])))
    two = jev.get_jev_logits(adapter.model(**batch([1, 2], 2)))
    torch.testing.assert_close(result[0], one[0])
    torch.testing.assert_close(result[1], two[0])


def test_categorical_hard_soft_masked_and_unknown_zero_graph():
    logits = torch.tensor([[.1, .8], [2., -1.], [1., 1.]], requires_grad=True)
    targets = torch.tensor([[0., 1.], [.3, .7], [float('nan'), float('nan')]])
    loss, count = jev.jev_loss(logits, targets, torch.tensor([True, True, False]))
    expected = -(targets[:2] * logits[:2].log_softmax(-1)).sum()
    torch.testing.assert_close(loss, expected)
    assert count == 2
    hard, hard_count = jev.jev_loss(logits, torch.tensor([1, 0, -100]))
    assert hard_count == 2 and torch.isfinite(hard)
    zero, count = jev.jev_loss(logits, torch.zeros_like(logits), torch.zeros(3, dtype=torch.bool))
    zero.backward()
    assert count == 0 and logits.grad is not None and logits.grad.eq(0).all()
    with pytest.raises(ValueError, match='summing to one'):
        jev.jev_loss(logits, torch.ones_like(logits))


def test_missing_labels_keep_all_head_parameters_in_zero_graph():
    adapter = TinyAdapter()
    jev.install_jev(adapter, CLASSES, head_dim=4)
    logits = jev.get_jev_logits(adapter.model(**batch([1, 2, 3, 4])))
    total, count = jev.jev_loss(logits, torch.zeros_like(logits), torch.tensor([False]))
    total.backward()
    assert count == 0
    assert all(p.grad is not None and p.grad.eq(0).all() for p in adapter.model.jev_head.parameters())


def test_manifest_hard_soft_unknown_source_identity(tmp_path):
    rows = [source(sid) for sid in ('a', 'b', 'c')]
    path = jsonl(tmp_path / 'labels.jsonl', [label(rows[0], genre='手势舞'),
        label(rows[1], genre_distribution={'hip-hop': .25, '手势舞': .75})])
    joined, metadata = jev.join_jev_labels(rows, path, CLASSES, 'train')
    assert metadata['known_labels'] == 2 and metadata['unknown_labels'] == 1
    targets, mask = jev.batch_jev_targets(joined, CLASSES)
    torch.testing.assert_close(targets, torch.tensor([[0., 1.], [.25, .75], [0., 0.]]))
    assert mask.tolist() == [True, True, False]
    no_labels, report = jev.join_jev_labels(rows, None, CLASSES, 'train')
    assert report['known_labels'] == 0 and all(row['jev_target'] is None for row in no_labels)
    assert 'jev_target' not in rows[0]


@pytest.mark.parametrize('changes', [dict(source_group_id='wrong'), dict(audio_sha256='f' * 64),
    dict(split='test'), dict(label_source=''), dict(genre='unknown'),
    dict(genre_distribution={'hip-hop': .5, 'unknown': .5}),
    dict(genre_distribution={'hip-hop': 1., '手势舞': 1.}), dict(genre_distribution={'hip-hop': float('nan')})])
def test_manifest_rejects_bad_labels_and_source_leakage(tmp_path, changes):
    row = source()
    target = label(row, genre='hip-hop')
    if 'genre_distribution' in changes:
        target.pop('genre')
    target.update(changes)
    path = jsonl(tmp_path / 'labels.jsonl', [target])
    with pytest.raises(ValueError):
        jev.join_jev_labels([row], path, CLASSES, 'train')


def test_classes_order_identity_and_no_fictitious_default(tmp_path):
    path = tmp_path / 'classes.json'
    path.write_text(json.dumps(CLASSES), encoding='utf-8')
    assert jev.read_classes(path) == CLASSES
    assert jev.classes_identity(CLASSES) != jev.classes_identity(list(reversed(CLASSES)))
    for classes in ([], ['one'], ['one', 'one'], ['one', ''], ['one', ' two '], {'one': 1, 'two': 2}):
        with pytest.raises(ValueError):
            jev.validate_classes(classes)


@pytest.mark.parametrize('family', ['moss', 'omni_captioner'])
def test_prefix_only_prediction_never_consumes_gold(family):
    adapter = TinyAdapter(family)
    jev.install_jev(adapter, CLASSES, head_dim=4)
    adapter.model.train()
    row = dict(source(), target_text='SECRET answer', genre='SECRET genre', image_path='SECRET.png')
    prediction = jev.predict_jev(adapter, row)
    assert adapter.model.training
    assert prediction['target_consumed'] is False and prediction['calibrated'] is False
    assert prediction['prefix_end'].tolist() == [4]
    torch.testing.assert_close(prediction['probabilities'].sum(-1), torch.ones(1))
    assert not any('SECRET' in str(value) for value in adapter.seen.values())


def test_prefix_only_prediction_preserves_variable_audio_duration():
    adapter = TinyAdapter()
    jev.install_jev(adapter, CLASSES, head_dim=4)
    row = dict(source(), audio_duration_seconds=9.25, audio_window_seconds=[0., 9.25], target_text='SECRET')
    jev.predict_jev(adapter, row)
    assert adapter.seen['audio_duration_seconds'] == 9.25
    assert adapter.seen['audio_window_seconds'] == [0., 9.25]
    assert 'target_text' not in adapter.seen


def test_export_reload_portable_state_identity_and_zeRO_guard(tmp_path):
    adapter = TinyAdapter(peft=True)
    jev.install_jev(adapter, CLASSES, head_dim=4)
    expected = jev.get_jev_logits(adapter.model(**batch([1, 2, 3, 4]))).detach()
    portable = {name: value.detach().clone() for name, value in adapter.model.named_parameters()}
    identity = dict(model_sha256='a' * 64, checkpoint_sha256='b' * 64)
    metadata = jev.export_jev(adapter, tmp_path / 'head', identity, state_dict=portable)
    # Use equal backbone weights but a freshly initialized head before loading.
    other = TinyAdapter(peft=True)
    other.model.load_state_dict({k: v for k, v in adapter.model.state_dict().items() if not k.startswith('jev_head.')})
    jev.load_jev(other, tmp_path / 'head', expected_identity=identity)
    actual = jev.get_jev_logits(other.model(**batch([1, 2, 3, 4]))).detach()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not any(p.requires_grad for p in other.model.jev_head.parameters())
    with pytest.raises(ValueError, match='identity'):
        jev.load_jev(other, tmp_path / 'head', expected_identity=dict(model_sha256='changed'))
    with pytest.raises(FileExistsError):
        jev.export_jev(adapter, tmp_path / 'head')
    for parameter in adapter.model.jev_head.parameters():
        parameter.ds_id = 1
        parameter.ds_shape = parameter.shape
    with pytest.raises(ValueError, match='gathered'):
        jev.export_jev(adapter, tmp_path / 'not-gathered')
    jev.export_jev(adapter, tmp_path / 'gathered', identity, state_dict=portable)
    assert metadata['installation']['classes'] == CLASSES


def test_export_rejects_missing_or_nonfinite_head_and_load_rejects_tamper(tmp_path):
    adapter = TinyAdapter()
    jev.install_jev(adapter, CLASSES, head_dim=4)
    portable = {name: value.detach().clone() for name, value in adapter.model.named_parameters()}
    key = next(name for name in portable if name.startswith('jev_head.'))
    partial = dict(portable)
    partial.pop(key)
    with pytest.raises(ValueError, match='all and only'):
        jev.export_jev(adapter, tmp_path / 'partial', state_dict=partial)
    portable[key].flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='FP32'):
        jev.export_jev(adapter, tmp_path / 'nan', state_dict=portable)
    jev.export_jev(adapter, tmp_path / 'good')
    (tmp_path / 'good/head.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='content identity'):
        jev.load_jev(TinyAdapter(), tmp_path / 'good')


def test_family_dictionary_hidden_mismatch_and_duplicate_install_rejected():
    adapter = TinyAdapter()
    metadata = jev.install_jev(adapter, CLASSES, head_dim=4)
    assert jev.install_jev(adapter, CLASSES, head_dim=4) == metadata
    with pytest.raises(ValueError, match='another dictionary'):
        jev.install_jev(adapter, list(reversed(CLASSES)), head_dim=4)
    with pytest.raises(ValueError, match='width'):
        jev.install_jev(TinyAdapter(), CLASSES, hidden_dim=123, head_dim=4)
    adapter = TinyAdapter()
    adapter.family = 'music_flamingo'
    with pytest.raises(ValueError, match='supports'):
        jev.install_jev(adapter, CLASSES, head_dim=4)
