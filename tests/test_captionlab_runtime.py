"""Tiny CPU tensor tests, never a pretrained-model or GPU acceptance claim."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as functional

from captionlab.runtime import score_target


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(13)
            self.embedding = nn.Embedding(16, 5)
            self.dropout = nn.Dropout(.8)
            self.head = nn.Linear(5, 16)
        self.calls = []
        self.corrupt = None

    def forward(self, input_ids, attention_mask, use_cache):
        self.calls.append(dict(use_cache=use_cache, grad=torch.is_grad_enabled(),
                               training=self.training, ids=input_ids.detach().clone()))
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        logits = self.head(self.dropout(hidden.cumsum(dim=1)))
        if self.corrupt == "nan":
            logits[:, -2, 0] = float("nan")
        elif self.corrupt == "unused_prefix_nan":
            logits[:, 0, 0] = float("nan")
        elif self.corrupt == "short":
            logits = logits[:, :-1]
        return SimpleNamespace(logits=logits)

    def generate(self, *args, **kwargs):
        raise AssertionError("Scoring must never generate")


class TinyAdapter:
    def __init__(self):
        self.model = TinyCausalModel()
        self.max_length = 64
        self.eos = [9, 10]
        self.batch_mutator = None
        self.move_mutator = None
        self.seen_rows = []

    def eos_token_ids(self):
        return self.eos

    def collate(self, rows):
        assert len(rows) == 1
        self.seen_rows.append(copy.deepcopy(rows[0]))
        rows[0]["collator_annotation"] = "must not escape to caller"
        ids = torch.tensor([[1, 4, 2, 5, 7, 9]], dtype=torch.long)
        batch = dict(input_ids=ids, labels=torch.tensor([[-100, -100, 2, 5, 7, 9]]),
                     attention_mask=torch.ones_like(ids))
        if self.batch_mutator:
            self.batch_mutator(batch)
        return batch

    def _move(self, batch):
        result = {key: value.to("cpu") if isinstance(value, torch.Tensor) else value
                  for key, value in batch.items() if key != "labels"}
        if self.move_mutator:
            self.move_mutator(result)
        return result


def row():
    return dict(target_text="Original header. Two movements.", audio_path="not-opened-by-fixture")


def test_causal_scores_include_terminal_eos_and_skip_prefix_without_training():
    adapter = TinyAdapter()
    original = row()
    before = {name: value.detach().clone() for name, value in adapter.model.named_parameters()}
    result = score_target(adapter, original, chunk_size=2)
    batch = adapter.collate([row()])
    with torch.no_grad():
        logits = adapter.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               use_cache=False).logits
        expected = functional.cross_entropy(logits[0, 1:5].float(), batch["labels"][0, 2:], reduction="none")
        wrong_shift = functional.cross_entropy(logits[0, 2:].float(), batch["labels"][0, 2:], reduction="none")
    assert result["token_nll"] == pytest.approx(expected.tolist())
    assert result["token_nll"] != pytest.approx(wrong_shift.tolist())
    assert result["target_token_ids"] == [2, 5, 7, 9]
    assert result["eos_token_id"] == 9
    assert (result["input_token_count"], result["prefix_token_count"], result["target_token_count"]) == (6, 2, 4)
    assert result["cache_parity"] == "not_verified"
    assert result["use_cache"] is False and result["gradient_enabled"] is False
    assert result["truncation_applied"] is False
    assert result["target_completeness"] == "original_collate_suffix_verified"
    assert all(not call["use_cache"] and not call["grad"] and not call["training"] for call in adapter.model.calls)
    assert "collator_annotation" not in original
    for name, parameter in adapter.model.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    # Stable finite JSON, with complete labels/mask identity rather than only a count.
    json.dumps(result, allow_nan=False)
    expected_mask = [[False, False, True, True, True, True]]
    encoded = json.dumps(expected_mask, separators=(",", ":")).encode()
    assert result["target_mask_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert result["mask_summary"]["assistant_end_exclusive"] == 6


def test_only_selected_chunk_rows_are_cast_to_fp32(monkeypatch):
    adapter = TinyAdapter()
    calls = []
    native_float = torch.Tensor.float
    native_ce = functional.cross_entropy

    def tracked_float(tensor, *args, **kwargs):
        assert tensor.ndim != 3, "Must not allocate full-sequence FP32 logits"
        return native_float(tensor, *args, **kwargs)

    def tracked_ce(logits, targets, **kwargs):
        calls.append(tuple(logits.shape))
        assert logits.dtype == torch.float32 and logits.shape[0] <= 3
        assert logits.device == targets.device and targets.dtype == torch.long
        return native_ce(logits, targets, **kwargs)

    monkeypatch.setattr(torch.Tensor, "float", tracked_float)
    monkeypatch.setattr(functional, "cross_entropy", tracked_ce)
    score_target(adapter, row(), chunk_size=3)
    assert calls == [(3, 16), (1, 16)]


def test_repeat_calls_disable_dropout_and_do_not_change_adapter_methods():
    adapter = TinyAdapter()
    collate, mover = adapter.collate.__func__, adapter._move.__func__
    first = score_target(adapter, row(), chunk_size=1)
    second = score_target(adapter, row(), chunk_size=32)
    assert first["token_nll"] == pytest.approx(second["token_nll"])
    assert adapter.collate.__func__ is collate and adapter._move.__func__ is mover
    assert all(not module.training for module in adapter.model.modules())


@pytest.mark.parametrize("size", [0, -1, 1.5, True, None])
def test_invalid_chunk_sizes_are_rejected_before_forward(size):
    adapter = TinyAdapter()
    with pytest.raises(ValueError, match="chunk_size"):
        score_target(adapter, row(), chunk_size=size)
    assert not adapter.model.calls


@pytest.mark.parametrize("mutation", [
    lambda b: b["labels"].__setitem__((0, 3), -100),
    lambda b: b["labels"].__setitem__((0, -1), -100),
    lambda b: b["labels"].__setitem__((0, 0), 1),
    lambda b: b["labels"].fill_(-100),
    lambda b: b["labels"].__setitem__((0, 3), 6),
    lambda b: b.__setitem__("labels", b["labels"].float()),
    lambda b: b.__setitem__("labels", b["labels"][:, :-1]),
    lambda b: b.__setitem__("input_ids", b["input_ids"].expand(2, -1)),
    lambda b: b["input_ids"].__setitem__((0, 0), -1),
])
def test_bad_assistant_support_or_labels_fail_before_forward(mutation):
    adapter = TinyAdapter()
    adapter.batch_mutator = mutation
    with pytest.raises(ValueError):
        score_target(adapter, row())
    assert not adapter.model.calls


@pytest.mark.parametrize("target", [[2, 5, 7, 8], [2, 9, 7, 9], [2, 10, 7, 9]])
def test_target_must_end_in_exactly_one_declared_eos(target):
    adapter = TinyAdapter()
    def change(batch):
        batch["input_ids"][0, 2:] = torch.tensor(target)
        batch["labels"][0, 2:] = torch.tensor(target)
    adapter.batch_mutator = change
    with pytest.raises(ValueError, match="exactly one EOS"):
        score_target(adapter, row())
    assert not adapter.model.calls


@pytest.mark.parametrize("eos", [[], [True], [-1], [9.0], None])
def test_malformed_eos_declaration_is_rejected(eos):
    adapter = TinyAdapter()
    adapter.eos = eos
    with pytest.raises(ValueError, match="EOS IDs"):
        score_target(adapter, row())


@pytest.mark.parametrize("mutation", [
    lambda b: b["attention_mask"].__setitem__((0, 4), 0),
    lambda b: b["attention_mask"].__setitem__((0, 1), 0),
    lambda b: b["attention_mask"].__setitem__((0, 0), 2),
    lambda b: b.pop("attention_mask"),
    lambda b: b.__setitem__("attention_mask", b["attention_mask"][:, :-1]),
])
def test_invisible_target_or_malformed_attention_rejected(mutation):
    adapter = TinyAdapter()
    adapter.batch_mutator = mutation
    with pytest.raises(ValueError, match="attention"):
        score_target(adapter, row())
    assert not adapter.model.calls


def test_context_overflow_is_not_truncated():
    adapter = TinyAdapter()
    adapter.max_length = 5
    with pytest.raises(ValueError, match="context contract"):
        score_target(adapter, row())
    assert not adapter.model.calls


@pytest.mark.parametrize("stage", ["batch", "move"])
def test_preexisting_kv_state_is_never_reused(stage):
    adapter = TinyAdapter()
    mutation = lambda b: b.__setitem__("past_key_values", ("other-audio-cache",))
    if stage == "batch":
        adapter.batch_mutator = mutation
    else:
        adapter.move_mutator = mutation
    with pytest.raises(ValueError, match="pre-existing KV cache"):
        score_target(adapter, row())
    assert not adapter.model.calls


def test_mover_cannot_change_input_identity():
    adapter = TinyAdapter()
    adapter.move_mutator = lambda b: b.__setitem__("input_ids", torch.zeros_like(b["input_ids"]))
    with pytest.raises(ValueError, match="Moving inputs"):
        score_target(adapter, row())
    assert not adapter.model.calls


@pytest.mark.parametrize("field", ["input_ids", "attention_mask"])
def test_mover_cannot_change_input_identity_in_place(field):
    adapter = TinyAdapter()
    adapter.move_mutator = lambda b: b[field].zero_()
    with pytest.raises(ValueError, match="Moving inputs"):
        score_target(adapter, row())
    assert not adapter.model.calls


def test_original_collate_nonempty_target_cannot_have_only_eos():
    adapter = TinyAdapter()
    def change(batch):
        batch["labels"][0, :-1] = -100
    adapter.batch_mutator = change
    with pytest.raises(ValueError, match="content before its EOS"):
        score_target(adapter, row())
    assert not adapter.model.calls


def test_bf16_autocast_keeps_mixed_precision_parameters_intact():
    adapter = TinyAdapter()
    adapter.model.to(dtype=torch.bfloat16)
    adapter.model.head.float()  # Analogous to an FP32 trainable native projector.
    before = {name: (p.dtype, p.detach().clone()) for name, p in adapter.model.named_parameters()}
    result = score_target(adapter, row())
    assert result["bf16_autocast"] is True
    assert all(value >= 0 for value in result["token_nll"])
    for name, parameter in adapter.model.named_parameters():
        assert parameter.dtype == before[name][0]
        torch.testing.assert_close(parameter, before[name][1], rtol=0, atol=0)


def test_bad_eval_override_is_not_accepted_as_dropout_disabled():
    adapter = TinyAdapter()
    adapter.model.eval = lambda: adapter.model
    with pytest.raises(ValueError, match="eval mode"):
        score_target(adapter, row())
    assert not adapter.model.calls


def test_nonfinite_scored_nll_is_rejected_but_unused_prefix_is_not_scored():
    adapter = TinyAdapter()
    adapter.model.corrupt = "nan"
    with pytest.raises(ValueError, match="finite"):
        score_target(adapter, row())
    adapter.model.corrupt = "unused_prefix_nan"
    assert len(score_target(adapter, row())["token_nll"]) == 4


def test_full_native_logits_shape_is_required():
    adapter = TinyAdapter()
    adapter.model.corrupt = "short"
    with pytest.raises(ValueError, match="full-sequence logits"):
        score_target(adapter, row())


def test_context_allows_full_target_exactly_at_limit():
    adapter = TinyAdapter()
    adapter.max_length = 6
    assert score_target(adapter, row())["input_token_count"] == 6
