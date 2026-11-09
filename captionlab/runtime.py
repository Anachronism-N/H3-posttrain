"""Teacher-forced diagnostics for an explicitly supplied original adapter.

This module never loads weights, generates text, trains, or implements a generic
native KV-cache driver. The caller must audit the adapter/checkpoint protocol.
In particular, the original ``collate`` owns tokenization (including any separate
header token boundary); this function neither re-encodes nor truncates its target.
"""
import copy
import hashlib
import json
from collections.abc import Mapping


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _check_eval(model):
    if not callable(getattr(model, "modules", None)):
        raise ValueError("Require a native torch model with inspectable eval mode")
    if any(module.training for module in model.modules()):
        raise ValueError("Every model module must be in eval mode for deterministic scoring")


def score_target(adapter, row, *, chunk_size=32):
    """Score one complete original-collate assistant suffix, including its EOS.

    The adapter must expose ``collate``, ``_move``, ``eos_token_ids`` and ``model``.
    All target IDs, masks and attention are validated before forward. Context
    limits fail rather than truncate. Only selected assistant logit rows are
    converted to FP32, at most ``chunk_size`` rows at once; native full-sequence
    logits can still occupy memory. The result contains ordinary finite JSON.

    ``input_token_count`` counts the complete teacher-forced sequence; the masked
    prefix and supervised target have separate counts. The masked prefix may
    include a protocol's deliberately unsupervised header, not just audio tokens.
    Token-ID hashes do not
    identify the audio waveform or features: the caller must bind those inputs.

    Suffix completeness is verified against the original collate output, not by
    a second tokenizer. The caller must separately verify that collate implements
    the audited raw-text/checkpoint contract without truncation.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(row, Mapping) or not isinstance(row.get("target_text"), str) or not row["target_text"].strip():
        raise ValueError("Require a row with nonempty original target_text")
    for name in ("collate", "_move", "eos_token_ids"):
        if not callable(getattr(adapter, name, None)):
            raise ValueError("Original adapter lacks " + name)
    model = getattr(adapter, "model", None)
    if not callable(model) or not callable(getattr(model, "eval", None)):
        raise ValueError("Original adapter lacks a callable model with eval mode")

    # Lazy import keeps prepare/check/CLI help usable without a Torch install.
    import torch
    import torch.nn.functional as functional

    with torch.no_grad():
        # Some collators annotate their input row; never alter the caller's row.
        batch = adapter.collate([copy.deepcopy(dict(row))])
        if not isinstance(batch, Mapping):
            raise ValueError("Original collate must return a tensor mapping")
        input_ids, labels = batch.get("input_ids"), batch.get("labels")
        if (not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor)
                or input_ids.ndim != 2 or input_ids.shape[0] != 1
                or input_ids.shape != labels.shape or input_ids.shape[1] < 2
                or input_ids.dtype != torch.long or labels.dtype != torch.long):
            raise ValueError("Require aligned int64 input_ids/labels [1,T], T >= 2")
        ids_cpu = input_ids.detach().cpu().clone()
        labels_cpu = labels.detach().cpu().clone()
        if (ids_cpu < 0).any().item() or ((labels_cpu < 0) & labels_cpu.ne(-100)).any().item():
            raise ValueError("Token IDs must be nonnegative; only -100 may mask labels")
        mask_cpu = labels_cpu.ne(-100)
        support = mask_cpu[0].nonzero(as_tuple=False).flatten()
        if not support.numel():
            raise ValueError("Require a nonempty supervised assistant suffix")
        prefix_length, sequence_length = int(support[0].item()), int(input_ids.shape[1])
        if prefix_length < 1 or not mask_cpu[0, prefix_length:].all().item():
            raise ValueError("Assistant labels must form one continuous suffix after a masked prefix")
        if not torch.equal(ids_cpu[:, prefix_length:], labels_cpu[:, prefix_length:]):
            raise ValueError("Assistant labels differ from the actual teacher-forced input tokens")
        target_ids = labels_cpu[0, prefix_length:].tolist()
        if len(target_ids) < 2:
            raise ValueError("Nonempty target text requires content before its EOS")
        eos_ids = adapter.eos_token_ids()
        if (not isinstance(eos_ids, (list, tuple)) or not eos_ids
                or any(type(value) is not int or value < 0 for value in eos_ids)):
            raise ValueError("Original adapter must declare nonnegative integer EOS IDs")
        eos_set = set(eos_ids)
        if target_ids[-1] not in eos_set or sum(value in eos_set for value in target_ids) != 1:
            raise ValueError("Assistant suffix must contain exactly one EOS, at its end")
        limit = getattr(adapter, "max_length", None)
        if limit is not None and (type(limit) is not int or limit < 1 or sequence_length > limit):
            raise ValueError("Complete teacher-forced target exceeds the adapter context contract")
        attention = batch.get("attention_mask")
        if (not isinstance(attention, torch.Tensor) or attention.shape != input_ids.shape
                or not ((attention == 0) | (attention == 1)).all().item()
                or not attention[0, prefix_length - 1:].eq(1).all().item()):
            raise ValueError("Require aligned binary attention with the last prefix and full target visible")
        attention_cpu = attention.detach().cpu().clone()
        if batch.get("past_key_values") is not None:
            raise ValueError("Teacher-forced scoring cannot consume a pre-existing KV cache")

        inputs = adapter._move(dict(batch))
        if not isinstance(inputs, Mapping):
            raise ValueError("Original adapter _move must return a model input mapping")
        inputs = dict(inputs)
        inputs.pop("labels", None)  # Do not ask the native model to compute another loss.
        if inputs.get("past_key_values") is not None:
            raise ValueError("Teacher-forced scoring cannot consume a pre-existing KV cache")
        moved_ids, moved_attention = inputs.get("input_ids"), inputs.get("attention_mask")
        if (not isinstance(moved_ids, torch.Tensor) or moved_ids.dtype != torch.long
                or not torch.equal(moved_ids.detach().cpu(), ids_cpu)
                or not isinstance(moved_attention, torch.Tensor)
                or not torch.equal(moved_attention.detach().cpu(), attention_cpu)):
            raise ValueError("Moving inputs changed the original IDs or attention contract")
        inputs["use_cache"] = False
        model.eval()
        _check_eval(model)
        parameter = next(model.parameters(), None)
        device_type = parameter.device.type if parameter is not None else moved_ids.device.type
        use_bf16_autocast = parameter is not None and parameter.dtype == torch.bfloat16
        # Match the native adapter's teacher_forced BF16 rule without recasting
        # any base/trainable parameter or replacing the original input mover.
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_bf16_autocast):
            output = model(**inputs)
        _check_eval(model)
        logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 3
                or tuple(logits.shape[:2]) != tuple(input_ids.shape)
                or logits.shape[-1] < 2 or not logits.is_floating_point()):
            raise ValueError("Native forward must return floating full-sequence logits [1,T,V]")
        if max(target_ids) >= logits.shape[-1]:
            raise ValueError("Target token ID is outside the native output vocabulary")
        token_nll = []
        for start in range(0, len(target_ids), chunk_size):
            end = min(start + chunk_size, len(target_ids))
            # The first assistant label is predicted at the last prefix position.
            selected = logits[0, prefix_length + start - 1:prefix_length + end - 1]
            targets = labels_cpu[0, prefix_length + start:prefix_length + end].to(logits.device)
            losses = functional.cross_entropy(selected.float(), targets, reduction="none")
            if not torch.isfinite(losses).all().item():
                raise ValueError("Teacher-forced assistant NLL must be finite for every token")
            token_nll.extend(losses.detach().cpu().tolist())

    return {
        "target_token_ids": target_ids,
        "token_nll": token_nll,
        "eos_token_id": target_ids[-1],
        "input_token_count": sequence_length,
        "prefix_token_count": prefix_length,
        "target_token_count": len(target_ids),
        "target_text_sha256": hashlib.sha256(row["target_text"].encode("utf-8")).hexdigest(),
        "input_ids_sha256": _digest(ids_cpu.tolist()),
        "prefix_ids_sha256": _digest(ids_cpu[0, :prefix_length].tolist()),
        "labels_sha256": _digest(labels_cpu.tolist()),
        "target_mask_sha256": _digest(mask_cpu.tolist()),
        "attention_mask_sha256": _digest(attention_cpu.tolist()),
        "mask_summary": {
            "shape": [1, sequence_length], "masked_prefix_tokens": prefix_length,
            "assistant_start": prefix_length, "assistant_end_exclusive": sequence_length,
            "assistant_is_contiguous_suffix": True, "eos_is_single_terminal_token": True,
        },
        "target_completeness": "original_collate_suffix_verified",
        "truncation_applied": False,
        "model_eval": True,
        "gradient_enabled": False,
        "use_cache": False,
        "bf16_autocast": use_bf16_autocast,
        "cache_parity": "not_verified",
    }
