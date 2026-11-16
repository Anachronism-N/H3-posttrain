"""Exact causal assistant-token CE without a full-sequence FP32 vocabulary copy.

Chunking controls the CE temporary, not the model's native output-logit storage.
An empty tail slot retains a zero-gradient graph and participates in collectives.

``token_weights`` scales per-token CE before summation, which is how the
per-window density experiments emphasise answer content. Weights are indexed with
the same (row, column) pairs as the labels, so they must be aligned with the
causally shifted label tensor (see ``assistant_token_weights``).
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_nll(logits, rows, columns, targets):
    return F.cross_entropy(logits[rows, columns].float(), targets, reduction='sum')


def _chunk_nll_weighted(logits, rows, columns, targets, weights):
    per_token = F.cross_entropy(logits[rows, columns].float(), targets, reduction='none')
    return (per_token * weights).sum()


def assistant_token_weights(labels, weight=2.0):
    """Return per-token CE weights (same shape as labels), or None to disable."""
    if weight <= 1.0:
        return None
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[labels.ne(-100)] = weight
    return weights


def assistant_token_nll(logits, labels, chunk_size=0, token_weights=None):
    if (logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape
            or logits.device != labels.device or labels.dtype != torch.long):
        raise ValueError('Require aligned logits [B,T,V] and int64 labels [B,T] on one device')
    if type(chunk_size) is not int or chunk_size < 0:
        raise ValueError('Loss chunk size must be a nonnegative integer')
    shifted = labels[:, 1:]
    rows, columns = shifted.ne(-100).nonzero(as_tuple=True)
    count = shifted.ne(-100).sum()
    if not rows.numel():
        # Do not multiply arbitrary (possibly nonfinite) logits by zero.
        return logits.reshape(-1)[:0].sum().float(), count
    if token_weights is not None:
        if token_weights.shape == labels.shape:
            token_weights = token_weights[:, 1:]
        if token_weights.shape != shifted.shape:
            raise ValueError('Token weights must align with labels or shifted labels')
    size = chunk_size or len(rows)
    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(rows), size):
        end = min(start + size, len(rows))
        if token_weights is None:
            inputs = (logits, rows[start:end], columns[start:end], shifted[rows[start:end], columns[start:end]])
            fn = _chunk_nll
        else:
            inputs = (logits, rows[start:end], columns[start:end],
                      shifted[rows[start:end], columns[start:end]],
                      token_weights[rows[start:end], columns[start:end]].to(logits.device))
            fn = _chunk_nll_weighted
        if chunk_size and torch.is_grad_enabled() and logits.requires_grad:
            part = checkpoint(fn, *inputs, use_reentrant=False, preserve_rng_state=False)
        else:
            part = fn(*inputs)
        total = total + part
    return total, count


def joint_update_objective(caption_sum, genre_sum, global_caption_tokens,
                           global_genre_sources, *, world, accum, genre_weight=0.,
                           count_sum=None, global_count_sources=None, count_weight=0.):
    """Undo framework GAS/DP means; normalize each task over its own global support.

    Accelerate DDP and managed DeepSpeed divide by GAS and average DP gradients.
    Thus each rank/microbatch contributes world*GAS times its own unnormalized
    sum. Unknown genre labels and padded tail slots never enter either count.
    """
    if world < 1 or accum < 1 or genre_weight < 0 or count_weight < 0:
        raise ValueError('Invalid distributed objective configuration')
    if global_caption_tokens.item() <= 0:
        raise ValueError('No valid assistant tokens in optimizer update')
    objective = caption_sum / global_caption_tokens
    if genre_weight and genre_sum is None:
        raise ValueError('Enabled JEV requires a graph-connected genre sum')
    if genre_sum is not None:
        # Keep the head in the graph even if this entire update is unlabelled.
        objective = objective + genre_weight * genre_sum / global_genre_sources.clamp_min(1)
    if count_weight:
        if count_sum is None or global_count_sources is None:
            raise ValueError('Enabled window-count loss requires a graph-connected count sum')
        objective = objective + count_weight * count_sum / global_count_sources.clamp_min(1)
    return objective * (world * accum)
