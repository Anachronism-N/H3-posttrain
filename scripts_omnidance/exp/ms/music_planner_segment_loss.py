"""Exact assistant-only CE with optional checkpointed vocabulary-chunk recompute."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_sums(logits, rows, columns, targets, segments):
    # Select before FP32 conversion. Checkpoint recomputes this temporary in backward.
    selected = logits[rows, columns].float()
    losses = F.cross_entropy(selected, targets, reduction='none')
    return losses.new_zeros(3).scatter_add(0, segments, losses)


def assistant_segment_sums(logits, shifted_labels, shifted_segments, chunk_size=0):
    if type(chunk_size) is not int or chunk_size < 0:
        raise ValueError('Loss chunk size must be a nonnegative integer')
    rows, columns = shifted_labels.ne(-100).nonzero(as_tuple=True)
    if not rows.numel():
        raise ValueError('No supervised assistant tokens')
    targets = shifted_labels[rows, columns]
    segments = shifted_segments[rows, columns]
    counts = torch.bincount(segments, minlength=3)
    sums = logits.new_zeros(3, dtype=torch.float32)
    size = chunk_size or len(rows)
    for start in range(0, len(rows), size):
        end = start + size
        inputs = (logits, rows[start:end], columns[start:end], targets[start:end], segments[start:end])
        if chunk_size and torch.is_grad_enabled() and logits.requires_grad:
            part = checkpoint(_chunk_sums, *inputs, use_reentrant=False, preserve_rng_state=False)
        else:
            part = _chunk_sums(*inputs)
        sums = sums + part
    return sums, counts
