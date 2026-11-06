"""Weighted assistant-token CE must equal the unweighted sum scaled by the weight."""
import importlib.util
import pathlib

import pytest

torch = pytest.importorskip('torch')
MODULE = pathlib.Path(__file__).resolve().parents[1] / 'trainer' / 'music_planner_token_loss.py'


def load_module():
    spec = importlib.util.spec_from_file_location('m2d_token_loss', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_ce_scales_exactly():
    module = load_module()
    torch.manual_seed(0)
    batch, steps, vocab = 2, 12, 25
    logits = torch.randn(batch, steps, vocab)
    labels = torch.full((batch, steps), -100, dtype=torch.long)
    labels[:, 6:] = torch.randint(0, vocab, (batch, steps - 6))
    labels[1, 6] = -100
    base, count = module.assistant_token_nll(logits, labels, chunk_size=4)
    weights = module.assistant_token_weights(labels, 2.0)
    weighted, count2 = module.assistant_token_nll(logits, labels, chunk_size=4, token_weights=weights)
    assert int(count) == int(count2)
    assert float(weighted) == pytest.approx(float(base) * 2.0, rel=1e-5)


def test_weights_none_for_unit_weight():
    module = load_module()
    labels = torch.tensor([[1, 2, -100]])
    assert module.assistant_token_weights(labels, 1.0) is None
    assert module.assistant_token_weights(labels, 2.0).shape == labels.shape
