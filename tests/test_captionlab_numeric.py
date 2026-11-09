import pytest

torch = pytest.importorskip("torch")
from captionlab.numeric_common import compare,digest_tensor,fixed_positions


def test_identical_logits_zero_controls():
    x=torch.randn(5,23)
    rows=compare(x,x.clone(),[1,2,3,4,5])
    assert all(r['max_logit_abs_difference']==0 and r['top1_equal'] and r['target_logprob_difference']==0 for r in rows)


def test_offset_is_not_probability_change():
    x=torch.tensor([[1.,2.,3.],[3.,1.,0.]])
    rows=compare(x,x+2,[2,0])
    assert all(r['max_logit_abs_difference']==2 for r in rows)
    assert all(r['max_centered_logit_abs_difference']<1e-6 and r['total_variation']==0 and r['target_logprob_difference']==0 for r in rows)


def test_distribution_shift_and_top1_change_visible():
    rows=compare(torch.tensor([[1.,2.,0.]]),torch.tensor([[3.,1.,0.]]),[1])
    assert not rows[0]['top1_equal'] and rows[0]['total_variation']>0 and rows[0]['target_logprob_difference']<0


def test_hash_binds_dtype_shape_and_values():
    x=torch.arange(8,dtype=torch.float32)
    assert digest_tensor(x)==digest_tensor(x.clone())
    assert digest_tensor(x)!=digest_tensor(x.to(torch.bfloat16))
    assert digest_tensor(x)!=digest_tensor(x.reshape(2,4))


def test_mismatched_logits_rejected():
    with pytest.raises(ValueError):compare(torch.randn(2,4),torch.randn(3,4))
