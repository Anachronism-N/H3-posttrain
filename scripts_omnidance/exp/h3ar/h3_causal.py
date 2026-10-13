import os
"""Block-causal attention for MiniMax-H3 Ref2VA (H3-AR, Causal Forcing++ stage 1-3).

Packed Ref2VA layout: [text | ref image | ref audio | target audio | target video | pad] (one cu segment + pad).
Rules (exact, via per-block key gathering + SDPA; no dense mask):
  prefix (text + references)  -> prefix                              (cacheable global context / sink)
  target audio (clean music)  -> prefix + all target audio           (music is a known input stream)
  video block j               -> prefix + target audio with time < end_j + lookahead + video blocks <= j
Video blocks: latent frame 0 alone (sink-like first frame), then `block` latent frames each.
Time = RoPE axis-0 coordinate (audio and target video share the origin; unit = one audio latent = 1/40 s).

Teacher forcing / AR sampling use the stock model_fn: context frames via input_latents_video + denoise_mask_video=0
(timestep 1.0 = clean), music via input_latents_audio + denoise_mask_audio=0.

usage:
    import h3_causal as HC
    HC.install()                                   # patch the DiT attention once
    plan = HC.make_plan(packed, n_ref_visual, n_ref_audio, latent_t, block=6, lookahead=8)
    with HC.use(plan): v_video, v_audio = model_fn(...)
"""
import contextlib
import torch
import torch.nn.functional as F

import diffsynth.models.minimax_h3_dit as HD

STATE = {"plan": None}
_orig = HD._sdpa_varlen_attention


def block_frames(latent_t, block, first=None):
    """first block = latents [0, first) (default 1 = the single-frame first latent; env OD_H3_FIRST).
    H3 video VAE groups latents by 5 (frames 1,4,4,4,4 = 17 frames): first=5, block=5 aligns every block to one VAE group."""
    first = int(os.environ.get("OD_H3_FIRST", "1")) if first is None else first
    b = [(0, first)]
    s = first
    while s < latent_t:
        b.append((s, min(s + block, latent_t)))
        s += block
    if os.environ.get("OD_H3_MERGE_TAIL") == "1" and len(b) > 2 and b[-1][1] - b[-1][0] < block:
        b = b[:-2] + [(b[-2][0], b[-1][1])]   # no short trailing block (H3's partial 5-frame VAE clip joins the previous block)
    return b


def make_plan(packed, n_ref_visual, n_ref_audio, latent_t, block=6, lookahead=8):
    img_pos, audio_pos, text_pos = packed["img_pos"], packed["audio_pos"], packed["text_pos"]
    g = packed["img_position_ids"][0, :, 0]
    dev = img_pos.device
    tv_pos = img_pos[n_ref_visual:]                     # target video rows (frame-major)
    ta_pos = audio_pos[n_ref_audio:]                    # target audio rows (channel-major)
    rows_per_frame = tv_pos.numel() // latent_t
    prefix = torch.cat([text_pos, img_pos[:n_ref_visual], audio_pos[:n_ref_audio]]).sort().values
    P = int(prefix.numel())
    assert torch.equal(prefix, torch.arange(P, device=dev)), "prefix must be contiguous [0, P)"
    A0, A1 = int(ta_pos.min()), int(ta_pos.max()) + 1
    V0, V1 = int(tv_pos.min()), int(tv_pos.max()) + 1
    assert A0 == P and V0 == A1 and V1 - V0 == tv_pos.numel(), (P, A0, A1, V0, V1)
    ta_time = g[A0:A1].float()
    vt = g[V0:V1].float().view(latent_t, rows_per_frame)[:, 0]
    frame_dt = float(vt[1] - vt[0]) if latent_t > 1 else 1.0
    groups = [(0, P, torch.arange(P, device=dev)), (A0, A1, torch.arange(A1, device=dev))]
    bounds = block_frames(latent_t, block)
    for f0, f1 in bounds:
        t_end = float(vt[f1 - 1]) + frame_dt + lookahead
        a_idx = torch.arange(A0, A1, device=dev)[ta_time < t_end]
        keys = torch.cat([torch.arange(P, device=dev), a_idx, torch.arange(V0, V0 + f1 * rows_per_frame, device=dev)])
        groups.append((V0 + f0 * rows_per_frame, V0 + f1 * rows_per_frame, keys))
    return dict(seq_used=V1, groups=groups, bounds=bounds, rows_per_frame=rows_per_frame, V0=V0)


def _causal_varlen(q, k, v, cu_seqlens, softmax_scale):
    plan = STATE["plan"]
    if plan is None or q.shape[0] < plan["seq_used"] or int(cu_seqlens[1]) != plan["seq_used"]:
        return _orig(q, k, v, cu_seqlens, softmax_scale)
    out = torch.zeros_like(q)
    for s, e, keys in plan["groups"]:
        qs = q[s:e].transpose(0, 1).unsqueeze(0)
        ks = k.index_select(0, keys).transpose(0, 1).unsqueeze(0)
        vs = v.index_select(0, keys).transpose(0, 1).unsqueeze(0)
        out[s:e] = F.scaled_dot_product_attention(qs, ks, vs, scale=softmax_scale).squeeze(0).transpose(0, 1)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[1:-1], bounds[2:]):     # padding segment(s): unchanged
        if stop > start:
            out[start:stop] = _orig(q[start:stop], k[start:stop], v[start:stop],
                                    torch.tensor([0, stop - start], device=cu_seqlens.device, dtype=cu_seqlens.dtype), softmax_scale)
    return out


def install():
    HD._sdpa_varlen_attention = _causal_varlen


@contextlib.contextmanager
def use(plan):
    old = STATE["plan"]; STATE["plan"] = plan
    try:
        yield
    finally:
        STATE["plan"] = old


def frame_mask(latent_t, clean_until, like):
    """denoise_mask_video for teacher forcing: frames < clean_until are clean context (0), others denoised (1)."""
    m = torch.ones_like(like)
    m[:, :, :clean_until] = 0
    return m
