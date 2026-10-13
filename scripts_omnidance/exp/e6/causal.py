"""E6: block-causal attention + per-frame timesteps for the OmniDance Wan2.2-TI2V-5B audio DiT.

Layout (latent frames): frame 0 = clean first-frame sink (its own block), then blocks of `block` latent frames.
Query tokens of block j attend to keys of blocks 0..j (exact, computed block-wise with unmasked SDPA, so no
dense mask is materialised). Audio/text cross-attention is untouched (music is known in advance -> non-causal OK).

Per-frame timesteps: pass `timestep` as a [T] tensor (sigma*1000 per latent frame); frame 0 is forced to 0.
Enable with `enable(tokens_per_frame, block)`; `disable()` restores full bidirectional attention.
"""
import torch
import torch.nn.functional as F
from einops import rearrange

import diffsynth.models.wan_video_dit as D
import diffsynth.pipelines.wan_video as P

STATE = {"on": False, "tpf": None, "block": 4}


def block_bounds(n_frames, block):
    """[(start_frame, end_frame)] for sink + chunks of `block` frames."""
    b = [(0, 1)]
    s = 1
    while s < n_frames:
        b.append((s, min(s + block, n_frames)))
        s += block
    return b


def _causal_self_attn_forward(self, x, freqs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)
    q = D.rope_apply(q, freqs, self.num_heads)
    k = D.rope_apply(k, freqs, self.num_heads)
    if not STATE["on"]:
        return self.o(self.attn(q, k, v))
    tpf = STATE["tpf"]
    L = q.shape[1]
    assert L % tpf == 0, (L, tpf)
    q, k, v = (rearrange(t, "b s (n d) -> b n s d", n=self.num_heads) for t in (q, k, v))
    outs = []
    for s, e in block_bounds(L // tpf, STATE["block"]):
        outs.append(F.scaled_dot_product_attention(q[:, :, s * tpf:e * tpf], k[:, :, :e * tpf], v[:, :, :e * tpf]))
    x = rearrange(torch.cat(outs, dim=2), "b n s d -> b s (n d)")
    return self.o(x)


_orig_prepare = P.prepare_wan_video_backbone_inputs


def _prepare_per_frame(dit, latents, timestep, context, *a, **kw):
    T = latents.shape[2]
    if timestep.numel() == T and T > 1 and dit.seperated_timestep and kw.get("fuse_vae_embedding_in_latents", False):
        hw = latents.shape[3] * latents.shape[4] // 4
        ts = timestep.to(latents.dtype).clone().view(T)
        ts[0] = 0
        out = _orig_prepare(dit, latents, timestep.flatten()[1:2], context, *a, **kw)  # build everything else
        pt = ts.view(T, 1).expand(T, hw).flatten()
        t = dit.time_embedding(D.sinusoidal_embedding_1d(dit.freq_dim, pt).unsqueeze(0))
        out["t"] = t
        out["t_mod"] = dit.time_projection(t).unflatten(2, (6, dit.dim))
        out["prepared_timestep"] = pt if out["prepared_timestep"].dim() == 1 else pt.unsqueeze(0)
        return out
    return _orig_prepare(dit, latents, timestep, context, *a, **kw)


D.SelfAttention.forward = _causal_self_attn_forward
P.prepare_wan_video_backbone_inputs = _prepare_per_frame


def enable(tokens_per_frame, block=4):
    STATE.update(on=True, tpf=tokens_per_frame, block=block)


def disable():
    STATE["on"] = False


def velocity_pf(pipe, shared, cond, x_t, sig_frames, grad=False, mode="gen"):
    """velocity with per-frame sigmas `sig_frames` [T] (frame 0 forced clean)."""
    import dmd_core as C
    C.DualLoRALinear.mode = mode
    x = x_t.clone()
    x[:, :, 0:1] = shared["first_frame_latents"]
    t = (sig_frames.to(x.device).float() * 1000.0).to(pipe.torch_dtype)
    kw = dict(shared)
    kw.update(cond)
    kw["latents"] = x
    kw["use_gradient_checkpointing"] = grad
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    with torch.set_grad_enabled(grad):
        return pipe.model_fn(**models, **kw, timestep=t)


@torch.no_grad()
def ar_sample(pipe, shared, posi, block=4, sigmas=None, seed=0, prefix_recompute=True, traj=None, prefix=None):
    """Autoregressive block-wise 4-step sampling. Context blocks are re-encoded at t=0 (exact KV-cache
    equivalent under block-causal attention). Returns full latents [1,C,T,H,W]."""
    import dmd_core as C
    sigmas = sigmas or C.STUDENT_SIGMAS
    lat = shared["latents"]
    T = lat.shape[2]
    gen = torch.Generator(device=lat.device).manual_seed(seed)
    out = torch.zeros_like(lat)
    out[:, :, 0:1] = shared["first_frame_latents"]
    P = 0 if prefix is None else prefix.shape[2]
    if P:
        out[:, :, 1:1 + P] = prefix.to(out.dtype)
    for s0, e0 in block_bounds(T, block)[1:]:
        if e0 <= 1 + P:
            continue
        x = torch.randn(lat[:, :, s0:e0].shape, generator=gen, device=lat.device, dtype=lat.dtype)
        xs = []
        for k, s in enumerate(sigmas):
            xs.append(x.clone().cpu())
            # full-length input: future blocks cannot influence block [s0,e0) under block-causal attention
            cur = torch.cat([out[:, :, :s0], x, torch.zeros_like(lat[:, :, e0:])], dim=2)
            sf = torch.ones(T, device=lat.device)
            sf[:s0] = 0.0
            sf[s0:e0] = s
            v = velocity_pf(pipe, shared, posi, cur, sf, grad=False, mode="gen")
            x0 = (cur - s * v)[:, :, s0:e0]
            if k + 1 < len(sigmas):
                sn = sigmas[k + 1]
                x = (1 - sn) * x0 + sn * torch.randn(x0.shape, generator=gen, device=x0.device, dtype=x0.dtype)
            else:
                x = x0
        out[:, :, s0:e0] = x
        if traj is not None:
            traj.append(dict(s0=s0, e0=e0, xs=torch.stack(xs), out=x.clone().cpu()))
    return out
