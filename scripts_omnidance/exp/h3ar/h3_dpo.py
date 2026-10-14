"""W1b (notes/19 §3): music-contrastive Diffusion-DPO on bidirectional MiniMax-H3.
Same video, same noise, same t, two music conditions:
  positive = the clip's own (clean, time-aligned) music;  negative = another clip's music (prob OD_H3_DPO_POTHER, swaps the
  reference-audio anchor too) or the clip's own music circularly shifted by 35-65 % of its length (rhythm-misaligned; ref anchor kept).
  e = ||v(x_t, music) - target||^2 (video only); ref = same base with the LoRA disabled (the CFG-fix adapter stays fused).
  L = (1/beta) * -log sigmoid(-beta * ((e_pos - ref_pos) - (e_neg - ref_neg)))  +  lambda * w(t) * e_pos   (SFT term keeps quality)
i.e. the policy must explain the real video better with the right music than with wrong / mis-timed music, relative to the base.
4 forwards per step (3 no-grad, 1 with grad: random branch, unbiased estimator); every rank does the same forwards (ZeRO-3 safe).
env: OD_H3_DPO=1, OD_H3_DPO_BETA (1000), OD_H3_DPO_SFT (1.0), OD_H3_DPO_POTHER (0.5)
"""
import os, random
import torch
import torch.nn.functional as F

STATE = {"prev": None, "n": 0, "acc": 0.0, "l": 0.0}


def _lora_layers(model):
    from peft.tuners.lora import LoraLayer
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def set_lora(model, on):
    for m in _lora_layers(model):
        m._disable_adapters = not on


def neg_music(inputs):
    a = inputs["audio_input_latents"]; ra = inputs.get("ref_audio_anchor")
    prev = STATE["prev"]
    STATE["prev"] = (a.detach().clone(), None if ra is None else ra.detach().clone())
    if prev is not None and prev[0].shape == a.shape and (ra is None or prev[1].shape == ra.shape) \
            and random.random() < float(os.environ.get("OD_H3_DPO_POTHER", "0.5")):
        return prev[0], prev[1], "other"
    k = int(a.shape[-1] * random.uniform(0.35, 0.65))
    return torch.roll(a, k, dims=-1), ra, "shift"


def dpo_loss(pipe, models, inputs, timestep_video, timestep_audio, target):
    dit = models["dit"]
    beta = float(os.environ.get("OD_H3_DPO_BETA", "1000")); lam = float(os.environ.get("OD_H3_DPO_SFT", "1.0"))
    a_neg, ra_neg, kind = neg_music(inputs)
    neg = dict(inputs)
    neg["audio_latents"] = a_neg; neg["input_latents_audio"] = a_neg; neg["audio_input_latents"] = a_neg
    if ra_neg is not None:
        neg["ref_audio_anchor"] = ra_neg

    V = {}

    def err(inp, grad, name=None):
        kw = dict(inp)
        if not grad:
            kw["use_gradient_checkpointing"] = False; kw["use_gradient_checkpointing_offload"] = False
        with torch.set_grad_enabled(grad):
            v, _ = pipe.model_fn(**models, **kw, timestep_video=timestep_video, timestep_audio=timestep_audio)
        if name:
            V[name] = v
        return ((v.float() - target.float()) ** 2).mean()

    # 10-14 base anchor (notes/22 §9): the SFT term regresses to the frozen base prediction at sigma >= OD_H3_ANCHOR_SIG
    # instead of the FM target (FM pulls the distilled high-noise predictions to the conditional mean -> blur)
    anchor = float(os.environ.get("OD_H3_ANCHOR", "0")); sig = float(timestep_video.reshape(-1)[0]) / 1000.0
    use_anchor = anchor > 0 and sig >= float(os.environ.get("OD_H3_ANCHOR_SIG", "0.5"))

    # ZeRO-3 cannot take gradients from two forwards of the same params in one backward (duplicate ds_id in the reduce bucket),
    # so only ONE branch carries gradient per step. c = sigmoid(beta * diff) = d(L_dpo/beta)/d diff (computed from all 4 values).
    # pos branch (p=1/2): 2*c*e_pos + 2*lam*w*e_pos ;  neg branch: -2*c*e_neg   -> unbiased for  c*(grad e_pos - grad e_neg) + lam*w*grad e_pos
    gpos = random.random() < 0.5
    with torch.no_grad():
        set_lora(dit, False)
        rp, rn = err(inputs, False, "base_pos"), err(neg, False)
        set_lora(dit, True)
        e_other = err(neg if gpos else inputs, False)
    e_grad = err(inputs if gpos else neg, True, "grad")
    ep, en = (e_grad, e_other) if gpos else (e_other, e_grad)
    diff = (ep.detach() - rp) - (en.detach() - rn)
    l_dpo = -F.logsigmoid(-beta * diff)
    c = torch.sigmoid(beta * diff)  # = d(L_dpo/beta)/d diff : DPO gradient rescaled by 1/beta so it is on the same scale as the SFT term
    w = pipe.scheduler.training_weight(timestep_video)
    if gpos and use_anchor:
        e_sft = anchor * ((V["grad"].float() - V["base_pos"].detach().float()) ** 2).mean()
    else:
        e_sft = e_grad
    loss = (2 * c * e_grad + 2 * lam * w * e_sft) if gpos else (-2 * c * e_grad)
    STATE["n"] += 1; STATE["acc"] += float(diff.item() < 0); STATE["l"] += float(l_dpo.item())
    if STATE["n"] % 10 == 1:
        print(f"[OD H3-DPO] n={STATE['n']} kind={kind} t={float(timestep_video):.0f} ep={ep.item():.4f} en={en.item():.4f} "
              f"rp={rp.item():.4f} rn={rn.item():.4f} margin={-diff.item():.2e} dpo={l_dpo.item():.4f} "
              f"acc_avg={STATE['acc'] / STATE['n']:.3f} dpo_avg={STATE['l'] / STATE['n']:.4f}", flush=True)
    return loss
