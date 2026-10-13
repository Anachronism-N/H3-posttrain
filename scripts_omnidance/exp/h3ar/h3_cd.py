"""H3 Causal Forcing++ stage 2: causal consistency distillation (CD) inside the upstream DiffSynth trainer (ZeRO-3, LoRA).

Three LoRA adapters on the same frozen 33B base (peft multi-adapter):
  default = student (trainable, initialised from the AR teacher via --lora_checkpoint)
  teacher = frozen stage-1 AR teacher (TF/DF mixed, block-causal)
  ema     = EMA of the student (consistency target)
Per step (clean GT history = teacher forcing context, block-causal attention, music = clean audio track):
  pick block b and teacher-grid index n (50-step, shift 5):  x_n = (1-s_n) z + s_n eps on b (future frames = noise)
  x_{n+1} = x_n + (s_{n+1}-s_n) v_teacher(x_n, s_n)                (one ODE step of the frozen AR teacher)
  target  = x_{n+1} - s_{n+1} v_ema(x_{n+1}, s_{n+1})   (= x_{n+1} if s_{n+1}=0)
  loss    = || x_n - s_n v_student(x_n, s_n) - sg(target) ||^2 on block b
H3 checkpoints have CFG distilled in (sampling cfg_scale=1) -> single conditional branch, no CFG here.
Enable: OD_H3_CD_TEACHER=<teacher lora .safetensors> (setup in train.py) + OD_H3_CD=1 (loss). Optional OD_H3_CD_EMA (0.999).
"""
import copy, os, random

import torch

STATE = {"sig": None, "ema_pairs": None}


def _lora_layers(model):
    from peft.tuners.lora import LoraLayer
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def set_active(model, name):
    for m in _lora_layers(model):
        m._active_adapter = name


def setup(pipe, model, teacher_path, rank, target_modules):
    """Called before accelerator.prepare: add frozen 'teacher' and 'ema' adapters loaded from the AR-teacher LoRA."""
    from peft import LoraConfig, inject_adapter_in_model
    from diffsynth.core import load_state_dict
    tm = target_modules[0] if isinstance(target_modules, list) and len(target_modules) == 1 else target_modules
    sd = load_state_dict(teacher_path)
    sd = pipe.lora_loader(torch_dtype=pipe.torch_dtype, device=pipe.device).convert_state_dict(sd)
    for name in ("teacher", "ema"):
        inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=rank, target_modules=tm), model, adapter_name=name)
        m = {}
        for k, v in sd.items():
            k2 = k.replace("lora_A.default.weight", "lora_A.weight").replace("lora_B.default.weight", "lora_B.weight")
            m[k2.replace("lora_A.weight", f"lora_A.{name}.weight").replace("lora_B.weight", f"lora_B.{name}.weight")] = v
        res = model.load_state_dict(m, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:4]
        print(f"[OD H3-CD] adapter '{name}' loaded {len(m)} tensors from {teacher_path}", flush=True)
    for n, p in model.named_parameters():
        if ".teacher." in n or ".ema." in n:
            p.requires_grad_(False)
            p.data = p.data.to(pipe.torch_dtype)
    set_active(model, "default")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OD H3-CD] trainable (student) params {n_tr / 1e6:.1f}M", flush=True)


def _part(p):
    t = getattr(p, "ds_tensor", None)
    return t if t is not None and t.numel() > 0 else p.data


@torch.no_grad()
def ema_update(model, decay):
    if STATE["ema_pairs"] is None:
        P = dict(model.named_parameters())
        STATE["ema_pairs"] = [(P[n.replace(".ema.", ".default.")], q) for n, q in P.items() if ".ema." in n]
    for s, e in STATE["ema_pairs"]:
        es, ss = _part(e), _part(s)
        es.mul_(decay).add_(ss.to(es.dtype), alpha=1 - decay)


def teacher_sigmas(pipe, steps=50, shift=None):
    shift = float(os.environ.get("OD_H3_CD_SHIFT", "12")) if shift is None else shift  # 10-10: official video shift 12 (was 5)
    if STATE["sig"] is None:
        sc = copy.deepcopy(pipe.scheduler)
        sc.set_timesteps(steps, shift=shift)
        STATE["sig"] = [float(s) for s in sc.sigmas[:steps]] + [0.0]
    return STATE["sig"]


def cd_loss(pipe, models, inputs, plan):
    dit = models["dit"]
    ema_update(dit, float(os.environ.get("OD_H3_CD_EMA", "0.999")))
    sig = teacher_sigmas(pipe, int(os.environ.get("OD_H3_CD_STEPS", "50")))
    z = inputs["input_latents"]; T = z.shape[2]
    f0, f1 = plan["bounds"][random.randint(0, len(plan["bounds"]) - 1)]
    n = random.randint(0, len(sig) - 2)
    sn, sm = sig[n], sig[n + 1]
    eps = torch.randn_like(z)
    x = eps.clone(); x[:, :, :f0] = z[:, :, :f0]
    x[:, :, f0:f1] = (1 - sn) * z[:, :, f0:f1] + sn * eps[:, :, f0:f1]
    m = torch.ones_like(z); m[:, :, :f0] = 0
    base = dict(inputs); base["input_latents_video"] = z; base["denoise_mask_video"] = m
    ta = torch.tensor([1000.0], device=z.device)  # audio rows are clean (mask 0) under OD_H3_CLEAN_AUDIO

    def fwd(xx, s, grad):
        kw = dict(base); kw["video_latents"] = xx
        if not grad:
            kw["use_gradient_checkpointing"] = False; kw["use_gradient_checkpointing_offload"] = False
        tv = torch.tensor([s * 1000.0], device=z.device)
        with torch.set_grad_enabled(grad):
            v, _ = pipe.model_fn(**models, **kw, timestep_video=tv, timestep_audio=ta)
        return v

    with torch.no_grad():
        set_active(dit, "teacher")
        vt = fwd(x, sn, False)
        xm = x.clone(); xm[:, :, f0:f1] = x[:, :, f0:f1] + (sm - sn) * vt[:, :, f0:f1]
        # always run the EMA forward: under ZeRO-3 every rank must issue the same number of forwards (param all-gathers),
        # otherwise a rank that skips it (s_{n+1}=0) deadlocks the others (seen in the 10-10 smoke: NCCL allgather timeout)
        set_active(dit, "ema")
        ve = fwd(xm, max(sm, 1e-3), False)
        target = xm[:, :, f0:f1] if sm <= 0 else xm[:, :, f0:f1] - sm * ve[:, :, f0:f1]
        set_active(dit, "default")
    v = fwd(x, sn, True)
    pred = x[:, :, f0:f1] - sn * v[:, :, f0:f1]
    loss = ((pred.float() - target.float()) ** 2).mean()
    if not getattr(cd_loss, "_logged", False):
        print(f"[OD H3-CD] blocks={plan['bounds']} n={n} s={sn:.3f}->{sm:.3f} target={(f0, f1)} loss={loss.item():.4f}", flush=True)
        cd_loss._logged = True
    return loss
