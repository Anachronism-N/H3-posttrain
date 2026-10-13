"""E6-3: Self-Forcing DMD for the block-causal OmniDance student.

Generator = S4-final + gen LoRA (init from E6-2 causal ODE ckpt), block-causal attention, AR rollout block by block
(4 steps/block, context re-encoded clean). Each block exits at a random step k_b; the exit step is recomputed with grad.
Real score = bidirectional S4-final teacher (3-branch CFG); fake score = bidirectional fake LoRA.
DMD grad g is computed once on the full rolled-out video; loss is separable over blocks, so each block's exit step is
back-propagated separately (memory of one forward).
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dmd"))
sys.path.insert(0, HERE)
import dmd_core as C  # noqa: E402
import causal  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--ckpt", default="/dev/shm/od/ckpt/S4-final/step-final.safetensors")
p.add_argument("--init_gen", required=True, help="gen LoRA from E6-2 (gen_lora_ema.pt)")
p.add_argument("--block", type=int, default=4)
p.add_argument("--ode_sigmas", default="", help="comma list; default = from init run args.json")
p.add_argument("--data", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/data/dmd/dmd_train.csv")
p.add_argument("--height", type=int, default=1280)
p.add_argument("--width", type=int, default=704)
p.add_argument("--num_frames", type=int, default=81)
p.add_argument("--rank", type=int, default=128)
p.add_argument("--lr_gen", type=float, default=2e-6)
p.add_argument("--lr_fake", type=float, default=2e-6)
p.add_argument("--gen_every", type=int, default=5, help="generator update every N critic updates")
p.add_argument("--max_iters", type=int, default=4000)
p.add_argument("--text_scale", type=float, default=2.0)
p.add_argument("--audio_peak", type=float, default=2.0)
p.add_argument("--audio_floor", type=float, default=1.0)
p.add_argument("--no_audio_cfg", action="store_true", help="E5c-a: teacher audio scale fixed at 1 (no audio guidance)")
p.add_argument("--cond_mix", default="joint:0.6,audio_only:0.2,text_only:0.2")
p.add_argument("--save_every", type=int, default=250)
p.add_argument("--sample_every", type=int, default=250)
p.add_argument("--max_grad_norm", type=float, default=1.0)
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step (both nets)")
p.add_argument("--ema", type=float, default=0.0, help="EMA decay for gen LoRA (0=off); samples/ckpt use EMA if on")
p.add_argument("--ema_start", type=int, default=100)
args = p.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = f"cuda:{local}"
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(args.out, exist_ok=True)
log_f = open(os.path.join(args.out, "train_log.jsonl"), "a") if rank == 0 else None


def log(msg):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if rank == 0:
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)

# stagger loading to avoid RAM spikes
time.sleep(local * 8)
pipe, infer_mod = C.build_pipe(args.ckpt, dev)
pipe.dit.to(dev)
names = C.inject_dual_lora(pipe.dit, args.rank, args.rank)
log(f"dual LoRA injected into {len(names)} linears")
gen_params, fake_params = C.lora_params(pipe.dit, "gen"), C.lora_params(pipe.dit, "fake")
for q in gen_params + fake_params:
    q.requires_grad_(True)
log(f"gen params {sum(q.numel() for q in gen_params)/1e6:.1f}M fake params {sum(q.numel() for q in fake_params)/1e6:.1f}M")

if not [d for d in os.listdir(args.out) if d.startswith("iter_")]:
    C.load_lora_state(pipe.dit, torch.load(args.init_gen, map_location="cpu"))
    log(f"gen LoRA initialised from {args.init_gen}")
if args.ode_sigmas:
    SIG = [float(x) for x in args.ode_sigmas.split(",")]
else:
    SIG = json.load(open(os.path.join(os.path.dirname(os.path.dirname(args.init_gen)), "args.json")))["ode_sigmas"]
log(f"student sigmas {SIG}")
TPF = (args.height // 32) * (args.width // 32)
T_LAT = (args.num_frames - 1) // 4 + 1
BOUNDS = causal.block_bounds(T_LAT, args.block)
causal.enable(TPF, args.block); causal.disable()

# resume
start_it = 0
ck = sorted([d for d in os.listdir(args.out) if d.startswith("iter_")])
opt_g = torch.optim.AdamW(gen_params, lr=args.lr_gen, betas=(0.0, 0.999), weight_decay=0.01)
opt_f = torch.optim.AdamW(fake_params, lr=args.lr_fake, betas=(0.0, 0.999), weight_decay=0.01)
if ck:
    last = os.path.join(args.out, ck[-1])
    C.load_lora_state(pipe.dit, torch.load(f"{last}/gen_lora.pt"))
    C.load_lora_state(pipe.dit, torch.load(f"{last}/fake_lora.pt"))
    if os.path.exists(f"{last}/opt.pt"):
        o = torch.load(f"{last}/opt.pt", map_location="cpu")
        opt_g.load_state_dict(o["g"]); opt_f.load_state_dict(o["f"])
    start_it = int(ck[-1].split("_")[1])
    log(f"resumed from {last}")
# sync initial LoRA (A init is random per rank)
for q in gen_params + fake_params:
    dist.broadcast(q.data, 0)

ema = None
if args.ema > 0:
    ema = [q.detach().clone() for q in gen_params]
    if ck and os.path.exists(os.path.join(args.out, ck[-1], "gen_lora_ema.pt")):
        sd = torch.load(os.path.join(args.out, ck[-1], "gen_lora_ema.pt"))
        names_g = [n for n, q in pipe.dit.named_parameters() if ".lora.gen." in n]
        ema = [sd[n].to(dev) for n in names_g]


def swap_ema():
    """swap live gen params with EMA copy (call twice to restore)."""
    if ema is None:
        return
    for q, e in zip(gen_params, ema):
        tmp = q.data.clone(); q.data.copy_(e); e.copy_(tmp)


sig_tab, sc_tab = C.audio_scale_table(args.audio_peak, args.audio_floor, infer_mod=infer_mod)
sig_tab, sc_tab = sig_tab.to(dev), sc_tab.to(dev)


def audio_scale_at(s):
    if args.no_audio_cfg:
        return 1.0
    return float(sc_tab[(sig_tab - s).abs().argmin()])


csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(args.data)))
rng = random.Random(args.seed + rank * 7919 + start_it)
mix = [(k, float(v)) for k, v in (x.split(":") for x in args.cond_mix.split(","))]


def pick_cond():
    r, acc = rng.random(), 0.0
    for k, w in mix:
        acc += w
        if r <= acc:
            return k
    return mix[-1][0]


def load_batch(it):
    row = rows[(it * world + rank) % len(rows)]
    cond = pick_cond()
    img = Image.open(row["image_path"]).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
    ae = None
    if cond != "text_only":
        ae = torch.load(row["mert_path"], map_location="cpu", weights_only=True)[..., :args.num_frames].to(dev, torch.bfloat16)
    prompt = " " if cond == "audio_only" else row["prompt"]
    shared, posi, nega = C.prepare_sample(pipe, prompt, img, ae, args.height, args.width, args.num_frames, seed=rng.randint(0, 2 ** 31))
    return cond, shared, posi, nega


def allreduce_grads(params):
    for q in params:
        if q.grad is None:
            q.grad = torch.zeros_like(q)
    flat = torch.cat([q.grad.flatten() for q in params])
    dist.all_reduce(flat)
    flat /= world
    off = 0
    for q in params:
        n = q.numel(); q.grad.copy_(flat[off:off + n].view_as(q)); off += n


def sample_timestep():
    u = random.uniform(0.02, 0.98)
    return C.shift_sigma(u)


def real_x0(shared, posi, nega, x_t, s, cond):
    i, ti, tia = C.cond_views(posi, nega)
    v_i = C.velocity(pipe, shared, i, x_t, s, "real")
    v_ti = C.velocity(pipe, shared, ti, x_t, s, "real")
    v = v_i + args.text_scale * (v_ti - v_i)
    if "audio_embeds" in posi:
        v_tia = C.velocity(pipe, shared, tia, x_t, s, "real")
        v = v + audio_scale_at(s) * (v_tia - v_ti)
    return x_t - s * v


@torch.no_grad()
def ar_rollout(shared, posi, gen=None):
    """AR rollout; returns full x0 video (no grad) + per-block (exit k, exit input x, sigma-vector) for re-forward."""
    lat = shared["latents"]
    out = torch.zeros_like(lat)
    out[:, :, 0:1] = shared["first_frame_latents"]
    exits = []
    causal.STATE["on"] = True
    for s0, e0 in BOUNDS[1:]:
        kx = random.randint(0, len(SIG) - 1)
        x = torch.randn(lat[:, :, s0:e0].shape, generator=gen, device=lat.device, dtype=lat.dtype)
        for k, sg in enumerate(SIG):
            cur = torch.cat([out[:, :, :s0], x, torch.zeros_like(lat[:, :, e0:])], dim=2)
            sf = torch.ones(T_LAT, device=lat.device); sf[:s0] = 0.0; sf[s0:e0] = sg
            v = causal.velocity_pf(pipe, shared, posi, cur, sf, mode="gen")
            x0 = (cur - sg * v)[:, :, s0:e0]
            if k == kx:
                exits.append((s0, e0, cur.clone(), sf.clone(), sg))
                x = x0
                break
            sn = SIG[k + 1]
            x = (1 - sn) * x0 + sn * torch.randn(x0.shape, generator=gen, device=x0.device, dtype=x0.dtype)
        out[:, :, s0:e0] = x
    causal.STATE["on"] = False
    return out, exits


def generator_step(shared, posi, nega, cond, accum):
    x0_gen, exits = ar_rollout(shared, posi)
    with torch.no_grad():
        s = sample_timestep()
        eps = torch.randn_like(x0_gen)
        x_t = (1 - s) * x0_gen + s * eps
        x0_real = real_x0(shared, posi, nega, x_t, s, cond)
        x0_fake = x_t - s * C.velocity(pipe, shared, posi, x_t, s, "fake")
        g = x0_fake - x0_real
        norm = (x0_gen - x0_real)[:, :, 1:].abs().mean().clamp_min(1e-6)
        g = torch.nan_to_num(g / norm)
    tot = 0.0
    n_lat = T_LAT - 1
    for s0, e0, cur, sf, sg in exits:
        causal.STATE["on"] = True
        v = causal.velocity_pf(pipe, shared, posi, cur, sf, grad=True, mode="gen")
        x0b = (cur - sg * v)[:, :, s0:e0]
        target = (x0b - g[:, :, s0:e0]).detach()
        # same normalisation as full-video MSE: sum over blocks == mean over all generated frames
        lb = 0.5 * F.mse_loss(x0b.float(), target.float()) * (e0 - s0) / n_lat
        C.DualLoRALinear.mode = "gen"
        causal.STATE["on"] = True  # recompute under checkpointing must see causal attention
        (lb / accum).backward()
        causal.STATE["on"] = False
        tot += lb.item()
    return tot, {"dmd_grad": g[:, :, 1:].abs().mean().item(), "t": s}


def critic_step(shared, posi):
    x0_gen, _ = ar_rollout(shared, posi)
    s = sample_timestep()
    eps = torch.randn_like(x0_gen)
    x_t = (1 - s) * x0_gen + s * eps
    v = C.velocity(pipe, shared, posi, x_t, s, "fake", grad=True)
    loss = F.mse_loss(v[:, :, 1:].float(), (eps - x0_gen)[:, :, 1:].float())
    return loss


val_row = rows[-1]


@torch.no_grad()
def save_sample(it):
    from diffsynth.utils.data import save_video
    img = Image.open(val_row["image_path"]).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
    ae = torch.load(val_row["mert_path"], map_location="cpu", weights_only=True)[..., :args.num_frames].to(dev, torch.bfloat16)
    shared, posi, _ = C.prepare_sample(pipe, val_row["prompt"], img, ae, args.height, args.width, args.num_frames, seed=0)
    causal.STATE["on"] = True
    lat = causal.ar_sample(pipe, shared, posi, block=args.block, sigmas=SIG, seed=0)
    causal.STATE["on"] = False
    video = C.decode(pipe, lat)
    import numpy as np
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in video])
    def hf(a):
        return float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=2)).mean())
    health = dict(iter=it, hf_ratio=hf(arr[1:]) / max(hf(arr[:1]), 1e-6),
                  tail_drift=float(np.abs(arr[-16:].mean(axis=(1, 2)) - arr[:1].mean(axis=(1, 2))).mean()),
                  sat=float(arr[1:].std(axis=-1).mean() / max(arr[:1].std(axis=-1).mean(), 1e-6)),
                  drift=float(np.abs(arr[1:].mean(axis=(1, 2)) - arr[:1].mean(axis=(1, 2))).mean()))
    open(os.path.join(args.out, "health.jsonl"), "a").write(json.dumps(health) + "\n")
    log(f"health {health}")
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    save_video(video, os.path.join(args.out, "samples", f"iter_{it:06d}.mp4"), fps=16)


def save_ckpt(it):
    d = os.path.join(args.out, f"iter_{it:06d}")
    os.makedirs(d, exist_ok=True)
    torch.save(C.lora_state_dict(pipe.dit, "gen"), f"{d}/gen_lora.pt")
    if ema is not None:
        swap_ema(); torch.save(C.lora_state_dict(pipe.dit, "gen"), f"{d}/gen_lora_ema.pt"); swap_ema()
    torch.save(C.lora_state_dict(pipe.dit, "fake"), f"{d}/fake_lora.pt")
    torch.save({"g": opt_g.state_dict(), "f": opt_f.state_dict()}, f"{d}/opt.pt")
    # keep only the latest 3 optimizer states
    olds = sorted(x for x in os.listdir(args.out) if x.startswith("iter_"))[:-3]
    for o in olds:
        op = os.path.join(args.out, o, "opt.pt")
        if os.path.exists(op):
            os.remove(op)


if start_it == 0 and rank == 0:
    swap_ema(); save_sample(0); swap_ema()
dist.barrier()
log("start training")
t_last = time.time()
for it in range(start_it, args.max_iters):
    stats = {"iter": it}
    if it % args.gen_every == 0:
        opt_g.zero_grad(set_to_none=True)
        acc = {}
        for a in range(args.accum):
            cond, shared, posi, nega = load_batch(it * args.accum + a)
            lg, info = generator_step(shared, posi, nega, cond, args.accum)
            acc["gen_loss"] = acc.get("gen_loss", 0) + lg / args.accum
            acc["dmd_grad"] = acc.get("dmd_grad", 0) + info["dmd_grad"] / args.accum
        allreduce_grads(gen_params)
        gn = torch.nn.utils.clip_grad_norm_(gen_params, args.max_grad_norm)
        opt_g.step()
        if ema is not None:
            d = args.ema if it >= args.ema_start else 0.0
            with torch.no_grad():
                for q, e in zip(gen_params, ema):
                    e.mul_(d).add_(q.data, alpha=1 - d)
        stats.update(gen_gn=gn.item(), **acc)
    opt_f.zero_grad(set_to_none=True)
    lfs = 0.0
    for a in range(args.accum):
        cond, shared, posi, nega = load_batch(10 ** 7 + it * args.accum + a)
        lf = critic_step(shared, posi)
        C.DualLoRALinear.mode = "fake"
        (lf / args.accum).backward()
        lfs += lf.item() / args.accum
    allreduce_grads(fake_params)
    fn = torch.nn.utils.clip_grad_norm_(fake_params, args.max_grad_norm)
    opt_f.step()
    stats.update(cond=cond, fake_loss=lfs, fake_gn=fn.item())
    now = time.time()
    stats["sec"] = now - t_last
    t_last = now
    stats["mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    if rank == 0:
        log_f.write(json.dumps(stats) + "\n"); log_f.flush()
        if it % 10 == 0:
            log(json.dumps(stats))
    done = it + 1
    if done % args.save_every == 0 and rank == 0:
        save_ckpt(done)
    if done % args.sample_every == 0 and rank == 0:
        swap_ema(); save_sample(done); swap_ema()
    if done % args.save_every == 0 or done % args.sample_every == 0:
        dist.barrier()
if rank == 0:
    save_ckpt(args.max_iters)
    swap_ema(); save_sample(args.max_iters); swap_ema()
    open(os.path.join(args.out, "DONE"), "w").write("ok")
dist.barrier()
log("finished")
