"""E6-2: causal ODE-regression init (CausVid / OmniForcing stage 1) for the OmniDance audio DiT.

Student = S4-final + gen LoRA, block-causal self-attention (sink frame + blocks of `--block` latent frames).
Data = teacher ODE trajectories (E6-1): latents at 4 student sigmas + final x0.
Each step: every block independently picks a trajectory point k in {0..3} (diffusion-forcing style); with prob
`--clean_ctx_p` a random prefix of blocks is set clean (sigma 0, as at AR inference). Loss = MSE(x0_pred, x0_teacher)
on the noisy blocks.

torchrun --nproc_per_node 8 train_causal_ode.py --out <dir> --data /dev/shm/od/ode
"""
import argparse, glob, json, os, random, sys, time

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
p.add_argument("--data", default="/dev/shm/od/ode")
p.add_argument("--height", type=int, default=1280)
p.add_argument("--width", type=int, default=704)
p.add_argument("--num_frames", type=int, default=81)
p.add_argument("--block", type=int, default=4)
p.add_argument("--rank", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max_iters", type=int, default=3000)
p.add_argument("--accum", type=int, default=1)
p.add_argument("--clean_ctx_p", type=float, default=0.3)
p.add_argument("--save_every", type=int, default=250)
p.add_argument("--sample_every", type=int, default=250)
p.add_argument("--max_grad_norm", type=float, default=1.0)
p.add_argument("--ema", type=float, default=0.99)
p.add_argument("--seed", type=int, default=1234)
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


files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
val_files = files[-4:]
files = files[:-4]
probe = torch.load(files[0], map_location="cpu")
ODE_SIG = [float(x) for x in probe["sigmas"][:4]]
if rank == 0:
    json.dump(dict(vars(args), ode_sigmas=ODE_SIG, n_files=len(files)), open(os.path.join(args.out, "args.json"), "w"), indent=1)
log(f"{len(files)} ODE files, sigmas {ODE_SIG}")

time.sleep(local * 8)
pipe, infer_mod = C.build_pipe(args.ckpt, dev)
pipe.dit.to(dev)
C.inject_dual_lora(pipe.dit, args.rank, args.rank)
gen_params = C.lora_params(pipe.dit, "gen")
for q in gen_params:
    q.requires_grad_(True)
for q in C.lora_params(pipe.dit, "fake"):
    q.requires_grad_(False)
log(f"gen LoRA params {sum(q.numel() for q in gen_params) / 1e6:.1f}M")
TPF = (args.height // 32) * (args.width // 32)
causal.enable(TPF, args.block)
T_LAT = (args.num_frames - 1) // 4 + 1
BOUNDS = causal.block_bounds(T_LAT, args.block)
log(f"block-causal: tpf={TPF} T={T_LAT} blocks={BOUNDS}")

opt = torch.optim.AdamW(gen_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
start_it = 0
ck = sorted(d for d in os.listdir(args.out) if d.startswith("iter_"))
if ck:
    last = os.path.join(args.out, ck[-1])
    C.load_lora_state(pipe.dit, torch.load(f"{last}/gen_lora.pt"))
    if os.path.exists(f"{last}/opt.pt"):
        opt.load_state_dict(torch.load(f"{last}/opt.pt", map_location="cpu"))
    start_it = int(ck[-1].split("_")[1])
    log(f"resumed from {last}")
for q in gen_params:
    dist.broadcast(q.data, 0)
ema = [q.detach().clone() for q in gen_params]
if ck and os.path.exists(os.path.join(args.out, ck[-1], "gen_lora_ema.pt")):
    sd = torch.load(os.path.join(args.out, ck[-1], "gen_lora_ema.pt"))
    names_g = [n for n, q in pipe.dit.named_parameters() if ".lora.gen." in n]
    ema = [sd[n].to(dev) for n in names_g]


def swap_ema():
    for q, e in zip(gen_params, ema):
        tmp = q.data.clone(); q.data.copy_(e); e.copy_(tmp)


rng = random.Random(args.seed + rank * 7919 + start_it)


def cond_inputs(d, seed):
    img = Image.open(d["image_path"]).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
    ae = None
    if d["cond"] != "text_only":
        ae = torch.load(d["mert_path"], map_location="cpu", weights_only=True)[..., :args.num_frames].to(dev, torch.bfloat16)
    return C.prepare_sample(pipe, d["prompt"], img, ae, args.height, args.width, args.num_frames, seed=seed)


def ode_step(idx):
    d = torch.load(files[idx % len(files)], map_location="cpu")
    lat = d["latents"].to(dev)  # [5,C,T,H,W]
    shared, posi, _ = cond_inputs(d, seed=rng.randint(0, 2 ** 31))
    nb = len(BOUNDS) - 1
    ks = [rng.randint(0, 3) for _ in range(nb)]
    if rng.random() < args.clean_ctx_p:
        j = rng.randint(1, nb - 1)
        for b in range(j):
            ks[b] = 4
    x = lat[4:5].clone()
    sf = torch.zeros(T_LAT, device=dev)
    mask = torch.zeros(T_LAT, device=dev)
    for b, (s, e) in enumerate(BOUNDS[1:]):
        x[:, :, s:e] = lat[ks[b]:ks[b] + 1, :, s:e]
        sig = 0.0 if ks[b] == 4 else ODE_SIG[ks[b]]
        sf[s:e] = sig
        mask[s:e] = 0.0 if ks[b] == 4 else 1.0
    v = causal.velocity_pf(pipe, shared, posi, x, sf, grad=True, mode="gen")
    x0 = x - sf.view(1, 1, -1, 1, 1).to(x.dtype) * v
    err = (x0.float() - lat[4:5].float()) ** 2
    per_frame = err.mean(dim=(0, 1, 3, 4))
    loss = (per_frame * mask).sum() / mask.sum().clamp_min(1)
    return loss, d["cond"], ks


@torch.no_grad()
def save_sample(it):
    from diffsynth.utils.data import save_video
    import numpy as np
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    errs = []
    for vi, vf in enumerate(val_files[:2]):
        d = torch.load(vf, map_location="cpu")
        shared, posi, _ = cond_inputs(d, seed=0)
        lat = causal.ar_sample(pipe, shared, posi, block=args.block, sigmas=ODE_SIG, seed=0)
        errs.append(float(((lat[:, :, 1:].float().cpu() - d["latents"][4:5, :, 1:].float()) ** 2).mean()))
        if vi == 0:
            video = C.decode(pipe, lat)
            arr = np.stack([np.asarray(f, dtype=np.float32) for f in video])
            health = dict(iter=it, sat=float(arr[1:].std(axis=-1).mean() / max(arr[:1].std(axis=-1).mean(), 1e-6)),
                          drift=float(np.abs(arr[1:].mean(axis=(1, 2)) - arr[:1].mean(axis=(1, 2))).mean()),
                          tail_drift=float(np.abs(arr[-16:].mean(axis=(1, 2)) - arr[:1].mean(axis=(1, 2))).mean()))
            save_video(video, os.path.join(args.out, "samples", f"iter_{it:06d}.mp4"), fps=16)
    health["ar_mse_vs_teacher"] = sum(errs) / len(errs)
    open(os.path.join(args.out, "health.jsonl"), "a").write(json.dumps(health) + "\n")
    log(f"health {health}")


def save_ckpt(it):
    dd = os.path.join(args.out, f"iter_{it:06d}")
    os.makedirs(dd, exist_ok=True)
    torch.save(C.lora_state_dict(pipe.dit, "gen"), f"{dd}/gen_lora.pt")
    swap_ema(); torch.save(C.lora_state_dict(pipe.dit, "gen"), f"{dd}/gen_lora_ema.pt"); swap_ema()
    torch.save(opt.state_dict(), f"{dd}/opt.pt")
    for o in sorted(x for x in os.listdir(args.out) if x.startswith("iter_"))[:-2]:
        op = os.path.join(args.out, o, "opt.pt")
        if os.path.exists(op):
            os.remove(op)


if start_it == 0 and rank == 0:
    save_sample(0)  # zero-shot causal masking of the bidirectional teacher (reference point)
dist.barrier()
log("start training")
t_last = time.time()
for it in range(start_it, args.max_iters):
    opt.zero_grad(set_to_none=True)
    ls = 0.0
    for a in range(args.accum):
        loss, cond, ks = ode_step((it * args.accum + a) * world + rank)
        C.DualLoRALinear.mode = "gen"
        (loss / args.accum).backward()
        ls += loss.item() / args.accum
    for q in gen_params:
        if q.grad is None:
            q.grad = torch.zeros_like(q)
    flat = torch.cat([q.grad.flatten() for q in gen_params]); dist.all_reduce(flat); flat /= world
    off = 0
    for q in gen_params:
        n = q.numel(); q.grad.copy_(flat[off:off + n].view_as(q)); off += n
    gn = torch.nn.utils.clip_grad_norm_(gen_params, args.max_grad_norm)
    opt.step()
    with torch.no_grad():
        dcy = args.ema if it >= 100 else 0.0
        for q, e in zip(gen_params, ema):
            e.mul_(dcy).add_(q.data, alpha=1 - dcy)
    now = time.time()
    st = dict(iter=it, loss=ls, gn=gn.item(), cond=cond, ks=ks, sec=now - t_last, mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    t_last = now
    if rank == 0:
        log_f.write(json.dumps(st) + "\n"); log_f.flush()
        if it % 10 == 0:
            log(json.dumps(st))
    done = it + 1
    if done % args.save_every == 0 and rank == 0:
        save_ckpt(done)
    if done % args.sample_every == 0 and rank == 0:
        swap_ema(); save_sample(done); swap_ema()
    if done % args.save_every == 0 or done % args.sample_every == 0:
        dist.barrier()
if rank == 0:
    save_ckpt(args.max_iters)
    open(os.path.join(args.out, "DONE"), "w").write("ok")
dist.barrier()
log("finished")
