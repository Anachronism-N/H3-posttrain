"""W5 / Causal Forcing stage 1: teacher-forcing multi-step AR teacher on OmniDance (Wan2.2-TI2V-5B audio DiT, LoRA).

Block-causal attention (e6/causal.py; frame 0 = clean first-frame sink, then blocks of `--block` latent frames).
Per step: pick a target block b; blocks < b = clean GT latents (sigma 0), block b = GT noised at sigma (flow
matching), blocks > b = pure noise (invisible to b under block-causal attention). Loss on block b only.
The result is a *multi-step* AR model whose PF-ODE is frame(block)-wise injective -> stage 2 (causal ODE /
consistency distillation) and stage 3 (asymmetric DMD) as in Causal Forcing / Causal Forcing++.

torchrun --nproc_per_node 8 train_ar_teacher.py --out <dir> [--block 4]
"""
import argparse, csv, json, os, random, sys, time

import decord
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dmd"))
sys.path.insert(0, os.path.join(HERE, "..", "e6"))
import dmd_core as C  # noqa: E402
import causal  # noqa: E402

VID = "/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/videos"
CAPS = "/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/metadata_stage5_motioncap.csv"

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--ckpt", default="/dev/shm/od/ckpt/S4-final/step-final.safetensors")
p.add_argument("--data", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/data/dmd/dmd_train.csv")
p.add_argument("--row_offset", type=int, default=4600)
p.add_argument("--height", type=int, default=1280)
p.add_argument("--width", type=int, default=704)
p.add_argument("--num_frames", type=int, default=81)
p.add_argument("--block", type=int, default=4)
p.add_argument("--hist_mode", default="tf", choices=["tf", "mix", "df"],
               help="tf: clean GT history; mix: with prob --df_p history noised at one sigma_h~U(0,--df_max) (TF-major + DF);"
                    " df: every history block noised at an independent sigma~U(0,1) (diffusion-forcing style)")
p.add_argument("--df_p", type=float, default=0.3)
p.add_argument("--df_max", type=float, default=0.2)
p.add_argument("--intended_prompt", type=int, default=0, help="1: prompt = caption + motion caption (stage5 metadata)")
p.add_argument("--rank", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max_iters", type=int, default=3000)
p.add_argument("--cond_mix", default="joint:0.5,audio_only:0.3,text_only:0.2")
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--sample_every", type=int, default=500)
p.add_argument("--sample_steps", type=int, default=20)
p.add_argument("--max_grad_norm", type=float, default=1.0)
p.add_argument("--seed", type=int, default=1111)
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
csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(args.data)))[args.row_offset:]
if args.intended_prompt:
    cap = {os.path.basename(r["video"])[:-4]: r["prompt"] for r in csv.DictReader(open(CAPS))}
    rows = [dict(r, prompt=cap[r["sample_id"]]) for r in rows if r["sample_id"] in cap]
val_row = rows[-1]
rows = rows[:-1]
log(f"{len(rows)} training clips")

time.sleep(local * 8)
pipe, infer_mod = C.build_pipe(args.ckpt, dev)
pipe.dit.to(dev)
C.inject_dual_lora(pipe.dit, args.rank, args.rank)
gen_params = C.lora_params(pipe.dit, "gen")
for q in gen_params:
    q.requires_grad_(True)
causal.enable((args.height // 32) * (args.width // 32), args.block)
causal.STATE["on"] = True
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
rng = random.Random(args.seed + rank * 7919 + start_it)
mix = [(k, float(v)) for k, v in (x.split(":") for x in args.cond_mix.split(","))]
T_LAT = (args.num_frames - 1) // 4 + 1
BOUNDS = causal.block_bounds(T_LAT, args.block)


def pick_cond():
    r, acc = rng.random(), 0.0
    for k, w in mix:
        acc += w
        if r <= acc:
            return k
    return mix[-1][0]


def read_clip(sid):
    vr = decord.VideoReader(f"{VID}/{sid}.mp4")
    fr = vr.get_batch(list(range(min(len(vr), args.num_frames)))).asnumpy()
    return [Image.fromarray(f).resize((args.width, args.height), Image.BICUBIC) for f in fr]


def ext_audio(path):
    return torch.load(path, map_location="cpu", weights_only=True)[..., :args.num_frames].to(dev, torch.bfloat16)


@torch.no_grad()
def encode_video(frames):
    v = pipe.preprocess_video(frames)
    return pipe.vae.encode(v, device=dev, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)).to(device=dev, dtype=pipe.torch_dtype)


def train_step(idx):
    row = rows[idx % len(rows)]
    while not os.path.exists(f"{VID}/{row['sample_id']}.mp4"):
        idx += 7919; row = rows[idx % len(rows)]
    cond = pick_cond()
    frames = read_clip(row["sample_id"])
    ae = None if cond == "text_only" else ext_audio(row["mert_path"])
    prompt = " " if cond == "audio_only" else row["prompt"]
    shared, posi, _ = C.prepare_sample(pipe, prompt, frames[0], ae, args.height, args.width, args.num_frames, seed=rng.randint(0, 2 ** 31))
    z = encode_video(frames)
    b = rng.randint(1, len(BOUNDS) - 1)
    s0, e0 = BOUNDS[b]
    s = C.shift_sigma(rng.uniform(0.02, 0.98))
    eps = torch.randn_like(z)
    x = z.clone()
    x[:, :, s0:e0] = (1 - s) * z[:, :, s0:e0] + s * eps[:, :, s0:e0]
    x[:, :, e0:] = eps[:, :, e0:]
    sf = torch.ones(T_LAT, device=dev); sf[:s0] = 0.0; sf[s0:e0] = s
    sh = 0.0
    if s0 > 1 and (args.hist_mode == "df" or (args.hist_mode == "mix" and rng.random() < args.df_p)):
        for hs, he in BOUNDS[1:b]:   # history blocks (frame 0 = clean sink stays clean)
            sh = rng.uniform(0.0, 1.0) if args.hist_mode == "df" else (sh or rng.uniform(0.0, args.df_max))
            x[:, :, hs:he] = (1 - sh) * z[:, :, hs:he] + sh * eps[:, :, hs:he]
            sf[hs:he] = sh
    causal.STATE["on"] = True
    v = causal.velocity_pf(pipe, shared, posi, x, sf, grad=True, mode="gen")
    loss = ((v[:, :, s0:e0].float() - (eps - z)[:, :, s0:e0].float()) ** 2).mean()
    return loss, dict(cond=cond, b=b, s=round(s, 3), sh=round(sh, 3))


@torch.no_grad()
def sample_ar_teacher(shared, posi, nega, steps, text_scale=2.0, audio_scale=2.0, seed=0):
    """multi-step AR sampling: per block, `steps`-step Euler ODE with 3-branch CFG; context = generated blocks (clean)."""
    i_v, ti_v, tia_v = C.cond_views(posi, nega)
    gen = torch.Generator(device=dev).manual_seed(seed)
    lat = shared["latents"]
    out = torch.zeros_like(lat); out[:, :, 0:1] = shared["first_frame_latents"]
    pipe.scheduler.set_timesteps(steps, denoising_strength=1.0, shift=5.0)
    sig = [float(q) for q in pipe.scheduler.sigmas[:steps]] + [0.0]
    for s0, e0 in BOUNDS[1:]:
        x = torch.randn(lat[:, :, s0:e0].shape, generator=gen, device=dev, dtype=lat.dtype)
        for k in range(steps):
            cur = torch.cat([out[:, :, :s0], x, torch.zeros_like(lat[:, :, e0:])], dim=2)
            sf = torch.ones(lat.shape[2], device=dev); sf[:s0] = 0; sf[s0:e0] = sig[k]
            v_i = causal.velocity_pf(pipe, shared, i_v, cur, sf, mode="gen")
            v_ti = causal.velocity_pf(pipe, shared, ti_v, cur, sf, mode="gen")
            v = v_i + text_scale * (v_ti - v_i)
            if "audio_embeds" in posi:
                asc = infer_mod.compute_text_only_audio_scale(audio_scale, int(k * 50 / steps), 50, hold_steps=5, decay_end_step=10, floor_scale=1.0)
                v = v + asc * (causal.velocity_pf(pipe, shared, tia_v, cur, sf, mode="gen") - v_ti)
            x = x + (sig[k + 1] - sig[k]) * v[:, :, s0:e0]
        out[:, :, s0:e0] = x
    return out


@torch.no_grad()
def save_sample(it):
    """held-out clip, ref = its frame 0 (the inference setting); health = first-frame copy + colour drift + identity proxy."""
    from diffsynth.utils.data import save_video
    row = val_row
    frames = read_clip(row["sample_id"])
    ae = ext_audio(row["mert_path"])
    shared, posi, nega = C.prepare_sample(pipe, row["prompt"], frames[0], ae, args.height, args.width, args.num_frames, seed=0)
    lat = sample_ar_teacher(shared, posi, nega, args.sample_steps)
    video = C.decode(pipe, lat)
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in video])
    ref = np.asarray(frames[0], dtype=np.float32)
    h = dict(iter=it, frames=int(len(arr)), tail_drift=float(np.abs(arr[-16:].mean(axis=(1, 2)) - ref.mean(axis=(0, 1))).mean()),
             drift=float(np.abs(arr.mean(axis=(1, 2)) - ref.mean(axis=(0, 1))).mean()),
             motion=float(np.abs(np.diff(arr[::4], axis=0)).mean()))
    open(os.path.join(args.out, "health.jsonl"), "a").write(json.dumps(h) + "\n")
    log(f"health {h}")
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    save_video(video, os.path.join(args.out, "samples", f"iter_{it:06d}.mp4"), fps=16)


def save_ckpt(it):
    dd = os.path.join(args.out, f"iter_{it:06d}")
    os.makedirs(dd, exist_ok=True)
    torch.save(C.lora_state_dict(pipe.dit, "gen"), f"{dd}/gen_lora.pt")
    torch.save(opt.state_dict(), f"{dd}/opt.pt")
    for o in sorted(x for x in os.listdir(args.out) if x.startswith("iter_"))[:-2]:
        op = os.path.join(args.out, o, "opt.pt")
        if os.path.exists(op):
            os.remove(op)


if start_it == 0 and rank == 0:
    save_sample(0)  # zero-shot: S4-final on the new layout
dist.barrier()
log("start training")
t_last = time.time()
for it in range(start_it, args.max_iters):
    opt.zero_grad(set_to_none=True)
    loss, info = train_step(it * world + rank)
    C.DualLoRALinear.mode = "gen"
    loss.backward()
    for q in gen_params:
        if q.grad is None:
            q.grad = torch.zeros_like(q)
    flat = torch.cat([q.grad.flatten() for q in gen_params]); dist.all_reduce(flat); flat /= world
    off = 0
    for q in gen_params:
        n = q.numel(); q.grad.copy_(flat[off:off + n].view_as(q)); off += n
    gn = torch.nn.utils.clip_grad_norm_(gen_params, args.max_grad_norm)
    opt.step()
    now = time.time()
    st = dict(iter=it, loss=loss.item(), gn=gn.item(), sec=now - t_last, mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, **info)
    t_last = now
    if rank == 0:
        log_f.write(json.dumps(st) + "\n"); log_f.flush()
        if it % 10 == 0:
            log(json.dumps(st))
    done = it + 1
    if done % args.save_every == 0 and rank == 0:
        save_ckpt(done)
    if done % args.sample_every == 0 and rank == 0:
        save_sample(done)
    if done % args.save_every == 0 or done % args.sample_every == 0:
        dist.barrier()
if rank == 0:
    save_ckpt(args.max_iters)
    open(os.path.join(args.out, "DONE"), "w").write(time.strftime("%F %T"))
log("done")
