"""E11: reference/first-frame decoupled SFT ((R+I)2V-style) on S4-final, LoRA.

Diagnosis (RESULTS.md, motion-factor analysis): motion is set by the reference image (= clip frame 0) + noise;
text/music barely matter. Here the clip's first frame is NOT given. Layout (latent T = 1 + 21 = 22):
  latent 0      = clean reference latent (single image, timestep 0) from a frame at a random time in the clip
                  (--ref_lo..--ref_hi), optionally augmented (scale/shift, flip, colour);
  latents 1..21 = the whole 81-frame clip (incl. its first frame), flow matching at one sigma, loss on all 21.
Audio: 4 replicated frames are prepended so the windowed audio encoder yields 22 anchors (anchor k+1 == old anchor k).
Attention stays bidirectional (per-frame timesteps via e6/causal.py patch only).

torchrun --nproc_per_node 8 train_e11.py --out <dir> [--ref_lo 40 --ref_hi 80 --aug 1]
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
p.add_argument("--ref_lo", type=int, default=40)
p.add_argument("--ref_hi", type=int, default=80)
p.add_argument("--aug", type=int, default=1, help="0 none, 1 scale/shift, 2 = 1 + hflip + colour jitter")
p.add_argument("--intended_prompt", type=int, default=0, help="1: prompt = caption + motion caption (stage5 metadata)")
p.add_argument("--rank", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max_iters", type=int, default=2500)
p.add_argument("--cond_mix", default="joint:0.5,audio_only:0.3,text_only:0.2")
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--sample_every", type=int, default=500)
p.add_argument("--sample_steps", type=int, default=30)
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
causal.disable()
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
NF_EXT = args.num_frames + 4
T_EXT = (NF_EXT - 1) // 4 + 1  # 22


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


def make_ref(img, r=None):
    r = r or rng
    if args.aug >= 1:
        s = r.uniform(0.85, 1.0)
        W, H = img.size
        w2, h2 = int(W * s), int(H * s)
        x0, y0 = r.randint(0, W - w2), r.randint(0, H - h2)
        img = img.crop((x0, y0, x0 + w2, y0 + h2)).resize((W, H), Image.BICUBIC)
    if args.aug >= 2:
        if r.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = ImageEnhance.Brightness(img).enhance(r.uniform(0.9, 1.1))
        img = ImageEnhance.Color(img).enhance(r.uniform(0.9, 1.1))
    return img


def ext_audio(path):
    a = torch.load(path, map_location="cpu", weights_only=True)[..., :args.num_frames]
    a = torch.cat([a[..., :1].expand(*a.shape[:-1], 4), a], dim=-1)
    return a.to(dev, torch.bfloat16)


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
    ref = make_ref(frames[rng.randint(args.ref_lo, min(args.ref_hi, len(frames) - 1))])
    ae = None if cond == "text_only" else ext_audio(row["mert_path"])
    prompt = " " if cond == "audio_only" else row["prompt"]
    shared, posi, _ = C.prepare_sample(pipe, prompt, ref, ae, args.height, args.width, NF_EXT, seed=rng.randint(0, 2 ** 31))
    z = encode_video(frames)  # [1,C,21,h,w]
    s = C.shift_sigma(rng.uniform(0.02, 0.98))
    eps = torch.randn_like(z)
    xt = (1 - s) * z + s * eps
    x = torch.cat([shared["first_frame_latents"].to(device=dev, dtype=z.dtype), xt], dim=2)
    sf = torch.full((T_EXT,), s, device=dev); sf[0] = 0.0
    v = causal.velocity_pf(pipe, shared, posi, x, sf, grad=True, mode="gen")
    loss = ((v[:, :, 1:].float() - (eps - z).float()) ** 2).mean()
    return loss, dict(cond=cond, s=round(s, 3))


@torch.no_grad()
def sample_e11(shared, posi, nega, steps, text_scale=2.0, audio_scale=2.0, seed=0):
    """3-branch CFG (same schedule as the teacher) on the 22-latent layout; returns clip latents [1,C,21,h,w]."""
    i_v, ti_v, tia_v = C.cond_views(posi, nega)
    gen = torch.Generator(device=dev).manual_seed(seed)
    lat = shared["latents"]
    x = torch.randn(lat.shape, generator=gen, device=dev, dtype=lat.dtype)
    pipe.scheduler.set_timesteps(steps, denoising_strength=1.0, shift=5.0)
    sig = [float(q) for q in pipe.scheduler.sigmas[:steps]] + [0.0]
    for k in range(steps):
        sf = torch.full((lat.shape[2],), sig[k], device=dev); sf[0] = 0
        v_i = causal.velocity_pf(pipe, shared, i_v, x, sf, mode="gen")
        v_ti = causal.velocity_pf(pipe, shared, ti_v, x, sf, mode="gen")
        v = v_i + text_scale * (v_ti - v_i)
        if "audio_embeds" in posi:
            asc = infer_mod.compute_text_only_audio_scale(audio_scale, int(k * 50 / steps), 50, hold_steps=5, decay_end_step=10, floor_scale=1.0)
            v = v + asc * (causal.velocity_pf(pipe, shared, tia_v, x, sf, mode="gen") - v_ti)
        x = x + (sig[k + 1] - sig[k]) * v
    return x[:, :, 1:]


@torch.no_grad()
def save_sample(it):
    """held-out clip, ref = its frame 0 (the inference setting); health = first-frame copy + colour drift + identity proxy."""
    from diffsynth.utils.data import save_video
    row = val_row
    frames = read_clip(row["sample_id"])
    ae = ext_audio(row["mert_path"])
    shared, posi, nega = C.prepare_sample(pipe, row["prompt"], frames[0], ae, args.height, args.width, NF_EXT, seed=0)
    lat = sample_e11(shared, posi, nega, args.sample_steps)
    video = C.decode(pipe, lat)
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in video])
    ref = np.asarray(frames[0], dtype=np.float32)
    h = dict(iter=it, frames=int(len(arr)), copy_l1=float(np.abs(arr[0] - ref).mean()),
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
