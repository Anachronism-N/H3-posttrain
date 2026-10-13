"""W5 / Causal Forcing++ stage 2: causal consistency distillation (CD) from the frozen multi-step AR teacher (stage 1).

Teacher = S4-final + AR-teacher LoRA (loaded into the frozen "fake" slot); student = "gen" slot, initialised from the
same LoRA; EMA copy of the student gives the consistency target. Block-causal attention, clean GT history (teacher
forcing context). Per step: block b, teacher-grid index n (50-step shifted schedule):
  x_n = (1-s_n) z + s_n eps on block b;  x_{n-1} = x_n + (s_{n-1}-s_n) * v_teacher^CFG(x_n)   (one ODE step, 3-branch CFG)
  loss = || f_student(x_n, s_n) - sg(f_ema(x_{n-1}, s_{n-1})) ||^2 on block b, f(x,s) = x - s*v;  f(x, 0) := x.
Sampling: 4-step multistep consistency AR sampler (e6/causal.ar_sample) with the student sigmas.

torchrun --nproc_per_node 8 train_causal_cd.py --out <dir> --teacher_lora <W5 iter dir>/gen_lora.pt
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
p.add_argument("--teacher_lora", required=True)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--teacher_steps", type=int, default=50)
p.add_argument("--text_scale", type=float, default=2.0)
p.add_argument("--audio_scale", type=float, default=2.0)
p.add_argument("--intended_prompt", type=int, default=0, help="1: prompt = caption + motion caption (stage5 metadata)")
p.add_argument("--rank", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--max_iters", type=int, default=2000)
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
_t = torch.load(args.teacher_lora, map_location="cpu")
C.load_lora_state(pipe.dit, _t)                                              # gen <- teacher (student init)
C.load_lora_state(pipe.dit, {k.replace(".lora.gen.", ".lora.fake."): v for k, v in _t.items()})  # fake <- teacher (frozen)
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
ema = [q.detach().clone() for q in gen_params]
if ck and os.path.exists(os.path.join(args.out, ck[-1], "gen_lora_ema.pt")):
    _e = torch.load(os.path.join(args.out, ck[-1], "gen_lora_ema.pt"), map_location="cpu")
    _names = [n for n, q in pipe.dit.named_parameters() if ".lora.gen." in n]
    ema = [_e[n].to(dev) for n in _names]


def swap_ema():
    for q, e in zip(gen_params, ema):
        t = q.data.clone(); q.data.copy_(e); e.copy_(t)


pipe.scheduler.set_timesteps(args.teacher_steps, denoising_strength=1.0, shift=5.0)
TSIG = [float(q) for q in pipe.scheduler.sigmas[:args.teacher_steps]] + [0.0]
SSIG = C.STUDENT_SIGMAS
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


def v_cfg(shared, posi, nega, x, sf, mode):
    i_v, ti_v, tia_v = C.cond_views(posi, nega)
    v_i = causal.velocity_pf(pipe, shared, i_v, x, sf, mode=mode)
    v_ti = causal.velocity_pf(pipe, shared, ti_v, x, sf, mode=mode)
    v = v_i + args.text_scale * (v_ti - v_i)
    if "audio_embeds" in posi:
        v = v + args.audio_scale * (causal.velocity_pf(pipe, shared, tia_v, x, sf, mode=mode) - v_ti)
    return v


def train_step(idx):
    row = rows[idx % len(rows)]
    while not os.path.exists(f"{VID}/{row['sample_id']}.mp4"):
        idx += 7919; row = rows[idx % len(rows)]
    cond = pick_cond()
    frames = read_clip(row["sample_id"])
    ae = None if cond == "text_only" else ext_audio(row["mert_path"])
    prompt = " " if cond == "audio_only" else row["prompt"]
    shared, posi, nega = C.prepare_sample(pipe, prompt, frames[0], ae, args.height, args.width, args.num_frames, seed=rng.randint(0, 2 ** 31))
    z = encode_video(frames)
    b = rng.randint(1, len(BOUNDS) - 1)
    s0, e0 = BOUNDS[b]
    n = rng.randint(0, args.teacher_steps - 1)
    sn, sm = TSIG[n], TSIG[n + 1]
    eps = torch.randn_like(z)
    x = z.clone(); x[:, :, e0:] = eps[:, :, e0:]
    x[:, :, s0:e0] = (1 - sn) * z[:, :, s0:e0] + sn * eps[:, :, s0:e0]
    sf = torch.ones(T_LAT, device=dev); sf[:s0] = 0.0
    causal.STATE["on"] = True
    with torch.no_grad():
        sf[s0:e0] = sn
        vt = v_cfg(shared, posi, nega, x, sf, "fake")
        xm = x.clone(); xm[:, :, s0:e0] = x[:, :, s0:e0] + (sm - sn) * vt[:, :, s0:e0]
        if sm <= 0:
            target = xm[:, :, s0:e0]
        else:
            swap_ema()
            sf[s0:e0] = sm
            ve = causal.velocity_pf(pipe, shared, posi, xm, sf, mode="gen")
            swap_ema()
            target = xm[:, :, s0:e0] - sm * ve[:, :, s0:e0]
    sf[s0:e0] = sn
    v = causal.velocity_pf(pipe, shared, posi, x, sf, grad=True, mode="gen")
    pred = x[:, :, s0:e0] - sn * v[:, :, s0:e0]
    loss = ((pred.float() - target.float()) ** 2).mean()
    return loss, dict(cond=cond, b=b, n=n, s=round(sn, 3))


@torch.no_grad()
def save_sample(it):
    """held-out clip, ref = its frame 0 (the inference setting); health = first-frame copy + colour drift + identity proxy."""
    from diffsynth.utils.data import save_video
    row = val_row
    frames = read_clip(row["sample_id"])
    ae = ext_audio(row["mert_path"])
    shared, posi, nega = C.prepare_sample(pipe, row["prompt"], frames[0], ae, args.height, args.width, args.num_frames, seed=0)
    swap_ema()
    causal.STATE["on"] = True
    lat = causal.ar_sample(pipe, shared, posi, block=args.block, sigmas=SSIG, seed=0)
    swap_ema()
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
    swap_ema(); torch.save(C.lora_state_dict(pipe.dit, "gen"), f"{dd}/gen_lora_ema.pt"); swap_ema()
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
    with torch.no_grad():
        for q, e in zip(gen_params, ema):
            e.mul_(args.ema).add_(q.data, alpha=1 - args.ema)
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
