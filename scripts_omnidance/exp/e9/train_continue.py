"""E9: continuation SFT (FlowAct-R2 style, history-as-condition) on S4-final, LoRA.

Per 81-frame GT clip (latent T=21): frame 0 = clean image latent (as in TI2V); frames 1..m = GT history,
lightly noised to sigma_h ~ U[0, hist_noise_max] (robustness to AR error); frames m+1..20 = target, standard flow
matching at sigma ~ shift(U). Per-frame timesteps via e6/causal.py patch (attention stays bidirectional).
Loss only on target frames. m ~ U{1..max_hist}.

Long-video inference (sample_long): clip k+1 uses the last m latents of clip k as history and the decoded video
frame right before them as frame 0 -> exactly the training layout.

torchrun --nproc_per_node 8 train_continue.py --out <dir>
"""
import argparse, csv, json, os, random, sys, time

import decord
import torch
import torch.distributed as dist
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dmd"))
sys.path.insert(0, os.path.join(HERE, "..", "e6"))
import dmd_core as C  # noqa: E402
import causal  # noqa: E402  (per-frame timestep patch; causal attention NOT enabled)

VID = "/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/videos"

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--ckpt", default="/dev/shm/od/ckpt/S4-final/step-final.safetensors")
p.add_argument("--data", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/data/dmd/dmd_train.csv")
p.add_argument("--row_offset", type=int, default=4000)
p.add_argument("--height", type=int, default=1280)
p.add_argument("--width", type=int, default=704)
p.add_argument("--num_frames", type=int, default=81)
p.add_argument("--max_hist", type=int, default=5)
p.add_argument("--hist_noise_max", type=float, default=0.3)
p.add_argument("--rank", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max_iters", type=int, default=3000)
p.add_argument("--cond_mix", default="joint:0.6,audio_only:0.2,text_only:0.2")
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--sample_every", type=int, default=500)
p.add_argument("--sample_steps", type=int, default=30)
p.add_argument("--max_grad_norm", type=float, default=1.0)
p.add_argument("--seed", type=int, default=4321)
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
T_LAT = (args.num_frames - 1) // 4 + 1


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
    ae = None if cond == "text_only" else torch.load(row["mert_path"], map_location="cpu", weights_only=True)[..., :args.num_frames].to(dev, torch.bfloat16)
    prompt = " " if cond == "audio_only" else row["prompt"]
    shared, posi, _ = C.prepare_sample(pipe, prompt, frames[0], ae, args.height, args.width, args.num_frames, seed=rng.randint(0, 2 ** 31))
    z = encode_video(frames)
    m = rng.randint(1, args.max_hist)
    s = C.shift_sigma(rng.uniform(0.02, 0.98))
    sh = rng.uniform(0.0, args.hist_noise_max)
    eps = torch.randn_like(z)
    sf = torch.full((T_LAT,), s, device=dev)
    sf[0] = 0.0
    sf[1:1 + m] = sh
    sfv = sf.view(1, 1, -1, 1, 1).to(z.dtype)
    x = (1 - sfv) * z + sfv * eps
    v = causal.velocity_pf(pipe, shared, posi, x, sf, grad=True, mode="gen")
    err = ((v.float() - (eps - z).float()) ** 2).mean(dim=(0, 1, 3, 4))
    loss = err[1 + m:].mean()
    return loss, dict(cond=cond, m=m, s=round(s, 3), sh=round(sh, 3))


@torch.no_grad()
def sample_continue(shared, posi, nega, hist, m, steps, sh=0.05, text_scale=2.0, seed=0):
    """3-branch CFG continuation: frames 1..m fixed to `hist` (noised to sh), generate frames m+1..T-1."""
    i_v, ti_v, tia_v = C.cond_views(posi, nega)
    gen = torch.Generator(device=dev).manual_seed(seed)
    lat = shared["latents"]
    x = torch.randn(lat.shape, generator=gen, device=dev, dtype=lat.dtype)
    hn = (1 - sh) * hist + sh * torch.randn(hist.shape, generator=gen, device=dev, dtype=hist.dtype)
    pipe.scheduler.set_timesteps(steps, denoising_strength=1.0, shift=5.0)
    sig = [float(q) for q in pipe.scheduler.sigmas[:steps]] + [0.0]
    for k in range(steps):
        x[:, :, 1:1 + m] = hn
        sf = torch.full((lat.shape[2],), sig[k], device=dev); sf[0] = 0; sf[1:1 + m] = sh
        v_i = causal.velocity_pf(pipe, shared, i_v, x, sf, mode="gen")
        v_ti = causal.velocity_pf(pipe, shared, ti_v, x, sf, mode="gen")
        v = v_i + text_scale * (v_ti - v_i)
        if "audio_embeds" in posi:
            asc = infer_mod.compute_text_only_audio_scale(2.0, int(k * 50 / steps), 50, hold_steps=5, decay_end_step=10, floor_scale=1.0)
            v = v + asc * (causal.velocity_pf(pipe, shared, tia_v, x, sf, mode="gen") - v_ti)
        x = x + (sig[k + 1] - sig[k]) * v
    x[:, :, 1:1 + m] = hist
    x[:, :, 0:1] = shared["first_frame_latents"]
    return x


@torch.no_grad()
def save_sample(it):
    """continuation from GT history (m=3) on a held-out clip; report latent MSE of generated part vs GT."""
    from diffsynth.utils.data import save_video
    import numpy as np
    row = val_row
    frames = read_clip(row["sample_id"])
    ae = torch.load(row["mert_path"], map_location="cpu", weights_only=True)[..., :args.num_frames].to(dev, torch.bfloat16)
    shared, posi, nega = C.prepare_sample(pipe, row["prompt"], frames[0], ae, args.height, args.width, args.num_frames, seed=0)
    z = encode_video(frames)
    m = 3
    lat = sample_continue(shared, posi, nega, z[:, :, 1:1 + m], m, args.sample_steps)
    video = C.decode(pipe, lat)
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in video])
    gt = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
    b = 4 * m + 1  # first generated video frame
    seam = float(np.abs(arr[b] - arr[b - 1]).mean() / max(np.abs(gt[b] - gt[b - 1]).mean(), 1e-3))
    h = dict(iter=it, seam_ratio=seam, drift=float(np.abs(arr[b:].mean(axis=(1, 2)) - gt[:b].mean(axis=(0, 1, 2))).mean()))
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
    save_sample(0)  # zero-shot: S4-final with extra clean frames (reference point)
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
    open(os.path.join(args.out, "DONE"), "w").write("ok")
dist.barrier()
log("finished")
