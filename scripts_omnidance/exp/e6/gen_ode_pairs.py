"""E6-1: teacher ODE trajectories for causal ODE-regression init (CausVid / OmniForcing recipe).
Teacher = S4-final, 50-step Euler flow matching (shift 5), text_first 3-branch CFG (text 2.0, audio 2.0 with the
inference audio schedule). For each sample, saves latents at the 4 student sigmas (closest teacher step) + final x0.
Output: <out>/<sid>_<cond>.pt = {latents: [5,C,T,H,W] bf16, sigmas: [5], sid, cond, prompt, mert_path, image_path}
python gen_ode_pairs.py --out <dir> --shard i --nshard n --num 1200
"""
import argparse, csv, os, random, sys, time
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dmd"))
import dmd_core as C  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--ckpt", default="/dev/shm/od/ckpt/S4-final/step-final.safetensors")
p.add_argument("--data", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/data/dmd/dmd_train.csv")
p.add_argument("--num", type=int, default=1200)
p.add_argument("--start", type=int, default=0)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--height", type=int, default=1280)
p.add_argument("--width", type=int, default=704)
p.add_argument("--num_frames", type=int, default=81)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--cond_mix", default="joint:0.6,audio_only:0.2,text_only:0.2")
a = p.parse_args()

csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(a.data)))[a.start:a.start + a.num]
rng = random.Random(a.start)
mix = [(k, float(v)) for k, v in (x.split(":") for x in a.cond_mix.split(","))]
def pick():
    r, acc = rng.random(), 0.0
    for k, w in mix:
        acc += w
        if r <= acc:
            return k
    return mix[-1][0]
jobs = [(a.start + i, r, pick()) for i, r in enumerate(rows)][a.shard::a.nshard]
os.makedirs(a.out, exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f"{a.out}/{j[1]['sample_id']}_{j[2]}.pt")]
print(f"[ode] {len(jobs)} todo", flush=True)
if not jobs:
    sys.exit(0)

dev = "cuda:0"
pipe, m = C.build_pipe(a.ckpt, dev)
pipe.dit.to(dev)
sched_sig = None
for i, r, cond in jobs:
    t0 = time.time()
    img = Image.open(r["image_path"]).convert("RGB").resize((a.width, a.height), Image.BICUBIC)
    ae = None if cond == "text_only" else torch.load(r["mert_path"], map_location="cpu", weights_only=True)[..., :a.num_frames].to(dev, torch.bfloat16)
    prompt = " " if cond == "audio_only" else r["prompt"]
    with torch.no_grad():
        shared, posi, nega = C.prepare_sample(pipe, prompt, img, ae, a.height, a.width, a.num_frames, seed=1000 + i)
        pipe.scheduler.set_timesteps(a.steps, denoising_strength=1.0, shift=5.0)
        sig = [float(s) for s in pipe.scheduler.sigmas[:a.steps]] + [0.0]
        keep = sorted({min(range(a.steps), key=lambda k: abs(sig[k] - s)) for s in C.STUDENT_SIGMAS})
        iv, tiv, tiav = C.cond_views(posi, nega)
        x = shared["latents"].clone()
        saved, saved_sig = [], []
        for k in range(a.steps):
            s = sig[k]
            if k in keep:
                xs = x.clone(); xs[:, :, 0:1] = shared["first_frame_latents"]
                saved.append(xs.to(torch.bfloat16).cpu()); saved_sig.append(s)
            v_i = C.velocity(pipe, shared, iv, x, s, "real")
            v_ti = C.velocity(pipe, shared, tiv, x, s, "real")
            v = v_i + 2.0 * (v_ti - v_i)
            if "audio_embeds" in posi:
                asc = m.compute_text_only_audio_scale(2.0, k, a.steps, hold_steps=5, decay_end_step=10, floor_scale=1.0)
                v = v + asc * (C.velocity(pipe, shared, tiav, x, s, "real") - v_ti)
            x = x + (sig[k + 1] - s) * v
            x[:, :, 0:1] = shared["first_frame_latents"]
        saved.append(x.to(torch.bfloat16).cpu()); saved_sig.append(0.0)
    torch.save(dict(latents=torch.cat(saved, 0), sigmas=torch.tensor(saved_sig), sid=r["sample_id"], cond=cond,
                    prompt=prompt, mert_path=r["mert_path"], image_path=r["image_path"]),
               f"{a.out}/{r['sample_id']}_{cond}.pt")
    if i % 50 == 0 or True:
        print(f"[ode] {r['sample_id']} {cond} {time.time() - t0:.0f}s keep={keep}", flush=True)
print("[ode] done", flush=True)
