"""Sample an E11 (reference-decoupled) LoRA on OD-Val-47 with the teacher protocol (50-step 3-branch CFG).
Layout: latent 0 = reference image (GT first frame at test time), latents 1..21 = the 81 output frames.
Output: <out>/<cond>/<sid>_<audiostem>.mp4 (same layout as sample_student, so build_manifest/eval reuse works).
python sample_e11.py --lora <gen_lora.pt> --out <dir> --csv_dir <dir with full_{cond}.csv> --conds joint,tswap,aswap --shard i --nshard n
"""
import argparse, csv, os, sys, time
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dmd"))
sys.path.insert(0, os.path.join(HERE, "..", "e6"))
import dmd_core as C  # noqa: E402
import causal  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--lora", required=True)
p.add_argument("--ckpt", default="/dev/shm/od/ckpt/S4-final/step-final.safetensors")
p.add_argument("--rank", type=int, default=128)
p.add_argument("--out", required=True)
p.add_argument("--csv_dir", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E7/val_full")
p.add_argument("--conds", default="joint,tswap,aswap")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--height", type=int, default=1280)
p.add_argument("--width", type=int, default=704)
p.add_argument("--num_frames", type=int, default=81)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--text_scale", type=float, default=2.0)
p.add_argument("--audio_scale", type=float, default=2.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--rows", type=int, default=0)
args = p.parse_args()

csv.field_size_limit(10 ** 9)
jobs = []
for c in args.conds.split(","):
    rows = list(csv.DictReader(open(f"{args.csv_dir}/full_{c}.csv")))
    jobs += [(c, r) for r in (rows[:args.rows] if args.rows else rows)]
jobs = jobs[args.shard::args.nshard]
todo = [(c, r, f"{args.out}/{c}/{r['sample_id']}_{os.path.basename(r['audio_path'])[:-4]}.mp4") for c, r in jobs]
todo = [t for t in todo if not os.path.exists(t[2])]
print(f"[e11] {len(todo)} todo", flush=True)
if not todo:
    sys.exit(0)

dev = "cuda:0"
pipe, m = C.build_pipe(args.ckpt, dev)
pipe.dit.to(dev)
C.inject_dual_lora(pipe.dit, args.rank, args.rank)
C.load_lora_state(pipe.dit, torch.load(args.lora, map_location="cpu"))
causal.disable()
from diffsynth.utils.data import save_video_with_audio  # noqa: E402
NF_EXT = args.num_frames + 4


@torch.no_grad()
def audio_ext(path):
    wav, sr = m.load_audio(path)
    f = pipe.mert_audio_encoder.get_audio_feats_per_inference(wav, sr, batch_frames=args.num_frames, fps=16, dtype=pipe.torch_dtype, device=pipe.device)
    a = (f[0] if isinstance(f, list) else f)[..., :args.num_frames]
    return torch.cat([a[..., :1].expand(*a.shape[:-1], 4), a], dim=-1)


@torch.no_grad()
def sample(shared, posi, nega):
    i_v, ti_v, tia_v = C.cond_views(posi, nega)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    lat = shared["latents"]
    x = torch.randn(lat.shape, generator=gen, device=dev, dtype=lat.dtype)
    pipe.scheduler.set_timesteps(args.steps, denoising_strength=1.0, shift=5.0)
    sig = [float(q) for q in pipe.scheduler.sigmas[:args.steps]] + [0.0]
    for k in range(args.steps):
        sf = torch.full((lat.shape[2],), sig[k], device=dev); sf[0] = 0
        v_i = causal.velocity_pf(pipe, shared, i_v, x, sf, mode="gen")
        v_ti = causal.velocity_pf(pipe, shared, ti_v, x, sf, mode="gen")
        v = v_i + args.text_scale * (v_ti - v_i)
        if "audio_embeds" in posi:
            asc = m.compute_text_only_audio_scale(args.audio_scale, int(k * 50 / args.steps), 50, hold_steps=5, decay_end_step=10, floor_scale=1.0)
            v = v + asc * (causal.velocity_pf(pipe, shared, tia_v, x, sf, mode="gen") - v_ti)
        x = x + (sig[k + 1] - sig[k]) * v
    return x[:, :, 1:]


for c, r, dst in todo:
    t0 = time.time()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = Image.open(r["image_path"]).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
    prompt = " " if c == "audio_only" else r["prompt"]
    ae = None if c == "text_only" else audio_ext(r["audio_path"])
    with torch.no_grad():
        shared, posi, nega = C.prepare_sample(pipe, prompt, img, ae, args.height, args.width, NF_EXT, args.seed)
        video = C.decode(pipe, sample(shared, posi, nega))
    save_video_with_audio(video, dst, r["audio_path"], fps=16)
    print(f"[e11] {c} {r['sample_id']} {time.time()-t0:.1f}s", flush=True)
print("[e11] done", flush=True)
