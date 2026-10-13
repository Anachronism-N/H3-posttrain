"""Sample the 4-step DMD student on OD-Val-47 (E2 protocol, 5 conditions). One GPU per shard.
python sample_student.py --lora <run>/iter_XXXX/gen_lora.pt --out <root> --shard i --nshard n [--height --width --num_frames]
Output: <root>/<cond>/<sid>_<audiostem>.mp4 (with audio)
"""
import argparse, csv, json, os, sys, time
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmd_core as C  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--lora", default="", help="gen_lora.pt; empty = run base S4 with 4 steps (sanity baseline)")
p.add_argument("--ckpt", default="/dev/shm/od/ckpt/S4-11000/step-11000.safetensors")
p.add_argument("--rank", type=int, default=128)
p.add_argument("--out", required=True)
p.add_argument("--conds", default="joint,audio_only,text_only,tswap,aswap")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--height", type=int, default=832)
p.add_argument("--width", type=int, default=480)
p.add_argument("--num_frames", type=int, default=49)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--rows", type=int, default=0, help="limit rows per cond (0=all 47)")
p.add_argument("--csv_dir", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E2")
p.add_argument("--ar_block", type=int, default=0, help=">0: block-causal AR student (E6) with this block size")
p.add_argument("--sigmas", default="", help="comma list of student sigmas (default STUDENT_SIGMAS)")
args = p.parse_args()

E2 = args.csv_dir
csv.field_size_limit(10 ** 9)
jobs = []
for c in args.conds.split(","):
    rows = list(csv.DictReader(open(f"{E2}/full_{c}.csv")))
    if args.rows:
        rows = rows[:args.rows]
    jobs += [(c, r) for r in rows]
jobs = jobs[args.shard::args.nshard]
todo = []
for c, r in jobs:
    dst = f"{args.out}/{c}/{r['sample_id']}_{os.path.basename(r['audio_path'])[:-4]}.mp4"
    if not os.path.exists(dst):
        todo.append((c, r, dst))
print(f"[student] {len(todo)} todo", flush=True)
if not todo:
    sys.exit(0)

dev = "cuda:0"
pipe, m = C.build_pipe(args.ckpt, dev)
pipe.dit.to(dev)
if args.lora:
    C.inject_dual_lora(pipe.dit, args.rank, args.rank)
    C.load_lora_state(pipe.dit, torch.load(args.lora, map_location="cpu"))
    mode = "gen"
else:
    mode = "real"
from diffsynth.utils.data import save_video_with_audio  # noqa: E402
SIG = [float(x) for x in args.sigmas.split(",")] if args.sigmas else C.STUDENT_SIGMAS
if args.ar_block:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e6"))
    import causal  # noqa: E402
    causal.enable((args.height // 32) * (args.width // 32), args.ar_block)

for c, r, dst in todo:
    t0 = time.time()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = Image.open(r["image_path"]).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
    prompt = " " if c == "audio_only" else r["prompt"]
    wav, sr = (None, None) if c == "text_only" else m.load_audio(r["audio_path"])
    with torch.no_grad():
        shared, posi, _ = C.prepare_sample_wave(pipe, prompt, img, wav, sr, args.height, args.width, args.num_frames, args.seed)
        if args.ar_block:
            x0 = causal.ar_sample(pipe, shared, posi, block=args.ar_block, sigmas=SIG, seed=args.seed)
            video = C.decode(pipe, x0)
            save_video_with_audio(video, dst, r["audio_path"], fps=16)
            print(f"[student] {c} {r['sample_id']} {time.time()-t0:.1f}s", flush=True)
            continue
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        noise = torch.randn(shared["latents"].shape, generator=gen, device=dev, dtype=shared["latents"].dtype)
        x = noise
        for k, s in enumerate(SIG):
            v = C.velocity(pipe, shared, posi, x, s, mode)
            x0 = x - s * v
            if k < len(SIG) - 1:
                sn = SIG[k + 1]
                x = (1 - sn) * x0 + sn * torch.randn(x.shape, generator=gen, device=dev, dtype=x.dtype)
        x0[:, :, 0:1] = shared["first_frame_latents"]
        video = C.decode(pipe, x0)
    save_video_with_audio(video, dst, r["audio_path"], fps=16)
    print(f"[student] {c} {r['sample_id']} {time.time()-t0:.1f}s", flush=True)
print("[student] done", flush=True)
