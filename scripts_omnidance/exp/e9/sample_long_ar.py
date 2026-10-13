"""E9-AR: long video by rolling the block-causal 4-step AR student (E6) segment by segment.
Segment k>0: frame0 = decoded frame g0, block 1 (latents 1..4) = clean re-encoding of video frames [g0, g0+16], AR student generates blocks 2..5 (64 new frames).
(derived from the E9 continuation sampler)

Segment 0: normal TI2V/MTI2V from the reference image (81 frames).
Segment k>0: frame0 = decoded video frame g0 = prev_start + 80 - 4m, history = VAE re-encoding of video frames
[g0, g0+4m] (latents 1..m, exactly the training layout), generate the remaining 80-4m frames.
m=0 => baseline "last-frame I2V chaining" (no history).
Long music = the sample's own clip followed by (segments-1) other val clips (concatenated) -- a mechanics test of
continuity, not of musical coherence.

python sample_long.py --out <dir> [--lora <gen_lora.pt>] --m 3 --segments 4 --shard i --nshard n
Output: <out>/<sid>.mp4 (with long audio) + <out>/<sid>.json (per-seam stats)
"""
import argparse, csv, json, os, sys, time

import numpy as np
import torch
import torchaudio
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dmd"))
sys.path.insert(0, os.path.join(HERE, "..", "e6"))
import dmd_core as C  # noqa: E402
import causal  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--lora", default="")
p.add_argument("--ckpt", default="/dev/shm/od/ckpt/S4-final/step-final.safetensors")
p.add_argument("--m", type=int, default=4, help="history latents = one AR block")
p.add_argument("--block", type=int, default=4)
p.add_argument("--anchor", default="prev", choices=["prev", "ref"], help="frame0 of segment k>0: previous decoded frame, or the original reference image")
p.add_argument("--sigmas", default="1.0,0.9405940771102905,0.8333333134651184,0.6372548937797546")
p.add_argument("--segments", type=int, default=4)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--hist_sigma", type=float, default=0.05)
p.add_argument("--rows", type=int, default=12)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--height", type=int, default=1280)
p.add_argument("--width", type=int, default=704)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

E2 = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E2"
csv.field_size_limit(10 ** 9)
allrows = list(csv.DictReader(open(f"{E2}/full_joint.csv")))
n = len(allrows)
jobs = [(i, allrows[i]) for i in range(min(args.rows, n))][args.shard::args.nshard]
os.makedirs(args.out, exist_ok=True)
jobs = [(i, r) for i, r in jobs if not os.path.exists(f"{args.out}/{r['sample_id']}.mp4")]
print(f"[long] {len(jobs)} todo", flush=True)
if not jobs:
    sys.exit(0)

dev = "cuda:0"
pipe, m_inf = C.build_pipe(args.ckpt, dev)
pipe.dit.to(dev)
C.inject_dual_lora(pipe.dit, 128, 128)
C.load_lora_state(pipe.dit, torch.load(args.lora, map_location="cpu"))
causal.enable((args.height // 32) * (args.width // 32), args.block)
SIG = [float(x) for x in args.sigmas.split(",")]
F = 81
SR = 16000


def long_audio(i):
    ws = []
    for k in range(args.segments + 1):
        w, sr = torchaudio.load(allrows[(i + 7 * k) % n]["audio_path"])
        w = w.mean(0)
        ws.append(torchaudio.functional.resample(w, sr, SR))
    return torch.cat(ws)


@torch.no_grad()
def encode_frames(frames):
    v = pipe.preprocess_video([Image.fromarray(f) for f in frames])
    return pipe.vae.encode(v, device=dev, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)).to(device=dev, dtype=pipe.torch_dtype)


@torch.no_grad()
def gen_segment(img, wav_seg, prompt, hist, m, seed):
    shared, posi, _ = C.prepare_sample_wave(pipe, prompt, img, wav_seg, SR, args.height, args.width, F, seed)
    x = causal.ar_sample(pipe, shared, posi, block=args.block, sigmas=SIG, seed=seed, prefix=hist if m else None)
    return np.stack([np.asarray(f) for f in C.decode(pipe, x)])


from diffsynth.utils.data import save_video  # noqa: E402

for i, r in jobs:
    t0 = time.time()
    wav = long_audio(i)
    img = Image.open(r["image_path"]).convert("RGB").resize((args.width, args.height), Image.BICUBIC)
    video = None  # global frames
    seams = []
    start = 0
    for s in range(args.segments):
        a0 = int(start / 16 * SR)
        wseg = wav[a0:a0 + int(F / 16 * SR) + SR // 16]
        if s == 0:
            seg = gen_segment(img, wseg, r["prompt"], None, 0, args.seed)
            video = seg
        else:
            m = args.m
            fr0 = video[start]
            hist = encode_frames(video[start:start + 4 * m + 1])[:, :, 1:1 + m] if m else None
            seg = gen_segment(img if args.anchor == "ref" else Image.fromarray(fr0), wseg, r["prompt"], hist, m, args.seed + s)
            b = 4 * m + 1  # first newly generated frame within segment
            new = seg[b:]
            prev = video[start + b - 1].astype(np.float32)
            seam = float(np.abs(new[0].astype(np.float32) - prev).mean())
            e = start + b
            inner = float(np.mean([np.abs(video[t].astype(np.float32) - video[t - 1].astype(np.float32)).mean() for t in range(e - 4, e)]))
            seams.append(dict(seg=s, seam_l1=seam, ref_l1=inner, ratio=seam / max(inner, 1e-3)))
            video = np.concatenate([video[:start + b], new], 0)
        start = len(video) - 1 - 4 * args.m if args.m else len(video) - 1
    dst = f"{args.out}/{r['sample_id']}.mp4"
    tmp_wav = f"{args.out}/{r['sample_id']}.wav"
    torchaudio.save(tmp_wav, wav[: int(len(video) / 16 * SR)].unsqueeze(0), SR)
    save_video([Image.fromarray(f) for f in video], dst + ".noaudio.mp4", fps=16)
    os.system(f"ffmpeg -loglevel error -y -i {dst}.noaudio.mp4 -i {tmp_wav} -c:v copy -c:a aac -shortest {dst} && rm {dst}.noaudio.mp4")
    first = video[0].astype(np.float32)
    stats = dict(sample_id=r["sample_id"], frames=len(video), seams=seams,
                 color_drift_last=float(np.abs(video[-16:].astype(np.float32).mean(axis=(0, 1, 2)) - first.mean(axis=(0, 1))).mean()),
                 sec=time.time() - t0)
    json.dump(stats, open(f"{args.out}/{r['sample_id']}.json", "w"))
    print(f"[long] {r['sample_id']} frames={len(video)} {time.time() - t0:.0f}s seams={[round(x['ratio'], 2) for x in seams]}", flush=True)
print("[long] done", flush=True)
