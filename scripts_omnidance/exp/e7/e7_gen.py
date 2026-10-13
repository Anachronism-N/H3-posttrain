"""E7-v1 step 1: group rollouts of the AR 4-step student (one GPU per shard).

For each pool row (with an E10 motion caption) the intended condition is
    prompt_full = original caption + " Detailed body movements: " + motion caption.
K branches share the prompt row but differ in condition-space exploration / noise:
    k=0 anchor (seed s, full prompt), k=1,2 other seeds (full prompt), k=3 seed s with ONE motion clause dropped.
Rewards are always computed against the full motion caption (the intended condition).
Output: <out>/<sid>_k<k>.mp4 (+ .pt latents [1,C,T,H,W]) and <out>/groups.jsonl (spec per branch).
python e7_gen.py --lora <gen_lora.pt> --out <dir> --round r --rows 64 --shard i --nshard n
"""
import argparse, csv, glob, json, os, random, re, sys, time

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dmd"))
sys.path.insert(0, os.path.join(HERE, "..", "e6"))
import dmd_core as C  # noqa: E402
import causal  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--lora", required=True)
p.add_argument("--out", required=True)
p.add_argument("--round", type=int, default=0)
p.add_argument("--rows", type=int, default=64)
p.add_argument("--K", type=int, default=4)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--pool", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E7/pool_caps.csv")
p.add_argument("--ckpt", default="/dev/shm/od/ckpt/S4-final/step-final.safetensors")
p.add_argument("--sigmas", default="1.0,0.9405940771102905,0.8333333134651184,0.6372548937797546")
p.add_argument("--block", type=int, default=4)
p.add_argument("--height", type=int, default=1280)
p.add_argument("--width", type=int, default=704)
p.add_argument("--num_frames", type=int, default=81)
a = p.parse_args()
SIG = [float(x) for x in a.sigmas.split(",")]


def full_prompt(row, caption):
    return row["prompt"].rstrip() + " Detailed body movements: " + caption.strip()


def drop_clause(caption, rng):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", caption.strip()) if s]
    if len(sents) < 3:
        return caption
    sents.pop(rng.randrange(len(sents)))
    return " ".join(sents)


csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(a.pool)))
rng0 = random.Random(1000 + a.round)
rng0.shuffle(rows)
rows = rows[:a.rows][a.shard::a.nshard]
os.makedirs(a.out, exist_ok=True)
todo = [r for r in rows if not os.path.exists(f"{a.out}/{r['sample_id']}_k{a.K - 1}.pt")]
print(f"[e7gen] {len(todo)} groups todo", flush=True)
if not todo:
    sys.exit(0)

dev = "cuda:0"
pipe, _ = C.build_pipe(a.ckpt, dev)
pipe.dit.to(dev)
C.inject_dual_lora(pipe.dit, 128, 128)
C.load_lora_state(pipe.dit, torch.load(a.lora, map_location="cpu"))
causal.enable((a.height // 32) * (a.width // 32), a.block)
from diffsynth.utils.data import save_video  # noqa: E402

gl = open(f"{a.out}/groups_{a.shard}.jsonl", "a")
for r in todo:
    t0 = time.time()
    sid = r["sample_id"]
    rng = random.Random(hash(sid) % (2 ** 31) + a.round)
    seed0 = rng.randint(0, 2 ** 30)
    img = Image.open(r["image_path"]).convert("RGB").resize((a.width, a.height), Image.BICUBIC)
    ae = torch.load(r["mert_path"], map_location="cpu", weights_only=True)[..., :a.num_frames].to(dev, torch.bfloat16)
    specs = []
    for k in range(a.K):
        cap = r["motion_caption"]
        seed = seed0 if k in (0, 3) else seed0 + k
        if k == 3:
            cap = drop_clause(cap, rng)
        specs.append(dict(k=k, seed=seed, prompt=full_prompt(r, cap)))
    for sp in specs:
        with torch.no_grad():
            shared, posi, _ = C.prepare_sample(pipe, sp["prompt"], img, ae, a.height, a.width, a.num_frames, seed=sp["seed"])
            traj = []
            lat = causal.ar_sample(pipe, shared, posi, block=a.block, sigmas=SIG, seed=sp["seed"], traj=traj)
            video = C.decode(pipe, lat)
        base = f"{a.out}/{sid}_k{sp['k']}"
        torch.save(dict(lat=lat.cpu(), traj=traj), base + ".pt")
        save_video(video, base + ".mp4", fps=16)
    gl.write(json.dumps(dict(sid=sid, round=a.round, image_path=r["image_path"], mert_path=r["mert_path"],
                             prompt_full=full_prompt(r, r["motion_caption"]),
                             motion_caption=r["motion_caption"], specs=specs)) + "\n")
    gl.flush()
    print(f"[e7gen] {sid} {time.time() - t0:.0f}s", flush=True)
print("[e7gen] done", flush=True)
