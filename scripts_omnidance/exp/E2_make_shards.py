"""Build E2 eval shards: OD-Val-47 x {joint, audio_only, text_only, tswap, aswap}.
Output: logs_omnidance/exp/E2/shards/n{node}_g{gpu}_{cond}.csv  (node 0..N-1, gpu 0..7)
"""
import csv, os, sys, json

R = "/apdcephfs_gy4/share_302533218/cedricnie"
M = f"{R}/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main"
OUT = os.environ.get("E2_DIR", f"{R}/diff/logs_omnidance/exp/E2")
n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 7
conds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["joint", "audio_only", "text_only", "tswap", "aswap"]

rows = list(csv.DictReader(open(f"{R}/OmniDance/part049_val.csv")))
base = []
for r in rows:
    s = r["sample_id"]
    base.append(dict(sample_id=s, image_path=f"{M}/first_frames/{s}.png", audio_path=f"{M}/audio/{s}.wav",
                     video=f"{M}/videos/{s}.mp4", prompt=r["prompt"]))
n = len(base)
SHIFT_T, SHIFT_A = 23, 17  # fixed derangements
os.makedirs(f"{OUT}/shards", exist_ok=True)
full = {}
for c in conds:
    lst = []
    for i, b in enumerate(base):
        d = dict(b)
        if c == "tswap":
            d["prompt"] = base[(i + SHIFT_T) % n]["prompt"]
            d["swap_src"] = base[(i + SHIFT_T) % n]["sample_id"]
        elif c == "aswap":
            d["audio_path"] = base[(i + SHIFT_A) % n]["audio_path"]
            d["swap_src"] = base[(i + SHIFT_A) % n]["sample_id"]
        else:
            d["swap_src"] = ""
        lst.append(d)
    full[c] = lst
    with open(f"{OUT}/full_{c}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lst[0].keys())); w.writeheader(); w.writerows(lst)

jobs = [(c, d) for c in conds for d in full[c]]
G = n_nodes * 8
shards = {}
for j, (c, d) in enumerate(jobs):
    g = j % G
    shards.setdefault((g // 8, g % 8, c), []).append(d)
for f in os.listdir(f"{OUT}/shards"):
    os.remove(f"{OUT}/shards/{f}")
for (nd, gp, c), lst in shards.items():
    with open(f"{OUT}/shards/n{nd}_g{gp}_{c}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lst[0].keys())); w.writeheader(); w.writerows(lst)
print(f"jobs={len(jobs)} gpus={G} shards={len(shards)} max_per_gpu={max(sum(len(v) for k,v in shards.items() if k[:2]==(a,b)) for a in range(n_nodes) for b in range(8))}")
