"""How much does each input move the generated motion?  (pose-sequence distance between paired videos)

For each val sample, distance between the body-pose sequence of the joint video and:
  seed  : same conditions, different noise seed          (sampling variability, the floor)
  text  : tswap video (prompt swapped)                   (text effect)
  audio : aswap video (music swapped)                    (music effect)
  other : another sample's joint video                   (different reference image + conditions, the ceiling)
Pose distance: keypoints centred on the neck, scaled by torso length, mean L2 over frames/joints (frame 0 excluded).
usage: motion_factors.py <tag> <joint_kps_dir> <seed_kps_dir> <tswap_kps_dir> <aswap_kps_dir>   (dirs may be the same)
"""
import glob, os, sys

import numpy as np

tag, dj, ds, dt, da = sys.argv[1:6]


def load(d, cond):
    out = {}
    for f in glob.glob(f"{d}/*__{cond}__*.npz"):
        sid = os.path.basename(f)[:-4].split("__")[-1]
        k = np.load(f)["kps"].astype(np.float32)[1:81]
        neck = k[:, 1:2]
        hip = np.nanmean(k[:, [8, 11]], axis=1, keepdims=True)
        torso = np.nanmedian(np.linalg.norm(neck - hip, axis=-1))
        if not np.isfinite(torso) or torso < 1:
            continue
        out[sid] = (k - neck) / torso
    return out


def dist(a, b):
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n] - b[:n], axis=-1)
    return float(np.nanmean(d))


J, Sd, Tx, Au = load(dj, "joint"), load(ds, "joint"), load(dt, "tswap"), load(da, "aswap")
ids = sorted(J)
res = {"seed": [], "text": [], "audio": [], "other": []}
for i, s in enumerate(ids):
    if s in Sd: res["seed"].append(dist(J[s], Sd[s]))
    if s in Tx: res["text"].append(dist(J[s], Tx[s]))
    if s in Au: res["audio"].append(dist(J[s], Au[s]))
    o = ids[(i + 11) % len(ids)]
    res["other"].append(dist(J[s], J[o]))
fmt = lambda v: f"{np.mean(v):.3f}±{np.std(v) / np.sqrt(len(v)):.3f} (n={len(v)})" if v else "–"
print(f"| {tag} | {fmt(res['seed'])} | {fmt(res['text'])} | {fmt(res['audio'])} | {fmt(res['other'])} |")
