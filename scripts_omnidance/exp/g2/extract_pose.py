"""G2: DWPose body keypoints for GT clips (training data for a music-motion contrastive discriminator).
python extract_pose.py --csv <csv sample_id,video,mert_path> --out <dir> --shard i --nshard n   (venv_eval, 1 GPU)
Output: <dir>/<sid>.npz  kps [T,18,2] (pixels, NaN if low conf), conf [T,18], h, w, fps
"""
import argparse, csv, os, sys

import cv2, decord
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--max_frames", type=int, default=81)
a = p.parse_args()

SCAIL = "/apdcephfs_gy4/share_302533218/cedricnie/pose-preprocess/third_party/scail_pose"
sys.path.insert(0, f"{SCAIL}/DWPoseProcess")
import torch  # noqa: E402,F401  (loads CUDA libs for onnxruntime)
import onnxruntime as _ort  # noqa: E402
_ort.preload_dlls()
from dwpose.wholebody import Wholebody  # noqa: E402

csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(a.csv)))[a.shard::a.nshard]
os.makedirs(a.out, exist_ok=True)
rows = [r for r in rows if not os.path.exists(f"{a.out}/{r['sample_id']}.npz")]
print(f"[pose] {len(rows)} todo", flush=True)
_cwd = os.getcwd(); os.chdir(SCAIL)
model = Wholebody(0)
os.chdir(_cwd)
for n, r in enumerate(rows):
    try:
        vr = decord.VideoReader(r["video"])
        T = min(len(vr), a.max_frames)
        fr = vr.get_batch(list(range(T))).asnumpy()
        H, W = fr.shape[1:3]
        K = np.full((T, 18, 2), np.nan, np.float32)
        S = np.zeros((T, 18), np.float32)
        for t in range(T):
            kps, sc, det = model(cv2.cvtColor(fr[t], cv2.COLOR_RGB2BGR))
            if len(kps) == 0:
                continue
            det = np.asarray(det).reshape(-1, 4)
            i = int(np.argmax((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))) if len(det) else 0
            k = kps[i, :18].astype(np.float32).copy()
            s = sc[i, :18].astype(np.float32)
            k[s < 0.3] = np.nan
            K[t], S[t] = k, s
        np.savez_compressed(f"{a.out}/{r['sample_id']}.npz", kps=K, conf=S, h=H, w=W, fps=float(vr.get_avg_fps()))
    except Exception as e:
        print(f"[pose] {r['sample_id']} error {str(e)[:120]}", flush=True)
    if n % 50 == 0:
        print(f"[pose] {n}/{len(rows)}", flush=True)
print("[pose] done", flush=True)
