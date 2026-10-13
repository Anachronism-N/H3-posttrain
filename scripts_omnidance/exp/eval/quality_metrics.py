"""quality_metrics.py (10-14): hand / face quality proxies for dance videos (complements eval_metrics.py, which has no detail metric).
All frames are resized to 480x832 (W x H) first so GT (704x1280) and every model are measured at the same resolution.
Per video (16 frames uniformly sampled from the first 5 s), DWPose wholebody on the largest person:
  hand_conf  mean keypoint score of both hands (21+21)          face_conf  mean score of the 68 face points
  hand_ok    fraction of frames where mean hand score > 0.5
  hand_sharp / face_sharp   Laplacian variance of a 64x64 crop around the hand / face points (only frames with score > 0.3)
  frame_sharp               Laplacian variance of the whole grey frame
These are proxies (blurred / smeared hands lower keypoint confidence and local high frequency); human review stays primary.
manifest csv: key, video, cond, tag (extra columns ignored)
python quality_metrics.py --manifest m.csv --out o.jsonl --shard i --nshard n
"""
import argparse, csv, json, os, sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0); p.add_argument("--nshard", type=int, default=1)
p.add_argument("--nframes", type=int, default=16)
a = p.parse_args()
csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(a.manifest)))[a.shard::a.nshard]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["key"] for l in open(a.out)}
rows = [r for r in rows if r["key"] not in done]
print(f"[q] shard {a.shard}/{a.nshard}: {len(rows)} todo", flush=True)
if not rows:
    sys.exit(0)
import torch  # noqa: E402,F401  (loads the CUDA/cuBLAS libs onnxruntime-gpu needs, as in eval_metrics.py)
import cv2, decord  # noqa: E402

SCAIL = "/apdcephfs_gy4/share_302533218/cedricnie/pose-preprocess/third_party/scail_pose"
sys.path.insert(0, f"{SCAIL}/DWPoseProcess")
_cwd = os.getcwd(); os.chdir(SCAIL)
import onnxruntime as _ort; _ort.preload_dlls()  # noqa: E402
from dwpose.wholebody import Wholebody  # noqa: E402
pose = Wholebody(0)
os.chdir(_cwd)
W, H = 480, 832
FACE, HANDS = list(range(24, 92)), list(range(92, 134))  # after neck insertion: 18 body + 6 feet + 68 face + 21 + 21 hands


def lapvar(img):
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def crop_sharp(grey, pts, sc, thr=0.3, size=64):
    ok = sc > thr
    if ok.sum() < 3:
        return None
    q = pts[ok]
    cx, cy = q.mean(0); half = max(16.0, 0.75 * max(np.ptp(q[:, 0]), np.ptp(q[:, 1])))
    x0, y0, x1, y1 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return lapvar(cv2.resize(grey[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA))


fo = open(a.out, "a")
for r in rows:
    try:
        vr = decord.VideoReader(r["video"])
        fps = float(vr.get_avg_fps()) or 24.0
        n = min(len(vr), int(5.0 * fps))
        idx = np.linspace(0, n - 1, a.nframes).astype(int)
        frames = vr.get_batch(list(idx)).asnumpy()
        hc, fc, hs, fs, frs, hok = [], [], [], [], [], []
        for f in frames:
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR); grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            frs.append(lapvar(grey))
            kps, sc, det = pose(bgr)
            if len(kps) == 0:
                hc.append(0.0); fc.append(0.0); hok.append(0.0); continue
            det = np.asarray(det).reshape(-1, 4)
            i = int(np.argmax((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))) if len(det) else 0
            k, s = kps[i], sc[i]
            hc.append(float(s[HANDS].mean())); fc.append(float(s[FACE].mean())); hok.append(float(s[HANDS].mean() > 0.5))
            for side in (HANDS[:21], HANDS[21:]):
                v = crop_sharp(grey, k[side], s[side])
                if v is not None:
                    hs.append(v)
            v = crop_sharp(grey, k[FACE], s[FACE])
            if v is not None:
                fs.append(v)
        rec = dict(key=r["key"], tag=r["tag"], cond=r["cond"], hand_conf=float(np.mean(hc)), face_conf=float(np.mean(fc)),
                   hand_ok=float(np.mean(hok)), hand_sharp=float(np.median(hs)) if hs else None,
                   face_sharp=float(np.median(fs)) if fs else None, frame_sharp=float(np.median(frs)))
        fo.write(json.dumps(rec) + "\n"); fo.flush()
        print(f"[q] {r['key']} " + " ".join(f"{k}={v:.3f}" for k, v in rec.items() if isinstance(v, float)), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        fo.write(json.dumps(dict(key=r["key"], tag=r["tag"], cond=r["cond"], error=str(e)[:300])) + "\n"); fo.flush()
