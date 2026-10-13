"""E4 unified metrics for generated dance videos (one GPU, one shard).

manifest csv columns: key, video, ref_image, prompt, alt_prompt, audio, alt_audio, cond, tag
Per video outputs (jsonl):
  bas / bas_alt       Pose-BAS vs conditioning audio / vs alt audio   (DWPose body keypoints, 16 fps)
  ta / ta_alt         TextAlign (Qwen3-VL-8B P(Yes) over motion clauses of prompt / alt prompt)
  ffpsnr              first frame vs reference image
  id_dino             mean DINOv2 cosine(ref, frames)
  motion              mean normalized joint speed
Usage: python eval_metrics.py --manifest m.csv --out o.jsonl --shard i --nshard n [--skip_ta]
"""
import argparse, csv, json, os, re, sys
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--skip_ta", action="store_true")
p.add_argument("--vlm", default="/apdcephfs_gy4/share_302533218/cedricnie/models/Qwen3-VL-8B-Instruct")
p.add_argument("--fps", type=float, default=16.0)
args = p.parse_args()

csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(args.manifest)))[args.shard::args.nshard]
done = set()
if os.path.exists(args.out):
    for l in open(args.out):
        done.add(json.loads(l)["key"])
rows = [r for r in rows if r["key"] not in done]
print(f"[eval] shard {args.shard}/{args.nshard}: {len(rows)} todo", flush=True)
if not rows:
    sys.exit(0)

import cv2, decord
from scipy.signal import find_peaks
from PIL import Image

# ---------------- DWPose
SCAIL = "/apdcephfs_gy4/share_302533218/cedricnie/pose-preprocess/third_party/scail_pose"
sys.path.insert(0, f"{SCAIL}/DWPoseProcess")
_cwd = os.getcwd(); os.chdir(SCAIL)
import onnxruntime as _ort; _ort.preload_dlls()
from dwpose.wholebody import Wholebody  # noqa: E402
pose_model = Wholebody(0)
os.chdir(_cwd)


def read_frames(path, max_dur=81 / 16):
    vr = decord.VideoReader(path)
    fps = float(vr.get_avg_fps()) or 16.0
    n = min(len(vr), int(round(max_dur * fps)))
    fr = vr.get_batch(list(range(n))).asnumpy()
    return fr, fps


def body_kps(frames):
    out = []
    for f in frames:
        kps, sc, det = pose_model(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        if len(kps) == 0:
            out.append(np.full((18, 2), np.nan, np.float32)); continue
        # pick largest bbox
        det = np.asarray(det).reshape(-1, 4)
        i = int(np.argmax((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))) if len(det) else 0
        k = kps[i, :18].astype(np.float32).copy()
        k[sc[i, :18] < 0.3] = np.nan
        out.append(k)
    return np.stack(out)


BEATS = json.load(open(os.environ.get("BEATS_JSON", "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/beats.json")))


def beat_times(audio, dur):
    bt = np.asarray(BEATS[audio], dtype=np.float32)
    return bt[bt <= dur]


def pose_bas(kps, bts, h, fps=16.0, tol=0.1, dist=0.3):
    k = kps / h
    d = np.linalg.norm(k[1:] - k[:-1], axis=-1)
    d = np.nan_to_num(d, nan=0.0).sum(-1)
    if len(bts) == 0 or len(d) < 3:
        return float("nan"), float(d.mean()) if len(d) else 0.0
    peaks, _ = find_peaks(d, distance=max(1, int(dist * fps)))
    pt = (peaks + 0.5) / fps
    if len(pt) == 0:
        return 0.0, float(d.mean())
    return float(np.mean([np.min(np.abs(pt - b)) < tol for b in bts])), float(d.mean())


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))


# ---------------- DINO identity
from transformers import AutoImageProcessor, AutoModel  # noqa: E402
DINO = "facebook/dinov2-base"
dproc = AutoImageProcessor.from_pretrained(DINO)
dino = AutoModel.from_pretrained(DINO, torch_dtype=torch.float16).cuda().eval()


@torch.no_grad()
def dino_feat(imgs):
    x = dproc(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
    f = dino(pixel_values=x).pooler_output.float()
    return torch.nn.functional.normalize(f, dim=-1)


# ---------------- TextAlign
if not args.skip_ta:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402
    vproc = AutoProcessor.from_pretrained(args.vlm)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(args.vlm, torch_dtype=torch.bfloat16).cuda().eval()
    tok = vproc.tokenizer
    YES = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes")]
    NO = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No")]


def motion_clauses(prompt, maxn=8):
    secs = re.split(r"\[\d\.[^\]]*\]", prompt)
    body = " ".join(s for s in secs[1:3]) if len(secs) >= 3 else prompt
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 20]
    return sents[:maxn] or [prompt[:300]]


@torch.no_grad()
def text_align(frames_pil, prompt):
    scores = []
    for c in motion_clauses(prompt):
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": frames_pil},
            {"type": "text", "text": f"Does the dancer's movement in this video match the following description? \"{c}\" Answer Yes or No."}]}]
        text = vproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = vproc(text=[text], videos=[frames_pil], return_tensors="pt", do_resize=True).to("cuda")
        logits = vlm(**inp).logits[0, -1].float()
        ly, ln = torch.logsumexp(logits[YES], 0), torch.logsumexp(logits[NO], 0)
        scores.append(torch.sigmoid(ly - ln).item())
    return float(np.mean(scores))


fo = open(args.out, "a")
for r in rows:
    try:
        fr, vfps = read_frames(r["video"])
        T, H, W, _ = fr.shape
        dur = T / vfps
        kps = body_kps(fr)
        kd = os.path.join(os.path.dirname(os.path.abspath(args.out)), "kps"); os.makedirs(kd, exist_ok=True)
        np.savez_compressed(os.path.join(kd, r["key"].replace("/", "__") + ".npz"), kps=kps, fps=vfps, h=H, w=W)
        bts, bts_alt = beat_times(r["audio"], dur), beat_times(r["alt_audio"], dur)
        bas, mot = pose_bas(kps, bts, H, vfps)
        bas_alt, _ = pose_bas(kps, bts_alt, H, vfps)
        ref = np.array(Image.open(r["ref_image"]).convert("RGB").resize((W, H), Image.BICUBIC))
        ff = psnr(fr[0], ref)
        idx = np.linspace(0, T - 1, 8).astype(int)
        pil = [Image.fromarray(fr[i]) for i in idx]
        fe = dino_feat([Image.fromarray(ref)] + pil)
        idd = float((fe[1:] @ fe[0]).mean())
        rec = dict(key=r["key"], tag=r["tag"], cond=r["cond"], bas=bas, bas_alt=bas_alt, motion=mot,
                   ffpsnr=ff, id_dino=idd, n_beats=int(len(bts)), frames=int(T), fps=vfps,
                   kp_valid=float(np.isfinite(kps).mean()))
        if not args.skip_ta:
            small = [im.resize((W * 448 // H // 28 * 28 or 224, 448)) for im in pil]
            rec["ta"] = text_align(small, r["prompt"])
            rec["ta_alt"] = text_align(small, r["alt_prompt"])
        fo.write(json.dumps(rec) + "\n"); fo.flush()
        print(f"[eval] {r['key']} {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k != 'key'})}", flush=True)
    except Exception as e:  # keep going, record error
        import traceback; traceback.print_exc()
        fo.write(json.dumps(dict(key=r["key"], tag=r["tag"], cond=r["cond"], error=str(e)[:300])) + "\n"); fo.flush()
print("[eval] shard done", flush=True)
