"""H0: convert OmniDance 16fps clips to H3 format: 480x832 (WxH), 24fps, 124 frames (17n+5), RIFE 4.22 interpolation.
Per clip -> <out>/video/<sid>.mp4, <out>/audio/<sid>.wav (copied), <out>/ref/<sid>_f0.png (first frame), <out>/ref/<sid>_mid.png (t~3s).
python prep_h3_data.py --csv <sample_id,video,audio> --out /dev/shm/h3data --shard i --nshard n   (one GPU)
"""
import argparse, csv, os, shutil, sys, time
import numpy as np, torch, torch.nn.functional as F, decord, imageio
from PIL import Image

RIFE = "/apdcephfs_gy4/share_302533218/cedricnie/models/rife422/Practical-RIFE"
sys.path.insert(0, RIFE)
p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--W", type=int, default=480)
p.add_argument("--H", type=int, default=832)
p.add_argument("--nf", type=int, default=124)
p.add_argument("--interp", default="rife", choices=["rife", "dup"], help="dup: no synthesized frames, 16->24 fps by nearest-frame repetition")
a = p.parse_args()
csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(a.csv)))[a.shard::a.nshard]
for d in ("video", "audio", "ref"):
    os.makedirs(f"{a.out}/{d}", exist_ok=True)

_cwd = os.getcwd(); os.chdir(RIFE)
from train_log.RIFE_HDv3 import Model  # noqa: E402
model = Model(); model.load_model(f"{RIFE}/train_log", -1); model.eval(); model.device()
os.chdir(_cwd)
dev = torch.device("cuda")


def load_frames(path):
    vr = decord.VideoReader(path)
    fr = vr.get_batch(list(range(len(vr)))).asnumpy()
    h, w = fr.shape[1:3]
    nh = int(round(h * a.W / w))
    x = torch.from_numpy(fr).to(dev).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(nh, a.W), mode="bicubic", align_corners=False).clamp(0, 1)
    top = max(0, (nh - a.H) // 2)
    x = x[:, :, top:top + a.H]
    if x.shape[2] < a.H:
        x = F.pad(x, (0, 0, 0, a.H - x.shape[2]), mode="replicate")
    return x, float(vr.get_avg_fps()) or 16.0


@torch.no_grad()
def to24(x, fps):
    n = x.shape[0]; out = []
    for k in range(a.nf):
        pos = min(k * fps / 24.0, n - 1)
        i = int(np.floor(pos)); f = pos - i
        if a.interp == "dup":
            out.append(x[min(int(round(pos)), n - 1)][None]); continue
        if f < 1e-3 or i + 1 >= n:
            out.append(x[i:i + 1])
        else:
            out.append(model.inference(x[i:i + 1], x[i + 1:i + 2], timestep=f).clamp(0, 1))
    return torch.cat(out)


done = 0; t0 = time.time()
for r in rows:
    sid = r["sample_id"]; vo = f"{a.out}/video/{sid}.mp4"
    if os.path.exists(vo):
        continue
    try:
        x, fps = load_frames(r["video"])
        y = (to24(x, fps).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        tmp = vo + ".tmp.mp4"
        imageio.mimwrite(tmp, list(y), fps=24, codec="libx264", quality=9, macro_block_size=16)
        os.replace(tmp, vo)
        shutil.copy(r["audio"], f"{a.out}/audio/{sid}.wav")
        Image.fromarray(y[0]).save(f"{a.out}/ref/{sid}_f0.png")
        Image.fromarray(y[min(72, len(y) - 1)]).save(f"{a.out}/ref/{sid}_mid.png")
        done += 1
        if done % 50 == 0:
            print(f"[prep] shard {a.shard} {done}/{len(rows)} {(time.time()-t0)/done:.2f}s/clip", flush=True)
    except Exception as e:
        print(f"[prep] ERR {sid} {str(e)[:120]}", flush=True)
print(f"[prep] shard {a.shard} done {done}", flush=True)
