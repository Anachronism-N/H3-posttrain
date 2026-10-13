"""Score videos with the G2 music-motion discriminator using saved eval keypoints.
python score_disc.py <disc.pt> <manifest.csv> <kps_dir> [<manifest> <kps_dir> ...]
For each row: s_own = D(motion, given audio), s_alt = D(motion, alt_audio). Prints per (tag,cond): mean s_own, s_alt,
paired Δ = s_own - s_alt with s.e., and AUC(own vs alt). For aswap rows "given" is the swapped music and alt = orig.
"""
import csv, os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
T = 81
MERT = "/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/mert_features"
csv.field_size_limit(10 ** 9)


def motion_feat(k):
    k = k.astype(np.float32)[:T]
    neck = k[:, 1:2]
    hip = np.nanmean(k[:, [8, 11]], axis=1, keepdims=True)
    torso = np.nanmedian(np.linalg.norm(neck - hip, axis=-1)) + 1e-3
    x = np.nan_to_num((k - neck) / torso, nan=0.0)
    v = np.diff(x, axis=0, prepend=x[:1])
    f = np.concatenate([x.reshape(len(x), -1), v.reshape(len(v), -1)], -1)
    return np.pad(f, ((0, T - len(f)), (0, 0))) if len(f) < T else f


cache = {}


def music_feat(audio):
    sid = os.path.basename(audio)[:-4]
    if sid not in cache:
        p = f"{MERT}/{sid}.pt" if os.path.exists(f"{MERT}/{sid}.pt") else f"/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/G2/mert_val/{sid}.pt"
        m = torch.load(p, map_location="cpu", weights_only=True)[0, 6:13].float().mean(0)[:, :T].t().numpy()
        cache[sid] = np.pad(m, ((0, T - len(m)), (0, 0))) if len(m) < T else m
    return cache[sid]


class Enc(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(din, 256, 5, padding=2), nn.GELU(), nn.Conv1d(256, 256, 5, padding=4, dilation=2), nn.GELU(),
                                 nn.Conv1d(256, 256, 5, padding=8, dilation=4), nn.GELU(), nn.Conv1d(256, 256, 1))

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


ck = torch.load(ARGS[0], map_location="cpu")
em, ea = Enc(72), Enc(1024)
em.load_state_dict({k[3:]: v for k, v in ck["state"].items() if k.startswith("em.")})
ea.load_state_dict({k[3:]: v for k, v in ck["state"].items() if k.startswith("ea.")})
em.eval(); ea.eval()


@torch.no_grad()
def score(mf, af):
    m = (torch.tensor(mf)[None] - ck["mu_m"]) / ck["sd_m"]
    x = (torch.tensor(af)[None] - ck["mu_a"]) / ck["sd_a"]
    a, b = F.normalize(em(m), dim=-1), F.normalize(ea(x), dim=-1)
    return float((a * b).sum(-1).mean())


def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return float((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())


G = {}
for man, kd in zip(ARGS[1::2], ARGS[2::2]):
    for r in csv.DictReader(open(man)):
        f = os.path.join(kd, r["key"].replace("/", "__") + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        if np.isfinite(z["kps"]).mean() < 0.3:
            continue
        mf = motion_feat(z["kps"])
        G.setdefault((r["tag"], r["cond"]), []).append((score(mf, music_feat(r["audio"])), score(mf, music_feat(r["alt_audio"]))))
print("| tag | cond | n | D(given) | D(alt) | Δ=given−alt | AUC |")
print("|---|---|---|---|---|---|---|")
for (t, c), L in sorted(G.items()):
    L = np.array(L); d = L[:, 0] - L[:, 1]
    print(f"| {t} | {c} | {len(L)} | {L[:, 0].mean():.3f} | {L[:, 1].mean():.3f} | {d.mean():+.3f}±{d.std() / np.sqrt(len(d)):.3f} | {auc(L[:, 0], L[:, 1]):.3f} |")
