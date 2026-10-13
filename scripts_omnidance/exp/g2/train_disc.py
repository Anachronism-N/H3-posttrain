"""G2: music-motion contrastive discriminator (replacement for beat metrics as RL reward).

motion: DWPose body keypoints [T,18,2] -> normalised (centre on neck, scale by torso), velocity -> [T, 72]
music:  cached MERT features [1,25,1024,T] -> mean of layers 6..12 -> [T,1024]  (T = 81 video frames @16fps)
Encoders: small temporal conv nets -> per-frame embeddings -> mean -> 256-d, cosine similarity / tau.
Loss: InfoNCE over the batch (other clips' music = negatives) + hard negatives = own music shifted by 6..20 frames.
Report: held-out AUC (own vs other music) and AUC (own vs shifted own music); G2 needs >= 0.7 on GT val clips.
python train_disc.py --pose_dir <dir> --csv <csv> --out <dir>
"""
import argparse, csv, json, os, random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--pose_dir", required=True)
p.add_argument("--csv", required=True)
p.add_argument("--out", required=True)
p.add_argument("--music_dir", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/G2/music")
p.add_argument("--val_frac", type=float, default=0.1)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--tau", type=float, default=0.07)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0); random.seed(0); np.random.seed(0)
T = 81


def motion_feat(z):
    k = z["kps"].astype(np.float32)[:T]
    neck = k[:, 1:2]
    hip = np.nanmean(k[:, [8, 11]], axis=1, keepdims=True)
    torso = np.nanmedian(np.linalg.norm(neck - hip, axis=-1)) + 1e-3
    x = (k - neck) / torso
    x = np.nan_to_num(x, nan=0.0)
    v = np.diff(x, axis=0, prepend=x[:1])
    f = np.concatenate([x.reshape(len(x), -1), v.reshape(len(v), -1)], -1)
    if len(f) < T:
        f = np.pad(f, ((0, T - len(f)), (0, 0)))
    return f


def music_feat(sid):
    m = np.load(f"{a.music_dir}/{sid}.npy").astype(np.float32)  # [<=T, 1024]
    if len(m) < T:
        m = np.pad(m, ((0, T - len(m)), (0, 0)))
    return m


from concurrent.futures import ThreadPoolExecutor
csv.field_size_limit(10 ** 9)
PACK = os.path.join(a.out, "packed.npz")
have_p = {f[:-4] for f in os.listdir(a.pose_dir) if f.endswith(".npz")}
have_m = {f[:-4] for f in os.listdir(a.music_dir) if f.endswith(".npy")}
rows = [r for r in csv.DictReader(open(a.csv)) if r["sample_id"] in have_p and r["sample_id"] in have_m]
def load_one(r):
    z = np.load(f"{a.pose_dir}/{r['sample_id']}.npz")
    if np.isfinite(z["kps"]).mean() < 0.6:
        return None
    return motion_feat(z), music_feat(r["sample_id"]).astype(np.float16), r["sample_id"]


if os.path.exists(PACK):
    zz = np.load(PACK); M, A, ids = zz["M"], zz["A"].astype(np.float32), list(zz["ids"])
else:
    with ThreadPoolExecutor(32) as ex:
        res = [x for x in ex.map(load_one, rows) if x is not None]
    M = np.stack([x[0] for x in res]); A = np.stack([x[1] for x in res]); ids = [x[2] for x in res]
    np.savez(PACK, M=M, A=A, ids=np.array(ids))
    A = A.astype(np.float32)
print(f"[disc] loaded {len(ids)} clips", flush=True)
M = torch.tensor(M); A = torch.tensor(A)
mu_m, sd_m = M.mean((0, 1)), M.std((0, 1)) + 1e-4
mu_a, sd_a = A.mean((0, 1)), A.std((0, 1)) + 1e-4
M = (M - mu_m) / sd_m; A = (A - mu_a) / sd_a
n = len(M); perm = torch.randperm(n); nv = int(n * a.val_frac)
vi, ti = perm[:nv], perm[nv:]
print(f"[disc] clips {n} train {len(ti)} val {nv}", flush=True)


class Enc(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(din, 256, 5, padding=2), nn.GELU(), nn.Conv1d(256, 256, 5, padding=4, dilation=2), nn.GELU(),
                                 nn.Conv1d(256, 256, 5, padding=8, dilation=4), nn.GELU(), nn.Conv1d(256, 256, 1))

    def forward(self, x):  # [B,T,D]
        return self.net(x.transpose(1, 2)).transpose(1, 2)  # [B,T,256]


class Disc(nn.Module):
    """frame-aligned: score = mean_t cos(m_t, a_t) — sensitive to temporal alignment, not only to global style."""
    def __init__(self):
        super().__init__()
        self.em, self.ea = Enc(M.shape[-1]), Enc(A.shape[-1])

    def emb(self, m, x):
        return F.normalize(self.em(m), dim=-1), F.normalize(self.ea(x), dim=-1)

    def pair(self, m, x):  # [B,B] score matrix
        em, ea = self.emb(m, x)
        return torch.einsum("itd,jtd->ij", em, ea) / em.shape[1]


dev = "cuda"
D = Disc().to(dev)
opt = torch.optim.AdamW(D.parameters(), lr=a.lr, weight_decay=1e-2)


def shift(x):
    s = random.randint(6, 20) * random.choice([-1, 1])
    return torch.roll(x, s, dims=1)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


@torch.no_grad()
def evaluate(idx):
    D.eval()
    m, x = M[idx].to(dev), A[idx].to(dev)
    S = D.pair(m, x).cpu().numpy()
    pos = np.diag(S); neg = S[~np.eye(len(S), dtype=bool)]
    em, _ = D.emb(m, x)
    _, eas = D.emb(m, shift(x))
    sh = (em * eas).sum(-1).mean(1).cpu().numpy()
    D.train()
    return auc(pos, neg), auc(pos, sh)


best = -1
for ep in range(a.epochs):
    order = ti[torch.randperm(len(ti))]
    tot = 0.0
    for b in range(0, len(order) - a.bs + 1, a.bs):
        bi = order[b:b + a.bs]
        m, x = M[bi].to(dev), A[bi].to(dev)
        S = D.pair(m, x) * (1 / a.tau)
        em, _ = D.emb(m, x)
        _, eas = D.emb(m, shift(x))
        hard = (em * eas).sum(-1).mean(1) / a.tau  # own music shifted
        logits = torch.cat([S, hard[:, None]], 1)
        lab = torch.arange(len(bi), device=dev)
        loss = F.cross_entropy(logits, lab) + F.cross_entropy(S.t(), lab)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    au, ash = evaluate(vi)
    print(f"[disc] ep {ep} loss {tot:.2f} val AUC other {au:.3f} shift {ash:.3f}", flush=True)
    if au + ash > best:
        best = au + ash
        torch.save(dict(state=D.state_dict(), mu_m=mu_m, sd_m=sd_m, mu_a=mu_a, sd_a=sd_a), f"{a.out}/disc.pt")
        json.dump(dict(ep=ep, auc_other=au, auc_shift=ash, n_train=len(ti), n_val=nv), open(f"{a.out}/best.json", "w"))
print("[disc] done", flush=True)
