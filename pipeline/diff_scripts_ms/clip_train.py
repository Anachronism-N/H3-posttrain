"""舞蹈配乐音频 encoder ↔ 冻结 H3 文本 encoder 的 CLIP 对齐（只训练音频侧；文本侧冻结，可选一个线性头）。
音频: MERT-330M 缓存 [1,25,1024,81]（只来自舞蹈视频的配乐）→ 取 7 层、时间 3 倍下采样 → 可学习层加权 + 2 层 Transformer + 注意力池化。
文本: text_embed.py 产出的冻结 H3 caption 嵌入（5120d），按训练集均值/方差标准化（无参数）。
损失: 对称 InfoNCE；目标用 caption-caption 相似度做软标签（同舞种/近似 caption 不当硬负例）。
评测: val 上 音频→caption 检索 R@1/5/10、检索结果的粗舞种/能量/位移等维度命中率（对比随机与多数类基线）。
用法:
  python clip_train.py prep --out feats.pt            # 压缩 MERT 缓存（CPU 多进程）
  python clip_train.py train --feats feats.pt --text text_emb.pt --caps captions.json --out DIR [--text_head]
"""
import argparse, json, os, random, sys, time, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

MERT = '/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/mert_features/{}.pt'
LAYERS = [0, 4, 8, 12, 16, 20, 24]


def _reduce(sid):
    try:
        x = torch.load(MERT.format(sid), map_location='cpu', weights_only=False)[0]  # [25,1024,T]
        x = x[LAYERS].float()
        T = x.shape[-1] // 3 * 3
        x = x[..., :T].reshape(len(LAYERS), 1024, T // 3, 3).mean(-1)  # [7,1024,27]
        return sid, x.permute(0, 2, 1).contiguous().half()             # [7,27,1024]
    except Exception as e:
        return sid, None


class AudioEnc(nn.Module):
    def __init__(self, d_out, nl=len(LAYERS), d=512, nlayer=2, drop=0.1):
        super().__init__()
        self.lw = nn.Parameter(torch.zeros(nl))
        self.inp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
        self.pos = nn.Parameter(torch.zeros(1, 64, d))
        L = nn.TransformerEncoderLayer(d, 8, 4 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(L, nlayer)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.att = nn.MultiheadAttention(d, 8, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_out))

    def forward(self, x):  # x [B,7,T,1024]
        w = torch.softmax(self.lw, 0)
        h = (x * w[None, :, None, None]).sum(1)
        h = self.inp(h) + self.pos[:, :h.shape[1]]
        h = self.tf(h)
        p, _ = self.att(self.q.expand(h.shape[0], -1, -1), h, h)
        return self.out(p[:, 0])


def dims_hit(caps, q_sids, r_sids, keys):
    out = {}
    for k in keys:
        out[k] = float(np.mean([caps[a]['dims'][k] == caps[b]['dims'][k] for a, b in zip(q_sids, r_sids)]))
    return out


def train(a):
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    feats = torch.load(a.feats); TE = torch.load(a.text); caps = json.load(open(a.caps))
    sids = sorted(set(feats) & set(TE) & set(caps))
    rng = random.Random(0); val = set(rng.sample(sids, int(len(sids) * a.val_frac)))
    tr = [s for s in sids if s not in val]; va = [s for s in sids if s in val]
    print(f'train {len(tr)} val {len(va)}', flush=True)
    X = torch.stack([feats[s] for s in sids]); Tm = torch.stack([TE[s].float() for s in sids])
    idx = {s: i for i, s in enumerate(sids)}
    tri = torch.tensor([idx[s] for s in tr]); vai = torch.tensor([idx[s] for s in va])
    mu, sd = Tm[tri].mean(0), Tm[tri].std(0) + 1e-4
    Tz = (Tm - mu) / sd
    if a.pca:  # 白化后再 PCA 到较低维，去掉高维噪声
        U, S, V = torch.pca_lowrank(Tz[tri], q=a.pca, center=False)
        Tz = Tz @ V
    dev = 'cuda'
    X, Tz = X.to(dev), Tz.to(dev)
    D = Tz.shape[1]
    d_emb = a.d_emb if a.text_head else D
    enc = AudioEnc(d_emb).to(dev)
    head = nn.Linear(D, d_emb, bias=False).to(dev) if a.text_head else None
    params = list(enc.parameters()) + (list(head.parameters()) if head else [])
    ls = nn.Parameter(torch.tensor(np.log(1 / 0.07), device=dev))
    opt = torch.optim.AdamW(params + [ls], lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(tr) // a.bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    tvec = lambda ii: F.normalize(head(Tz[ii]) if head else Tz[ii], dim=-1)
    Tn_all = F.normalize(Tz, dim=-1)

    def evaluate(ii, tag):
        enc.eval()
        with torch.no_grad():
            A = torch.cat([F.normalize(enc(X[ii[j:j + 512]].float()), dim=-1) for j in range(0, len(ii), 512)])
            T = tvec(ii)
            S = A @ T.t()
            rk = (S > S.diag()[:, None]).sum(1)
            r = {f'R@{k}': float((rk < k).float().mean()) for k in (1, 5, 10)}
            r['medR'] = float(rk.float().median() + 1)
            nn_ = S.argmax(1).cpu().numpy()
            q = [sids[i] for i in ii.tolist()]; rr = [sids[ii[j]] for j in nn_]
            r.update({f'hit_{k}': v for k, v in dims_hit(caps, q, rr, ['coarse', 'genre', 'energy', 'tempo', 'travel', 'quality']).items()})
            # 随机检索基线
            perm = np.random.permutation(len(q)); r.update({f'rand_{k}': v for k, v in dims_hit(caps, q, [q[p] for p in perm], ['coarse', 'genre', 'energy']).items()})
        enc.train(); return r

    best, step, t0 = -1, 0, time.time()
    for ep in range(a.epochs):
        perm = tri[torch.randperm(len(tri))]
        for j in range(0, len(perm) - a.bs + 1, a.bs):
            ii = perm[j:j + a.bs].to(dev)
            x = X[ii].float()
            if a.aug:  # 时间随机裁剪/平移 + 特征 dropout
                sh = random.randint(0, 4); x = torch.roll(x, sh, dims=2)
                x = F.dropout(x, 0.1)
            A = F.normalize(enc(x), dim=-1); T = tvec(ii)
            logit = ls.exp().clamp(max=100) * A @ T.t()
            with torch.no_grad():
                tt = Tn_all[ii] @ Tn_all[ii].t()
                tgt = torch.softmax(tt / a.soft_tau, 1) if a.soft_tau > 0 else torch.eye(len(ii), device=dev)
            loss = 0.5 * (-(tgt * F.log_softmax(logit, 1)).sum(1).mean() - (tgt.t() * F.log_softmax(logit.t(), 1)).sum(1).mean())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); step += 1
            assert torch.isfinite(loss), 'non-finite loss'
        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            r = evaluate(vai, 'val')
            print(f'ep {ep + 1} loss {loss.item():.3f} T {1 / ls.exp().item():.3f} ' + ' '.join(f'{k}={v:.3f}' for k, v in r.items()) + f' {time.time() - t0:.0f}s', flush=True)
            score = r['R@10'] + r['hit_coarse']
            if score > best:
                best = score
                torch.save(dict(enc=enc.state_dict(), head=head.state_dict() if head else None, mu=mu, sd=sd, args=vars(a), val=sorted(va), metrics=r), os.path.join(a.out, 'best.pt'))
    json.dump(dict(best=r, args=vars(a)), open(os.path.join(a.out, 'final_metrics.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd')
    ap.add_argument('--out', required=True); ap.add_argument('--ids', default='/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/ids_all.txt')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--feats'); ap.add_argument('--text'); ap.add_argument('--caps')
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--val_frac', type=float, default=0.1); ap.add_argument('--soft_tau', type=float, default=0.05)
    ap.add_argument('--text_head', action='store_true'); ap.add_argument('--d_emb', type=int, default=512); ap.add_argument('--pca', type=int, default=0)
    ap.add_argument('--aug', action='store_true'); ap.add_argument('--eval_every', type=int, default=5); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'prep':
        from multiprocessing import Pool
        ids = [l.strip() for l in open(a.ids) if l.strip()]
        out = {}
        with Pool(a.workers) as p:
            for i, (s, x) in enumerate(p.imap_unordered(_reduce, ids, chunksize=8)):
                if x is not None: out[s] = x
                if (i + 1) % 2000 == 0: print(i + 1, flush=True)
        torch.save(out, a.out); print('DONE', len(out))
    else:
        os.makedirs(a.out, exist_ok=True); train(a)
