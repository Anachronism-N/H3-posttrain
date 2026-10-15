"""genre_ae.py (10-16, mentor stage 1: music <-> H3-text autoencoder with CLIP alignment). H3 and its text encoder stay frozen.
Encoder E: MERT-v1-330M per-frame features (layers 6/12/18/24, 81 frames @16 fps for 5.17 s) -> per-frame MLP -> 2-layer
Transformer -> K=32 learned queries (cross-attention) -> K tokens in H3's text-encoder INPUT embedding space (5120-d, the
embed_tokens table). Decoder D: tokens -> mean -> MLP -> reconstructs the (normalised) MERT clip vector (music autoencoder).
Losses (text side = frozen H3 embed_tokens of one "genre + matching dance" caption per class, from genre_clip.py):
  L_clip : InfoNCE between pooled E(music) and the pooled class captions (which caption does this music go with)
  L_tok  : token-wise MSE + (1 - cos) to the caption tokens of the true class (tokens must be readable by frozen H3)
  L_rec  : MSE of D(E(music)) to the MERT clip vector (tokens keep music information beyond the class)
Writes <out>/ae.pt and per-clip tokens for any extra audio (val dance music, g2 clips) via --export.
python genre_ae.py --feat GENRE8/feat --text GENRE8/clip/text_targets.pt --out GENRE8/ae
python genre_ae.py --export_only --out GENRE8/ae --wavs a.wav,b.wav --mert_cache <dir>   (tokens -> <out>/tokens.pt)
"""
import argparse, glob, json, os, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

LAYERS = (6, 12, 18, 24)


class AE(nn.Module):
    def __init__(self, K=32, d=768, d_txt=5120, d_in=4096):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, 96, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        self.q = nn.Parameter(torch.randn(1, K, d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_txt))
        self.dec = nn.Sequential(nn.Linear(d_txt, 1024), nn.GELU(), nn.Linear(1024, d_in))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def encode(self, x):  # x [B, T, 4096]
        h = self.inp(x) + self.pos[:, :x.shape[1]]
        h = self.enc(h)
        z, _ = self.xattn(self.q.expand(x.shape[0], -1, -1), h, h)
        return self.out(z)  # [B, K, 5120]

    def forward(self, x):
        tok = self.encode(x)
        return tok, self.dec(tok.mean(1))


def mert_seq(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        s = d["seq"].float()                        # [4,1024,T]
    else:
        s = d[0][list(LAYERS)].float()              # dance cache [1,25,1024,T]
    s = s[..., :81]
    if s.shape[-1] < 81:
        s = F.pad(s, (0, 81 - s.shape[-1]))
    return s.permute(2, 0, 1).reshape(81, -1)     # [81, 4096]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feat", default=""); p.add_argument("--text", default=""); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=25); p.add_argument("--w_tok", type=float, default=1.0); p.add_argument("--w_rec", type=float, default=0.2)
    p.add_argument("--export_only", action="store_true"); p.add_argument("--wavs", default=""); p.add_argument("--mert_files", default="")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not a.export_only and not a.mert_files:
        TT = torch.load(a.text, weights_only=False); T = TT["T"].float(); G = TT["genres"]   # [C,K,5120]
        items = [(g, f) for g in G for f in sorted(glob.glob(f"{a.feat}/{g}/*.pt"))]
        random.Random(0).shuffle(items)                                                    # same split as genre_clip.py
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(32) as ex:
            X = torch.stack(list(ex.map(mert_seq, [f for _, f in items])))                 # [N,81,4096]
        Y = torch.tensor([G.index(g) for g, _ in items])
        mu, sd = X.mean((0, 1)), X.std((0, 1)) + 1e-6; X = (X - mu) / sd
        R = X.mean(1)                                                                       # reconstruction target
        nv = len(items) // 10; tr = torch.arange(nv, len(items)); va = torch.arange(nv)
        m = AE(K=T.shape[1]).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
        Td = T.to(dev); Tm = F.normalize(Td.mean(1), dim=-1); tsc = Td.pow(2).mean().sqrt()
        best = -1
        for ep in range(a.epochs):
            m.train(); perm = tr[torch.randperm(len(tr))]
            for i in range(0, len(perm), 128):
                b = perm[i:i + 128]; x = X[b].to(dev); y = Y[b].to(dev)
                tok, rec = m(x)
                lc = F.cross_entropy(m.logit_scale.exp().clamp(max=100) * F.normalize(tok.mean(1), dim=-1) @ Tm.t(), y)
                tgt = Td[y]
                lt = F.mse_loss(tok / tsc, tgt / tsc) + (1 - F.cosine_similarity(tok, tgt, dim=-1)).mean()
                lr_ = F.mse_loss(rec, R[b].to(dev))
                loss = lc + a.w_tok * lt + a.w_rec * lr_
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                tok, rec = m(X[va].to(dev)); pv = (F.normalize(tok.mean(1), dim=-1) @ Tm.t()).argmax(1).cpu()
                cos_true = F.cosine_similarity(tok, Td[Y[va].to(dev)], dim=-1).mean().item()
                rel_rec = (F.mse_loss(rec, R[va].to(dev)) / R[va].pow(2).mean()).item()
            acc = float((pv == Y[va]).float().mean())
            print(f"ep {ep} loss {loss.item():.3f} clip {lc.item():.3f} tok {lt.item():.3f} rec {lr_.item():.3f} | val acc {acc:.3f} tokcos {cos_true:.3f} relrec {rel_rec:.3f}", flush=True)
            if acc >= best:
                best = acc; torch.save({"state": m.state_dict(), "mu": mu, "sd": sd, "genres": G, "K": T.shape[1], "val_acc": acc, "tokcos": cos_true}, f"{a.out}/ae.pt")
        lv = {"rock": 2, "hiphop": 2, "dance": 2, "funk": 2, "jazz": 1, "orchestral": 1, "zen": 0, "chinese": 0}
        ck = torch.load(f"{a.out}/ae.pt", weights_only=False); m.load_state_dict(ck["state"])
        with torch.no_grad():
            tok, _ = m(X[va].to(dev)); pv = (F.normalize(tok.mean(1), dim=-1) @ Tm.t()).argmax(1).cpu()
        L = torch.tensor([lv[g] for g in G])
        json.dump({"val_acc": ck["val_acc"], "tokcos": ck["tokcos"], "energy3_acc": float((L[pv] == L[Y[va]]).float().mean())}, open(f"{a.out}/eval.json", "w"), indent=1)
        print(open(f"{a.out}/eval.json").read())
    if a.mert_files:  # export per-clip tokens: comma list of "<key>=<mert .pt path>"
        ck = torch.load(f"{a.out}/ae.pt", weights_only=False); m = AE(K=ck["K"]).to(dev); m.load_state_dict(ck["state"]); m.eval()
        out = {}
        for kv in a.mert_files.split(","):
            k, f = kv.split("=", 1)
            x = ((mert_seq(f) - ck["mu"]) / ck["sd"]).unsqueeze(0).to(dev)
            with torch.no_grad():
                out[k] = m.encode(x)[0].to(torch.bfloat16).cpu()
        torch.save(out, f"{a.out}/tokens.pt"); print("exported", list(out))
