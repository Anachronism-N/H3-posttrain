"""genre_clip.py (10-15, mentor music->H3-text-space CLIP, G1).
Text side (frozen): H3's own text encoder input embedding table (Qwen3-VL embed_tokens) applied to one caption per music
genre. Because G0 showed H3 ignores bare genre words, each caption states the genre AND the matching dance (high-energy /
medium / calm). Captions are tokenised with H3's tokenizer and right-padded with '.' to a common length K -> T_c [K,5120].
Audio side (trained): MERT-v1-330M time-mean of layers 6/12/18/24 (4096) -> MLP -> 5120, CLIP logits = cos(a, mean(T_c))/tau.
At inference the predicted class distribution p mixes the token sequences: sum_c p_c T_c (soft prompt), which is written into
H3's text-encoder input at placeholder positions (sample_h3_ar.py --audio_text).
python genre_clip.py --feat GENRE8/feat --out GENRE8/clip     (train + eval, CPU/GPU)
python genre_clip.py --embed_only ...                          (only build text targets)
"""
import argparse, glob, json, os, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

CAP = {  # genre -> caption body (inside H3's template, replaces the per-video description)
    "rock": "<Subject 1> dances to loud rock music with big, explosive, high-energy moves: wide arm swings above the head, high kicks and jumps.",
    "hiphop": "<Subject 1> dances to hip-hop music with big, hard-hitting, high-energy moves: wide arm swings, sharp hits, kicks and jumps.",
    "dance": "<Subject 1> dances to electronic dance music with big, bouncy, high-energy moves: wide arm swings above the head, kicks and jumps.",
    "funk": "<Subject 1> dances to funk music with big, groovy, high-energy moves: wide arm swings, hip bounces, kicks and jumps.",
    "jazz": "<Subject 1> dances to jazz music with moderate, swinging steps and clear, relaxed arm gestures.",
    "orchestral": "<Subject 1> dances to orchestral music with moderate, flowing steps and graceful, clear arm gestures.",
    "zen": "<Subject 1> dances very calmly to soft meditative music, barely moving: gentle swaying and soft hand gestures, feet staying in place.",
    "chinese": "<Subject 1> dances calmly to soft traditional Chinese music, barely moving: gentle swaying and soft hand gestures, feet staying in place.",
}
GENRES = list(CAP)
LAYERS = (6, 12, 18, 24)


def text_targets(w="/dev/shm/h3/Ref2VA"):
    from transformers import AutoTokenizer
    from safetensors import safe_open
    tok = AutoTokenizer.from_pretrained(f"{w}/processor")
    ids = [tok(CAP[g], add_special_tokens=False)["input_ids"] for g in GENRES]
    dot = tok(".", add_special_tokens=False)["input_ids"][0]
    K = max(map(len, ids)); ids = [x + [dot] * (K - len(x)) for x in ids]
    idx = json.load(open(f"{w}/text_encoder/model.safetensors.index.json"))["weight_map"]
    key = "model.language_model.embed_tokens.weight"
    with safe_open(f"{w}/text_encoder/{idx[key]}", "pt") as f:
        E = f.get_slice(key)
        T = torch.stack([torch.stack([E[i:i + 1][0] for i in x]) for x in ids]).float()  # [C,K,5120]
    return T, ids, K


class AudioEnc(nn.Module):
    def __init__(self, d_in=4096, d_out=5120):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, d_out))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def mert_vec(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = d["mean"] if isinstance(d, dict) else d[0].float().mean(-1)  # GENRE8 dict or [1,25,1024,T] cache
    return torch.cat([m[l].float() for l in LAYERS])


def logits(enc, x, Tm):
    return enc.logit_scale.exp().clamp(max=100) * enc(x) @ F.normalize(Tm, dim=-1).t()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feat", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--embed_only", action="store_true")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    T, ids, K = text_targets()
    torch.save({"T": T.to(torch.bfloat16), "ids": ids, "K": K, "genres": GENRES, "captions": CAP}, f"{a.out}/text_targets.pt")
    Tm = T.mean(1)  # [C,5120] pooled caption embedding (frozen text side)
    print("text targets", tuple(T.shape), "pairwise cos of pooled captions:\n", (F.normalize(Tm, dim=-1) @ F.normalize(Tm, dim=-1).t()).numpy().round(3))
    if a.embed_only:
        raise SystemExit
    items = [(g, f) for g in GENRES for f in sorted(glob.glob(f"{a.feat}/{g}/*.pt"))]
    random.Random(0).shuffle(items)
    print("items", len(items), {g: sum(1 for x in items if x[0] == g) for g in GENRES}, flush=True)
    X = torch.stack([mert_vec(f) for _, f in items]); Y = torch.tensor([GENRES.index(g) for g, _ in items])
    mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    nv = len(items) // 10; Xv, Yv, Xt, Yt = X[:nv], Y[:nv], X[nv:], Y[nv:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = AudioEnc().to(dev); opt = torch.optim.AdamW(enc.parameters(), lr=3e-4, weight_decay=0.05); Tmd = Tm.to(dev)
    best = 0
    for ep in range(a.epochs):
        enc.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; loss = F.cross_entropy(logits(enc, Xt[b].to(dev), Tmd), Yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval()
        with torch.no_grad():
            pv = logits(enc, Xv.to(dev), Tmd).argmax(1).cpu()
        acc = float((pv == Yv).float().mean())
        if acc >= best:
            best = acc; torch.save({"state": enc.state_dict(), "mu": mu, "sd": sd, "genres": GENRES, "layers": LAYERS, "val_acc": acc}, f"{a.out}/audio_enc.pt")
        print(f"ep {ep} loss {loss.item():.3f} val_acc {acc:.3f}", flush=True)
    ck = torch.load(f"{a.out}/audio_enc.pt", weights_only=False); enc.load_state_dict(ck["state"])
    with torch.no_grad():
        pv = logits(enc, Xv.to(dev), Tmd).argmax(1).cpu()
    cm = np.zeros((len(GENRES), len(GENRES)), int)
    for y, q in zip(Yv.tolist(), pv.tolist()):
        cm[y, q] += 1
    lv = {"rock": 2, "hiphop": 2, "dance": 2, "funk": 2, "jazz": 1, "orchestral": 1, "zen": 0, "chinese": 0}
    L = torch.tensor([lv[g] for g in GENRES])
    res = {"val_acc": ck["val_acc"], "energy3_acc": float((L[pv] == L[Yv]).float().mean()), "confusion": cm.tolist(), "genres": GENRES, "n_val": nv}
    json.dump(res, open(f"{a.out}/eval.json", "w"), indent=1)
    print(json.dumps(res))
