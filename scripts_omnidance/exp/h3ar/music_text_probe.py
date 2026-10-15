"""music_text_probe.py (10-15, stage-0 of the mentor's music->text autoencoder idea).
Before training any music->text-space encoder, check whether the music carries information about what the (VLM) caption
of the paired video says about the dance. Labels are derived lexically from the motion sections ([1] Technical Execution,
[2] Choreographic Content) of the rich prompt: energy (high vs low words), tempo (fast vs slow words) and genre keywords.
Features: MERT-v1-330M cache, mean over time of 4 layers -> 4096-d. Linear logistic probe, held-out AUC; a label-shuffled
probe gives the chance level. AUC ~0.5 -> music does not predict the caption -> a music->text encoder cannot learn it.
"""
import argparse, glob, json, os, random, re
import numpy as np, torch
from concurrent.futures import ThreadPoolExecutor

p = argparse.ArgumentParser()
p.add_argument("--prompts", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/all_rich_prompts.json")
p.add_argument("--mert", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/mert_features")
p.add_argument("--n", type=int, default=3000); p.add_argument("--out", required=True)
a = p.parse_args()
P = json.load(open(a.prompts))
ids = [s for s in P if os.path.exists(f"{a.mert}/{s}.pt")]
random.Random(0).shuffle(ids); ids = ids[:a.n]
print("clips", len(ids), flush=True)
LEX = {"energy_hi": r"\b(explosive|powerful|energetic|high-energy|vigorous|dynamic|intense|sharp|forceful|hard-hitting|aggressive)\b",
       "energy_lo": r"\b(gentle|soft|subtle|relaxed|calm|delicate|graceful|smooth|restrained|light)\b",
       "fast": r"\b(fast|quick|rapid|brisk|swift|upbeat)\b", "slow": r"\b(slow|languid|unhurried|leisurely|sustained)\b",
       "hiphop": r"\b(hip-hop|hip hop|street|urban|krump|breaking|popping|locking)\b", "kpop": r"\b(k-pop|kpop)\b",
       "jazz": r"\bjazz\b", "contemporary": r"\b(contemporary|lyrical|modern)\b", "latin": r"\b(latin|salsa|samba|bachata)\b",
       "chinese": r"\b(chinese|classical|traditional)\b"}


def motion_text(s):
    m = re.search(r"\[1\. Technical Execution.*?(?=\[3\.|\Z)", s, re.S)
    return (m.group(0) if m else s).lower()


Y = {k: np.array([bool(re.search(v, motion_text(P[s]))) for s in ids]) for k, v in LEX.items()}


def feat(s):
    x = torch.load(f"{a.mert}/{s}.pt", map_location="cpu", weights_only=False).float()  # [1,25,1024,T]
    return torch.cat([x[0, l].mean(-1) for l in (6, 12, 18, 24)]).numpy()


with ThreadPoolExecutor(16) as ex:
    X = np.stack(list(ex.map(feat, ids)))
X = (X - X.mean(0)) / (X.std(0) + 1e-6)
ntr = int(0.8 * len(ids))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
res = {}
for k, y in Y.items():
    if y[:ntr].sum() < 20 or y[ntr:].sum() < 5 or (~y[ntr:]).sum() < 5:
        res[k] = {"pos_rate": float(y.mean()), "skip": True}; continue
    def fit(yy):
        c = LogisticRegression(C=0.01, max_iter=2000).fit(X[:ntr], yy[:ntr]); return roc_auc_score(yy[ntr:], c.predict_proba(X[ntr:])[:, 1])
    ys = y.copy(); np.random.default_rng(0).shuffle(ys)
    res[k] = {"pos_rate": round(float(y.mean()), 3), "auc": round(float(fit(y)), 3), "auc_shuffled": round(float(fit(ys)), 3)}
    print(k, res[k], flush=True)
json.dump({"n": len(ids), "res": res}, open(a.out, "w"), indent=1)
print("written", a.out)
