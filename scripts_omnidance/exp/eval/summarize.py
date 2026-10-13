"""Summarize metrics jsonl dirs -> per (tag, cond) table + controllability deltas.
Delta_audio (aswap): BAS(generated vs *its conditioning* music) - BAS(vs original music)  >0 => follows given audio
Delta_text  (tswap): TA(new prompt) - TA(original prompt)                                   >0 => follows given text
Also: joint BAS vs a mismatched music (bas_alt) = chance-level reference.
Usage: summarize.py <out.md> <jsonl dirs or files...>
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

out = sys.argv[1]
recs = []
for p in sys.argv[2:]:
    files = glob.glob(f"{p}/*.jsonl") if os.path.isdir(p) else [p]
    for f in files:
        recs += [json.loads(l) for l in open(f)]
recs = [r for r in recs if "error" not in r]
# de-dup by key (last wins)
recs = list({r["key"]: r for r in recs}.values())
G = defaultdict(list)
for r in recs:
    G[(r["tag"], r["cond"])].append(r)

def m(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return (np.mean(v), np.std(v) / max(1, np.sqrt(len(v)))) if v else (float("nan"), 0)

cols = ["bas", "bas_alt", "ta", "ta_alt", "ffpsnr", "id_dino", "motion"]
lines = ["| tag | cond | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 3)]
for (t, c), L in sorted(G.items()):
    cells = []
    for k in cols:
        mu, se = m([r.get(k) for r in L])
        cells.append("-" if np.isnan(mu) else f"{mu:.3f}±{se:.3f}")
    lines.append(f"| {t} | {c} | {len(L)} | " + " | ".join(cells) + " |")
lines += ["", "### Controllability (paired over samples)", "| tag | Δaudio=BAS(given)-BAS(orig) [aswap] | Δbeat_joint=BAS(own)-BAS(mismatch) [joint] | Δtext=TA(new)-TA(orig) [tswap] | Δtext_joint=TA(own)-TA(other) [joint] |", "|---|---|---|---|---|"]
for t in sorted(set(k[0] for k in G)):
    def d(cond, a, b):
        L = G.get((t, cond), [])
        v = [r[a] - r[b] for r in L if a in r and b in r and np.isfinite(r[a]) and np.isfinite(r[b])]
        if not v:
            return "-"
        mu, se = np.mean(v), np.std(v) / np.sqrt(len(v))
        return f"{mu:+.3f}±{se:.3f} (n={len(v)})"
    lines.append(f"| {t} | {d('aswap','bas','bas_alt')} | {d('joint','bas','bas_alt')} | {d('tswap','ta','ta_alt')} | {d('joint','ta','ta_alt')} |")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
