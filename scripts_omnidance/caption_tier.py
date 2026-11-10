#!/usr/bin/env python3
"""Screen captions into quality tiers by templating / compression signals."""
import json, re, sys, collections

PATH = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/caption_tiers.json"

TEMPLATE = re.compile(
    r"^(?:one side|the other side)?\s*(?:wrist|elbow|knee|shoulder|hip|ankle|foot)\b", re.I)
LINK_ONLY = re.compile(r"^(?:then|and then|then the other side|continues?)\b", re.I)


def body(cap):
    out = []
    for ln in cap.split("\n"):
        if ln.startswith("Dance style:"):
            continue
        out.append(re.sub(r"^\[[^\]]+\]\s*", "", ln))
    return "\n".join(out)


def sents(txt):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", txt) if s.strip()]


def analyse(cap):
    b = body(cap)
    ss = sents(b)
    n = len(ss)
    uniq = len({s.lower() for s in ss})
    dup = 1 - uniq / n if n else 0
    tmpl = sum(1 for s in ss if TEMPLATE.match(s)) / n if n else 0
    words = re.findall(r"[a-z]+", b.lower())
    # 8-gram repetition
    g = collections.Counter(tuple(words[i:i + 8]) for i in range(len(words) - 7))
    rep8 = sum(c - 1 for c in g.values() if c > 1) / max(len(words) - 7, 1)
    # narrative signal: connectors / verbs of travel / orientation words
    narr = len(re.findall(
        r"\b(step|steps|turn|turns|pivot|travel|walk|lunge|shift(?:s)? weight|"
        r"faces?|facing|reaches?|extends?|lifts?|raises?|swings?|rotates?|"
        r"leans?|crouches?|jumps?|slides?|cross(?:es)?|opens?|closes?)\b", b, re.I))
    narr_per_1k = narr / max(len(words) / 1000, 1)
    # how much the sentence count collapsed vs source event count
    return dict(n_sent=n, dup=dup, tmpl=tmpl, rep8=rep8, narr=narr_per_1k, words=len(words))


def main():
    rows = []
    with open(PATH) as f:
        for line in f:
            d = json.loads(line)
            m = analyse(d["motion_caption"])
            m["id"] = d["id"]
            rows.append(m)
    N = len(rows)
    # tier: good = low template ratio, low dup, decent narrative density
    def tier(m):
        if m["tmpl"] >= 0.75 and m["dup"] >= 0.35:
            return "C_raw_template"
        if m["tmpl"] >= 0.9:
            return "C_raw_template"
        if m["dup"] >= 0.25 or m["rep8"] >= 0.35:
            return "B_partly_rewritten"
        if m["narr"] < 8 and m["tmpl"] >= 0.5:
            return "B_partly_rewritten"
        return "A_narrative"
    tiers = collections.Counter(tier(m) for m in rows)
    print(f"records={N}")
    for k, v in sorted(tiers.items()):
        print(f"{k:>16}: {v:>7} ({v/N*100:.1f}%)")
    print()
    for k in sorted(tiers):
        sub = [m for m in rows if tier(m) == k]
        if not sub:
            continue
        n = len(sub)
        print(f"--- {k} (n={n})")
        for f in ("tmpl", "dup", "rep8", "narr"):
            v = sorted(x[f] for x in sub)
            print(f"   {f:>5}: p10={v[n//10]:.3f} p50={v[n//2]:.3f} p90={v[n*9//10]:.3f}")
        print(f"   mean sentences {sum(x['n_sent'] for x in sub)/n:.1f}, mean words {sum(x['words'] for x in sub)/n:.0f}")
    json.dump({"tiers": {k: [m['id'] for m in rows if tier(m) == k] for k in sorted(tiers)},
               "metrics": rows}, open(OUT, "w"))


if __name__ == "__main__":
    main()
