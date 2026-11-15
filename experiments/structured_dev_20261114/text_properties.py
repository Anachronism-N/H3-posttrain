#!/usr/bin/env python3
"""Objective text-level properties of a choreography plan (no judge, no audio).

  duplicate_rate   share of events whose action text is a near-duplicate of an
                   earlier event in the same song (cosine >= 0.90 on action words)
  exact_duplicate  share of events repeating an earlier action verbatim
  adjacency_jaccard mean Jaccard overlap of content words between consecutive
                   events (higher = more continuous phrasing, but too high = monotone)
  distinct_ratio   unique action strings / events

These are literal text measurements. They say nothing about musical quality and
must not be reported as one.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

WORDS = re.compile(r"[a-z]+")
STOP = set("""a an the and or of to in on with while as into then its their his her
both each other another this that these those it is are be by from at for""".split())


def words(text):
    return [w for w in WORDS.findall(text.lower()) if w not in STOP]


def cosine(a, b):
    ca, cb = Counter(a), Counter(b)
    shared = set(ca) & set(cb)
    numerator = sum(ca[w] * cb[w] for w in shared)
    denominator = math.sqrt(sum(v * v for v in ca.values()) * sum(v * v for v in cb.values()))
    return numerator / denominator if denominator else 0.0


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def measure(plan):
    events = [e for e in plan.get("events", []) if e.get("action")]
    texts = [e["action"] for e in events]
    tokenised = [words(t) for t in texts]
    duplicates = 0
    for index in range(len(tokenised)):
        if any(cosine(tokenised[index], tokenised[j]) >= 0.90 for j in range(index)):
            duplicates += 1
    exact = len(texts) - len(set(texts))
    adjacent = ([jaccard(tokenised[i], tokenised[i + 1]) for i in range(len(tokenised) - 1)]
                if len(tokenised) > 1 else [])
    return dict(events=len(events), duplicate_rate=round(duplicates / len(events), 3) if events else None,
                exact_duplicate_rate=round(exact / len(events), 3) if events else None,
                adjacency_jaccard=round(sum(adjacent) / len(adjacent), 3) if adjacent else None,
                distinct_ratio=round(len(set(texts)) / len(events), 3) if events else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="directory with <arm>/<song>/plan.json")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for arm in args.arms:
        for plan_path in sorted((Path(args.runs) / arm).glob("*/plan.json")):
            plan = json.loads(plan_path.read_text())
            rows.append(dict(arm=arm, song=plan_path.parent.name, **measure(plan)))
    payload = dict(schema="choreobench.text_properties.v1", rows=rows,
                   note="Literal text measurements of the finished plan; not a quality score.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    for arm in args.arms:
        mine = [r for r in rows if r["arm"] == arm and r["events"]]
        if not mine:
            continue
        print(arm, "songs", len(mine),
              "dup", round(sum(r["duplicate_rate"] for r in mine) / len(mine), 3),
              "exact", round(sum(r["exact_duplicate_rate"] for r in mine) / len(mine), 3),
              "adj", round(sum(r["adjacency_jaccard"] for r in mine) / len(mine), 3),
              "distinct", round(sum(r["distinct_ratio"] for r in mine) / len(mine), 3))


if __name__ == "__main__":
    main()
