#!/usr/bin/env python3
"""Aggregate one or more model-judge files into a paired result with a bootstrap CI.

Inputs are the judge outputs (positional A/B) plus the private bindings that map
positions back to method names. Deltas are always treatment minus baseline, so
swapped pairs cannot flip the sign.
"""
import argparse
import json
import random
from pathlib import Path

CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", action="append", required=True, help="judge JSON")
    parser.add_argument("--bindings", action="append", required=True, help="matching private bindings JSON")
    parser.add_argument("--baseline", default="B")
    parser.add_argument("--treatment", default="S4")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len(args.ratings) != len(args.bindings):
        raise SystemExit("pass one bindings file per ratings file")
    rows = {}
    for ratings_path, bindings_path in zip(args.ratings, args.bindings):
        judge = json.loads(Path(ratings_path).read_text())
        bindings = {b["pair_id"]: b for b in json.loads(Path(bindings_path).read_text())["pairs"]}
        for pair in judge["pairs"]:
            if pair.get("error"):
                continue
            binding = bindings[pair["pair_id"]]
            score = {binding["A"]: pair["scores"]["A"], binding["B"]: pair["scores"]["B"]}
            if args.baseline not in score or args.treatment not in score:
                continue
            base, treat = score[args.baseline], score[args.treatment]
            if any(base[c] is None or treat[c] is None for c in CRITERIA):
                continue
            rows[pair["song_id"]] = dict(
                delta={c: treat[c] - base[c] for c in CRITERIA},
                total_delta=sum(treat[c] for c in CRITERIA) - sum(base[c] for c in CRITERIA))
    songs = sorted(rows)
    if not songs:
        raise SystemExit("no complete pairs to aggregate")
    totals = [rows[s]["total_delta"] for s in songs]
    mean_total = sum(totals) / len(totals)
    random.seed(args.seed)
    boots = sorted(sum(random.choice(totals) for _ in totals) / len(totals) for _ in range(args.resamples))
    lo, hi = boots[int(0.025 * args.resamples)], boots[int(0.975 * args.resamples) - 1]
    per_criterion = {c: round(sum(rows[s]["delta"][c] for s in songs) / len(songs), 2) for c in CRITERIA}
    report = dict(
        schema="choreobench.judge_aggregate.v1", baseline=args.baseline, treatment=args.treatment,
        songs=songs, n=len(songs), per_song={s: rows[s] for s in songs},
        per_criterion_mean_delta=per_criterion,
        total_mean_delta=round(mean_total, 2), bootstrap_ci95=[round(lo, 2), round(hi, 2)],
        resamples=args.resamples, wins=dict(treatment=sum(1 for t in totals if t > 0),
                                            baseline=sum(1 for t in totals if t < 0),
                                            tie=sum(1 for t in totals if t == 0)),
        excludes_zero=bool(lo > 0 or hi < 0),
        note=("Model-assisted paired comparison. Deltas are treatment minus baseline on a 1-5 scale per "
              "criterion; song-level bootstrap over whole songs. Not human review."))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("n", "per_criterion_mean_delta", "total_mean_delta",
                                             "bootstrap_ci95", "wins", "excludes_zero")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
