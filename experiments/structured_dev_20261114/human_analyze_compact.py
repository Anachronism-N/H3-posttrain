#!/usr/bin/env python3
"""Analyze compact (forced-choice) human review answers.

Answers come from human_pack_html.py: per song an overall choice and one choice
per criterion, each "A"/"B"/"无法判断". Positions are mapped back to methods with
the private bindings. Reports per-reviewer wins, cross-reviewer agreement, and
every disagreement, so a small panel cannot be summarized into a false consensus.
"""
import argparse
import itertools
import json
from pathlib import Path

CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")
ABSTAIN = ("无法判断", "cannot tell", "tie", "", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True, help="review-pairs.private.json")
    parser.add_argument("--answers", action="append", required=True, help="answers_*.json per reviewer")
    parser.add_argument("--baseline", default="B")
    parser.add_argument("--treatment", default="S4")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bindings = {p["pair_id"]: p for p in json.loads(Path(args.bindings).read_text())["pairs"]}
    reviewers, rows = [], []
    for path in args.answers:
        payload = json.loads(Path(path).read_text())
        reviewer = payload.get("reviewer_id") or Path(path).stem
        reviewers.append(reviewer)
        for pair in payload["pairs"]:
            binding = bindings.get(pair["pair_id"])
            if binding is None:
                raise SystemExit("unknown pair in answers: " + str(pair["pair_id"]))
            if {binding["A"], binding["B"]} != {args.baseline, args.treatment}:
                continue

            def method(choice):
                if choice in ABSTAIN:
                    return None
                return binding[choice]

            row = dict(reviewer=reviewer, pair_id=pair["pair_id"], song_id=pair["song_id"],
                       heard_audio=pair.get("heard_audio"), reason=pair.get("reason", ""),
                       overall_choice=pair.get("overall"), overall_method=method(pair.get("overall")),
                       criteria={c: method(pair.get(c)) for c in CRITERIA})
            rows.append(row)
    by_reviewer = {}
    for reviewer in reviewers:
        mine = [r for r in rows if r["reviewer"] == reviewer]
        votes = [r["overall_method"] for r in mine]
        by_reviewer[reviewer] = dict(
            songs=len(mine), abstained=sum(1 for v in votes if v is None),
            treatment_wins=sum(1 for v in votes if v == args.treatment),
            baseline_wins=sum(1 for v in votes if v == args.baseline),
            criterion_votes={c: dict(treatment=sum(1 for r in mine if r["criteria"][c] == args.treatment),
                                     baseline=sum(1 for r in mine if r["criteria"][c] == args.baseline),
                                     abstain=sum(1 for r in mine if r["criteria"][c] is None))
                             for c in CRITERIA},
            songs_without_audio=[r["song_id"] for r in mine if r["heard_audio"] is False])
    agreement = []
    for first, second in itertools.combinations(reviewers, 2):
        f = {r["pair_id"]: r["overall_method"] for r in rows if r["reviewer"] == first}
        s = {r["pair_id"]: r["overall_method"] for r in rows if r["reviewer"] == second}
        shared = [p for p in f if p in s and f[p] is not None and s[p] is not None]
        agree = [p for p in shared if f[p] == s[p]]
        observed = len(agree) / len(shared) if shared else None
        # Cohen's kappa for the two-category choice, ties/abstains dropped.
        p_yes = (sum(1 for p in shared if f[p] == args.treatment) / len(shared)
                 + sum(1 for p in shared if s[p] == args.treatment) / len(shared)) / 2 if shared else None
        expected = (p_yes ** 2 + (1 - p_yes) ** 2) if shared else None
        kappa = None if not shared or expected in (None, 1) else (observed - expected) / (1 - expected)
        agreement.append(dict(reviewers=[first, second], shared_pairs=len(shared),
                              observed_agreement=observed, kappa=kappa,
                              disagreements=[dict(pair_id=p, first=f[p], second=s[p])
                                             for p in shared if f[p] != s[p]]))
    combined = {}
    for pair_id in {r["pair_id"] for r in rows}:
        votes = [r["overall_method"] for r in rows if r["pair_id"] == pair_id and r["overall_method"]]
        combined[pair_id] = dict(treatment=sum(1 for v in votes if v == args.treatment),
                                 baseline=sum(1 for v in votes if v == args.baseline), n=len(votes))
    payload = dict(schema="choreobench.human_analysis_compact.v1", baseline=args.baseline,
                   treatment=args.treatment, reviewers=reviewers, per_reviewer=by_reviewer,
                   agreement=agreement, per_song_panel=combined, rows=rows,
                   note="Abstains are excluded, never imputed. With 2 reviewers, disagreement is reported, "
                        "not averaged away.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(reviewers=reviewers, per_reviewer=by_reviewer, agreement=agreement),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
