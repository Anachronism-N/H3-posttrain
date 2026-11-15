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


def cohen_kappa(first, second):
    """Binary marginal agreement; undefined for empty or degenerate panels."""
    if len(first) != len(second):
        raise ValueError("Reviewer vectors have different lengths")
    if not first:
        return None
    if any(type(v) is not bool for v in list(first) + list(second)):
        raise ValueError("Kappa expects binary reviewer vectors")
    p1, p2 = sum(first) / len(first), sum(second) / len(second)
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True, help="review-pairs.private.json")
    parser.add_argument("--answers", action="append", required=True, help="answers_*.json per reviewer")
    parser.add_argument("--baseline", default="B")
    parser.add_argument("--treatment", default="O")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.baseline == args.treatment:
        parser.error("baseline and treatment must differ")
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    bindings = {p["pair_id"]: p for p in json.loads(Path(args.bindings).read_text())["pairs"]}
    reviewers, rows = [], []
    for path in args.answers:
        payload = json.loads(Path(path).read_text())
        reviewer = payload.get("reviewer_id") or Path(path).stem
        if reviewer in reviewers:
            raise ValueError("Duplicate reviewer identity: " + reviewer)
        reviewers.append(reviewer)
        seen_pairs = set()
        for pair in payload["pairs"]:
            if pair["pair_id"] in seen_pairs:
                raise ValueError("Repeated pair from the same reviewer")
            seen_pairs.add(pair["pair_id"])
            binding = bindings.get(pair["pair_id"])
            if binding is None:
                raise SystemExit("unknown pair in answers: " + str(pair["pair_id"]))
            if {binding["A"], binding["B"]} != {args.baseline, args.treatment}:
                continue

            def method(choice):
                if choice in ABSTAIN:
                    return None
                if choice not in ("A", "B"):
                    raise ValueError("Invalid review choice: " + str(choice))
                return binding[choice]

            row = dict(reviewer=reviewer, pair_id=pair["pair_id"], song_id=pair["song_id"],
                       heard_audio=pair.get("heard_audio"), reason=pair.get("reason", ""),
                       overall_choice=pair.get("overall"), overall_method=method(pair.get("overall")),
                       criteria={c: (None if c == "musical_fit" and pair.get("heard_audio") is not True
                                     else method(pair.get(c))) for c in CRITERIA})
            rows.append(row)
    by_reviewer = {}
    if not rows:
        raise ValueError("No matching reviews; check baseline/treatment labels and bindings")
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
            songs_without_verified_audio_declaration=[r["song_id"] for r in mine if r["heard_audio"] is not True])
    agreement = []
    for first, second in itertools.combinations(reviewers, 2):
        f = {r["pair_id"]: r["overall_method"] for r in rows if r["reviewer"] == first}
        s = {r["pair_id"]: r["overall_method"] for r in rows if r["reviewer"] == second}
        shared = [p for p in f if p in s and f[p] is not None and s[p] is not None]
        agree = [p for p in shared if f[p] == s[p]]
        observed = len(agree) / len(shared) if shared else None
        # Cohen's kappa for the two-category choice, ties/abstains dropped.
        kappa = cohen_kappa([f[p] == args.treatment for p in shared],
                            [s[p] == args.treatment for p in shared])
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
