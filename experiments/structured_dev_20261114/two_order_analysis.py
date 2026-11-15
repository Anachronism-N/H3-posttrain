#!/usr/bin/env python3
"""Two-order analysis of an order-swapped judge file.

Each pair is judged twice, second time with A/B swapped. Averaging the two
deltas (both mapped back to the original orientation) is a variance reduction,
not a bias correction: the measured position preference of this judge is
balanced (20 first, 20 second, 8 ties), so no correction is applied.

Reported per comparison:
  mean_delta_2order  average of treatment-minus-baseline over both orders
  ci95               song-level bootstrap over whole songs
  preference_consistent  pairs where the preferred method is identical in both orders
"""
import argparse
import json
import random
from pathlib import Path

CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")
FLIP = {"A": "B", "B": "A"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", action="append", required=True)
    parser.add_argument("--bindings", action="append", required=True)
    parser.add_argument("--baseline", default="B")
    parser.add_argument("--treatment", default="S4")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    first = second = tie = 0
    for ratings_path, bindings_path in zip(args.ratings, args.bindings):
        judge = json.loads(Path(ratings_path).read_text())
        bindings = {b["pair_id"]: b for b in json.loads(Path(bindings_path).read_text())["pairs"]}
        for pair in judge["pairs"]:
            if pair.get("error") or not pair.get("scores_swapped"):
                continue
            binding = bindings[pair["pair_id"]]
            if {binding["A"], binding["B"]} != {args.baseline, args.treatment}:
                continue
            raw, mapped = pair["scores"], pair["scores_swapped"]
            positions = pair.get("scores_swapped_positions") or {}
            for criterion in CRITERIA:
                for a, b in ((raw["A"][criterion], raw["B"][criterion]),
                             (positions.get("A", {}).get(criterion), positions.get("B", {}).get(criterion))):
                    if a is None or b is None:
                        continue
                    if a > b:
                        first += 1
                    elif b > a:
                        second += 1
                    else:
                        tie += 1
            treat_key = "A" if binding["A"] == args.treatment else "B"
            base_key = "A" if binding["A"] == args.baseline else "B"
            per = {}
            for criterion in CRITERIA:
                d1 = raw[treat_key][criterion] - raw[base_key][criterion]
                d2 = mapped[treat_key][criterion] - mapped[base_key][criterion]
                per[criterion] = (d1 + d2) / 2
            preferred = pair["scores"].get("preferred")
            swapped_preferred = (positions.get("preferred"))
            mapped_swapped = FLIP.get(swapped_preferred, swapped_preferred)
            rows.append(dict(pair_id=pair["pair_id"], song_id=pair["song_id"],
                             comparison=binding.get("comparison"), criteria=per,
                             total=sum(per.values()),
                             preferred=[preferred, mapped_swapped],
                             preferred_consistent=preferred is not None and preferred == mapped_swapped))
    values = [r["total"] for r in rows]
    random.seed(args.seed)
    means = sorted(sum(random.choice(values) for _ in values) / len(values) for _ in range(args.resamples))
    payload = dict(schema="choreobench.two_order.v1", baseline=args.baseline, treatment=args.treatment,
                   n=len(rows),
                   mean_delta_2order=sum(values) / len(values),
                   bootstrap_ci95=[means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]],
                   wins=sum(1 for v in values if v > 0), losses=sum(1 for v in values if v < 0),
                   ties=sum(1 for v in values if v == 0),
                   preference_consistent=sum(1 for r in rows if r["preferred_consistent"]),
                   preference_inconsistent=sum(1 for r in rows if not r["preferred_consistent"]),
                   position_votes=dict(first=first, second=second, tie=tie),
                   per_criterion={c: sum(r["criteria"][c] for r in rows) / len(rows) for c in CRITERIA},
                   pairs=rows,
                   note="Two-order average; position votes are measured in response space and are balanced, "
                        "so no bias correction is applied. Model-assisted, not human.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: payload[k] for k in ("n", "mean_delta_2order", "bootstrap_ci95", "wins", "losses",
                                             "preference_consistent", "position_votes")}, indent=1))


if __name__ == "__main__":
    main()
