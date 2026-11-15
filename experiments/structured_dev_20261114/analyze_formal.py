#!/usr/bin/env python3
"""Analyze the frozen formal run: blind bundle, two-order judging, split-aware aggregate.

Only completed songs with both arms are paired. Failures are counted and reported,
never dropped silently. Dev and test songs are aggregated separately, because the
protocol forbids tuning on test.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")


def bootstrap(values, seed=7, resamples=5000):
    if not values:
        return None, None
    random.seed(seed)
    means = sorted(sum(random.choice(values) for _ in values) / len(values) for _ in range(resamples))
    return round(sum(values) / len(values), 3), [round(means[int(0.025 * resamples)], 3),
                                                 round(means[int(0.975 * resamples) - 1], 3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="root with <arm>/<song>/plan.json")
    parser.add_argument("--inventory", required=True, help="formal_song_inventory.json for splits")
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", help="provider json for the judge server (required unless --skip-judge)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8014/v1")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()
    if not args.skip_judge and not args.provider:
        parser.error("--provider is required for judging")
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    inventory = {s["song_id"]: s for s in json.loads(Path(args.inventory).read_text())["songs"]}
    status = {}
    for arm in ("B", "O"):
        for song in inventory:
            result = runs / arm / song / "results.json"
            plan = runs / arm / song / "plan.json"
            if not result.is_file():
                status[(arm, song)] = "missing"
                continue
            payload = json.loads(result.read_text())
            status[(arm, song)] = "complete" if payload.get("full_song_succeeded") is True and plan.is_file() else "failed"
    paired = sorted(song for song in inventory
                    if status[("B", song)] == "complete" and status[("O", song)] == "complete")
    completion = dict(
        B=dict(complete=sum(1 for a, s in status if a == "B" and status[(a, s)] == "complete"),
               failed=sum(1 for a, s in status if a == "B" and status[(a, s)] == "failed"),
               missing=sum(1 for a, s in status if a == "B" and status[(a, s)] == "missing")),
        O=dict(complete=sum(1 for a, s in status if a == "O" and status[(a, s)] == "complete"),
               failed=sum(1 for a, s in status if a == "O" and status[(a, s)] == "failed"),
               missing=sum(1 for a, s in status if a == "O" and status[(a, s)] == "missing")))
    summary = dict(schema="choreobench.formal_analysis.v1", paired=paired, completion=completion,
                   per_song_status={"%s/%s" % key: value for key, value in status.items()})
    if paired and not args.skip_judge:
        songs = " ".join("--song-id " + song for song in paired)
        subprocess.run([sys.executable, str(HERE / "report_multi.py"), *songs.split(),
                        "--arm", "B=%s/B" % runs, "--arm", "O=%s/O" % runs, "--pair", "B,O",
                        "--out", str(out / "bundle")], check=True, cwd=HERE)
        subprocess.run([sys.executable, str(HERE / "judge_pairs.py"),
                        "--pairs", str(out / "bundle/review-pairs.public.json"),
                        "--music-root", str(Path(args.provider).resolve().parent / "music_all"),
                        "--out", str(out / "judge.json"), "--repeat-swapped", "--hide-motif",
                        "--action-chars", "0", "--max-tokens", "1800",
                        "--base-url", args.base_url, "--model", args.model], check=True, cwd=HERE)
        judge = json.loads((out / "judge.json").read_text())
        bindings = {p["pair_id"]: p for p in json.loads((out / "bundle/review-pairs.private.json").read_text())["pairs"]}
        rows = []
        for pair in judge["pairs"]:
            if pair.get("error") or not pair.get("scores_swapped"):
                continue
            binding = bindings[pair["pair_id"]]
            treat = "A" if binding["A"] == "O" else "B"
            base = "A" if binding["A"] == "B" else "B"
            total = 0.0
            for criterion in CRITERIA:
                d1 = pair["scores"][treat][criterion] - pair["scores"][base][criterion]
                d2 = pair["scores_swapped"][treat][criterion] - pair["scores_swapped"][base][criterion]
                total += (d1 + d2) / 2
            preferred = pair["scores"].get("preferred")
            swapped = pair["scores_swapped_positions"].get("preferred")
            mapped = {"A": "B", "B": "A"}.get(swapped, swapped)
            rows.append(dict(pair_id=pair["pair_id"], song_id=pair["song_id"], delta=round(total, 3),
                             preferred=[preferred, mapped], consistent=preferred is not None and preferred == mapped))
        for name, subset in (("all", rows),
                             ("dev", [r for r in rows if inventory[r["song_id"]]["split"] == "dev"]),
                             ("test", [r for r in rows if inventory[r["song_id"]]["split"] == "test"])):
            values = [r["delta"] for r in subset]
            mean, ci = bootstrap(values)
            summary[name] = dict(n=len(values), mean=mean, ci95=ci,
                                 wins=sum(1 for v in values if v > 0), losses=sum(1 for v in values if v < 0),
                                 preferred_consistent=sum(1 for r in subset if r["consistent"]))
        summary["rows"] = rows
    (out / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k not in ("rows", "per_song_status")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
