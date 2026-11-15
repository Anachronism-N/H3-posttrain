#!/usr/bin/env python3
"""Generic arm table + blinded pairwise bundle.

Usage:
  report_multi.py --song-id S1 --song-id S2 ... \
    --arm B=runs/B --arm BB=runs/BB --pair B,BB --pair BB,SB --out DIR

Each arm maps to a directory containing <song>/results.json and
<final_plan.json|plan.json>. Pairs are emitted with the A/B order swapped on
alternate songs; the method binding stays in the private file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from choreobench.schema import validate_plan

CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def facts(arm_dir, song):
    run = Path(arm_dir) / song
    try:
        results = _load(run / "results.json")
    except (ValueError, OSError) as error:
        return dict(status="invalid_result", full_song_succeeded=False, error=str(error))
    if results is None:
        return dict(status="missing")
    row = dict(status="complete" if results.get("full_song_succeeded") is True else "failed",
               full_song_succeeded=results.get("full_song_succeeded") is True,
               calls=results.get("total_calls", results.get("calls_used")),
               input_tokens=results.get("total_input_tokens", results.get("input_tokens")),
               output_tokens=results.get("total_output_tokens", results.get("output_tokens")),
               outcome=results.get("outcome") or results.get("stop_reason"))
    try:
        plan = _load(run / "final_plan.json") or _load(run / "plan.json")
        if row["full_song_succeeded"]:
            if plan is None:
                raise ValueError("Success result has no final plan")
            validate_plan(plan)
            if plan["song_id"] != song:
                raise ValueError("Final plan song identity differs")
    except (ValueError, TypeError, KeyError, OSError) as error:
        row.update(status="invalid_plan", full_song_succeeded=False, error=str(error))
        return row
    if results.get("changed_event_ids") is not None:
        row["changed_events"] = len(results.get("changed_event_ids") or [])
    if plan:
        rate = plan.get("sample_rate", 16000)
        events = plan.get("events", [])
        durations = [(e["end"] - e["start"]) / float(rate) for e in events] or [0.0]
        seen, repeated = {}, 0
        for e in events:
            key = " ".join(e["action"].lower().split())
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 2:
                repeated += 1
        motifs = {e.get("motif_id") for e in events if e.get("motif_id")}
        row.update(events=len(events), mean_seconds=round(sum(durations) / len(durations), 2),
                   repeated_action_types=repeated, distinct_motifs=len(motifs),
                   plan_path=str(run / ("final_plan.json" if (run / "final_plan.json").is_file() else "plan.json")))
    return row


def _actions(plan):
    """Plan times are sample indices; review bundles must use seconds."""
    rate = float(plan["sample_rate"])
    return [dict(event_id=e["event_id"], start=e["start"] / rate, end=e["end"] / rate,
                 track=e["track"], action=e["action"], motif_id=e.get("motif_id"))
            for e in plan.get("events", [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--song-id", action="append", required=True)
    parser.add_argument("--arm", action="append", required=True, help="name=dir")
    parser.add_argument("--pair", action="append", default=[], help="A,B")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    arms = {}
    for item in args.arm:
        name, _, directory = item.partition("=")
        if not name or not directory or name in arms:
            parser.error("--arm must uniquely name a nonempty directory")
        arms[name] = directory
    songs = args.song_id
    if len(set(songs)) != len(songs):
        parser.error("Duplicate registered song")
    table = {s: {name: facts(d, s) for name, d in arms.items()} for s in songs}
    pairs, private, excluded = [], [], []
    counter = 0
    for spec in args.pair:
        base, _, treatment = spec.partition(",")
        if base not in arms or treatment not in arms or base == treatment:
            parser.error("--pair must name two distinct registered arms")
        for index, song in enumerate(songs):
            if not all(table[song][m].get("full_song_succeeded") is True for m in (base, treatment)):
                excluded.append(dict(song_id=song, comparison=spec,
                    statuses={m: table[song][m]["status"] for m in (base, treatment)}))
                continue
            a_dir, b_dir = Path(arms[base]) / song, Path(arms[treatment]) / song
            a_plan = _load(a_dir / "final_plan.json") or _load(a_dir / "plan.json")
            b_plan = _load(b_dir / "final_plan.json") or _load(b_dir / "plan.json")
            if not a_plan or not b_plan:
                continue
            swap = counter % 2 == 1
            first, second = (treatment, base) if swap else (base, treatment)
            pair = dict(pair_id="pair-%02d" % counter, song_id=song, comparison="%s_vs_%s" % (base, treatment),
                        time_unit="seconds",
                        A=_actions(_load((Path(arms[first]) / song / "final_plan.json"))
                                   or _load((Path(arms[first]) / song / "plan.json"))),
                        B=_actions(_load((Path(arms[second]) / song / "final_plan.json"))
                                   or _load((Path(arms[second]) / song / "plan.json"))),
                        criteria=list(CRITERIA),
                        instructions=("Compare the two timed scores. Cite event ids and action text. Judge musical fit "
                                      "against the supplied acoustic evidence only; judge motif development by whether "
                                      "repetition is intended or mechanical; executability by whether the action fits its "
                                      "duration; boundary connection by whether adjacent events continue coherently. "
                                      "Sustained movement and intentional motif return are not defects."))
            pairs.append(pair)
            private.append(dict(pair_id=pair["pair_id"], song_id=song, comparison=pair["comparison"],
                                A=first, B=second, swapped=swap))
            counter += 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(dict(schema="choreobench.multi_arm_report.v1", songs=songs,
                                                     arms=arms, per_song=table, excluded_pairs=excluded,
                                                     registered_pairs=len(songs) * len(args.pair)), ensure_ascii=False, indent=1) + "\n")
    (out / "review-pairs.public.json").write_text(json.dumps(dict(
        schema="choreobench.pairwise_blind.v2", time_unit="seconds", criteria=list(CRITERIA), pairs=pairs),
        ensure_ascii=False, indent=1) + "\n")
    (out / "review-pairs.private.json").write_text(json.dumps(dict(
        schema="choreobench.pairwise_bindings.v1", pairs=private,
        bundle_sha256=hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()),
        ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(songs=len(songs), arms=list(arms), pairs=len(pairs))))


if __name__ == "__main__":
    main()
