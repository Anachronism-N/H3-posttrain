#!/usr/bin/env python3
"""Action-level adherence of a finished plan to the acoustic summary.

For each planning unit, the acoustic direction is recomputed deterministically
from the summary, and the plan's event shape is checked against it:

  increase -> two or more events in the unit
  decrease -> exactly one event covering at least half of the unit
  steady   -> one or two events

This is a literal engineering check, not a musical quality score.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from choreobench.music_structure_summary import direction  # noqa: E402


def unit_shape(plan, unit, rate):
    events = [e for e in plan["events"] if unit["start"] <= e["start"] < unit["end"]]
    if not events:
        return dict(events=0, mean_duration=0.0)
    duration = (unit["end"] - unit["start"]) / float(rate)
    return dict(events=len(events),
                mean_duration=sum((e["end"] - e["start"]) / float(rate) for e in events) / len(events),
                unit_duration=duration)


def matches(direction, shape):
    if direction == "increase":
        return shape["events"] >= 2
    if direction == "decrease":
        return shape["events"] == 1 and shape["mean_duration"] >= 0.5 * shape.get("unit_duration", 0)
    if direction == "steady":
        return 1 <= shape["events"] <= 2
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--source", required=True, help="complete catalog song view")
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--music-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    from choreobench.composition_workflows import _source
    cfg = dict(unit_config=dict(bars_per_block=4, fallback_seconds=10.125, max_unit_seconds=20,
                                meter_status="unverified", min_time_block_seconds=2))
    units = _source(args.source, args.song_id, cfg, {})["units"]["units"]
    summary = json.loads(Path(args.music_summary).read_text())
    rate = plan["sample_rate"]
    rows, skipped = [], 0
    for unit in units:
        want = direction(summary, unit["start"], unit["end"])
        shape = unit_shape(plan, unit, rate)
        ok = matches(want, shape)
        if ok is None:
            skipped += 1
            continue
        rows.append(dict(unit_id=unit["unit_id"], acoustic_direction=want,
                         events=shape["events"], mean_duration=round(shape["mean_duration"], 2),
                         unit_duration=round(shape.get("unit_duration", 0), 2), matches=ok))
    rate_ok = (sum(1 for r in rows if r["matches"]) / len(rows)) if rows else None
    report = dict(schema="choreobench.plan_adherence.v1", plan=args.plan, units=len(rows),
                  skipped_unknown=skipped, adherence=rate_ok, rows=rows,
                  note="Literal event-shape check against a deterministic block-energy direction.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(plan=Path(args.plan).parent.name, units=len(rows), adherence=rate_ok)))


if __name__ == "__main__":
    main()
