#!/usr/bin/env python3
"""2x2 report for B / S / R / S+R on the dev songs.

Reads completed runs only and never calls a model. Produces:
  report.json             per-song completion, calls, tokens, revision outcome
  review-pairs.public.json  blinded A/B bundles for S-vs-B, R-vs-B, S+R-vs-S, S+R-vs-B
  review-pairs.private.json method bindings for the blind bundle

R and S+R always start from the frozen draft of their base arm, so their cost
row includes the draft cost and is reported as total method cost.
"""
import argparse
import hashlib
import json
from pathlib import Path

ARMS = ("B", "S", "R", "SR")
PAIRINGS = (("B", "S"), ("B", "R"), ("S", "SR"), ("B", "SR"))
CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")


def _load(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def run_facts(run_dir):
    results = _load(run_dir / "results.json")
    if results is None:
        return dict(status="missing")
    plan_path = run_dir / "final_plan.json"
    if not plan_path.is_file():
        plan_path = run_dir / "plan.json"
    plan = _load(plan_path)
    row = dict(status="complete" if results.get("complete") else "incomplete",
               full_song_succeeded=results.get("full_song_succeeded"))
    if "total_calls" in results:  # revision arm
        row.update(calls=results.get("total_calls"), input_tokens=results.get("total_input_tokens"),
                   output_tokens=results.get("total_output_tokens"),
                   draft_calls=results.get("draft_calls"), revision_calls=results.get("revision_calls"),
                   outcome=results.get("outcome"), changed_event_ids=results.get("changed_event_ids"))
    else:
        row.update(calls=results.get("calls_used"), input_tokens=results.get("input_tokens"),
                   output_tokens=results.get("output_tokens"), outcome=results.get("stop_reason"))
    if results.get("music_structure"):
        row["music_structure"] = results["music_structure"]
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
        row.update(events=len(events), min_seconds=round(min(durations), 3),
                   mean_seconds=round(sum(durations) / len(durations), 3),
                   repeated_action_types=repeated,
                   motifs_used=sorted({e.get("motif_id") for e in events if e.get("motif_id")}))
        row["plan_path"] = str(plan_path)
    return row


def _actions(plan):
    return [dict(event_id=e["event_id"], start=e["start"], end=e["end"], track=e["track"],
                 action=e["action"], motif_id=e.get("motif_id")) for e in plan.get("events", [])]


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--song-id", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.runs_root), Path(args.out)
    songs = args.song_id
    facts = {s: {arm: run_facts(root / arm / s) for arm in ARMS} for s in songs}
    pairs, private = [], []
    for index, song in enumerate(songs):
        for base, treatment in PAIRINGS:
            for size in (0, 1):
                left, right = (base, treatment) if (index + len(pairs)) % 2 == 0 else (treatment, base)
                a_dir, b_dir = root / left / song, root / right / song
                a_plan = _load(a_dir / ("final_plan.json")) or _load(a_dir / "plan.json")
                b_plan = _load(b_dir / ("final_plan.json")) or _load(b_dir / "plan.json")
                if not a_plan or not b_plan:
                    continue
                pair_id = "pair-%02d" % len(pairs)
                pairs.append(dict(
                    pair_id=pair_id, song_id=song, comparison="%s_vs_%s" % (base, treatment),
                    A=_actions(a_plan), B=_actions(b_plan), criteria=list(CRITERIA),
                    instructions=("Compare the two timed choreography scores for this song. Cite concrete event ids "
                                  "and action text. Judge musical fit against the supplied acoustic evidence only; "
                                  "do not invent heard audio. Judge motif development by whether repetition is "
                                  "intended or mechanical, executability by whether the action fits its duration, "
                                  "and boundary connection by whether adjacent events continue coherently. "
                                  "Sustained movement and intentional motif return are not defects.")))
                private.append(dict(pair_id=pair_id, song_id=song, comparison="%s_vs_%s" % (base, treatment),
                                    A=left, B=right))
                break
    report = dict(schema="choreobench.sr_2x2_report.v1", songs=songs, arms=list(ARMS),
                  pairings=[list(p) for p in PAIRINGS], per_song=facts,
                  note=("Completion, calls and tokens only. R/S+R rows include their draft cost. "
                        "Quality claims require the blinded pairwise review; nothing here is fed back to generation."))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    (out / "review-pairs.public.json").write_text(json.dumps(dict(
        schema="choreobench.pairwise_blind.v1", criteria=list(CRITERIA), pairs=pairs),
        ensure_ascii=False, indent=1) + "\n")
    (out / "review-pairs.private.json").write_text(json.dumps(dict(
        schema="choreobench.pairwise_bindings.v1", pairs=private, bundle_sha256=_sha(pairs)),
        ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(songs=len(songs), pairs=len(pairs))))


if __name__ == "__main__":
    main()
