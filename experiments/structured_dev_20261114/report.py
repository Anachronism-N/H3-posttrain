#!/usr/bin/env python3
"""Per-song cost/failure table and a blinded pairwise bundle for the dev matrix.

Reads completed runs only. Missing or failed cells stay explicit; nothing is
imputed. The blinded bundle swaps A/B order per pair and keeps the method
mapping private, so a reviewer or model judge cannot infer the method from
position. The report itself never calls a model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

METHODS = ("workflow", "react", "choreography", "structured")
CONTROLLER = {"workflow": "v1", "react": "v2", "choreography": "v2", "structured": "v2"}
CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")


def load(run_dir):
    results, plan = Path(run_dir) / "results.json", Path(run_dir) / "plan.json"
    if not results.is_file():
        return dict(status="missing")
    try:
        value = json.loads(results.read_text())
    except json.JSONDecodeError:
        return dict(status="unreadable")
    out = dict(status="complete" if value.get("complete") else "incomplete", results=value)
    if plan.is_file():
        out["plan"] = json.loads(plan.read_text())
    return out


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def facts(entry):
    if entry["status"] in ("missing", "unreadable"):
        return dict(status=entry["status"])
    value = entry["results"]
    stats = value.get("structured_statistics") or {}
    return dict(status=entry["status"], full_song_succeeded=value.get("full_song_succeeded"),
                stop_reason=value.get("stop_reason"), calls_used=value.get("calls_used"),
                input_tokens=value.get("input_tokens"), output_tokens=value.get("output_tokens"),
                committed_units=value.get("committed_units"), total_units=value.get("total_units"),
                blocks_with_issues=stats.get("blocks_with_detected_issues"),
                revision_calls=stats.get("revision_calls"),
                token_projection_ratio=value.get("token_projection_ratio"))


def _plan_actions(plan):
    return [dict(event_id=e["event_id"], start=e["start"], end=e["end"], track=e["track"],
                 action=e["action"], motif_id=e.get("motif_id")) for e in plan.get("events", [])]


def _clauses(action):
    lowered = " " + action.lower() + " "
    return 1 + sum(lowered.count(m) for m in (";", " then ", ", then ", " followed by ", " and then "))


def literal_facts(plan, sample_rate):
    """Cheap, non-quality diagnostics used to point reviewers at real content."""
    events = plan.get("events", [])
    if not events:
        return dict(events=0)
    durations = [(e["end"] - e["start"]) / float(sample_rate) for e in events]
    repeated = 0
    seen = {}
    for e in events:
        key = " ".join(e["action"].lower().split())
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            repeated += 1
    dense = sum(1 for e in events if _clauses(e["action"]) >= 3 and (e["end"] - e["start"]) / float(sample_rate) < 1.5)
    return dict(events=len(events), min_seconds=round(min(durations), 3), mean_seconds=round(sum(durations) / len(durations), 3),
                repeated_action_types=repeated, long_clause_short_events=dense,
                motifs_used=sorted({e.get("motif_id") for e in events if e.get("motif_id")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--song-id", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_root, out = Path(args.out_root), Path(args.out)
    songs = args.song_id
    runs = {s: {m: load(out_root / s / m) for m in METHODS} for s in songs}
    table = {s: {m: facts(runs[s][m]) for m in METHODS} for s in songs}
    pairs = []
    private = []
    for index, song in enumerate(songs):
        for left, right in (("workflow", "react"), ("workflow", "choreography"),
                            ("workflow", "structured"), ("choreography", "structured")):
            a, b = runs[song][left], runs[song][right]
            if "plan" not in a or "plan" not in b:
                continue
            swap = (index + len(pairs)) % 2 == 1
            first, second = (right, left) if swap else (left, right)
            pair_id = "pair-%02d" % len(pairs)
            pairs.append(dict(
                pair_id=pair_id, song_id=song, order=("B_then_A" if swap else "A_then_B"),
                A=dict(plan=_plan_actions(runs[song][first]["plan"])),
                B=dict(plan=_plan_actions(runs[song][second]["plan"])),
                criteria=list(CRITERIA),
                instructions=("Compare the two timed choreography scores for this song. Cite concrete event ids "
                              "and action text for every judgement. Judge musical fit only from the supplied "
                              "independent music evidence; do not invent heard audio. Judge motif development by "
                              "whether repetition is intended or mechanical, executability by whether actions fit "
                              "their durations, and boundary connection by adjacent event compatibility. "
                              "Unknown physical state is not a defect.")))
            private.append(dict(pair_id=pair_id, song_id=song, A=first, B=second, swapped=swap))
    report = dict(schema="choreobench.structured_dev_report.v1", songs=songs, methods=list(METHODS),
                  controller_version=CONTROLLER, per_song=table,
                  literal=dict((s, dict((m, literal_facts(runs[s][m]["plan"], 16000))
                                        for m in METHODS if "plan" in runs[s][m])) for s in songs),
                  note=("Cost and completion only. Model/human quality judgements are recorded separately "
                        "and never fed back into generation."))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    (out / "review-pairs.public.json").write_text(json.dumps(dict(
        schema="choreobench.pairwise_blind.v1", criteria=list(CRITERIA),
        template=dict(pair_id="", song_id="", a_score=None, b_score=None, a_reason="", b_reason="",
                      preferred="", confidence="", reviewer_id="", heard_audio=False),
        pairs=pairs), ensure_ascii=False, indent=1) + "\n")
    (out / "review-pairs.private.json").write_text(json.dumps(dict(
        schema="choreobench.pairwise_bindings.v1", pairs=private,
        bundle_sha256=_sha(pairs)), ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(songs=len(songs), pairs=len(pairs), cells=len(songs) * len(METHODS))))


if __name__ == "__main__":
    main()
