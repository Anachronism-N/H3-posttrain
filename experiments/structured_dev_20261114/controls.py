#!/usr/bin/env python3
"""Small real-model controls for Module S and Module R.

1. shuffled summary: the outline must lose the original block correspondence
   (or fail validation), i.e. the mapping is not driven by section order alone.
2. explicit continuity conflict: the auditor must name it.
3. reasonable motif return: the auditor must not call it a mechanical repeat.

These are implementation probes, not quality evidence.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from choreobench import composition_music as cm          # noqa: E402
from choreobench import whole_song_revision as wsr      # noqa: E402
from choreobench.composition_workflows import _source    # noqa: E402
from choreobench.io import read_json                     # noqa: E402
from choreobench.music_structure_summary import load as load_summary  # noqa: E402
from choreobench.providers import create_provider        # noqa: E402


def cfg():
    return dict(seed=42, max_output_tokens=2048, outline_max_output_tokens=4096,
                max_request_characters=300000, max_events_per_unit=3, structured_unit_output=True,
                use_outline=True, use_history=True, draft_revision_limit=1, event_time_format="boundary_ids",
                event_grid_seconds=.25, min_event_seconds=.5, event_schedule_policy="balanced_full_unit",
                unit_config=dict(bars_per_block=4, fallback_seconds=10.125, max_unit_seconds=20,
                                 meter_status="unverified", min_time_block_seconds=2),
                outline_proposal_char_limit=160, outline_candidates_per_window=1,
                max_calls=1)


def shuffled(summary):
    value = copy.deepcopy(summary)
    blocks = value["blocks"]
    for index, block in enumerate(blocks):
        mirror = blocks[len(blocks) - 1 - index]
        block["relative_energy_db"], block["onset_rate_per_second"] = (
            mirror["relative_energy_db"], mirror["onset_rate_per_second"])
        block["energy_trend"] = mirror["energy_trend"]
    value["relations"] = []
    value["_control_shuffled"] = True
    return value


def call(provider, request):
    response = provider.generate(request)
    if response.get("stop_reason") not in ("stop", "eos"):
        raise SystemExit("control call did not finish: " + str(response.get("stop_reason")))
    return response


def probe_outline(provider, source, summary, label):
    request = cm.outline_request(source["song"], source["options"], source["units"], summary, cfg())
    response = call(provider, request)
    try:
        outline = cm.read_outline(response["text"], source["song"], summary, source["units"])
        return dict(label=label, valid=True,
                    blocks=[[m for m in s["metadata"]["music_blocks"]] for s in outline["sections"]],
                    quantity=[s["metadata"]["quantity_change"] for s in outline["sections"]])
    except Exception as error:
        return dict(label=label, valid=False, error=str(error))


def corrupted(plan):
    value = copy.deepcopy(plan)
    events = value["events"]
    if len(events) < 3:
        return value
    gap = int(3.0 * value["sample_rate"])
    for event in events[1:]:
        event["start"] += gap
        event["end"] += gap
    return value


def reasonable_return(plan):
    value = copy.deepcopy(plan)
    events = value["events"]
    if len(events) >= 3:
        events[2]["action"] = events[0]["action"]
        events[2]["motif_id"] = events[0].get("motif_id")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--music-summary", required=True)
    parser.add_argument("--draft-plan", required=True)
    parser.add_argument("--outline", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sources = {}
    source = _source(args.source, args.song_id, cfg(), sources)
    summary = load_summary(args.music_summary)
    provider = create_provider(read_json(args.provider), "planner", Path(args.out))
    report = dict(schema="choreobench.sr_controls.v1", song_id=args.song_id)
    report["outline_normal"] = probe_outline(provider, source, summary, "normal")
    report["outline_shuffled"] = probe_outline(provider, source, shuffled(summary), "shuffled")
    same = (report["outline_normal"].get("blocks") == report["outline_shuffled"].get("blocks"))
    report["shuffled_changed_mapping"] = not same
    plan = read_json(args.draft_plan)
    outline = read_json(args.outline)
    audit_cfg = dict(seed=42, max_request_characters=300000)
    for label, mutated in (("explicit_gap", corrupted(plan)), ("reasonable_motif_return", reasonable_return(plan))):
        request = wsr.audit_request(mutated, outline, summary, audit_cfg, args.song_id + "/control/" + label)
        response = call(provider, request)
        try:
            issues = wsr.read_audit(response["text"], mutated, summary)
            report[label] = dict(valid=True, codes=[i["problem_type"] for i in issues], issues=issues)
        except Exception as error:
            report[label] = dict(valid=False, error=str(error), raw=response["text"][:800])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "controls.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in v if kk in ("valid", "codes")})
                      for k, v in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
