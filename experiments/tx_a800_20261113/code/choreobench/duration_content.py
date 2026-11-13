"""Paired brief/duration organization with frozen slots and common length limits.

This is a fixed-original-prefix development probe, not physical validation.
Historical v25 serves as bound input/slot evidence, never as same-round control.
"""
import argparse
import copy
import json
from pathlib import Path
import re

from .coordination import unit_response_schema
from .io import file_sha, json_text, object_sha, read_json
from .memory_replay import _prepare_requests, _snapshot, audit_completed_replay, project_history, run_memory_replay
from .request_budget import guard_request
from .timing import event_clock


LIMITS = dict(target_min_words=12, target_max_words=24, max_words=32, max_characters=240)
COMMON = (
    " This is a FIXED-SLOT CONTENT probe. The supplied intervals and event count are frozen, not selectable. "
    "Write concise English movement text: target 12 to 24 whitespace-delimited words per action; "
    "hard maximum 32 words and 240 Unicode characters per action. Do not shorten by using opaque abbreviations. "
    "No new numeric timestamps or beat numbers in action prose; the JSON schedule is authoritative. "
    "Use the full meaning of the supplied actions, not merely music-fitting adjectives. "
    "The targets are writing constraints, not quality scores. No event deletion, reordering or extra time. "
)
DURATION = (
    " Additionally organize the movement to fit EACH slot_duration_seconds. Distinguish successive phases "
    "from simultaneous limb/torso modifiers; simultaneous gestures do not require successive transitions. "
    "Below one second prefer one clear primary movement or one movement phase with few simultaneous modifiers; "
    "do not pack a whole spin then landing then crouching then touching the floor into that slot. "
    "Longer slots may develop a clear phrase rather than reducing everything to a still pose. "
    "Use a natural connection to adjacent slots; if initial body state is unknown do not claim it verified. "
    "Do not add exact repetition counts or verified support/contact states to satisfy the rule. "
    "This sub-second rule is a design hypothesis, not a physiological law. "
)


def validate_content(case, events):
    expected = case["fixed_intervals"]
    if [dict(start=e["start"], end=e["end"]) for e in events] != expected:
        raise ValueError("Content probe changed the frozen interval array or count")
    limits = case["content_limits"]
    for event in events:
        if len(event["action"]) > limits["max_characters"] or len(event["action"].split()) > limits["max_words"]:
            raise ValueError("Content action exceeds the common word/character ceiling; no truncation")
        if re.search(r'\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|seconds?|samples?)\b', event["action"], re.I):
            raise ValueError("Content action added a numeric time claim outside the frozen JSON schedule")


def content_statistics(events, limits):
    return [dict(event_id=e["event_id"], words=len(e["action"].split()), characters=len(e["action"]),
                 target_word_range_met=limits["target_min_words"] <= len(e["action"].split()) <= limits["target_max_words"])
            for e in events]


def prepare_duration_requests(inventory_path, sources, source_run, adaptive_boundary=False):
    source = Path(source_run).resolve(strict=True)
    audit_completed_replay(source)
    prior_contract = _snapshot(source / "contract.json", sources)
    prior_result = _snapshot(source / "results.json", sources)
    if prior_contract.get("duration_source_run") is not None or prior_contract["modes"] != ["full", "metadata_only"]:
        raise ValueError("Duration probe requires the original full/metadata_only completed replay")
    if str(Path(inventory_path).resolve()) not in prior_contract["sources"]:
        raise ValueError("Duration probe inventory differs from historical memory replay")
    cases, _ = _prepare_requests(inventory_path, sources)
    rows = {(r["song_id"], r["unit_id"], r["mode"]): r for r in prior_result["rows"]}
    requests = []
    for case in cases:
        key = (case["song_id"], case["unit_id"])
        full_row, meta_row = rows[key + ("full",)], rows[key + ("metadata_only",)]
        if full_row["status"] != "valid" or meta_row["status"] != "valid" or full_row["actual_intervals"] != meta_row["actual_intervals"]:
            raise ValueError("Historical arms do not provide one common valid fixed schedule")
        historical = _snapshot(source / case["song_id"] / "metadata_only/events.json", sources)
        probe = dict(copy.deepcopy(case), fixed_intervals=copy.deepcopy(meta_row["actual_intervals"]), content_limits=copy.deepcopy(LIMITS))
        if [dict(start=e["start"], end=e["end"]) for e in historical] != probe["fixed_intervals"]:
            raise ValueError("Historical event intervals differ from the audited schedule")
        request = project_history(case["request"], "metadata_only")
        payload = json.loads(request["messages"][-1]["content"])
        clock = copy.deepcopy(case["clock"])
        if clock["format"] != "boundary_ids" or clock.get("schedule_policy") != "balanced_full_unit":
            raise ValueError("Content probe requires the public boundary-ID complete schedule")
        samples = clock["boundary_samples"]
        schedule = [dict(start_boundary=samples.index(i["start"]), end_boundary=samples.index(i["end"])) for i in probe["fixed_intervals"]]
        if schedule not in clock["permitted_schedules"]:
            raise ValueError("Historical fixed intervals are outside the original public menu")
        if adaptive_boundary:
            adaptive_config = dict(case["config"], min_event_seconds=2)
            clock = event_clock(case["unit"], case["before"]["sample_rate"], adaptive_config)
            schedule = clock["permitted_schedules"][-1]
            probe["fixed_intervals"] = [dict(start=clock["boundary_samples"][i["start_boundary"]],
                                            end=clock["boundary_samples"][i["end_boundary"]]) for i in schedule]
            payload["output_contract"]["max_events"] = len(schedule)
        clock["permitted_schedules"] = [schedule]
        probe["clock"] = clock
        payload["event_clock"] = clock
        payload["content_contract"] = dict(limits=LIMITS, fixed_schedule=schedule,
            slot_duration_seconds=[(i["end"]-i["start"])/case["before"]["sample_rate"] for i in probe["fixed_intervals"]],
            timing_rule_is_physical_evidence=False)
        schema = unit_response_schema(case["unit"], payload["output_contract"]["allowed_motif_ids"],
            [c["call_id"] for c in payload["local_proposals"]], max_events=len(schedule), clock=clock,
            max_request_characters=case["config"]["max_request_characters"])
        def bound_actions(value):
            if isinstance(value, dict):
                properties = value.get("properties", {})
                if "action" in properties:
                    properties["action"]["maxLength"] = LIMITS["max_characters"]
                for item in value.values():
                    bound_actions(item)
            elif isinstance(value, list):
                for item in value:
                    bound_actions(item)
        bound_actions(schema)
        modes = ("metadata_only", "boundary_compact") if adaptive_boundary else ("brief", "duration")
        for mode in modes:
            new_request = copy.deepcopy(request)
            namespace = "adaptive-boundary" if adaptive_boundary else "duration-content"
            new_request["call_id"] = case["song_id"] + "/" + namespace + "/" + case["unit_id"] + "/" + mode
            view = copy.deepcopy(payload)
            if adaptive_boundary:
                view["boundary_actions"] = ([{k: copy.deepcopy(e[k]) for k in
                    ("event_id", "start", "end", "track", "action", "exit")} for e in case["before"]["events"]
                    if e["end"] == case["unit"]["start"]] if mode == "boundary_compact" else [])
                view["content_contract"]["adaptive_slot_minimum_seconds"] = 2
                view["content_contract"]["boundary_actions_are_observed_state"] = False
            new_request["messages"][-1]["content"] = json.dumps(view, ensure_ascii=False, sort_keys=True, allow_nan=False)
            new_request["messages"][0]["content"] += COMMON + (DURATION if adaptive_boundary or mode == "duration" else "")
            if adaptive_boundary:
                new_request["messages"][0]["content"] += (" Any boundary_actions are prior planned text, each stable event shown once, "
                    "not measured body state. Use them only for a plausible transition; do not copy their whole phrase. "
                    "Empty boundary_actions do not imply no prior movement. Two seconds is an engineering "
                    "slot-capacity setting, not a physical feasibility guarantee. ")
            new_request["response_schema"] = schema
            guard_request(new_request, case["config"])
            requests.append((probe, mode, new_request))
    return [requests[i][0] for i in range(0, len(requests), 2)], requests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adaptive-boundary", action="store_true")
    args = parser.parse_args(argv)
    result = run_memory_replay(args.inventory, read_json(args.provider), args.out_dir, args.resume,
                               duration_source_run=args.source_run, adaptive_boundary=args.adaptive_boundary)
    print(json_text({k: result[k] for k in ("complete", "expected_calls", "calls_used", "rows", "quality_verified")}))


if __name__ == "__main__":
    main()
