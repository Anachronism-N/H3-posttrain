"""Aggregate source-bound reviews without treating missing labels as success."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ms_body_review import load_facts
from ms_nlf_rgb32 import read_rows
from ms_optimize import atomic_write, json_bytes, sha256


GATES = {"false_positive_fraction_max": .05, "false_negative_fraction_max": .15,
         "boundary_error_p95_seconds_max": .20, "decidable_support_fraction_min": .90,
         "critical_source_time_mapping_errors_max": 0, "incorrect_removed_occurrences_max": 0,
         "required_main_video_reviews": 200, "required_independent_human_reviews_min": 40}


def check_occurrence_coverage(facts):
    source = {event["id"]: event for event in facts["events"]}
    summary = facts["summary"]
    if len(source) != len(facts["events"]):
        raise ValueError("Repeated event IDs in facts")
    if {event["id"]: event for event in summary["source_events"]} != source:
        raise ValueError("Summary altered or lost source events")
    selected = {event["id"] for event in summary["selected_events"]}
    covered = {item["event_id"] for group in summary["groups"] for item in group["occurrences"]}
    all_ids = {sid for group in summary["groups"] for sid in group["source_ids"]}
    if covered != selected or all_ids != set(source):
        raise ValueError("Source occurrences disappeared from grouping")


def evaluate_review(review, facts, arrays, facts_sha, row):
    if (review.get("schema") != "native-body-review-template-v1" or review.get("id") != row["id"]
            or review.get("video_sha256") != row["video_sha256"] or review.get("facts_sha256") != facts_sha):
        raise ValueError("Review is not bound to this source/fact revision")
    if (review.get("review_completed") is not True
            or review.get("reference_built_from_rgb_before_reading_candidates") is not True
            or review.get("reviewer_type") not in ("assistant", "human")
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
        return None
    if review.get("absolute_3d_scale_or_depth_certified") is not False:
        raise ValueError("Monocular RGB review cannot certify absolute 3D scale/depth")
    check_occurrence_coverage(facts)
    events = {event["id"]: event for event in facts["events"]}
    references = review["reference_occurrences"]
    reference_map = {}
    for reference in references:
        key = reference["id"]
        if not isinstance(key, str) or not key or key in reference_map:
            raise ValueError("Reference occurrence IDs must be unique")
        start, end = reference["start"], reference["end"]
        if (not np.isfinite([start, end]).all() or start < 0 or end <= start
                or end > row["media_probe"]["time"][-1] + 1e-9):
            raise ValueError("Reference interval must lie within observed source frame times")
        if not isinstance(reference.get("decidable"), bool):
            raise ValueError("Every reference must explicitly declare decidability")
        reference_map[key] = reference
    decisions = review["event_decisions"]
    decision_ids = [item["event_id"] for item in decisions]
    if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != set(events):
        raise ValueError("All candidates must receive exactly one review decision")
    counts = {"tp": 0, "fp": 0, "unknown": 0, "references": sum(ref["decidable"] for ref in references)}
    matched, boundaries = set(), []
    for decision in decisions:
        verdict = decision["verdict"]
        if verdict not in ("supported", "false_positive", "unknown"):
            raise ValueError("Unsupported review verdict")
        if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            raise ValueError("Review decisions require an evidence reason")
        event = events[decision["event_id"]]
        if verdict == "supported":
            reference = reference_map.get(decision.get("reference_id"))
            if reference is None or not reference["decidable"]:
                raise ValueError("Supported candidates must match a decidable RGB reference")
            if (reference["body_part"] != event["body_part"] or reference["action"] != event["action"]
                    or reference["measurement_name"] != event["evidence"]["measurement_name"]
                    or min(reference["end"], event["end"]) <= max(reference["start"], event["start"])):
                raise ValueError("Reference and supported candidate differ in primitive or timing")
            counts["tp"] += 1
            matched.add(reference["id"])
            boundaries.extend([abs(reference["start"] - event["start"]), abs(reference["end"] - event["end"])])
        else:
            counts["fp" if verdict == "false_positive" else "unknown"] += 1
    counts["fn"] = counts["references"] - len(matched)
    time = arrays["tracks"]["time"]
    signals = arrays["signals"]
    names = signals["body_signal_names"].tolist()
    visible = {}
    for interval in review["decidable_intervals"]:
        start, end = interval["start"], interval["end"]
        if (not np.isfinite([start, end]).all() or start < 0 or end <= start
                or end > row["media_probe"]["time"][-1] + 1e-9):
            raise ValueError("Decidable support interval is outside source motion observations")
        name = interval["body_part"] + "." + interval["measurement_name"]
        visible.setdefault(name, np.zeros(len(time), bool))[:] |= (time >= start - 1e-9) & (time <= end + 1e-9)
    support_total, support_valid = 0, 0
    for name, mask in visible.items():
        support_total += int(mask.sum())
        if name in names:
            support_valid += int((signals["body_signal_valid"][:, names.index(name)] & mask).sum())
    counts.update(support_total=support_total, support_valid=support_valid,
                  critical_errors=len(review["critical_source_time_mapping_errors"]),
                  incorrect_removed_occurrences=len(review["incorrect_removed_occurrence_ids"]))
    return {"id": row["id"], "reviewer": review["reviewer"], "reviewer_type": review["reviewer_type"],
            **counts, "boundary_errors": boundaries, "unmatched_decidable_reference_ids": [
                reference["id"] for reference in references if reference["decidable"] and reference["id"] not in matched]}


def metrics(cases):
    total = {key: sum(case[key] for case in cases) for key in
             ("tp", "fp", "fn", "unknown", "references", "support_total", "support_valid", "critical_errors", "incorrect_removed_occurrences")}
    errors = [value for case in cases for value in case["boundary_errors"]]
    return {"counts": total, "false_positive_fraction": total["fp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else None,
            "false_negative_fraction": total["fn"] / total["references"] if total["references"] else None,
            "unknown_candidate_fraction": total["unknown"] / (total["tp"] + total["fp"] + total["unknown"]) if total["tp"] + total["fp"] + total["unknown"] else None,
            "boundary_error_p95_seconds": float(np.percentile(errors, 95)) if errors else None,
            "decidable_support_fraction": total["support_valid"] / total["support_total"] if total["support_total"] else None}


def video_bootstrap(cases, iterations=1000):
    keys = ("false_positive_fraction", "false_negative_fraction", "boundary_error_p95_seconds", "decidable_support_fraction")
    if len(cases) < 2:
        return {key: None for key in keys}
    samples = {key: [] for key in keys}
    rng = np.random.default_rng(0)
    for _ in range(iterations):
        values = metrics([cases[i] for i in rng.integers(0, len(cases), len(cases))])
        for key in keys:
            if values[key] is not None:
                samples[key].append(values[key])
    return {key: np.percentile(values, [2.5, 97.5]).tolist() if values else None for key, values in samples.items()}


def numeric_gates_pass(values):
    limits = (("false_positive_fraction", .05, False), ("false_negative_fraction", .15, False),
              ("boundary_error_p95_seconds", .20, False), ("decidable_support_fraction", .90, True))
    return all(values[key] is not None and (values[key] >= limit if minimum else values[key] <= limit)
               for key, limit, minimum in limits)


def run(args):
    selection = Path(args.selection).resolve()
    rows = read_rows(selection, 0)
    if len(rows) != 200 or len({row["source_group"] for row in rows}) != 200 or any(row.get("panel_role") != "acceptance" for row in rows):
        raise ValueError("Exact frozen 200 acceptance sources required")
    protocol = json.loads(Path(args.protocol).read_text())
    if protocol["acceptance_manifest_sha256"] != sha256(selection):
        raise ValueError("Measurement protocol belongs to a different acceptance cohort")
    panel = json.loads(Path(args.independent_panel).read_text())
    if panel["acceptance_manifest_sha256"] != sha256(selection) or panel["count"] != 40:
        raise ValueError("Independent panel does not match frozen acceptance sources")
    selected_ids = {row["id"] for row in rows}
    base_human_ids = [item["id"] for item in panel["cases"]]
    if len(base_human_ids) != 40 or len(set(base_human_ids)) != 40 or not set(base_human_ids) <= selected_ids:
        raise ValueError("Independent review must contain 40 distinct registered source IDs")
    required_human = set(base_human_ids) | {item["id"] for item in panel["additional_source_disputes"]}
    if not required_human <= selected_ids:
        raise ValueError("Independent dispute case is outside the frozen source panel")
    cases, main_reviews, human_reviews, pending, invalid, disagreements = [], {}, {}, [], [], []
    configs, code_versions = set(), set()
    for row in rows:
        sid = row["id"]
        facts_dir = Path(args.facts_root) / sid
        review_path = Path(args.reviews) / (sid + ".json")
        if not (facts_dir / "facts.json").is_file() or not review_path.is_file():
            pending.append({"id": sid, "reason": "missing facts or source RGB review"})
            continue
        try:
            facts, arrays, facts_sha = load_facts(facts_dir)
            if (facts["id"] != sid or facts["source_group"] != row["source_group"]
                    or facts["video_sha256"] != row["video_sha256"] or sha256(Path(row["video_path"])) != row["video_sha256"]):
                raise ValueError("Source video or fact binding mismatch")
            track = arrays["tracks"]
            probe = row["media_probe"]
            if (not np.array_equal(track["raw_pts"], probe["pts"])
                    or not np.array_equal(track["raw_frame_indices"], probe["frame_indices"])
                    or not np.array_equal(track["time_base"], probe["time_base"])
                    or not np.allclose(track["raw_time"], probe["time"], rtol=0, atol=1e-9)):
                raise ValueError("Processed source frame/PTS mapping differs from frozen media")
            if (facts["parameter_config"] != protocol["config"]
                    or facts["measurement_code_sha256"] != protocol["measurement_code_sha256"]
                    or facts["implementation_sha256"] != protocol["front_code_sha256"]):
                raise ValueError("Facts are not the frozen acceptance code/parameter revision")
            configs.add(json.dumps(facts["parameter_config"], sort_keys=True))
            code_versions.add(json.dumps([facts["measurement_code_sha256"], facts["implementation_sha256"]], sort_keys=True))
            review = json.loads(review_path.read_text())
            result = evaluate_review(review, facts, arrays, facts_sha, row)
            if result is None:
                pending.append({"id": sid, "reason": "source RGB review incomplete"})
                continue
            cases.append(result)
            main_reviews[sid] = result
            if result["fp"] or result["critical_errors"] or result["incorrect_removed_occurrences"]:
                required_human.add(sid)
            human_path = Path(args.independent_reviews) / (sid + ".json")
            if human_path.is_file():
                human_review = json.loads(human_path.read_text())
                independent = evaluate_review(human_review, facts, arrays, facts_sha, row)
                if (independent is not None and independent["reviewer_type"] == "human"
                        and independent["reviewer"] != result["reviewer"]):
                    human_reviews[sid] = independent
                    main_decisions = {item["event_id"]: item["verdict"] for item in review["event_decisions"]}
                    human_decisions = {item["event_id"]: item["verdict"] for item in human_review["event_decisions"]}
                    changed = [event_id for event_id in main_decisions if main_decisions[event_id] != human_decisions[event_id]]
                    if changed:
                        disagreements.append({"id": sid, "event_ids": changed, "status": "requires_adjudication"})
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid.append({"id": sid, "error_type": type(error).__name__, "error": str(error)})
    measured = metrics(cases)
    missing_human = sorted(required_human - set(human_reviews))
    complete = (len(main_reviews) == 200 and len(human_reviews) >= 40 and not pending and not invalid
                and not missing_human and not disagreements and len(configs) == 1 and len(code_versions) == 1)
    usable_metrics = all(measured[key] is not None for key in
                         ("false_positive_fraction", "false_negative_fraction", "boundary_error_p95_seconds", "decidable_support_fraction"))
    passed = (complete and usable_metrics and measured["false_positive_fraction"] <= .05
              and measured["false_negative_fraction"] <= .15 and measured["boundary_error_p95_seconds"] <= .20
              and measured["decidable_support_fraction"] >= .90 and not measured["counts"]["critical_errors"]
              and not measured["counts"]["incorrect_removed_occurrences"]
              and all(not case["critical_errors"] and not case["incorrect_removed_occurrences"] for case in human_reviews.values()))
    status = "passed" if passed else "failed_quality_gate" if complete and usable_metrics else "pending_or_insufficient_review"
    strata = {}
    for label, predicate in (("partial_body", lambda row: row["rgb_strata"]["framing"] == "partial_or_feet_outside_screen"),
                             ("body_in_frame", lambda row: row["rgb_strata"]["framing"] == "body_in_frame"),
                             ("low_contrast", lambda row: row["rgb_strata"]["dark_or_low_contrast_scene"])):
        subset = [main_reviews[row["id"]] for row in rows if row["id"] in main_reviews and predicate(row)]
        strata[label] = {"reviewed_videos": len(subset), "metrics": metrics(subset), "video_bootstrap_ci95": video_bootstrap(subset)}
    independent_metrics = metrics(list(human_reviews.values()))
    metric_keys = ("false_positive_fraction", "false_negative_fraction", "boundary_error_p95_seconds", "decidable_support_fraction")
    usable_metrics = usable_metrics and all(independent_metrics[key] is not None for key in metric_keys)
    usable_metrics = usable_metrics and all(value["metrics"][key] is not None for value in strata.values() for key in metric_keys)
    strata_passed = all(numeric_gates_pass(value["metrics"]) for value in strata.values())
    passed = passed and strata_passed and numeric_gates_pass(independent_metrics)
    status = "passed" if passed else "failed_quality_gate" if complete and usable_metrics else "pending_or_insufficient_review"
    report = {"schema": "native-body-real-quality-report-v1", "status": status, "quality_gate_passed": bool(passed),
              "independent_human_metrics": independent_metrics,
              "unresolved_independent_disagreements": disagreements,
              "all_strata_numeric_gates_passed": strata_passed,
              "registered_videos": 200, "completed_source_rgb_reviews": len(main_reviews),
              "completed_independent_human_reviews": len(human_reviews), "required_independent_ids": sorted(required_human),
              "missing_independent_reviews": missing_human, "pending": pending, "invalid_reviews_or_bindings": invalid,
              "gates": GATES, "metrics_scope": "review-completed source videos only; not a full-200 claim while pending",
              "metrics": measured, "video_bootstrap_ci95": video_bootstrap(cases), "strata": strata,
              "cases": cases, "unknowns_are_not_true_positives": True, "visible_abstentions_count_as_misses": True,
              "zero_metric_denominators_are_not_passes": True, "absolute_scale_or_depth_certified": False,
              "inconsistent_parameter_configs": len(configs) > 1, "selection_sha256": sha256(selection),
              "independent_panel_sha256": sha256(Path(args.independent_panel)), "implementation_sha256": sha256(Path(__file__)),
              "frozen_protocol_sha256": sha256(Path(args.protocol))}
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError("Use a new quality output; preserve previous review revisions")
    out.mkdir(parents=True)
    payload = json_bytes(report)
    atomic_write(out / "quality_report.receipt.json", json_bytes({"quality_report_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "quality_report.json", payload)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("selection", "facts-root", "reviews", "independent-panel", "independent-reviews", "protocol", "out"):
        ap.add_argument("--" + key, required=True)
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
