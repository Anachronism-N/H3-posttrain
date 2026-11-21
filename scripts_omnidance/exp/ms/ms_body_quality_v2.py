"""Source-reviewed, one-to-one motion validation with explicit unknown coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ms_body_quality import GATES, check_occurrence_coverage, metrics, numeric_gates_pass, video_bootstrap
from ms_body_review import load_facts
from ms_landmarks import load_bundle
from ms_nlf_rgb32 import read_rows
from ms_optimize import atomic_write, json_bytes, sha256


SCHEMA = "native-body-source-review-v2"
GEOMETRY_CHECKS = (
    "anatomical_mapping", "source_frame_time_alignment", "required_point_observability",
    "coordinate_reference_labeling", "primitive_direction", "amplitude_speed_trends", "occurrence_preservation",
)


def channel_key(item):
    return item["body_part"] + "." + item["measurement_name"]


def interval_bounds(item, start, end):
    a, b = item["start"], item["end"]
    if (type(a) not in (int, float) or type(b) not in (int, float) or not np.isfinite([a, b]).all()
            or a < start - 1e-9 or b > end + 1e-9 or b <= a):
        raise ValueError("Review intervals must lie inside observed source frame times")
    return float(a), float(b)


def assessments(review, facts, arrays, row):
    expected = {channel_key(item) for item in facts["signal_metadata"]["signals"]}
    columns = arrays["signals"]["body_signal_names"].tolist()
    if len(columns) != len(set(columns)) or expected != set(columns):
        raise ValueError("Fact operator inventory differs from its signal cache")
    items = review["channel_assessments"]
    keys = [channel_key(item) for item in items]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Every measured channel requires exactly one complete source assessment")
    start, end = row["media_probe"]["time"][0], row["media_probe"]["time"][-1]
    if review.get("reference_inventory_complete") is not True:
        return None
    time = arrays["tracks"]["time"]
    visible = {}
    unknown_seconds = 0.
    for item in items:
        if item.get("reference_inventory_complete") is not True:
            return None
        intervals = sorted(item["intervals"], key=lambda interval: interval["start"])
        cursor = start
        good = np.zeros(len(time), dtype=bool)
        for interval in intervals:
            a, b = interval_bounds(interval, start, end)
            if abs(a - cursor) > 1e-9:
                raise ValueError("Channel observability must cover the entire source without gaps or overlaps")
            if interval.get("observability") not in ("decidable", "unknown"):
                raise ValueError("Source observability must be decidable or unknown")
            if not isinstance(interval.get("reason"), str) or not interval["reason"].strip():
                raise ValueError("Each source-observability interval requires an evidence reason")
            if interval["observability"] == "decidable":
                good |= (time >= a - 1e-9) & (time <= b + 1e-9)
            else:
                unknown_seconds += b - a
            cursor = b
        if not intervals or abs(cursor - end) > 1e-9:
            raise ValueError("Channel observability must reach the last observed source time")
        visible[channel_key(item)] = {"mask": good, "intervals": intervals}
    return visible, unknown_seconds, len(expected) * (end - start)


def reference_is_decidable(reference, intervals):
    cursor = reference["start"]
    for interval in intervals:
        if interval["end"] <= cursor + 1e-9:
            continue
        if interval["start"] >= reference["end"] - 1e-9:
            break
        if interval["observability"] != "decidable" or interval["start"] > cursor + 1e-9:
            return False
        cursor = min(reference["end"], interval["end"])
        if cursor >= reference["end"] - 1e-9:
            return True
    return False


def evaluate_review(review, facts, arrays, facts_sha, row):
    if (review.get("schema") != SCHEMA or review.get("id") != row["id"]
            or review.get("video_sha256") != row["video_sha256"] or review.get("facts_sha256") != facts_sha):
        raise ValueError("Review source/fact revision is not bound")
    if (review.get("review_completed") is not True
            or review.get("reference_built_from_rgb_before_reading_candidates") is not True
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or review.get("reviewer_type") not in ("assistant", "human")):
        return None
    if review.get("absolute_3d_scale_or_depth_certified") is not False:
        raise ValueError("Monocular source review cannot certify absolute scale or depth")
    check_occurrence_coverage(facts)
    coverage = assessments(review, facts, arrays, row)
    if coverage is None:
        return None
    visible, unknown_seconds, operator_seconds = coverage
    check_items = review["geometry_checks"]
    check_names = [item["name"] for item in check_items]
    if len(check_names) != len(set(check_names)) or set(check_names) != set(GEOMETRY_CHECKS):
        raise ValueError("Every geometry/source check must be explicitly assessed")
    for item in check_items:
        if item.get("verdict") not in ("pass", "fail", "unknown") or not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("Geometry checks require an explicit verdict and source evidence")
    references = {}
    reference_signatures = set()
    for item in review["reference_occurrences"]:
        rid = item["id"]
        if not isinstance(rid, str) or not rid or rid in references:
            raise ValueError("Reference occurrence IDs must be nonempty and unique")
        interval_bounds(item, row["media_probe"]["time"][0], row["media_probe"]["time"][-1])
        key = channel_key(item)
        if key not in visible or type(item.get("decidable")) is not bool:
            raise ValueError("References must belong to assessed channels with explicit decidability")
        if item["decidable"]:
            if not all(isinstance(item.get(name), str) and item[name] for name in ("action", "direction")):
                raise ValueError("Decidable references require explicit primitive action and direction")
            if not reference_is_decidable(item, visible[key]["intervals"]):
                raise ValueError("A decidable reference cannot use unknown source support")
        signature = (key, item.get("action"), item.get("direction"), item["start"], item["end"], item["decidable"])
        if signature in reference_signatures:
            raise ValueError("Duplicated reference occurrences cannot increase reference or TP counts")
        reference_signatures.add(signature)
        references[rid] = item
    events = {event["id"]: event for event in facts["events"]}
    decisions = review["event_decisions"]
    ids = [item["event_id"] for item in decisions]
    if len(ids) != len(set(ids)) or set(ids) != set(events):
        raise ValueError("Every original candidate requires exactly one decision")
    counts = {"tp": 0, "fp": 0, "unknown": 0, "references": sum(item["decidable"] for item in references.values())}
    matched, boundaries = set(), []
    for decision in decisions:
        verdict = decision["verdict"]
        if verdict not in ("supported", "false_positive", "unknown") or not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            raise ValueError("Candidate verdicts require a source-evidence reason")
        if verdict == "supported":
            event = events[decision["event_id"]]
            reference = references.get(decision.get("reference_id"))
            if reference is None or not reference["decidable"]:
                raise ValueError("Supported candidates require a decidable RGB reference")
            if reference["id"] in matched:
                raise ValueError("One reference occurrence cannot support multiple candidate occurrences")
            if (reference["body_part"] != event["body_part"] or reference["action"] != event["action"]
                    or reference["direction"] != event["direction"]
                    or reference["measurement_name"] != event["evidence"]["measurement_name"]
                    or min(reference["end"], event["end"]) <= max(reference["start"], event["start"])):
                raise ValueError("Supported candidate must match reference channel, direction, primitive and time")
            matched.add(reference["id"])
            counts["tp"] += 1
            boundaries.extend([abs(reference["start"] - event["start"]), abs(reference["end"] - event["end"])])
        else:
            counts["fp" if verdict == "false_positive" else "unknown"] += 1
    counts["fn"] = counts["references"] - len(matched)
    names = arrays["signals"]["body_signal_names"].tolist()
    counts["support_total"] = sum(int(item["mask"].sum()) for item in visible.values())
    counts["support_valid"] = sum(int((arrays["signals"]["body_signal_valid"][:, names.index(key)] & item["mask"]).sum())
                                  for key, item in visible.items())
    counts.update(critical_errors=len(review["critical_source_time_mapping_errors"]),
                  incorrect_removed_occurrences=len(review["incorrect_removed_occurrence_ids"]))
    return {"id": row["id"], "reviewer": review["reviewer"], "reviewer_type": review["reviewer_type"],
            **counts, "boundary_errors": boundaries, "geometry_checks_passed": all(item["verdict"] == "pass" for item in check_items),
            "unknown_operator_seconds": unknown_seconds, "total_operator_seconds": operator_seconds,
            "unknown_operator_fraction": unknown_seconds / operator_seconds if operator_seconds else None,
            "unmatched_decidable_reference_ids": [rid for rid, item in references.items() if item["decidable"] and rid not in matched]}


def review_disagreements(main, independent, tolerance=.20):
    problems = []
    a = {item["event_id"]: item["verdict"] for item in main["event_decisions"]}
    b = {item["event_id"]: item["verdict"] for item in independent["event_decisions"]}
    changed = sorted(key for key in a if a[key] != b.get(key))
    if changed:
        problems.append({"kind": "candidate_verdict", "event_ids": changed})
    main_references = {item["id"]: item for item in main["reference_occurrences"]}
    human_references = {item["id"]: item for item in independent["reference_occurrences"]}
    human_decisions = {item["event_id"]: item for item in independent["event_decisions"]}
    for decision in main["event_decisions"]:
        other_decision = human_decisions.get(decision["event_id"])
        if decision["verdict"] != "supported" or other_decision is None or other_decision["verdict"] != "supported":
            continue
        left = main_references[decision["reference_id"]]
        right = human_references[other_decision["reference_id"]]
        if ((channel_key(left), left["action"], left["direction"]) != (channel_key(right), right["action"], right["direction"])
                or max(abs(left["start"] - right["start"]), abs(left["end"] - right["end"])) > tolerance + 1e-9):
            problems.append({"kind": "supported_reference_assignment", "event_id": decision["event_id"]})
    used = set()
    for item in main["reference_occurrences"]:
        signature = (channel_key(item), item.get("action"), item.get("direction"), item["decidable"])
        candidates = [(i, other) for i, other in enumerate(independent["reference_occurrences"]) if i not in used
                      and (channel_key(other), other.get("action"), other.get("direction"), other["decidable"]) == signature]
        match = min(candidates, key=lambda pair: abs(item["start"] - pair[1]["start"]) + abs(item["end"] - pair[1]["end"]), default=None)
        if match is None:
            problems.append({"kind": "missing_or_different_reference", "reference_id": item["id"]})
            continue
        index, other = match
        used.add(index)
        if max(abs(item["start"] - other["start"]), abs(item["end"] - other["end"])) > tolerance + 1e-9:
            problems.append({"kind": "reference_boundary", "reference_id": item["id"], "independent_reference_id": other["id"]})
    if len(used) != len(independent["reference_occurrences"]):
        problems.append({"kind": "additional_independent_references"})
    main_channels = {channel_key(item): item for item in main["channel_assessments"]}
    human_channels = {channel_key(item): item for item in independent["channel_assessments"]}
    for name, item in main_channels.items():
        other = human_channels.get(name)
        if other is None:
            problems.append({"kind": "missing_channel", "channel": name})
            continue
        intervals = item["intervals"] + other["intervals"]
        boundaries = sorted({bound for interval in intervals for bound in (interval["start"], interval["end"])})
        for start, end in zip(boundaries, boundaries[1:]):
            value = (start + end) / 2
            left = next((interval["observability"] for interval in item["intervals"] if interval["start"] <= value < interval["end"]), None)
            right = next((interval["observability"] for interval in other["intervals"] if interval["start"] <= value < interval["end"]), None)
            if left != right:
                problems.append({"kind": "source_observability", "channel": name, "start": start, "end": end})
    a = {item["name"]: item["verdict"] for item in main["geometry_checks"]}
    b = {item["name"]: item["verdict"] for item in independent["geometry_checks"]}
    if a != b:
        problems.append({"kind": "geometry_check_disagreement"})
    return problems


def contract_review_ids(path, base_ids, source_hashes, selection_sha, panel_sha):
    payload = Path(path).read_bytes()
    contract = json.loads(payload)
    if (contract.get("schema") != "native-body-independent-review-contract-v1"
            or contract.get("selection_sha256") != selection_sha
            or contract.get("original_independent_panel_sha256") != panel_sha):
        raise ValueError("Independent review contract does not match the frozen source cohort")
    required = set(base_ids)
    supplement_hashes = {}
    for item in contract["supplemental_panels"]:
        content = Path(item["path"]).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("Supplemental review panel changed")
        panel = json.loads(content)
        if panel.get("original_independent_panel_sha256") != panel_sha:
            raise ValueError("Supplemental review panel belongs to a different original panel")
        key = {"native-body-supplemental-independent-review-v1": "video_sha256",
               "native-body-supplemental-geometry-review-v1": "source_video_sha256"}.get(panel.get("schema"))
        if key is None:
            raise ValueError("Unknown supplemental review panel schema")
        case_ids = [case["id"] for case in panel["cases"]]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Duplicated supplemental source IDs")
        for case in panel["cases"]:
            if case["id"] not in source_hashes or case[key] != source_hashes[case["id"]]:
                raise ValueError("Supplemental review source is not bound to the registered video")
            required.add(case["id"])
        supplement_hashes[item["path"]] = digest
    declared = contract["required_independent_ids"]
    if (len(declared) != len(set(declared)) or set(declared) != required
            or not required <= set(source_hashes)):
        raise ValueError("Independent review contract omitted or added unbound required sources")
    return required, hashlib.sha256(payload).hexdigest(), supplement_hashes


def run(args):
    selection = Path(args.selection).resolve()
    rows = read_rows(selection, 0)
    if len(rows) != 200 or len({row["source_group"] for row in rows}) != 200 or any(row.get("panel_role") != "acceptance" for row in rows):
        raise ValueError("Exactly200 distinct frozen acceptance sources required")
    protocol = json.loads(Path(args.protocol).read_text())
    panel = json.loads(Path(args.independent_panel).read_text())
    registry = json.loads(Path(args.registry).read_text())
    if (protocol["acceptance_manifest_sha256"] != sha256(selection) or panel["acceptance_manifest_sha256"] != sha256(selection)
            or registry.get("expected") != 200 or not registry.get("failures_in_denominator")):
        raise ValueError("Source selection, independent panel or result registry differs")
    if (sha256(Path(args.independent_panel)) != protocol["independent_review_panel_sha256"]
            or sha256(Path(__file__).with_name("ms_body_quality.py")) != protocol["quality_code_sha256"]
            or registry.get("frozen_protocol_sha256") != sha256(Path(args.protocol))):
        raise ValueError("Frozen independent cohort, metric implementation or result protocol changed")
    records = {item["id"]: item for item in registry["records"]}
    ids = {row["id"] for row in rows}
    if len(records) != len(registry["records"]) or set(records) != ids:
        raise ValueError("Result registry silently dropped or duplicated sources")
    human_ids = [item["id"] for item in panel["cases"]]
    if panel["count"] != 40 or len(human_ids) != 40 or len(set(human_ids)) != 40 or not set(human_ids) <= ids:
        raise ValueError("Independent panel must contain 40 distinct registered sources")
    required_human = set(human_ids) | {item["id"] for item in panel["additional_source_disputes"]}
    if not required_human <= ids:
        raise ValueError("Independent source disputes fall outside the registered panel")
    required_human, review_contract_sha, supplemental_panel_hashes = contract_review_ids(
        args.review_contract, required_human, {row["id"]: row["video_sha256"] for row in rows},
        sha256(selection), sha256(Path(args.independent_panel)))
    cases, human, pending, invalid, disputes, main_documents = {}, {}, [], [], [], {}
    review_hashes = {}
    for row in rows:
        sid = row["id"]
        record = records[sid]
        if record["status"] != "candidate_evidence_complete":
            pending.append({"id": sid, "reason": "registered source extraction or evidence incomplete"})
            continue
        main_path = Path(args.reviews) / (sid + ".json")
        if not main_path.is_file():
            pending.append({"id": sid, "reason": "source review absent"})
            continue
        try:
            raw, source, manifest_sha = load_bundle(Path(record["bundle"]))
            facts, arrays, facts_sha = load_facts(Path(record["facts_dir"]))
            if (source["id"] != sid or source["source_group"] != row["source_group"]
                    or source["source_sha256"]["video"] != row["video_sha256"] or sha256(Path(row["video_path"])) != row["video_sha256"]
                    or source["source_sha256"]["model"] != protocol["model_sha256"]
                    or source["source_sha256"]["mapping"] != protocol["mapping_sha256"]
                    or source["point_count"] != 20 or source["source"] != "nlf_official_native_rgb_body"
                    or facts["source_manifest_sha256"] != manifest_sha or facts_sha != record["facts_sha256"]
                    or facts["measurement_code_sha256"] != protocol["measurement_code_sha256"]
                    or facts["implementation_sha256"] != protocol["front_code_sha256"] or facts["parameter_config"] != protocol["config"]):
                raise ValueError("Facts/source/model/parameters are not the frozen revision")
            probe = row["media_probe"]
            if (not np.array_equal(raw["pts"], probe["pts"]) or not np.array_equal(raw["frame_indices"], probe["frame_indices"])
                    or not np.array_equal(raw["time_base"], probe["time_base"])
                    or not np.allclose(raw["timestamps"], probe["time"], rtol=0, atol=1e-9)):
                raise ValueError("Native source timing differs from the selected video")
            document = json.loads(main_path.read_text())
            result = evaluate_review(document, facts, arrays, facts_sha, row)
            if result is None:
                pending.append({"id": sid, "reason": "source/reference/channel review incomplete"})
                continue
            cases[sid], main_documents[sid] = result, document
            review_hashes[sid] = {"main": sha256(main_path)}
            if (result["fp"] or result["critical_errors"] or result["incorrect_removed_occurrences"]
                    or not result["geometry_checks_passed"]
                    or document.get("primary_reference_compilation_after_candidate_view") is True):
                required_human.add(sid)
            human_path = Path(args.independent_reviews) / (sid + ".json")
            if human_path.is_file():
                independent_document = json.loads(human_path.read_text())
                independent = evaluate_review(independent_document, facts, arrays, facts_sha, row)
                if independent is not None and independent["reviewer_type"] == "human" and independent["reviewer"] != result["reviewer"]:
                    human[sid] = independent
                    review_hashes[sid]["independent"] = sha256(human_path)
                    differences = review_disagreements(document, independent_document)
                    if differences:
                        disputes.append({"id": sid, "issues": differences, "status": "requires_adjudication"})
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid.append({"id": sid, "error_type": type(error).__name__, "reason": str(error)})
    measured, independent_metrics = metrics(list(cases.values())), metrics(list(human.values()))
    strata = {}
    for label, predicate in (("partial_body", lambda row: row["rgb_strata"]["framing"] == "partial_or_feet_outside_screen"),
                             ("body_in_frame", lambda row: row["rgb_strata"]["framing"] == "body_in_frame"),
                             ("low_contrast", lambda row: row["rgb_strata"]["dark_or_low_contrast_scene"])):
        subset = [cases[row["id"]] for row in rows if row["id"] in cases and predicate(row)]
        strata[label] = {"registered_videos": sum(predicate(row) for row in rows), "reviewed_videos": len(subset),
                         "metrics": metrics(subset), "video_bootstrap_ci95": video_bootstrap(subset)}
    missing_human = sorted(required_human - set(human))
    complete = (len(cases) == 200 and len(human) >= 40 and not pending and not invalid and not disputes
                and not missing_human and all(item["geometry_checks_passed"] for item in [*cases.values(), *human.values()]))
    sets = [measured, independent_metrics, *[item["metrics"] for item in strata.values()]]
    keys = ("false_positive_fraction", "false_negative_fraction", "boundary_error_p95_seconds", "decidable_support_fraction")
    decidable = all(all(item[key] is not None for key in keys) for item in sets)
    passed = (complete and decidable and all(numeric_gates_pass(item) for item in sets)
              and all(not item["critical_errors"] and not item["incorrect_removed_occurrences"] for item in [*cases.values(), *human.values()]))
    unknown_seconds = sum(item["unknown_operator_seconds"] for item in cases.values())
    operator_seconds = sum(item["total_operator_seconds"] for item in cases.values())
    report = {"schema": "native-body-real-quality-report-v2", "status": "passed" if passed else "failed_quality_gate" if complete and decidable else "pending_or_insufficient_review",
              "quality_gate_passed": bool(passed), "registered_videos": 200, "completed_source_rgb_reviews": len(cases),
              "completed_independent_human_reviews": len(human), "pending": pending, "invalid_reviews_or_bindings": invalid,
              "missing_independent_reviews": missing_human, "unresolved_independent_disagreements": disputes,
              "required_independent_review_ids": sorted(required_human), "independent_review_contract_sha256": review_contract_sha,
              "supplemental_review_panel_sha256": supplemental_panel_hashes,
              "metrics": measured, "independent_human_metrics": independent_metrics, "strata": strata,
              "video_bootstrap_ci95": video_bootstrap(list(cases.values())), "gates": GATES,
              "unknown_operator_seconds": unknown_seconds, "total_reviewed_operator_seconds": operator_seconds,
              "unknown_operator_fraction": unknown_seconds / operator_seconds if operator_seconds else None,
              "cases": list(cases.values()), "review_revision_sha256": review_hashes,
              "one_reference_supports_at_most_one_candidate": True,
              "direction_and_reference_coverage_verified": len(cases) == 200,
              "direction_reference_verification_scope": "all200 registered source reviews; pending sources remain unverified",
              "all_measured_channels_require_full_source_coverage": True, "geometry_unknowns_block_promotion": True,
              "metrics_scope": "completed source reviews only; no full-200 inference while pending",
              "absolute_scale_or_depth_certified": False, "frozen_measurements_changed": False,
              "selection_sha256": sha256(selection), "frozen_protocol_sha256": sha256(Path(args.protocol)),
              "postprocess_registry_sha256": sha256(Path(args.registry)), "implementation_sha256": sha256(Path(__file__)),
              "frozen_verifier_v1_sha256": protocol["quality_code_sha256"]}
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError("Preserve previous quality report revisions")
    out.mkdir(parents=True)
    payload = json_bytes(report)
    atomic_write(out / "quality_report.receipt.json", json_bytes({"quality_report_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "quality_report.json", payload)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("selection", "registry", "reviews", "independent-panel", "independent-reviews", "review-contract", "protocol", "out"):
        ap.add_argument("--" + key, required=True)
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
