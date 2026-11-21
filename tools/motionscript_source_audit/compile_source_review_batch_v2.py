"""Compile two further source-first audits and persist independent review obligations."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementation/scripts_omnidance/exp/ms"))
from compile_source_review_batch_v1 import references
from ms_body_quality_v2 import GEOMETRY_CHECKS, contract_review_ids, evaluate_review
from ms_body_review import load_facts
from ms_landmarks import load_bundle
from ms_optimize import atomic_write, json_bytes, sha256

WORK = ROOT / "acceptance-review-workspace-v1"
OBS = WORK / "source_observations"
UP = "wrist_position_along_torso_axis"
ANGLE = "joint_interior_angle"
ELEVATION = "upper_arm_elevation_relative_to_torso"
LATERAL = "wrist_position_along_shoulder_axis"
CASES = {
    "drKSnQygeQBpvOwH": {
        "inventory": "drKSnQygeQBpvOwH.reference_inventory.v2.json",
        "sha256": "3c047c7f72413cad82ef2032080c93f475954cd57a19d67a039f2274fcc5ec41",
        "matches": {
            12: ("left_wrist", UP, 0, 11), 13: ("right_wrist", UP, 0, 11),
            25: ("left_wrist", UP, 11, 17), 32: ("right_wrist", UP, 11, 17),
            129: ("left_wrist", UP, 40, 51), 161: ("left_wrist", UP, 56, 65),
            177: ("left_wrist", UP, 65, 69), 190: ("left_wrist", UP, 69, 74),
            204: ("left_wrist", UP, 74, 79),
            14: ("left_elbow", ANGLE, 0, 11), 15: ("right_elbow", ANGLE, 0, 11),
            44: ("left_elbow", ANGLE, 17, 21), 58: ("left_elbow", ANGLE, 21, 28),
            68: ("left_elbow", ANGLE, 28, 36), 115: ("left_elbow", ANGLE, 40, 51),
            134: ("left_elbow", ANGLE, 51, 56), 156: ("left_elbow", ANGLE, 56, 65),
            172: ("left_elbow", ANGLE, 65, 69), 189: ("left_elbow", ANGLE, 69, 74),
            196: ("left_elbow", ANGLE, 74, 79),
            33: ("left_upper_arm", ELEVATION, 11, 17),
            201: ("left_upper_arm", ELEVATION, 74, 79),
            191: ("right_upper_arm", ELEVATION, 74, 79),
        },
    },
    "CXnOwcDSVHzIUwtx": {
        "inventory": "CXnOwcDSVHzIUwtx.reference_inventory.json",
        "sha256": "71d50353c603e2cfeffbd9714f7141f490924eddac6af6e1ed5880cfe8e8ef8d",
        "matches": {
            14: ("left_wrist", LATERAL, 16, 21),
            44: ("left_elbow", ANGLE, 58, 63),
            48: ("left_wrist", UP, 58, 63),
            50: ("right_elbow", ANGLE, 59, 63),
            51: ("right_wrist", UP, 59, 66),
        },
    },
}


def source_intervals(inventory, item, time):
    key = item["body_part"] + "." + item["measurement_name"]
    spans = inventory["decidable_channel_frame_windows"].get(key, [])
    bounds = sorted({0, len(time) - 1, *(value for span in spans for value in span)})
    result = []
    for a, b in zip(bounds, bounds[1:]):
        visible = any(lo <= (a + b) / 2 < hi for lo, hi in spans)
        result.append({"start": time[a], "end": time[b], "observability": "decidable" if visible else "unknown",
                       "reason": "Source-only proximal-chain tracking permits gross primitive direction in this interval; exact3D values remain uncertified." if visible else inventory["unknown_channel_reason"]})
    return result


def main():
    selection = ROOT / "panel-200-v1/acceptance.jsonl"
    rows = {row["id"]: row for row in map(json.loads, selection.read_text().splitlines())}
    registry = {row["id"]: row for row in json.loads((ROOT / "acceptance-panel-v1-postprocess/finished.json").read_text())["records"]}
    prepared = []
    dependency_sha = sha256(Path(__file__).with_name("compile_source_review_batch_v1.py"))
    for sid, case in CASES.items():
        inventory_path = OBS / case["inventory"]
        if sha256(inventory_path) != case["sha256"]:
            raise ValueError("Frozen source-reference revision changed")
        inventory = json.loads(inventory_path.read_text())
        row, record = rows[sid], registry[sid]
        if inventory["video_sha256"] != row["video_sha256"] or sha256(Path(row["video_path"])) != row["video_sha256"]:
            raise ValueError("Frozen source RGB changed")
        raw, source, manifest_sha = load_bundle(Path(record["bundle"]))
        for a, b in (("frame_indices", "frame_indices"), ("pts", "pts"), ("time_base", "time_base"), ("timestamps", "time")):
            if not np.array_equal(raw[a], row["media_probe"][b]):
                raise ValueError("Native frame/PTS source binding changed")
        facts, arrays, facts_sha = load_facts(Path(record["facts_dir"]))
        if facts_sha != record["facts_sha256"] or facts["source_manifest_sha256"] != manifest_sha:
            raise ValueError("Frozen fact/raw binding changed")
        path = WORK / "main_reviews_v2" / (sid + ".json")
        old_bytes = path.read_bytes()
        review = json.loads(old_bytes)
        if review.get("reviewer") is not None or review.get("review_completed") is not False:
            raise ValueError("Do not replace an authored source review")
        for item in review["source_rgb_sequences"]:
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("Source-only review sequence changed")
        time, pts, base = [row["media_probe"][key] for key in ("time", "pts", "time_base")]
        refs = references(inventory, time, pts, base)
        lookup = {(ref["body_part"], ref["measurement_name"], *ref["source_frames"]): ref for ref in refs}
        review.update(reviewer="assistant source-first compiler and candidate comparator with independent source-only chain crosscheck",
                      reviewer_type="assistant", review_completed=True, reference_built_from_rgb_before_reading_candidates=True,
                      reference_inventory_complete=True, reference_inventory_sha256=case["sha256"], reference_occurrences=refs,
                      reference_origin=inventory["reference_origin"], primary_reference_compilation_after_candidate_view=True,
                      source_reference_revised_after_candidate_view=inventory.get("source_revision_after_candidate_view", False),
                      absolute_3d_scale_or_depth_certified=False, independent_human_review_completed=False,
                      critical_source_time_mapping_errors=[], incorrect_removed_occurrence_ids=[])
        for item in review["channel_assessments"]:
            item["reference_inventory_complete"] = True
            item["intervals"] = source_intervals(inventory, item, time)
        passed = {
            "source_frame_time_alignment": "All81 native source frames and integer PTS/time_base match the frozen video; actual source seconds are used throughout.",
            "coordinate_reference_labeling": "Frozen facts state estimated camera/local axes and no calibrated world, gravity, ground or absolute-depth claims.",
            "occurrence_preservation": "Every original event/selected occurrence is retained; strict coverage validation is unchanged.",
        }
        review["geometry_checks"] = [{"name": name, "verdict": "pass" if name in passed else "unknown",
                                      "reason": passed.get(name, "Partial source support and unobservable/overlapping joints do not certify this check over the full source. Independent human adjudication is required.")} for name in GEOMETRY_CHECKS]
        decisions = []
        for event in facts["events"]:
            number = int(event["id"].rsplit("_", 1)[1])
            match = case["matches"].get(number)
            if match is None:
                decisions.append({"event_id": event["id"], "verdict": "unknown",
                    "reason": "No source reference independently verifies this entire candidate: unobservable3D channel, hidden required chain, long boundaries spanning holds/reversals, or interrupted support. Unknown is not stillness or a true positive."})
            else:
                ref = lookup[match]
                decisions.append({"event_id": event["id"], "verdict": "supported", "reference_id": ref["id"],
                    "reason": f"Coarse source direction corresponds to native frames{ref['source_frames']}. This diagnostic correspondence does not certify exact3D angle, distance, speed or extrema; full-geometry checks remain unknown."})
        review["event_decisions"] = decisions
        result = evaluate_review(review, facts, arrays, facts_sha, row)
        if result is None:
            raise ValueError("Source/channel audit is incomplete")
        snapshot = OBS / (sid + ".pre_review_template.json")
        audit_path = OBS / (sid + ".completed_audit.json")
        if snapshot.exists() or audit_path.exists():
            raise ValueError("Preserve earlier diagnostic revisions")
        audit = {"schema": "native-body-completed-assistant-diagnostic-audit-v1", "id": sid,
                 "source_all81_frames_inspected": True, "candidate_count": len(facts["events"]), "all_candidate_decisions_present": True,
                 "diagnostic_counts": {key: result[key] for key in ("tp", "fp", "fn", "unknown", "references")},
                 "unmatched_support_reference_ids": result["unmatched_decidable_reference_ids"],
                 "unmatched_support_parts_are_not_confirmed_distinct_physical_misses": True,
                 "geometry_checks_passed": result["geometry_checks_passed"], "quality_gate_passed": False,
                 "primary_reference_compilation_after_candidate_view": True,
                 "source_reference_revised_after_candidate_view": inventory.get("source_revision_after_candidate_view", False),
                 "reference_inventory_sha256": case["sha256"], "implementation_sha256": sha256(Path(__file__)),
                 "reference_compiler_dependency_sha256": dependency_sha, "independent_human_review_completed": False}
        prepared.append((path, old_bytes, review, snapshot, audit_path, audit))
    package = ROOT / "independent-review-supplement-sourcebatch2-v1"
    panel_path = WORK / "supplemental_independent_sourcebatch2_review.json"
    contract_path = WORK / "independent_review_contract.v3.json"
    if package.exists() or panel_path.exists() or contract_path.exists():
        raise ValueError("Preserve previous independent review materials")
    for path, old_bytes, review, snapshot, audit_path, audit in prepared:
        atomic_write(snapshot, old_bytes)
        atomic_write(path, json_bytes(review))
        audit["review_sha256"] = sha256(path)
        atomic_write(audit_path, json_bytes(audit))
        print(json.dumps({"id": audit["id"], "diagnostic_counts": audit["diagnostic_counts"],
                          "quality_gate_passed": False}), flush=True)
    package.mkdir()
    for sid in CASES:
        row, probe = rows[sid], rows[sid]["media_probe"]
        os.symlink(row["video_path"], package / (sid + ".mp4"))
        os.symlink(ROOT / "acceptance-source-sequence-review-v1" / sid, package / (sid + ".source_rgb_frames"))
        with (package / (sid + ".source_time.csv")).open("x", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["native_frame_index", "integer_pts", "time_base_numerator", "time_base_denominator", "relative_seconds"])
            writer.writerows([frame, point, *probe["time_base"], second] for frame, point, second in zip(probe["frame_indices"], probe["pts"], probe["time"]))
    old_contract_path = WORK / "independent_review_contract.v2.json"
    if sha256(old_contract_path) != "e814c50d265f2e3303bbc654f557c02511f49ee40c8e01946e937d0cd69214e0":
        raise ValueError("Prior independent review contract changed")
    contract = json.loads(old_contract_path.read_text())
    panel = {"schema": "native-body-supplemental-independent-review-v1",
             "original_independent_panel_sha256": contract["original_independent_panel_sha256"],
             "acceptance_manifest_sha256": sha256(selection),
             "cases": [{"id": sid, "video_sha256": rows[sid]["video_sha256"], "source_only_package": str(package)} for sid in CASES],
             "instructions": "Inspect source RGB and actual source PTS first, freeze proximal-chain references, then compare predictions; do not copy assistant labels. Do not infer a hidden arm from the opposite arm.",
             "human_reviews_completed": 0, "human_verdicts_prefilled": False, "numeric_quality_gates_changed": False}
    atomic_write(panel_path, json_bytes(panel))
    contract.update(revision=3, previous_contract_sha256=sha256(old_contract_path))
    contract["required_independent_ids"] = sorted(set(contract["required_independent_ids"]) | set(CASES))
    contract["required_distinct_independent_review_count"] = len(contract["required_independent_ids"])
    contract["supplemental_panels"].append({"path": str(panel_path), "sha256": sha256(panel_path)})
    atomic_write(contract_path, json_bytes(contract))
    original_panel_path = ROOT / "panel-200-v1/independent_review40.json"
    original_panel = json.loads(original_panel_path.read_text())
    base = {case["id"] for case in original_panel["cases"]} | {case["id"] for case in original_panel["additional_source_disputes"]}
    ids, digest, _ = contract_review_ids(contract_path, base, {sid: row["video_sha256"] for sid, row in rows.items()},
                                        sha256(selection), sha256(original_panel_path))
    print(json.dumps({"required_distinct_human_sources": len(ids), "contract_sha256": digest,
                      "frozen_model_facts_or_thresholds_changed": False}), flush=True)


if __name__ == "__main__":
    main()
