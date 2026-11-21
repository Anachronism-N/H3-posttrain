"""Compile two source-envelope audits without altering frozen motion candidates."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementation/scripts_omnidance/exp/ms"))
from compile_source_review_batch_v2 import references, source_intervals
from ms_body_quality_v2 import GEOMETRY_CHECKS, contract_review_ids, evaluate_review
from ms_body_review import load_facts
from ms_landmarks import load_bundle
from ms_optimize import atomic_write, json_bytes, sha256

WORK = ROOT / "acceptance-review-workspace-v1"
OBS = WORK / "source_observations"
UP = "wrist_position_along_torso_axis"
ANGLE = "joint_interior_angle"
ELEVATION = "upper_arm_elevation_relative_to_torso"
CASES = {
    "NyfVAyPijuUenemx": {
        "sha256": "109af8926851a18333cc4eeafb8c6a8ca2c96015e03a49d2bffc6a29d60d4476",
        "matches": {
            82: ("right_upper_arm", ELEVATION, 29, 37),
            143: ("left_upper_arm", ELEVATION, 43, 56),
            144: ("left_wrist", UP, 43, 56),
            159: ("right_upper_arm", ELEVATION, 57, 61),
            165: ("left_upper_arm", ELEVATION, 56, 66),
            177: ("right_upper_arm", ELEVATION, 61, 67),
            189: ("left_upper_arm", ELEVATION, 66, 74),
            196: ("right_upper_arm", ELEVATION, 67, 71),
        },
    },
    "FyUgxWjRsOITaKzM": {
        "sha256": "c2fa7cc5fca37ca35f7741b096ae82fe266818bb4f0edc987e8bef554ff9a6d8",
        "matches": {
            10: ("right_elbow", ANGLE, 8, 12),
            44: ("right_elbow", ANGLE, 32, 38),
            59: ("right_elbow", ANGLE, 38, 45),
            63: ("right_elbow", ANGLE, 45, 54),
            77: ("right_elbow", ANGLE, 54, 59),
            11: ("left_elbow", ANGLE, 8, 12),
            21: ("left_elbow", ANGLE, 12, 27),
            39: ("left_elbow", ANGLE, 27, 32),
            42: ("left_elbow", ANGLE, 32, 38),
            66: ("left_elbow", ANGLE, 45, 54),
            98: ("left_elbow", ANGLE, 73, 80),
            20: ("right_wrist", UP, 12, 18),
            57: ("right_wrist", UP, 38, 43),
            65: ("right_wrist", UP, 45, 54),
            54: ("left_wrist", UP, 38, 43),
            64: ("left_wrist", UP, 45, 54),
        },
    },
}


def main():
    selection = ROOT / "panel-200-v1/acceptance.jsonl"
    rows = {row["id"]: row for row in map(json.loads, selection.read_text().splitlines())}
    registry = {row["id"]: row for row in json.loads((ROOT / "acceptance-panel-v1-postprocess/finished.json").read_text())["records"]}
    prepared = []
    for sid, case in CASES.items():
        inventory_path = OBS / (sid + ".reference_inventory.json")
        if sha256(inventory_path) != case["sha256"]:
            raise ValueError("Frozen source reference changed")
        inventory = json.loads(inventory_path.read_text())
        row, record = rows[sid], registry[sid]
        if sha256(Path(row["video_path"])) != row["video_sha256"] or inventory["video_sha256"] != row["video_sha256"]:
            raise ValueError("Frozen source RGB changed")
        raw, source, manifest_sha = load_bundle(Path(record["bundle"]))
        facts, arrays, facts_sha = load_facts(Path(record["facts_dir"]))
        if facts_sha != record["facts_sha256"] or facts["source_manifest_sha256"] != manifest_sha:
            raise ValueError("Frozen fact/raw binding changed")
        for a, b in (("frame_indices", "frame_indices"), ("pts", "pts"), ("time_base", "time_base"), ("timestamps", "time")):
            if not np.array_equal(raw[a], row["media_probe"][b]):
                raise ValueError("Native frame/PTS binding changed")
        path = WORK / "main_reviews_v2" / (sid + ".json")
        old_bytes = path.read_bytes()
        review = json.loads(old_bytes)
        if review.get("reviewer") is not None or review.get("review_completed") is not False:
            raise ValueError("Do not replace an authored source review")
        for item in review["source_rgb_sequences"]:
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("Source-only sequence changed")
        probe = row["media_probe"]
        refs = references(inventory, probe["time"], probe["pts"], probe["time_base"])
        lookup = {(ref["body_part"], ref["measurement_name"], *ref["source_frames"]): ref for ref in refs}
        review.update(reviewer="assistant source-envelope compiler with independent source-only survey and prior partial diagnostic context",
            reviewer_type="assistant", review_completed=True, reference_built_from_rgb_before_reading_candidates=True,
            reference_inventory_complete=True, reference_inventory_sha256=case["sha256"], reference_occurrences=refs,
            reference_origin=inventory["reference_origin"], primary_reference_compilation_after_candidate_view=True,
            primary_prior_partial_diagnostic_exposure=True, absolute_3d_scale_or_depth_certified=False,
            independent_human_review_completed=False, critical_source_time_mapping_errors=[], incorrect_removed_occurrence_ids=[])
        for item in review["channel_assessments"]:
            item["reference_inventory_complete"] = True
            item["intervals"] = source_intervals(inventory, item, probe["time"])
        passed = {
            "source_frame_time_alignment": "All81 input source frames, integer PTS/time_base and actual seconds match the frozen source; no nominalfps substitution.",
            "coordinate_reference_labeling": "Frozen facts explicitly describe model-camera/local reference axes, not calibrated world, gravity, ground contact or full wrist displacement.",
            "occurrence_preservation": "Every original candidate and selected occurrence is retained; strict occurrence coverage validator remains unchanged.",
        }
        review["geometry_checks"] = [{"name": name, "verdict": "pass" if name in passed else "unknown",
            "reason": passed.get(name, "Overlap/prop/garment and unobservable3D quantities leave this whole-source check uncertified. Independent human review is required.")} for name in GEOMETRY_CHECKS]
        decisions = []
        for event in facts["events"]:
            number = int(event["id"].rsplit("_", 1)[1])
            target = case["matches"].get(number)
            if target is None:
                decisions.append({"event_id": event["id"], "verdict": "unknown",
                    "reason": "No source envelope independently verifies this entire candidate: hidden grip/overlapping endpoints, unobservable3D reference, echo, or boundaries spanning source reversals/unknown support. Unknown is not stillness or supported3D truth."})
            else:
                ref = lookup[target]
                decisions.append({"event_id": event["id"], "verdict": "supported", "reference_id": ref["id"],
                    "reason": f"Coarse main-person source configuration corresponds to native frames{ref['source_frames']}; fan fabric/ghosts/finger tips are not wrist observations. This correspondence does not certify exact3D quantity or complete precise boundaries."})
        review["event_decisions"] = decisions
        result = evaluate_review(review, facts, arrays, facts_sha, row)
        if result is None:
            raise ValueError("Source/interval/candidate review is incomplete")
        snapshot = OBS / (sid + ".pre_review_template.json")
        audit_path = OBS / (sid + ".completed_audit.json")
        if snapshot.exists() or audit_path.exists():
            raise ValueError("Preserve existing diagnostic revisions")
        audit = {"schema": "native-body-completed-assistant-diagnostic-audit-v1", "id": sid,
            "source_all81_frames_inspected": True, "candidate_count": len(facts["events"]), "all_candidate_decisions_present": True,
            "diagnostic_counts": {key: result[key] for key in ("tp", "fp", "fn", "unknown", "references")},
            "unmatched_support_reference_ids": result["unmatched_decidable_reference_ids"],
            "unmatched_support_parts_are_not_confirmed_distinct_physical_misses": True,
            "geometry_checks_passed": result["geometry_checks_passed"], "quality_gate_passed": False,
            "primary_prior_partial_diagnostic_exposure": True, "primary_reference_compilation_after_candidate_view": True,
            "reference_inventory_sha256": case["sha256"], "implementation_sha256": sha256(Path(__file__)),
            "compiler_dependency_sha256": sha256(Path(__file__).with_name("compile_source_review_batch_v2.py")),
            "reference_dependency_sha256": sha256(Path(__file__).with_name("compile_source_review_batch_v1.py")),
            "independent_human_review_completed": False}
        prepared.append((path, old_bytes, review, snapshot, audit_path, audit))
    package = ROOT / "independent-review-supplement-sourcebatch3-v1"
    panel_path = WORK / "supplemental_independent_sourcebatch3_review.json"
    contract_path = WORK / "independent_review_contract.v4.json"
    old_contract_path = WORK / "independent_review_contract.v3.json"
    if package.exists() or panel_path.exists() or contract_path.exists():
        raise ValueError("Preserve prior review materials")
    if sha256(old_contract_path) != "7873a5cc9d21efe75181b8c03af7aec1c80090435ebd0625bf529504caefd117":
        raise ValueError("Previous independent contract changed")
    for path, old_bytes, review, snapshot, audit_path, audit in prepared:
        atomic_write(snapshot, old_bytes)
        atomic_write(path, json_bytes(review))
        audit["review_sha256"] = sha256(path)
        atomic_write(audit_path, json_bytes(audit))
        print(json.dumps({"id": audit["id"], "diagnostic_counts": audit["diagnostic_counts"], "quality_gate_passed": False}), flush=True)
    package.mkdir()
    for sid in CASES:
        row, probe = rows[sid], rows[sid]["media_probe"]
        os.symlink(row["video_path"], package / (sid + ".mp4"))
        os.symlink(ROOT / "acceptance-source-sequence-review-v1" / sid, package / (sid + ".source_rgb_frames"))
        with (package / (sid + ".source_time.csv")).open("x", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["native_frame_index", "integer_pts", "time_base_numerator", "time_base_denominator", "relative_seconds"])
            writer.writerows([frame, pts, *probe["time_base"], t] for frame, pts, t in zip(probe["frame_indices"], probe["pts"], probe["time"]))
    contract = json.loads(old_contract_path.read_text())
    panel = {"schema": "native-body-supplemental-independent-review-v1",
        "original_independent_panel_sha256": contract["original_independent_panel_sha256"], "acceptance_manifest_sha256": sha256(selection),
        "cases": [{"id": sid, "video_sha256": rows[sid]["video_sha256"], "source_only_package": str(package)} for sid in CASES],
        "instructions": "Inspect source RGB and actual PTS first, freeze proximal-chain references before viewing predictions. Cloth/ghost motion is not wrist motion; moving-body projections are not world wrist displacement.",
        "human_reviews_completed": 0, "human_verdicts_prefilled": False, "numeric_quality_gates_changed": False}
    atomic_write(panel_path, json_bytes(panel))
    contract.update(revision=4, previous_contract_sha256=sha256(old_contract_path))
    contract["required_independent_ids"] = sorted(set(contract["required_independent_ids"]) | set(CASES))
    contract["required_distinct_independent_review_count"] = len(contract["required_independent_ids"])
    contract["supplemental_panels"].append({"path": str(panel_path), "sha256": sha256(panel_path)})
    atomic_write(contract_path, json_bytes(contract))
    original_panel_path = ROOT / "panel-200-v1/independent_review40.json"
    original_panel = json.loads(original_panel_path.read_text())
    base = {case["id"] for case in original_panel["cases"]} | {case["id"] for case in original_panel["additional_source_disputes"]}
    ids, digest, _ = contract_review_ids(contract_path, base, {sid: row["video_sha256"] for sid, row in rows.items()}, sha256(selection), sha256(original_panel_path))
    print(json.dumps({"required_distinct_human_sources": len(ids), "contract_sha256": digest,
        "source_or_frozen_pose_or_thresholds_changed": False}), flush=True)


if __name__ == "__main__":
    main()
