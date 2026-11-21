"""Compile explicitly assigned source-first audits and independent review obligations."""
from __future__ import annotations

import argparse
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


def basename(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("Batch output/input names must be plain basenames")
    return value


def main(args):
    batch_path = Path(args.batch).resolve()
    batch = json.loads(batch_path.read_text())
    if batch.get("schema") != "native-body-source-audit-batch-v1":
        raise ValueError("Unknown source-audit batch schema")
    selection = ROOT / "panel-200-v1/acceptance.jsonl"
    rows = {row["id"]: row for row in map(json.loads, selection.read_text().splitlines())}
    registry = {row["id"]: row for row in json.loads((ROOT / "acceptance-panel-v1-postprocess/finished.json").read_text())["records"]}
    cases = batch["cases"]
    if not cases or not set(cases) <= set(rows) or set(rows) != set(registry) or len(rows) != 200:
        raise ValueError("Batch must use the exact frozen source registry")
    package = ROOT / basename(batch["source_package"])
    panel_path = WORK / basename(batch["supplemental_panel"])
    contract_path = WORK / basename(batch["new_contract"])
    old_contract_path = WORK / basename(batch["previous_contract"])
    if package.exists() or panel_path.exists() or contract_path.exists():
        raise ValueError("Preserve previous independent materials")
    if sha256(old_contract_path) != batch["previous_contract_sha256"]:
        raise ValueError("Previous independent review contract changed")
    prepared = []
    for sid, case in cases.items():
        inventory_path = OBS / basename(case["inventory"])
        if sha256(inventory_path) != case["sha256"]:
            raise ValueError("Frozen source-reference revision changed")
        inventory = json.loads(inventory_path.read_text())
        row, record = rows[sid], registry[sid]
        if inventory["id"] != sid or inventory["video_sha256"] != row["video_sha256"] or sha256(Path(row["video_path"])) != row["video_sha256"]:
            raise ValueError("Source-reference/RGB binding changed")
        raw, source, manifest_sha = load_bundle(Path(record["bundle"]))
        facts, arrays, facts_sha = load_facts(Path(record["facts_dir"]))
        if facts_sha != record["facts_sha256"] or facts["source_manifest_sha256"] != manifest_sha:
            raise ValueError("Frozen source/fact revision changed")
        for a, b in (("frame_indices", "frame_indices"), ("pts", "pts"), ("time_base", "time_base"), ("timestamps", "time")):
            if not np.array_equal(raw[a], row["media_probe"][b]):
                raise ValueError("Native frame/PTS binding changed")
        path = WORK / "main_reviews_v2" / (sid + ".json")
        old_bytes = path.read_bytes()
        review = json.loads(old_bytes)
        if review.get("reviewer") is not None or review.get("review_completed") is not False:
            raise ValueError("Do not overwrite an authored review")
        for item in review["source_rgb_sequences"]:
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("Source-only review sequence changed")
        probe = row["media_probe"]
        refs = references(inventory, probe["time"], probe["pts"], probe["time_base"])
        lookup = {(ref["body_part"], ref["measurement_name"], *ref["source_frames"]): ref for ref in refs}
        review.update(reviewer="assistant source-first survey/compiler and candidate comparator; independent source-only checks are not human reviews",
            reviewer_type="assistant", review_completed=True, reference_built_from_rgb_before_reading_candidates=True,
            reference_inventory_complete=True, reference_inventory_sha256=case["sha256"], reference_occurrences=refs,
            reference_origin=inventory["reference_origin"], primary_reference_compilation_after_candidate_view=True,
            source_reference_revised_after_candidate_view=inventory.get("source_revision_after_candidate_view", False),
            primary_prior_partial_diagnostic_exposure=inventory.get("prior_partial_diagnostic_exposure", False),
            absolute_3d_scale_or_depth_certified=False, independent_human_review_completed=False,
            critical_source_time_mapping_errors=[], incorrect_removed_occurrence_ids=[])
        for item in review["channel_assessments"]:
            item["reference_inventory_complete"] = True
            item["intervals"] = source_intervals(inventory, item, probe["time"])
        passed = {
            "source_frame_time_alignment": "Native source frames, integer PTS/time_base and actual seconds bind to the frozen source; no index/fps substitution.",
            "coordinate_reference_labeling": "Facts explicitly state estimated camera/local axes, not calibrated world gravity, true scale, ground contact or whole3D limb path.",
            "occurrence_preservation": "All original events/selected occurrences remain present; strict occurrence-coverage validation is unchanged.",
        }
        review["geometry_checks"] = [{"name": name, "verdict": "pass" if name in passed else "unknown",
            "reason": passed.get(name, "Source projection, hidden joints, uncertainty and unresolved3D quantities do not certify this whole-source check; real independent review remains required.")} for name in GEOMETRY_CHECKS]
        decisions = []
        for event in facts["events"]:
            key = str(int(event["id"].rsplit("_", 1)[1]))
            target = case["matches"].get(key)
            if target is None:
                decisions.append({"event_id": event["id"], "verdict": "unknown",
                    "reason": "No independently decidable source reference verifies the full candidate: ambiguous3D direction, hidden chain, unsupported source interval, unresolved side, or boundaries spanning reversals. Unknown is not stillness or a true positive."})
            else:
                ref = lookup[tuple(target)]
                decisions.append({"event_id": event["id"], "verdict": "supported", "reference_id": ref["id"],
                    "reason": f"Coarse source direction/configuration corresponds to native frames{ref['source_frames']}; this correspondence does not certify exact3D quantity, precise boundaries or physical action count."})
        review["event_decisions"] = decisions
        result = evaluate_review(review, facts, arrays, facts_sha, row)
        if result is None:
            raise ValueError("Source/reference/channel audit is incomplete")
        snapshot, audit_path = [OBS / (sid + suffix) for suffix in (".pre_review_template.json", ".completed_audit.json")]
        if snapshot.exists() or audit_path.exists():
            raise ValueError("Preserve prior source-audit revisions")
        audit = {"schema": "native-body-completed-assistant-diagnostic-audit-v1", "id": sid,
            "source_all81_frames_inspected": True, "candidate_count": len(facts["events"]), "all_candidate_decisions_present": True,
            "diagnostic_counts": {key: result[key] for key in ("tp", "fp", "fn", "unknown", "references")},
            "unmatched_support_reference_ids": result["unmatched_decidable_reference_ids"],
            "unmatched_support_parts_are_not_confirmed_distinct_physical_misses": True,
            "geometry_checks_passed": result["geometry_checks_passed"], "quality_gate_passed": False,
            "source_reference_revised_after_candidate_view": inventory.get("source_revision_after_candidate_view", False),
            "primary_reference_compilation_after_candidate_view": True,
            "reference_inventory_sha256": case["sha256"], "implementation_sha256": sha256(Path(__file__)),
            "batch_sha256": sha256(batch_path), "compiler_dependency_sha256": sha256(Path(__file__).with_name("compile_source_review_batch_v2.py")),
            "reference_dependency_sha256": sha256(Path(__file__).with_name("compile_source_review_batch_v1.py")),
            "independent_human_review_completed": False}
        prepared.append((path, old_bytes, review, snapshot, audit_path, audit))
    for path, old_bytes, review, snapshot, audit_path, audit in prepared:
        atomic_write(snapshot, old_bytes)
        atomic_write(path, json_bytes(review))
        audit["review_sha256"] = sha256(path)
        atomic_write(audit_path, json_bytes(audit))
        print(json.dumps({"id": audit["id"], "diagnostic_counts": audit["diagnostic_counts"], "quality_gate_passed": False}), flush=True)
    package.mkdir()
    for sid in cases:
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
        "cases": [{"id": sid, "video_sha256": rows[sid]["video_sha256"], "source_only_package": str(package)} for sid in cases],
        "instructions": "Review source RGB/actual PTS first and freeze proximal-chain references before predictions. Knee folding, shoe motion and apparent2D direction are not interchangeable3D truth; do not infer missing sides or motion from endpoint states.",
        "human_reviews_completed": 0, "human_verdicts_prefilled": False, "numeric_quality_gates_changed": False}
    atomic_write(panel_path, json_bytes(panel))
    contract.update(revision=contract["revision"] + 1, previous_contract_sha256=sha256(old_contract_path))
    contract["required_independent_ids"] = sorted(set(contract["required_independent_ids"]) | set(cases))
    contract["required_distinct_independent_review_count"] = len(contract["required_independent_ids"])
    contract["supplemental_panels"].append({"path": str(panel_path), "sha256": sha256(panel_path)})
    atomic_write(contract_path, json_bytes(contract))
    original_panel_path = ROOT / "panel-200-v1/independent_review40.json"
    original_panel = json.loads(original_panel_path.read_text())
    base = {case["id"] for case in original_panel["cases"]} | {case["id"] for case in original_panel["additional_source_disputes"]}
    ids, digest, _ = contract_review_ids(contract_path, base, {sid: row["video_sha256"] for sid, row in rows.items()}, sha256(selection), sha256(original_panel_path))
    print(json.dumps({"required_distinct_human_sources": len(ids), "contract_sha256": digest,
        "source_model_facts_or_event_thresholds_changed": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    main(parser.parse_args())
