"""Compile two source-first assistant diagnostics without changing frozen facts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementation/scripts_omnidance/exp/ms"))
from ms_body_quality_v2 import GEOMETRY_CHECKS, evaluate_review
from ms_body_review import load_facts
from ms_landmarks import load_bundle
from ms_optimize import atomic_write, json_bytes, sha256

WORK = ROOT / "acceptance-review-workspace-v1"
OBS = WORK / "source_observations"
ANGLE = "joint_interior_angle"
UP = "wrist_position_along_torso_axis"
LATERAL = "wrist_position_along_shoulder_axis"

SOURCE_HASHES = {
    "kucgimcrldnunIqv": "e45a7cfe917a4f9d1d1a338782dd9f2b7c1ecf15bc7af68fe5dfd3c3df801d2d",
    "rRwNMQMKDoQTyHuS": "a3bd91f729caf28ce6f9c22444964c28222587b81b9f3536134006df67e8c7c8",
}
MATCHES = {
    "kucgimcrldnunIqv": {
        9: ("right_wrist", UP, 6, 9),
        10: ("left_wrist", UP, 7, 11),
        15: ("right_wrist", UP, 9, 15),
        17: ("left_wrist", UP, 11, 13),
        23: ("left_wrist", UP, 13, 18),
        29: ("right_wrist", UP, 15, 21),
        34: ("left_wrist", UP, 24, 27),
        41: ("left_wrist", UP, 30, 35),
        46: ("right_wrist", UP, 35, 42),
        52: ("left_wrist", UP, 44, 57),
        55: ("right_wrist", UP, 42, 49),
        63: ("right_wrist", UP, 62, 71),
        69: ("left_wrist", UP, 57, 62),
        88: ("right_wrist", UP, 76, 79),
        60: ("left_wrist", LATERAL, 52, 56),
        64: ("left_wrist", LATERAL, 56, 62),
        11: ("left_elbow", ANGLE, 7, 11),
        13: ("right_elbow", ANGLE, 9, 15),
        16: ("left_elbow", ANGLE, 11, 13),
        20: ("right_elbow", ANGLE, 15, 17),
        21: ("left_elbow", ANGLE, 13, 18),
        31: ("right_elbow", ANGLE, 17, 21),
        32: ("left_elbow", ANGLE, 24, 27),
        40: ("right_elbow", ANGLE, 35, 39),
        47: ("right_elbow", ANGLE, 39, 42),
        53: ("right_elbow", ANGLE, 42, 45),
        58: ("right_elbow", ANGLE, 45, 49),
        61: ("left_elbow", ANGLE, 52, 56),
        65: ("right_elbow", ANGLE, 62, 66),
        75: ("right_elbow", ANGLE, 66, 71),
    },
    "rRwNMQMKDoQTyHuS": {
        8: ("right_wrist", UP, 14, 19),
        17: ("right_wrist", UP, 25, 28),
        25: ("right_wrist", UP, 57, 62),
        28: ("left_wrist", UP, 57, 62),
        3: ("right_elbow", ANGLE, 14, 19),
        16: ("right_elbow", ANGLE, 25, 28),
    },
}


def references(inventory, time, pts, base):
    rows = []
    for track in inventory["tracks"]:
        frames, directions = track["frames"], track["directions"]
        if len(frames) != len(directions) + 1:
            raise ValueError("Frozen source reversal track length differs")
        name = {"elbow_internal_angle": ANGLE, "wrist_position_along_body_right_axis": LATERAL}.get(track["measurement_name"], track["measurement_name"])
        for a, b, direction in zip(frames, frames[1:], directions):
            if not 0 <= a < b < len(time):
                raise ValueError("Reference native frame interval differs")
            action = track.get("action")
            if name == ANGLE:
                action, direction = {"bend": ("bends", "decreasing_joint_angle"), "extend": ("extends", "increasing_joint_angle")}[direction]
            direction = {"positive_body_right_axis": "positive_anatomical_right_axis", "negative_body_right_axis": "negative_anatomical_right_axis"}.get(direction, direction)
            rows.append({"id": f"ref_{len(rows) + 1:03d}", "body_part": track["body_part"], "measurement_name": name,
                         "action": action, "direction": direction, "start": time[a], "end": time[b],
                         "source_frames": [a, b], "source_pts": [pts[a], pts[b]], "time_base": base,
                         "decidable": track.get("decidable", True),
                         "reason": track.get("reason", "Frozen source shoulder-elbow-wrist chain supports this coarse trend; exact3D magnitude, depth and extrema are not certified."),
                         "exact3d_amplitude_or_depth_certified": False})
    return rows


def channel_intervals(sid, part, name, time):
    spans = []
    if sid == "kucgimcrldnunIqv":
        if name == UP:
            spans = [(0, 71)] if part == "left_wrist" else [(0, 71), (76, 80)]
        elif name == ANGLE and part in ("left_elbow", "right_elbow"):
            spans = [(7 if part == "left_elbow" else 9, 71)]
        elif name == LATERAL and part == "left_wrist":
            spans = [(52, 62)]
    elif name == UP:
        spans = [(12, 29), (43, 80)]
    elif part == "right_elbow" and name == ANGLE:
        spans = [(14, 28)]
    boundaries = sorted({0, 80, *(value for span in spans for value in span)})
    rows = []
    for a, b in zip(boundaries, boundaries[1:]):
        middle = (a + b) / 2
        visible = any(lo <= middle < hi for lo, hi in spans)
        rows.append({"start": time[a], "end": time[b], "observability": "decidable" if visible else "unknown",
                     "reason": "Source-only proximal-chain tracking supports gross primitive direction here; exact3D values are not certified." if visible else
                     "Source does not independently expose this channel: screen exclusion, clothing, overlap, small ambiguous movement, or unobservable depth/reference. No algorithm output is used to declare source invisibility."})
    return rows


def main():
    rows = {row["id"]: row for row in map(json.loads, (ROOT / "panel-200-v1/acceptance.jsonl").read_text().splitlines())}
    registry = {row["id"]: row for row in json.loads((ROOT / "acceptance-panel-v1-postprocess/finished.json").read_text())["records"]}
    for sid, assignments in MATCHES.items():
        inventory_path = OBS / (sid + ".reference_inventory.json")
        inventory_sha = sha256(inventory_path)
        if SOURCE_HASHES[sid] and inventory_sha != SOURCE_HASHES[sid]:
            raise ValueError("Frozen source reference inventory changed")
        inventory = json.loads(inventory_path.read_text())
        row, record = rows[sid], registry[sid]
        if inventory["video_sha256"] != row["video_sha256"] or sha256(Path(row["video_path"])) != row["video_sha256"]:
            raise ValueError("Frozen source video changed")
        raw, source, _ = load_bundle(Path(record["bundle"]))
        for key, probe_key in (("pts", "pts"), ("time_base", "time_base"), ("timestamps", "time"), ("frame_indices", "frame_indices")):
            if not np.array_equal(raw[key], row["media_probe"][probe_key]):
                raise ValueError("Frozen source frame/PTS binding changed")
        facts, arrays, digest = load_facts(Path(record["facts_dir"]))
        if digest != record["facts_sha256"]:
            raise ValueError("Frozen measurement facts changed")
        refs = references(inventory, row["media_probe"]["time"], row["media_probe"]["pts"], row["media_probe"]["time_base"])
        lookup = {(x["body_part"], x["measurement_name"], *x["source_frames"]): x for x in refs}
        review_path = WORK / "main_reviews_v2" / (sid + ".json")
        original_bytes = review_path.read_bytes()
        review = json.loads(original_bytes)
        if review.get("reviewer") is not None or review.get("review_completed") is not False:
            raise ValueError("Do not replace a previously authored review")
        review.update(reviewer="assistant source-first reference compiler and candidate comparator", reviewer_type="assistant",
                      review_completed=True, reference_built_from_rgb_before_reading_candidates=True, reference_inventory_complete=True,
                      reference_inventory_sha256=inventory_sha, reference_occurrences=refs,
                      reference_origin=inventory["reference_origin"], primary_reference_compilation_after_candidate_view=True,
                      source_reference_label_normalization_after_candidate_view=True,
                      source_reference_intervals_or_trends_changed_after_candidate_view=False,
                      absolute_3d_scale_or_depth_certified=False, independent_human_review_completed=False,
                      critical_source_time_mapping_errors=[], incorrect_removed_occurrence_ids=[])
        for item in review["channel_assessments"]:
            item["reference_inventory_complete"] = True
            item["intervals"] = channel_intervals(sid, item["body_part"], item["measurement_name"], row["media_probe"]["time"])
        verdicts = {
            "source_frame_time_alignment": ("pass", "All81 native frames and actual integer PTS/time_base match the frozen source; source detail and frozen projection frames are aligned."),
            "coordinate_reference_labeling": ("pass", "Facts explicitly use estimated camera/local axes, no world/gravity/ground or absolute-depth certification."),
            "occurrence_preservation": ("pass", "Frozen events and selected occurrences remain unchanged; the strict occurrence-coverage validator checks every original candidate."),
        }
        review["geometry_checks"] = [{"name": key, "verdict": verdicts.get(key, ("unknown", "Partial source observations cannot certify this check over every candidate/required point; independent human adjudication remains required."))[0],
                                     "reason": verdicts.get(key, ("unknown", "Partial source observations cannot certify this check over every candidate/required point; independent human adjudication remains required."))[1]} for key in GEOMETRY_CHECKS]
        decisions = []
        for event in facts["events"]:
            number = int(event["id"].rsplit("_", 1)[1])
            assignment = assignments.get(number)
            if assignment is None:
                reason = "No independently decidable source reference verifies this complete candidate: unobservable3D channel, ambiguous support/boundary, or a span across source reversal/overlap. Unknown is neither true positive nor evidence of stillness."
                decisions.append({"event_id": event["id"], "verdict": "unknown", "reason": reason})
            else:
                ref = lookup[assignment]
                decisions.append({"event_id": event["id"], "verdict": "supported", "reference_id": ref["id"],
                                  "reason": f"Coarse source direction corresponds to native frames{ref['source_frames']}; this diagnostic match does not certify exact3D magnitude or boundary. Full-geometry checks remain unknown and independent adjudication is required."})
        review["event_decisions"] = decisions
        result = evaluate_review(review, facts, arrays, digest, row)
        if result is None:
            raise ValueError("Assistant diagnostic review is incomplete")
        snapshot = OBS / (sid + ".pre_review_template.json")
        audit = OBS / (sid + ".completed_audit.json")
        if snapshot.exists() or audit.exists():
            raise ValueError("Preserve previous source diagnostic revisions")
        payload = json_bytes(review)
        atomic_write(snapshot, original_bytes)
        atomic_write(review_path, payload)
        atomic_write(audit, json_bytes({"schema": "native-body-completed-assistant-diagnostic-audit-v1", "id": sid,
            "source_all81_frames_inspected": True, "candidate_count": len(facts["events"]), "all_candidate_decisions_present": True,
            "diagnostic_counts": {key: result[key] for key in ("tp", "fp", "fn", "unknown", "references")},
            "unmatched_support_reference_ids": result["unmatched_decidable_reference_ids"],
            "unmatched_support_parts_are_not_confirmed_distinct_physical_misses": True,
            "geometry_checks_passed": result["geometry_checks_passed"], "quality_gate_passed": False,
            "primary_reference_compilation_after_candidate_view": True,
            "source_reference_intervals_changed_after_candidate_view": False,
            "review_sha256": sha256(review_path), "reference_inventory_sha256": inventory_sha,
            "implementation_sha256": sha256(Path(__file__)), "independent_human_review_completed": False}))
        print(json.dumps({"id": sid, "diagnostic_counts": {key: result[key] for key in ("tp", "fp", "fn", "unknown", "references")},
                          "geometry_checks_passed": result["geometry_checks_passed"], "quality_gate_passed": False}), flush=True)


if __name__ == "__main__":
    main()
