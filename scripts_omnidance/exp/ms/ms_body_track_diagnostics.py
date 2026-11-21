"""Non-destructive native-joint continuity diagnostics, not action validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ms_landmarks import load_bundle, require_new_output
from ms_optimize import atomic_write, json_bytes, sha256


BONES = tuple((f"{side}_{a}", f"{side}_{b}") for side in ("left", "right")
              for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("hip", "knee"), ("knee", "ankle")))
POLICY = {"side_swap_cost_ratio_max": .25, "relative_bone_length_ratio_min": 1.5,
          "common_scale_min_supported_bones": 4, "minimum_length_m": 1e-5,
          "warning_thresholds_are_not_semantic_admission_rules": True}


def source_pair(raw, end):
    return {"frame_start": int(raw["frame_indices"][end - 1]), "frame_end": int(raw["frame_indices"][end]),
            "pts_start": int(raw["pts"][end - 1]), "pts_end": int(raw["pts"][end]),
            "seconds_start": float(raw["timestamps"][end - 1]), "seconds_end": float(raw["timestamps"][end]),
            "time_base": np.asarray(raw["time_base"]).tolist()}


def diagnose(raw):
    if np.asarray(raw["units"]).item() != "m" or np.asarray(raw["coordinate_frame"]).item() != "camera_xright_ydown_zforward":
        raise ValueError("Native metre-valued camera-coordinate predictions required")
    points = np.asarray(raw["points_m"], dtype=float)
    time = np.asarray(raw["timestamps"], dtype=float)
    names = np.asarray(raw["names"]).tolist()
    pixels = np.asarray(raw["points2d_px"], dtype=float)
    valid = np.asarray(raw["valid"])
    size = np.asarray(raw["frame_size"])
    if (points.shape != (len(time), len(names), 3) or pixels.shape != (len(time), len(names), 2)
            or valid.shape != points.shape[:2] or len(set(names)) != len(names)
            or time.ndim != 1 or not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)
            or size.shape != (2,) or np.any(size <= 0)
            or valid.dtype.kind not in "biu" or np.any((valid != 0) & (valid != 1))):
        raise ValueError("Native point shapes, timestamps or support masks do not match")
    frame_indices, pts, base = [np.asarray(raw[key]) for key in ("frame_indices", "pts", "time_base")]
    if (frame_indices.shape != time.shape or pts.shape != time.shape or base.shape != (2,)
            or any(value.dtype.kind not in "iu" for value in (frame_indices, pts, base))
            or np.any(frame_indices < 0) or np.any(np.diff(frame_indices.astype(np.int64)) != 1)
            or np.any(np.diff(pts.astype(np.int64)) <= 0) or np.any(base <= 0)
            or not np.allclose(time - time[0], (pts.astype(np.int64) - int(pts[0])) * (int(base[0]) / int(base[1])), rtol=0, atol=1e-9)):
        raise ValueError("Consecutive native frame indices and integer PTS must match source seconds")
    good = (valid.astype(bool) & np.isfinite(points).all(axis=-1) & (points[:, :, 2] > 0)
            & np.isfinite(pixels).all(axis=-1) & (pixels >= 0).all(axis=-1) & (pixels < size).all(axis=-1))
    lookup = {name: i for i, name in enumerate(names)}
    dt = np.diff(time)
    side_candidates, bone_candidates, peaks = [], [], []
    side_support = {}
    for group, joints in (("arms", ("shoulder", "elbow", "wrist")), ("legs", ("hip", "knee", "ankle"))):
        labels = [f"{side}_{joint}" for side in ("left", "right") for joint in joints]
        if not all(name in lookup for name in labels):
            side_support[group] = {"assessable_intervals": 0, "reason": "required native joint names absent"}
            continue
        indices = [lookup[name] for name in labels]
        supported = good[:, indices].all(axis=1)
        usable = supported[1:] & supported[:-1]
        local = points[:, indices].copy()
        local -= .5 * (local[:, 0:1] + local[:, 3:4])
        direct = np.sum((local[1:] - local[:-1]) ** 2, axis=(1, 2))
        swapped = np.sum((local[1:] - local[:-1, [3, 4, 5, 0, 1, 2]]) ** 2, axis=(1, 2))
        ratio = np.divide(swapped, direct, out=np.full(len(dt), np.inf), where=direct > 1e-8)
        flagged = usable & (direct > 1e-8) & (ratio <= POLICY["side_swap_cost_ratio_max"])
        side_support[group] = {"assessable_intervals": int(usable.sum()), "required_names": labels}
        for index in np.flatnonzero(flagged):
            side_candidates.append({"region": group, **source_pair(raw, index + 1),
                                    "direct_centered_cost_m2": float(direct[index]),
                                    "swapped_centered_cost_m2": float(swapped[index]), "swapped_to_direct_cost_ratio": float(ratio[index]),
                                    "confirmed_side_swap": False, "true_turn_or_crossing_is_possible": True})
    bone_labels, lengths, supports = [], [], []
    for a, b in BONES:
        if a not in lookup or b not in lookup:
            continue
        i, j = lookup[a], lookup[b]
        length = np.linalg.norm(points[:, i] - points[:, j], axis=-1)
        supported = good[:, i] & good[:, j] & np.isfinite(length) & (length > POLICY["minimum_length_m"])
        bone_labels.append(a + "--" + b)
        lengths.append(length)
        supports.append(supported)
    scale_available = np.zeros(len(time), dtype=bool)
    if lengths:
        length = np.column_stack(lengths)
        support = np.column_stack(supports)
        reference = np.full(len(lengths), np.nan)
        for column in range(len(lengths)):
            if support[:, column].any():
                reference[column] = np.median(length[support[:, column], column])
        ratios = np.where(support, length / reference, np.nan)
        common_scale = np.full(len(time), np.nan)
        for row in range(len(time)):
            selected = ratios[row, np.isfinite(ratios[row])]
            if len(selected) >= POLICY["common_scale_min_supported_bones"]:
                common_scale[row] = np.median(selected)
                scale_available[row] = True
        normalized = length / common_scale[:, None]
        usable = support[1:] & support[:-1] & scale_available[1:, None] & scale_available[:-1, None]
        change = normalized[1:] / normalized[:-1]
        magnitude = np.maximum(change, 1 / change)
        for index, column in np.argwhere(usable & (magnitude >= POLICY["relative_bone_length_ratio_min"])):
            bone_candidates.append({"bone": bone_labels[column], **source_pair(raw, int(index) + 1),
                                    "raw_length_start_m": float(length[index, column]), "raw_length_end_m": float(length[index + 1, column]),
                                    "common_model_scale_start": float(common_scale[index]), "common_model_scale_end": float(common_scale[index + 1]),
                                    "normalized_length_change_ratio": float(change[index, column]), "confirmed_joint_error": False})
    for i, name in enumerate(names):
        usable = good[1:, i] & good[:-1, i]
        available = np.flatnonzero(usable)
        if not len(available):
            peaks.append({"point": name, "peak_camera_displacement_speed_m_per_s": None,
                          "assessable_intervals": 0, "reason": "no consecutive native screen-supported predictions"})
            continue
        speed = np.linalg.norm(points[1:, i] - points[:-1, i], axis=-1) / dt
        peak = int(available[np.argmax(speed[available])])
        peaks.append({"point": name, **source_pair(raw, peak + 1), "assessable_intervals": len(available),
                      "peak_camera_displacement_speed_m_per_s": float(speed[peak]),
                      "speed_includes_camera_and_model_motion": True, "peak_is_not_a_confirmed_jump_error": True})
    return {"schema": "native-body-track-diagnostics-v1", "source_frames": len(time), "native_point_count": len(names),
            "policy": POLICY, "side_correspondence_support": side_support, "side_swap_candidates": side_candidates,
            "relative_bone_length_change_candidates": bone_candidates, "common_scale_available_frames": int(scale_available.sum()),
            "point_displacement_peaks": peaks, "quality_mask_is_observability_ground_truth": False,
            "centered_cost_removes_common_translation_not_true_body_rotation": True,
            "bone_ratio_removes_common_model_scale_not_certified_metric_scale": True,
            "source_gaps_interpolated_or_stitched": False, "resampling_or_smoothing_performed": False,
            "automatic_relabeling": False, "candidate_events_deleted": False, "physical_impossibility_inferred": False,
            "semantic_quality_gate_passed": False, "independent_human_review_completed": False}


def run(args):
    registry_path, protocol_path, out = map(lambda path: Path(path).resolve(), (args.registry, args.protocol, args.out))
    require_new_output(out, [registry_path, protocol_path])
    registry = json.loads(registry_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    records = registry["records"]
    if registry["expected"] != 200 or len(records) != 200 or len({record["id"] for record in records}) != 200:
        raise ValueError("Exactly200 registered sources required")
    if registry.get("frozen_protocol_sha256") != sha256(protocol_path):
        raise ValueError("Diagnostic inputs differ from the frozen measurement protocol")
    out.mkdir(parents=True)
    summaries = []
    for record in records:
        sid = record["id"]
        if record["status"] != "candidate_evidence_complete":
            summaries.append({"id": sid, "status": "upstream_incomplete", "source_record": record})
            continue
        try:
            raw, source, manifest_sha = load_bundle(Path(record["bundle"]))
            if (source["id"] != sid or source["source_sha256"]["model"] != protocol["model_sha256"]
                    or source["source_sha256"]["mapping"] != protocol["mapping_sha256"]):
                raise ValueError("Native model, mapping or source identity changed")
            report = diagnose(raw)
            report.update(id=sid, source_video_sha256=source["source_sha256"]["video"], raw_sha256=source["raw_sha256"],
                          source_manifest_sha256=manifest_sha, source_group=source["source_group"],
                          implementation_sha256=sha256(Path(__file__)))
            payload = json_bytes(report)
            atomic_write(out / (sid + ".receipt.json"), json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest()}))
            atomic_write(out / (sid + ".json"), payload)
            summaries.append({"id": sid, "status": "diagnostic_complete", "report_file": sid + ".json",
                              "report_sha256": hashlib.sha256(payload).hexdigest(),
                              "side_swap_candidates": len(report["side_swap_candidates"]),
                              "bone_change_candidates": len(report["relative_bone_length_change_candidates"]),
                              "candidate_flags_are_not_confirmed_errors": True})
        except (OSError, ValueError, KeyError) as error:
            summaries.append({"id": sid, "status": "diagnostic_failed", "error_type": type(error).__name__, "reason": str(error)})
    terminal = {"schema": "native-body-track-diagnostic-panel-v1", "expected": 200, "records": summaries,
                "registered_count": len(summaries), "all_diagnostics_complete": all(record["status"] == "diagnostic_complete" for record in summaries),
                "policy": POLICY, "input_registry_sha256": sha256(registry_path), "frozen_protocol_sha256": sha256(protocol_path),
                "hypotheses_are_not_motion_false_positives": True, "source_or_facts_modified": False,
                "semantic_quality_gate_passed": False}
    payload = json_bytes(terminal)
    atomic_write(out / "finished.receipt.json", json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "finished.json", payload)
    return terminal


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("registry", "protocol", "out"):
        ap.add_argument("--" + key, required=True)
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
