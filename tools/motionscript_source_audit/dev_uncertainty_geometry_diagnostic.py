"""Developer-only uncertainty/geometry association; warnings are not ground truth."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementation/scripts_omnidance/exp/ms"))
from ms_landmarks import load_bundle
from ms_optimize import atomic_write, json_bytes, sha256


def profile(value, controls):
    if not len(controls):
        return {"value_native": float(value), "control_count": 0, "control_median_native": None,
                "midrank_fraction": None, "value_over_control_median": None}
    median = float(np.median(controls))
    rank = float((np.sum(controls < value) + .5 * np.sum(controls == value)) / len(controls))
    return {"value_native": float(value), "control_count": len(controls), "control_median_native": median,
            "midrank_fraction": rank, "value_over_control_median": float(value / median) if median > 0 else None}


def main():
    development_path = ROOT / "panel-200-v1/development.jsonl"
    registry_path = ROOT / "dev-panel-v1/finished.json"
    diagnostic_path = ROOT / "dev-panel-v1-postprocess/native_track_diagnostics.json"
    mapping_path = ROOT / "assets/body20_mapping.v1.json"
    asset_path = ROOT / "assets/nlf_asset.json"
    out = ROOT / "development-uncertainty-geometry-v1"
    if out.exists():
        raise ValueError("Preserve previous developer diagnostic revisions")
    development = {row["id"]: row for row in map(json.loads, development_path.read_text().splitlines())}
    registry = json.loads(registry_path.read_text())
    records = {row["id"]: row for row in registry["records"]}
    original = json.loads(diagnostic_path.read_text())
    prior_records = {row["id"]: row for row in original["records"]}
    warnings = original["ranked_bone_review_hypotheses"]
    if (len(development) != 20 or len(records) != 20 or len(prior_records) != 20
            or set(development) != set(records) or set(development) != set(prior_records)
            or registry["expected"] != 20 or original["expected"] != 20):
        raise ValueError("Exactly20 frozen development sources required")
    if len(warnings) != sum(row["bone_hypotheses"] for row in prior_records.values()):
        raise ValueError("Reuse the complete prior warning inventory, not a ranked subset")
    model_sha = json.loads(asset_path.read_text())["sha256"]
    mapping_sha = sha256(mapping_path)
    out.mkdir()
    summaries = []
    shorter_profiles = {}
    all_profiles = {}
    for sid, row in development.items():
        try:
            if row["panel_role"] != "development" or row["split"] != "train":
                raise ValueError("No acceptance source may enter developer calibration")
            raw, source, manifest_sha = load_bundle(Path(records[sid]["bundle"]))
            if (source["id"] != sid or source["raw_sha256"] != prior_records[sid]["source_raw_sha256"]
                    or source["source_sha256"]["video"] != row["video_sha256"]
                    or sha256(Path(row["video_path"])) != row["video_sha256"]
                    or source["source_sha256"]["model"] != model_sha or source["source_sha256"]["mapping"] != mapping_sha):
                raise ValueError("Frozen source/model/mapping changed")
            for a, b in (("frame_indices", "frame_indices"), ("pts", "pts"), ("time_base", "time_base"), ("timestamps", "time")):
                if not np.array_equal(raw[a], row["media_probe"][b]):
                    raise ValueError("Native source frame/PTS binding changed")
            points = np.asarray(raw["points_m"], dtype=float)
            pixels = np.asarray(raw["points2d_px"], dtype=float)
            uncertainty = np.asarray(raw["uncertainty"], dtype=float)
            names = raw["names"].tolist()
            size = np.asarray(raw["frame_size"])
            good = (raw["valid"].astype(bool) & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
                    & np.isfinite(pixels).all(axis=-1) & (pixels >= 0).all(axis=-1) & (pixels < size).all(axis=-1))
            if uncertainty.shape != good.shape or len(set(names)) != len(names) or len(names) != 20:
                raise ValueError("Uncertainty/native-joint order or shape changed")
            finite_uncertainty = np.isfinite(uncertainty) & (uncertainty >= 0)
            lookup = {name: i for i, name in enumerate(names)}
            frame_rows = {int(frame): i for i, frame in enumerate(raw["frame_indices"])}
            case_warnings = [warning for warning in warnings if warning["id"] == sid]
            if len(case_warnings) != prior_records[sid]["bone_hypotheses"]:
                raise ValueError("Prior per-source warnings changed")
            flagged = {name: set() for name in names}
            for warning in case_warnings:
                for name in warning["bone"].split("--"):
                    flagged[name].update((warning["frame_start"], warning["frame_end"]))
            controls, point_profiles = {}, []
            for name, column in lookup.items():
                excluded = np.isin(raw["frame_indices"], list(flagged[name]))
                allowed = good[:, column] & finite_uncertainty[:, column] & ~excluded
                controls[name] = uncertainty[allowed, column]
                sample = controls[name]
                point_profiles.append({"point": name, "model_screen_supported_frames": int(good[:, column].sum()),
                    "nonwarning_control_frames": int(allowed.sum()), "warning_endpoint_frames": sorted(flagged[name]),
                    "control_uncertainty_quantiles_native": np.quantile(sample, [.5, .75, .9, .95]).tolist() if len(sample) else None,
                    "control_predictions_are_not_certified_correct": True})
            comparisons = []
            for warning in case_warnings:
                a, b = warning["bone"].split("--")
                lengths = [warning["raw_length_start_m"] / warning["common_model_scale_start"],
                           warning["raw_length_end_m"] / warning["common_model_scale_end"]]
                shorter_frame = warning["frame_start"] if lengths[0] <= lengths[1] else warning["frame_end"]
                endpoints = []
                for suffix in ("start", "end"):
                    frame = warning["frame_" + suffix]
                    index = frame_rows[frame]
                    if (warning["pts_" + suffix] != int(raw["pts"][index])
                            or warning["seconds_" + suffix] != float(raw["timestamps"][index])
                            or warning["time_base"] != raw["time_base"].tolist()):
                        raise ValueError("Prior warning source PTS differs")
                    length = float(np.linalg.norm(points[index, lookup[a]] - points[index, lookup[b]]))
                    if not np.isclose(length, warning["raw_length_" + suffix + "_m"], rtol=1e-7, atol=1e-10):
                        raise ValueError("Prior raw geometric values changed")
                    for name in (a, b):
                        column = lookup[name]
                        if not good[index, column] or not finite_uncertainty[index, column]:
                            raise ValueError("Prior warning endpoint is not model-screen supported")
                        values = profile(uncertainty[index, column], controls[name])
                        values.update(point=name, frame_index=int(frame), pts=int(raw["pts"][index]),
                                      source_seconds=float(raw["timestamps"][index]), shorter_normalized_bone_frame=frame == shorter_frame)
                        endpoints.append(values)
                        all_profiles[(sid, frame, name)] = {"id": sid, **values}
                        if frame == shorter_frame:
                            shorter_profiles[(sid, frame, name)] = {"id": sid, **values}
                comparisons.append({**warning, "shorter_normalized_bone_frame": shorter_frame,
                    "shorter_frame_is_not_ground_truth_error_label": True, "endpoint_uncertainty_profiles": endpoints})
            report = {"schema": "nlf-developer-uncertainty-geometry-diagnostic-v1", "id": sid,
                "source_role": "development", "source_manifest_sha256": manifest_sha, "raw_sha256": source["raw_sha256"],
                "source_video_sha256": row["video_sha256"], "model_sha256": model_sha, "mapping_sha256": mapping_sha,
                "source_frames": len(raw["timestamps"]), "uncertainty_description": source["uncertainty_units"],
                "model_screen_support_is_not_rgb_observability": True,
                "controls": "Same source and native joint, finite/in-frame predictions, excluding all prior joint warning endpoints. Other predictions may also be wrong.",
                "point_profiles": point_profiles, "warning_profiles": comparisons,
                "warning_labels_are_not_ground_truth": True, "uncertainty_detection_accuracy_measured": False,
                "uncertainty_threshold_selected": False, "inference_or_event_thresholds_changed": False,
                "raw_or_fact_values_modified": False, "quality_gate_passed": False,
                "implementation_sha256": sha256(Path(__file__))}
            payload = json_bytes(report)
            atomic_write(out / (sid + ".json"), payload)
            summaries.append({"id": sid, "status": "diagnostic_complete", "source_frames": len(raw["timestamps"]),
                              "bone_warnings_reused": len(case_warnings), "report_sha256": hashlib.sha256(payload).hexdigest()})
        except (OSError, ValueError, KeyError, TypeError) as error:
            summaries.append({"id": sid, "status": "diagnostic_failed", "reason": str(error), "error_type": type(error).__name__})
    selected = list(shorter_profiles.values())
    ranked = sorted((value for value in selected if value["midrank_fraction"] is not None),
                    key=lambda value: (value["midrank_fraction"], value["id"], value["frame_index"], value["point"]))
    terminal = {"schema": "nlf-developer-uncertainty-geometry-terminal-v1", "expected": 20, "records": summaries,
        "complete": all(value["status"] == "diagnostic_complete" for value in summaries),
        "source_frames": sum(value.get("source_frames", 0) for value in summaries),
        "prior_bone_warning_count": len(warnings), "unique_warning_endpoint_profiles": len(all_profiles),
        "unique_shorter_bone_frame_endpoint_profiles": len(selected),
        "shorter_endpoint_profiles_not_above_same_joint_control_median": sum(value["midrank_fraction"] <= .5 for value in ranked),
        "shorter_endpoint_profiles_rank_at_least_0_90": sum(value["midrank_fraction"] >= .9 for value in ranked),
        "rank_values_are_descriptive_not_admission_thresholds": True,
        "lowest_rank_shorter_endpoint_profiles": ranked[:12],
        "warning_vs_nonwarning_is_not_error_vs_correct": True,
        "independent_error_or_visibility_labels_available": False, "uncertainty_threshold_selected": False,
        "source_selection_or_model_or_frozen_acceptance_changed": False, "quality_gate_passed": False,
        "development_manifest_sha256": sha256(development_path), "raw_registry_sha256": sha256(registry_path),
        "prior_diagnostic_sha256": sha256(diagnostic_path), "implementation_sha256": sha256(Path(__file__))}
    payload = json_bytes(terminal)
    atomic_write(out / "finished.receipt.json", json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "finished.json", payload)
    print(json.dumps({key: terminal[key] for key in ("expected", "complete", "source_frames", "prior_bone_warning_count",
        "unique_shorter_bone_frame_endpoint_profiles", "shorter_endpoint_profiles_not_above_same_joint_control_median",
        "shorter_endpoint_profiles_rank_at_least_0_90", "uncertainty_threshold_selected", "quality_gate_passed")}), flush=True)


if __name__ == "__main__":
    main()
