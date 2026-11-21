"""Exact developer-only decomposition of the frozen wrist/torso scalar."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementation/scripts_omnidance/exp/ms"))
from motionscript_v2.body_events import _body_inputs, _good, _signals
from motionscript_v2.events import _runs
from motionscript_v2.landmark_events import _unit
from ms_body_review import load_facts
from ms_landmarks import load_bundle
from ms_optimize import atomic_write, json_bytes, save_npz, sha256, source_hashes

NAME = "wrist_position_along_torso_axis"


def decompose(relative, axis):
    scalar = np.sum(relative * axis, axis=-1)
    vector_term = np.sum(np.diff(relative, axis=0) * (axis[1:] + axis[:-1]) * .5, axis=-1)
    axis_term = np.sum((relative[1:] + relative[:-1]) * .5 * np.diff(axis, axis=0), axis=-1)
    return scalar, np.diff(scalar), vector_term, axis_term


def supported_steps(good, time, segments):
    result = np.zeros(max(0, len(time) - 1), dtype=bool)
    for first, last in _runs(good, time, joint_segment_id=segments):
        if last - first > 1:
            result[first:last - 1] = True
    return result


def contribution_summary(delta, vector_term, axis_term):
    vector_absolute = float(np.abs(vector_term).sum())
    axis_absolute = float(np.abs(axis_term).sum())
    total = vector_absolute + axis_absolute
    scalar_path = float(np.abs(delta).sum())
    return {"net_scalar_delta_m": float(delta.sum()),
            "net_relative_vector_term_m": float(vector_term.sum()),
            "net_reference_axis_term_m": float(axis_term.sum()),
            "absolute_relative_vector_terms_m": vector_absolute,
            "absolute_reference_axis_terms_m": axis_absolute,
            "scalar_path_change_m": scalar_path,
            "axis_absolute_term_fraction": axis_absolute / total if total else None,
            "term_cancellation_fraction": max(0., 1 - scalar_path / total) if total else None}


def main():
    selection_path = ROOT / "panel-200-v1/development.jsonl"
    registry_path = ROOT / "dev-panel-v1-postprocess/finished.json"
    protocol_path = ROOT / "panel-200-v1/measurement_protocol.v1.json"
    out = ROOT / "development-torso-axis-decomposition-v1"
    if out.exists():
        raise ValueError("Preserve previous decomposition revisions")
    rows = {row["id"]: row for row in map(json.loads, selection_path.read_text().splitlines())}
    registry = json.loads(registry_path.read_text())
    records = {row["id"]: row for row in registry["records"]}
    protocol = json.loads(protocol_path.read_text())
    if len(rows) != 20 or len(records) != 20 or set(rows) != set(records) or registry["expected"] != 20:
        raise ValueError("Exactly20 frozen developer sources required")
    if source_hashes() != protocol["measurement_code_sha256"]:
        raise ValueError("Frozen measurement implementation changed")
    out.mkdir()
    cases, event_records, channel_records = [], [], []
    for sid, row in rows.items():
        try:
            if row["panel_role"] != "development" or row["split"] != "train":
                raise ValueError("No acceptance source may enter developer analysis")
            record = records[sid]
            raw, source, manifest_sha = load_bundle(Path(record["bundle"]))
            facts, arrays, facts_sha = load_facts(Path(record["facts_dir"]))
            if (source["id"] != sid or source["source_sha256"]["video"] != row["video_sha256"]
                    or sha256(Path(row["video_path"])) != row["video_sha256"]
                    or facts_sha != record["facts_sha256"] or facts["source_manifest_sha256"] != manifest_sha
                    or facts["measurement_code_sha256"] != protocol["measurement_code_sha256"]
                    or facts["parameter_config"] != protocol["config"]):
                raise ValueError("Source/fact/code/config binding changed")
            tracks = arrays["tracks"]
            for a, b in (("raw_frame_indices", "frame_indices"), ("raw_pts", "pts"), ("raw_time", "timestamps"), ("time_base", "time_base")):
                if not np.array_equal(tracks[a], raw[b]):
                    raise ValueError("Processed/source frame and PTS binding changed")
            data = _body_inputs(tracks)
            p, time, lookup = data["points"], data["time"], data["lookup"]
            up_vector = p[:, lookup["neck"]] - p[:, lookup["spine3"]]
            axis, axis_good = _unit(up_vector)
            length = np.linalg.norm(up_vector, axis=-1)
            axis_available = axis_good & data["available"][:, [lookup["neck"], lookup["spine3"]]].all(axis=1)
            axis_angles = np.degrees(np.arccos(np.clip(np.sum(axis[1:] * axis[:-1], axis=-1), -1., 1.)))
            exported = {"time": time, "interval_start_time": time[:-1], "interval_end_time": time[1:],
                "neck_spine3_axis": np.where(axis_available[:, None], axis, np.nan),
                "neck_spine3_length_m": np.where(axis_available, length, np.nan),
                "raw_frame_indices": tracks["raw_frame_indices"], "raw_time": tracks["raw_time"],
                "raw_pts": tracks["raw_pts"], "time_base": tracks["time_base"]}
            summaries, events = [], []
            for signal in _signals(data):
                if signal.name != NAME:
                    continue
                side = signal.body_part.split("_")[0]
                relative = p[:, lookup[side + "_wrist"]] - p[:, lookup[side + "_shoulder"]]
                scalar, delta, vector_term, axis_term = decompose(relative, axis)
                good = _good(signal, data)
                segments = data["point_segment_id"][:, signal.indices]
                eligible = supported_steps(good, time, segments)
                column = arrays["signals"]["body_signal_names"].tolist().index(signal.body_part + "." + NAME)
                cached = arrays["signals"]["body_signal_values"][:, column]
                if not np.array_equal(good, arrays["signals"]["body_signal_valid"][:, column]):
                    raise ValueError("Frozen signal support differs")
                error = float(np.max(np.abs(scalar[good] - cached[good]))) if good.any() else None
                decomposition_error = float(np.max(np.abs(delta[eligible] - vector_term[eligible] - axis_term[eligible]))) if eligible.any() else None
                if ((error is not None and error > 1e-9)
                        or (decomposition_error is not None and decomposition_error > 1e-9)):
                    raise ValueError("Exact scalar/decomposition identity differs from the cache")
                minimal_names = [side + "_wrist", side + "_shoulder", "neck", "spine3"]
                minimal_ids = [lookup[name] for name in minimal_names]
                minimal_good = axis_good & data["available"][:, minimal_ids].all(axis=1)
                minimal_steps = supported_steps(minimal_good, time, data["point_segment_id"][:, minimal_ids])
                opposite = "right_shoulder" if side == "left" else "left_shoulder"
                summary = {"id": sid, "side": side, "signal": signal.body_part + "." + NAME,
                    "required_point_names": [data["names"][i] for i in signal.indices],
                    "literal_up_scalar_dependencies": minimal_names,
                    "valid_rows": int(good.sum()), "valid_steps": int(eligible.sum()),
                    "algebraic_minimal_supported_but_full_basis_rejected_rows": int((minimal_good & ~good).sum()),
                    "algebraic_minimal_supported_but_full_basis_rejected_steps": int((minimal_steps & ~eligible).sum()),
                    "extra_rows_with_opposite_shoulder_unavailable": int((minimal_good & ~good & ~data["available"][:, lookup[opposite]]).sum()),
                    "minimal_formula_is_not_admitted_visibility_or_quality": True,
                    "scalar_cache_max_error_m": error, "decomposition_max_error_m": decomposition_error,
                    "axis_length_min_m": float(length[good].min()) if good.any() else None,
                    "axis_length_median_m": float(np.median(length[good])) if good.any() else None,
                    "axis_step_angle_p95_degrees": float(np.quantile(axis_angles[eligible], .95)) if eligible.any() else None,
                    "axis_step_angle_max_degrees": float(axis_angles[eligible].max()) if eligible.any() else None,
                    **contribution_summary(delta[eligible], vector_term[eligible], axis_term[eligible])}
                summaries.append(summary)
                channel_records.append(summary)
                exported.update({side + "_scalar_m": np.where(good, scalar, np.nan),
                    side + "_valid_rows": good, side + "_valid_steps": eligible,
                    side + "_delta_scalar_m": np.where(eligible, delta, np.nan),
                    side + "_relative_vector_terms_m": np.where(eligible, vector_term, np.nan),
                    side + "_reference_axis_terms_m": np.where(eligible, axis_term, np.nan)})
                for event in facts["events"]:
                    if event["body_part"] != signal.body_part or event["evidence"]["measurement_name"] != NAME:
                        continue
                    a, b = event["evidence"]["sample_start"], event["evidence"]["sample_end"]
                    if not eligible[a:b].all() or abs(time[a] - event["start"]) > 1e-9 or abs(time[b] - event["end"]) > 1e-9:
                        raise ValueError("Event crosses a source support segment or its cached boundary")
                    part = slice(a, b)
                    values = contribution_summary(delta[part], vector_term[part], axis_term[part])
                    if abs(values["net_scalar_delta_m"] - event["measurement"]["delta"]) > 1e-9:
                        raise ValueError("Frozen event delta differs")
                    source_frames = event["evidence"]["source_frames"]
                    for bracket in (source_frames["start_bracket"], source_frames["end_bracket"]):
                        for item in bracket:
                            i = item["raw_index"]
                            if (item["frame_index"] != int(raw["frame_indices"][i]) or item["pts"] != int(raw["pts"][i])
                                    or item["raw_time"] != float(raw["timestamps"][i])):
                                raise ValueError("Frozen event native-frame/PTS bracket differs")
                    item = {"id": sid, "event_id": event["id"], "side": side,
                        "start": event["start"], "end": event["end"], "cached_delta_m": event["measurement"]["delta"],
                        "source_start_bracket": source_frames["start_bracket"],
                        "source_end_bracket": source_frames["end_bracket"], "time_base": source_frames["time_base"],
                        "axis_length_min_m": float(length[a:b + 1].min()),
                        "axis_step_angle_max_degrees": float(axis_angles[part].max()), **values,
                        "reference_axis_change_can_be_true_torso_motion": True, "error_ground_truth_measured": False}
                    events.append(item)
                    event_records.append(item)
            cache_path = out / (sid + ".decomposition.npz")
            save_npz(cache_path, exported)
            report = {"schema": "nlf-developer-torso-axis-decomposition-v1", "id": sid,
                "source_role": "development", "raw_sha256": source["raw_sha256"], "facts_sha256": facts_sha,
                "source_video_sha256": row["video_sha256"], "source_manifest_sha256": manifest_sha,
                "grid_rows": len(time), "source_frames": len(raw["timestamps"]),
                "formula": "Delta(dot(P,U)) = dot(Delta(P),(U1+U0)/2) + dot((P1+P0)/2,Delta(U))",
                "relative_vector_definition": "estimated wrist minus same-side shoulder; includes body/camera/model rotation",
                "reference_axis_definition": "unit(estimated neck-spine3), not world gravity",
                "absolute_term_fraction_is_not_fraction_of_physical_motion_or_error": True,
                "channel_summaries": summaries, "event_decompositions": events,
                "cache_sha256": sha256(cache_path), "implementation_sha256": sha256(Path(__file__)),
                "frozen_scalar_definition_or_thresholds_changed": False,
                "axis_policy_or_dependency_changes_promoted": False, "quality_gate_passed": False}
            payload = json_bytes(report)
            atomic_write(out / (sid + ".json"), payload)
            cases.append({"id": sid, "status": "diagnostic_complete", "event_decompositions": len(events),
                          "report_sha256": hashlib.sha256(payload).hexdigest()})
        except (OSError, ValueError, KeyError, TypeError) as error:
            cases.append({"id": sid, "status": "diagnostic_failed", "error_type": type(error).__name__, "reason": str(error)})
    ranked = sorted((event for event in event_records if event["axis_absolute_term_fraction"] is not None),
                    key=lambda event: (-event["axis_absolute_term_fraction"], event["id"], event["event_id"]))
    terminal = {"schema": "nlf-developer-axis-decomposition-terminal-v1", "expected": 20, "records": cases,
        "complete": all(case["status"] == "diagnostic_complete" for case in cases),
        "wrist_channels_analyzed": len(channel_records), "events_analyzed": len(event_records),
        "extra_algebraically_supported_rows_rejected_by_full_basis": sum(item["algebraic_minimal_supported_but_full_basis_rejected_rows"] for item in channel_records),
        "extra_rows_with_opposite_shoulder_unavailable": sum(item["extra_rows_with_opposite_shoulder_unavailable"] for item in channel_records),
        "events_with_axis_absolute_term_fraction_above_half": sum(event["axis_absolute_term_fraction"] > .5 for event in ranked),
        "half_is_descriptive_not_a_quality_threshold": True, "ranked_reference_axis_sensitive_events": ranked[:12],
        "contribution_fractions_are_not_error_detection_accuracy": True,
        "body_reference_axis_change_is_not_automatically_an_error": True,
        "no_support_gap_stitching_or_source_extrapolation": True,
        "source_selection_or_model_or_events_changed": False, "quality_gate_passed": False,
        "development_manifest_sha256": sha256(selection_path), "registry_sha256": sha256(registry_path),
        "protocol_sha256": sha256(protocol_path), "implementation_sha256": sha256(Path(__file__))}
    payload = json_bytes(terminal)
    atomic_write(out / "finished.receipt.json", json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "finished.json", payload)
    print(json.dumps({key: terminal[key] for key in ("expected", "complete", "wrist_channels_analyzed", "events_analyzed",
        "extra_algebraically_supported_rows_rejected_by_full_basis", "extra_rows_with_opposite_shoulder_unavailable",
        "events_with_axis_absolute_term_fraction_above_half", "quality_gate_passed")}), flush=True)


if __name__ == "__main__":
    main()
