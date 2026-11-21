"""Independent body-rule candidates from direct RGB-predicted 3D tracks."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .events import _excursions, _runs
from .landmark_events import _inputs, _measured_angle, _unit, _MIN_LENGTH_M

__all__ = ["measure_body_signals", "extract_body_events", "render_body_caption"]
SOURCE = "nlf_direct_rgb_body_rules"
_FRAME = "camera_xright_ydown_zforward"
_CONTINUITY = "independent_required_point_segments"
_NORMALIZATION = "diag(1,-1,-1): camera X-right, Y-up, Z-backward; proper rotation"


@dataclass(frozen=True)
class _Signal:
    values: np.ndarray
    good: np.ndarray
    indices: tuple[int, ...]
    body_part: str
    name: str
    operator: str
    unit: str
    reference: str
    positive: tuple[str, str]
    negative: tuple[str, str]
    rigid_invariant: bool = True
    circular: bool = False


def _positive_scalar(value, name):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all() or raw <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return float(raw)


def _time_base(value):
    raw = np.asarray(value)
    try:
        if raw.shape == () and raw.dtype.kind in "U":
            base = Fraction(raw.item())
        elif raw.shape == () and raw.dtype.kind in "iuf" and np.isfinite(raw).all():
            base = Fraction(str(raw.item()))
        elif raw.shape == (2,) and raw.dtype.kind in "iu":
            base = Fraction(int(raw[0]), int(raw[1]))
        else:
            raise ValueError("unsupported representation")
        if base <= 0 or not np.isfinite(float(base)) or float(base) <= 0:
            raise ValueError("nonpositive time base")
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("time_base must be positive seconds per PTS tick (scalar or rational pair)") from error
    return base


def _source_inputs(arrays, time, point_count):
    """Validate native timing up to one explicit constant PTS offset."""
    frame_keys = ("raw_frame_indices", "raw_pts", "time_base")
    has_frames = any(key in arrays for key in frame_keys)
    if has_frames and not all(key in arrays for key in ("raw_time",) + frame_keys):
        raise ValueError("raw_time, raw_frame_indices, raw_pts and time_base must be supplied together")
    if "raw_quality_valid" in arrays and "raw_time" not in arrays:
        raise ValueError("raw_quality_valid requires raw_time")
    if "raw_time" not in arrays:
        return None
    raw_time = np.asarray(arrays["raw_time"])
    if (raw_time.ndim != 1 or raw_time.dtype.kind not in "iuf"
            or not np.isfinite(raw_time).all() or np.any(raw_time[1:] <= raw_time[:-1])):
        raise ValueError("raw_time must be finite, real and strictly increasing with shape [R]")
    raw_time = raw_time.astype(float)
    if len(time) and (not len(raw_time) or time[0] < raw_time[0] - 1e-9 or time[-1] > raw_time[-1] + 1e-9):
        raise ValueError("time must be within raw_time support; source-frame extrapolation is not allowed")
    quality = None
    if "raw_quality_valid" in arrays:
        quality = np.asarray(arrays["raw_quality_valid"])
        if (quality.shape != (len(raw_time), point_count) or quality.dtype.kind not in "biuf"
                or not np.isfinite(quality).all() or not ((quality == 0) | (quality == 1)).all()):
            raise ValueError("raw_quality_valid must contain finite boolean/0/1 values with shape [R,K]")
        quality = quality.astype(bool)
    if not has_frames:
        return None
    native = {}
    for key in ("raw_frame_indices", "raw_pts"):
        raw = np.asarray(arrays[key])
        if (raw.shape != raw_time.shape or raw.dtype.kind not in "iu"
                or np.any(raw[1:] <= raw[:-1])
                or (key == "raw_frame_indices" and np.any(raw < 0))):
            raise ValueError(f"{key} must contain strictly increasing native integers with shape [R]")
        native[key] = raw
    base = _time_base(arrays["time_base"])
    with np.errstate(over="ignore", invalid="ignore"):
        pts_seconds = native["raw_pts"].astype(float) * float(base)
        offset = float(pts_seconds[0] - raw_time[0]) if len(raw_time) else 0.
        agrees = np.isfinite(pts_seconds).all() and np.isfinite(offset) and np.allclose(
            pts_seconds - offset, raw_time, atol=1e-6, rtol=0)
    if not agrees:
        raise ValueError("raw_pts/time_base must reproduce raw_time up to one constant PTS time offset")
    return {**native, "raw_time": raw_time, "quality": quality,
            "time_base": str(base), "pts_time_offset_seconds": offset}


def _body_inputs(arrays):
    # _inputs supplies independent masks and a proper rotation, not a root frame.
    # Reject silent string/complex/bool coercion before its numeric conversions.
    for key in ("time", "points_m"):
        if key in arrays and np.asarray(arrays[key]).dtype.kind not in "iuf":
            raise ValueError(f"{key} must contain real numeric values")
    data = _inputs(arrays)
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.isfinite(np.diff(data["time"])).all():
            raise ValueError("time intervals must be finite")
    data["source_frames"] = _source_inputs(arrays, data["time"], len(data["names"]))
    return data


def _local_axes(up_vector, lateral_vector):
    up, up_good = _unit(up_vector)
    lateral, lateral_good = _unit(lateral_vector)
    projected = lateral - np.sum(lateral * up, axis=-1)[:, None] * up
    right, right_good = _unit(projected)
    normal = np.cross(right, up)
    good = up_good & lateral_good & right_good & (np.linalg.norm(projected, axis=-1) >= .1)
    return (right, up, normal), good


def _signals(data):
    p, lookup = data["points"], data["lookup"]

    def ids(names):
        return tuple(lookup[name] for name in names) if all(name in lookup for name in names) else None

    for side in ("left", "right"):
        for part, endpoints in (("elbow", ("shoulder", "elbow", "wrist")),
                                ("knee", ("hip", "knee", "ankle"))):
            indices = ids([f"{side}_{name}" for name in endpoints])
            if indices is None:
                continue
            a, pivot, b = indices
            angle, good = _measured_angle(p[:, a] - p[:, pivot], p[:, b] - p[:, pivot])
            yield _Signal(
                angle, good, indices, f"{side}_{part}", "joint_interior_angle",
                f"angle({side}_{endpoints[0]}-{side}_{part}, {side}_{endpoints[2]}-{side}_{part})",
                "degrees", "independent_point_relative_3d",
                ("extends", "increasing_joint_angle"), ("bends", "decreasing_joint_angle"),
            )

        # neck-spine3 is the only allowed torso-up axis; no pelvis/hip fallback.
        indices = ids([f"{side}_shoulder", f"{side}_elbow", "neck", "spine3"])
        if indices is not None:
            shoulder, elbow, neck, spine3 = indices
            up = p[:, neck] - p[:, spine3]
            angle, good = _measured_angle(p[:, elbow] - p[:, shoulder], -up)
            yield _Signal(
                angle, good, indices, side + "_upper_arm", "upper_arm_elevation_relative_to_torso",
                f"angle({side}_elbow-{side}_shoulder, -(neck-spine3))",
                "degrees", "neck_spine3_torso_relative_3d",
                ("raises", "increasing_relative_elevation"), ("lowers", "decreasing_relative_elevation"),
            )

        for distal, proximal in (("wrist", "shoulder"), ("ankle", "hip")):
            indices = ids([f"{side}_{distal}", f"{side}_{proximal}"])
            if indices is not None:
                vector = p[:, indices[0]] - p[:, indices[1]]
                _, good = _unit(vector)
                distance = np.linalg.norm(vector, axis=-1)
                yield _Signal(
                    distance, good, indices, f"{side}_{distal}", f"{distal}_to_{proximal}_distance",
                    f"norm({side}_{distal}-{side}_{proximal})", "m", "independent_point_relative_3d",
                    ("increases_relative_distance", f"farther_from_{proximal}"),
                    ("decreases_relative_distance", f"closer_to_{proximal}"),
                )

        for distal, proximal, belt in (("wrist", "shoulder", "shoulder"), ("ankle", "hip", "hip")):
            required = [f"{side}_{distal}", f"{side}_{proximal}", f"left_{belt}", f"right_{belt}", "neck", "spine3"]
            required = list(dict.fromkeys(required))
            indices = ids(required)
            if indices is None:
                continue
            axes, good = _local_axes(p[:, lookup["neck"]] - p[:, lookup["spine3"]],
                                      p[:, lookup[f"right_{belt}"]] - p[:, lookup[f"left_{belt}"]])
            displacement = p[:, lookup[f"{side}_{distal}"]] - p[:, lookup[f"{side}_{proximal}"]]
            for axis, label, positive, negative in zip(axes,
                    (belt + "_axis", "torso_axis", belt + "_torso_plane_normal"),
                    ("positive_anatomical_right_axis", "positive_torso_up_axis", "positive_ordered_plane_normal"),
                    ("negative_anatomical_right_axis", "negative_torso_up_axis", "negative_ordered_plane_normal")):
                yield _Signal(
                    np.sum(displacement * axis, axis=-1), good, indices, f"{side}_{distal}",
                    f"{distal}_position_along_{label}",
                    f"dot({side}_{distal}-{side}_{proximal}, {label} from right_{belt}-left_{belt} and neck-spine3)",
                    "m", f"{belt}_neck_spine3_local_frame",
                    ("changes_local_position", positive), ("changes_local_position", negative))

        indices = ids([f"{side}_ankle", f"{side}_foot", f"{side}_knee"])
        if indices is not None:
            ankle, foot, knee = indices
            angle, good = _measured_angle(p[:, foot] - p[:, ankle], p[:, knee] - p[:, ankle])
            yield _Signal(
                angle, good, indices, side + "_foot", "foot_long_axis_to_shin_angle",
                f"angle({side}_foot-{side}_ankle, {side}_knee-{side}_ankle)",
                "degrees", "ankle_foot_axis_to_shin_relative_3d",
                ("changes_relative_orientation", "increasing_relative_angle"),
                ("changes_relative_orientation", "decreasing_relative_angle"),
            )

    pelvis = lookup.get("pelvis")
    if pelvis is not None:
        for axis, label in enumerate(("x", "y_up", "z_backward")):
            yield _Signal(p[:, pelvis, axis], np.ones(len(p), dtype=bool), (pelvis,), "root",
                          "pelvis_camera_" + label, "estimated pelvis coordinate in fixed normalized camera axes",
                          "m", "fixed_camera_y_up_z_backward_m",
                          ("changes_camera_coordinate_position", "positive_camera_" + label),
                          ("changes_camera_coordinate_position", "negative_camera_" + label), rigid_invariant=False)

    indices = ids(["left_shoulder", "right_shoulder", "neck", "spine3"])
    if indices is not None:
        left, right, neck, spine3 = indices
        axes, good = _local_axes(p[:, neck] - p[:, spine3], p[:, right] - p[:, left])
        normal = axes[2]
        good &= np.hypot(normal[:, 0], normal[:, 2]) >= .1
        yaw = np.degrees(np.arctan2(normal[:, 0], normal[:, 2]))
        yield _Signal(yaw, good, indices, "torso", "shoulder_plane_camera_yaw",
                      "atan2(ordered_shoulder_torso_normal.camera_X, ordered_shoulder_torso_normal.camera_Z_backward)",
                      "degrees", "fixed_camera_y_up_projected_shoulder_plane",
                      ("changes_camera_coordinate_orientation", "positive_camera_yaw"),
                      ("changes_camera_coordinate_orientation", "negative_camera_yaw"),
                      rigid_invariant=False, circular=True)

    indices = ids(["neck", "pelvis"])
    if indices is not None:
        # _inputs rotates camera (0,-1,0) to (0,1,0). This is not gravity.
        angle, good = _measured_angle(p[:, indices[0]] - p[:, indices[1]],
                                      np.broadcast_to([0., 1., 0.], (len(p), 3)))
        yield _Signal(
            angle, good, indices, "torso", "torso_camera_up_angle",
            "angle(neck-pelvis, camera_Y_up=(0,-1,0))", "degrees", "fixed_camera_y_up_3d",
            ("changes_camera_coordinate_orientation", "away_from_camera_y_up"),
            ("changes_camera_coordinate_orientation", "toward_camera_y_up"), rigid_invariant=False,
        )


def _good(signal, data):
    return signal.good & data["available"][:, signal.indices].all(axis=1) & np.isfinite(signal.values)


def _description(signal, data):
    return {
        "name": signal.body_part + "." + signal.name,
        "measurement_name": signal.name, "body_part": signal.body_part,
        "operator": signal.operator, "coordinate_frame": _FRAME,
        "reference_frame": signal.reference, "normalization": _NORMALIZATION,
        "required_joints": list(signal.indices),
        "required_point_names": [data["names"][i] for i in signal.indices],
        "unit": signal.unit, "speed_unit": signal.unit + "/s", "circular": signal.circular,
        "value_representation": "raw_camera_yaw_branch_degrees" if signal.circular else "operator_value",
        "plane_normal_is_verified_body_front": False, "camera_axes_are_world_or_gravity": False,
        "common_so3_translation_invariant": signal.rigid_invariant,
        "continuity_scope": _CONTINUITY, "calibrated": False,
    }


def measure_body_signals(arrays: dict) -> tuple[dict, dict]:
    """Retain all signal trajectories and independent required-point segments."""
    data = _body_inputs(arrays)
    columns, validity, names, units, descriptions = [], [], [], [], []
    for signal in _signals(data):
        good = _good(signal, data)
        description = _description(signal, data)
        description["required_joint_segment_ids"] = [
            [int(value) for value in row] for row in data["point_segment_id"][:, signal.indices]]
        names.append(description["name"])
        units.append(signal.unit)
        columns.append(np.where(good, signal.values, np.nan))
        validity.append(good)
        descriptions.append(description)
    n = len(data["time"])
    signal_arrays = {
        "time": data["time"].copy(),
        "body_signal_values": np.column_stack(columns) if columns else np.empty((n, 0), float),
        "body_signal_valid": np.column_stack(validity) if validity else np.empty((n, 0), bool),
        "body_signal_names": np.asarray(names, dtype="U"),
        "body_signal_units": np.asarray(units, dtype="U"),
    }
    metadata = {
        "source": SOURCE, "signals": descriptions, "units": sorted(set(units)),
        "input_units": "m", "input_coordinate_frame": _FRAME,
        "continuity_scope": _CONTINUITY, "invalid_value": "NaN in numeric arrays only",
        "semantic_status": "measurement_derived_candidate", "semantic_approval": False,
        "calibrated": False, "quality_is_probability": False,
        "source_frame_evidence_available": data["source_frames"] is not None,
        "minimum_vector_length_m": _MIN_LENGTH_M,
        "notice": "Uncalibrated direct RGB point geometry; local distance is between named points. Torso camera-up angle describes camera-coordinate orientation only.",
    }
    return signal_arrays, metadata


def _bracket(times, value):
    right = int(np.searchsorted(times, value))
    for index in (right, right - 1):
        if 0 <= index < len(times) and abs(float(times[index]) - value) <= 1e-9:
            return index, index
    return max(0, right - 1), min(len(times) - 1, right)


def _frame_record(source, index):
    return {"raw_index": int(index), "frame_index": int(source["raw_frame_indices"][index]),
            "pts": int(source["raw_pts"][index]), "raw_time": float(source["raw_time"][index])}


def _frame_evidence(source, start_time, end_time, indices, names):
    """Retain native quality-support brackets, separately from smoothing weights."""
    start_bracket = _bracket(source["raw_time"], start_time)
    end_bracket = _bracket(source["raw_time"], end_time)
    lo, hi = start_bracket[0], end_bracket[1]
    endpoint_support = {}
    if source["quality"] is not None:
        for index in indices:
            usable = np.flatnonzero(source["quality"][:, index])
            pairs = []
            for value in (start_time, end_time):
                if not len(usable):
                    pairs.append(None)
                    continue
                # None on an unsupported side rather than extrapolated evidence.
                a, b = _bracket(source["raw_time"][usable], value)
                pair = [int(usable[a]) if source["raw_time"][usable[a]] <= value + 1e-9 else None,
                        int(usable[b]) if source["raw_time"][usable[b]] >= value - 1e-9 else None]
                pairs.append(pair)
                actual = [item for item in pair if item is not None]
                if actual:
                    lo, hi = min(lo, min(actual)), max(hi, max(actual))
            endpoint_support[index] = pairs
    raw_indices = list(range(lo, hi + 1))
    result = {
        "scope": "native_source_brackets_and_required_point_quality_support_not_smoothing_weights",
        "range_inclusive": True, "time_base": source["time_base"],
        "pts_time_offset_seconds": source["pts_time_offset_seconds"],
        "time_definition": "raw_time = raw_pts * time_base - pts_time_offset_seconds",
        "start_bracket": [_frame_record(source, i) for i in start_bracket],
        "end_bracket": [_frame_record(source, i) for i in end_bracket],
        "range_start": _frame_record(source, lo), "range_end": _frame_record(source, hi),
        "raw_indices": raw_indices,
        "frame_indices": [int(source["raw_frame_indices"][i]) for i in raw_indices],
        "pts": [int(source["raw_pts"][i]) for i in raw_indices],
        "raw_times": [float(source["raw_time"][i]) for i in raw_indices],
        "raw_quality_valid_available": source["quality"] is not None,
        "required_point_support": [],
    }
    for index in indices:
        support = {"name": names[index], "index": int(index), "raw_quality_valid_support_indices": None}
        if source["quality"] is not None:
            kept = [i for i in raw_indices if source["quality"][i, index]]
            support.update({
                "raw_quality_valid_support_indices": kept,
                "frame_indices": [int(source["raw_frame_indices"][i]) for i in kept],
                "pts": [int(source["raw_pts"][i]) for i in kept],
                "start_support_bracket": endpoint_support[index][0],
                "end_support_bracket": endpoint_support[index][1],
            })
        result["required_point_support"].append(support)
    return result


def _measurement(values, time, unit):
    duration = float(time[-1] - time[0])
    delta = float(values[-1] - values[0])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        steps = np.abs(np.diff(values))
        speed = steps / np.diff(time)
        peak = int(np.argmax(speed))
        result = {
            "start": float(values[0]), "end": float(values[-1]), "delta": delta,
            "unit": unit, "duration_seconds": duration, "path_change": float(steps.sum()),
            "net_speed_per_s": abs(delta) / duration,
            "mean_speed_per_s": float(steps.sum() / duration),
            "peak_speed_per_s": float(speed[peak]),
            "peak_speed_time": float(time[peak] + (time[peak + 1] - time[peak]) / 2),
            "speed_unit": unit + "/s",
        }
    return result if all(np.isfinite(value) for value in result.values() if isinstance(value, float)) else None


def extract_body_events(arrays: dict, *, min_duration_seconds=.15,
                        min_angle_change_degrees=20., min_distance_change_m=.12) -> list[dict]:
    """Extract supported candidates; sampled speed magnitudes do not imply force."""
    duration_min = _positive_scalar(min_duration_seconds, "min_duration_seconds")
    angle_min = _positive_scalar(min_angle_change_degrees, "min_angle_change_degrees")
    distance_min = _positive_scalar(min_distance_change_m, "min_distance_change_m")
    data = _body_inputs(arrays)
    time, events = data["time"], []
    for signal in _signals(data):
        threshold = distance_min if signal.unit == "m" else angle_min
        segments = data["point_segment_id"][:, signal.indices]
        for first, last in _runs(_good(signal, data), time, joint_segment_id=segments):
            values = signal.values[first:last].copy()
            if signal.circular:
                values = np.degrees(np.unwrap(np.radians(values)))
            for a, b in _excursions(values, threshold * .2):
                start, end = first + a, first + b
                delta = float(values[b] - values[a])
                if time[end] - time[start] + 1e-12 < duration_min or abs(delta) + 1e-9 < threshold:
                    continue
                part = slice(start, end + 1)
                measurement = _measurement(values[a:b + 1], time[part], signal.unit)
                if measurement is None:
                    continue
                observed = data["point_observed"][part][:, signal.indices]
                interpolated = data["point_interpolated"][part][:, signal.indices]
                evidence = _description(signal, data)
                evidence.update({
                    "required_joint_segment_ids": [int(value) for value in segments[start]],
                    "point_quality": [
                        {"name": data["names"][index], "index": index,
                         "observed_fraction": float(observed[:, column].mean()),
                         "interpolated_fraction": float(interpolated[:, column].mean()),
                         "segment_id": int(segments[start, column])}
                        for column, index in enumerate(signal.indices)],
                    "observed_fraction": float(observed.all(axis=1).mean()),
                    "interpolated_fraction": float(interpolated.any(axis=1).mean()),
                    "coverage_scope": "required_point_timestamp_support", "quality_is_probability": False,
                    "observed_fraction_definition": "fraction of output samples with all required points observed",
                    "interpolated_fraction_definition": "fraction of output samples with any required point interpolated",
                    "anatomical_side_source": "supplied point names, not camera X sign",
                    "sample_start": start, "sample_end": end, "sample_count": end - start + 1,
                    "max_sample_gap_seconds": float(np.diff(time[part]).max()),
                    "minimum_change": threshold, "minimum_duration_seconds": duration_min,
                    "minimum_vector_length_m": _MIN_LENGTH_M, "reversal_deadband": threshold * .2,
                    "boundary_method": "sampled_extrema_with_reversal_hysteresis",
                    "net_speed_definition": "abs(delta)/duration_seconds",
                    "mean_speed_definition": "sum(abs(diff(values)))/duration_seconds",
                    "peak_speed_definition": "max(abs(diff(values))/diff(time)); first exact tie",
                    "peak_speed_time_definition": "midpoint of the peak sampled interval",
                })
                if data["source_frames"] is not None:
                    evidence["source_frames"] = _frame_evidence(
                        data["source_frames"], float(time[start]), float(time[end]), signal.indices, data["names"])
                action, direction = signal.positive if delta > 0 else signal.negative
                events.append({
                    "id": "", "start": float(time[start]), "end": float(time[end]),
                    "duration_seconds": measurement["duration_seconds"],
                    "body_part": signal.body_part, "action": action, "direction": direction,
                    "measurement": measurement, "evidence": evidence, "source": SOURCE,
                    "semantic_status": "measurement_derived_candidate", "semantic_approval": False,
                })
    events.sort(key=lambda e: (e["start"], e["end"], e["body_part"], e["evidence"]["measurement_name"], e["direction"]))
    for index, event in enumerate(events, 1):
        event["id"] = f"body_event_{index:04d}"
    return events


def render_body_caption(events: list) -> list[str]:
    """One attributed candidate sentence per event, preserving operator scope."""
    captions = []
    for event in sorted(events, key=lambda e: (e["start"], e["end"], e["id"])):
        name = event["evidence"]["measurement_name"]
        part = event["body_part"].replace("_", " ").capitalize()
        delta = event["measurement"]["delta"]
        amount, change = abs(delta), "increases" if delta > 0 else "decreases"
        if name == "joint_interior_angle":
            action = "extends" if delta > 0 else "bends"
            text = f"{part} {action}: its 3D interior angle {change} by {amount:.1f} degrees."
        elif name == "upper_arm_elevation_relative_to_torso":
            text = f"{part} elevation relative to the neck-spine3 torso axis {change} by {amount:.1f} degrees."
        elif name in ("wrist_to_shoulder_distance", "ankle_to_hip_distance"):
            side = event["body_part"].split("_", 1)[0].capitalize()
            noun = name.replace("_", " ")
            text = f"{side} {noun} {change} by {amount:.3f} m (distance between these points)."
        elif "_position_along_" in name:
            axis = name.split("_position_along_", 1)[1].replace("_", " ")
            text = f"{part} position relative to its proximal joint along the local {axis} {change} by {amount:.3f} m."
        elif name.startswith("pelvis_camera_"):
            axis = name.removeprefix("pelvis_camera_").replace("_", " ")
            text = f"Estimated pelvis camera-{axis} coordinate {change} by {amount:.3f} m (not world translation or ground height)."
        elif name == "shoulder_plane_camera_yaw":
            text = f"Shoulder-torso plane normal camera yaw {change} by {amount:.1f} degrees (not a verified world/body-front direction)."
        elif name == "foot_long_axis_to_shin_angle":
            text = f"{part} ankle-to-foot long-axis angle relative to the shin {change} by {amount:.1f} degrees."
        elif name == "torso_camera_up_angle":
            text = f"Torso angle to camera Y-up {change} by {amount:.1f} degrees (camera-coordinate orientation change)."
        else:
            raise ValueError(f"Unsupported body measurement: {name}")
        captions.append(f"[{event['id']} | {event['start']:.2f}-{event['end']:.2f} s | {event['source']}] Candidate: {text}")
    return captions
