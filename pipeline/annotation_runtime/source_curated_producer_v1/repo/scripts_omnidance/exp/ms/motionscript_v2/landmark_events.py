"""Relative micro-geometry from named, directly image-predicted 3D points.

No SMPL mesh/FK points are generated here. Anatomical sides come exclusively
from supplied names. Operators are invariant to a common proper rigid transform
at every timestamp; none claims isolated wrist rotation, contact, or a dance step.
The first version uses body22 plus two palm anchors per hand and three foot
anchors per foot (32 points total). It does not extract individual finger joints.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import _angle, _excursions, _runs

__all__ = ["measure_landmark_signals", "extract_landmark_events", "render_landmark_caption"]
SOURCE = "nlf_direct_rgb_landmarks"
_MIN_LENGTH_M = 1e-5
_MIN_PLANE_SINE = .10
_MIN_ARM_PLANE_SINE = .15


@dataclass(frozen=True)
class _Signal:
    values: np.ndarray
    good: np.ndarray
    indices: tuple[int, ...]
    body_part: str
    name: str
    operator: str
    frame: str
    circular: bool = False


def _inputs(arrays):
    required = ("time", "points_m", "names", "point_valid", "point_segment_id",
                "point_observed", "point_interpolated", "coordinate_frame", "units")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError("Missing landmark arrays: " + ", ".join(missing))
    for key, expected in (("coordinate_frame", "camera_xright_ydown_zforward"), ("units", "m")):
        value = np.asarray(arrays[key])
        if value.shape != () or value.item() != expected:
            raise ValueError(f"{key} must be the scalar {expected!r}")
    time = np.asarray(arrays["time"], dtype=float)
    points = np.asarray(arrays["points_m"], dtype=float)
    names = np.asarray(arrays["names"])
    if time.ndim != 1 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time must be finite, strictly increasing, and one dimensional")
    if names.ndim != 1 or names.dtype.kind not in "UO":
        raise ValueError("names must be a one-dimensional sequence of unique strings")
    names = names.tolist()
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("names must be nonempty unique strings")
    n, k = len(time), len(names)
    if points.shape != (n, k, 3):
        raise ValueError("points_m must have shape [T,K,3]")
    data = {"time": time, "names": names, "lookup": {name: i for i, name in enumerate(names)}}
    finite = np.isfinite(points).all(axis=-1)
    # A proper 180-degree X rotation, not a reflection: angle/roll handedness is
    # retained while camera Y-down/Z-forward becomes Y-up/Z-backward.
    data["points"] = np.where(np.isfinite(points), points, 0.) * [1., -1., -1.]
    for key in ("point_valid", "point_observed", "point_interpolated"):
        raw = np.asarray(arrays[key])
        if raw.shape != (n, k):
            raise ValueError(f"{key} must have shape [T,K]")
        if (raw.dtype.kind not in "biuf" or not np.isfinite(raw).all()
                or not ((raw == 0) | (raw == 1)).all()):
            raise ValueError(f"{key} must contain finite boolean/0/1 values")
        data[key] = raw.astype(bool)
    segment = np.asarray(arrays["point_segment_id"])
    if segment.shape != (n, k):
        raise ValueError("point_segment_id must have shape [T,K]")
    if (segment.dtype.kind not in "iuf" or not np.isfinite(segment).all()
            or np.any(segment < -1) or np.any(segment != np.floor(segment))):
        raise ValueError("point_segment_id must contain finite integer IDs >= -1")
    data["point_segment_id"] = segment
    data["available"] = (finite & data["point_valid"] & (segment >= 0)
                         & (data["point_observed"] | data["point_interpolated"]))
    return data


def _unit(vector):
    norm = np.linalg.norm(vector, axis=-1)
    good = np.isfinite(vector).all(-1) & (norm > _MIN_LENGTH_M)
    return vector / np.maximum(norm[..., None], _MIN_LENGTH_M), good


def _plane(u, v, min_sine=_MIN_PLANE_SINE):
    a, ag = _unit(u)
    b, bg = _unit(v)
    cross = np.cross(a, b)
    normal, good = _unit(cross)
    return normal, good & ag & bg & (np.linalg.norm(cross, axis=-1) >= min_sine)


def _measured_angle(u, v):
    angle, good = _angle(u, v)
    return angle, good & (np.linalg.norm(u, axis=-1) > _MIN_LENGTH_M) & (np.linalg.norm(v, axis=-1) > _MIN_LENGTH_M)


def _signals(data):
    p, lookup = data["points"], data["lookup"]

    def ids(names):
        return tuple(lookup[name] for name in names) if all(name in lookup for name in names) else None

    for side in ("left", "right"):
        palm = ids([f"{side}_{name}" for name in ("wrist", "index_mcp", "little_mcp", "elbow")])
        if palm is not None:
            wrist, index, little, elbow = palm
            normal, normal_good = _plane(p[:, index] - p[:, wrist], p[:, little] - p[:, wrist])
            forearm = p[:, wrist] - p[:, elbow]
            angle, good = _measured_angle(normal, forearm)
            yield _Signal(
                angle, good & normal_good, palm, side + "_palm", "palm_normal_to_forearm_angle",
                "angle(cross(index_mcp-wrist, little_mcp-wrist), wrist-elbow)", "forearm_relative_angle",
            )
            shoulder = lookup.get(side + "_shoulder")
            if shoulder is not None:
                f, fg = _unit(forearm)
                reference, rg = _plane(p[:, elbow] - p[:, shoulder], forearm, _MIN_ARM_PLANE_SINE)
                projection = normal - np.sum(normal * f, axis=-1)[:, None] * f
                palm_radial, pg = _unit(projection)
                # A straight arm has no stable bend plane. A normal parallel to
                # the forearm has no defined roll around that axis either.
                good = (normal_good & fg & rg & pg
                        & (np.linalg.norm(projection, axis=-1) >= _MIN_ARM_PLANE_SINE))
                roll = np.degrees(np.arctan2(np.sum(f * np.cross(reference, palm_radial), axis=-1),
                                             np.sum(reference * palm_radial, axis=-1)))
                yield _Signal(
                    roll, good, palm + (shoulder,), side + "_palm", "palm_roll_relative_to_arm_plane",
                    "signed_angle_about(wrist-elbow, cross(elbow-shoulder, wrist-elbow), projected_palm_normal)",
                    "upper_arm_forearm_plane_relative_roll", circular=True,
                )

        foot = ids([f"{side}_{name}" for name in ("heel", "big_toe", "little_toe", "ankle", "knee")])
        if foot is not None:
            heel, big, little, ankle, knee = foot
            shin = p[:, knee] - p[:, ankle]
            normal, normal_good = _plane(p[:, big] - p[:, heel], p[:, little] - p[:, heel])
            angle, good = _measured_angle(normal, shin)
            yield _Signal(
                angle, good & normal_good, foot, side + "_foot", "sole_normal_to_shin_angle",
                "angle(cross(big_toe-heel, little_toe-heel), knee-ankle)", "shin_relative_angle",
            )
            longitudinal = .5 * (p[:, big] + p[:, little]) - p[:, heel]
            angle, good = _measured_angle(longitudinal, shin)
            yield _Signal(
                angle, good, foot, side + "_foot", "foot_long_axis_to_shin_angle",
                "angle((big_toe+little_toe)/2-heel, knee-ankle)", "shin_relative_angle",
            )

def measure_landmark_signals(arrays: dict) -> tuple[dict, dict]:
    """Keep complete relative-angle trajectories, independently of event thresholds.

    Invalid values are NaN. Circular signals retain their raw atan2 branch;
    nothing is unwrapped across timestamps or point continuity segments. The
    returned arrays use numeric/Unicode dtypes and load without pickle.
    """
    data = _inputs(arrays)
    columns, validity, names, descriptions = [], [], [], []
    for signal in _signals(data):
        good = (signal.good & data["available"][:, signal.indices].all(axis=1)
                & np.isfinite(signal.values))
        name = signal.body_part + "." + signal.name
        names.append(name)
        columns.append(np.where(good, signal.values, np.nan))
        validity.append(good)
        descriptions.append({
            "name": name, "measurement_name": signal.name, "body_part": signal.body_part,
            "operator": signal.operator, "coordinate_frame": signal.frame,
            "required_points": [data["names"][index] for index in signal.indices],
            "required_point_indices": list(signal.indices), "unit": "degrees",
            "circular": signal.circular,
            "value_representation": "raw_atan2_branch_degrees" if signal.circular else "inter_vector_angle_degrees",
            "calibrated": False,
        })
    n = len(data["time"])
    signal_arrays = {
        "landmark_signal_values": np.column_stack(columns) if columns else np.empty((n, 0), dtype=float),
        "landmark_signal_valid": np.column_stack(validity) if validity else np.empty((n, 0), dtype=bool),
        "landmark_signal_names": np.asarray(names, dtype="U"),
    }
    metadata = {
        "source": SOURCE, "signals": descriptions, "calibrated": False,
        "input_coordinate_frame": "camera_xright_ydown_zforward", "unit": "degrees",
        "notice": "Uncalibrated relative geometry from image-predicted points; not isolated wrist/ankle articulation or contact ground truth.",
        "continuity": "Validity is per timestamp. Before temporal differences or unwrap, split by every required point's point_segment_id and validity.",
        "invalid_value": "NaN", "circular_values": "raw branch [-180,180]; no temporal unwrap",
    }
    return signal_arrays, metadata


def extract_landmark_events(arrays: dict, *, min_duration_seconds=.15, min_angle_change_degrees=20.) -> list[dict]:
    """Extract local measured changes without camera-frame fallback semantics.

    Missing names disable their operator; invalid points or any required-point
    segment change split support. Point fractions are support metadata, not
    probabilities, visibility estimates, or evidence of physical contact.
    """
    for name, value in (("min_duration_seconds", min_duration_seconds),
                        ("min_angle_change_degrees", min_angle_change_degrees)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    data = _inputs(arrays)
    time, events = data["time"], []
    for signal in _signals(data):
        indices = signal.indices
        good = signal.good & data["available"][:, indices].all(axis=1) & np.isfinite(signal.values)
        segments = data["point_segment_id"][:, indices]
        for first, last in _runs(good, time, joint_segment_id=segments):
            values = signal.values[first:last].copy()
            if signal.circular:
                values = np.degrees(np.unwrap(np.radians(values)))
            for a, b in _excursions(values, min_angle_change_degrees * .2):
                start, end = first + a, first + b
                delta = float(values[b] - values[a])
                if (time[end] - time[start] + 1e-12 < min_duration_seconds
                        or abs(delta) + 1e-9 < min_angle_change_degrees):
                    continue
                part = slice(start, end + 1)
                observed = data["point_observed"][part][:, indices]
                interpolated = data["point_interpolated"][part][:, indices]
                if signal.circular:
                    action = "rotates_relative_to_arm_plane"
                    direction = "positive_roll" if delta > 0 else "negative_roll"
                else:
                    action = "changes_relative_orientation"
                    direction = "increasing_relative_angle" if delta > 0 else "decreasing_relative_angle"
                events.append({
                    "id": "", "start": float(time[start]), "end": float(time[end]),
                    "body_part": signal.body_part, "action": action, "direction": direction,
                    "measurement": {"start": float(values[a]), "end": float(values[b]), "delta": delta, "unit": "degrees"},
                    "evidence": {
                        "required_points": [data["names"][i] for i in indices],
                        "required_point_indices": list(indices),
                        "required_point_segment_ids": [int(i) for i in segments[start]],
                        "point_quality": [{"name": data["names"][index], "index": index,
                                           "observed_fraction": float(observed[:, column].mean()),
                                           "interpolated_fraction": float(interpolated[:, column].mean()),
                                           "segment_id": int(segments[start, column])}
                                          for column, index in enumerate(indices)],
                        "observed_fraction": float(observed.all(axis=1).mean()),
                        "interpolated_fraction": float(interpolated.any(axis=1).mean()),
                        "coverage_scope": "required_point_timestamp_support", "quality_is_probability": False,
                        "measurement_name": signal.name, "operator": signal.operator,
                        "coordinate_frame": signal.frame, "input_coordinate_frame": "camera_xright_ydown_zforward",
                        "normalization": "diag(1,-1,-1): X-right, Y-up, Z-backward; proper rotation",
                        "anatomical_side_source": "supplied point names, not camera X sign",
                        "common_so3_translation_invariant": True,
                        "articulation_claim": "relative geometry; not isolated wrist/ankle joint rotation",
                        "normal_sign": "ordered cross product, not a verified palmar/sole facing label",
                        "sample_start": start, "sample_end": end, "sample_count": end - start + 1,
                        "max_sample_gap_seconds": float(np.diff(time[part]).max()),
                        "minimum_change_degrees": float(min_angle_change_degrees),
                        "minimum_duration_seconds": float(min_duration_seconds),
                        "minimum_vector_length_m": _MIN_LENGTH_M, "minimum_plane_sine": _MIN_PLANE_SINE,
                        "minimum_arm_plane_sine": _MIN_ARM_PLANE_SINE,
                        "boundary_method": "sampled_extrema_with_reversal_hysteresis",
                    }, "source": SOURCE,
                })
    events.sort(key=lambda event: (event["start"], event["end"], event["body_part"], event["evidence"]["measurement_name"]))
    for index, event in enumerate(events, 1):
        event["id"] = f"landmark_event_{index:04d}"
    return events


def render_landmark_caption(events: list) -> list[str]:
    """Render strictly measured detail; never substitute wrist/contact labels."""
    captions = []
    for event in sorted(events, key=lambda item: (item["start"], item["end"], item["id"])):
        part = event["body_part"].replace("_", " ").capitalize()
        amount = abs(event["measurement"]["delta"])
        name = event["evidence"]["measurement_name"]
        change = "increases" if event["measurement"]["delta"] > 0 else "decreases"
        if name == "palm_roll_relative_to_arm_plane":
            sign = "positive" if event["measurement"]["delta"] > 0 else "negative"
            text = f"{part} normal rotates {amount:.1f} degrees relative to the upper-arm/forearm plane ({sign} roll)."
        else:
            side = event["body_part"].split("_", 1)[0].capitalize()
            noun = {"palm_normal_to_forearm_angle": "palm-normal-to-forearm angle",
                    "sole_normal_to_shin_angle": "sole-normal-to-shin angle",
                    "foot_long_axis_to_shin_angle": "foot-long-axis-to-shin angle"}.get(name)
            if noun is None:
                raise ValueError(f"Unsupported landmark measurement: {name}")
            text = f"{side} {noun} {change} by {amount:.1f} degrees."
        captions.append(f"[{event['id']} | {event['start']:.2f}-{event['end']:.2f} s | {event['source']}] {text}")
    return captions
