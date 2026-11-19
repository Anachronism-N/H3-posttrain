"""Conservative, auditable movement events from the v2 skeleton arrays.

Angles and distances are measurements, not action recognition probabilities.
The rules deliberately do not infer contact, clapping, jumping, walking, finger
gestures, or dance style.  X signs are retained as coordinate directions rather
than silently interpreted as an anatomical left/right side.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np


SOURCE = "motionscript_v2_rules"
JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee",
    "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck", "left_collar", "right_collar",
    "head", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist",
)
__all__ = ["extract_events", "render_caption", "make_llm_payload"]


@dataclass(frozen=True)
class _Signal:
    name: str
    values: np.ndarray
    good: np.ndarray
    joints: tuple[int, ...]
    body_part: str
    unit: str
    frame: str
    positive: tuple[str, str]
    negative: tuple[str, str]
    circular: bool = False


def _inputs(arrays: dict) -> dict:
    """Validate structure without treating missing/degenerate poses as data."""
    numeric = ("time", "body", "translation", "yaw_deg")
    masks = ("valid", "joint_valid", "observed", "interpolated")
    missing = [key for key in numeric + masks if key not in arrays]
    if missing:
        raise ValueError("Missing event arrays: " + ", ".join(missing))
    data = {key: np.asarray(arrays[key], dtype=float) for key in numeric}
    joint_masks = ("joint_observed", "joint_interpolated")
    if any(key in arrays for key in joint_masks):
        if not all(key in arrays for key in joint_masks):
            raise ValueError("joint_observed and joint_interpolated must be supplied together")
        masks += joint_masks
    for key in masks:
        raw = np.asarray(arrays[key])
        if (raw.dtype.kind not in "biuf" or not np.isfinite(raw).all()
                or not ((raw == 0) | (raw == 1)).all()):
            raise ValueError(f"{key} must contain only finite boolean/0/1 values")
        data[key] = raw.astype(bool)
    if data["time"].ndim != 1:
        raise ValueError("time must have shape [N]")
    n = len(data["time"])
    shapes = {
        "body": (n, 22, 3), "translation": (n, 3), "yaw_deg": (n,),
        "valid": (n,), "joint_valid": (n, 22), "observed": (n,),
        "interpolated": (n,),
    }
    for key in joint_masks:
        if key in data:
            shapes[key] = (n, 22)
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}, got {data[key].shape}")
    if "global_joints" in arrays and np.shape(arrays["global_joints"]) != (n, 22, 3):
        raise ValueError("global_joints must have shape [N,22,3]")
    if not np.isfinite(data["time"]).all() or np.any(np.diff(data["time"]) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    for key, shape in (("segment_id", (n,)), ("joint_segment_id", (n, 22))):
        if key in arrays:
            segment_id = np.asarray(arrays[key])
            if segment_id.shape != shape:
                raise ValueError(f"{key} must have shape {shape}")
            if (segment_id.dtype.kind not in "iuf" or not np.isfinite(segment_id).all()
                    or np.any(segment_id < -1) or np.any(segment_id != np.floor(segment_id))):
                raise ValueError(f"{key} must contain finite integer IDs >= -1")
            data[key] = segment_id
    return data


def _runs(good: np.ndarray, time: np.ndarray,
          segment_id: np.ndarray | None = None,
          joint_segment_id: np.ndarray | None = None) -> Iterable[tuple[int, int]]:
    """Yield half-open runs; never unwrap/filter/detect across invalid gaps.

    Explicit root and required-joint source segment boundaries remain
    authoritative even when resampling has skipped all invalid timestamps.
    Joint IDs are compared as a tuple, not collapsed into a minimum/maximum.
    Also split unmarked timestamp holes exceeding three nominal sample periods.
    This is a continuity guard, not a claim about sub-frame timing accuracy.
    """
    steps = np.diff(time)
    max_step = 3.0 * float(np.median(steps)) if len(steps) else np.inf
    start = None
    for i, usable in enumerate(good):
        usable = bool(usable and (segment_id is None or segment_id[i] >= 0))
        if joint_segment_id is not None:
            usable = usable and bool(np.all(joint_segment_id[i] >= 0))
        gap = i > 0 and time[i] - time[i - 1] > max_step + 1e-12
        if segment_id is not None and i > 0:
            gap = gap or segment_id[i] != segment_id[i - 1]
        if joint_segment_id is not None and i > 0:
            gap = gap or bool(np.any(joint_segment_id[i] != joint_segment_id[i - 1]))
        if start is not None and (not usable or gap):
            yield start, i
            start = None
        if usable and start is None:
            start = i
    if start is not None:
        yield start, len(time)


def _angle(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    un = np.linalg.norm(u, axis=-1)
    vn = np.linalg.norm(v, axis=-1)
    good = np.isfinite(u).all(-1) & np.isfinite(v).all(-1) & (un > 1e-6) & (vn > 1e-6)
    cosine = np.sum(u * v, axis=-1) / np.maximum(un * vn, 1e-12)
    values = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return values, good


def _signals(data: dict) -> Iterable[_Signal]:
    b = data["body"]
    # Availability is evaluated per rule: a masked wrist must not suppress a
    # valid knee event, nor silently support an elbow measurement.
    def available(joints):
        return (data["joint_valid"][:, joints].all(1)
                & np.isfinite(b[:, joints]).all(axis=(1, 2)))

    body_frame = "body_yaw_removed_y_up_m"
    for side, shoulder, elbow, wrist, hip, knee, ankle in (
        ("left", 16, 18, 20, 1, 4, 7),
        ("right", 17, 19, 21, 2, 5, 8),
    ):
        for part, a, pivot, c in (("elbow", shoulder, elbow, wrist),
                                   ("knee", hip, knee, ankle)):
            angle, good = _angle(b[:, a] - b[:, pivot], b[:, c] - b[:, pivot])
            joints = (a, pivot, c)
            yield _Signal(
                "joint_interior_angle", angle, good & available(joints), joints,
                f"{side}_{part}", "degrees", body_frame,
                ("extends", "increasing_joint_angle"),
                ("bends", "decreasing_joint_angle"),
            )
        joints = (0, 12, shoulder, elbow)
        # Refer to the torso axis so leaning the entire body does not, on its
        # own, masquerade as raising an arm relative to the torso.
        angle, good = _angle(b[:, elbow] - b[:, shoulder], b[:, 0] - b[:, 12])
        yield _Signal(
            "upper_arm_elevation_relative_to_torso", angle, good & available(joints),
            joints, f"{side}_upper_arm", "degrees", body_frame,
            ("raises", "upward"), ("lowers", "downward"),
        )

    torso = b[:, 12] - b[:, 0]
    torso_norm = np.linalg.norm(torso, axis=-1)
    # The first version covers upright lean rather than trying to name inverted
    # acrobatics from a projected angle with ambiguous orientation.
    torso_good = available((0, 12)) & (torso_norm > 1e-6) & (torso[:, 1] > 0.05 * torso_norm)
    for axis, name, positive, negative in (
        (0, "torso_side_tilt", "positive_x", "negative_x"),
        (2, "torso_forward_tilt", "forward", "backward"),
    ):
        values = np.degrees(np.arctan2(torso[:, axis], torso[:, 1]))
        yield _Signal(name, values, torso_good, (0, 12), "torso", "degrees", body_frame,
                      ("leans", positive), ("leans", negative))

    hips = b[:, 2] - b[:, 1]
    shoulders = b[:, 17] - b[:, 16]
    hx, hz = hips[:, 0], hips[:, 2]
    sx, sz = shoulders[:, 0], shoulders[:, 2]
    twist = np.degrees(np.arctan2(hz * sx - hx * sz, hx * sx + hz * sz))
    twist_good = (available((1, 2, 16, 17))
                  & (np.hypot(hx, hz) > 1e-6) & (np.hypot(sx, sz) > 1e-6))
    yield _Signal(
        "shoulder_yaw_relative_to_hips", twist, twist_good, (1, 2, 16, 17),
        "torso", "degrees", body_frame,
        ("twists", "positive_yaw"), ("twists", "negative_yaw"), circular=True,
    )

    root_good = available((0,))
    yield _Signal(
        "continuous_root_yaw_relative_to_first_valid_frame", data["yaw_deg"],
        root_good & np.isfinite(data["yaw_deg"]), (0,), "root", "degrees",
        "continuous_relative_root_yaw",
        ("turns", "positive_yaw"), ("turns", "negative_yaw"),
    )
    for axis, positive, negative in ((0, "positive_x", "negative_x"),
                                     (2, "positive_z", "negative_z")):
        values = data["translation"][:, axis]
        yield _Signal(
            "pelvis_translation_" + "xyz"[axis], values,
            root_good & np.isfinite(values), (0,), "root", "m",
            "first_valid_heading_relative_translation_y_up_m",
            ("translates", positive), ("translates", negative),
        )


def _excursions(values: np.ndarray, deadband: float) -> Iterable[tuple[int, int]]:
    """Directional-change segmentation with hysteresis, without fixed bins.

    A reversal must exceed ``deadband`` before the previous excursion ends.
    Flat starts/ends are excluded so an instantaneous spike plus a long hold
    cannot pass the movement-duration requirement.  Endpoints are actual
    measured extrema; there is no interpolation to a threshold crossing.
    """
    if len(values) < 2:
        return
    low = high = start = extreme = 0
    direction = 0
    for i in range(1, len(values)):
        value = values[i]
        if direction == 0:
            if value <= values[low]:
                low = i
            if value >= values[high]:
                high = i
            if value - values[low] >= deadband:
                start, extreme, direction = low, i, 1
            elif values[high] - value >= deadband:
                start, extreme, direction = high, i, -1
        elif direction > 0:
            if value > values[extreme]:
                extreme = i
            elif values[extreme] - value >= deadband:
                yield start, extreme
                start, extreme, direction = extreme, i, -1
        else:
            if value < values[extreme]:
                extreme = i
            elif value - values[extreme] >= deadband:
                yield start, extreme
                start, extreme, direction = extreme, i, 1
    if direction:
        yield start, extreme


def extract_events(
    arrays: dict, *, min_duration_seconds: float = 0.15,
    min_angle_change_degrees: float = 20.0, min_distance_change_m: float = 0.12,
) -> list[dict]:
    """Extract supported changes, preserving side, timing, and provenance.

    ``valid`` controls continuity, including any short interpolations explicitly
    accepted by the upstream adapter.  Long gaps must remain invalid.  Rule-
    specific joint masks and finite/degeneracy checks further restrict support.
    With joint support masks, coverage is all-required-joints observed / any-
    required-joint interpolated per timestamp. The legacy fallback is explicitly
    root timestamp support only. Neither scope expresses confidence.
    """
    for name, value in (
        ("min_duration_seconds", min_duration_seconds),
        ("min_angle_change_degrees", min_angle_change_degrees),
        ("min_distance_change_m", min_distance_change_m),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    data = _inputs(arrays)
    time = data["time"]
    events = []
    supported = data["valid"] & (data["observed"] | data["interpolated"])
    for signal in _signals(data):
        threshold = min_distance_change_m if signal.unit == "m" else min_angle_change_degrees
        deadband = threshold * 0.2
        good = supported & signal.good & np.isfinite(signal.values)
        joint_segments = (data["joint_segment_id"][:, signal.joints]
                          if "joint_segment_id" in data else None)
        for run_start, run_end in _runs(good, time, data.get("segment_id"), joint_segments):
            values = signal.values[run_start:run_end].copy()
            if signal.circular:
                values = np.degrees(np.unwrap(np.radians(values)))
            for first, last in _excursions(values, deadband):
                start, end = run_start + first, run_start + last
                delta = float(values[last] - values[first])
                duration = float(time[end] - time[start])
                if duration + 1e-12 < min_duration_seconds or abs(delta) + 1e-9 < threshold:
                    continue
                action, direction = signal.positive if delta > 0 else signal.negative
                segment = slice(start, end + 1)
                if "joint_observed" in data:
                    observed = data["joint_observed"][segment][:, signal.joints].all(axis=1)
                    interpolated = data["joint_interpolated"][segment][:, signal.joints].any(axis=1)
                    coverage_scope = "required_joint_timestamp_support"
                else:
                    observed = data["observed"][segment]
                    interpolated = data["interpolated"][segment]
                    coverage_scope = "root_timestamp_only"
                variation = float(np.abs(np.diff(values[first:last + 1])).sum())
                events.append({
                    "id": "", "start": float(time[start]), "end": float(time[end]),
                    "body_part": signal.body_part, "action": action, "direction": direction,
                    "measurement": {
                        "start": float(values[first]), "end": float(values[last]),
                        "delta": delta, "unit": signal.unit,
                    },
                    "evidence": {
                        "required_joints": list(signal.joints),
                        "required_joint_names": [JOINT_NAMES[j] for j in signal.joints],
                        "observed_fraction": float(observed.mean()),
                        "interpolated_fraction": float(interpolated.mean()),
                        "coverage_scope": coverage_scope,
                        "observed_fraction_definition": ("fraction of timestamps with all required joints observed"
                                                          if "joint_observed" in data else
                                                          "fraction of timestamps with root observed"),
                        "interpolated_fraction_definition": ("fraction of timestamps with any required joint interpolated"
                                                              if "joint_interpolated" in data else
                                                              "fraction of timestamps with root interpolated"),
                        "quality_is_probability": False,
                        "coordinate_frame": signal.frame,
                        "axis_convention": "Y up; +Z forward; X sign is not an anatomical side label",
                        "positive_yaw_convention": "rotation from +Z toward +X",
                        "measurement_name": signal.name,
                        "sample_start": int(start), "sample_end": int(end),
                        "sample_count": int(end - start + 1),
                        "continuity_segment_id": (int(data["segment_id"][start])
                                                  if "segment_id" in data else None),
                        "required_joint_segment_ids": ([int(value) for value in joint_segments[start]]
                                                       if joint_segments is not None else None),
                        "continuity_scope": ("root_and_required_joint_segments"
                                             if joint_segments is not None and "segment_id" in data else
                                             "required_joint_segments" if joint_segments is not None else
                                             "root_segment_only" if "segment_id" in data else
                                             "valid_masks_and_timestamps_only"),
                        "max_sample_gap_seconds": float(np.diff(time[segment]).max()),
                        "net_change_over_total_variation": min(1.0, abs(delta) / max(variation, 1e-12)),
                        "minimum_change": float(threshold),
                        "minimum_duration_seconds": float(min_duration_seconds),
                        "reversal_deadband": float(deadband),
                        "boundary_method": "sampled_extrema_with_reversal_hysteresis",
                    },
                    "source": SOURCE,
                })
    events.sort(key=lambda e: (e["start"], e["end"], e["body_part"],
                               e["evidence"]["measurement_name"], e["direction"]))
    for i, event in enumerate(events, 1):
        event["id"] = f"event_{i:04d}"
    return events


def _ordered(events: list) -> list:
    return sorted(events, key=lambda e: (e["start"], e["end"], e["id"]))


def render_caption(events: list) -> list[str]:
    """One chronological, attributed sentence per event; no inferred ordering links."""
    captions = []
    for event in _ordered(events):
        part = event["body_part"].replace("_", " ").capitalize()
        action, direction = event["action"], event["direction"]
        measurement = event["measurement"]
        amount = abs(measurement["delta"])
        if action in ("bends", "extends", "raises", "lowers"):
            text = f"{part} {action} by {amount:.1f} degrees."
        elif action in ("twists", "turns"):
            sign = "positive" if direction == "positive_yaw" else "negative"
            text = f"{part} {action} by {amount:.1f} degrees ({sign} yaw)."
        elif action == "leans":
            toward = {"positive_x": "body +X", "negative_x": "body -X",
                      "forward": "forward", "backward": "backward"}[direction]
            text = f"{part} changes tilt by {amount:.1f} degrees toward {toward}."
        elif action == "translates":
            toward = {"positive_x": "+X", "negative_x": "-X",
                      "positive_z": "+Z", "negative_z": "-Z"}[direction]
            text = f"Pelvis moves {amount:.2f} m along initial-heading {toward}."
        else:
            raise ValueError(f"Unsupported event action: {action}")
        prefix = f"[{event['id']} | {event['start']:.2f}-{event['end']:.2f} s | {event['source']}]"
        captions.append(prefix + " " + text)
    return captions


def make_llm_payload(events: list) -> dict:
    """Build a constrained rewrite request locally; never call an external API."""
    ordered = copy.deepcopy(_ordered(events))
    return {
        "task": "fact_preserving_motion_event_rewrite",
        "schema_version": 1,
        "abstain": not bool(ordered),
        "events": ordered,
        "fallback_captions": render_caption(ordered),
        "constraints": [
            "Use only the supplied event facts; do not add actions, poses, contacts, counts, intention, style, or emotion.",
            "Retain every event ID verbatim, its source, start/end times, anatomical side, and measured direction.",
            "Return one sentence per event, ordered by start time; do not merge left and right events or remove events.",
            "Do not use then, while, simultaneously, or other inferred temporal relationships; retain explicit times.",
            "Positive/negative coordinate directions are not anatomical left/right; never relabel them as such.",
            "Observed/interpolated fractions describe timestamp support within evidence.coverage_scope, not probability or confidence.",
            "root_timestamp_only coverage cannot establish observation or visibility of individual body parts; never upgrade its scope.",
            "Do not infer jump, landing, foot contact, clapping, hand/finger gestures, walking, or running.",
            "Preserve numeric measurements and units if mentioned; do not turn angular change into a count of full spins.",
            "If events is empty, return abstain=true and sentences=[]; do not invent a static-pose description.",
        ],
        "response_schema": {
            "abstain": "boolean",
            "sentences": [{"event_id": "verbatim event ID", "start": "event start seconds",
                           "end": "event end seconds", "source": SOURCE,
                           "text": "one concise English sentence supported by that event"}],
        },
    }
