"""Media windows and conservative slices of direct landmark tracks.

Windows locate the input supplied to a descriptor. Their boundaries are not
action boundaries, and the first/last supported samples are not whole-window
evidence. No detector, beat grid, coordinate reset, or gap repair runs here.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np


SCHEMA = "motionscript_windows_v1"
_TOLERANCE = 1e-9
_TEMPORAL = {
    "time", "points_m", "point_valid", "point_segment_id", "point_observed",
    "point_interpolated", "landmark_signal_values", "landmark_signal_valid",
}


def _number(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else " and nonnegative" if nonnegative else ""))
    return value


def _bounds(start, end):
    start = _number(start, "media/window start", nonnegative=True)
    end = _number(end, "media/window end", nonnegative=True)
    if end <= start:
        raise ValueError("media/window end must be greater than start")
    return start, end


def fixed_windows(media_start, media_end, chunk_seconds=2.0, min_tail_seconds=1.5):
    """Cover media with adjacent half-open windows; merge a short final tail.

    A clip shorter than the requested size remains a single complete window.
    This function needs an actual media end; it never estimates it from fps or
    the final pose sample. Four positional arguments are intentionally allowed.
    """
    start, end = _bounds(media_start, media_end)
    chunk_seconds = _number(chunk_seconds, "chunk_seconds", positive=True)
    min_tail_seconds = _number(min_tail_seconds, "min_tail_seconds", nonnegative=True)
    if start + chunk_seconds <= start:
        raise ValueError("chunk_seconds is too small for the media time origin")
    result = []
    previous = start
    index = 1
    while previous < end:
        boundary = min(end, start + index * chunk_seconds)
        if boundary <= previous:
            raise ValueError("chunk_seconds cannot advance the media time grid")
        result.append({"start": previous, "end": boundary})
        previous = boundary
        index += 1
    if len(result) > 1 and result[-1]["end"] - result[-1]["start"] < min_tail_seconds:
        tail = result.pop()
        result[-1]["end"] = tail["end"]
    return result


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError("window specification and provenance must contain finite JSON data") from error


def _beat_source(source):
    if not isinstance(source, dict) or not source:
        raise ValueError("beat strategy requires a beat_source provenance object")
    origin = [source.get(key) for key in ("source", "method", "detector", "path")]
    if not any((isinstance(value, str) and value.strip()) or (isinstance(value, dict) and value)
               for value in origin):
        raise ValueError("beat_source must declare its source, method, detector or path")
    reliability = source.get("reliability")
    if not ((isinstance(reliability, str) and reliability.strip())
            or (isinstance(reliability, dict) and reliability)):
        raise ValueError("beat_source must declare reliability as a nonempty string or object")
    grid_methods = {"global_bpm", "global_bpm_grid", "bpm_grid", "tempo_grid", "uniform_grid", "synthetic_grid"}
    for key in ("method", "kind", "type", "origin"):
        value = source.get(key)
        if isinstance(value, str) and re.sub(r"[ -]+", "_", value.strip().lower()) in grid_methods:
            raise ValueError("beat boundaries require actual beat timestamps, not a global-BPM grid")
    if source.get("derived_from_global_bpm") is True or source.get("synthetic") is True:
        raise ValueError("beat boundaries require actual beat timestamps, not a global-BPM grid")


def validate_windows(spec, *, media_start, media_end):
    """Normalize externally selected complete windows and declared provenance.

    Beat validation checks declared provenance and timestamp alignment only;
    it cannot verify that a detector/annotator found real beats in the audio.
    A regular beat sequence is allowed when it is actual supplied beat data.
    No beat sequence or adaptive segmentation is constructed by this function.
    """
    start, end = _bounds(media_start, media_end)
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError(f"window specification schema must be {SCHEMA}")
    normalized = _json_copy(spec)
    strategy = normalized.get("strategy")
    if strategy not in ("fixed", "beat", "explicit"):
        raise ValueError("window strategy must be fixed, beat or explicit")
    for key, expected in (("media_start", start), ("media_end", end)):
        if key in normalized and abs(_number(normalized[key], key, nonnegative=True) - expected) > _TOLERANCE:
            raise ValueError(f"{key} differs from the actual media bounds")
        normalized[key] = expected
    if "video_sha256" in normalized:
        digest = normalized["video_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("video_sha256 must contain exactly 64 hexadecimal characters")
        normalized["video_sha256"] = digest.lower()
    windows = normalized.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("windows must be a nonempty list covering the complete media interval")
    previous = start
    for index, window in enumerate(windows):
        if not isinstance(window, dict) or "start" not in window or "end" not in window:
            raise ValueError("each window must contain start and end")
        left, right = _bounds(window["start"], window["end"])
        if abs(left - previous) > _TOLERANCE:
            raise ValueError("windows must be continuous and nonoverlapping from media_start")
        left = previous
        if index == len(windows) - 1:
            if abs(right - end) > _TOLERANCE:
                raise ValueError("windows must cover exactly through media_end")
            right = end
        if right > end or right <= left:
            raise ValueError("window bounds must be positive-duration intervals within the media")
        window.update(start=left, end=right)
        previous = right
    if strategy == "beat":
        _beat_source(normalized.get("beat_source"))
        beats = normalized.get("beats")
        if not isinstance(beats, list) or not beats:
            raise ValueError("beat strategy requires an actual beats timestamp list")
        beats = [_number(value, "beat", nonnegative=True) for value in beats]
        if (any(right <= left for left, right in zip(beats, beats[1:]))
                or beats[0] < start or beats[-1] > end):
            raise ValueError("beats must be strictly increasing and inside the media interval")
        for index, window in enumerate(windows[:-1]):
            boundary = window["end"]
            choices = [beat for beat in beats if abs(beat - boundary) <= _TOLERANCE]
            if not choices:
                raise ValueError("every interior beat-window boundary must match a supplied beat")
            boundary = min(choices, key=lambda beat: abs(beat - boundary))
            windows[index]["end"] = windows[index + 1]["start"] = boundary
        if any(window["end"] <= window["start"] for window in windows):
            raise ValueError("beat-boundary normalization produced an empty window")
        normalized["beats"] = beats
    normalized["validation"] = {
        "half_open_windows": True, "continuous_exact_media_coverage": True,
        "boundary_tolerance_seconds": _TOLERANCE,
        "source_provenance_verified": False, "beats_verified_against_audio": False,
        "beat_boundary_alignment_checked": strategy == "beat",
        "window_boundaries_are_action_boundaries": False,
        "time_basis": "global_media_seconds",
    }
    return normalized


def slice_landmarks(arrays, start, end):
    """Select target-time rows in [start,end), retaining raw-source arrays whole.

    The explicit temporal allowlist avoids corrupting raw tracks or point/signal
    names when their length happens to equal the number of target samples.
    Global times, coordinates, validity and segment IDs are never rebased or
    repaired. Returned metadata distinguishes the requested media window from
    the actual sample support. No whole-window action claim is made.
    """
    start, end = _bounds(start, end)
    if not isinstance(arrays, dict):
        raise ValueError("arrays must be a dictionary")
    required = ("time", "points_m", "point_valid", "point_observed", "point_interpolated", "point_segment_id")
    if any(key not in arrays for key in required):
        raise ValueError("missing landmark time, point geometry or point support arrays")
    time = np.asarray(arrays["time"])
    if (time.ndim != 1 or time.dtype.kind not in "fiu" or not np.isfinite(time).all()
            or np.any(time < 0) or np.any(np.diff(time.astype(np.float64)) <= 0)):
        raise ValueError("time must be finite, nonnegative and strictly increasing")
    points = np.asarray(arrays["points_m"])
    if points.ndim != 3 or points.shape[0] != len(time) or points.shape[1] < 1 or points.shape[2] != 3:
        raise ValueError("points_m must have shape [N,K,3]")
    for key in required[2:]:
        if np.shape(arrays[key]) != points.shape[:2]:
            raise ValueError(f"{key} must have shape [N,K]")
    for key in ("point_valid", "point_observed", "point_interpolated"):
        value = np.asarray(arrays[key])
        if (value.dtype.kind not in "biuf" or not np.isfinite(value).all()
                or not ((value == 0) | (value == 1)).all()):
            raise ValueError(f"{key} must contain finite boolean/0/1 values")
    segment = np.asarray(arrays["point_segment_id"])
    if (segment.dtype.kind not in "iuf" or not np.isfinite(segment).all()
            or np.any(segment < -1) or np.any(segment != np.floor(segment))):
        raise ValueError("point_segment_id must contain finite integer IDs >= -1")
    for key in ("landmark_signal_values", "landmark_signal_valid"):
        if key in arrays:
            value = np.asarray(arrays[key])
            if value.ndim != 2 or value.shape[0] != len(time):
                raise ValueError(f"{key} must have shape [N,S]")
    if "landmark_signal_values" in arrays and "landmark_signal_valid" in arrays:
        if np.shape(arrays["landmark_signal_values"]) != np.shape(arrays["landmark_signal_valid"]):
            raise ValueError("landmark signal values and validity must have matching shapes")
    selection = np.flatnonzero((time >= start) & (time < end))
    sliced = dict(arrays)
    for key in _TEMPORAL & arrays.keys():
        sliced[key] = np.asarray(arrays[key])[selection].copy()
    point_valid = np.asarray(sliced["point_valid"], dtype=bool)
    empty = len(selection) == 0
    supported = (point_valid & np.isfinite(sliced["points_m"]).all(axis=2)
                 & (np.asarray(sliced["point_segment_id"]) >= 0)
                 & (np.asarray(sliced["point_observed"], dtype=bool)
                    | np.asarray(sliced["point_interpolated"], dtype=bool)))
    any_usable = bool(supported.any())
    metadata = {
        "window_start": start, "window_end": end, "window_interval": "[start,end)",
        "time_basis": "global_media_seconds", "sample_count": len(selection),
        "first_sample_time": None if empty else float(time[selection[0]]),
        "last_sample_time": None if empty else float(time[selection[-1]]),
        "first_sample_index": None if empty else int(selection[0]),
        "last_sample_index": None if empty else int(selection[-1]),
        "selected_indices": selection.tolist(),
        "sample_index_basis": "indices into the provided target-time array; not raw/video frame indices",
        "sample_span_definition": "first/last selected sample timestamps, both inclusive; distinct from the half-open media window",
        "sample_span_is_complete_window_evidence": False,
        "empty": empty, "abstain": empty or not any_usable,
        "abstain_reason": "no_samples_in_window" if empty else "no_valid_point_support" if not any_usable else None,
        "valid_point_samples": int(point_valid.sum()),
        "usable_point_samples": int(supported.sum()),
        "point_support_fraction": float(point_valid.mean()) if point_valid.size else 0.0,
        "coordinate_reset_applied": False, "segment_ids_preserved": True,
        "raw_arrays_retained_full_length": True,
        "sliced_time_fields": sorted(_TEMPORAL & arrays.keys()),
        "preserved_fields": sorted(arrays.keys() - _TEMPORAL),
        "limits": "Per-point masks and original segment IDs still govern usable continuous evidence; local support must not be relabeled as a complete-window action.",
    }
    return sliced, metadata
