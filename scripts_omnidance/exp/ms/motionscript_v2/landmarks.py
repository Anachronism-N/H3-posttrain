"""Named arbitrary-point predictions, without a body template or SMPL FK.

The extractor supplies actual per-point camera-coordinate predictions in metres.
This module validates their declared frame and resamples each point separately;
it never supplies a missing point from another point or a rest-body template.
"""

from __future__ import annotations

import json
import re

import numpy as np

from .coordinates import _interpolate, _runs, _scalar, _segments, _supported_grid


SPEC_SCHEMA = "motionscript_landmark_spec_v1"
CANONICAL_FRAME = "nlf_canonical_smplx"
CAMERA_FRAME = "camera_xright_ydown_zforward"
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def _text(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar string")
    return str(array.item())


def _names(value, count):
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind != "U":
        raise ValueError("names must be a [K] array of canonical string identifiers")
    names = array.tolist()
    if any(not _NAME.fullmatch(name) for name in names) or len(set(names)) != count:
        raise ValueError("names must be unique lowercase identifiers matching [a-z][a-z0-9_]*")
    return np.asarray(names, dtype=str)


def _numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    return array.astype(np.float64)


def validate_point_spec(spec: dict) -> dict:
    """Return a JSON-safe canonical query specification, without certifying it.

    Coordinates are already in metres in NLF's canonical SMPL-X frame. Names
    identify queries; they do not establish that a query is anatomically correct
    or that a trained predictor can reliably observe it. Optional ``source``
    records declared provenance, which this validator does not authenticate.
    """
    if not isinstance(spec, dict) or spec.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"point specification schema must be {SPEC_SCHEMA}")
    if spec.get("units") != "m":
        raise ValueError("canonical point specification units must be m; convert upstream")
    if spec.get("canonical_frame") != CANONICAL_FRAME:
        raise ValueError(f"canonical_frame must be {CANONICAL_FRAME}")
    points = spec.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("points must be a nonempty list of named canonical queries")
    names, normalized = [], []
    for point in points:
        if not isinstance(point, dict) or set(point) != {"name", "canonical"}:
            raise ValueError("each point must contain exactly name and canonical")
        name = point["name"]
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("point names must be lowercase canonical identifiers")
        canonical = _numeric(point["canonical"], "canonical")
        if (canonical.shape != (3,) or not np.isfinite(canonical).all()
                or any(isinstance(value, (bool, np.bool_)) for value in point["canonical"])):
            raise ValueError("canonical must be a finite real [3] metre coordinate")
        names.append(name)
        normalized.append({"name": name, "canonical": canonical.tolist()})
    _names(names, len(names))
    source = spec.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("source provenance must be a JSON object")
    try:
        source = json.loads(json.dumps(source, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError("source provenance must be finite JSON data") from error
    return {"schema": SPEC_SCHEMA, "units": "m", "canonical_frame": CANONICAL_FRAME,
            "points": normalized, "source": source}


def _exact_source_matches(source_time, time):
    """Match timestamps once, without a T-by-N coincidence matrix."""
    result = np.full(len(time), -1, dtype=np.int64)
    insertion = np.searchsorted(source_time, time)
    for shift in (-1, 0):
        candidate = np.clip(insertion + shift, 0, len(source_time) - 1)
        matches = np.isclose(time, source_time[candidate], rtol=0, atol=1e-9)
        result[matches] = candidate[matches]
    return result


def preprocess_landmarks(raw: dict, *, fps=20.0, max_gap_seconds=0.2,
                         smooth_seconds=0.0625,
                         max_point_uncertainty=None) -> tuple[dict, dict]:
    """Resample independent measured/predicted point tracks, retaining gaps.

    Required raw keys are ``points_m[T,K,3]``, ``valid[T,K]``, ``timestamps[T]``,
    ``names[K]``, ``fps``, ``coordinate_frame`` and ``units``. Optional
    ``uncertainty[T,K]`` gates only when ``max_point_uncertainty`` is supplied.
    A missing/invalid point stays NaN outside its own bounded support spans.

    Output ``point_observed`` means a timestamp matched a quality-usable raw
    prediction, not a direct sensor measurement or a calibrated confidence.
    Smoothing can change its value. ``point_interpolated`` is remaining valid
    support, including resampling. IDs in ``point_segment_id`` are local to
    each point and must not be joined across a gap hidden by a coarse grid.
    Frame/units are preserved verbatim; this function never recovers world
    coordinates, applies FK, fits a body, or infers another landmark's value.
    """
    if not isinstance(raw, dict):
        raise ValueError("raw must be a dictionary")
    fps = _scalar(fps, "fps")
    source_fps = _scalar(raw.get("fps"), "raw fps")
    max_gap_seconds = _scalar(max_gap_seconds, "max_gap_seconds", positive=False)
    smooth_seconds = _scalar(smooth_seconds, "smooth_seconds", positive=False)
    if max_point_uncertainty is not None:
        max_point_uncertainty = _scalar(max_point_uncertainty, "max_point_uncertainty", positive=False)
    frame = _text(raw.get("coordinate_frame"), "coordinate_frame")
    if frame != CAMERA_FRAME:
        raise ValueError(f"coordinate_frame must be {CAMERA_FRAME}; no implicit axis conversion")
    if _text(raw.get("units"), "units") != "m":
        raise ValueError("points_m units must be m; millimetre conversion belongs in the extractor")
    points = _numeric(raw.get("points_m"), "points_m")
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[0] < 1 or points.shape[1] < 1:
        raise ValueError("points_m must have shape [T,K,3] with T,K >= 1")
    source_count, point_count = points.shape[:2]
    names = _names(raw.get("names"), point_count)
    source_time = _numeric(raw.get("timestamps"), "timestamps")
    if (source_time.shape != (source_count,) or not np.isfinite(source_time).all()
            or source_time[0] < 0 or np.any(np.diff(source_time) <= 0)):
        raise ValueError("timestamps must be finite, nonnegative, strictly increasing [T] seconds")
    raw_valid = np.asarray(raw.get("valid"))
    if (raw_valid.shape != (source_count, point_count) or raw_valid.dtype.kind not in "biu"
            or np.any((raw_valid != 0) & (raw_valid != 1))):
        raise ValueError("valid must be a boolean or binary integer [T,K] mask")
    raw_valid = raw_valid.astype(bool)
    raw_quality_valid = raw_valid & np.isfinite(points).all(axis=2)
    uncertainty = None
    if "uncertainty" in raw:
        uncertainty = _numeric(raw["uncertainty"], "uncertainty")
        if uncertainty.shape != (source_count, point_count):
            raise ValueError("uncertainty must have shape [T,K]")
    if max_point_uncertainty is not None:
        if uncertainty is None:
            raise ValueError("uncertainty is required when max_point_uncertainty is supplied")
        raw_quality_valid &= np.isfinite(uncertainty) & (uncertainty >= 0)
        raw_quality_valid &= uncertainty <= max_point_uncertainty

    output_count = int(np.floor((source_time[-1] - source_time[0]) * fps + 1e-10)) + 1
    time = source_time[0] + np.arange(output_count, dtype=np.float64) / fps
    time = time[time <= source_time[-1]]
    maximum_interval = 1 / source_fps + max_gap_seconds
    output = np.full((len(time), point_count, 3), np.nan, dtype=np.float64)
    point_valid = np.zeros((len(time), point_count), dtype=bool)
    point_segment_id = np.full((len(time), point_count), -1, dtype=np.int32)
    per_point = []
    for point in range(point_count):
        selected, supported = _interpolate(source_time, points[:, point], raw_quality_valid[:, point],
                                            time, maximum_interval, smooth_seconds * fps)
        output[:, point] = selected
        point_valid[:, point] = supported
        segments = _segments(source_time, raw_quality_valid[:, point], maximum_interval)
        for identifier, indices in enumerate(segments):
            point_segment_id[_supported_grid(source_time, indices, time), point] = identifier
        per_point.append({"name": str(names[point]), "raw_valid_count": int(raw_valid[:, point].sum()),
                          "raw_quality_valid_count": int(raw_quality_valid[:, point].sum()),
                          "supported_source_segments": len(segments),
                          "long_gap_count": max(0, len(segments) - 1),
                          "raw_missing_runs": len(_runs(~raw_quality_valid[:, point])),
                          "valid_output_count": int(supported.sum()),
                          "valid_output_fraction": float(supported.mean())})
    matches = _exact_source_matches(source_time, time)
    point_observed = np.zeros_like(point_valid)
    matched = matches >= 0
    point_observed[matched] = raw_quality_valid[matches[matched]]
    point_observed &= point_valid
    point_segment_id[~point_valid] = -1
    output[~point_valid] = np.nan
    arrays = {
        "time": time, "points_m": output.astype(np.float32), "names": names,
        "point_valid": point_valid, "point_observed": point_observed,
        "point_interpolated": point_valid & ~point_observed,
        "point_segment_id": point_segment_id,
        "raw_points_m": points.astype(np.float32), "raw_time": source_time.copy(),
        "raw_valid": raw_valid.copy(), "raw_quality_valid": raw_quality_valid,
        "coordinate_frame": np.asarray(frame), "units": np.asarray("m"),
    }
    if uncertainty is not None:
        arrays["raw_uncertainty"] = uncertainty.astype(np.float32)
    uncertainty_units = raw.get("uncertainty_units", "model_native_unspecified")
    uncertainty_units = _text(uncertainty_units, "uncertainty_units")
    metadata = {
        "schema": "motionscript_landmark_tracks_v1", "source_frames": source_count,
        "output_frames": len(time), "point_count": point_count,
        "source_fps": source_fps, "output_fps": fps, "coordinate_frame": frame, "units": "m",
        "coordinate_transform_applied": False, "world_coordinates_recovered": False,
        "geometry_source": "input per-point coordinates, declared upstream RGB predictions; no SMPL FK or template fill",
        "source_provenance_verified": False,
        "point_names_are_verified_anatomy": False,
        "output_geometry_dtype": "float32", "time_dtype": "float64",
        "max_gap_seconds": max_gap_seconds,
        "gap_duration_definition": "bracketing_observation_interval_minus_one_nominal_source_frame",
        "edge_extrapolation": False, "smooth_seconds": smooth_seconds,
        "smoothing": "per-point output-grid Gaussian positions within independent support segments",
        "cross_point_filling": False, "smoothing_crosses_unsupported_gap": False,
        "uncertainty_gate_enabled": max_point_uncertainty is not None,
        "max_point_uncertainty": max_point_uncertainty, "uncertainty_units": uncertainty_units,
        "uncertainty_is_calibrated_confidence": False,
        "raw_uncertainty_preserved": uncertainty is not None, "uncertainty_resampled": False,
        "raw_valid_definition": "original upstream per-point validity mask, preserved before finite/uncertainty gating",
        "point_valid_definition": "finite point prediction with optional uncertainty gate and bounded temporal support; no skeleton parent chain",
        "point_observed_definition": "output timestamp matches a quality-usable raw prediction for this point; smoothing may change its value; not a calibrated observation confidence",
        "point_segment_id_definition": "independent source support segment for each point; invalid=-1; adjacent output samples can have distinct IDs",
        "input_masked_nonfinite_count": int((raw_valid & ~np.isfinite(points).all(axis=2)).sum()),
        "semantic_approval": False, "per_point": per_point,
    }
    return arrays, metadata
