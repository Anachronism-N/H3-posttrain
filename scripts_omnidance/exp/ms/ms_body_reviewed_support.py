"""Keep explicitly reviewed local identity/geometry exclusions as hard support cuts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ms_landmarks import load_bundle, require_new_output
from ms_optimize import atomic_write, json_bytes, save_npz, sha256, source_hashes
from motionscript_v2.body_events import extract_body_events, measure_body_signals, render_body_caption
from motionscript_v2.coordinates import _interpolate, _runs, _segments, _supported_grid
from motionscript_v2.landmarks import _exact_source_matches, preprocess_landmarks
from motionscript_v2.selection import co_supported_groups, summarize_events


REVIEW_KINDS = {
    "identity": ("native-body-identity-support-review-v1", "anatomical_identity_continuity_unknown", "identity_continuity_unknown", 32),
    "geometry": ("native-body-geometry-support-review-v1", "local_geometric_support_unknown", "geometry_support_unknown", 64),
}


def validated_review_mask(review, raw, source, *, kind):
    if kind not in REVIEW_KINDS:
        raise ValueError("Local support review kind must be identity or geometry")
    schema, reason, _, _ = REVIEW_KINDS[kind]
    if (review.get("schema") != schema or review.get("id") != source["id"]
            or review.get("video_sha256") != source["source_sha256"]["video"]
            or review.get("raw_sha256") != source["raw_sha256"]):
        raise ValueError("Local exclusions must bind to the exact source video and raw revision")
    if (review.get("reviewer_type") not in ("assistant", "human")
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
        raise ValueError("An explicit reviewer and reviewer type are required")
    names = np.asarray(raw["names"]).tolist()
    frames = np.asarray(raw["frame_indices"])
    excluded = np.zeros(np.asarray(raw["valid"]).shape, dtype=bool)
    for item in review["exclusions"]:
        if item.get("reason_code") != reason:
            raise ValueError("Exclusion reason does not match the declared local review kind")
        if (not isinstance(item.get("reason"), str) or not item["reason"].strip()
                or not isinstance(item.get("evidence"), list) or not item["evidence"]):
            raise ValueError("Each local exclusion needs an evidence reason")
        selected_names, selected_frames = item["point_names"], item["frame_indices"]
        if (not selected_names or len(selected_names) != len(set(selected_names)) or not set(selected_names) <= set(names)
                or not selected_frames or len(selected_frames) != len(set(selected_frames))
                or any(type(frame) is not int for frame in selected_frames)
                or not set(selected_frames) <= set(frames.tolist())):
            raise ValueError("Local exclusion names and native frame indices must match the source")
        for evidence in item["evidence"]:
            if sha256(Path(evidence["path"])) != evidence["sha256"]:
                raise ValueError("Local review evidence changed")
        indices = [names.index(name) for name in selected_names]
        excluded[np.ix_(np.flatnonzero(np.isin(frames, selected_frames)), indices)] = True
    return excluded


def preprocess_reviewed_support(raw, excluded, *, kind, **config):
    if kind not in REVIEW_KINDS:
        raise ValueError("Local support review kind must be identity or geometry")
    arrays, quality = preprocess_landmarks(raw, **config)
    mask = np.asarray(excluded)
    if mask.shape != arrays["raw_quality_valid"].shape or mask.dtype != np.bool_:
        raise ValueError("Explicit local exclusion must be a boolean native [T,K] mask")
    _, _, reason_name, reason_bit = REVIEW_KINDS[kind]
    baseline_quality = arrays["raw_quality_valid"].copy()
    arrays["raw_" + kind + "_excluded"] = mask.copy()
    arrays["raw_quality_valid"] &= ~mask
    arrays["raw_mask_reason"][mask] |= reason_bit
    source_time, time = arrays["raw_time"], arrays["time"]
    fps = quality["output_fps"]
    maximum_interval = 1 / quality["source_fps"] + quality["max_gap_seconds"]
    sigma = quality["smooth_seconds"] * fps
    matches = _exact_source_matches(source_time, time)
    for point in np.flatnonzero(mask.any(axis=0)):
        values = np.full((len(time), 3), np.nan)
        usable = np.zeros(len(time), bool)
        point_segments = np.full(len(time), -1, np.int32)
        segments = []
        allowed = ~mask[:, point] & ~arrays["raw_screen_excluded"][:, point]
        for start, stop in _runs(allowed):
            selected, supported = _interpolate(source_time[start:stop], raw["points_m"][start:stop, point],
                                               baseline_quality[start:stop, point], time, maximum_interval, sigma)
            values[supported] = selected[supported]
            usable |= supported
            segments.extend(indices + start for indices in _segments(source_time[start:stop],
                baseline_quality[start:stop, point], maximum_interval))
        for identifier, indices in enumerate(segments):
            point_segments[_supported_grid(source_time, indices, time)] = identifier
        observed = np.zeros(len(time), bool)
        matched = matches >= 0
        observed[matched] = arrays["raw_quality_valid"][matches[matched], point]
        observed &= usable
        values[~usable] = np.nan
        point_segments[~usable] = -1
        arrays["points_m"][:, point] = values.astype(np.float32)
        arrays["point_valid"][:, point] = usable
        arrays["point_observed"][:, point] = observed
        arrays["point_interpolated"][:, point] = usable & ~observed
        arrays["point_segment_id"][:, point] = point_segments
        item = quality["per_point"][point]
        item.update(raw_quality_valid_count=int(arrays["raw_quality_valid"][:, point].sum()),
                    supported_source_segments=len(segments), long_gap_count=max(0, len(segments) - 1),
                    raw_missing_runs=len(_runs(~arrays["raw_quality_valid"][:, point])),
                    valid_output_count=int(usable.sum()), valid_output_fraction=float(usable.mean()))
        item[kind + "_excluded_source_frames"] = int(mask[:, point].sum())
    quality["raw_mask_reason_bits"][reason_name] = reason_bit
    quality.update(identity_side_relabeling_performed=False, raw_prediction_arrays_preserved=True,
                   semantic_approval=False, independent_human_review_completed=False)
    quality.update({"explicit_" + kind + "_exclusion_enabled": True,
                    kind + "_excluded_native_points": int(mask.sum()),
                    kind + "_excluded_native_frames": int(mask.any(axis=1).sum()),
                    kind + "_exclusion_crossed_by_smoothing": False,
                    kind + "_exclusion_filled_by_interpolation": False})
    return arrays, quality


def run(args):
    bundle, review_path, protocol_path, out = map(lambda path: Path(path).resolve(),
                                               (args.bundle, args.review, args.protocol, args.out))
    require_new_output(out, [bundle / "raw.npz", review_path, protocol_path])
    raw, source, source_manifest_sha = load_bundle(bundle)
    if sha256(Path(source["source_paths"]["video"])) != source["source_sha256"]["video"]:
        raise ValueError("Bound source video changed")
    review_bytes, protocol_bytes = review_path.read_bytes(), protocol_path.read_bytes()
    review, protocol = json.loads(review_bytes), json.loads(protocol_bytes)
    if (source["source_sha256"]["model"] != protocol["model_sha256"]
            or source["source_sha256"]["mapping"] != protocol["mapping_sha256"]
            or source_hashes() != protocol["measurement_code_sha256"]):
        raise ValueError("Original model, mapping or frozen measurement algorithms changed")
    mask = validated_review_mask(review, raw, source, kind=args.kind)
    config = protocol["config"]
    arrays, quality = preprocess_reviewed_support(raw, mask, kind=args.kind, fps=config["fps"],
        max_gap_seconds=config["max_gap_seconds"], smooth_seconds=config["smooth_seconds"],
        max_point_uncertainty=config["max_point_uncertainty"])
    signals, metadata = measure_body_signals(arrays)
    events = extract_body_events(arrays, min_duration_seconds=config["min_duration_seconds"],
                                min_angle_change_degrees=config["min_angle_change_degrees"],
                                min_distance_change_m=config["min_distance_change_m"])
    summary = summarize_events(events, renderer=render_body_caption)
    out.mkdir(parents=True)
    save_npz(out / "reviewed_tracks.npz", arrays)
    save_npz(out / "signals.npz", signals)
    report = {"schema": "native-body-reviewed-local-support-candidates-v1", "review_kind": args.kind, "id": source["id"],
              "source_manifest_sha256": source_manifest_sha, "source_video_sha256": source["source_sha256"]["video"],
              "original_raw_sha256": source["raw_sha256"], "local_review": review,
              "local_review_sha256": hashlib.sha256(review_bytes).hexdigest(),
              "frozen_measurement_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(), "parameter_config": config,
              "tracks_sha256": sha256(out / "reviewed_tracks.npz"), "signals_sha256": sha256(out / "signals.npz"),
              "quality": quality, "signal_metadata": metadata, "events": events, "summary": summary,
              "co_supported_groups": co_supported_groups(summary["selected_events"]), "captions": render_body_caption(events),
              "automatic_warning_to_exclusion_conversion": False, "frozen_results_modified": False,
              "visible_excluded_motion_still_requires_recall_review": True,
              "output_is_separate_review_diagnostic_not_original_holdout_result": True,
              "semantic_approval": False, "human_review_completed": False, "quality_gate_passed": False,
              "implementation_sha256": sha256(Path(__file__))}
    payload = json_bytes(report)
    atomic_write(out / "candidates.receipt.json", json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "candidates.json", payload)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "review", "protocol", "out"):
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--kind", choices=tuple(REVIEW_KINDS), required=True)
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
