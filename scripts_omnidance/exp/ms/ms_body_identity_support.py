"""Apply explicit source-bound identity exclusions without guessing new sides."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ms_landmarks import load_bundle, require_new_output
from ms_optimize import atomic_write, json_bytes, save_npz, sha256, source_hashes
from motionscript_v2.body_events import extract_body_events, measure_body_signals, render_body_caption
from ms_body_reviewed_support import preprocess_reviewed_support, validated_review_mask
from motionscript_v2.selection import co_supported_groups, summarize_events


SCHEMA = "native-body-identity-support-review-v1"


def review_mask(review, raw, source):
    return validated_review_mask(review, raw, source, kind="identity")


def preprocess_identity_support(raw, excluded, **config):
    return preprocess_reviewed_support(raw, excluded, kind="identity", **config)


def run(args):
    bundle, review_path, protocol_path, out = map(lambda path: Path(path).resolve(),
                                               (args.bundle, args.review, args.protocol, args.out))
    require_new_output(out, [bundle / "raw.npz", review_path, protocol_path])
    raw, source, source_manifest_sha = load_bundle(bundle)
    if sha256(Path(source["source_paths"]["video"])) != source["source_sha256"]["video"]:
        raise ValueError("Bound source video changed")
    review_bytes = review_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    review = json.loads(review_bytes)
    protocol = json.loads(protocol_bytes)
    review_sha = hashlib.sha256(review_bytes).hexdigest()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if (source["source_sha256"]["model"] != protocol["model_sha256"]
            or source["source_sha256"]["mapping"] != protocol["mapping_sha256"]
            or source_hashes() != protocol["measurement_code_sha256"]):
        raise ValueError("Original model, mapping or frozen measurement algorithms changed")
    mask = review_mask(review, raw, source)
    config = protocol["config"]
    arrays, quality = preprocess_identity_support(raw, mask, fps=config["fps"], max_gap_seconds=config["max_gap_seconds"],
        smooth_seconds=config["smooth_seconds"], max_point_uncertainty=config["max_point_uncertainty"])
    signals, metadata = measure_body_signals(arrays)
    events = extract_body_events(arrays, min_duration_seconds=config["min_duration_seconds"],
                                min_angle_change_degrees=config["min_angle_change_degrees"],
                                min_distance_change_m=config["min_distance_change_m"])
    summary = summarize_events(events, renderer=render_body_caption)
    out.mkdir(parents=True)
    save_npz(out / "identity_supported_tracks.npz", arrays)
    save_npz(out / "signals.npz", signals)
    report = {"schema": "native-body-explicit-identity-support-candidates-v1", "id": source["id"],
              "source_manifest_sha256": source_manifest_sha, "source_video_sha256": source["source_sha256"]["video"],
              "original_raw_sha256": source["raw_sha256"], "identity_review": review, "identity_review_sha256": review_sha,
              "frozen_measurement_protocol_sha256": protocol_sha, "parameter_config": config,
              "tracks_sha256": sha256(out / "identity_supported_tracks.npz"), "signals_sha256": sha256(out / "signals.npz"),
              "quality": quality, "signal_metadata": metadata, "events": events, "summary": summary,
              "co_supported_groups": co_supported_groups(summary["selected_events"]),
              "captions": render_body_caption(events), "original_frozen_results_modified": False,
              "visible_motion_in_excluded_intervals_must_still_count_for_recall": True,
              "output_is_independent_diagnostic_revision_not_original_holdout_result": True,
              "semantic_approval": False, "human_review_completed": False, "quality_gate_passed": False,
              "implementation_sha256": sha256(Path(__file__)),
              "support_dependency_sha256": sha256(Path(__file__).with_name("ms_body_reviewed_support.py"))}
    payload = json_bytes(report)
    atomic_write(out / "candidates.receipt.json", json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "candidates.json", payload)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "review", "protocol", "out"):
        ap.add_argument("--" + key, required=True)
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
