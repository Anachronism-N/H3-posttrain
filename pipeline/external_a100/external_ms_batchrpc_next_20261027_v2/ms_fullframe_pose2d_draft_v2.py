"""CPU-only, write-once fullframe drafts with one explicit additional NPZ contract.

Accepts the two frozen v1 fullvideo schemas unchanged, or exact
ms_fullframe_pose2d_npz_v1 with float boxes[N,4], scalar width/height/duration.
Only validation receives an in-memory boxes->bbox/schema bridge. Generation
and binding use the original arrays and manifest, including the original schema.
Optional source_pts/source_time_base must be paired and reproduce native PTS.
Optional decoder_frame_timestamps are float [N,2] source-PTS start/end pairs,
checked against recorded actual_probe frame starts and pkt_duration_time, never
an inferred FPS. Raw-effective finite scores must still exceed 0.3; finite
image-external predictions remain in the cache/binding but are excluded from
rule geometry/event support by a separate mask and reported as local evidence
limitations, never clipped. No media/model/audio/ROI receipt/bar is accessed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np

import ms_fullframe_pose2d_draft_v1 as v1

rules = v1.rules
SCHEMA = "ms_fullframe_pose2d_rule_draft_v2"
MANIFEST_SCHEMA = v1.MANIFEST_SCHEMA
BINDING_SCHEMA = v1.BINDING_SCHEMA
FLAGS = v1.FLAGS
BOXES_SCHEMA = "ms_fullframe_pose2d_npz_v1"
VALIDATION_SCHEMA = "ms_fullvideo_pose2d_diagnostic_v1"
INPUT_SCHEMAS = v1.INPUT_SCHEMAS | {BOXES_SCHEMA}
REQUIRED_BY_SCHEMA = {schema: v1.REQUIRED for schema in v1.INPUT_SCHEMAS}
REQUIRED_BY_SCHEMA[BOXES_SCHEMA] = (v1.REQUIRED - {"bbox"}) | {"boxes", "width", "height", "duration"}
check_row = v1.check_row
fullframe_binding = v1.fullframe_binding


def input_schema(data):
    if "schema" not in data:
        raise ValueError("Missing full-frame NPZ schema")
    schema = rules.scalar_text(data, "schema")
    if schema not in INPUT_SCHEMAS:
        raise ValueError("Not an explicitly accepted full-frame NPZ schema; aliases/old ROI are not accepted")
    missing = REQUIRED_BY_SCHEMA[schema] - set(data)
    if missing:
        raise ValueError(f"Missing full-frame NPZ fields: {sorted(missing)}")
    if any(value.dtype.hasobject for value in data.values()):
        raise ValueError("Object arrays are never accepted")
    return schema


def validate_boxes_metadata(manifest, data):
    for key, manifest_key in (("width", "width"), ("height", "height"), ("duration", "duration_seconds")):
        value = data[key]
        expected = v1.finite_number(manifest.get(manifest_key), manifest_key)
        if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
            raise ValueError(f"01 {key} must be a finite numeric scalar")
        if not np.isclose(float(value), expected, atol=1e-6, rtol=0):
            raise ValueError(f"01 NPZ/fullframe {key} mismatch")
    if ("source_pts" in data) != ("source_time_base" in data):
        raise ValueError("source_pts and source_time_base must be provided together")
    n = manifest["frame_count"]
    if "source_pts" in data:
        pts = data["source_pts"]
        base_text = rules.scalar_text(data, "source_time_base")
        if re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", base_text) is None:
            raise ValueError("source_time_base must be a positive numerator/denominator text scalar")
        if pts.shape != (n,) or pts.dtype.kind not in "iu":
            raise ValueError("source_pts must contain native integer PTS ticks[N]")
        timestamps = pts.astype(np.float64) * float(Fraction(base_text))
        if not np.allclose(timestamps, data["source_timestamps"], atol=1e-6, rtol=0):
            raise ValueError("source_pts/source_time_base disagree with native source_timestamps")
        if "source_pts" in manifest:
            values = manifest["source_pts"]
            if not isinstance(values, list) or any(type(v) is not int for v in values) or not np.array_equal(pts, values):
                raise ValueError("Fullframe manifest/NPZ source_pts mismatch")
        if "source_time_base" in manifest and manifest["source_time_base"] != base_text:
            raise ValueError("Fullframe manifest/NPZ source_time_base mismatch")
    if "decoder_frame_timestamps" in data:
        decoder = data["decoder_frame_timestamps"]
        if decoder.shape != (n, 2) or decoder.dtype.kind != "f" or not np.isfinite(decoder).all():
            raise ValueError("decoder_frame_timestamps must be finite float source-PTS start/end[N,2]")
        if not np.allclose(decoder[:, 0], data["source_timestamps"], atol=1e-6, rtol=0):
            raise ValueError("Decoder start PTS disagree with native source_timestamps")
        if np.any(decoder[:, 1] <= decoder[:, 0]) or np.any(decoder[:-1, 1] > decoder[1:, 0] + 1e-6):
            raise ValueError("Decoder frame start/end intervals must be positive and nonoverlapping")
        if np.any(decoder[:, 1] - manifest["pts_origin"] > manifest["duration_seconds"] + 1e-6):
            raise ValueError("Decoder frame end exceeds recorded video duration")
        provenance = manifest.get("metadata_provenance", {})
        probe = provenance.get("actual_probe", {}) if isinstance(provenance, dict) else {}
        frames = probe.get("frames") if isinstance(probe, dict) else None
        if not isinstance(frames, list) or len(frames) != n:
            raise ValueError("Decoder [N,2] contract requires recorded actual_probe.frames to verify start/end meaning")
        for i, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ValueError("Recorded actual_probe frame must be an object")
            try:
                start = float(frame["best_effort_timestamp_time"])
                duration = float(frame["pkt_duration_time"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Recorded native decoder start/end evidence is missing") from error
            if not np.isfinite([start, duration]).all() or duration <= 0:
                raise ValueError("Recorded native frame PTS/duration must be finite and positive")
            if not np.allclose(decoder[i], [start, start + duration], atol=1e-6, rtol=0):
                raise ValueError("Decoder start/end disagree with recorded native frame PTS/duration")
            if "source_pts" in data and frame.get("best_effort_timestamp") != int(data["source_pts"][i]):
                raise ValueError("source_pts disagree with recorded native frame PTS ticks")


def in_frame_points(manifest, data):
    xy = data["xy"]
    return (np.isfinite(xy).all(axis=2) & (xy[:, :, 0] >= 0) & (xy[:, :, 0] < manifest["width"])
            & (xy[:, :, 1] >= 0) & (xy[:, :, 1] < manifest["height"]))


def validate_observations(row, manifest, data):
    check_row(row)
    schema = input_schema(data)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Expected an explicit standalone fullframe manifest schema")
    if manifest.get("input_schema", schema) != schema:
        raise ValueError("Fullframe manifest/NPZ original input_schema mismatch")
    bridge_data, bridge_manifest = data, manifest
    if schema == BOXES_SCHEMA:
        if "bbox" in data:
            raise ValueError("01 contract uses boxes only; an additional bbox alias is not accepted")
        # Copies are validation-only; frozen NPZ arrays and manifest are never rewritten.
        bridge_data = {key: value.copy() for key, value in data.items()}
        bridge_data["bbox"] = bridge_data.pop("boxes")
        bridge_data["schema"] = np.array(VALIDATION_SCHEMA)
        bridge_manifest = dict(manifest)
        if "input_schema" in bridge_manifest:
            bridge_manifest["input_schema"] = VALIDATION_SCHEMA
    try:
        mask = v1.validate_observations(row, bridge_manifest, bridge_data)
    except ValueError as error:
        if str(error) != "Effective valid xy must be in native pixel bounds":
            raise
        # This exact v1 failure occurs AFTER raw-effective finite/threshold checks.
        # Retry the entire validator so its subsequent tracked-bbox check still runs.
        bounded_data = dict(bridge_data)
        bounded_data["valid"] = bridge_data["valid"] & in_frame_points(manifest, data)
        mask = v1.validate_observations(row, bridge_manifest, bounded_data)
    if schema == BOXES_SCHEMA:
        validate_boxes_metadata(manifest, data)
    return mask


def load_observations(row):
    check_row(row)
    manifest = rules.read_json(row["fullframe_manifest_path"])
    raw = Path(row["npz_path"]).read_bytes()
    if "npz_sha256" in manifest and manifest["npz_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Fullframe manifest/actual NPZ byte SHA256 mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    mask = validate_observations(row, manifest, data)
    return manifest, data, mask


def spatial_evidence(manifest, data, mask):
    raw_effective = data["valid"] & data["tracked"][:, None]
    excluded = raw_effective & ~in_frame_points(manifest, data)
    per_signal = []
    if excluded.any():
        original_signals, _ = rules.rule_signals(data, raw_effective)
        usable_signals, _ = rules.rule_signals(data, mask)
        for original, usable in zip(original_signals, usable_signals):
            count = int(usable["valid"].sum())
            affected = original["valid"] & ~usable["valid"]
            per_signal.append({"signal": usable["name"], "usable_frame_count": count,
                               "excluded_support_frame_count": int(affected.sum()),
                               "excluded_support_frame_indices": data["frame_indices"][affected].tolist(),
                               "minimum_total_frames": rules.RULES["min_frames"],
                               "below_minimum_total_frames": count < rules.RULES["min_frames"],
                               "continuous_event_support_not_guaranteed": True})
    return {"strategy": "rule_observation_mask = raw_valid & tracked & finite_native_pixel_in_frame; no xy clipping, rescaling or raw_valid rewrite",
            "excluded_point_count": int(excluded.sum()),
            "affected_frame_indices": data["frame_indices"][excluded.any(axis=1)].tolist(),
            "affected_joint_indices": np.flatnonzero(excluded.any(axis=0)).tolist(),
            "excluded_point_count_by_joint": [int(v) for v in excluded.sum(axis=0)],
            "usable_point_count": int(mask.sum()), "per_signal_support": per_signal,
            "local_evidence_limited": bool(excluded.any()),
            "whole_source_unusable_claimed": False}


def generate_draft(row, manifest, data, mask):
    # v1 generation uses pose/time arrays directly; binding sees ORIGINAL fields.
    draft = v1.generate_draft(row, manifest, data, mask)
    draft["schema"] = SCHEMA
    schema = rules.scalar_text(data, "schema")
    draft["source"]["adapter"] = {
        "name": "ms_fullframe_pose2d_draft_v2",
        "input_schema": schema,
        "validation_schema": VALIDATION_SCHEMA if schema == BOXES_SCHEMA else schema,
        "mapping": {"boxes": "bbox", "schema": VALIDATION_SCHEMA} if schema == BOXES_SCHEMA else {},
        "mapping_scope": "validation-only in-memory copies; generation and fullframe_binding use original NPZ arrays and manifest",
        "original_cache_rewritten": False,
        "extraction_recipe_equivalence_claimed": False,
        "original_bbox_field_claimed": schema != BOXES_SCHEMA,
        "decoder_timestamps_contract": "recorded native source-PTS [start,end] checked against actual_probe frame PTS and packet duration" if schema == BOXES_SCHEMA and "decoder_frame_timestamps" in data else None,
        "rule_mask_policy": "raw_valid & tracked & finite_native_pixel_in_frame; original arrays retained",
    }
    evidence = spatial_evidence(manifest, data, mask)
    draft["observation"]["spatial_evidence"] = evidence
    draft["source"]["provenance"]["local_spatial_evidence_limited"] = evidence["local_evidence_limited"]
    if evidence["local_evidence_limited"]:
        draft["status"] = "partial" if draft["events"] else "needs_regeneration"
        draft["limitations"].append("Some raw-effective finite predictions are outside the native image and excluded only from rule/event support; unaffected local evidence remains usable. Original xy, valid and binding are unchanged.")
    return draft


def process_row(row, out):
    check_row(row)
    path = Path(out) / row["split"] / row["id"] / "draft.json"
    if path.exists():
        return {"id": row["id"], "split": row["split"], "status": "existing_not_overwritten", "path": str(path)}
    started = time.perf_counter()
    manifest, data, mask = load_observations(row)
    loaded = time.perf_counter()
    draft = generate_draft(row, manifest, data, mask)
    generated = time.perf_counter()
    try:
        rules.write_new(path, draft)
    except FileExistsError:
        return {"id": row["id"], "split": row["split"], "status": "existing_not_overwritten", "path": str(path)}
    finished = time.perf_counter()
    return {"id": row["id"], "split": row["split"], "status": draft["status"], "path": str(path), **FLAGS,
            "input_schema": draft["source"]["input_schema"], "event_count": len(draft["events"]),
            "window_count": len(draft["windows"]), "observation": draft["observation"], "ms_text": draft["ms_text"],
            "event_boundaries": [{"event_id": e["event_id"], "start": e["start"], "end": e["end"],
                                  "signal": e["support"]["signal"], "support_frame_count": len(e["support"]["frame_indices"])} for e in draft["events"]],
            "profile_seconds": {"read_and_validate": loaded - started, "generate": generated - loaded,
                                "write_back": finished - generated, "total": finished - started}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Explicit JSONL: id/split/npz_path/fullframe_manifest_path")
    parser.add_argument("--out", required=True, help="Write-once OUT/SPLIT/ID/draft.json")
    parser.add_argument("--limit", type=int, default=3, help="Maximum sources, default 3; increase explicitly")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive; no implicit all-sources mode")
    started = time.perf_counter()
    attempted = written = existing = errors = events = 0
    statuses = {"draft_generated": 0, "partial": 0, "needs_regeneration": 0}
    seen = set()
    with Path(args.manifest).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if attempted >= args.limit:
                break
            attempted += 1
            row_started = time.perf_counter()
            row = None
            try:
                row = check_row(json.loads(line))
                if row["id"] in seen:
                    raise ValueError("Duplicate source ID, including across train/val splits")
                seen.add(row["id"])
                result = process_row(row, args.out)
                if result["status"] == "existing_not_overwritten":
                    existing += 1
                else:
                    written += 1
                    events += result["event_count"]
                    statuses[result["status"]] += 1
                print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
            except (ValueError, KeyError, TypeError, OSError, EOFError, zipfile.BadZipFile) as error:
                errors += 1
                print(json.dumps({"id": row.get("id") if isinstance(row, dict) else None,
                                  "split": row.get("split") if isinstance(row, dict) else None,
                                  "status": "rejected", "error": str(error),
                                  "profile_seconds": {"failed_e2e": time.perf_counter() - row_started},
                                  **FLAGS}, ensure_ascii=False), file=sys.stderr, flush=True)
    print(json.dumps({"summary": True, "sources_attempted": attempted, "sources_written": written,
                      "sources_existing": existing, "sources_rejected": errors, "status_counts": statuses,
                      "event_count": events, "elapsed_e2e_seconds": time.perf_counter() - started, **FLAGS,
                      "scope": "standalone CPU diagnostic; existing sources, no additional unique-source/production/ticket claims"}, ensure_ascii=False), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
