"""CPU-only, write-once, unapproved drafts from explicit native full-frame 2D NPZs.

Input JSONL: id, split, npz_path, fullframe_manifest_path. Each standalone manifest
has schema ms_fullframe_pose2d_manifest_v1, the same id/split/npz_path, video_path,
video_sha256, frame_count, width, height, pts_origin, duration_seconds,
frame_indices, times, source_timestamps, joint_layout, score_threshold=0.3,
diagnostic_not_production=true and the four approval/revision flags below.
Optional npz_sha256 is checked against the actual NPZ bytes; metadata_provenance
records the real ffprobe/model/receipt sources, not an old ROI binding.
Native frames must cover 0..frame_count-1. No media, weights, GPU, old ROI or bar
plan is accessed. Manifest consistency is not an independent media re-probe.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np

import ms_pose2d_draft as rules

MANIFEST_SCHEMA = "ms_fullframe_pose2d_manifest_v1"
SCHEMA = "ms_fullframe_pose2d_rule_draft_v1"
BINDING_SCHEMA = "ms_fullframe_pose2d_binding_v1"
INPUT_SCHEMAS = {"ms_fullvideo_pose2d_diagnostic_v1", "ms_fullvideo_pose2d_cache_v1"}
FLAGS = {**rules.FLAGS, "diagnostic_not_production": True}
REQUIRED = {"schema", "id", "split", "video_sha256", "joint_layout", "xy", "score",
            "valid", "tracked", "bbox", "frame_indices", "times", "source_timestamps", *FLAGS}


def check_row(row):
    if not isinstance(row, dict):
        raise ValueError("Manifest row must be an object")
    for key in ("id", "split", "npz_path", "fullframe_manifest_path"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Manifest requires nonempty {key}")
    for key in ("id", "split"):
        if re.fullmatch(r"[A-Za-z0-9_-]+", row[key]) is None:
            raise ValueError(f"Unsafe {key}; expected a bare identifier")
    for key in ("npz_path", "fullframe_manifest_path"):
        if not Path(row[key]).is_absolute():
            raise ValueError(f"{key} must be absolute (no implicit working-directory binding)")
        path = Path(row[key]).resolve()
        if path.parent.name != row["id"] or path.parent.parent.name != row["split"]:
            raise ValueError(f"{key} must belong to SPLIT/ID, not another source/split")
    if Path(row["npz_path"]).resolve().parent != Path(row["fullframe_manifest_path"]).resolve().parent:
        raise ValueError("Fullframe manifest and NPZ must share their source directory")
    if any(key in row for key in ("roi_manifest_path", "bar_plan_path")):
        raise ValueError("Old ROI/bar metadata is not accepted by the full-frame entrypoint")
    if "diagnostic_not_production" in row and row["diagnostic_not_production"] is not True:
        raise ValueError("This entrypoint is diagnostic only")
    return row


def finite_number(value, key):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite JSON number")
    return float(value)


def sha_text(value, key):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{key} must be a lowercase SHA256 digest")
    return value


def validate_observations(row, manifest, data):
    check_row(row)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Expected an explicit standalone fullframe manifest schema")
    for key in ("id", "split"):
        if manifest.get(key) != row[key] or rules.scalar_text(data, key) != row[key]:
            raise ValueError(f"Fullframe manifest/NPZ {key} mismatch")
    schema = rules.scalar_text(data, "schema")
    if schema not in INPUT_SCHEMAS:
        raise ValueError("Not an accepted fullvideo 2D schema; old ROI caches are not accepted")
    if manifest.get("input_schema", schema) != schema:
        raise ValueError("Fullframe manifest/NPZ input_schema mismatch")
    if not isinstance(manifest.get("npz_path"), str) or not Path(manifest["npz_path"]).is_absolute():
        raise ValueError("Fullframe manifest requires an absolute npz_path")
    if Path(manifest["npz_path"]).resolve() != Path(row["npz_path"]).resolve():
        raise ValueError("NPZ path is not the fullframe-bound path")
    video_path = manifest.get("video_path")
    if not isinstance(video_path, str) or not Path(video_path).is_absolute() or Path(video_path).stem != row["id"]:
        raise ValueError("Recorded video_path must be absolute and bound to the source ID")
    sha = sha_text(manifest.get("video_sha256"), "video_sha256")
    if rules.scalar_text(data, "video_sha256") != sha or row.get("video_sha256", sha) != sha:
        raise ValueError("Fullframe manifest/NPZ video_sha256 mismatch")
    if "npz_sha256" in manifest:
        sha_text(manifest["npz_sha256"], "npz_sha256")
    for key, expected in FLAGS.items():
        value = data[key]
        if manifest.get(key) is not expected or value.shape != () or value.dtype.kind != "b" or value.item() is not expected:
            raise ValueError(f"Unapproved diagnostic flag {key} mismatch")
    layout = json.loads(rules.scalar_text(data, "joint_layout"))
    if layout != rules.joint_layout() or manifest.get("joint_layout") != layout:
        raise ValueError("Fullframe/NPZ layout must equal the real pixel_xy Wholebody134 mapping")
    for key in ("frame_count", "width", "height"):
        if type(manifest.get(key)) is not int or manifest[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    n, width, height = (manifest[k] for k in ("frame_count", "width", "height"))
    specs = {"xy": ((n, 134, 2), "f"), "score": ((n, 134), "f"),
             "valid": ((n, 134), "b"), "tracked": ((n,), "b"), "bbox": ((n, 4), "f"),
             "frame_indices": ((n,), "iu"), "times": ((n,), "f"), "source_timestamps": ((n,), "f")}
    for key, (shape, kinds) in specs.items():
        if data[key].shape != shape or data[key].dtype.kind not in kinds:
            raise ValueError(f"Invalid {key} shape/dtype; full-frame xy must be [N,134,2]")
    indices = manifest.get("frame_indices")
    if not isinstance(indices, list) or any(type(v) is not int for v in indices) or indices != list(range(n)):
        raise ValueError("Native frame_indices must cover every frame 0..frame_count-1")
    if not np.array_equal(data["frame_indices"], indices):
        raise ValueError("NPZ/native full-frame indices mismatch or omitted frames")
    for key in ("times", "source_timestamps"):
        values = manifest.get(key)
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(f"Fullframe manifest requires {key}[frame_count]")
        expected = np.asarray([finite_number(v, key) for v in values], dtype=float)
        if not np.isfinite(data[key]).all() or np.any(np.diff(data[key]) <= 0) or np.any(np.diff(expected) <= 0):
            raise ValueError(f"{key} must be finite and strictly increasing")
        if not np.allclose(data[key], expected, atol=1e-6, rtol=0):
            raise ValueError(f"Fullframe manifest/NPZ native {key} mismatch")
    origin = finite_number(manifest.get("pts_origin"), "pts_origin")
    duration = finite_number(manifest.get("duration_seconds"), "duration_seconds")
    times = data["times"]
    if abs(float(times[0])) > 1e-6 or np.any(times < 0) or duration <= 0 or duration < float(times[-1]):
        raise ValueError("Full native relative PTS must start at zero and stay within video duration")
    if not np.allclose(data["source_timestamps"] - times, origin, atol=1e-6, rtol=0):
        raise ValueError("Native source PTS offset disagrees with pts_origin")
    threshold = finite_number(manifest.get("score_threshold"), "score_threshold")
    if threshold != 0.3:
        raise ValueError("This rule adapter requires the recorded score_threshold=0.3")
    for key in ("frame_count", "width", "height", "pts_origin", "duration_seconds", "score_threshold"):
        if key in data:
            value = data[key]
            if value.shape != () or value.dtype.kind not in "iuf" or not math.isclose(float(value), manifest[key], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Optional NPZ/fullframe {key} mismatch")
    effective = data["valid"] & data["tracked"][:, None]
    if not np.isfinite(data["xy"][effective]).all() or not np.isfinite(data["score"][effective]).all():
        raise ValueError("Effective valid points must have finite xy and raw score")
    if np.any(data["score"][effective] <= threshold):
        raise ValueError("Effective valid points contradict the recorded raw-score threshold")
    points = data["xy"][effective]
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        raise ValueError("Effective valid xy must be in native pixel bounds")
    if not np.isfinite(data["bbox"][data["tracked"]]).all():
        raise ValueError("Tracked bbox must be finite")
    # Invalid NaNs and raw scores >1 are kept, never padded or treated as probabilities.
    return effective & (data["score"] > threshold)


def load_observations(row):
    check_row(row)
    manifest = rules.read_json(row["fullframe_manifest_path"])
    raw = Path(row["npz_path"]).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if "npz_sha256" in manifest and manifest["npz_sha256"] != actual_sha:
        raise ValueError("Fullframe manifest/actual NPZ byte SHA256 mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if not REQUIRED.issubset(archive.files):
            raise ValueError(f"Missing full-frame NPZ fields: {sorted(REQUIRED - set(archive.files))}")
        data = {key: archive[key] for key in archive.files}
    mask = validate_observations(row, manifest, data)
    return manifest, data, mask


def fullframe_binding(manifest, data):
    observations = hashlib.sha256()
    for key in sorted(data):
        value = data[key]
        if value.dtype.hasobject:
            raise ValueError("Object arrays are never accepted")
        descriptor = json.dumps([key, value.dtype.str, list(value.shape)], separators=(",", ":")).encode("utf-8")
        raw = value.tobytes(order="C")
        for part in (descriptor, raw):
            observations.update(len(part).to_bytes(8, "big"))
            observations.update(part)
    payload = {"schema": BINDING_SCHEMA, "fullframe_manifest": manifest,
               "observations_sha256": observations.hexdigest()}
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"schema": BINDING_SCHEMA, "sha256": hashlib.sha256(raw).hexdigest(),
            "observations_sha256": observations.hexdigest(),
            "algorithm": "SHA256 canonical compact sorted-key UTF8 JSON of schema/fullframe_manifest/observations_sha256; observations are sorted fields, each descriptor [name,dtype.str,shape] and C-order bytes prefixed by 8-byte big-endian length",
            "scope": "provided fullframe metadata and actual NPZ arrays, not an ROI reference or independent media verification"}


def generate_draft(row, manifest, data, mask):
    binding = fullframe_binding(manifest, data)
    # Only an in-memory bridge for the frozen generator's duration/source keys.
    # This is not a fabricated ROI receipt and is never passed to its ROI validator.
    bridge_row = {"id": row["id"], "split": row["split"], "npz_path": row["npz_path"],
                  "roi_manifest_path": row["fullframe_manifest_path"]}
    bridge = {"probe": {"duration": manifest["duration_seconds"]},
              "video_sha256": manifest["video_sha256"], "binding_sha256": binding["sha256"]}
    draft = rules.generate_draft(bridge_row, bridge, data, mask)
    draft["schema"] = SCHEMA
    draft.update(FLAGS)
    draft["source"] = {"npz_path": str(Path(row["npz_path"]).resolve()),
                       "fullframe_manifest_path": str(Path(row["fullframe_manifest_path"]).resolve()),
                       "input_schema": rules.scalar_text(data, "schema"),
                       "video_path": manifest["video_path"], "video_sha256": manifest["video_sha256"],
                       "fullframe_binding": binding,
                       "provenance": {"metadata_provenance": manifest.get("metadata_provenance", {}),
                                      "verification_scope": "NPZ/fullframe manifest identity, native PTS, 2D layout, masks and full coverage consistency",
                                      "media_or_weights_accessed": False, "old_roi_receipt_used": False,
                                      "audio_video_alignment_verified": False, "beat_phase_verified": False,
                                      "motions_on_beats_verified": False}}
    draft["observation"].update({"native_width": manifest["width"], "native_height": manifest["height"],
                                 "native_pts_origin": manifest["pts_origin"],
                                 "native_frame_count": manifest["frame_count"], "all_native_frames_present": True,
                                 "score_threshold": manifest["score_threshold"], "score_is_probability": False})
    draft["limitations"].append("Full-frame metadata/NPZ consistency only; video duration tail beyond last PTS remains unknown. No new audio beat alignment.")
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
            "window_count": len(draft["windows"]), "observation": draft["observation"],
            "ms_text": draft["ms_text"],
            "event_boundaries": [{"event_id": e["event_id"], "start": e["start"], "end": e["end"],
                                  "signal": e["support"]["signal"], "support_frame_count": len(e["support"]["frame_indices"])} for e in draft["events"]],
            "profile_seconds": {"read_and_validate": loaded - started, "generate": generated - loaded,
                                "write_back": finished - generated, "total": finished - started}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Explicit JSONL: id/split/npz_path/fullframe_manifest_path")
    parser.add_argument("--out", required=True, help="Write-once OUT/SPLIT/ID/draft.json")
    parser.add_argument("--limit", type=int, default=3, help="Maximum sources, default 3; explicitly increase to process more")
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
                                  "status": "rejected", "error": str(error), **FLAGS}, ensure_ascii=False), file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - started
    print(json.dumps({"summary": True, "sources_attempted": attempted, "sources_written": written,
                      "sources_existing": existing, "sources_rejected": errors, "status_counts": statuses,
                      "event_count": events, "elapsed_e2e_seconds": elapsed, **FLAGS,
                      "scope": "standalone CPU diagnostic; no production or ticket budget allocation claims"}, ensure_ascii=False), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
