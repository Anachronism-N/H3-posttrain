"""Cache raw Wholebody 134-point observations for an explicit, fixed clip-ID list.

Low-score coordinates remain unmodified. Missing/rejected subjects are NaN;
valid[T,134] is score > 0.3 and tracked[T] is the bbox continuity mask.
Coverage is a score-threshold statistic, not calibrated confidence or labels.
GPU/video dependencies load only when the CLI has uncached clips to process.
"""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

SCAIL = Path("/apdcephfs_gy4/share_302533218/cedricnie/pose-preprocess/third_party/scail_pose")
VIDEO_ROOT = Path("/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/videos")
MAX_SECONDS = 81 / 16
SCORE_THRESHOLD = 0.3
MIN_IOU = 0.3
SCHEMA_VERSION = 1
PARTS = {
    "body": (0, 18),
    "foot": (18, 24),
    "face": (24, 92),
    "hand": (92, 134),
    "left_hand": (92, 113),
    "right_hand": (113, 134),
}


def joint_layout():
    names = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip",
        "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
        "right_eye", "left_eye", "right_ear", "left_ear",
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel",
    ]
    names += [f"face_{i:02d}" for i in range(68)]
    for side in ("left", "right"):
        names.append(f"{side}_hand_wrist")
        names += [f"{side}_{finger}_{i}" for finger in
                  ("thumb", "index", "middle", "ring", "little") for i in range(1, 5)]
    source = list(range(133))
    source.insert(17, None)
    original = source.copy()
    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    for target, origin in zip(openpose_idx, mmpose_idx):
        source[target] = original[origin]
    return {
        "name": "DWPose_Wholebody_134_neck_insert17_OpenPose_body_reorder",
        "coordinate_system": "pixel_xy",
        "joint_count": 134,
        "names": names,
        "source_coco_wholebody133_indices": source,
        "parts_half_open": {key: list(value) for key, value in PARTS.items()},
        "neck": {"index": 1, "derived_by_Wholebody": True,
                 "xy": "mean(COCO133 shoulders 5,6)",
                 "score": "float(left_shoulder_score > 0.3 and right_shoulder_score > 0.3)"},
    }


LAYOUT_JSON = json.dumps(joint_layout(), sort_keys=True, separators=(",", ":"))


def read_ids(path):
    ids = Path(path).read_text(encoding="utf-8").splitlines()
    ids = [sid.strip() for sid in ids if sid.strip()]
    if not ids:
        raise ValueError("The explicit ID list is empty")
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", sid) is None for sid in ids):
        raise ValueError("IDs must be bare clip IDs, not paths, extensions, or glob patterns")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip IDs would make sharding/outputs ambiguous")
    return ids


def bbox_iou(boxes, previous):
    overlap = np.maximum(0, np.minimum(boxes[:, 2:], previous[2:])
                         - np.maximum(boxes[:, :2], previous[:2]))
    intersection = overlap[:, 0] * overlap[:, 1]
    area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    previous_area = np.prod(previous[2:] - previous[:2])
    return intersection / (area + previous_area - intersection)


def select_subject(result, previous, min_iou=MIN_IOU):
    """Return (xy, score, box, reason): 0=matched, 1=no detections, 2=low IoU."""
    points, scores, detections = result
    boxes = np.asarray(detections, dtype=np.float32)
    # Upstream pose preprocessing invents a full-frame crop if detections are empty.
    if boxes.size == 0:
        return None, None, None, 1
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Unexpected detector boxes shape: {boxes.shape}")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Detector returned non-finite or degenerate boxes")
    points = np.asarray(points, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if points.shape != (len(boxes), 134, 2) or scores.shape != (len(boxes), 134):
        raise ValueError(f"Expected raw 134-point Wholebody output, got {points.shape}, {scores.shape}")
    if previous is None:
        index = int(np.argmax(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)))
    else:
        overlaps = bbox_iou(boxes, previous)
        index = int(np.argmax(overlaps))
        if overlaps[index] < min_iou:
            return None, None, None, 2
    if not np.isfinite(points[index]).all() or not np.isfinite(scores[index]).all():
        raise ValueError("Selected Wholebody observation contains non-finite coordinates/scores")
    return points[index].copy(), scores[index].copy(), boxes[index].copy(), 0


def pack_observations(results, fps, h, w, sid, min_iou=MIN_IOU):
    observations = list(results)
    n = len(observations)
    if n == 0:
        raise ValueError("Cannot cache an empty clip")
    xy = np.full((n, 134, 2), np.nan, dtype=np.float32)
    score = np.full((n, 134), np.nan, dtype=np.float32)
    boxes = np.full((n, 4), np.nan, dtype=np.float32)
    tracked = np.zeros(n, dtype=np.bool_)
    reasons = np.empty(n, dtype=np.uint8)
    previous = None
    for t, result in enumerate(observations):
        points, scores, box, reason = select_subject(result, previous, min_iou)
        reasons[t] = reason
        if box is None:
            continue
        xy[t], score[t], boxes[t] = points, scores, box
        tracked[t] = True
        previous = box
    return {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int32),
        "sid": np.array(sid),
        "xy": xy, "score": score, "boxes": boxes,
        "valid": tracked[:, None] & (score > SCORE_THRESHOLD),
        "tracked": tracked, "missing_reason": reasons,
        "fps": np.array(fps, dtype=np.float64),
        "h": np.array(h, dtype=np.int32), "w": np.array(w, dtype=np.int32),
        "joint_layout": np.array(LAYOUT_JSON),
        "frame_indices": np.arange(n, dtype=np.int64),
        "score_threshold": np.array(SCORE_THRESHOLD, dtype=np.float64),
        "min_iou": np.array(min_iou, dtype=np.float64),
        "max_seconds": np.array(MAX_SECONDS, dtype=np.float64),
    }


def validate_cache(data, sid):
    expected = {"schema_version", "sid", "xy", "score", "boxes", "valid", "tracked",
                "missing_reason", "fps", "h", "w", "joint_layout", "frame_indices",
                "score_threshold", "min_iou", "max_seconds"}
    if set(data) != expected:
        raise ValueError(f"Cache schema keys mismatch: {set(data) ^ expected}")
    for key in ("schema_version", "sid", "fps", "h", "w", "joint_layout",
                "score_threshold", "min_iou", "max_seconds"):
        if data[key].shape != ():
            raise ValueError(f"Cache {key} must be scalar")
    scalar_types = {"schema_version": np.int32, "fps": np.float64,
                    "h": np.int32, "w": np.int32, "score_threshold": np.float64,
                    "min_iou": np.float64, "max_seconds": np.float64}
    for key, dtype in scalar_types.items():
        if data[key].dtype != np.dtype(dtype):
            raise ValueError(f"Cache {key} scalar dtype mismatch")
    if any(data[key].dtype.kind != "U" for key in ("sid", "joint_layout")):
        raise ValueError("Cache ID/layout must be non-pickled Unicode scalars")
    if int(data["schema_version"]) != SCHEMA_VERSION or str(data["sid"]) != sid:
        raise ValueError("Cache version/clip ID mismatch")
    if str(data["joint_layout"]) != LAYOUT_JSON:
        raise ValueError("Cache is not the verified raw Wholebody 134-point layout")
    if float(data["score_threshold"]) != SCORE_THRESHOLD or float(data["min_iou"]) != MIN_IOU:
        raise ValueError("Cache threshold/tracking settings differ from this entrypoint")
    if float(data["max_seconds"]) != MAX_SECONDS:
        raise ValueError("Cache duration limit mismatch")
    fps = float(data["fps"])
    n = len(data["frame_indices"])
    if not math.isfinite(fps) or fps <= 0 or n == 0 or n > int(math.floor(MAX_SECONDS * fps + 1e-6)):
        raise ValueError("Cache has invalid FPS or exceeds the fixed duration limit")
    if int(data["h"]) <= 0 or int(data["w"]) <= 0:
        raise ValueError("Cache has invalid dimensions")
    specs = {"xy": ((n, 134, 2), np.float32), "score": ((n, 134), np.float32),
             "boxes": ((n, 4), np.float32), "valid": ((n, 134), np.bool_),
             "tracked": ((n,), np.bool_), "missing_reason": ((n,), np.uint8),
             "frame_indices": ((n,), np.int64)}
    for key, (shape, dtype) in specs.items():
        if data[key].shape != shape or data[key].dtype != np.dtype(dtype):
            raise ValueError(f"Cache {key} shape/dtype mismatch")
    if not np.array_equal(data["frame_indices"], np.arange(n, dtype=np.int64)):
        raise ValueError("Cache must retain contiguous original native frame indices")
    tracked = data["tracked"]
    if np.any(data["missing_reason"] > 2) or not np.array_equal(tracked, data["missing_reason"] == 0):
        raise ValueError("Cache frame missing reasons disagree with the tracking mask")
    for key in ("xy", "score", "boxes"):
        if not np.isfinite(data[key][tracked]).all() or not np.isnan(data[key][~tracked]).all():
            raise ValueError(f"Cache {key} must be finite on tracked frames and NaN on missing frames")
    if np.any(data["boxes"][tracked, 2:] <= data["boxes"][tracked, :2]):
        raise ValueError("Cache contains degenerate selected boxes")
    if not np.array_equal(data["valid"], tracked[:, None] & (data["score"] > SCORE_THRESHOLD)):
        raise ValueError("Cache validity mask is not the specified score-threshold mask")


def load_cache(path, sid):
    try:
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        validate_cache(data, sid)
    except Exception as error:
        raise RuntimeError(f"Invalid existing cache {path}: {error}") from error
    return data


def coverage_report(data):
    n = len(data["tracked"])
    matched = int(data["tracked"].sum())
    report = {
        "schema_version": SCHEMA_VERSION, "sid": str(data["sid"]),
        "frames": n, "tracked_frames": matched,
        "missing_frames": {"no_detection": int((data["missing_reason"] == 1).sum()),
                           "low_iou": int((data["missing_reason"] == 2).sum())},
        "fps": float(data["fps"]), "h": int(data["h"]), "w": int(data["w"]),
        "score_threshold": SCORE_THRESHOLD, "min_iou": float(data["min_iou"]),
        "note": "Raw score thresholds only, not calibrated confidence, action labels, or identity verification. "
                "Wholebody neck is a shoulder-derived point with a binary score.",
        "tracking": "First detected frame: largest box; later: highest IoU against last accepted box. "
                    "No interpolation, prediction, or reinitialization after rejection.",
        "parts": {},
    }
    for part, (start, end) in PARTS.items():
        scores = data["score"][:, start:end]
        valid = data["valid"][:, start:end]
        observed = scores[data["tracked"]].reshape(-1)
        count = int(valid.sum())
        report["parts"][part] = {
            "joint_indices_half_open": [start, end], "joint_count": end - start,
            "valid_observations": count, "possible_observations": n * (end - start),
            "score_coverage_all_frames": count / (n * (end - start)),
            "score_coverage_tracked_frames": count / (matched * (end - start)) if matched else None,
            "frames_any_valid": int(valid.any(axis=1).sum()),
            "frames_all_valid": int(valid.all(axis=1).sum()),
            "per_joint_valid_count": valid.sum(axis=0).astype(int).tolist(),
            "raw_score_mean_tracked": float(observed.mean()) if matched else None,
            "raw_score_quantiles_tracked": np.quantile(observed, [0.1, 0.5, 0.9]).tolist() if matched else None,
        }
    return report


def atomic_write(path, writer):
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output parent does not exist: {path.parent}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_coverage(path, data):
    report = coverage_report(data)
    if Path(path).exists():
        try:
            existing = json.loads(Path(path).read_text(encoding="utf-8"))
            if existing != report:
                raise ValueError("Coverage JSON does not match the validated cache")
        except Exception as error:
            raise RuntimeError(f"Invalid existing coverage {path}: {error}") from error
        return
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(encoded))


def load_wholebody(num_threads=1):
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    if os.environ.get("DWPOSE_ALLOW_CPU", "0") == "1":
        raise RuntimeError("DWPOSE_ALLOW_CPU=1 is forbidden for full-point extraction")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(num_threads)
    import torch  # Load the environment's CUDA libraries before ONNX Runtime initializes CUDA.
    torch.cuda.init()
    process_root = SCAIL / "DWPoseProcess"
    sys.path.insert(0, str(process_root))
    wholebody = importlib.import_module("dwpose.wholebody")
    if Path(wholebody.__file__).resolve() != (process_root / "dwpose/wholebody.py").resolve():
        raise RuntimeError("Imported Wholebody is not the verified local SCAIL implementation")
    original_weights = wholebody.ModelDataPathPrefix
    wholebody.ModelDataPathPrefix = SCAIL / "pretrained_weights"
    try:
        model = wholebody.Wholebody(0, use_batch=False)
    finally:
        wholebody.ModelDataPathPrefix = original_weights
    for session in (model.session_det, model.session_pose):
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("DWPose failed to initialize CUDA; CPU fallback is forbidden")
    return model


def extract_clip(path, model, sid, num_threads=1):
    import cv2
    import decord

    cv2.setNumThreads(num_threads)
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=num_threads, fault_tol=0)
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video has invalid native FPS: {path}")
    n = min(len(reader), int(math.floor(MAX_SECONDS * fps + 1e-6)))
    if n <= 0:
        raise ValueError(f"Video has no complete frames in the fixed time window: {path}")
    first = reader[0].asnumpy()
    h, w = first.shape[:2]

    def results():
        for index in range(n):
            rgb = first if index == 0 else reader[index].asnumpy()
            if rgb.shape != (h, w, 3):
                raise ValueError(f"Video dimensions/channels changed at frame {index}: {path}")
            yield model(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return pack_observations(results(), fps, h, w, sid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True, help="Text file containing one known bare clip ID per line")
    parser.add_argument("--out", required=True, help="Output directory; its parent must already exist")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.nshard < 1 or not 0 <= args.shard < args.nshard:
        parser.error("Require nshard > 0 and 0 <= shard < nshard")
    if args.num_threads < 1:
        parser.error("num_threads must be positive")
    ids = read_ids(args.ids)[args.shard::args.nshard]
    out = Path(args.out)
    if not out.parent.is_dir():
        raise FileNotFoundError(f"Output directory parent must already exist: {out.parent}")
    out.mkdir(exist_ok=True)
    pending = []
    for sid in ids:
        cache_path = out / f"{sid}.npz"
        coverage_path = out / f"{sid}.coverage.json"
        if cache_path.exists():
            data = load_cache(cache_path, sid)
            write_coverage(coverage_path, data)
            print(f"[validated] {sid}", flush=True)
        else:
            if coverage_path.exists():
                raise RuntimeError(f"Orphan existing coverage without NPZ: {coverage_path}")
            path = VIDEO_ROOT / f"{sid}.mp4"
            if not path.is_file():
                raise FileNotFoundError(f"Explicit clip ID has no normalized video: {path}")
            pending.append((sid, path))
    print(f"[dwpose-full] shard={args.shard}/{args.nshard} ids={len(ids)} pending={len(pending)}", flush=True)
    if not pending:
        return
    model = load_wholebody(args.num_threads)
    for sid, path in pending:
        data = extract_clip(path, model, sid, args.num_threads)
        validate_cache(data, sid)
        cache_path = out / f"{sid}.npz"
        atomic_write(cache_path, lambda stream: np.savez_compressed(stream, **data))
        verified = load_cache(cache_path, sid)
        write_coverage(out / f"{sid}.coverage.json", verified)
        print(f"[cached] {sid} frames={len(data['tracked'])} tracked={int(data['tracked'].sum())}", flush=True)


if __name__ == "__main__":
    main()
