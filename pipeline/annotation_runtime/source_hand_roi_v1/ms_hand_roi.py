"""Independent native-RGB hand evidence, not a hand-shape classifier.

prepare is CPU/provenance only. extract requires real CUDA ONNX sessions. Raw
Wholebody scores are admission thresholds, never calibrated probabilities.
No old pose cache, interpolation, synthetic points, rotation or mirroring is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types

import numpy as np

from dwpose_full_extract import SCAIL, bbox_iou, joint_layout

SCHEMA = "ms_native_hand_roi_v1"
HANDS = {"left_hand": (92, 113, 7), "right_hand": (113, 134, 4)}
DEFAULT_PARAMS = {"score_threshold": .3, "min_points": 6, "min_iou": .3,
                  "padding": .35, "min_crop_side": 32, "min_hand_extent": 12.,
                  "min_blur_variance": 4., "wrist_alignment_factor": .75,
                  "max_hand_jump": 2., "sample_fps": 8., "max_frames": 32,
                  "num_threads": 1, "device": 0, "jpeg_quality": 95}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def snapshot(path):
    value = Path(path).stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns, "device": value.st_dev, "inode": value.st_ino}


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = snapshot(path)
    result = {"path": str(path), "sha256": sha256(path), "stat": before}
    if snapshot(path) != before:
        raise ValueError(f"File changed while hashing: {path}")
    return result


def verify_identity(value):
    path = Path(value["path"])
    if str(path.resolve(strict=True)) != value["path"] or snapshot(path) != value["stat"]:
        raise ValueError(f"Identity file changed: {path}")


def write_new(path, raw):
    """Publish completely written bytes without replacing any existing output."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json_new(path, value):
    write_new(path, json_bytes(value))


def validate_params(params):
    if set(params) != set(DEFAULT_PARAMS):
        raise ValueError("ROI parameter schema mismatch")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
           for v in params.values()):
        raise ValueError("ROI parameters must be finite numeric values")
    for key in ("min_points", "min_crop_side", "max_frames", "num_threads", "device", "jpeg_quality"):
        if not isinstance(params[key], int):
            raise ValueError(f"{key} must be integer")
    if not 1 <= params["min_points"] <= 21 or params["min_crop_side"] < 1:
        raise ValueError("Invalid support point/crop limits")
    if not 0 <= params["score_threshold"] < 1 or not 0 <= params["min_iou"] <= 1:
        raise ValueError("Invalid raw threshold/IoU")
    if (params["max_frames"] < 1 or params["sample_fps"] <= 0 or params["num_threads"] < 1
            or params["device"] < 0 or not 1 <= params["jpeg_quality"] <= 100):
        raise ValueError("Invalid sampling/runtime parameters")
    for key in ("padding", "min_hand_extent", "min_blur_variance", "wrist_alignment_factor", "max_hand_jump"):
        if params[key] < 0:
            raise ValueError(f"{key} cannot be negative")


def read_manifest(path, ids=None, limit=3):
    if limit < 0:
        raise ValueError("limit must be zero (all) or positive")
    identity = file_identity(path)
    rows, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        original = json.loads(line)
        row = original.get("manifest_record", original) if isinstance(original, dict) else None
        if not isinstance(row, dict) or any(key not in row for key in ("id", "video_path", "source_group")):
            raise ValueError(f"Manifest line {number} needs id/video_path/source_group")
        sid = row["id"]
        if not isinstance(sid, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sid) is None:
            raise ValueError("Unsafe manifest ID")
        if sid.casefold() in seen:
            raise ValueError(f"Duplicate manifest ID: {sid}")
        seen.add(sid.casefold())
        if not isinstance(row["video_path"], str) or not Path(row["video_path"]).is_absolute():
            raise ValueError("video_path must be an explicit absolute path")
        if not isinstance(row["source_group"], str) or not row["source_group"].strip():
            raise ValueError("source_group must be nonempty")
        rows.append({"id": sid, "manifest_record": row, "input_line": number,
                     "input_row_sha256": digest(original)})
    verify_identity(identity)
    if not rows:
        raise ValueError("Empty manifest")
    if ids is not None:
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("IDs must be nonempty and unique")
        missing = set(ids) - {row["id"] for row in rows}
        if missing:
            raise ValueError(f"IDs absent from manifest: {sorted(missing)}")
        rows = [row for row in rows if row["id"] in set(ids)]
    return (rows[:limit] if limit else rows), identity


def probe_video(path, ffprobe):
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
               "-show_entries", "frame=best_effort_timestamp,best_effort_timestamp_time,width,height:"
               "stream=width,height,time_base,duration,avg_frame_rate", "-of", "json", str(path)]
    value = json.loads(subprocess.run(command, check=True, capture_output=True).stdout)
    streams, frames = value.get("streams", []), value.get("frames", [])
    if len(streams) != 1 or not frames:
        raise ValueError("Probe requires one video stream and real decoded frames")
    stream = streams[0]
    times = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Actual video PTS must be finite, unique and strictly increasing")
    width, height = int(stream["width"]), int(stream["height"])
    if min(width, height) <= 0 or any((int(f.get("width", width)), int(f.get("height", height)))
                                  != (width, height) for f in frames):
        raise ValueError("Video dimensions must remain native and constant")
    ticks = [int(frame["best_effort_timestamp"]) if "best_effort_timestamp" in frame else None for frame in frames]
    return {"width": width, "height": height, "decoded_video_frames": len(times),
            "actual_video_pts": times, "actual_video_pts_ticks": ticks,
            "time_base": stream.get("time_base"), "actual_video_pts_sha256": digest({"times": times, "ticks": ticks}),
            "frame_times": [t - times[0] for t in times], "time_origin": times[0],
            "duration": stream.get("duration"), "avg_frame_rate": stream.get("avg_frame_rate"),
            "pts_source": "ffprobe decoded best_effort_timestamp_time"}


def select_frame_indices(probe, params, explicit=None):
    times = np.asarray(probe["frame_times"], dtype=np.float64)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Duplicate/nonfinite/nonmonotone PTS")
    if explicit is not None:
        if (not isinstance(explicit, list) or not explicit or
                any(not isinstance(i, int) or isinstance(i, bool) for i in explicit)):
            raise ValueError("Explicit source frame indices must be nonempty integers")
        if min(explicit) < 0 or max(explicit) >= len(times) or any(b <= a for a, b in zip(explicit, explicit[1:])):
            raise ValueError("Source indices must be unique, increasing and in bounds")
        return explicit
    chosen = []
    for target in np.arange(0., times[-1] + 1e-9, 1. / params["sample_fps"]):
        position = int(np.searchsorted(times, target))
        candidates = [i for i in (position - 1, position) if 0 <= i < len(times)]
        index = min(candidates, key=lambda i: (abs(times[i] - target), i))
        if not chosen or index != chosen[-1]:
            chosen.append(index)
        if len(chosen) == params["max_frames"]:
            break
    return chosen


def model_identity(scail=SCAIL):
    scail = Path(scail).resolve(strict=True)
    weights = scail / "pretrained_weights/DWPose"
    code = scail / "DWPoseProcess/dwpose"
    files = {"detector": file_identity(weights / "yolox_l.onnx"),
             "pose": file_identity(weights / "dw-ll_ucoco_384.onnx"),
             "wholebody": file_identity(code / "wholebody.py"),
             "onnxdet": file_identity(code / "onnxdet.py"),
             "onnxpose": file_identity(code / "onnxpose.py"),
             "dwpose_init": file_identity(code / "__init__.py"),
             "layout_source": file_identity(Path(__file__).with_name("dwpose_full_extract.py")),
             "roi_source": file_identity(__file__)}
    return {"scail_root": str(scail), "files": files, "sha256": digest(files),
            "joint_layout": joint_layout(), "full_hash_once_per_process": True,
            "upstream_package_init_executed": False}


def verify_model(identity):
    for value in identity["files"].values():
        verify_identity(value)


def prepare_source(row, manifest_identity, model_ref, params, ffprobe, ffmpeg, explicit=None):
    video = file_identity(row["manifest_record"]["video_path"])
    probe = probe_video(video["path"], ffprobe)
    verify_identity(video)
    indices = select_frame_indices(probe, params, explicit)
    frames = [{"frame_index": i, "time": probe["frame_times"][i],
               "source_timestamp": probe["actual_video_pts"][i],
               "source_pts": probe["actual_video_pts_ticks"][i]} for i in indices]
    source = {"schema": SCHEMA, "id": row["id"], "source": video,
              "video_sha256": video["sha256"], "source_group": row["manifest_record"]["source_group"],
              "input_row": row, "source_manifest": manifest_identity, "model_ref": model_ref,
              "params": params, "probe": probe, "selected_frames": frames,
              "tools": {"ffprobe": str(Path(ffprobe).resolve()), "ffmpeg": str(Path(ffmpeg).resolve())},
              "sampling_policy": "nearest real PTS to sample_fps targets, first max_frames; explicit overrides allowed",
              "native_rgb_only": True, "old_cache_reuse": False,
              "score_is_probability": False, "anatomical_side_guaranteed": False}
    source["binding_sha256"] = digest(source)
    return source


def ensure_prepared(out, source):
    directory = Path(out) / source["id"]
    directory.mkdir(exist_ok=True)
    path = directory / "source_manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != source:
            raise ValueError(f"Refusing resume with different source/SHA/PTS/model/params: {path}")
    else:
        if any(directory.iterdir()):
            raise ValueError(f"Unbound/orphan output directory: {directory}")
        write_json_new(path, source)
    receipt = {"schema": SCHEMA, "status": "prepared_not_extracted", "id": source["id"],
               "binding_sha256": source["binding_sha256"], "source_manifest_sha256": sha256(path),
               "cuda_loaded": False, "semantic_approval": False}
    receipt_path = directory / "prepare_receipt.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != receipt:
            raise ValueError("Prepare receipt mismatch")
    else:
        write_json_new(receipt_path, receipt)
    return directory


def point_valid(xy, scores, width, height, threshold=.3, tracked=True):
    xy, scores = np.asarray(xy), np.asarray(scores)
    return (tracked & np.isfinite(xy).all(axis=-1) & np.isfinite(scores) & (scores > threshold)
            & (xy[..., 0] >= 0) & (xy[..., 0] < width) & (xy[..., 1] >= 0) & (xy[..., 1] < height))


def select_raw_subject(result, previous, min_iou):
    """Same largest/IoU policy as DWPose; nonfinite points stay invalid locally."""
    points, scores, boxes = (np.asarray(value, dtype=np.float32) for value in result)
    if boxes.size == 0:
        return None, None, None, "no_detection"
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Invalid detector boxes")
    if points.shape != (len(boxes), 134, 2) or scores.shape != (len(boxes), 134):
        raise ValueError("Expected verified raw OpenPose Wholebody 134-point layout")
    if previous is None:
        index = int(np.argmax(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)))
    else:
        overlaps = bbox_iou(boxes, previous)
        index = int(np.argmax(overlaps))
        if overlaps[index] < min_iou:
            return None, None, None, "low_subject_iou"
    return points[index].copy(), scores[index].copy(), boxes[index].copy(), "tracked"


def hand_roi(rgb, xy, scores, part, *, tracked=True, params=None, previous_box=None):
    """Return a crop proposal or an explicit unknown; never fill absent support."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    validate_params(params)
    if part not in HANDS:
        raise ValueError("Expected left_hand/right_hand")
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("ROI requires native uint8 RGB")
    xy, scores = np.asarray(xy), np.asarray(scores)
    if xy.shape != (134, 2) or scores.shape != (134,):
        raise ValueError("ROI requires raw 134 points, not NLF or COCO133")
    h, w = rgb.shape[:2]
    start, end, body_wrist = HANDS[part]
    valid = point_valid(xy, scores, w, h, params["score_threshold"], tracked)
    hand_valid = valid[start:end]
    supported = np.flatnonzero(hand_valid) + start
    result = {"part": part, "status": "unknown", "reason": None, "crop_xyxy": None,
              "supportpointindices": supported.astype(int).tolist(),
              "valid_hand_points": int(hand_valid.sum()), "quality_is_probability": False,
              "wrist_alignment_px": None, "hand_bbox_jump_normalized": None,
              "original_hand_pixel_size": None, "blur_laplacian_variance": None,
              "body_wrist_index": body_wrist, "hand_wrist_index": start,
              "anatomical_side_guaranteed": False}
    if not tracked:
        result["reason"] = "subject_not_continuously_tracked"
        return result
    if not valid[body_wrist] or not valid[start]:
        result["reason"] = "missing_body_or_hand_wrist"
        return result
    if hand_valid.sum() < params["min_points"]:
        result["reason"] = "insufficient_in_frame_hand_points"
        return result
    points = xy[supported]
    low, high = points.min(axis=0), points.max(axis=0)
    extent = high - low
    result["original_hand_pixel_size"] = extent.astype(float).tolist()
    result["hand_support_xyxy"] = np.r_[low, high].astype(float).tolist()
    alignment = float(np.linalg.norm(xy[start] - xy[body_wrist]))
    result["wrist_alignment_px"] = alignment
    raw_scores = scores[supported].astype(float)
    result["scores_summary"] = {"raw_min": float(raw_scores.min()), "raw_mean": float(raw_scores.mean()),
                                "raw_median": float(np.median(raw_scores)),
                                "threshold": params["score_threshold"], "is_probability": False}
    if float(extent.min()) < params["min_hand_extent"]:
        result["reason"] = "native_hand_extent_too_small"
        return result
    if alignment > max(12., float(extent.max()) * params["wrist_alignment_factor"]):
        result["reason"] = "body_hand_wrist_misaligned"
        return result
    side = max(params["min_crop_side"], int(math.ceil(float(extent.max()) * (1. + 2. * params["padding"]))))
    side = min(side, w, h)
    if side < params["min_crop_side"] or side < float(extent.max()):
        result["reason"] = "image_too_small_for_supported_hand_region"
        return result
    center = (low + high) / 2.
    x0 = int(np.clip(math.floor(center[0] - side / 2.), 0, w - side))
    y0 = int(np.clip(math.floor(center[1] - side / 2.), 0, h - side))
    box = np.array([x0, y0, x0 + side, y0 + side], dtype=np.float64)
    if np.any(points < box[:2]) or np.any(points >= box[2:]):
        result["reason"] = "crop_cannot_contain_real_support"
        return result
    if previous_box is not None:
        previous = np.asarray(previous_box, dtype=np.float64)
        if previous.shape != (4,) or not np.isfinite(previous).all() or np.any(previous[2:] <= previous[:2]):
            raise ValueError("Invalid previous hand bbox")
        jump = float(np.linalg.norm((box[:2] + box[2:] - previous[:2] - previous[2:]) / 2.)
                     / max(previous[2:] - previous[:2]))
        result["hand_bbox_jump_normalized"] = jump
        if jump > params["max_hand_jump"]:
            result["reason"] = "hand_bbox_jump_not_supported"
            return result
    import cv2
    support = rgb[int(math.floor(low[1])):min(h, int(math.ceil(high[1])) + 1),
                  int(math.floor(low[0])):min(w, int(math.ceil(high[0])) + 1)]
    blur = float(cv2.Laplacian(cv2.cvtColor(support, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    result["blur_laplacian_variance"] = blur
    result["blur_is_heuristic_not_truth"] = True
    if blur < params["min_blur_variance"]:
        result["reason"] = "native_hand_blur_heuristic_unknown"
        return result
    result.update({"status": "supported", "reason": "native_rgb_region_with_raw_point_support",
                   "crop_xyxy": box.astype(int).tolist(), "source_frame_size": [w, h],
                   "ROI_to_source_affine": [[1., 0., float(x0)], [0., 1., float(y0)]],
                   "source_to_ROI_affine": [[1., 0., float(-x0)], [0., 1., float(-y0)]],
                   "crop_pixel_size": [side, side], "resized": False, "rotated": False, "mirrored": False,
                   "crop_adds_native_detail": False})
    return result


def save_crop(rgb, proposal, path, quality=95):
    if proposal["status"] != "supported" or proposal["crop_xyxy"] is None:
        raise ValueError("Unknown ROI cannot produce a crop")
    x0, y0, x1, y1 = proposal["crop_xyxy"]
    h, w = rgb.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Refusing crop outside native frame")
    import cv2
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    write_new(path, encoded.tobytes())
    return {"path": str(Path(path).resolve()), "sha256": sha256(path),
            "crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest()}


def load_verified_wholebody_module(identity):
    # The upstream package __init__ mutates KMP_DUPLICATE_LIB_OK. Import only the
    # verified inference files under an isolated package, without executing it.
    package_name = "_ms_hand_roi_local_dwpose"
    code_root = str(Path(identity["files"]["wholebody"]["path"]).parent)
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [code_root]
        sys.modules[package_name] = package
    elif list(sys.modules[package_name].__path__) != [code_root]:
        raise RuntimeError("Existing isolated DWPose package has different import source")
    name = package_name + ".wholebody"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, identity["files"]["wholebody"]["path"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    for suffix, key in (("wholebody", "wholebody"), ("onnxdet", "onnxdet"), ("onnxpose", "onnxpose")):
        module = importlib.import_module(package_name + "." + suffix)
        if str(Path(module.__file__).resolve()) != identity["files"][key]["path"]:
            raise RuntimeError(f"Unexpected imported DWPose source: {suffix}")
    return sys.modules[name]


def load_cuda_model(identity, params):
    """Use verified Wholebody inference with explicit sessions; change no env/weights."""
    verify_model(identity)
    if os.environ.get("DWPOSE_ALLOW_CPU") == "1":
        raise RuntimeError("CPU fallback is forbidden")
    import torch
    import onnxruntime as ort
    if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("Real torch CUDA and ONNX CUDAExecutionProvider are required")
    torch.cuda.init()
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    wholebody = load_verified_wholebody_module(identity)
    options = ort.SessionOptions()
    options.intra_op_num_threads = params["num_threads"]
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    providers = [("CUDAExecutionProvider", {"device_id": params["device"], "arena_extend_strategy": "kSameAsRequested",
                  "cudnn_conv_algo_search": "HEURISTIC", "do_copy_in_default_stream": "1"})]
    model = wholebody.Wholebody.__new__(wholebody.Wholebody)
    model.use_batch = False
    model.session_det = ort.InferenceSession(identity["files"]["detector"]["path"], sess_options=options, providers=providers)
    model.session_pose = ort.InferenceSession(identity["files"]["pose"]["path"], sess_options=options, providers=providers)
    for session in (model.session_det, model.session_pose):
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("ONNX CUDA session initialization failed; CPU is forbidden")
    verify_model(identity)
    return model


def open_native_reader(source, num_threads):
    import decord
    reader = decord.VideoReader(source["source"]["path"], ctx=decord.cpu(0), num_threads=num_threads, fault_tol=0)
    probe = source["probe"]
    if len(reader) != probe["decoded_video_frames"]:
        raise ValueError("Decoder frame count disagrees with actual ffprobe PTS")
    decoded = np.asarray(reader.get_frame_timestamp(list(range(len(reader)))))[:, 0]
    actual = np.asarray(probe["actual_video_pts"])
    relative = np.asarray(probe["frame_times"])
    if np.allclose(decoded, actual, rtol=0, atol=1e-4):
        basis = "decord timestamp equals actual ffprobe PTS"
    elif np.allclose(decoded, relative, rtol=0, atol=1e-4):
        basis = "decord relative timestamp plus recorded ffprobe origin equals actual PTS"
    else:
        raise ValueError("Decoder timestamps disagree with actual video PTS; no synthetic FPS mapping")
    verify_identity(source["source"])
    return reader, basis


def extract_source(directory, source, model):
    verify_model(source["model_ref"])
    verify_identity(source["source"])
    if (directory / "roi_manifest.json").exists() or (directory / "extract_receipt.json").exists():
        return validate_completed(directory, source)
    allowed = {"source_manifest.json", "prepare_receipt.json"}
    if any(path.name not in allowed for path in directory.iterdir()):
        raise ValueError("Partial/orphan extraction exists; refuse to overwrite")
    import cv2
    reader, timestamp_basis = open_native_reader(source, source["params"]["num_threads"])
    frames, proposals, previous, previous_hands = [], [], None, {}
    n = len(source["selected_frames"])
    xy = np.full((n, 134, 2), np.nan, np.float32)
    scores = np.full((n, 134), np.nan, np.float32)
    boxes = np.full((n, 4), np.nan, np.float32)
    tracked = np.zeros(n, np.bool_)
    valid = np.zeros((n, 134), np.bool_)
    reasons = []
    crop_dir = directory / "crops"
    crop_dir.mkdir()
    width, height = source["probe"]["width"], source["probe"]["height"]
    for t, frame in enumerate(source["selected_frames"]):
        rgb = reader[frame["frame_index"]].asnumpy()
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Decoded image is not native RGB at probed dimensions")
        record = dict(frame, source_frame_size=[width, height],
                      rawframe_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest())
        points, raw_scores, box, reason = select_raw_subject(model(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
                                                           previous, source["params"]["min_iou"])
        reasons.append(reason)
        if box is not None:
            xy[t], scores[t], boxes[t], tracked[t] = points, raw_scores, box, True
            previous = box
            valid[t] = point_valid(points, raw_scores, width, height, source["params"]["score_threshold"])
        record.update({"tracked": bool(tracked[t]), "subject_reason": reason,
                       "subject_bbox": box.astype(float).tolist() if box is not None else None})
        frames.append(record)
        for part in HANDS:
            proposal = hand_roi(rgb, xy[t], scores[t], part, tracked=bool(tracked[t]),
                                params=source["params"], previous_box=previous_hands.get(part))
            proposal.update({key: record[key] for key in ("frame_index", "time", "source_timestamp", "source_pts", "rawframe_sha256")})
            if proposal["status"] == "supported":
                path = crop_dir / f"f{frame['frame_index']:06d}_{part}.jpg"
                proposal.update(save_crop(rgb, proposal, path, source["params"]["jpeg_quality"]))
                previous_hands[part] = proposal["crop_xyxy"]
            proposals.append(proposal)
    verify_identity(source["source"])
    verify_model(source["model_ref"])
    raw_path = directory / "raw_wholebody134.npz"
    with tempfile.SpooledTemporaryFile() as stream:
        np.savez_compressed(stream, xy=xy, score=scores, valid=valid, tracked=tracked, boxes=boxes,
                            left_hand_xy=xy[:, 92:113], right_hand_xy=xy[:, 113:134],
                            left_hand_score=scores[:, 92:113], right_hand_score=scores[:, 113:134],
                            left_hand_valid=valid[:, 92:113], right_hand_valid=valid[:, 113:134],
                            frame_indices=np.array([f["frame_index"] for f in frames], np.int64),
                            times=np.array([f["time"] for f in frames], np.float64),
                            source_timestamps=np.array([f["source_timestamp"] for f in frames], np.float64),
                            rawframe_sha256=np.array([f["rawframe_sha256"] for f in frames]),
                            subject_reason=np.array(reasons), binding_sha256=np.array(source["binding_sha256"]),
                            joint_layout=np.array(json.dumps(joint_layout(), sort_keys=True)),
                            video_sha256=np.array(source["video_sha256"]))
        stream.seek(0)
        write_new(raw_path, stream.read())
    manifest = {"schema": SCHEMA, "id": source["id"], "source": source["source"],
                "video_sha256": source["video_sha256"], "binding_sha256": source["binding_sha256"],
                "source_manifest": {"path": str((directory / "source_manifest.json").resolve()),
                                    "sha256": sha256(directory / "source_manifest.json")},
                "source_group": source["source_group"], "model_ref": source["model_ref"],
                "params": source["params"], "probe": source["probe"], "selected_frames": frames,
                "decoder_timestamp_basis": timestamp_basis,
                "raw": {"path": str(raw_path.resolve()), "sha256": sha256(raw_path)},
                "crops": [p for p in proposals if p["status"] == "supported"],
                "hand_observations": proposals, "quality_is_probability": False,
                "anatomical_side_guaranteed": False, "semantic_approval": False,
                "note": "Crops add no native detail and do not establish V/OK/heart. Two hands in one frame are one temporal observation."}
    write_json_new(directory / "roi_manifest.json", manifest)
    receipt = {"schema": SCHEMA, "id": source["id"], "binding_sha256": source["binding_sha256"],
               "status": "extracted_not_semantically_reviewed", "cuda_required": True,
               "roi_manifest_sha256": sha256(directory / "roi_manifest.json"), "raw_sha256": manifest["raw"]["sha256"],
               "selected_frame_count": n, "crop_count": len(manifest["crops"]), "semantic_approval": False}
    write_json_new(directory / "extract_receipt.json", receipt)
    return validate_completed(directory, source)


def validate_completed(directory, source):
    directory = Path(directory)
    manifest_path, receipt_path = directory / "roi_manifest.json", directory / "extract_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise ValueError("Incomplete extraction receipt/manifest")
    manifest, receipt = json.loads(manifest_path.read_text()), json.loads(receipt_path.read_text())
    if (manifest["binding_sha256"] != source["binding_sha256"] or receipt["binding_sha256"] != source["binding_sha256"]
            or manifest["video_sha256"] != source["video_sha256"] or manifest["model_ref"] != source["model_ref"]
            or receipt["roi_manifest_sha256"] != sha256(manifest_path)):
        raise ValueError("Completed source/model/manifest SHA mismatch")
    if sha256(directory / "source_manifest.json") != manifest["source_manifest"]["sha256"]:
        raise ValueError("Source manifest SHA mismatch")
    for asset in [manifest["raw"]] + manifest["crops"]:
        if sha256(asset["path"]) != asset["sha256"]:
            raise ValueError("Raw/crop content SHA mismatch")
    expected = source["selected_frames"]
    actual = [{key: f[key] for key in expected[0]} for f in manifest["selected_frames"]]
    if actual != expected:
        raise ValueError("Completed source frame/PTS mismatch")
    with np.load(manifest["raw"]["path"], allow_pickle=False) as raw:
        if (str(raw["binding_sha256"]) != source["binding_sha256"] or str(raw["video_sha256"]) != source["video_sha256"]
                or not np.array_equal(raw["frame_indices"], [f["frame_index"] for f in expected])
                or not np.array_equal(raw["source_timestamps"], [f["source_timestamp"] for f in expected])):
            raise ValueError("Raw source/frame/PTS binding mismatch")
    return manifest


def select_roi_evidence(manifest, start, end, requiredindices=None, max_per_hand=3):
    """CPU helper: at least two distinct real frames per hand, no hand-shape labels."""
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        source = json.loads(path.with_name("source_manifest.json").read_text())
        manifest = validate_completed(path.parent, source)
    if manifest.get("schema") != SCHEMA or not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Invalid ROI manifest/time interval")
    if max_per_hand not in (2, 3):
        raise ValueError("Select two or three distinct frames at most")
    if requiredindices is not None:
        if (any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in requiredindices)
                or len(requiredindices) != len(set(requiredindices))):
            raise ValueError("Required frame indices must be unique original integers")
        requiredindices = set(requiredindices)
    frames = manifest["selected_frames"]
    indices = [f["frame_index"] for f in frames]
    if len(indices) != len(set(indices)) or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("Duplicate/nonmonotone source frame indices")
    times = [f["time"] for f in frames]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Duplicate/nonmonotone real source PTS")
    lookup = {f["frame_index"]: f for f in frames}
    groups = {part: [] for part in HANDS}
    seen = set()
    for crop in manifest["crops"]:
        part, index = crop["part"], crop["frame_index"]
        if part not in HANDS or (part, index) in seen or index not in lookup:
            raise ValueError("Duplicate/unknown source crop binding")
        seen.add((part, index))
        frame = lookup[index]
        if any(crop.get(key) != frame.get(key) for key in ("time", "source_timestamp", "rawframe_sha256")):
            raise ValueError("Crop frame/time/raw RGB hash mismatch")
        if crop.get("status") != "supported" or crop.get("quality_is_probability") is not False:
            raise ValueError("Unknown or probability-shaped ROI is not evidence")
        if start <= frame["time"] < end and (requiredindices is None or index in requiredindices):
            if sha256(crop["path"]) != crop["sha256"]:
                raise ValueError("Evidence JPEG SHA mismatch")
            groups[part].append(crop)
    output = {}
    for part, candidates in groups.items():
        candidates.sort(key=lambda c: c["frame_index"])
        if len(candidates) < 2:
            output[part] = {"status": "unknown", "reason": "fewer_than_two_distinct_supported_source_frames",
                            "frame_indices": [], "crops": []}
            continue
        positions = np.linspace(0, len(candidates) - 1, min(max_per_hand, len(candidates)), dtype=int)
        selected = [candidates[int(i)] for i in positions]
        output[part] = {"status": "supported", "reason": "native_roi_evidence_only_not_hand_shape",
                        "frame_indices": [c["frame_index"] for c in selected], "crops": selected}
    output["distinct_source_frame_indices"] = sorted({i for part in HANDS for i in output[part]["frame_indices"]})
    output["same_frame_two_hands_is_one_temporal_observation"] = True
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "extract"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="Independent new output root; never a legacy cache directory")
    parser.add_argument("--ids", help="Existing ID text file or comma-separated exact IDs")
    parser.add_argument("--limit", type=int, default=3, help="Default three cases; 0 means all selected IDs")
    parser.add_argument("--frame-indices-json", help="JSON mapping each selected SID to actual source frame indices")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Recorded tool only; native decoding uses decord without scaling")
    parser.add_argument("--scail-root", default=str(SCAIL))
    for key, value in DEFAULT_PARAMS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    params = {key: getattr(args, key) for key in DEFAULT_PARAMS}
    validate_params(params)
    ids = None
    if args.ids:
        ids = [i.strip() for i in (Path(args.ids).read_text().splitlines() if Path(args.ids).is_file()
                                   else args.ids.split(",")) if i.strip()]
    rows, manifest_ref = read_manifest(args.manifest, ids, args.limit)
    frame_map = None
    if args.frame_indices_json:
        frame_map = json.loads(Path(args.frame_indices_json).read_text())
        if not isinstance(frame_map, dict) or any(row["id"] not in frame_map for row in rows):
            raise ValueError("Frame-index map must contain every selected SID")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {row["id"] for row in rows} for p in out.iterdir()):
        if any(not p.is_dir() or not (p / "source_manifest.json").is_file() for p in out.iterdir()):
            raise ValueError("Output root contains unrelated/legacy artifacts")
    model_ref = model_identity(args.scail_root)
    pending = []
    for row in rows:
        source = prepare_source(row, manifest_ref, model_ref, params, args.ffprobe, args.ffmpeg,
                                frame_map[row["id"]] if frame_map is not None else None)
        directory = ensure_prepared(out, source)
        if args.command == "extract":
            if (directory / "roi_manifest.json").exists() or (directory / "extract_receipt.json").exists():
                validate_completed(directory, source)
            else:
                pending.append((directory, source))
        print(f"[{args.command}] {row['id']} native_frames={len(source['selected_frames'])} binding={source['binding_sha256']}", flush=True)
    if args.command == "extract" and pending:
        model = load_cuda_model(model_ref, params)
        for directory, source in pending:
            manifest = extract_source(directory, source, model)
            print(f"[extracted] {source['id']} crops={len(manifest['crops'])} no_semantic_approval", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
