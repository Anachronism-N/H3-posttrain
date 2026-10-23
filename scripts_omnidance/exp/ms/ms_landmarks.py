"""Predict additional named landmarks from RGB through NLF, without SMPL fitting.

build-spec requires the NLF model's own SMPL-X canonical assets. extract reuses
cached person boxes, not cached pose geometry. describe consumes direct point
predictions, never the fixed-body FK used by ms_optimize.py.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from ms_optimize import atomic_write, json_bytes, overlaps, save_npz, sha256
from ms_visual_observe import strict_json


SCHEMA = "motionscript_direct_landmarks_v1"
NLF_SOURCE = "https://github.com/isarandi/nlf/blob/29046738a8101c15dc366ab45945b45e40debf17/nlf/pt/multiperson/multiperson_model.py"
SMPLX_SOURCE = "https://github.com/vchoutas/smplx/tree/1265df7ba545e8b00f72e7c557c766e15c71632f/smplx"
BODY_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
# Factual anatomical indices in the official SMPL-X topology; no body assets
# or canonical coordinate values are bundled or synthesized here.
FOOT_VERTICES = (
    ("left_big_toe", 5770), ("left_little_toe", 5780), ("left_heel", 8846),
    ("right_big_toe", 8463), ("right_little_toe", 8474), ("right_heel", 8635),
)
PALM_JOINTS = (
    ("left_index_mcp", 25), ("left_little_mcp", 31),
    ("right_index_mcp", 40), ("right_little_mcp", 46),
)


def build_smplx_spec(joints, vertices, *, source=None):
    """Index user-supplied NLF canonical arrays; do not substitute a T-pose."""
    from motionscript_v2.landmarks import validate_point_spec

    joints, vertices = np.asarray(joints), np.asarray(vertices)
    if (joints.shape != (55, 3) or vertices.shape != (10475, 3)
            or joints.dtype.kind not in "fiu" or vertices.dtype.kind not in "fiu"
            or not np.isfinite(joints).all() or not np.isfinite(vertices).all()):
        raise ValueError("need finite NLF SMPL-X canonical joints [55,3] and vertices [10475,3], in metres")
    rows = [(name, "joint", i) for i, name in enumerate(BODY_NAMES)]
    rows.extend((name, "joint", index) for name, index in PALM_JOINTS)
    rows.extend((name, "vertex", index) for name, index in FOOT_VERTICES)
    spec = {
        "schema": "motionscript_landmark_spec_v1", "units": "m",
        "canonical_frame": "nlf_canonical_smplx",
        "purpose": "body22 plus palm-plane and foot-surface anchors; no finger articulation or fingertips",
        "points": [{"name": name, "canonical": (joints if kind == "joint" else vertices)[index].tolist()}
                   for name, kind, index in rows],
        "source": {**(source or {}), "mapping_reference": SMPLX_SOURCE,
                   "point_origins": [{"name": name, "kind": kind, "index": index} for name, kind, index in rows],
                   "canonical_asset_identity": "supplied_by_operator; hashes identify bytes, not official authenticity",
                   "not_generic_smplx_rest_pose": True},
    }
    validate_point_spec(spec)
    return spec


def probe_video(path, ffprobe="ffprobe"):
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
        "-show_entries", "stream=width,height,avg_frame_rate,start_time,duration:stream_tags=rotate:stream_side_data=rotation:frame=best_effort_timestamp_time,duration_time,pkt_duration_time",
        "-of", "json", str(path)], check=True, capture_output=True)
    value = strict_json(result.stdout)
    if len(value.get("streams", [])) != 1:
        raise ValueError("cannot probe the first video stream")
    stream = value["streams"][0]
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    if rotation % 360 or any(float(row.get("rotation", 0)) % 360 for row in stream.get("side_data_list", [])):
        raise ValueError("normalize video rotation and regenerate matching boxes before extraction")
    time = np.asarray([float(row["best_effort_timestamp_time"]) for row in value["frames"]], np.float64)
    if not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("decoded video timestamps must be finite and strictly increasing")
    offset = float(time[0])
    time -= offset
    try:
        fps = float(Fraction(stream["avg_frame_rate"]))
    except (ValueError, ZeroDivisionError):
        fps = 0.
    if not np.isfinite(fps) or fps <= 0:
        if len(time) < 2:
            raise ValueError("cannot determine source frame rate")
        fps = float(1 / np.median(np.diff(time)))
    media_end, end_source = None, "unavailable; do not invent a final frame duration"
    for key in ("duration_time", "pkt_duration_time"):
        try:
            duration = float(value["frames"][-1].get(key, 0))
        except (ValueError, TypeError):
            continue
        if np.isfinite(duration) and duration > 0:
            media_end, end_source = float(time[-1] + duration), "last decoded frame " + key
            break
    if media_end is None:
        try:
            stop = float(stream["start_time"]) + float(stream["duration"]) - offset
        except (KeyError, ValueError, TypeError):
            stop = np.nan
        if np.isfinite(stop) and stop > time[-1]:
            media_end, end_source = float(stop), "video stream start_time + duration - first frame PTS"
    return {"time": time, "fps": fps, "width": int(stream["width"]),
            "height": int(stream["height"]), "pts_offset": offset,
            "media_end": media_end, "media_end_source": end_source}


def decode_batches(path, info, batch_size, ffmpeg="ffmpeg"):
    """Decode v:0 without resize/autorotation; retain every probed frame."""
    size = info["height"] * info["width"] * 3
    if size <= 0 or batch_size < 1:
        raise ValueError("invalid video dimensions or batch size")
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([
            ffmpeg, "-nostdin", "-v", "error", "-noautorotate", "-i", str(path),
            "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
        ], stdout=subprocess.PIPE, stderr=errors)
        try:
            batch, count = [], 0
            while True:
                data = bytearray()
                while len(data) < size:
                    chunk = process.stdout.read(size - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                if not data:
                    break
                if len(data) != size:
                    raise ValueError("truncated RGB frame")
                batch.append(np.frombuffer(data, np.uint8).reshape(info["height"], info["width"], 3))
                count += 1
                if len(batch) == batch_size:
                    yield count - len(batch), np.stack(batch)
                    batch = []
            if batch:
                yield count - len(batch), np.stack(batch)
            if process.wait(timeout=30) != 0:
                errors.seek(0)
                raise ValueError("FFmpeg decode failed: " + errors.read(2000).decode("utf-8", errors="replace"))
            if count != len(info["time"]):
                raise ValueError("decoded frame count differs from timestamp probe")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)


def validate_boxes(cache, info):
    box, valid = np.asarray(cache["box"]), np.asarray(cache["valid"])
    n = len(info["time"])
    if box.shape not in ((n, 4), (n, 5)) or valid.shape != (n,) or valid.dtype.kind != "b":
        raise ValueError("cached box/valid shape must match every decoded source frame")
    if (not np.isfinite(box[valid, :4]).all() or np.any(box[valid, 2:4] <= 0)):
        raise ValueError("valid person boxes must have finite XYWH and positive sizes")
    if "timestamps" in cache:
        stamps = np.asarray(cache["timestamps"], np.float64)
        if stamps.shape != (n,) or not np.allclose(stamps, info["time"], rtol=0, atol=1e-5):
            raise ValueError("box timestamps differ from source video")
    else:
        fps = np.asarray(cache.get("fps"))
        if fps.ndim != 0 or fps.dtype.kind not in "fiu" or not np.isfinite(fps) or not np.isclose(float(fps), info["fps"], rtol=.001):
            raise ValueError("legacy box cache frame rate differs from source video")
    return box[:, :4].astype(np.float32), valid.copy()


def unpack_predictions(result, counts, point_count):
    """Validate direct nonparametric output; coordinates are mm, unc stays native."""
    def numpy(value):
        return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)

    if any(key not in result or len(result[key]) != len(counts) for key in ("poses3d", "poses2d", "uncertainties")):
        raise ValueError("NLF direct result needs per-frame poses3d, poses2d and uncertainties")
    points = np.full((len(counts), point_count, 3), np.nan, np.float32)
    pixels = np.full((len(counts), point_count, 2), np.nan, np.float32)
    uncertainty = np.full((len(counts), point_count), np.nan, np.float32)
    valid = np.zeros_like(uncertainty, bool)
    for i, count in enumerate(counts):
        p, q, u = numpy(result["poses3d"][i]), numpy(result["poses2d"][i]), numpy(result["uncertainties"][i])
        if p.shape != (count, point_count, 3) or q.shape != (count, point_count, 2) or u.shape != (count, point_count):
            raise ValueError("NLF direct result changed point order/count or person count")
        if count:
            points[i], pixels[i], uncertainty[i] = p[0] / 1000., q[0], u[0]
            valid[i] = np.isfinite(points[i]).all(axis=1) & np.isfinite(q[0]).all(axis=1) & np.isfinite(u[0]) & (u[0] >= 0)
    return points, uncertainty, valid, pixels


class NLFDirectPredictor:
    """The only model path: named query weights + RGB crops, never a SMPL call."""
    def __init__(self, model_path, spec, device, num_aug):
        import torch
        import torchvision  # noqa: F401 -- registers torchvision ops for TorchScript

        self.torch, self.device, self.num_aug = torch, device, num_aug
        self.model = torch.jit.load(str(model_path), map_location=device).eval()
        methods = ("get_weights_for_canonical_points", "estimate_poses_batched")
        if not all(hasattr(self.model, name) for name in methods):
            raise ValueError("checkpoint does not export NLF direct-point methods; no SMPL fallback is allowed")
        self.method_schemas = {name: str(getattr(self.model, name).schema) for name in methods}
        with torch.inference_mode():
            query = torch.tensor([row["canonical"] for row in spec["points"]], dtype=torch.float32, device=device)
            self.weights = self.model.get_weights_for_canonical_points(query)

    def __call__(self, frames, boxes, valid):
        torch = self.torch
        with torch.inference_mode():
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device)
            crops = [torch.as_tensor(box[None] if ok else np.empty((0, 4), np.float32),
                                     dtype=torch.float32, device=self.device)
                     for box, ok in zip(boxes, valid)]
            # No extrinsics are supplied: the direct output retains camera axes.
            return self.model.estimate_poses_batched(images, crops, self.weights, num_aug=self.num_aug)


def require_new_output(out, sources):
    if out.exists():
        raise ValueError("use a new output directory")
    if any(overlaps(out, path.parent) for path in sources):
        raise ValueError("output must be separate from all input directories")


def extract(args, predictor_factory=NLFDirectPredictor):
    from motionscript_v2.landmarks import validate_point_spec

    video, boxes_path, spec_path, model_path, out = [Path(p).resolve() for p in
        (args.video, args.boxes_nlf, args.spec, args.model, args.out)]
    sources = (video, boxes_path, spec_path, model_path)
    require_new_output(out, sources)
    if args.bs < 1 or args.num_aug < 1:
        raise ValueError("batch size and num_aug must be positive")
    source_hashes = {label: sha256(path) for label, path in zip(("video", "boxes", "spec", "model"), sources)}
    spec_bytes, box_bytes = spec_path.read_bytes(), boxes_path.read_bytes()
    if (hashlib.sha256(spec_bytes).hexdigest() != source_hashes["spec"]
            or hashlib.sha256(box_bytes).hexdigest() != source_hashes["boxes"]):
        raise ValueError("input changed during snapshot read")
    spec = strict_json(spec_bytes)
    validate_point_spec(spec)
    info = probe_video(video, args.ffprobe)
    with np.load(io.BytesIO(box_bytes), allow_pickle=False) as source:
        boxes, detection_valid = validate_boxes(dict(source), info)
    predictor = predictor_factory(model_path, spec, args.device, args.num_aug)
    n, k = len(info["time"]), len(spec["points"])
    points, uncertainty = np.full((n, k, 3), np.nan, np.float32), np.full((n, k), np.nan, np.float32)
    pixels = np.full((n, k, 2), np.nan, np.float32)
    valid = np.zeros((n, k), bool)
    for start, frames in decode_batches(video, info, args.bs, args.ffmpeg):
        stop = start + len(frames)
        support = detection_valid[start:stop]
        if not support.any():
            continue
        result = predictor(frames, boxes[start:stop], support)
        points[start:stop], uncertainty[start:stop], valid[start:stop], pixels[start:stop] = unpack_predictions(result, support.astype(int), k)
    for label, path in zip(("video", "boxes", "spec", "model"), sources):
        if sha256(path) != source_hashes[label]:
            raise ValueError(f"{label} changed during extraction")
    out.mkdir(parents=True)
    save_npz(out / "raw.npz", {
        "points_m": points, "points2d_px": pixels, "uncertainty": uncertainty, "valid": valid,
        "names": np.asarray([row["name"] for row in spec["points"]]), "timestamps": info["time"],
        "fps": np.asarray(info["fps"]), "coordinate_frame": np.asarray("camera_xright_ydown_zforward"),
        "units": np.asarray("m"), "box": boxes, "detection_valid": detection_valid,
    })
    manifest = {"schema": SCHEMA, "source": "nlf_direct_rgb_landmarks", "point_count": k,
        "source_paths": dict(zip(("video", "boxes", "spec", "model"), map(str, sources))),
        "source_sha256": source_hashes, "point_spec": spec, "raw_sha256": sha256(out / "raw.npz"),
        "model_method_schemas": predictor.method_schemas,
        "video_stream_selector": "0:v:0", "source_pts_offset_seconds": info["pts_offset"],
        "media_bounds": {"start": 0., "end": info["media_end"], "end_source": info["media_end_source"]},
        "frame_count": n, "frame_size": [info["width"], info["height"]],
        "num_aug": args.num_aug, "device": args.device, "default_camera_fov_degrees": 55.,
        "box_source_binding": "operator_supplied; frame count and timing checked; identity/media pairing not proven",
        "person_tracking_added": False, "smpl_fitting_used": False, "template_fk_used": False,
        "body_anchors_queried_together_with_extras": True,
        "query_set_can_change_body_predictions": True,
        "visibility_verified": False, "semantic_approval": False,
        "uncertainty_units": "NLF estimate_poses_batched native; not legacy joint_uncertainties scale or calibrated probability",
        "implementation_sha256": sha256(Path(__file__)), "api_reference": NLF_SOURCE}
    payload = json_bytes(manifest)
    atomic_write(out / "manifest.receipt.json", json_bytes({"manifest_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "manifest.json", payload)
    return manifest


def load_bundle(bundle):
    raw_manifest = (bundle / "manifest.json").read_bytes()
    receipt = strict_json((bundle / "manifest.receipt.json").read_bytes())
    if receipt.get("manifest_sha256") != hashlib.sha256(raw_manifest).hexdigest():
        raise ValueError("direct-landmark manifest was altered")
    manifest = strict_json(raw_manifest)
    raw = (bundle / "raw.npz").read_bytes()
    if manifest.get("schema") != SCHEMA or hashlib.sha256(raw).hexdigest() != manifest["raw_sha256"]:
        raise ValueError("direct-landmark raw cache changed or has the wrong schema")
    with np.load(io.BytesIO(raw), allow_pickle=False) as cache:
        arrays = dict(cache)
    expected_names = [row["name"] for row in manifest["point_spec"]["points"]]
    if list(arrays["names"]) != expected_names:
        raise ValueError("point names differ from the extraction query order")
    return arrays, manifest, hashlib.sha256(raw_manifest).hexdigest()


def describe(args):
    from motionscript_v2.landmarks import preprocess_landmarks
    from motionscript_v2.landmark_events import (
        extract_landmark_events, measure_landmark_signals, render_landmark_caption,
    )

    bundle, out = Path(args.bundle).resolve(), Path(args.out).resolve()
    require_new_output(out, [bundle / "raw.npz"])
    raw, source, manifest_sha = load_bundle(bundle)
    window_spec, window_sha = None, None
    if args.chunk_seconds is not None or args.windows is not None:
        from motionscript_v2.chunks import fixed_windows, validate_windows

        bounds = source["media_bounds"]
        if bounds["end"] is None:
            raise ValueError("chunk annotation needs a known media end, not just the last sampled timestamp")
        if args.windows is not None:
            window_bytes = Path(args.windows).read_bytes()
            window_sha = hashlib.sha256(window_bytes).hexdigest()
            window_spec = strict_json(window_bytes)
            video_digest = window_spec.get("video_sha256")
            if not isinstance(video_digest, str) or video_digest.lower() != source["source_sha256"]["video"]:
                raise ValueError("window manifest must bind the same source video SHA256")
        else:
            window_spec = {"schema": "motionscript_windows_v1", "strategy": "fixed",
                "windows": fixed_windows(bounds["start"], bounds["end"], args.chunk_seconds, args.min_tail_seconds),
                "video_sha256": source["source_sha256"]["video"],
                "source": {"chunk_seconds": args.chunk_seconds, "min_tail_seconds": args.min_tail_seconds}}
        window_spec = validate_windows(window_spec, media_start=bounds["start"], media_end=bounds["end"])
    config = {key: getattr(args, key) for key in
              ("fps", "max_gap_seconds", "smooth_seconds", "max_point_uncertainty")}
    arrays, metadata = preprocess_landmarks(raw, **config)
    arrays["raw_points2d_px"] = raw["points2d_px"]
    signal_arrays, signal_metadata = measure_landmark_signals(arrays)
    arrays.update(signal_arrays)
    events = extract_landmark_events(arrays, min_duration_seconds=args.min_duration_seconds,
                                    min_angle_change_degrees=args.min_angle_change_degrees)
    report = {"schema": SCHEMA, "source": "nlf_direct_rgb_landmarks", "metadata": metadata,
        "point_count": len(arrays["names"]), "names": arrays["names"].tolist(),
        "continuous_signals": signal_metadata,
        "events": events, "caption": render_landmark_caption(events),
        "semantic_approval": False, "new_information_accuracy_validated": False,
        "source_manifest_sha256": manifest_sha, "raw_sha256": source["raw_sha256"],
        "media_bounds": source["media_bounds"],
        "uncertainty_units": source["uncertainty_units"],
        "config": {**config, "min_duration_seconds": args.min_duration_seconds,
                   "min_angle_change_degrees": args.min_angle_change_degrees},
        "implementation_sha256": {name: sha256(Path(__file__).parent / name) for name in
            ("ms_landmarks.py", "motionscript_v2/landmarks.py", "motionscript_v2/landmark_events.py")},
        "limits": ["Extra points are image-conditioned estimates, not ground-truth observations.",
                   "Relative palm/foot geometry is not a complete dance action vocabulary.",
                   "No Omni/H3 training input was changed."]}
    if window_spec is not None:
        from motionscript_v2.chunks import slice_landmarks

        chunks = []
        for chunk_number, window in enumerate(window_spec["windows"], 1):
            chunk_id = f"chunk_{chunk_number:04d}"
            sliced, support = slice_landmarks(arrays, window["start"], window["end"])
            local_events = extract_landmark_events(sliced, min_duration_seconds=args.min_duration_seconds,
                                                   min_angle_change_degrees=args.min_angle_change_degrees)
            for event in local_events:
                event["id"] = chunk_id + "_" + event["id"]
                evidence = event["evidence"]
                evidence["sample_index_basis"] = "indices into this chunk's selected target-time rows"
                evidence["global_target_sample_start"] = support["selected_indices"][evidence["sample_start"]]
                evidence["global_target_sample_end"] = support["selected_indices"][evidence["sample_end"]]
            chunks.append({"id": chunk_id, "window": window, "support": support, "events": local_events,
                "evidence_caption": render_landmark_caption(local_events),
                "source": "nlf_direct_rgb_landmarks_chunk_candidates",
                "semantic_approval": False, "caption_is_training_target": False,
                "empty_events_do_not_mean_static_pose": True})
        report.update(window_spec=window_spec, windows_input_sha256=window_sha, chunks=chunks,
                      window_time_meaning="annotation window, not exact action boundary")
        report["implementation_sha256"]["motionscript_v2/chunks.py"] = sha256(Path(__file__).parent / "motionscript_v2/chunks.py")
    out.mkdir(parents=True)
    save_npz(out / "coords.npz", arrays)
    report["coords_sha256"] = sha256(out / "coords.npz")
    payload = json_bytes(report)
    atomic_write(out / "report.receipt.json", json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "report.json", payload)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-spec")
    p.add_argument("--canonical-joints", required=True)
    p.add_argument("--canonical-vertices", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("extract")
    for key in ("video", "boxes-nlf", "spec", "model", "out"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--num-aug", type=int, default=3)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p = sub.add_parser("describe")
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=20.)
    p.add_argument("--max-gap-seconds", type=float, default=.2)
    p.add_argument("--smooth-seconds", type=float, default=.025)
    p.add_argument("--max-point-uncertainty", type=float)
    p.add_argument("--min-duration-seconds", type=float, default=.15)
    p.add_argument("--min-angle-change-degrees", type=float, default=20.)
    window_args = p.add_mutually_exclusive_group()
    window_args.add_argument("--chunk-seconds", type=float)
    window_args.add_argument("--windows", help="External contiguous window manifest, bound to the source video SHA")
    p.add_argument("--min-tail-seconds", type=float, default=1.5)
    return ap


def main():
    args = parser().parse_args()
    if args.command == "build-spec":
        first, second, out = map(Path, (args.canonical_joints, args.canonical_vertices, args.out))
        if out.exists():
            raise ValueError("build-spec needs a new output file")
        joint_bytes, vertex_bytes = first.read_bytes(), second.read_bytes()
        result = build_smplx_spec(np.load(io.BytesIO(joint_bytes), allow_pickle=False),
            np.load(io.BytesIO(vertex_bytes), allow_pickle=False),
            source={"canonical_joints_sha256": hashlib.sha256(joint_bytes).hexdigest(),
                    "canonical_vertices_sha256": hashlib.sha256(vertex_bytes).hexdigest()})
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(out, json_bytes(result))
    else:
        result = extract(args) if args.command == "extract" else describe(args)
    print(json.dumps({"command": args.command, "point_count": result.get("point_count", len(result.get("points", []))),
                      "semantic_approval": False}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
