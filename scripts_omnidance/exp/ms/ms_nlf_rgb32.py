"""Fresh NLF detection and direct RGB32 queries, without legacy boxes or fitting."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

from ms_landmarks import SCHEMA, decode_batches, probe_video, require_new_output, unpack_predictions
from ms_optimize import atomic_write, json_bytes, save_npz, sha256
from ms_visual_observe import strict_json
from motionscript_v2.landmarks import validate_point_spec


def numpy(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def select_largest_detections(result, batch_count):
    keys = ("boxes", "poses3d", "poses2d", "uncertainties")
    if not isinstance(result, dict) or any(key not in result or len(result[key]) != batch_count for key in keys):
        raise ValueError("NLF detector output must align all four ragged fields to the RGB batch")
    selected = {key: [] for key in keys[1:]}
    counts, indices, boxes, all_boxes, all_points, all_pixels, all_uncertainty = [], [], [], [], [], [], []
    for t in range(batch_count):
        b, p, q, u = [numpy(result[key][t]) for key in keys]
        if b.ndim != 2 or b.shape[1] != 5:
            raise ValueError("NLF detection boxes must be [people,5] XYWH+score")
        count = len(b)
        if p.shape != (count, 32, 3) or q.shape != (count, 32, 2) or u.shape != (count, 32):
            raise ValueError("Detection counts and direct32 geometry do not match")
        if any(x.dtype.kind not in "fiu" for x in (b, p, q, u)):
            raise ValueError("Detection fields must be numeric arrays")
        if not np.isfinite(b).all() or np.any(b[:, 2:4] <= 0):
            raise ValueError("Detected XYWH+score must be finite with positive width/height")
        index = int(np.argmax(b[:, 2] * b[:, 3])) if count else -1
        counts.append(count)
        indices.append(index)
        boxes.append(b[index] if count else np.full(5, np.nan, np.float32))
        for key, values in zip(keys[1:], (p, q, u)):
            selected[key].append(values[index:index + 1] if count else values)
        all_boxes.append(b.astype(np.float32))
        all_points.append(p.astype(np.float32) / 1000.)
        all_pixels.append(q.astype(np.float32))
        all_uncertainty.append(u.astype(np.float32))
    points, uncertainty, valid, pixels = unpack_predictions(selected, np.asarray(counts) > 0, 32)
    return dict(points_m=points, points2d_px=pixels, uncertainty=uncertainty, valid=valid,
        box=np.asarray(boxes, np.float32), detection_valid=np.asarray(counts) > 0,
        detection_count=np.asarray(counts, np.int32), selected_index=np.asarray(indices, np.int32),
        all_boxes=np.concatenate(all_boxes), all_points_m=np.concatenate(all_points),
        all_points2d_px=np.concatenate(all_pixels), all_uncertainty=np.concatenate(all_uncertainty))


class FreshDetector32:
    def __init__(self, model, spec, device, num_aug, detector_threshold, detector_nms_iou_threshold):
        import torch
        import torchvision  # noqa: F401 -- registers TorchScript torchvision ops

        self.torch, self.device, self.num_aug = torch, device, num_aug
        self.model = torch.jit.load(str(model), map_location=device).eval()
        methods = ("get_weights_for_canonical_points", "detect_poses_batched")
        if not all(hasattr(self.model, name) for name in methods):
            raise ValueError("Model must export fresh detection + direct canonical queries")
        self.method_schemas = {name: str(getattr(self.model, name).schema) for name in methods}
        self.detector_threshold, self.nms = detector_threshold, detector_nms_iou_threshold
        with torch.inference_mode():
            query = torch.tensor([row["canonical"] for row in spec["points"]], dtype=torch.float32, device=device)
            self.weights = self.model.get_weights_for_canonical_points(query)

    def __call__(self, frames):
        torch = self.torch
        with torch.inference_mode():
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device)
            return self.model.detect_poses_batched(images, self.weights, num_aug=self.num_aug,
                detector_threshold=self.detector_threshold, detector_nms_iou_threshold=self.nms,
                max_detections=150, suppress_implausible_poses=True, extra_boxes=None)


def read_rows(path, limit):
    rows, seen = [], set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = strict_json(line.encode())
            if not isinstance(row, dict):
                raise ValueError("Input manifest rows must be objects")
            sid = row.get("id")
            if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sid) or sid in seen:
                raise ValueError("IDs must be unique safe identifiers")
            for key in ("video_path", "source_group", "split"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError("Input requires nonempty " + key)
            if not Path(row["video_path"]).is_absolute():
                raise ValueError("Input video paths must be absolute")
            if row["split"] not in ("train", "val", "test"):
                raise ValueError("Input split must be train/val/test")
            rows.append(row)
            seen.add(sid)
    if not rows or limit < 0:
        raise ValueError("Nonempty input and nonnegative limit required")
    return rows[:limit] if limit else rows


def extract_one(row, out, model, spec_path, spec, predictor, args, manifest_path, manifest_sha):
    video = Path(row["video_path"])
    require_new_output(out, [video, model, spec_path, manifest_path])
    inputs = {"video": video, "model": model, "spec": spec_path, "input_manifest": manifest_path}
    hashes = {key: sha256(path) for key, path in inputs.items()}
    if hashes["input_manifest"] != manifest_sha:
        raise ValueError("Input manifest changed before extraction")
    info = probe_video(video, args.ffprobe)
    n = len(info["time"])
    selected = {key: [] for key in ("points_m", "points2d_px", "uncertainty", "valid", "box", "detection_valid", "detection_count", "selected_index")}
    ragged = {key: [] for key in ("all_boxes", "all_points_m", "all_points2d_px", "all_uncertainty")}
    consumed = 0
    for start, frames in decode_batches(video, info, args.bs, args.ffmpeg):
        if start != consumed:
            raise ValueError("Decoded batches must preserve all source frames in order")
        result = select_largest_detections(predictor(frames), len(frames))
        for key in selected:
            selected[key].append(result[key])
        for key in ragged:
            ragged[key].append(result[key])
        consumed += len(frames)
    if consumed != n:
        raise ValueError("Actual decoded frames differ from probed PTS")
    arrays = {key: np.concatenate(values) for key, values in selected.items()}
    arrays.update({key: np.concatenate(values) for key, values in ragged.items()})
    arrays.update(names=np.asarray([row["name"] for row in spec["points"]]), timestamps=info["time"],
        fps=np.asarray(info["fps"]), coordinate_frame=np.asarray("camera_xright_ydown_zforward"), units=np.asarray("m"),
        detection_offsets=np.concatenate(([0], np.cumsum(arrays["detection_count"], dtype=np.int64))))
    for key, path in inputs.items():
        if sha256(path) != hashes[key]:
            raise ValueError(key + " changed during fresh extraction")
    out.mkdir(parents=True)
    save_npz(out / "raw.npz", arrays)
    report = dict(schema=SCHEMA, id=row["id"], source="nlf_fresh_detector_rgb_landmarks",
        source_group=row["source_group"], split=row["split"], input_record=row,
        source_paths={key: str(path) for key, path in inputs.items()}, source_sha256=hashes,
        point_count=32, point_spec=spec, raw_sha256=sha256(out / "raw.npz"),
        model_method_schemas=predictor.method_schemas, video_stream_selector="0:v:0",
        source_pts_offset_seconds=info["pts_offset"], media_bounds=dict(start=0., end=info["media_end"], end_source=info["media_end_source"]),
        frame_count=n, frame_size=[info["width"], info["height"]], num_aug=args.num_aug, bs=args.bs, device=args.device,
        detection_count=arrays["detection_count"].tolist(), selected_index=arrays["selected_index"].tolist(),
        selection_policy="per-frame largest XYWH area; argmax first tie; not tracking or guaranteed same identity",
        all_detection_boxes_and_direct_geometry_preserved=True, fresh_detector_used=True,
        no_external_boxes_supplied=True, external_pose_or_box_cache_conditioning=False,
        legacy_nlf_path_provenance_only=True, smpl_fitting_used=False, template_fk_used=False,
        body_anchors_queried_together_with_extras=True, default_camera_fov_degrees=55.,
        detector_config=dict(threshold=args.detector_threshold, nms_iou_threshold=args.detector_nms_iou_threshold,
            max_detections=150, suppress_implausible_poses=True, extra_boxes=None),
        detector_is_separate_network=True, crop_features_shared_across_32_queries=True,
        augmentation_crops_per_detection=args.num_aug, internal_batch_size=64,
        single_total_network_forward=False, query_set_can_change_body_predictions=True,
        person_tracking_added=False, zero_detections_keep_nan_mask=True, visibility_verified=False,
        semantic_approval=False, raw_is_ground_truth=False,
        uncertainty_units="NLF detect_poses_batched native; not calibrated visibility or legacy joint_uncertainties scale",
        implementation_sha256=sha256(Path(__file__)),
        dependency_sha256={name: sha256(Path(__file__).parent / name) for name in
            ("ms_landmarks.py", "ms_optimize.py", "ms_visual_observe.py", "motionscript_v2/landmarks.py")})
    payload = json_bytes(report)
    atomic_write(out / "manifest.receipt.json", json_bytes(dict(manifest_sha256=hashlib.sha256(payload).hexdigest())))
    atomic_write(out / "manifest.json", payload)
    return report


def run(args, factory=FreshDetector32):
    if args.bs < 1 or args.num_aug < 1 or not 0 <= args.detector_threshold <= 1 or not 0 <= args.detector_nms_iou_threshold <= 1:
        raise ValueError("Invalid batch, augmentation or detector settings")
    manifest, model, spec_path, out = map(lambda p: Path(p).resolve(), (args.manifest, args.model, args.spec, args.out))
    require_new_output(out, [manifest, model, spec_path])
    manifest_sha = sha256(manifest)
    rows = read_rows(manifest, args.limit)
    spec_bytes = spec_path.read_bytes()
    spec = strict_json(spec_bytes)
    model_sha = sha256(model)
    spec_sha = hashlib.sha256(spec_bytes).hexdigest()
    validate_point_spec(spec)
    if len(spec["points"]) != 32:
        raise ValueError("Exactly32 named canonical queries required")
    for row in rows:
        require_new_output(out, [Path(row["video_path"]), manifest, model, spec_path])
    predictor = factory(model, spec, args.device, args.num_aug, args.detector_threshold, args.detector_nms_iou_threshold)
    if sha256(model) != model_sha or sha256(spec_path) != spec_sha:
        raise ValueError("Model or spec changed while loading detector")
    reports = []
    for row in rows:
        if sha256(model) != model_sha or sha256(spec_path) != spec_sha:
            raise ValueError("Model or spec changed between clips")
        report = extract_one(row, out / "raw" / row["id"], model, spec_path, spec, predictor, args, manifest, manifest_sha)
        reports.append(dict(id=row["id"], manifest_path=str(out / "raw" / row["id"] / "manifest.json"),
            manifest_sha256=sha256(out / "raw" / row["id"] / "manifest.json"), raw_sha256=report["raw_sha256"],
            frame_count=report["frame_count"], detected_frames=sum(count > 0 for count in report["detection_count"]),
            multiple_person_frames=sum(count > 1 for count in report["detection_count"])))
        print(json.dumps(reports[-1]), flush=True)
    if sha256(manifest) != manifest_sha:
        raise ValueError("Input manifest changed")
    terminal = dict(schema="nlf-fresh-detector32-terminal-v1", complete=True, expected=len(rows), actual=len(reports),
        cases=reports, input_manifest_path=str(manifest), input_manifest_sha256=manifest_sha,
        no_external_boxes_supplied=True, person_tracking_added=False, smpl_fitting_used=False,
        template_fk_used=False, visibility_verified=False, semantic_approval=False,
        implementation_sha256=sha256(Path(__file__)))
    atomic_write(out / "finished.json", json_bytes(terminal))
    return terminal


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "spec", "model", "out"):
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--num-aug", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--detector-threshold", type=float, default=.3)
    ap.add_argument("--detector-nms-iou-threshold", type=float, default=.7)
    ap.add_argument("--ffprobe", default="/opt/ms_tools/ffprobe")
    ap.add_argument("--ffmpeg", default="/usr/local/bin/ffmpeg")
    return ap


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
