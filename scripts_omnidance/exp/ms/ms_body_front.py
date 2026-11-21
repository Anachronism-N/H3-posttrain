"""Native NLF body candidates with source-bound frames, without music or LLMs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

from ms_landmarks import SCHEMA, decode_batches, load_bundle, probe_video, require_new_output
from ms_nlf_rgb32 import read_rows, select_largest_detections
from ms_optimize import atomic_write, json_bytes, save_npz, sha256, source_hashes
from motionscript_v2.body_events import extract_body_events, measure_body_signals, render_body_caption
from motionscript_v2.events import JOINT_NAMES
from motionscript_v2.landmarks import SPEC_SCHEMA, preprocess_landmarks, validate_point_spec
from motionscript_v2.selection import co_supported_groups, summarize_events


NATIVE_ALIASES = {
    "pelv": "pelvis", "lhip": "left_hip", "rhip": "right_hip", "spi1": "spine1",
    "lkne": "left_knee", "rkne": "right_knee", "spi2": "spine2", "lank": "left_ankle",
    "rank": "right_ankle", "spi3": "spine3", "neck": "neck", "lcla": "left_collar",
    "rcla": "right_collar", "head": "head", "lsho": "left_shoulder", "rsho": "right_shoulder",
    "lelb": "left_elbow", "relb": "right_elbow", "lwri": "left_wrist", "rwri": "right_wrist",
}


def validate_native_aliases(mapping):
    if (mapping.get("schema") != "nlf_native_body_mapping_v1"
            or mapping.get("mapping_version") != "smpl24-body20-no-toes-v1"
            or mapping.get("skeleton") != "smpl_24"):
        raise ValueError("Admitted SMPL24/body20 native mapping required")
    joints = mapping.get("joints")
    if not isinstance(joints, list) or len(joints) != len(NATIVE_ALIASES):
        raise ValueError("Exactly20 admitted native body joints required")
    if {row["model_name"] for row in joints} != set(NATIVE_ALIASES):
        raise ValueError("Mapping contains missing, duplicated, finger or toe native joints")
    if any(row["name"] != NATIVE_ALIASES[row["model_name"]] for row in joints):
        raise ValueError("Anatomical alias differs from admitted native model meaning")


class NativeBodyPredictor:
    def __init__(self, model_path, mapping, device, num_aug):
        validate_native_aliases(mapping)
        import torch
        import torchvision

        if mapping.get("schema") != "nlf_native_body_mapping_v1":
            raise ValueError("Native body mapping schema required")
        if sha256(model_path) != mapping.get("model_sha256"):
            raise ValueError("Mapping belongs to a different model")
        self.torch, self.device, self.num_aug = torch, device, num_aug
        self.model = torch.jit.load(str(model_path), map_location=device).eval()
        methods = ("get_weights_for_canonical_points", "detect_poses_batched")
        if not all(hasattr(self.model, name) for name in methods):
            raise ValueError("Official direct detection methods are absent")
        self.method_schemas = {name: str(getattr(self.model, name).schema) for name in methods}
        skeleton = mapping["skeleton"]
        model_names = list(self.model.per_skeleton_joint_names[skeleton])
        model_indices = self.model.per_skeleton_indices[skeleton].detach().cpu().tolist()
        joints = mapping["joints"]
        if not joints or len({row["name"] for row in joints}) != len(joints):
            raise ValueError("Nonempty unique body mapping required")
        if any(row["name"] not in JOINT_NAMES for row in joints):
            raise ValueError("Only body22/coarse foot names are admitted; no finger or toe queries")
        if len({row["model_index"] for row in joints}) != len(joints):
            raise ValueError("A native joint cannot supply multiple body landmarks")
        for row in joints:
            index = row["model_index"]
            if (type(index) is not int or not 0 <= index < len(model_names)
                    or row["model_name"] != model_names[index]
                    or type(row["canonical_index"]) is not int
                    or row["canonical_index"] != model_indices[index]):
                raise ValueError("Mapping differs from actual model joint names/indices")
        with torch.inference_mode():
            canonical = self.model.crop_model.canonical_locs()
            query = canonical[[row["canonical_index"] for row in joints]].to(device)
            self.weights = self.model.get_weights_for_canonical_points(query)
        self.spec = validate_point_spec({
            "schema": SPEC_SCHEMA, "units": "m", "canonical_frame": "nlf_learned_native_joints",
            "points": [{"name": row["name"], "canonical": point}
                       for row, point in zip(joints, query.detach().cpu().tolist())],
            "source": {"method": "actual_official_model_native_joint_table", "skeleton": skeleton,
                       "model_sha256": mapping["model_sha256"], "mapping": joints,
                       "canonical_sha256": hashlib.sha256(query.detach().cpu().numpy().tobytes()).hexdigest(),
                       "template_fk_used": False, "anatomy_mapping_requires_rgb_review": True}})
        self.point_count = len(joints)
        self.detector_threshold = .3
        self.nms = .7

    def __call__(self, frames):
        with self.torch.inference_mode():
            images = self.torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device)
            return self.model.detect_poses_batched(
                images, self.weights, num_aug=self.num_aug, detector_threshold=self.detector_threshold,
                detector_nms_iou_threshold=self.nms, max_detections=150,
                suppress_implausible_poses=True)


def extract_one(row, out, predictor, args, source_sha):
    video = Path(row["video_path"])
    require_new_output(out, [video, Path(args.model), Path(args.mapping), Path(args.manifest)])
    video_sha = sha256(video)
    if row.get("video_sha256") and row["video_sha256"] != video_sha:
        raise ValueError("Video differs from selected source media")
    info = probe_video(video, args.ffprobe)
    if not all(key in info for key in ("frame_indices", "pts", "time_base")):
        raise ValueError("Native integer PTS/time_base are required for body extraction")
    keys = ("points_m", "points2d_px", "uncertainty", "valid", "box", "detection_valid",
            "detection_count", "selected_index", "all_boxes", "all_points_m", "all_points2d_px", "all_uncertainty")
    values = {key: [] for key in keys}
    consumed = 0
    for start, frames in decode_batches(video, info, args.bs, args.ffmpeg):
        if start != consumed:
            raise ValueError("Source frame order changed")
        result = select_largest_detections(predictor(frames), len(frames), predictor.point_count)
        for key in keys:
            values[key].append(result[key])
        consumed += len(frames)
    if consumed != len(info["time"]):
        raise ValueError("Decoded RGB count differs from source PTS")
    raw = {key: np.concatenate(value) for key, value in values.items()}
    raw["positive_camera_depth"] = np.isfinite(raw["points_m"][:, :, 2]) & (raw["points_m"][:, :, 2] > 0)
    raw["valid"] &= raw["positive_camera_depth"]
    raw.update(names=np.asarray([point["name"] for point in predictor.spec["points"]]),
               timestamps=info["time"], frame_indices=info["frame_indices"], pts=info["pts"],
               time_base=info["time_base"], fps=np.asarray(info["fps"]),
               frame_size=np.array([info["width"], info["height"]]),
               coordinate_frame=np.asarray("camera_xright_ydown_zforward"), units=np.asarray("m"),
               detection_offsets=np.r_[0, np.cumsum(raw["detection_count"], dtype=np.int64)])
    if sha256(video) != video_sha:
        raise ValueError("Video changed during extraction")
    out.mkdir(parents=True)
    save_npz(out / "raw.npz", raw)
    report = {
        "schema": SCHEMA, "id": row["id"], "source_group": row["source_group"], "split": row["split"],
        "source": "nlf_official_native_rgb_body", "input_record": row,
        "source_paths": {"video": str(video), "model": args.model, "mapping": args.mapping,
                         "input_manifest": args.manifest},
        "source_sha256": {**source_sha, "video": video_sha}, "point_spec": predictor.spec,
        "point_count": predictor.point_count, "raw_sha256": sha256(out / "raw.npz"),
        "model_method_schemas": predictor.method_schemas, "frame_count": consumed,
        "frame_size": [info["width"], info["height"]], "source_pts_offset_seconds": info["pts_offset"],
        "media_bounds": {"start": 0., "end": info["media_end"], "end_source": info["media_end_source"]},
        "native_integer_pts_preserved": True, "num_aug": args.num_aug, "batch_size": args.bs,
        "selection_policy": "largest per-frame XYWH area; all raw detections retained; not identity tracking",
        "smpl_fitting_used": False, "template_fk_used": False, "hand_fingers_observed": False,
        "camera_frame_is_world": False, "default_camera_fov_degrees": 55.,
        "semantic_approval": False, "human_review_completed": False, "visibility_verified": False,
        "uncertainty_units": "NLF native direct-query values; not calibrated visibility",
        "implementation_sha256": sha256(Path(__file__)), "measurement_code_sha256": source_hashes(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "input_model_pose_units": "mm", "cached_pose_units": "m"}
    payload = json_bytes(report)
    atomic_write(out / "manifest.receipt.json", json_bytes({"manifest_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "manifest.json", payload)
    return report


def extract(args, factory=NativeBodyPredictor):
    if args.bs < 1 or args.num_aug < 1:
        raise ValueError("Positive batch size and augmentation count required")
    out = Path(args.out).resolve()
    sources = {key: Path(getattr(args, key)).resolve() for key in ("manifest", "model", "mapping")}
    require_new_output(out, list(sources.values()))
    hashes = {key: sha256(path) for key, path in sources.items()}
    rows = read_rows(sources["manifest"], args.limit)
    mapping = json.loads(sources["mapping"].read_text())
    predictor = factory(sources["model"], mapping, args.device, args.num_aug)
    if any(sha256(path) != hashes[key] for key, path in sources.items()):
        raise ValueError("Input changed while loading model")
    out.mkdir(parents=True)
    records = []
    for row in rows:
        try:
            report = extract_one(row, out / row["id"], predictor, args, hashes)
            record = {"id": row["id"], "status": "extracted", "bundle": str(out / row["id"]),
                      "raw_sha256": report["raw_sha256"], "frames": report["frame_count"]}
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            record = {"id": row["id"], "status": "failed", "error_type": type(error).__name__, "error": str(error)}
        records.append(record)
        atomic_write(out / "progress.json", json_bytes({"expected": len(rows), "records": records, "complete": False}))
        print(json.dumps(record), flush=True)
    if any(sha256(path) != hashes[key] for key, path in sources.items()):
        raise ValueError("Inputs changed during body run")
    terminal = {"schema": "native-body-extraction-v1", "source_sha256": hashes, "expected": len(rows),
                "records": records, "complete": True, "semantic_approval": False,
                "failures_in_denominator": True, "human_review_completed": False}
    atomic_write(out / "finished.json", json_bytes(terminal))
    return terminal


def describe(args):
    bundle, out = Path(args.bundle).resolve(), Path(args.out).resolve()
    require_new_output(out, [bundle / "raw.npz"])
    raw, source, manifest_sha = load_bundle(bundle)
    video = Path(source["source_paths"]["video"])
    if sha256(video) != source["source_sha256"]["video"]:
        raise ValueError("Source video changed")
    info = probe_video(video, args.ffprobe)
    if (not np.array_equal(raw["pts"], info["pts"])
            or not np.array_equal(raw["time_base"], info["time_base"])
            or not np.array_equal(raw["frame_indices"], info["frame_indices"])
            or not np.allclose(raw["timestamps"], info["time"], rtol=0, atol=1e-9)):
        raise ValueError("Raw cache timing differs from the bound RGB video")
    arrays, quality = preprocess_landmarks(raw, fps=args.fps, max_gap_seconds=args.max_gap_seconds,
                                         smooth_seconds=args.smooth_seconds,
                                         max_point_uncertainty=args.max_point_uncertainty)
    signals, signal_metadata = measure_body_signals(arrays)
    events = extract_body_events(arrays, min_duration_seconds=args.min_duration_seconds,
                                 min_angle_change_degrees=args.min_angle_change_degrees,
                                 min_distance_change_m=args.min_distance_change_m)
    summary = summarize_events(events, renderer=render_body_caption)
    out.mkdir(parents=True)
    save_npz(out / "tracks.npz", arrays)
    save_npz(out / "signals.npz", signals)
    report = {"schema": "native-body-facts-v1", "id": source["id"], "source_group": source["source_group"],
              "source_manifest_sha256": manifest_sha, "video_sha256": source["source_sha256"]["video"],
              "tracks_sha256": sha256(out / "tracks.npz"), "signals_sha256": sha256(out / "signals.npz"),
              "quality": quality, "signal_metadata": signal_metadata, "events": events, "summary": summary,
              "co_supported_groups": co_supported_groups(summary["selected_events"]),
              "captions": render_body_caption(events), "status": "candidate" if events else "abstain_no_supported_event",
              "parameter_config": {key: getattr(args, key) for key in ("fps", "max_gap_seconds", "smooth_seconds",
                  "max_point_uncertainty", "min_duration_seconds", "min_angle_change_degrees", "min_distance_change_m")},
              "implementation_sha256": sha256(Path(__file__)), "measurement_code_sha256": source_hashes(),
              "numpy_version": np.__version__, "scipy_version": scipy.__version__, "semantic_approval": False,
              "human_review_completed": False, "visibility_verified": False, "absence_of_events_means_static": False}
    payload = json_bytes(report)
    atomic_write(out / "facts.receipt.json", json_bytes({"facts_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "facts.json", payload)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    extraction = sub.add_parser("extract")
    for key in ("manifest", "model", "mapping", "out"):
        extraction.add_argument("--" + key, required=True)
    extraction.add_argument("--device", default="cpu")
    extraction.add_argument("--bs", type=int, default=4)
    extraction.add_argument("--num-aug", type=int, default=3)
    extraction.add_argument("--limit", type=int, default=0)
    extraction.add_argument("--ffmpeg", required=True)
    extraction.add_argument("--ffprobe", required=True)
    facts = sub.add_parser("describe")
    for key in ("bundle", "out", "ffprobe"):
        facts.add_argument("--" + key, required=True)
    facts.add_argument("--fps", type=float, default=20.)
    facts.add_argument("--max-gap-seconds", type=float, default=.2)
    facts.add_argument("--smooth-seconds", type=float, default=.0625)
    facts.add_argument("--max-point-uncertainty", type=float)
    facts.add_argument("--min-duration-seconds", type=float, default=.15)
    facts.add_argument("--min-angle-change-degrees", type=float, default=20.)
    facts.add_argument("--min-distance-change-m", type=float, default=.12)
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    {"extract": extract, "describe": describe}[args.command](args)
