"""Source-bound RGB and 3D diagnostics, without semantic certification."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from ms_landmarks import decode_batches, load_bundle, probe_video, require_new_output
from ms_optimize import atomic_write, json_bytes, sha256


_EDGES = (("neck", "head"), ("neck", "spine3"), ("spine3", "spine2"),
          ("spine2", "spine1"), ("spine1", "pelvis")) + tuple(
    (f"{side}_{a}", f"{side}_{b}") for side in ("left", "right")
    for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("hip", "knee"),
                 ("knee", "ankle"), ("ankle", "foot")))


def load_facts(directory):
    payload = (directory / "facts.json").read_bytes()
    receipt = json.loads((directory / "facts.receipt.json").read_text())
    if receipt.get("facts_sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("Facts receipt does not match")
    report = json.loads(payload)
    if report.get("schema") != "native-body-facts-v1":
        raise ValueError("Native body facts required")
    arrays = {}
    for name in ("tracks", "signals"):
        data = (directory / (name + ".npz")).read_bytes()
        if hashlib.sha256(data).hexdigest() != report[name + "_sha256"]:
            raise ValueError(name + " cache differs from bound facts")
        with np.load(io.BytesIO(data), allow_pickle=False) as cache:
            arrays[name] = dict(cache)
    return report, arrays, hashlib.sha256(payload).hexdigest()


def at_source_times(tracks, source_time):
    time = tracks["time"]
    points = tracks["points_m"]
    valid = tracks["point_valid"]
    segments = tracks["point_segment_id"]
    result = np.full((len(source_time), points.shape[1], 3), np.nan)
    supported = np.zeros(result.shape[:2], bool)
    for row, value in enumerate(source_time):
        right = int(np.searchsorted(time, value))
        exact = next((i for i in (right, right - 1)
                      if 0 <= i < len(time) and abs(float(time[i]) - value) <= 1e-9), None)
        if exact is not None:
            good = valid[exact] & (segments[exact] >= 0)
            result[row, good] = points[exact, good]
        elif 0 < right < len(time):
            left = right - 1
            good = valid[left] & valid[right] & (segments[left] >= 0) & (segments[left] == segments[right])
            weight = (value - time[left]) / (time[right] - time[left])
            result[row, good] = points[left, good] * (1 - weight) + points[right, good] * weight
        else:
            continue
        supported[row] = good & np.isfinite(result[row]).all(axis=-1)
    return result, supported


def project_camera(points, width, height, fov_degrees=55.):
    focal = max(width, height) / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    good = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    pixels = np.full(points.shape[:-1] + (2,), np.nan)
    pixels[good] = points[good, :2] / points[good, 2:3] * focal + [(width - 1) / 2, (height - 1) / 2]
    return pixels, good


def draw_points(image, pixels, names, valid):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    lookup = {name: i for i, name in enumerate(names)}
    inside = (valid & np.isfinite(pixels).all(axis=-1) & (pixels >= 0).all(axis=-1)
              & (pixels < np.array(image.size)).all(axis=-1))
    for left, right in _EDGES:
        if left in lookup and right in lookup and inside[lookup[left]] and inside[lookup[right]]:
            a, b = pixels[lookup[left]], pixels[lookup[right]]
            draw.line([tuple(a), tuple(b)], fill=(40, 230, 70), width=3)
    for i in np.flatnonzero(inside):
        x, y = pixels[i]
        color = (255, 90, 90) if names[i].startswith("left_") else (90, 170, 255)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        if names[i].endswith(("wrist", "ankle")):
            draw.text((x + 4, y), names[i], fill=color)
    return result, inside


def export(args):
    bundle, facts_dir, out = Path(args.bundle).resolve(), Path(args.facts).resolve(), Path(args.out).resolve()
    require_new_output(out, [bundle / "raw.npz", facts_dir / "facts.json"])
    raw, source, source_manifest_sha = load_bundle(bundle)
    facts, arrays, facts_sha = load_facts(facts_dir)
    if (facts["id"] != source["id"] or facts["source_manifest_sha256"] != source_manifest_sha
            or facts["video_sha256"] != source["source_sha256"]["video"]):
        raise ValueError("Facts and native tracks do not belong to the same source")
    video = Path(source["source_paths"]["video"])
    if sha256(video) != source["source_sha256"]["video"]:
        raise ValueError("Source RGB video changed")
    info = probe_video(video, args.ffprobe)
    if (not np.array_equal(raw["pts"], info["pts"])
            or not np.array_equal(raw["time_base"], info["time_base"])
            or not np.allclose(raw["timestamps"], info["time"], rtol=0, atol=1e-9)):
        raise ValueError("Source RGB and native 3D have different timing")
    if list(raw["names"]) != list(arrays["tracks"]["names"]):
        raise ValueError("Processed point mapping differs from native mapping")
    processed, processed_valid = at_source_times(arrays["tracks"], raw["timestamps"])
    pixels, positive_depth = project_camera(processed, info["width"], info["height"])
    processed_valid &= positive_depth
    raw_projection, raw_positive = project_camera(raw["points_m"], info["width"], info["height"])
    compare = raw_positive & np.isfinite(raw["points2d_px"]).all(axis=-1)
    residual = np.linalg.norm(raw_projection[compare] - raw["points2d_px"][compare], axis=-1)
    out.mkdir(parents=True)
    images_dir = out / "frames"
    images_dir.mkdir()
    records, frame_files = [], []
    names = raw["names"].tolist()
    width, height = info["width"], info["height"]
    for start, frames in decode_batches(video, info, 4, args.ffmpeg):
        for offset, frame in enumerate(frames):
            i = start + offset
            original = Image.fromarray(frame)
            native, native_inside = draw_points(original, raw["points2d_px"][i], names, raw["valid"][i])
            smooth, smooth_inside = draw_points(original, pixels[i], names, processed_valid[i])
            canvas = Image.new("RGB", (width * 3 + (width * 3) % 2, height + 150 + height % 2), "black")
            draw = ImageDraw.Draw(canvas)
            for column, (image, title) in enumerate(((original, "Original RGB"), (native, "Native NLF pixels"),
                                                    (smooth, "Processed 3D projection: support only"))):
                canvas.paste(image, (width * column, 28))
                draw.text((width * column + 6, 6), title, fill="white")
            time = float(raw["timestamps"][i])
            active = [event for event in facts["events"] if event["start"] - 1e-9 <= time <= event["end"] + 1e-9]
            draw.text((6, height + 34), f'{source["id"]} frame={int(raw["frame_indices"][i])} PTS={int(raw["pts"][i])} seconds={time:.6f}', fill="white")
            draw.text((6, height + 50), "Geometry is estimated, not world/visibility/force ground truth. Red=anatomical left name; blue=right name.", fill="white")
            draw.text((6, height + 66), "Native screen-excluded/unknown: " + ", ".join(name for j, name in enumerate(names) if not native_inside[j]), fill="white")
            draw.text((6, height + 82), "Processed unsupported/unknown: " + ", ".join(name for j, name in enumerate(names) if not smooth_inside[j]), fill="white")
            draw.text((6, height + 98), "Active candidates: " + ", ".join(event["id"] for event in active), fill="white")
            filename = images_dir / f"{i:06d}.png"
            canvas.save(filename, compress_level=1)
            frame_files.append(filename)
            records.append({"frame_index": int(raw["frame_indices"][i]), "pts": int(raw["pts"][i]),
                            "seconds": time, "native_screen_supported": native_inside.tolist(),
                            "processed_screen_supported": smooth_inside.tolist(),
                            "active_event_ids": [event["id"] for event in active],
                            "image": str(filename), "sha256": sha256(filename)})
    if len(records) != len(raw["timestamps"]):
        raise ValueError("Evidence export lost source frames")
    num, den = map(int, raw["time_base"])
    concat = ["ffconcat version 1.0"]
    for i, filename in enumerate(frame_files):
        escaped = str(filename).replace("'", "'\\''")
        concat.extend([f"file '{escaped}'", f"option framerate {den}/{num}"])
        if i + 1 < len(frame_files):
            concat.append(f"duration {raw['timestamps'][i + 1] - raw['timestamps'][i]:.12f}")
    atomic_write(out / "frames.ffconcat", ("\n".join(concat) + "\n").encode())
    subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(out / "frames.ffconcat"), "-fps_mode", "vfr", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-threads", "2", "-preset", "veryfast", "-crf", "18",
                    "-video_track_timescale", str(den), str(out / "review.mp4")], check=True)
    rendered = probe_video(out / "review.mp4", args.ffprobe)
    tolerance = max(2e-6, num / den)
    if len(rendered["time"]) != len(records) or not np.allclose(rendered["time"], raw["timestamps"], rtol=0, atol=tolerance):
        raise ValueError("Rendered evidence timestamps differ from source frames")
    review = {"schema": "native-body-review-template-v1", "id": source["id"],
              "video_sha256": facts["video_sha256"], "facts_sha256": facts_sha,
              "source_video_path": str(video),
              "source_frame_namespace": "decoded frame indices of the selected input clip; no parent-video frame mapping inferred",
              "review_order": "watch source RGB and establish reference occurrences first, then inspect geometry/candidate events",
              "reviewer": None, "reviewer_type": None, "review_completed": False,
              "reference_built_from_rgb_before_reading_candidates": False,
              "reference_occurrences": [], "event_decisions": [], "decidable_intervals": [],
              "critical_source_time_mapping_errors": [], "incorrect_removed_occurrence_ids": [],
              "unknown_reasons": [], "absolute_3d_scale_or_depth_certified": False}
    atomic_write(out / "review_template.json", json_bytes(review))
    report = {"schema": "native-body-evidence-v1", "id": source["id"], "facts_sha256": facts_sha,
              "source_manifest_sha256": source_manifest_sha, "source_video_sha256": facts["video_sha256"],
              "frames": records, "frame_count": len(records), "time_base": raw["time_base"].tolist(),
              "source_pts_offset_seconds": source["source_pts_offset_seconds"], "source_media_bounds": source["media_bounds"],
              "review_video": str(out / "review.mp4"), "review_video_sha256": sha256(out / "review.mp4"),
              "display_to_source_rgb_affine": {name: [[1, 0, -column * width], [0, 1, -28], [0, 0, 1]]
                  for column, name in enumerate(("original", "native", "processed"))},
              "source_rgb_panel_lossless_png": True, "external_resize_or_autorotation": False,
              "rendered_time_tolerance_seconds": tolerance, "rendered_final_frame_duration_is_not_source_evidence": True,
              "default_intrinsics_reference": "NLF ptu3d.intrinsic_matrix_from_field_of_view(55): max side focal, (width-1,height-1)/2 principal point",
              "native_3d_to_stored_pixels_residual_p95_px": float(np.percentile(residual, 95)) if len(residual) else None,
              "native_3d_to_stored_pixels_residual_is_semantic_accuracy": False,
              "processed_projection_alignment": "interpolate only within same supported point segment; no edge extrapolation",
              "point_support_is_visibility": False, "semantic_approval": False, "human_review_completed": False,
              "implementation_sha256": sha256(Path(__file__))}
    payload = json_bytes(report)
    atomic_write(out / "evidence.receipt.json", json_bytes({"evidence_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(out / "evidence.json", payload)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "facts", "out", "ffmpeg", "ffprobe"):
        ap.add_argument("--" + key, required=True)
    return ap


if __name__ == "__main__":
    export(parser().parse_args())
