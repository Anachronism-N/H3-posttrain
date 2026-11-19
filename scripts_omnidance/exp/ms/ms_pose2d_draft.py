"""CPU-only, unapproved 2D MotionScript drafts from an explicit native-ROI list.

No video/model access, interpolation, anatomical-side, 3D, beat or gesture claims.
Every event retains its observed frames and rule signal; missing data stay unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np

from dwpose_full_extract import joint_layout

SCHEMA = "ms_pose2d_rule_draft_v1"
FLAGS = {"semantic_approval": False, "quality_approved": False,
         "training_eligible": False, "needs_visual_revision": True}
RULES = {"min_frames": 3, "min_duration_s": 0.15,
         "wrist_min_delta_body_scales": 0.20, "wrist_min_speed_body_scales_s": 0.15,
         "angle_min_delta_degrees": 20.0, "angle_min_speed_degrees_s": 15.0,
         "max_gap_s": 0.5, "gap_median_multiplier": 3.0}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def check_row(row):
    if not isinstance(row, dict):
        raise ValueError("Manifest row must be an object")
    for key in ("id", "split", "npz_path", "roi_manifest_path"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Manifest requires nonempty {key}")
    for key in ("id", "split"):
        if re.fullmatch(r"[A-Za-z0-9_-]+", row[key]) is None:
            raise ValueError(f"Unsafe {key}; expected a bare identifier")
    return row


def scalar_text(data, key):
    value = data[key]
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(f"{key} must be a non-pickled text scalar")
    text = value.item()
    return text.decode("utf-8") if isinstance(text, bytes) else text


def validate_observations(row, roi, data):
    check_row(row)
    if roi.get("id") != row["id"] or ("split" in roi and roi["split"] != row["split"]):
        raise ValueError("ROI source ID/split mismatch")
    if Path(roi["source"]["path"]).stem != row["id"]:
        raise ValueError("ROI source path ID mismatch")
    if Path(roi["raw"]["path"]).resolve() != Path(row["npz_path"]).resolve():
        raise ValueError("NPZ path is not the ROI-bound raw path")
    for key in ("binding_sha256", "video_sha256"):
        recorded = roi.get(key)
        if not isinstance(recorded, str) or not recorded or scalar_text(data, key) != recorded:
            raise ValueError(f"NPZ/ROI {key} mismatch")
        if key in row and row[key] != recorded:
            raise ValueError(f"Manifest/ROI {key} mismatch")
    if roi["source"].get("sha256") != roi["video_sha256"]:
        raise ValueError("ROI source/video binding mismatch")
    for key in ("id", "sid"):
        if key in data and scalar_text(data, key) != row["id"]:
            raise ValueError("NPZ source ID mismatch")
    layout = json.loads(scalar_text(data, "joint_layout"))
    if layout != joint_layout():
        raise ValueError("Expected the real pixel_xy Wholebody 134 names/mapping, not 3D")
    if "joint_layout" in roi.get("model_ref", {}) and roi["model_ref"]["joint_layout"] != layout:
        raise ValueError("ROI/NPZ joint layout mismatch")
    times = data["times"]
    n = len(times) if times.ndim == 1 else 0
    specs = {"xy": ((n, 134, 2), "f"), "score": ((n, 134), "f"),
             "valid": ((n, 134), "b"), "tracked": ((n,), "b"),
             "times": ((n,), "f"), "source_timestamps": ((n,), "f"),
             "frame_indices": ((n,), "iu")}
    if not n:
        raise ValueError("An empty cache has no observations")
    for key, (shape, kinds) in specs.items():
        if data[key].shape != shape or data[key].dtype.kind not in kinds:
            raise ValueError(f"Invalid {key} shape/dtype; require raw 2D xy[T,134,2]")
    for key in ("times", "source_timestamps", "frame_indices"):
        if not np.isfinite(data[key]).all() or np.any(np.diff(data[key]) <= 0):
            raise ValueError(f"{key} must be finite and strictly increasing")
    if times[0] < 0 or data["frame_indices"][0] < 0:
        raise ValueError("Negative relative time/frame index")
    frames = roi["selected_frames"]
    if len(frames) != n:
        raise ValueError("ROI selected frame count mismatch")
    for key, field in (("frame_indices", "frame_index"), ("tracked", "tracked")):
        expected = [f[field] for f in frames]
        if key == "frame_indices" and any(type(v) is not int for v in expected):
            raise ValueError("ROI frame indices must be integers")
        if key == "tracked" and any(type(v) is not bool for v in expected):
            raise ValueError("ROI tracked values must be booleans")
        if not np.array_equal(data[key], expected):
            raise ValueError(f"ROI selected_frames/{key} mismatch")
    for key, field in (("times", "time"), ("source_timestamps", "source_timestamp")):
        expected = np.asarray([f[field] for f in frames], dtype=float)
        if not np.isfinite(expected).all() or not np.allclose(data[key], expected, atol=1e-6, rtol=0):
            raise ValueError(f"ROI selected_frames/{key} PTS mismatch")
    origin = data["source_timestamps"] - times
    if not np.allclose(origin, origin[0], atol=1e-6, rtol=0):
        raise ValueError("Relative/source timestamp origin mismatch")
    probe = roi.get("probe", {})
    if "time_origin" in probe and not math.isclose(origin[0], float(probe["time_origin"]), abs_tol=1e-6):
        raise ValueError("Recorded source PTS origin mismatch")
    for key, probe_key in (("times", "frame_times"), ("source_timestamps", "actual_video_pts")):
        if probe_key in probe:
            pts = np.asarray(probe[probe_key], dtype=float)
            indices = data["frame_indices"]
            if pts.ndim != 1 or indices[-1] >= len(pts) or not np.allclose(data[key], pts[indices], atol=1e-6, rtol=0):
                raise ValueError("Selected observations disagree with recorded native PTS")
    if "actual_video_pts_ticks" in probe:
        ticks = probe["actual_video_pts_ticks"]
        if data["frame_indices"][-1] >= len(ticks) or any(f["source_pts"] != ticks[int(i)] for f, i in zip(frames, data["frame_indices"])):
            raise ValueError("Selected source_pts disagree with recorded native PTS ticks")
    # Invalid raw NaNs are retained; effective valid points may never be fabricated.
    mask = data["valid"] & data["tracked"][:, None]
    if not np.isfinite(data["xy"][mask]).all() or not np.isfinite(data["score"][mask]).all():
        raise ValueError("Effective valid points must have finite xy and raw score")
    return mask


def load_observations(row):
    check_row(row)
    roi = read_json(row["roi_manifest_path"])
    required = {"xy", "score", "valid", "tracked", "times", "source_timestamps",
                "frame_indices", "joint_layout", "binding_sha256", "video_sha256"}
    with np.load(row["npz_path"], allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError(f"Missing native ROI NPZ fields: {sorted(required - set(archive.files))}")
        data = {key: archive[key] for key in required | ({"id", "sid"} & set(archive.files))}
    mask = validate_observations(row, roi, data)
    return roi, data, mask


def rule_signals(data, mask):
    names = joint_layout()["names"]
    index = {name: i for i, name in enumerate(names)}
    xy, n = data["xy"].astype(float), len(data["times"])
    torso = [index[k] for k in ("right_shoulder", "left_shoulder", "right_hip", "left_hip")]
    r, l, rh, lh = torso
    width = np.linalg.norm(xy[:, r] - xy[:, l], axis=1)
    length = np.linalg.norm((xy[:, r] + xy[:, l] - xy[:, rh] - xy[:, lh]) / 2, axis=1)
    scale = (width + length) / 2
    body_ok = mask[:, torso].all(axis=1) & np.isfinite(scale) & (width >= 1) & (length >= 1)
    ratio = np.full(n, np.nan)
    np.divide(width, length, out=ratio, where=length >= 1)
    body_ok &= (ratio >= 0.1) & (ratio <= 5)
    signals = []
    for side, label in (("right", "一侧"), ("left", "另一侧")):
        shoulder, elbow, wrist, hip, knee, ankle = [index[f"{side}_{part}"] for part in
                                                  ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")]
        arm_ok = body_ok & mask[:, [shoulder, wrist]].all(axis=1)
        for axis, axis_label in ((0, "horizontal"), (1, "vertical")):
            values = np.full(n, np.nan)
            values[arm_ok] = (xy[arm_ok, wrist, axis] - xy[arm_ok, shoulder, axis]) / scale[arm_ok]
            signals.append({"name": f"{side}_wrist_relative_shoulder_{axis_label}", "values": values,
                            "valid": arm_ok, "joints": torso + [wrist], "label": label,
                            "kind": axis_label, "unit": "body_scale"})
        for kind, joints in (("elbow", [shoulder, elbow, wrist]), ("knee", [hip, knee, ankle])):
            a, b, c = joints
            u, v = xy[:, a] - xy[:, b], xy[:, c] - xy[:, b]
            ul, vl = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
            ok = body_ok & mask[:, joints].all(axis=1) & (ul >= 1) & (vl >= 1)
            values = np.full(n, np.nan)
            cosine = np.sum(u[ok] * v[ok], axis=1) / (ul[ok] * vl[ok])
            values[ok] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            signals.append({"name": f"{side}_{kind}_projected_angle", "values": values,
                            "valid": ok, "joints": sorted(set(torso + joints)), "label": label,
                            "kind": kind, "unit": "degrees"})
    return signals, body_ok


def contiguous_spans(ok, links):
    spans, start = [], None
    for i, present in enumerate(ok):
        if present:
            if start is None:
                start = i
            elif not links[i - 1]:
                spans.append((start, i - 1))
                start = i
        elif start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(ok) - 1))
    return spans


def extract_events(data, signals, links):
    times, indices, events = data["times"], data["frame_indices"], []
    for signal in signals:
        values, ok = signal["values"], signal["valid"]
        angle = signal["unit"] == "degrees"
        min_speed = RULES["angle_min_speed_degrees_s" if angle else "wrist_min_speed_body_scales_s"]
        min_delta = RULES["angle_min_delta_degrees" if angle else "wrist_min_delta_body_scales"]
        for lo, hi in contiguous_spans(ok, links):
            if hi - lo < 2:
                continue
            speed = np.diff(values[lo:hi + 1]) / np.diff(times[lo:hi + 1])
            directions = np.where(speed > min_speed, 1, np.where(speed < -min_speed, -1, 0))
            edge = 0
            while edge < len(directions):
                end = edge + 1
                direction = int(directions[edge])
                while end < len(directions) and directions[end] == direction:
                    end += 1
                a, b = lo + edge, lo + end
                delta, duration = float(values[b] - values[a]), float(times[b] - times[a])
                if direction and b - a + 1 >= RULES["min_frames"] and duration >= RULES["min_duration_s"] and abs(delta) + 1e-9 >= min_delta:
                    kind, label = signal["kind"], signal["label"]
                    if kind == "vertical":
                        action = "上移" if direction < 0 else "下移"
                        text = f"{label}手腕相对肩部在画面中{action}。"
                    elif kind == "horizontal":
                        action = "向画面右侧移动" if direction > 0 else "向画面左侧移动"
                        text = f"{label}手腕相对肩部{action}。"
                    else:
                        action = "增大" if direction > 0 else "减小"
                        joint_label = "肘" if kind == "elbow" else "膝"
                        text = f"{label}{joint_label}部的二维投影夹角{action}。"
                    events.append({"start": float(times[a]), "end": float(times[b]),
                                   "kind": kind, "direction": direction, "text": text,
                                   "support": {"signal": signal["name"], "unit": signal["unit"],
                                               "joint_indices": signal["joints"], "frame_indices": indices[a:b + 1].tolist(),
                                               "times": times[a:b + 1].tolist(), "values": values[a:b + 1].tolist(),
                                               "dt_seconds": np.diff(times[a:b + 1]).tolist(),
                                               "delta": delta, "mean_speed_per_s": delta / duration}})
                edge = end
    events.sort(key=lambda e: (e["start"], e["end"], e["support"]["signal"]))
    for i, event in enumerate(events):
        event["event_id"] = f"e{i:04d}"
    return events


def load_bar_windows(row, roi):
    if not row.get("bar_plan_path"):
        return None
    plan = read_json(row["bar_plan_path"])
    if not isinstance(plan, dict) or plan.get("id") != row["id"]:
        raise ValueError("Bar plan must be an object bound to the same source ID")
    for key in ("split", "binding_sha256", "video_sha256"):
        expected = row["split"] if key == "split" else roi[key]
        if key in plan and plan[key] != expected:
            raise ValueError(f"Bar plan {key} mismatch")
    record = plan.get("manifest_record", {})
    if isinstance(record, dict):
        for key in ("id", "split"):
            if key in record and record[key] != row[key]:
                raise ValueError(f"Bar manifest_record {key} mismatch")
    manifest = plan.get("window_manifest", {})
    if not isinstance(manifest, dict):
        raise ValueError("window_manifest must be an object")
    if manifest and manifest.get("id") != row["id"]:
        raise ValueError("Window manifest source ID mismatch")
    binding = plan.get("binding")
    if not isinstance(binding, dict) or binding.get("id") != row["id"] or binding.get("video_sha256") != roi["video_sha256"]:
        raise ValueError("Window plan requires matching source ID/video binding")
    if manifest and manifest.get("binding") != binding:
        raise ValueError("Top-level/window_manifest binding mismatch")
    probe = roi.get("probe", {})
    if "actual_video_pts_sha256" in binding:
        if "frame_times" not in probe:
            raise ValueError("Recorded relative PTS list required to verify window binding")
        # ROI hashes {absolute times, ticks}; music windows hash a pretty-JSON relative list.
        pts_bytes = (json.dumps(probe["frame_times"], ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if hashlib.sha256(pts_bytes).hexdigest() != binding["actual_video_pts_sha256"]:
            raise ValueError("Bar binding/ROI relative actual_video_pts_sha256 mismatch")
    if "decoded_video_frames" in binding and "decoded_video_frames" in probe and binding["decoded_video_frames"] != probe["decoded_video_frames"]:
        raise ValueError("Bar binding/ROI decoded_video_frames mismatch")
    start_bound, end_bound = float(binding["media_start"]), float(binding["media_end"])
    offset = float(binding["video_pts_offset"])
    if not all(math.isfinite(v) for v in (start_bound, end_bound, offset)) or start_bound < 0 or end_bound <= start_bound:
        raise ValueError("Invalid bound media range/PTS offset")
    if not math.isclose(offset, float(probe.get("time_origin", 0)), abs_tol=1e-6):
        raise ValueError("Bar video_pts_offset/ROI source PTS mismatch")
    if "duration" in probe and end_bound > float(probe["duration"]) + 1e-6:
        raise ValueError("Window binding exceeds recorded video duration")
    if manifest and manifest.get("time_basis") != "global_media_seconds":
        raise ValueError("Window time basis must be explicit global_media_seconds")
    bars = plan.get("windows", manifest.get("windows"))
    if "windows" in plan and "windows" in manifest and plan["windows"] != manifest["windows"]:
        raise ValueError("Top-level/window_manifest windows mismatch")
    if not isinstance(bars, list) or not bars:
        raise ValueError("Bar plan requires explicit nonempty windows with start/end seconds")
    windows = []
    for i, bar in enumerate(bars):
        start, end = float(bar["start"]), float(bar["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < start_bound or end > end_bound + 1e-6 or end <= start:
            raise ValueError("Invalid explicit media window")
        if windows and start < windows[-1]["end"]:
            raise ValueError("Media windows must be ordered and nonoverlapping")
        windows.append({"window_id": i, "start": start, "end": end, "basis": "provided_media_window",
                        "audio_video_alignment_verified": False, "beat_phase_verified": False,
                        "motions_on_beats_verified": False, "one_window_is_exactly_one_bar": False,
                        "window_boundaries_are_action_boundaries": False})
    return windows


def generate_draft(row, roi, data, mask, bar_windows=None):
    signals, body_ok = rule_signals(data, mask)
    times = data["times"]
    dt = np.diff(times)
    gap_limit = min(RULES["max_gap_s"], max(0.15, RULES["gap_median_multiplier"] * float(np.median(dt)))) if len(dt) else RULES["max_gap_s"]
    links = dt <= gap_limit
    events = extract_events(data, signals, links)
    observed = [{"start": float(times[a]), "end": float(times[b]),
                 "frame_indices": data["frame_indices"][a:b + 1].tolist()}
                for a, b in contiguous_spans(body_ok, links)]
    unknown = []
    if len(times) == 1 and not body_ok[0]:
        unknown.append({"start": float(times[0]), "end": float(times[0]),
                        "reason": "invalid_tracking_points_or_scale"})
    for i in range(len(dt)):
        if not links[i] or not (body_ok[i] and body_ok[i + 1]):
            unknown.append({"start": float(times[i]), "end": float(times[i + 1]),
                            "reason": "time_gap" if not links[i] else "invalid_tracking_points_or_scale"})
    duration = roi.get("probe", {}).get("duration")
    if duration is not None:
        duration = float(duration)
        if not math.isfinite(duration) or duration < times[-1]:
            raise ValueError("Recorded duration is shorter than the observed time range")
        if times[0] > 0:
            unknown.insert(0, {"start": 0.0, "end": float(times[0]), "reason": "outside_cached_observations"})
        if duration > times[-1]:
            unknown.append({"start": float(times[-1]), "end": duration, "reason": "outside_cached_observations"})
    windows = bar_windows
    if windows is None:
        windows = [{"window_id": i, "start": span["start"], "end": span["end"], "basis": "observed_time"}
                   for i, span in enumerate(observed)]
        windows += [{"window_id": f"unknown_{i}", "start": item["start"], "end": item["end"], "basis": "unobserved", "reason": item["reason"]}
                    for i, item in enumerate(unknown)]
    windows = [dict(window) for window in windows]
    for window in windows:
        a, b = window["start"], window["end"]
        covered = sum(max(0.0, min(b, span["end"]) - max(a, span["start"])) for span in observed)
        refs = []
        for event in events:
            support = event["support"]
            selected = [i for i, t in enumerate(support["times"]) if a <= t < b]
            if len(selected) < RULES["min_frames"]:
                continue
            lo, hi = selected[0], selected[-1]
            start, end = support["times"][lo], support["times"][hi]
            delta = support["values"][hi] - support["values"][lo]
            min_delta = RULES["angle_min_delta_degrees" if support["unit"] == "degrees" else "wrist_min_delta_body_scales"]
            if end - start >= RULES["min_duration_s"] and abs(delta) + 1e-9 >= min_delta:
                refs.append({"event_id": event["event_id"], "observed_overlap_start": start,
                             "observed_overlap_end": end, "frame_indices": support["frame_indices"][lo:hi + 1],
                             "times": support["times"][lo:hi + 1], "values": support["values"][lo:hi + 1]})
        window["event_refs"] = refs
        window["observed_duration_s"] = covered
        window["unknown"] = covered < max(0.0, b - a) - 1e-6 or window["basis"] == "unobserved" or b == a
        window["observation_status"] = "unknown" if not covered else ("partial" if window["unknown"] else "observed")
        window["status"] = "needs_regeneration" if not refs else ("partial" if window["unknown"] else "draft_generated")
        window["text"] = " ".join(dict.fromkeys(e["text"] for e in events if any(ref["event_id"] == e["event_id"] for ref in refs))) or "unknown：没有足够的动态动作证据。"
    all_signals_ok = np.stack([s["valid"] for s in signals]).all(axis=0)
    status = "needs_regeneration" if not events else "draft_generated"
    if events and (unknown or not all_signals_ok.all() or not links.all() or (bar_windows is not None and any(w["unknown"] or w["status"] == "needs_regeneration" for w in windows))):
        status = "partial"
    return {"schema": SCHEMA, "id": row["id"], "split": row["split"], "status": status,
            **FLAGS, "anatomical_side_guaranteed": False, "quality_is_probability": False,
            "beat_alignment_claimed": False, "coordinate_system": "pixel_xy_2d",
            "source": {"npz_path": str(Path(row["npz_path"]).resolve()),
                       "roi_manifest_path": str(Path(row["roi_manifest_path"]).resolve()),
                       "bar_plan_path": row.get("bar_plan_path"),
                       "video_sha256": roi["video_sha256"], "binding_sha256": roi["binding_sha256"]},
            "observation": {"start": float(times[0]), "end": float(times[-1]), "frame_count": len(times),
                            "tracked_frames": int(data["tracked"].sum()), "body_scale_valid_frames": int(body_ok.sum()),
                            "observed_spans": observed, "unknown_intervals": unknown,
                            "outside_observation_range": "unknown; never extrapolated", "gap_limit_s": gap_limit},
            "rules": RULES, "events": events, "windows": windows,
            "ms_text": " ".join(dict.fromkeys(e["text"] for e in events)) or "unknown：没有足够的动态动作证据。",
            "limitations": ["二维投影规则草稿；必须视觉修订，不能直接训练。", "一侧/另一侧是模型索引标签，不保证解剖左右。",
                            "不推断3D朝向、承重、手指手势、音乐风格或节拍同步。", "没有动作证据不等于静止；未观测部分为unknown。"]}


def write_new(path, draft):
    path = Path(path)
    raw = (json.dumps(draft, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".draft.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def process_row(row, out):
    check_row(row)
    path = Path(out) / row["split"] / row["id"] / "draft.json"
    if path.exists():
        return {"id": row["id"], "split": row["split"], "status": "existing_not_overwritten", "path": str(path)}
    started = time.perf_counter()
    roi, data, mask = load_observations(row)
    bars = load_bar_windows(row, roi)
    loaded = time.perf_counter()
    draft = generate_draft(row, roi, data, mask, bars)
    diagnostic = row.get("diagnostic_not_production") is True
    draft["diagnostic_not_production"] = diagnostic
    generated = time.perf_counter()
    try:
        write_new(path, draft)
    except FileExistsError:
        return {"id": row["id"], "split": row["split"], "status": "existing_not_overwritten", "path": str(path)}
    finished = time.perf_counter()
    return {"id": row["id"], "split": row["split"], "status": draft["status"], "path": str(path),
            "diagnostic_not_production": diagnostic, "quality_approved": False, "training_eligible": False,
            "event_count": len(draft["events"]), "window_count": len(draft["windows"]),
            "observation": draft["observation"],
            "event_boundaries": [{"event_id": e["event_id"], "start": e["start"], "end": e["end"],
                                  "signal": e["support"]["signal"], "support_frame_count": len(e["support"]["frame_indices"])} for e in draft["events"]],
            "windows": [{k: w[k] for k in ("window_id", "start", "end", "status", "observation_status", "unknown", "observed_duration_s", "event_refs")} for w in draft["windows"]],
            "profile_seconds": {"read_and_validate": loaded - started, "generate": generated - loaded,
                                "write_back": finished - generated, "total": finished - started}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Explicit JSONL rows: id/split/npz_path/roi_manifest_path, optional bar_plan_path")
    parser.add_argument("--out", required=True, help="Write-once root: OUT/SPLIT/ID/draft.json")
    parser.add_argument("--limit", type=int, default=3, help="Maximum sources, default 3; must be explicitly increased")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive; no implicit all-sources mode")
    started = time.perf_counter()
    attempted = written = existing = errors = event_count = window_count = diagnostic_written = 0
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
                key = (row["split"], row["id"])
                if key in seen:
                    raise ValueError("Duplicate id/split in explicit input list")
                seen.add(key)
                result = process_row(row, args.out)
                if result["status"] == "existing_not_overwritten":
                    existing += 1
                else:
                    written += 1
                    diagnostic_written += int(result.get("diagnostic_not_production", False))
                    statuses[result["status"]] += 1
                event_count += result.get("event_count", 0)
                window_count += result.get("window_count", 0)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors += 1
                print(json.dumps({"id": row.get("id") if isinstance(row, dict) else None,
                                  "status": "rejected", "error": str(error)}, ensure_ascii=False), file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - started
    print(json.dumps({"summary": True, "sources_attempted": attempted, "sources_written": written,
                      "sources_existing": existing, "sources_rejected": errors,
                      "diagnostic_sources_written": diagnostic_written, "quality_approved_sources": 0,
                      "diagnostic_not_production": written > 0 and diagnostic_written == written,
                      "status_counts": statuses,
                      "source_per_hour": written * 3600 / elapsed if elapsed else 0,
                      "event_count": event_count, "window_count": window_count,
                      "elapsed_e2e_seconds": elapsed}, ensure_ascii=False), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
