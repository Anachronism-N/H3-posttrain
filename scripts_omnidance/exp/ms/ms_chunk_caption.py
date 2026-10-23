"""Original-MS whole-clip and fixed-window candidates on matched video/NLF time.

Explicit JSONL requires id (or sid), nlf_path, video_path, audio_path,
source_group and split. Read full NLF once; preprocess once before half-open
slices. Only consumed shard files are statted/hashed. Actual media end must
come from video probing, never a declared duration or the last pose plus fps.
Outputs are independent candidate records, not training or human approval.
"""

from __future__ import annotations

import argparse
import contextlib
from collections import Counter
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys

import numpy as np
import scipy

from motionscript_v2.chunks import fixed_windows
from motionscript_v2.coordinates import preprocess
from ms_landmarks import probe_video
from ms_optimize import (atomic_write, json_bytes, original_captions, overlaps,
                         save_npz, sha256, source_hashes, upstream_hashes)

SCHEMA = "motionscript_original_fixed_chunks_v2"
CLEAN_RULES = {
    "version": "neutral_dancer_no_synthetic_seconds_v1",
    "subject": "he/she/<s/he> -> the dancer; his/her -> the dancer's; him -> the dancer",
    "synthetic_seconds": "a second later/a few seconds later -> afterward",
    "raw_original_text_retained": True, "action_paraphrase": False,
    "repetition_or_real_counts_removed": False,
}
# Source arrays can have the same length as the output grid. Never infer which
# fields to slice from shape, or reset heading/translation after slicing.
TEMPORAL_FIELDS = frozenset({
    "time", "body", "translation", "yaw_deg", "valid", "joint_valid",
    "observed", "joint_observed", "joint_interpolated", "interpolated",
    "segment_id", "joint_segment_id", "root_rotation", "global_joints", "coords",
})
_WORKER = None


def load_manifest(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    records, seen = [], set()
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(row, dict):
                raise ValueError("manifest row must be an object")
            sid = row.get("id", row.get("sid"))
            if "id" in row and "sid" in row and row["id"] != row["sid"]:
                raise ValueError("id and sid disagree")
            if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sid):
                raise ValueError("id must be a portable bare identifier")
            if sid.casefold() in seen or sid.casefold() in {"protocol", "manifest", "chunks"}:
                raise ValueError("duplicate/colliding/reserved id")
            seen.add(sid.casefold())
            normalized = dict(row)
            normalized.pop("sid", None)
            normalized["id"] = sid
            for key in ("nlf_path", "video_path", "audio_path"):
                value = row.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{key} must be an explicit nonempty path")
                # Lexical normalization only: no 12k x resolve/stat for each shard.
                normalized[key] = os.path.abspath(os.path.join(path.parent, value))
            if Path(normalized["nlf_path"]).suffix != ".npz":
                raise ValueError("nlf_path must name an NPZ")
            if not isinstance(row.get("source_group"), str) or not row["source_group"].strip():
                raise ValueError("source_group must be explicitly nonempty; never guessed from id")
            if row.get("split") not in {"train", "val", "test"}:
                raise ValueError("split must be train, val or test")
            if row.get("media_start", row.get("video_start_seconds", 0)) != 0:
                raise ValueError("normalized video windows must start at zero")
            normalized["manifest_line"] = line_number
            records.append(normalized)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"manifest line {line_number}: {error}") from error
    if not records:
        raise ValueError("manifest has no records")
    return records, hashlib.sha256(content).hexdigest()


def slice_arrays(arrays, start, end):
    fixed_windows(start, end)
    time = np.asarray(arrays["time"])
    indices = np.flatnonzero((time >= start) & (time < end))
    sliced = dict(arrays)
    for key in TEMPORAL_FIELDS & arrays.keys():
        sliced[key] = np.asarray(arrays[key])[indices].copy()
    return sliced, indices


def neutral_clean(text):
    """Mechanical cleanup only; preserve actual repetitions and action counts."""
    text = re.sub(r"<s/he>|\b(?:he|she)\b", "the dancer", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:his|her)\b", "the dancer's", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhim\b", "the dancer", text, flags=re.IGNORECASE)
    # These exact phrases come from upstream's randomly chosen transition
    # templates, not elapsed-time measurement. Do not remove other numbers.
    text = re.sub(r"\b(?:a second later|a few seconds later)\b", "afterward", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def describe_windows(arrays, windows, src, shim, *, captioner=None):
    captioner = original_captions if captioner is None else captioner
    chunks = []
    for index, window in enumerate(windows):
        sliced, selected = slice_arrays(arrays, window["start"], window["end"])
        segments = captioner(sliced, src, shim)
        texts, supports, described_indices = [], [], set()
        for segment in segments:
            text = " ".join(neutral_clean(item) for item in segment["ms"] if item.strip())
            selection = selected[(sliced["time"] >= segment["start"]) & (sliced["time"] <= segment["end"])]
            described_indices.update(selection.tolist())
            supports.append({"start": segment["start"], "end": segment["end"],
                             "interval": "inclusive first/last usable sample timestamps, not action boundaries",
                             "target_indices": selection.tolist(), "sample_count": len(selection), "text": text})
            if text:
                texts.append(text)
        source_time = np.asarray(arrays["source_time"])
        source_indices = np.flatnonzero((source_time >= window["start"]) & (source_time < window["end"]))
        source_usable = arrays["source_valid"][source_indices] & arrays["source_joint_valid"][source_indices].all(axis=1)
        usable = sliced["valid"] & sliced["joint_valid"].all(axis=1)
        # Valid adjacent output samples may still be separated by an unobserved
        # source gap; original_captions keeps those spans separate.
        continuous = (len(selected) > 0 and len(segments) == 1 and bool(usable.all())
                      and bool(source_usable.all()) and len(described_indices) == len(selected))
        partial = bool(segments) and not continuous
        reason = ("no_samples_in_window" if not len(selected) else "partial_support" if partial
                  else "no_output" if not texts else None)
        candidate = bool(texts) and continuous
        candidate_text = " ".join(texts)
        timed = f"[{window['start']:.6f}, {window['end']:.6f}) seconds: {candidate_text}" if candidate else ""
        chunks.append({
            "chunk_index": index, "window_start": window["start"], "window_end": window["end"],
            "window_interval": "[start,end)", "time_basis": "global_media_seconds",
            "status": "candidate" if candidate else "abstain", "abstain_reason": reason,
            "support_status": "partial" if partial else "complete_available_samples" if continuous else "none",
            "selected_target_indices": selected.tolist(), "selected_sample_count": len(selected),
            "source_frame_indices_in_window": source_indices.tolist(),
            "first_sample_time": float(sliced["time"][0]) if len(selected) else None,
            "last_sample_time": float(sliced["time"][-1]) if len(selected) else None,
            "usable_target_sample_count": int(usable.sum()),
            "original_captions": segments, "supported_descriptions": supports,
            "partial_supported_motion_text": candidate_text if partial else "",
            "cleaned_motion_text": timed, "clean_rules": CLEAN_RULES,
            "semantic_approval": False, "human_review_completed": False,
            "window_boundaries_are_action_boundaries": False,
            "sample_span_is_complete_window_evidence": False,
            "complete_available_samples_is_verified_window_semantics": False,
            "coordinate_reset_applied": False, "segment_ids_preserved": True,
            "source_arrays_retained_full_length": True,
            "sample_index_basis": "indices into full preprocessed target grid, not original video frames",
            "final_pose_frame_duration_is_not_extrapolated": True,
        })
    return chunks


def media_bounds(row, nlf, *, ffprobe):
    info = probe_video(row["video_path"], ffprobe=ffprobe)
    if info["media_end"] is None:
        raise ValueError("actual video media end unavailable; declared end cannot bypass probing")
    fixed_windows(0, info["media_end"])
    pose = np.asarray(nlf.get("pose"))
    if pose.ndim != 2 or len(pose) != len(info["time"]):
        raise ValueError("NLF/video decoded frame count mismatch")
    fps_value = np.asarray(nlf.get("fps"))
    if "fps" in nlf:
        if fps_value.ndim != 0 or fps_value.dtype.kind not in "fiu" or not np.isfinite(fps_value) or fps_value <= 0:
            raise ValueError("NLF fps must be a finite positive scalar")
        if not np.isclose(float(fps_value), info["fps"], rtol=0, atol=1e-5):
            raise ValueError("NLF/video fps mismatch")
    elif "timestamps" not in nlf:
        raise ValueError("NLF fps or explicit timestamps required")
    if "timestamps" in nlf:
        times = np.asarray(nlf["timestamps"], dtype=np.float64)
    else:
        times = np.arange(len(pose), dtype=np.float64) / float(fps_value)
    if times.shape != info["time"].shape or not np.isfinite(times).all() or not np.allclose(times, info["time"], rtol=0, atol=1e-5):
        raise ValueError("NLF/video relative timestamps mismatch")
    if "fps" not in nlf and len(times) > 1 and not np.isclose(1 / np.median(np.diff(times)), info["fps"], rtol=0, atol=1e-5):
        raise ValueError("NLF/video inferred fps mismatch")
    for key in ("media_end", "video_end_seconds"):
        if key in row:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not np.isclose(value, info["media_end"], rtol=0, atol=1e-5):
                raise ValueError(f"{key} differs from actual video end")
    return float(info["media_end"]), {
        "media_start": 0.0, "media_end": float(info["media_end"]), "media_end_source": info["media_end_source"],
        "media_end_verified_from_video": True, "nlf_video_frame_count_matched": True,
        "nlf_video_timestamps_matched": True, "nlf_video_fps_matched": True, "timestamp_tolerance_seconds": 1e-5,
        "video_pts_offset": info["pts_offset"], "decoded_video_frames": len(info["time"]),
        "video_fps": info["fps"], "video_width": info["width"], "video_height": info["height"],
        "video_first_sample": float(info["time"][0]), "video_last_sample": float(info["time"][-1]),
        "audio_offset_verification": "unknown; audio bytes linked, not temporal alignment verified",
        "original_source_window_mapping": row.get("original_source_window_mapping", "unknown"),
        "cache_video_identity_verified": False,
        "cache_video_identity_limit": "matching frame times/count/fps is not evidence that cached poses came from these RGB bytes",
    }


def init_worker(context):
    global _WORKER
    _WORKER = context
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    src, shim = Path(context["src"]), Path(context["shim"]) if context["shim"] else None
    for root in (src, shim):
        if root:
            sys.path.insert(0, str(root))
    with contextlib.redirect_stdout(io.StringIO()):
        from text2pose.posescript import captioning
    if not any(Path(captioning.__file__).resolve().is_relative_to(root) for root in (src, shim) if root):
        raise ValueError("imported original MotionScript is outside configured source/shim")


def process_one(row):
    context = _WORKER
    out, sid = Path(context["out"]), row["id"]
    report_path = out / f"{sid}.json"
    paths = {key: Path(row[key]).resolve() for key in ("nlf_path", "video_path", "audio_path")}
    for key, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing consumed {key}: {path}")
        if overlaps(out, path.parent):
            raise ValueError("--out must be separate from consumed NLF/video/audio input directories")
    source_bytes = paths["nlf_path"].read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    video_sha, audio_sha = sha256(paths["video_path"]), sha256(paths["audio_path"])
    if report_path.exists():
        if not context["resume"]:
            raise ValueError(f"output exists: {sid}")
        report = json.loads(report_path.read_bytes())
        if (report.get("fingerprint") != context["fingerprint"] or report.get("input_sha256") != source_sha
                or report.get("video_sha256") != video_sha or report.get("audio_sha256") != audio_sha):
            raise ValueError(f"resume input/protocol mismatch: {sid}")
        receipt = json.loads((out / f"{sid}.receipt.json").read_bytes())
        if receipt.get("report_sha256") != sha256(report_path):
            raise ValueError(f"altered report: {sid}")
        if report.get("coords_file") != f"{sid}.coords.npz" or sha256(out / f"{sid}.coords.npz") != report.get("coords_sha256"):
            raise ValueError(f"altered coordinates: {sid}")
        return report
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as data:
        nlf = {key: data[key] for key in data.files}
    video_stat = paths["video_path"].stat()
    end, media = media_bounds(row, nlf, ffprobe=context["ffprobe"])
    after = paths["video_path"].stat()
    if (video_stat.st_size, video_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("video changed during probing")
    arrays, quality = preprocess(nlf, context["template"], **context["preprocess"])
    src, shim = Path(context["src"]), Path(context["shim"]) if context["shim"] else None
    whole = describe_windows(arrays, [{"start": 0., "end": end}], src, shim)[0]
    whole["mode"] = "A_original_rules_whole_clip_same_coordinates"
    windows = fixed_windows(0, end, context["chunk_seconds"], context["min_tail_seconds"])
    chunks = describe_windows(arrays, windows, src, shim)
    coords_path = out / f"{sid}.coords.npz"
    save_npz(coords_path, arrays)
    report = {"schema": SCHEMA, "id": sid, "source_group": row["source_group"], "split": row["split"],
              "source": "original_motionscript_rules_on_v2_coordinates", "fingerprint": context["fingerprint"],
              "config_hash": context["config_hash"], "manifest_record": row,
              "input_sha256": source_sha, "video_sha256": video_sha, "audio_sha256": audio_sha,
              "input_path": str(paths["nlf_path"]), "coords_file": coords_path.name, "coords_sha256": sha256(coords_path),
              "quality": quality, "media": media, "whole_clip_original": whole, "chunks": chunks,
              "status": "candidate" if any(chunk["status"] == "candidate" for chunk in chunks) else "abstain",
              "mode": "B_fixed_windows_same_whole_clip_coordinates", "dataset_processed": True, "accepted": False,
              "semantic_approval": False, "human_review_completed": False, "hand_fingers_observed": False,
              "limitations": "NLF body22/fixed SMPL-H template, unobserved fingers, heuristic floor/camera axes; original-rule text is a candidate, not verified motion semantics."}
    payload = json_bytes(report)
    atomic_write(out / f"{sid}.receipt.json", json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest()}))
    atomic_write(report_path, payload)
    return report


def safe_process(row):
    try:
        return process_one(row), None
    except Exception as error:
        result = {"id": row["id"], "error_type": type(error).__name__, "error": str(error),
                  "cached_report_rejected": (Path(_WORKER["out"]) / (row["id"] + ".json")).exists()}
        atomic_write(Path(_WORKER["out"]) / (row["id"] + ".error.json"), json_bytes(result))
        return None, result


def run(args):
    if args.workers < 1 or args.nshard < 1 or not 0 <= args.shard < args.nshard or args.limit < 0:
        raise ValueError("invalid workers/shard/nshard/limit")
    fixed_windows(0, 1, args.chunk_seconds, args.min_tail_seconds)
    manifest_path, template_path, out = [Path(value).resolve() for value in (args.manifest, args.template, args.out)]
    src, shim = Path(args.upstream_src).resolve(), Path(args.upstream_shim).resolve() if args.upstream_shim else None
    rows, manifest_sha = load_manifest(manifest_path)
    selected = (rows[:args.limit] if args.limit else rows)[args.shard::args.nshard]
    if not selected:
        raise ValueError("manifest/shard selection is empty")
    # Cheap lexical guard now; realpath/symlink guard only for consumed records.
    for row in selected:
        for key in ("nlf_path", "video_path", "audio_path"):
            parent = Path(row[key]).parent
            if out == parent or out in parent.parents or parent in out.parents:
                raise ValueError("--out must be separate from consumed input directories")
    for source in (manifest_path, template_path):
        if out == source or out in source.parents:
            raise ValueError("--out must not contain manifest/template inputs")
    if overlaps(out, src) or (shim and overlaps(out, shim)):
        raise ValueError("--out must be separate from upstream source/shim")
    template_bytes = template_path.read_bytes()
    with np.load(io.BytesIO(template_bytes), allow_pickle=False) as data:
        template = {key: data[key] for key in data.files}
    import torch
    config = {name: getattr(args, name) for name in (
        "chunk_seconds", "min_tail_seconds", "max_gap_seconds", "smooth_seconds",
        "max_joint_uncertainty", "shard", "nshard", "limit", "ffprobe", "workers")}
    config_hash = hashlib.sha256(json_bytes(config)).hexdigest()
    protocol = {"schema": SCHEMA, "manifest_sha256": manifest_sha, "selected_ids": [row["id"] for row in selected],
                "template_sha256": hashlib.sha256(template_bytes).hexdigest(),
                "code_sha256": {**source_hashes(), Path(__file__).name: sha256(Path(__file__)),
                                "ms_landmarks.py": sha256(Path(__file__).with_name("ms_landmarks.py"))},
                "upstream_sha256": upstream_hashes(src, shim), "config": config, "config_hash": config_hash,
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
                "output_fps": 20.0, "clean_rules": CLEAN_RULES,
                "semantic_approval": False, "human_review_completed": False}
    fingerprint = hashlib.sha256(json_bytes(protocol)).hexdigest()
    context = {"out": str(out), "src": str(src), "shim": str(shim) if shim else None,
               "template": template, "fingerprint": fingerprint, "config_hash": config_hash, "resume": args.resume,
               "chunk_seconds": args.chunk_seconds, "min_tail_seconds": args.min_tail_seconds,
               "ffprobe": args.ffprobe, "preprocess": {"fps": 20.0, "max_gap_seconds": args.max_gap_seconds,
                   "smooth_seconds": args.smooth_seconds, "max_joint_uncertainty": args.max_joint_uncertainty}}
    init_worker(context)  # Dependency failures stop the parent before pool spawn.
    out.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in out.iterdir()):
        raise ValueError("output artifacts must not be symlinks")
    lock = out / ".run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        marker = out / "protocol.json"
        marker_bytes = json_bytes({"fingerprint": fingerprint, "protocol": protocol})
        if marker.exists():
            if not args.resume or marker.read_bytes() != marker_bytes:
                raise ValueError("output exists or resume protocol differs; use a new independent --out")
        elif any(path.name != lock.name for path in out.iterdir()):
            raise ValueError("--out must be empty or contain an identical chunk protocol")
        else:
            atomic_write(marker, marker_bytes)
        for name in ("manifest.json", "chunks.jsonl"):
            path = out / name
            if path.exists():
                receipt = json.loads((out / "manifest.receipt.json").read_bytes())
                if receipt.get(name + "_sha256") != sha256(path):
                    raise ValueError(f"altered final artifact: {name}")
        results, errors = [], []
        pool = None
        if args.workers == 1:
            iterator = map(safe_process, selected)
        else:
            pool = mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(context,))
            iterator = pool.imap_unordered(safe_process, selected)
        try:
            for report, error in iterator:
                if error:
                    errors.append(error)
                    print(json.dumps(error, ensure_ascii=False), flush=True)
                else:
                    results.append(report)
                    print(json.dumps({"id": report["id"], "status": report["status"], "chunks": len(report["chunks"])}), flush=True)
        finally:
            if pool:
                pool.close()
                pool.join()
        order = {row["id"]: index for index, row in enumerate(selected)}
        results.sort(key=lambda row: order[row["id"]])
        errors.sort(key=lambda row: order[row["id"]])
        chunk_rows = [{"schema": SCHEMA, "id": report["id"], "source_group": report["source_group"], "split": report["split"],
                       "fingerprint": fingerprint, "config_hash": config_hash,
                       "input_sha256": report["input_sha256"], "video_sha256": report["video_sha256"], "audio_sha256": report["audio_sha256"],
                       "coords_file": report["coords_file"], "coords_sha256": report["coords_sha256"],
                       "accepted": False, **chunk} for report in results for chunk in report["chunks"]]
        chunk_bytes = b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for row in chunk_rows)
        summary = {"schema": SCHEMA, "fingerprint": fingerprint, "config_hash": config_hash,
                   "input_manifest_count": len(rows), "selected_count": len(selected), "completed_count": len(results),
                   "chunk_count": len(chunk_rows), "complete": not errors, "dataset_processed": not errors,
                   "accepted": False, "all_input_manifest_processed": not errors and len(selected) == len(rows),
                   "status_counts": dict(Counter(row["status"] for row in chunk_rows)), "errors": errors,
                   "semantic_approval": False, "human_review_completed": False,
                   "records": [{"id": report["id"], "report_file": report["id"] + ".json",
                                "report_sha256": sha256(out / (report["id"] + ".json"))} for report in results]}
        if any(error["cached_report_rejected"] for error in errors):
            # Continue other records, but never overwrite a completed aggregate
            # with a subset after rejecting an existing report during resume.
            atomic_write(out / "resume.errors.json", json_bytes(summary))
        else:
            summary_bytes = json_bytes(summary)
            atomic_write(out / "chunks.jsonl", chunk_bytes)
            atomic_write(out / "manifest.receipt.json", json_bytes({
                "manifest.json_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                "chunks.jsonl_sha256": hashlib.sha256(chunk_bytes).hexdigest()}))
            atomic_write(out / "manifest.json", summary_bytes)
        return summary
    finally:
        lock.unlink()


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Explicit JSONL: id/nlf_path/video_path/audio_path/source_group/split")
    ap.add_argument("--template", required=True)
    ap.add_argument("--upstream-src", required=True)
    ap.add_argument("--upstream-shim")
    ap.add_argument("--out", required=True, help="Independent output directory; each shard uses its own")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk-seconds", type=float, default=2.0)
    ap.add_argument("--min-tail-seconds", type=float, default=1.5)
    ap.add_argument("--max-gap-seconds", type=float, default=0.2)
    ap.add_argument("--smooth-seconds", type=float, default=0.0625)
    ap.add_argument("--max-joint-uncertainty", type=float, default=None)
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--resume", action="store_true")
    return ap


if __name__ == "__main__":
    try:
        result = run(parser().parse_args())
        print(json.dumps(result, ensure_ascii=False), flush=True)
        sys.exit(0 if result["complete"] else 1)
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
