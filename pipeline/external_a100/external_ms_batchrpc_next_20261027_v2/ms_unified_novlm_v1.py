"""Finite MotionScript-only worker. Production requires a live native caller gate.

CLI defaults to deny: its ticket alone never grants GPU authorization. Caller
must hold its existing native resource lease throughout run_ticket. New edition
claims are never removed/retried; batch commit is the only success marker.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np
import dwpose_full_extract as pose
import ms_fullframe_pose2d_draft_v2 as v2

SCHEMA = "ms_unified_novlm_ticket_v1"
CACHE_SCHEMA = "ms_fullvideo_pose2d_cache_v1"
GIB = 1024 ** 3
FILES = ("pose.npz", "fullframe_manifest.json", "beats.json", "draft.json", "receipt.json")


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_once(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".publish-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)


def claim_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def check_ticket(ticket, limit=3):
    if not isinstance(ticket, dict) or ticket.get("schema") != SCHEMA or limit < 1:
        raise ValueError("Explicit new-edition ticket and positive finite limit required")
    for key in ("edition", "ticket_id"):
        if not isinstance(ticket.get(key), str) or re.fullmatch(r"[A-Za-z0-9_-]+", ticket[key]) is None:
            raise ValueError("Unsafe ticket/edition identifier")
    if ticket.get("authorization") != "native_parent" or type(ticket.get("gpu_index")) is not int or ticket["gpu_index"] < 0:
        raise ValueError("Native parent authorization and physical single GPU index required")
    for key in ("output_root", "local_cache_root", "beat_checkpoint", "scail_root"):
        if not isinstance(ticket.get(key), str) or not Path(ticket[key]).is_absolute():
            raise ValueError(f"Absolute {key} required")
    for key in ("ffprobe_path", "ffmpeg_path"):
        if key in ticket and (not isinstance(ticket[key], str) or not Path(ticket[key]).is_absolute()):
            raise ValueError(f"Optional {key} must be an explicit absolute local executable")
    sources = ticket.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Finite explicit assignment list required")
    seen = set()
    for row in sources:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or re.fullmatch(r"[A-Za-z0-9_-]+", row[k]) is None for k in ("id", "split")):
            raise ValueError("Unsafe source ID/split")
        if row["id"] in seen:
            raise ValueError("Duplicate source ID, including split mixing")
        seen.add(row["id"])
        for key in ("video_path", "audio_path"):
            if not isinstance(row.get(key), str) or not Path(row[key]).is_absolute() or Path(row[key]).stem != row["id"]:
                raise ValueError("Source paths must retain original source ID")
        if any(k in row for k in ("roi_manifest_path", "bar_plan_path", "npz_path")):
            raise ValueError("Old ROI/cache claims are not an input")
    budget = ticket.get("max_cache_bytes", 8 * GIB)
    if type(budget) is not int or budget < 8 * GIB:
        raise ValueError("At least 8 GiB bounded local cache required")
    return sources[:limit], budget


def resource_snapshot(local_root):
    """Read actual cgroup v1/v2 ancestor headroom and independent local disk."""
    groups = [line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()]
    unified = next((entry for entry in groups if entry[0] == "0" and not entry[1]), None)
    memory = next((entry for entry in groups if "memory" in entry[1].split(",")), None)
    if unified:
        root, relative, limit_name, usage_name = Path("/sys/fs/cgroup"), unified[2], "memory.max", "memory.current"
    elif memory:
        root, relative, limit_name, usage_name = Path("/sys/fs/cgroup/memory"), memory[2], "memory.limit_in_bytes", "memory.usage_in_bytes"
    else:
        raise PermissionError("Actual memory cgroup cannot be verified")
    current = root / relative.lstrip("/")
    headrooms = []
    while current.is_relative_to(root):
        maximum = (current / limit_name).read_text().strip()
        used = int((current / usage_name).read_text())
        if maximum != "max":
            headrooms.append(int(maximum) - used)
        if current == root:
            break
        current = current.parent
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    headrooms.append(available)
    local = Path(local_root).resolve(strict=True)
    mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    matches = [entry for entry in mounts if local.is_relative_to(Path(entry[1]))]
    filesystem = max(matches, key=lambda entry: len(entry[1]))[2]
    if filesystem in {"ceph", "nfs", "nfs4", "cifs", "fuse.ceph", "fuse.ceph-fuse", "lustre"}:
        raise PermissionError("Cache must be independent local disk")
    return {"cgroup_headroom": min(headrooms), "local_free": shutil.disk_usage(local).free}


def audio_metadata(path):
    import soundfile
    info = soundfile.info(str(path))
    return {"duration_seconds": info.frames / info.samplerate, "sample_rate": info.samplerate,
            "channels": info.channels, "frames": info.frames}

def deny_gate(ticket):
    raise PermissionError("Production disabled: native caller resource gate/lease not integrated")


def probe_video(path, executable="ffprobe"):
    command = [str(executable), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames", "-show_entries", "stream=width,height,duration,nb_frames,time_base:frame=best_effort_timestamp,best_effort_timestamp_time,pkt_duration_time", "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def video_metadata(actual):
    streams, frames = actual["streams"], actual["frames"]
    if len(streams) != 1 or not frames:
        raise ValueError("Exactly one selected native video stream and frames required")
    stream = streams[0]
    pts = np.array([float(frame["best_effort_timestamp_time"]) for frame in frames], dtype=np.float64)
    base = Fraction(stream["time_base"])
    ticks = np.array([int(frame["best_effort_timestamp"]) for frame in frames], dtype=np.int64)
    if base <= 0 or not np.allclose(ticks.astype(float) * float(base), pts, atol=1e-6, rtol=0):
        raise ValueError("Native PTS ticks/time_base disagree with recorded source timestamps")
    duration = float(stream["duration"])
    width, height = int(stream["width"]), int(stream["height"])
    if not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0) or not np.isfinite(duration) or duration <= 0 or duration < pts[-1] - pts[0] or min(width, height) <= 0:
        raise ValueError("Invalid/reversed native video PTS, dimensions or video duration")
    if stream.get("nb_frames") not in (None, "N/A") and int(stream["nb_frames"]) != len(frames):
        raise ValueError("Native frame count disagrees with actual probe frames")
    return pts, duration, width, height


def decode_bgr(path, width, height, executable="ffmpeg"):
    command = [str(executable), "-v", "error", "-threads", "1", "-noautorotate", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            size = width * height * 3
            while True:
                raw = process.stdout.read(size)
                if not raw:
                    break
                if len(raw) != size:
                    raise ValueError("Truncated native raw BGR frame")
                yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read(4096).decode(errors="replace"))
        finally:
            process.stdout.close()
            process.wait()  # Closing pipe ends unfinished decoder; no process signals.


def load_models(ticket):
    if os.environ.get("DWPOSE_ALLOW_CPU") == "1":
        raise PermissionError("DWPose CPU fallback forbidden")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(ticket["gpu_index"]):
        raise PermissionError("Caller must set the authorized physical GPU visibility; core never remaps it")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    checkpoint = Path(ticket["beat_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError("Local BeatThis checkpoint required; downloads forbidden")
    import torch
    torch.set_num_threads(1)
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise PermissionError("Exactly one authorized CUDA device required")
    scail_root = Path(ticket["scail_root"])
    process_root = scail_root / "DWPoseProcess"
    sys.path.insert(0, str(process_root))
    module = importlib.import_module("dwpose.wholebody")
    if Path(module.__file__).resolve() != (process_root / "dwpose/wholebody.py").resolve():
        raise RuntimeError("Unverified Wholebody module")
    weights = module.ModelDataPathPrefix
    module.ModelDataPathPrefix = scail_root / "pretrained_weights"
    try:
        wholebody = module.Wholebody(0, use_batch=False)
    finally:
        module.ModelDataPathPrefix = weights
    if any("CUDAExecutionProvider" not in session.get_providers() for session in (wholebody.session_det, wholebody.session_pose)):
        raise RuntimeError("Actual detector/pose CUDA sessions required")
    from beat_this.inference import File2Beats
    beats = File2Beats(checkpoint_path=str(checkpoint), device="cuda:0", float16=False, dbn=False)
    if not all(parameter.device.type == "cuda" and parameter.dtype == torch.float32 for parameter in beats.model.parameters()):
        raise RuntimeError("BeatThis actual CUDA FP32 required")
    return wholebody, beats


def copy_source(row, root, max_bytes, profile=None):
    started = time.perf_counter()
    try:
        destination = root / row["split"] / row["id"] / "inputs"
        destination.mkdir(parents=True)
        copied, hashes, total = {}, {}, 0
        for key in ("video_path", "audio_path"):
            source = Path(row[key])
            target = (destination / ("video" if key == "video_path" else "audio")).with_suffix(source.suffix)
            digest = hashlib.sha256()
            with source.open("rb", buffering=0) as reader, target.open("xb", buffering=0) as writer:
                while raw := reader.read(256 * 1024):
                    total += len(raw)
                    if total > max_bytes:
                        raise ValueError("Source exceeds bounded two-source prefetch byte budget")
                    digest.update(raw)
                    writer.write(raw)
            copied[key], hashes[key] = target, digest.hexdigest()
            if row.get(key.replace("_path", "_sha256"), hashes[key]) != hashes[key]:
                raise ValueError("Input hash differs from explicit immutable assignment")
        return copied, hashes
    
    finally:
        if profile is not None:
            profile["copy"] = time.perf_counter() - started

def extract_and_rule(row, copied, hashes, models, local_case, bound_case, probe=probe_video, decoder=decode_bgr, max_output_bytes=4 * GIB, audio_info=audio_metadata, profile=None, source_started=None):
    profile = profile if profile is not None else {}
    case_started = time.perf_counter()
    started = case_started
    local_case.mkdir(parents=True)
    actual = probe(copied["video_path"])
    pts, duration, width, height = video_metadata(actual)
    profile["probe"] = time.perf_counter() - started
    started = time.perf_counter()
    n = len(pts)
    xy, score = np.full((n, 134, 2), np.nan, np.float32), np.full((n, 134), np.nan, np.float32)
    bbox, tracked, reasons = np.full((n, 4), np.nan, np.float32), np.zeros(n, bool), np.empty(n, np.uint8)
    previous, count = None, 0
    for index, frame in enumerate(decoder(copied["video_path"], width, height)):
        if index >= n or frame.shape != (height, width, 3) or frame.dtype != np.uint8:
            raise ValueError("Native decoder frame count/dimensions mismatch")
        points, scores, box, reason = pose.select_subject(models[0](frame), previous, 0.3)
        reasons[index], count = reason, index + 1
        if box is not None:
            xy[index], score[index], bbox[index], tracked[index], previous = points, scores, box, True, box
    if count != n:
        raise ValueError("Native decoder omitted frames")
    profile["decode_pose"] = time.perf_counter() - started
    started = time.perf_counter()
    data = {"schema": np.array(CACHE_SCHEMA), "id": np.array(row["id"]), "split": np.array(row["split"]), "video_sha256": np.array(hashes["video_path"]), "joint_layout": np.array(pose.LAYOUT_JSON), "xy": xy, "score": score, "bbox": bbox, "tracked": tracked, "missing_reason": reasons, "valid": tracked[:, None] & (score > 0.3), "frame_indices": np.arange(n, dtype=np.int64), "source_timestamps": pts, "times": pts - pts[0], **{key: np.array(value) for key, value in v2.FLAGS.items()}}
    bound_row = {"id": row["id"], "split": row["split"], "npz_path": str(bound_case / "pose.npz"), "fullframe_manifest_path": str(bound_case / "fullframe_manifest.json")}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **data)
    raw = buffer.getvalue()
    manifest = {**bound_row, "schema": v2.MANIFEST_SCHEMA, "input_schema": CACHE_SCHEMA, "video_path": row["video_path"], "video_sha256": hashes["video_path"], "npz_sha256": hashlib.sha256(raw).hexdigest(), "frame_count": n, "width": width, "height": height, "duration_seconds": duration, "pts_origin": float(pts[0]), "joint_layout": pose.joint_layout(), "score_threshold": 0.3, **{key: data[key].tolist() for key in ("frame_indices", "times", "source_timestamps")}, **v2.FLAGS, "metadata_provenance": {"actual_probe": actual, "decoder": "ffmpeg native all frames BGR24; no resize, rotate or FPS sampling", "tracking": "largest first; IoU>=0.3 against last accepted box; no reinitialization/interpolation", "input_source_row": row}}
    mask = v2.validate_observations(bound_row, manifest, data)
    draft = v2.generate_draft(bound_row, manifest, data, mask)
    raw_body_missing = int((~mask[:, :18].all(axis=1)).sum())
    if raw_body_missing or not tracked.all():
        draft["status"] = "partial" if draft["events"] else "needs_regeneration"
        draft["limitations"].append("Missing raw body/track evidence stays partial/unknown; never imputed or quality-complete.")
    profile["rules"] = time.perf_counter() - started
    started = time.perf_counter()
    audio = audio_info(copied["audio_path"])
    audio_duration = float(audio["duration_seconds"])
    if not np.isfinite(audio_duration) or audio_duration <= 0:
        raise ValueError("Actual audio duration must be finite and positive")
    beats, downbeats = (np.asarray(values, dtype=float) for values in models[1](str(copied["audio_path"])))
    if any(values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > audio_duration + 0.02) or np.any(np.diff(values) <= 0) for values in (beats, downbeats)) or any(not np.isclose(beat, beats, atol=1e-6, rtol=0).any() for beat in downbeats):
        raise ValueError("Invalid actual BeatThis beat/downbeat output")
    profile["beat"] = time.perf_counter() - started
    started = time.perf_counter()
    sidecar = {"schema": "ms_beatthis_audio_clockonly_v1", "id": row["id"], "split": row["split"], "audio_path": row["audio_path"], "audio_sha256": hashes["audio_path"], "beats": beats.tolist(), "downbeats": downbeats.tolist(), "emptybeats": not bool(beats.size), "method": "beat_this.inference.File2Beats", "device": "cuda:0", "float16": False, "dbn": False, "clock": "audio_only", "actual_audio_metadata": audio, "audio_video_alignment_verified": False, "motions_on_beats_verified": False}
    receipt = {"schema": "ms_unified_novlm_case_receipt_v1", "source_row": row, "status": "extracted_rule_draft_unapproved", "input_copy_completed": True, "profile_seconds": profile, "draft_status": draft["status"], "frame_count": n, "tracked_frames": int(tracked.sum()), "raw_body_missing_frames": raw_body_missing, "video_duration_seconds": duration, "tail_unknown": {"start": float(data["times"][-1]), "end": duration}, "beat_alignment_claimed": False, "vlm_called": False, "rgb_or_crops_written": False, **v2.FLAGS}
    payloads = dict(zip(FILES, (raw, json_bytes(manifest), json_bytes(sidecar), json_bytes(draft), json_bytes(receipt))))
    if sum(map(len, payloads.values())) > max_output_bytes:
        raise ValueError("Rule artifacts exceed bounded local cache output budget")
    for name, content in payloads.items():
        if name != "receipt.json":
            write_once(local_case / name, content)
    profile["write"] = time.perf_counter() - started
    profile["extract_rule_e2e"] = time.perf_counter() - case_started
    profile["whole_case_e2e"] = time.perf_counter() - (source_started if source_started is not None else case_started)
    write_once(local_case / "receipt.json", json_bytes(receipt))
    return receipt


def run_ticket(ticket, limit=3, resource_gate=deny_gate, model_loader=load_models, probe=probe_video,
               decoder=decode_bgr, resource_reader=None, audio_info=audio_metadata, stage_gate=None):
    started_e2e = time.perf_counter()
    ticket = json.loads(json_bytes(ticket))
    sources, budget = check_ticket(ticket, limit)
    local_root = Path(ticket["local_cache_root"])
    with resource_gate(ticket) as evidence:
        if not isinstance(evidence, dict) or evidence.get("granted") is not True or evidence.get("owner_pid") != os.getpid() or evidence.get("gpu_index") != ticket["gpu_index"] or not evidence.get("gpu_uuid") or evidence.get("gpu_observed_free") is not True or evidence.get("native_memory_gate_preserved") is not True or not 0 <= time.time() - evidence.get("observed_unix", 0) <= 30:
            raise PermissionError("Fresh actual GPU ownership/free/native memory evidence required")
        snapshot = (resource_reader or resource_snapshot)(local_root)
        if snapshot["cgroup_headroom"] < 32 * GIB or snapshot["local_free"] < budget + 8 * GIB:
            raise PermissionError("Need actual >=32 GiB cgroup headroom plus cache and 8 GiB local reserve")
        edition = Path(ticket["output_root"]) / ticket["edition"]
        if not edition.is_absolute() or edition.resolve() != edition:
            raise PermissionError("Output edition may not traverse symlinks")
        claim = edition / "claims" / "tickets" / (ticket["ticket_id"] + ".json")
        claim_once(claim, {"ticket": ticket, "owner_pid": os.getpid(), "resource_evidence": evidence})
        started = time.perf_counter()
        models = model_loader(ticket)  # Native caller wrapper runs its actual gate(True) here.
        profile = {"models_load_once": time.perf_counter() - started}
        if stage_gate:
            stage_gate(False)
        actual_probe = (lambda path: probe_video(path, ticket["ffprobe_path"])) if probe is probe_video and "ffprobe_path" in ticket else probe
        actual_decoder = (lambda path, width, height: decode_bgr(path, width, height, ticket["ffmpeg_path"])) if decoder is decode_bgr and "ffmpeg_path" in ticket else decoder
        receipts = []
        batch = edition / "batches" / ticket["ticket_id"]
        with tempfile.TemporaryDirectory(prefix="ms-unified-", dir=local_root) as directory, ThreadPoolExecutor(max_workers=1) as executor:
            work = Path(directory)
            pending, todo = deque(), iter(sources)
            def enqueue():
                row = next(todo, None)
                if row is None:
                    return False
                source_claim = edition / "claims" / "sources" / (row["id"] + ".json")
                try:
                    claim_once(source_claim, {"source_row": row, "ticket_id": ticket["ticket_id"], "owner_pid": os.getpid(), "budget_consumed": True, "requeue_allowed": False})
                except FileExistsError:
                    receipts.append({"source_row": row, "status": "existing_claim_not_reassigned", "input_copy_completed": False})
                    return True
                case_profile = {}
                source_started = time.perf_counter()
                pending.append((row, executor.submit(copy_source, row, work, budget // 4, case_profile), case_profile, source_started))
                return True
            while len(pending) < 2 and enqueue():
                pass
            while pending:
                row, future, case_profile, case_started = pending.popleft()
                relative = Path("cases") / row["split"] / row["id"]
                case = work / relative
                copy_completed = False
                try:
                    copied, hashes = future.result()
                    copy_completed = True
                    used = sum(path.stat().st_size for path in (work / "cases").rglob("*") if path.is_file())
                    result = extract_and_rule(row, copied, hashes, models, case, batch / relative, actual_probe, actual_decoder, budget // 2 - used, audio_info, case_profile, case_started)
                except Exception as error:
                    case_profile["whole_case_e2e"] = time.perf_counter() - case_started
                    result = {"schema": "ms_unified_novlm_case_receipt_v1", "source_row": row, "status": "failed_sealed_no_retry", "input_copy_completed": copy_completed, "profile_seconds": case_profile, "error": str(error), "budget_consumed": True, **v2.FLAGS}
                    case.mkdir(parents=True, exist_ok=True)
                    write_once(case / "receipt.json", json_bytes(result))
                receipts.append(result)
                shutil.rmtree(work / row["split"] / row["id"], ignore_errors=False)
                if stage_gate:
                    stage_gate(False)
                while len(pending) < 2 and enqueue():
                    pass
            executor.shutdown(wait=True)
            prefetch_drained = not pending
            started = time.perf_counter()
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as pack:
                for source in sorted((work / "cases").glob("*/*")):
                    for name in FILES:
                        path = source / name
                        if path.is_file():
                            pack.add(path, arcname=str(path.relative_to(work)), recursive=False)
                            write_once(batch / path.relative_to(work), path.read_bytes())
            raw = archive.getvalue()
            write_once(batch / "batch.tar", raw)
            profile["pack_publish"] = time.perf_counter() - started
            if stage_gate:
                stage_gate(False)
            profile["whole_e2e"] = time.perf_counter() - started_e2e
            receipts.sort(key=lambda item: next(i for i, source in enumerate(sources) if source["id"] == item["source_row"]["id"]))
            commit = {"schema": "ms_unified_novlm_batch_commit_v1", "ticket_id": ticket["ticket_id"], "edition": ticket["edition"], "batch_sha256": hashlib.sha256(raw).hexdigest(), "receipts": receipts, "prefetch_drained": prefetch_drained, "profile_seconds": profile, "publication_scope": "Small pilot: five real filesystem files/case plus identical batch tar members; no input media", "unique_sources_attempted": sum(item["status"] != "existing_claim_not_reassigned" for item in receipts), "quality_approved": False, "beat_alignment_claimed": False, "production_complete_claimed": False}
            write_once(batch / "commit.json", json_bytes(commit))
            return commit

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--cpu-test", action="store_true", help="Mocked fixtures only; no production models/claims")
    args = parser.parse_args(argv)
    if args.cpu_test:
        if args.ticket:
            parser.error("--cpu-test never accepts production tickets")
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_ms_unified_novlm_v1.py")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.ticket:
        parser.error("--ticket required")
    ticket = json.loads(Path(args.ticket).read_text())
    check_ticket(ticket, args.limit)
    parser.error("Production disabled: integrate run_ticket with existing native caller resource manager; CLI ticket grants no GPU/claims")


if __name__ == "__main__":
    raise SystemExit(main())
