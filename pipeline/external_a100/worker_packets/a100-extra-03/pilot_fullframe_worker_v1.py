"""Ticket-bound, write-once native full-frame pose/actual-audio cache pilot."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import select
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid

BASE = Path("/tmp/ms_pose2d_draft_20261026T1427Z")
EDITION = "external_a100_fullframe_pilot_20261026_v1"
ROLES = {"a100-extra-03": "launcher", "a100-extra-04": "worker-0",
         "a100-extra-05": "worker-1", "a100-extra-06": "worker-2"}
CODE_REL = ("ms_hand_roi.py", "dwpose_full_extract.py", "ms_pose2d_draft.py",
            "dwpose/wholebody.py", "dwpose/onnxdet.py", "dwpose/onnxpose.py", "dwpose/__init__.py")
OUTPUTS = ("new_fullvideo_wholebody134.npz", "actual_audio_beats.json",
           "fullframe_manifest.json", "case_receipt.json")
FLAGS = {"diagnostic_not_production": True, "semantic_approval": False,
         "quality_approved": False, "training_eligible": False, "needs_visual_revision": True}
SCHEMA = "ms_fullvideo_pose2d_diagnostic_v1"
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
GiB = 2 ** 30


def now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(path):
    s = Path(path).stat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("input is not a regular file: " + str(path))
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "device": s.st_dev, "inode": s.st_ino}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=pairs,
                      parse_constant=bad_constant)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mkdir_scoped(path, scope):
    path, scope = Path(path), Path(scope)
    if path != scope and scope not in path.parents:
        raise ValueError("directory escaped authorized scope")
    path.mkdir(parents=True, exist_ok=True)
    if path.resolve(strict=True) != path or scope.resolve(strict=True) != scope:
        raise ValueError("directory scope is redirected")
    return path


def write_local(path, value, limit=None):
    raw = json_bytes(value)
    if limit is not None and len(raw) > limit:
        raise BudgetStop("JSON output exceeds reserved remaining cache bytes")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir(Path(path).parent)


def publish_file(local, shared, expected_sha, scope):
    shared = Path(shared)
    mkdir_scoped(shared.parent, scope)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=shared.parent, prefix=".pilot-", delete=False) as target:
            temporary = Path(target.name)
            h, copied = hashlib.sha256(), 0
            with Path(local).open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    target.write(chunk)
                    h.update(chunk)
                    copied += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        if h.hexdigest() != expected_sha or copied != Path(local).stat().st_size:
            raise ValueError("publish copy SHA/size mismatch")
        os.link(temporary, shared, follow_symlinks=False)
        fsync_dir(shared.parent)
        if shared.is_symlink() or shared.stat().st_size != copied or sha(shared) != expected_sha:
            raise ValueError("published output SHA/size mismatch")
        return {"path": str(shared), "bytes": copied, "sha256": expected_sha}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_json(value, shared, scope):
    shared = Path(shared)
    mkdir_scoped(shared.parent, scope)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=shared.parent, prefix=".pilot-", delete=False) as stream:
            temporary = Path(stream.name)
            raw = json_bytes(value)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, shared, follow_symlinks=False)
        fsync_dir(shared.parent)
        if sha(shared) != hashlib.sha256(raw).hexdigest():
            raise ValueError("published JSON SHA mismatch")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_wrapper(path, root):
    path = Path(path)
    if path != root / "fullvideo_validated_helpers_ed283d684.py" or path.resolve(strict=True) != path:
        raise ValueError("validated wrapper is not the owned frozen executed helper snapshot")
    if sha(path) != "ed283d6844830a1c2da109361391dbc44858a16c0e0262ca3ace117b3c48cb78":
        raise ValueError("validated helper snapshot SHA256 mismatch")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("_pilot_validated_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("gpu_gate", "memory_contract", "import_official", "local_model_identity",
                 "probe_native", "coverage", "process_identity"):
        if not callable(getattr(module, name, None)):
            raise ValueError("validated helper missing: " + name)
    return module


def validate(args):
    if args.worker not in ROLES:
        raise ValueError("only the explicitly assigned worker is allowed; no implicit all")
    root = Path(args.root)
    if root != BASE / ROLES[args.worker] or root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("root must be the current worker own fixed BASE role")
    ticket_path, manifest_path = Path(args.ticket), Path(args.manifest)
    if not ticket_path.is_absolute() or not manifest_path.is_absolute():
        raise ValueError("ticket and assignment require explicit absolute paths")
    ticket_raw = ticket_path.read_bytes()
    ticket = strict_json(ticket_raw)
    required = {"schema": "external_a100_fullframe_pilot_ticket_v1", "worker_id": args.worker,
                "edition": EDITION, "source_count": 100, "copy_threads": 1,
                "max_inflight_sources": 2, "local_cache_budget_GiB": 8,
                "models_loaded_once_per_worker": True, "allowed_gpu_indices": [0], "processes": 1,
                "automatic_retry": False, "write_once": True, "old_claims_modified": False,
                "repeat_weight_media_sha_or_profile_each_source": False,
                "semantic_approval": False, "quality_approved": False, "training_eligible": False,
                "needs_visual_revision": True, "formal_final_caption_sources": 0,
                "monitor_seconds": 60, "not_32b_caption_pipeline": True,
                "task_stage": "new_fullframe_pose_and_beats_cache", "claim_key": "split/id.json", "required_outputs": list(OUTPUTS)}
    for key, expected in required.items():
        if ticket.get(key) != expected or type(ticket.get(key)) is not type(expected):
            raise ValueError("ticket field mismatch: " + key)
    edition_root = ticket_path.parent.parent
    if (edition_root.name != EDITION or ticket_path != edition_root / "assignments" / (args.worker + ".ticket.json")
            or manifest_path != edition_root / "assignments" / (args.worker + ".jsonl")
            or ticket.get("assignment_path") != str(manifest_path)
            or ticket.get("claims_root") != str(edition_root / "claims")
            or ticket.get("output_root") != str(edition_root / "results" / args.worker)
            or edition_root.resolve(strict=True) != edition_root):
        raise ValueError("ticket/assignment/claim/output must be this new edition and worker")
    raw = manifest_path.read_bytes()
    assignment_sha = hashlib.sha256(raw).hexdigest()
    if assignment_sha != ticket.get("assignment_sha256"):
        raise ValueError("complete assignment bytes do not match exact ticket SHA256; cursor remains 0")
    if not raw or b"\0" in raw or not raw.endswith(b"\n"):
        raise ValueError("assignment must be nonempty NUL-free UTF8 JSONL with terminal LF; cursor remains 0")
    rows, seen = [], set()
    keys = {"id", "split", "batch_id", "source_ordinal", "video_path", "audio_path", "meta_path"}
    for number, line in enumerate(raw.split(b"\n")[:-1], 1):
        try:
            row = strict_json(line)
            if not isinstance(row, dict) or set(row) != keys:
                raise ValueError("row schema mismatch")
            if (any(not isinstance(row[k], str) or not SAFE.fullmatch(row[k]) for k in ("id", "split", "batch_id"))
                    or type(row["source_ordinal"]) is not int or row["source_ordinal"] < 1):
                raise ValueError("unsafe id/split/batch or source ordinal")
            identity = (row["split"].casefold(), row["id"].casefold())
            if identity in seen:
                raise ValueError("duplicate split/id")
            seen.add(identity)
            for key in ("video_path", "audio_path", "meta_path"):
                if not isinstance(row[key], str) or not Path(row[key]).is_absolute() or "\0" in row[key]:
                    raise ValueError("input paths must be explicit absolute paths")
            rows.append(row)
        except Exception as error:
            raise ValueError("assignment line %d rejected; no lines skipped; cursor remains 0: %s" % (number, error)) from error
    if len(rows) != 100:
        raise ValueError("assignment must contain exactly 100 rows; cursor remains 0")
    code_root = root / "cache/current/assigned_validation_packet_v2/code"
    if code_root.resolve(strict=True) != code_root:
        raise ValueError("official snapshot scope redirected")
    for relative in CODE_REL:
        path = code_root / relative
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("official copied snapshot missing/redirected: " + relative)
    wrapper = load_wrapper(args.validated_wrapper, root)
    return root, ticket, rows, wrapper, code_root, {
        "ticket_path": str(ticket_path), "ticket_sha256": hashlib.sha256(ticket_raw).hexdigest(),
        "assignment_path": str(manifest_path), "assignment_sha256": assignment_sha,
        "code_sha256": sha(Path(__file__)), "validated_wrapper_sha256": sha(args.validated_wrapper),
        "official_code_sha256": {relative: sha(code_root / relative) for relative in CODE_REL},
        "assignment_read_node_pid": os.getpid(), "cursor": 0}


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, limit):
        self.limit = limit
        self.condition = threading.Condition()
        self.amounts = {}
        self.failed = set()
        self.stopped = False

    def set(self, key, amount):
        with self.condition:
            if amount > self.limit:
                raise BudgetStop("source input/output/array reservation exceeds 8GiB")
            while sum(self.amounts.values()) - self.amounts.get(key, 0) + amount > self.limit:
                failed_bytes = sum(self.amounts[k] for k in self.failed if k != key)
                if amount + failed_bytes > self.limit:
                    raise BudgetStop("retained failed cache prevents reservation; stop without retry/reassignment")
                if self.stopped:
                    raise BudgetStop("run stopped during cache reservation")
                self.condition.wait(timeout=1)
            if self.stopped:
                raise BudgetStop("run stopped; no further claim/copy")
            self.amounts[key] = amount
            self.condition.notify_all()

    def retain_failed(self, key, amount):
        with self.condition:
            self.amounts[key] = amount
            self.failed.add(key)
            if sum(self.amounts.values()) > self.limit:
                self.stopped = True
            self.condition.notify_all()

    def release_success(self, key):
        with self.condition:
            self.amounts.pop(key, None)
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"cachebytes": sum(self.amounts.values()), "cache_budget_bytes": self.limit,
                    "failed_cache_bytes": sum(self.amounts[k] for k in self.failed),
                    "cache_reservations": dict(self.amounts)}


class Progress:
    def __init__(self, root, identity, budget, total):
        self.path = mkdir_scoped(root / "logs", root) / "pilot_fullframe_progress.json"
        self.events = self.path.parent / ("pilot_fullframe_events_" + identity["runid"] + ".jsonl")
        self.lock, self.done = threading.RLock(), threading.Event()
        self.budget = budget
        self.value = {**FLAGS, **identity, "status": "starting", "model_init_counts": {"dwpose": 0, "beatthis": 0},
                      "processed": 0, "success": 0, "partial": 0, "failed": 0, "skipped": 0,
                      "total_sources": total, "totalframes": 0, "remaining": total, "cursor": 0,
                      "current_source": None, "frames_done": 0, "total_frames": 0,
                      "copy_seconds": 0.0, "inference_seconds": 0.0, "publish_seconds": 0.0,
                      "inflight": 0, "queuedwriteback": 0}
        self.thread = threading.Thread(target=self.heartbeat, name="pilot-heartbeat", daemon=True)

    def start(self):
        self.update(event="run_start")
        self.thread.start()

    def heartbeat(self):
        while not self.done.wait(60):
            try:
                self.update()
            except Exception as error:
                self.budget.stop()
                print(json.dumps({"status": "heartbeat_failed", "error": str(error)}), file=sys.stderr, flush=True)
                return

    def update(self, event=None, **fields):
        with self.lock:
            self.value.update(fields, time=now(), **self.budget.snapshot())
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=".pilot-progress-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(json_bytes(self.value))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
                fsync_dir(self.path.parent)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            if event:
                with self.events.open("ab") as stream:
                    stream.write(json_bytes({"event": event, **self.value}))
                    stream.flush()
                    os.fsync(stream.fileno())

    def finish_case(self, result):
        with self.lock:
            status = result["status"]
            self.value[status] += 1
            if status != "skipped":
                self.value["processed"] += 1
            self.value["totalframes"] += result.get("frames_done", 0)
            self.value["remaining"] -= 1
            self.value["cursor"] += 1
            for stage in ("copy", "inference", "publish"):
                self.value[stage + "_seconds"] += result.get("stage_seconds", {}).get(stage, 0.0)
            self.update(event="case_" + status, current_source=result["id"], case_status=status)

    def close(self):
        self.done.set()
        self.thread.join(timeout=5)


class LimitedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, raw):
        if self.stream.tell() + len(raw) > self.limit:
            raise BudgetStop("output write exceeds reserved remaining cache bytes")
        return self.stream.write(raw)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def bounded_probe(command):
    decoder = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, bufsize=0)
    data, deadline = bytearray(), time.monotonic() + 300
    try:
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError("native ffprobe timed out")
            ready, _, _ = select.select([decoder.stdout], [], [], 1)
            if not ready:
                continue
            chunk = os.read(decoder.stdout.fileno(), 1024 * 1024)
            if not chunk:
                break
            if len(data) + len(chunk) > 16 * 1024 ** 2:
                raise BudgetStop("full native PTS probe exceeds bounded 16MiB metadata reservation")
            data.extend(chunk)
        if decoder.wait(timeout=max(.1, deadline - time.monotonic())) != 0:
            raise RuntimeError("ffprobe full native-frame PTS query failed")
        return bytes(data)
    finally:
        decoder.stdout.close()
        if decoder.poll() is None:
            decoder.terminate()
            try:
                decoder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                decoder.kill()
                decoder.wait(timeout=10)


def probe_native(path, np):
    # Adapt the accepted probe helper only to include this video container duration.
    command = ["ffprobe", "-v", "error", "-threads", "2", "-select_streams", "v:0",
               "-show_frames", "-show_streams", "-show_format", "-show_entries",
               "frame=best_effort_timestamp_time,pts_time,width,height:"
               "stream=width,height,duration,avg_frame_rate,time_base:format=duration",
               "-of", "json", str(path)]
    value = strict_json(bounded_probe(command))
    streams, frames = value.get("streams", []), value.get("frames", [])
    if len(streams) != 1 or not frames:
        raise ValueError("require one native video stream with real decoded frame PTS")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    if width <= 0 or height <= 0:
        raise ValueError("invalid native video dimensions")
    times, origins = [], set()
    for frame in frames:
        key = "best_effort_timestamp_time" if "best_effort_timestamp_time" in frame else "pts_time"
        if key not in frame:
            raise ValueError("native frame has no real PTS; no FPS-derived timestamps")
        times.append(float(frame[key]))
        origins.add(key)
        if (int(frame.get("width", width)), int(frame.get("height", height))) != (width, height):
            raise ValueError("native dimensions changed midstream")
    pts = np.asarray(times, dtype=np.float64)
    if not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("native PTS must be finite and strictly increasing")
    duration, duration_source = None, None
    for candidate, field in ((stream.get("duration"), "video_stream.duration"),
                             (value.get("format", {}).get("duration"), "video_file.format.duration")):
        try:
            candidate = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate) and candidate > 0:
            duration, duration_source = candidate, field
            break
    if duration is None:
        raise ValueError("video probe has no explicit positive duration; audio duration is not a substitute")
    return {"width": width, "height": height, "frame_count": len(pts), "pts": pts,
            "time_origin": float(pts[0]), "pts_fields": sorted(origins), "duration": duration,
            "duration_source": duration_source, "time_base": stream.get("time_base"),
            "avg_frame_rate_metadata_only": stream.get("avg_frame_rate"), "command": command}


def tree_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    if path.is_symlink():
        return path.lstat().st_size
    for item in path.rglob("*"):
        s = item.lstat()
        if stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode):
            total += s.st_size
    return total


def claim_and_copy(row, context):
    root, ticket, cache, run, budget, np = (context[k] for k in ("root", "ticket", "cache", "run", "budget", "np"))
    key = row["split"] + "/" + row["id"]
    out = Path(ticket["output_root"]) / row["split"] / row["id"]
    if os.path.lexists(out):
        return {"row": row, "status": "skipped", "reason": "output_already_exists"}
    claims_root = Path(ticket["claims_root"])
    claim = mkdir_scoped(claims_root / row["split"], claims_root) / (row["id"] + ".json")
    if budget.stopped:
        return {"row": row, "status": "unattempted", "reason": "run_stopped_before_claim"}
    record = {"schema": "ms_fullframe_pilot_claim_v1", "edition": ticket["edition"], **run,
              "id": row["id"], "split": row["split"], "time": now(),
              "input_original": {k: row[k] for k in ("video_path", "audio_path", "meta_path")},
              "automatic_retry": False}
    try:
        with claim.open("xb") as stream:
            stream.write(json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        fsync_dir(claim.parent)
    except FileExistsError:
        return {"row": row, "status": "skipped", "reason": "claim_already_exists", "claim_path": str(claim)}
    except Exception as error:
        budget.stop()
        return {"row": row, "status": "failed", "claim_path": str(claim), "key": key,
                "error": type(error).__name__ + ": " + str(error), "failed_stage": "claim_write_fsync", "stop": True}
    case = cache / row["split"] / row["id"]
    result = {"row": row, "status": "prepared", "claim_path": str(claim), "case": str(case),
              "key": key, "stage_seconds": {}, "frames_done": 0}
    before = time.monotonic()
    try:
        snapshots = {k: stamp(row[k]) for k in ("video_path", "audio_path", "meta_path")}
        input_bytes = sum(s["bytes"] for s in snapshots.values())
        if input_bytes > budget.limit:
            raise BudgetStop("sum of source video/audio/meta bytes exceeds 8GiB; claim retained")
        budget.set(key, input_bytes + 16 * 1024 ** 2)
        if shutil.disk_usage(root).free < input_bytes + 16 * 1024 ** 2 + 2 * GiB:
            raise BudgetStop("insufficient local disk for bounded input copy plus independent 2GiB headroom")
        mkdir_scoped(case.parent, cache)
        case.mkdir(exist_ok=False)
        write_local(case / "own_marker.json", {"runid": run["runid"], "worker": run["worker"],
                                               "key": key, "cache_root": str(cache)})
        assets, local = [], {}
        for k in ("video_path", "audio_path", "meta_path"):
            original = Path(row[k])
            suffix = {"video_path": ".video", "audio_path": ".audio", "meta_path": ".meta"}[k] + original.suffix
            path = case / ("input" + suffix)
            h, copied = hashlib.sha256(), 0
            with original.open("rb") as src, path.open("xb") as dst:
                opened = os.fstat(src.fileno())
                if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns, opened.st_dev, opened.st_ino) != tuple(snapshots[k][x] for x in ("bytes", "mtime_ns", "ctime_ns", "device", "inode")):
                    raise ValueError("source changed between stat and copy open")
                for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
                    if budget.stopped:
                        raise BudgetStop("run stopped during copy; partial cache retained")
                    copied += len(chunk)
                    if copied > snapshots[k]["bytes"]:
                        raise ValueError("source grew during copy")
                    dst.write(chunk)
                    h.update(chunk)
                dst.flush()
                os.fsync(dst.fileno())
                after_fd = os.fstat(src.fileno())
                if (after_fd.st_size, after_fd.st_mtime_ns, after_fd.st_ctime_ns, after_fd.st_dev, after_fd.st_ino) != tuple(snapshots[k][x] for x in ("bytes", "mtime_ns", "ctime_ns", "device", "inode")):
                    raise ValueError("opened source changed during copy")
            if stamp(original) != snapshots[k] or copied != snapshots[k]["bytes"]:
                raise ValueError("source path/size/stat changed during one-pass copy")
            local[k] = str(path)
            assets.append({"key": k, "original_path": str(original), "local_path": str(path),
                           "bytes": copied, "sha256": h.hexdigest(), "source_stat_before": snapshots[k],
                           "source_stat_after": stamp(original), "local_stat_after_copy": stamp(path)})
        fsync_dir(case)
        source = {**row, **local, "original_paths": {k: row[k] for k in local}, "assets": assets}
        result["source"] = source
        result["stage_seconds"]["copy"] = time.monotonic() - before
        probe_start = time.monotonic()
        probe = probe_native(source["video_path"], np)
        result["stage_seconds"]["probe"] = time.monotonic() - probe_start
        import soundfile as sf
        info = sf.info(source["audio_path"])
        if not math.isfinite(info.duration) or info.duration <= 0 or info.frames <= 0:
            raise ValueError("actual audio must have a positive finite duration")
        # Covers raw arrays, their NPZ+bundle copies, JSON timelines, decode buffer,
        # and actual-audio buffers before allocating anything proportional to N.
        reserve = (input_bytes + probe["frame_count"] * 8192 + probe["width"] * probe["height"] * 12
                   + info.frames * (info.channels + 2) * 4 + 16 * 1024 ** 2)
        budget.set(key, reserve)
        if shutil.disk_usage(root).free < reserve - input_bytes + 2 * GiB:
            raise BudgetStop("insufficient local disk for conservative arrays/output/bundle reservation")
        result.update(probe=probe, audio_info={"duration_seconds": float(info.duration),
                      "sample_rate": info.samplerate, "channels": info.channels, "sample_frames": info.frames},
                      reserved_bytes=reserve, input_bytes=input_bytes)
        return result
    except Exception as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error),
                      failed_stage="claim_copy_probe_budget", stop=isinstance(error, BudgetStop))
        result["stage_seconds"].setdefault("copy", time.monotonic() - before)
        budget.retain_failed(key, tree_bytes(case))
        if result["stop"]:
            budget.stop()
        return result


def load_models(context):
    root, wrapper, code_root = (context[k] for k in ("root", "wrapper", "code_root"))
    api, _unused_rules = wrapper.import_official(code_root)
    import numpy as np
    import torch
    from beat_this.inference import File2Beats
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly physical GPU0 must be visible and CUDA available")
    identity = wrapper.local_model_identity(api, root, code_root)
    binding_path = root / "model_cache/BeatThis/local_source_manifest.json"
    binding = strict_json(binding_path.read_bytes())
    checkpoint = Path(binding["checkpoint_path"]).resolve(strict=True)
    checkpoint_stat = stamp(checkpoint)
    if ((root / "model_cache/BeatThis") not in checkpoint.parents
            or not binding.get("official_sha1_match") or not binding.get("model_bytes_unchanged")
            or binding.get("schema") != "beatthis_checkpoint_source_v1"
            or checkpoint.stat().st_size != binding.get("checkpoint_bytes")
            or sha(checkpoint) != binding.get("checkpoint_sha256")
            or stamp(checkpoint) != checkpoint_stat):
        raise ValueError("formal local BeatThis source/checkpoint binding mismatch")
    model = api.load_cuda_model(identity, {"num_threads": 4, "device": 0}, profile_root=None)
    context["run"]["model_init_counts"]["dwpose"] = 1
    with checkpoint.open("rb") as stream:
        beat = File2Beats(checkpoint_path=stream, device="cuda:0", float16=False, dbn=False)
    context["run"]["model_init_counts"]["beatthis"] = 1
    if stamp(checkpoint) != checkpoint_stat:
        raise ValueError("local checkpoint changed during model initialization")
    torch.cuda.synchronize()
    devices = sorted({str(p.device) for p in beat.model.parameters()})
    if str(beat.device) != "cuda:0" or devices != ["cuda:0"] or beat.float16:
        raise RuntimeError("actual File2Beats device/parameters must be cuda:0, float16=False")
    context.update(api=api, np=np, torch=torch, pose_model=model, beat_model=beat,
                   model_identity={"dwpose": identity, "beatthis": {"binding": binding,
                   "local_binding_sha256": sha(binding_path), "actual_parameter_devices": devices,
                   "actual_parameter_dtypes": sorted({str(p.dtype) for p in beat.model.parameters()}),
                   "float16": False, "dbn": False, "library": "official beat_this.inference.File2Beats"},
                   "actual_session_providers": {"detector": model.session_det.get_providers(),
                                                "pose": model.session_pose.get_providers()},
                   "weight_hash_scope": "startup_once; no per-source weight/media rehash or profiles"})


def extract_fullvideo(prepared, context, receipt):
    source, probe, case = prepared["source"], prepared["probe"], Path(prepared["case"])
    api, model, np, torch, progress, budget = (context[k] for k in ("api", "pose_model", "np", "torch", "progress", "budget"))
    n, width, height = probe["frame_count"], probe["width"], probe["height"]
    array_bytes = n * (134 * 2 * 4 + 134 * 4 + 134 + 1 + 4 * 4 + 3 * 8)
    if array_bytes > prepared["reserved_bytes"] - prepared["input_bytes"]:
        raise BudgetStop("native array bytes exceed reserved remaining cache before allocation")
    data = {"xy": np.full((n, 134, 2), np.nan, np.float32),
            "score": np.full((n, 134), np.nan, np.float32),
            "valid": np.zeros((n, 134), np.bool_), "tracked": np.zeros(n, np.bool_),
            "bbox": np.full((n, 4), np.nan, np.float32),
            "frame_indices": np.arange(n, dtype=np.int64),
            "times": probe["pts"] - probe["time_origin"], "source_timestamps": probe["pts"]}
    if sum(v.nbytes for v in data.values()) > prepared["reserved_bytes"] - prepared["input_bytes"]:
        raise BudgetStop("native array bytes do not fit reserved remaining cache budget")
    reasons, previous, done, decoder = [], None, 0, None
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-noautorotate", "-i", source["video_path"],
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0", "-threads", "2",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    receipt["decoder_command"] = command
    progress.update(status="running", stage="fullvideo_pose", current_source=source["id"], frames_done=0, total_frames=n)
    started = time.monotonic()
    stderr_thread, stderr_failures = None, []
    errors = (case / "ffmpeg_decoder.stderr.log").open("xb")
    def drain_stderr():
        kept = 0
        try:
            while True:
                chunk = decoder.stderr.read(8192)
                if not chunk:
                    break
                retained = chunk[:max(0, 65536 - kept)]
                errors.write(retained)
                kept += len(retained)
            errors.flush()
            os.fsync(errors.fileno())
        except Exception as error:
            stderr_failures.append(str(error))
            budget.stop()
    try:
        with errors:
            decoder = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            stderr_thread = threading.Thread(target=drain_stderr, name="pilot-decoder-log", daemon=True)
            stderr_thread.start()
            receipt["decoder_pid"] = decoder.pid
            while True:
                if budget.stopped:
                    raise BudgetStop("stop requested during native frame stream")
                buffer = bytearray(width * height * 3)
                view, filled = memoryview(buffer), 0
                while filled < len(buffer):
                    amount = decoder.stdout.readinto(view[filled:])
                    if not amount:
                        break
                    filled += amount
                if not filled:
                    break
                if filled != len(buffer) or done >= n:
                    raise ValueError("partial or extra native decoded frame; no padding/truncation")
                frame = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
                with torch.inference_mode():
                    points, scores, box, reason = api.select_raw_subject(model(frame), previous, .3)
                reasons.append(reason)
                if box is not None:
                    data["xy"][done], data["score"][done], data["bbox"][done] = points, scores, box
                    data["tracked"][done] = True
                    data["valid"][done] = api.point_valid(points, scores, width, height, .3, tracked=True)
                    previous = box
                done += 1
                prepared["frames_done"] = done
                del frame, view, buffer, points, scores, box
                if done % 30 == 0 or done == n:
                    progress.update(frames_done=done, total_frames=n, source_timestamp=float(probe["pts"][done - 1]))
            rc = decoder.wait(timeout=30)
            stderr_thread.join(timeout=10)
            if stderr_thread.is_alive() or stderr_failures:
                raise RuntimeError("decoder stderr logging did not finish reliably")
            if rc != 0 or done != n:
                raise ValueError("ffmpeg exit/frame count differs from native PTS probe")
    finally:
        if decoder is not None:
            if decoder.stdout is not None:
                decoder.stdout.close()
            if decoder.poll() is None:
                decoder.terminate()
                try:
                    decoder.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    decoder.kill()
                    decoder.wait(timeout=10)
            if stderr_thread is not None:
                stderr_thread.join(timeout=10)
            if decoder.stderr is not None:
                decoder.stderr.close()
            receipt["decoder_returncode"] = decoder.returncode
        if not errors.closed:
            errors.close()
        receipt["decoded_frame_count"] = done
        receipt["native_pts_full_coverage"] = done == n
        receipt["pose_decode_seconds"] = time.monotonic() - started
    coverage = context["wrapper"].coverage(data, np)
    if coverage["effective_valid_nonfinite_xy_count"] or coverage["effective_valid_nonfinite_score_count"]:
        raise ValueError("effective valid observations are nonfinite")
    receipt["coverage"] = {**coverage, "decoded_equals_probed_framecount": True, "interpolation": False,
                           "native_pts_strictly_increasing": True, "frame_indices_contiguous": True}
    receipt["subject_reason_counts"] = dict(Counter(reasons))
    path = case / OUTPUTS[0]
    room = prepared["reserved_bytes"] - tree_bytes(case)
    with path.open("xb") as stream:
        np.savez_compressed(LimitedWriter(stream, room), **data, schema=np.array(SCHEMA), id=np.array(source["id"]),
                            split=np.array(source["split"]), subject_reason=np.asarray(reasons),
                            joint_layout=np.array(json.dumps(api.joint_layout(), sort_keys=True)),
                            video_sha256=np.array(next(a["sha256"] for a in source["assets"] if a["key"] == "video_path")),
                            **{k: np.array(v) for k, v in FLAGS.items()})
        stream.flush()
        os.fsync(stream.fileno())
    return data


def actual_audio(prepared, context):
    source, np, torch, beat = (prepared["source"], context["np"], context["torch"], context["beat_model"])
    context["progress"].update(stage="actual_audio_beats")
    started = time.monotonic()
    with torch.inference_mode():
        beats, downbeats = beat(source["audio_path"])
    torch.cuda.synchronize()
    arrays = {"beats": np.asarray(beats, dtype=np.float64), "downbeats": np.asarray(downbeats, dtype=np.float64)}
    duration = prepared["audio_info"]["duration_seconds"]
    for name, values in arrays.items():
        if (values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0)
                or np.any(np.diff(values) <= 0) or np.any(values > duration + .02)):
            raise ValueError(name + " must be finite/nonnegative/strictly increasing within actual audio duration")
    audio = {"schema": "ms_actual_audio_beats_v1", **FLAGS, "id": source["id"], "split": source["split"],
             "source_audio_path": source["original_paths"]["audio_path"], "actual_input_path": source["audio_path"],
             "audio_sha256": next(a["sha256"] for a in source["assets"] if a["key"] == "audio_path"),
             **prepared["audio_info"], **{k: v.tolist() for k, v in arrays.items()},
             "empty_beats": len(arrays["beats"]) == 0, "empty_downbeats": len(arrays["downbeats"]) == 0,
             "validation_tolerance_s": .02, "beat_accuracy_verified": False,
             "audio_video_alignment_verified": False, "model_identity": context["model_identity"]["beatthis"],
             "inference_seconds": time.monotonic() - started}
    case = Path(prepared["case"])
    write_local(case / OUTPUTS[1], audio, prepared["reserved_bytes"] - tree_bytes(case))
    return audio


def cleanup_success(prepared, context):
    case, cache = Path(prepared["case"]), context["cache"]
    if cache not in case.parents or case.resolve(strict=True) != case:
        raise ValueError("refuse cleanup outside exact owned case root")
    marker = strict_json((case / "own_marker.json").read_bytes())
    if marker != {"runid": context["run"]["runid"], "worker": context["run"]["worker"],
                  "key": prepared["key"], "cache_root": str(cache)}:
        raise ValueError("refuse cleanup without exact own marker")
    expected = {"own_marker.json", "ffmpeg_decoder.stderr.log", "source.tar.gz", *OUTPUTS}
    expected.update(Path(a["local_path"]).name for a in prepared["source"]["assets"])
    for path in case.iterdir():
        if path.is_symlink() or not path.is_file() or path.name not in expected:
            raise ValueError("unknown cache entry; refuse removal")
    shutil.rmtree(case)
    fsync_dir(case.parent)
    context["budget"].release_success(prepared["key"])


def process_source(prepared, context):
    row = prepared["row"]
    result = {"id": row["id"], "split": row["split"], "status": prepared["status"],
              "claim_path": prepared.get("claim_path"), "frames_done": prepared.get("frames_done", 0),
              "stage_seconds": prepared.get("stage_seconds", {}), **FLAGS}
    if prepared["status"] in ("skipped", "unattempted", "failed"):
        for key in ("reason", "error", "failed_stage", "stop", "case"):
            if key in prepared:
                result[key] = prepared[key]
        return result
    source, probe, case, run, ticket = (prepared["source"], prepared["probe"], Path(prepared["case"]), context["run"], context["ticket"])
    output_root = Path(ticket["output_root"])
    shared_case = output_root / row["split"] / row["id"]
    bundle_path = output_root / "bundles" / run["runid"] / (row["split"] + "--" + row["id"] + ".tar.gz")
    started = time.monotonic()
    receipt = {"schema": "ms_fullframe_pilot_case_receipt_v1", **FLAGS, **run, "id": row["id"], "split": row["split"],
               "edition": ticket["edition"], "started": now(), "claim_path": prepared["claim_path"],
               "input_assets": source["assets"], "input_row": row, "stage_seconds": result["stage_seconds"],
               "new_npz_schema": SCHEMA, "original_roi_binding_not_impersonated": True,
               "automatic_retry": False, "finalcaption": 0}
    committed, infer_start, publish_start = False, None, None
    try:
        for asset in source["assets"]:
            if stamp(asset["local_path"]) != asset["local_stat_after_copy"]:
                raise ValueError("local copied input changed before inference")
        infer_start = time.monotonic()
        receipt["current_stage"] = "fullvideo_pose"
        data = extract_fullvideo(prepared, context, receipt)
        receipt["current_stage"] = "actual_audio"
        audio = actual_audio(prepared, context)
        result["stage_seconds"]["inference"] = time.monotonic() - infer_start
        for asset in source["assets"]:
            if stamp(asset["local_path"]) != asset["local_stat_after_copy"]:
                raise ValueError("local copied input changed during inference")
        manifest = {"schema": "ms_fullframe_pose2d_manifest_v1", **FLAGS, "id": row["id"], "split": row["split"],
                    "npz_path": str(shared_case / OUTPUTS[0]), "video_path": row["video_path"],
                    "video_sha256": next(a["sha256"] for a in source["assets"] if a["key"] == "video_path"),
                    "frame_count": probe["frame_count"], "width": probe["width"], "height": probe["height"],
                    "pts_origin": probe["time_origin"], "duration_seconds": probe["duration"],
                    "frame_indices": data["frame_indices"].tolist(), "times": data["times"].tolist(),
                    "source_timestamps": data["source_timestamps"].tolist(), "joint_layout": context["api"].joint_layout(),
                    "score_threshold": .3, "score_is_probability": False, "interpolation": False,
                    "metadata_provenance": {"probe": {k: v for k, v in probe.items() if k != "pts"},
                                            "model_identity": context["model_identity"], **context["binding"],
                                            "copy_assets": source["assets"], "meta_usage": "provenance_only",
                                            "source_meta_path": row["meta_path"]}}
        del data
        write_local(case / OUTPUTS[2], manifest, prepared["reserved_bytes"] - tree_bytes(case))
        result["status"] = "partial" if receipt["coverage"]["missing_subject_frames"] or audio["empty_beats"] or audio["empty_downbeats"] else "success"
        receipt.update(status=result["status"], finished=now(), audio_empty_beats=audio["empty_beats"],
                       audio_empty_downbeats=audio["empty_downbeats"], current_stage="write_once_publication",
                       bundle_path=str(bundle_path), commit_policy="receipt linked last after verified bundle and three other outputs",
                       outputs={name: {"path": str(shared_case / name), "bytes": (case / name).stat().st_size,
                                       "sha256": sha(case / name)} for name in OUTPUTS[:-1]})
        receipt["case_compute_seconds"] = time.monotonic() - started
        write_local(case / OUTPUTS[3], receipt, prepared["reserved_bytes"] - tree_bytes(case))
        if tree_bytes(case) * 2 > prepared["reserved_bytes"] + prepared["input_bytes"]:
            raise BudgetStop("actual output plus pending bundle exceeds conservative cache reservation")
        receipt_sha = sha(case / OUTPUTS[3])
        publish_start = time.monotonic()
        context["progress"].update(stage="publish", queuedwriteback=1)
        room = prepared["reserved_bytes"] - tree_bytes(case)
        with (case / "source.tar.gz").open("xb") as stream:
            with tarfile.open(fileobj=LimitedWriter(stream, room), mode="w:gz") as bundle:
                for name in OUTPUTS:
                    bundle.add(case / name, arcname=name, recursive=False)
            stream.flush()
            os.fsync(stream.fileno())
        if tree_bytes(case) > prepared["reserved_bytes"]:
            raise BudgetStop("local inputs/output/bundle exceed reserved cache bytes")
        bundle_sha = sha(case / "source.tar.gz")
        published_bundle = publish_file(case / "source.tar.gz", bundle_path, bundle_sha, output_root)
        result["bundle"] = published_bundle
        # An existing directory (even empty) belongs to another publication; never merge into it.
        mkdir_scoped(shared_case.parent, output_root)
        shared_case.mkdir(exist_ok=False)
        fsync_dir(shared_case.parent)
        published = {}
        result["outputs"] = published
        for name in OUTPUTS[:-1]:
            published[name] = publish_file(case / name, shared_case / name, receipt["outputs"][name]["sha256"], output_root)
        published[OUTPUTS[3]] = publish_file(case / OUTPUTS[3], shared_case / OUTPUTS[3], receipt_sha, output_root)
        committed = True
        result.update(bundle=published_bundle, outputs=published, publication_committed=True, receipt_path=str(shared_case / OUTPUTS[3]),
                      frames_done=prepared["frames_done"], case_total_seconds=time.monotonic() - started)
        result["stage_seconds"]["publish"] = time.monotonic() - publish_start
        cleanup_success(prepared, context)
        result["cache_reclaimed"] = True
        context["progress"].update(queuedwriteback=0)
    except Exception as error:
        if infer_start is not None:
            result["stage_seconds"].setdefault("inference", time.monotonic() - infer_start)
        if publish_start is not None:
            result["stage_seconds"]["publish"] = time.monotonic() - publish_start
        result.update(status="failed", error=type(error).__name__ + ": " + str(error),
                      failed_stage=receipt.get("current_stage", "publication_or_cleanup"),
                      frames_done=prepared["frames_done"], case=str(case),
                      stop=isinstance(error, BudgetStop), publication_committed=committed,
                      case_total_seconds=time.monotonic() - started)
        context["budget"].retain_failed(prepared["key"], tree_bytes(case))
        if result["stop"]:
            context["budget"].stop()
        context["progress"].update(queuedwriteback=0)
    return result


def record_failure(result, context):
    value = {"schema": "ms_fullframe_pilot_failure_v1", **context["run"], **result,
             "time": now(), "automatic_retry": False, "claim_and_failed_cache_retained": True}
    local = context["local_run"] / "errors" / result["split"]
    mkdir_scoped(local, context["local_run"])
    write_local(local / (result["id"] + ".json"), value)
    output = Path(context["ticket"]["output_root"])
    publish_json(value, output / "errors" / context["run"]["runid"] / result["split"] / (result["id"] + ".json"), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--validated-wrapper", required=True)
    parser.add_argument("--check-only", action="store_true", help="validate exact assignment/scope; no claims, writes, models or GPU operations")
    args = parser.parse_args()
    root, ticket, rows, wrapper, code_root, binding = validate(args)
    if args.check_only:
        print(json.dumps({"status": "check_only_pass", "worker": args.worker, "root": str(root),
                          "source_count": len(rows), **binding, "claims_created": 0, "models_loaded": 0,
                          "shared_writes": 0, "finalcaption": 0, **FLAGS}, allow_nan=False), flush=True)
        return 0
    if Path(sys.prefix).resolve() != (root / "venv").resolve():
        raise RuntimeError("must use the own node-local venv")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    memory = wrapper.memory_contract(root)
    gpu = wrapper.gpu_gate(fresh=True)
    runid = args.worker + "-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
    run = {**wrapper.process_identity(), "worker": args.worker, "worker_id": args.worker,
           "runid": runid, "started": now(), "model_init_counts": {"dwpose": 0, "beatthis": 0},
           "copy_threads": 1, "max_inflight_sources": 2, "models_loaded_once_per_worker": True,
           "physical_gpu_index": 0, "edition": ticket["edition"], "finalcaption": 0}
    budget = Budget(8 * GiB)
    local_run = root / "pilot_runs" / runid
    mkdir_scoped(local_run.parent, root)
    local_run.mkdir(exist_ok=False)
    cache = root / "cache" / "pilot_fullframe" / runid
    mkdir_scoped(cache.parent, root)
    cache.mkdir(exist_ok=False)
    context = {"root": root, "ticket": ticket, "wrapper": wrapper, "code_root": code_root,
               "binding": binding, "run": run, "budget": budget, "cache": cache, "local_run": local_run}
    progress = Progress(root, run, budget, len(rows))
    context["progress"] = progress
    def stop_signal(number, frame):
        budget.stop()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, stop_signal)
    start, results, futures, next_index, executor = time.monotonic(), [], deque(), 0, None
    summary = {"schema": "ms_fullframe_pilot_summary_v1", **FLAGS, **run, **binding,
               "memory_contract": memory, "gpu0_initial_gate": gpu, "source_count": len(rows),
               "output_root": ticket["output_root"], "status": "running", "cases": results,
               "automatic_retry": False, "caption_completion": False, "speedup_claimed": False}
    progress.start()
    try:
        load_models(context)
        summary["gpu0_after_model_initialization_gate"] = wrapper.gpu_gate(fresh=False)
        progress.update(stage="models_ready", model_init_counts=run["model_init_counts"], event="models_loaded_once")
        identity = {"schema": "ms_fullframe_pilot_run_identity_v1", **FLAGS, **run, **binding,
                    "memory_contract": memory, "gpu0_initial_gate": gpu,
                    "model_identity": context["model_identity"], "source_count": len(rows),
                    "output_root": ticket["output_root"], "models_reused_for_all_sources": True}
        write_local(local_run / "run_identity.json", identity)
        output = Path(ticket["output_root"])
        publish_json(identity, output / "pilot_runs" / runid / "run_identity.json", output)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pilot-copy")
        while next_index < len(rows) or futures:
            while not budget.stopped and next_index < len(rows) and len(futures) < 2:
                futures.append((rows[next_index], executor.submit(claim_and_copy, rows[next_index], context)))
                next_index += 1
            if not futures:
                break
            progress.update(inflight=len(futures))
            submitted_row, future = futures.popleft()
            try:
                prepared = future.result()
            except Exception as error:
                budget.stop()
                prepared = {"row": submitted_row, "status": "failed", "stop": True,
                            "error": type(error).__name__ + ": " + str(error),
                            "failed_stage": "copy_future_exception",
                            "claim_path": str(Path(ticket["claims_root"]) / submitted_row["split"] / (submitted_row["id"] + ".json"))}
            if prepared["status"] == "unattempted":
                continue
            result = process_source(prepared, context)
            results.append(result)
            if result["status"] == "failed":
                record_failure(result, context)
            progress.finish_case(result)
            if result.get("stop"):
                budget.stop()
        summary["status"] = "stopped" if budget.stopped else "failed" if any(r["status"] == "failed" for r in results) else "partial" if any(r["status"] == "partial" for r in results) else "success"
    except Exception as error:
        budget.stop()
        summary.update(status="failed", error=type(error).__name__ + ": " + str(error))
    finally:
        if executor is not None:
            budget.stop()
            executor.shutdown(wait=True, cancel_futures=False)
        # Drain each already submitted source so a claim made during a stop has a failure report.
        for submitted_row, future in futures:
            try:
                try:
                    prepared = future.result()
                except Exception as error:
                    prepared = {"row": submitted_row, "status": "failed", "stop": True,
                                "error": type(error).__name__ + ": " + str(error),
                                "failed_stage": "copy_future_exception_during_stop",
                                "claim_path": str(Path(ticket["claims_root"]) / submitted_row["split"] / (submitted_row["id"] + ".json"))}
                if prepared["status"] == "unattempted":
                    continue
                if prepared["status"] == "prepared":
                    prepared.update(status="failed", error="run stopped after copy; no automatic retry",
                                    failed_stage="stopped_before_inference", stop=True)
                    budget.retain_failed(prepared["key"], tree_bytes(prepared["case"]))
                result = process_source(prepared, context)
                results.append(result)
                if result["status"] == "failed":
                    record_failure(result, context)
                progress.finish_case(result)
            except Exception as error:
                summary.setdefault("shutdown_errors", []).append(type(error).__name__ + ": " + str(error))
        summary.update(finished=now(), e2e_seconds=time.monotonic() - start,
                       actual_processed=sum(r["status"] != "skipped" for r in results),
                       skipped=sum(r["status"] == "skipped" for r in results),
                       failed=sum(r["status"] == "failed" for r in results),
                       success=sum(r["status"] == "success" for r in results),
                       partial=sum(r["status"] == "partial" for r in results),
                       totalframes=sum(r.get("frames_done", 0) for r in results),
                       framecounts={r["split"] + "/" + r["id"]: r.get("frames_done", 0) for r in results},
                       remaining=len(rows) - len(results), cursor=len(results),
                       model_init_counts=run["model_init_counts"],
                       output_bundle_paths=[r["bundle"]["path"] for r in results if r.get("bundle")],
                       **budget.snapshot())
        if not (local_run / "run_identity.json").exists():
            identity = {"schema": "ms_fullframe_pilot_run_identity_v1", **FLAGS, **run, **binding,
                        "status": "initialization_failed", "error": summary.get("error"),
                        "memory_contract": memory, "gpu0_initial_gate": gpu}
            write_local(local_run / "run_identity.json", identity)
            try:
                output = Path(ticket["output_root"])
                publish_json(identity, output / "pilot_runs" / runid / "run_identity.json", output)
            except Exception as error:
                summary["identity_publication_error"] = str(error)
        write_local(local_run / "summary.json", summary)
        try:
            output = Path(ticket["output_root"])
            publish_json(summary, output / "pilot_runs" / runid / "summary.json", output)
        except Exception as error:
            summary["summary_publication_error"] = str(error)
            summary["status"] = "failed"
            print(json.dumps({"status": "summary_publication_failed", "error": str(error),
                              "local_summary": str(local_run / "summary.json")}), file=sys.stderr, flush=True)
        try:
            progress.update(status=summary["status"], event="run_finished", inflight=0, queuedwriteback=0,
                            current_source=None, remaining=summary["remaining"], cursor=summary["cursor"],
                            model_init_counts=run["model_init_counts"], e2e_seconds=summary["e2e_seconds"],
                            summary_path=str(local_run / "summary.json"))
        finally:
            progress.close()
        print(json.dumps({"status": summary["status"], "runid": runid, "summary_path": str(local_run / "summary.json"),
                          "actual_processed": summary["actual_processed"], "skipped": summary["skipped"],
                          "failed": summary["failed"], "remaining": summary["remaining"], "finalcaption": 0}), flush=True)
    return 1 if summary["status"] in ("failed", "stopped") or summary.get("summary_publication_error") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "rejected", "error": type(error).__name__ + ": " + str(error),
                          "cursor": 0, "automatic_retry": False, "finalcaption": 0, **FLAGS}),
              file=sys.stderr, flush=True)
        raise SystemExit(1)
