"""Local, video-only open-vocabulary observations over explicit manifest windows.

prepare never imports vLLM. run uses local vLLM LLM.chat, not a service or a
model download. Every output is an unreviewed candidate, abstention, or failure;
prepared evidence alone is never counted as a successful VLM observation.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

# Append only an explicitly supplied existing CPU dependency directory. Do not
# prepend another environment: native vLLM/Torch packages must keep precedence.
_cpu_site = os.environ.get("MS_VISUAL_CPU_SITEPATH")
if _cpu_site:
    if not Path(_cpu_site).is_absolute() or not Path(_cpu_site).is_dir():
        raise ValueError("MS_VISUAL_CPU_SITEPATH must be an existing absolute dependency directory")
    sys.path.append(_cpu_site)

from ms_optimize import atomic_write, json_bytes, overlaps, sha256
from ms_visual_observe import SCHEMA as VISUAL_SCHEMA, SYSTEM, parse_response, strict_json, validate_candidate
from motionscript_v2.chunks import fixed_windows

SCHEMA = "motionscript_local_visual_batch_v1"
FIELDS = ("id", "video_path", "audio_path", "nlf_path", "source_group", "split")
SOURCE_FIELDS = ("video_path", "audio_path", "nlf_path")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def snapshot(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "device": stat.st_dev, "inode": stat.st_ino}


def file_identity(path):
    path = path.resolve(strict=True)
    before = snapshot(path)
    value = {"path": str(path), "sha256": sha256(path), "stat": before}
    if snapshot(path) != before:
        raise ValueError(f"file changed while hashing: {path}")
    return value


def identity_tree(root, paths):
    files = {str(path.relative_to(root)): file_identity(path) for path in paths}
    content = {name: {"path": row["path"], "sha256": row["sha256"]}
               for name, row in files.items()}
    return {"root": str(root), "files": files, "sha256": digest({"root": str(root), "files": content}),
            "full_content_hash_computed_once_per_process": True}


def model_files(root):
    return sorted(path for path in root.rglob("*") if path.is_file()
                  and not any(part.startswith(".") for part in path.relative_to(root).parts))


def model_identity(model):
    root = Path(model).resolve(strict=True)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError("--model must be an existing local model directory with config.json")
    paths = model_files(root)
    if not any(path.suffix in (".safetensors", ".bin", ".pt", ".gguf") for path in paths):
        raise ValueError("local model directory contains no model weight files")
    return identity_tree(root, paths)


def code_identity():
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ("ms_visual_batch.py", "ms_visual_observe.py", "ms_optimize.py")]
    paths.extend(sorted((root / "motionscript_v2").glob("*.py")))
    return identity_tree(root, paths)


def verify_current(identity, *, rehash=False, model=False):
    root = Path(identity["root"])
    if model and {str(p.relative_to(root)) for p in model_files(root)} != set(identity["files"]):
        raise ValueError("model file inventory changed before loading")
    for name, value in identity["files"].items():
        path = root / name
        if str(path.resolve(strict=True)) != value["path"] or snapshot(path) != value["stat"]:
            raise ValueError(f"identity file changed before loading: {path}")
        if rehash and sha256(path) != value["sha256"]:
            raise ValueError(f"identity content SHA changed before loading: {path}")


def read_manifest(path, shard=0, nshard=1, limit=0):
    raw = Path(path).read_bytes()
    rows, seen = [], set()
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = strict_json(line)
        if not isinstance(value, dict) or any(key not in value for key in FIELDS):
            raise ValueError(f"manifest line {number} must explicitly contain {FIELDS}")
        row = {key: value[key] for key in FIELDS}
        if not isinstance(row["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", row["id"]):
            raise ValueError(f"manifest line {number} has an unsafe ID")
        if row["id"].casefold() in seen:
            raise ValueError(f"duplicate manifest ID: {row['id']}")
        seen.add(row["id"].casefold())
        if row["split"] not in ("train", "val", "validation", "test"):
            raise ValueError(f"manifest line {number}: split must explicitly declare train/val/validation/test")
        if not isinstance(row["source_group"], str) or not row["source_group"].strip():
            raise ValueError(f"manifest line {number}: source_group must be a nonempty string")
        for key in SOURCE_FIELDS:
            if not isinstance(row[key], str) or not Path(row[key]).is_absolute():
                raise ValueError(f"manifest line {number}: {key} must be an explicit absolute path")
        row["input_line"] = number
        row["input_row_sha256"] = digest(value)
        rows.append(row)
    if not rows:
        raise ValueError("empty input manifest")
    selected = rows[shard::nshard]
    return (selected[:limit] if limit else selected), hashlib.sha256(raw).hexdigest()


def source_identity(row):
    files = {}
    for key in SOURCE_FIELDS:
        path = Path(row[key])
        if not path.is_file():
            raise ValueError(f"{key} is missing; no placeholder asset is admitted: {path}")
        files[key] = file_identity(path)
    return {"files": files, "sha256": digest({"row": row, "files": {
        key: {"path": value["path"], "sha256": value["sha256"]} for key, value in files.items()}}),
        "audio_and_nlf_are_provenance_only_not_model_inputs": True}


def verify_source(source):
    for value in source["files"].values():
        path = Path(value["path"])
        if snapshot(path) != value["stat"]:
            raise ValueError(f"source asset changed: {path}")


def probe_video(video, ffprobe):
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_streams", "-show_format", "-show_entries",
        "frame=best_effort_timestamp_time,pkt_duration_time,duration_time:stream=duration:format=duration",
        "-of", "json", str(video)], check=True, capture_output=True)
    value = json.loads(result.stdout)
    frames = value.get("frames", [])
    raw_times = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    if (len(raw_times) < 2 or any(not math.isfinite(t) for t in raw_times)
            or any(b <= a for a, b in zip(raw_times, raw_times[1:]))):
        raise ValueError("video needs at least two finite strictly increasing decoded frame timestamps")
    times = [t - raw_times[0] for t in raw_times]
    duration = value.get("streams", [{}])[0].get("duration")
    basis = "ffprobe video stream duration"
    if duration is None or duration == "N/A":
        last = frames[-1]
        tail_duration = last.get("duration_time", last.get("pkt_duration_time"))
        if tail_duration is not None and tail_duration != "N/A" and float(tail_duration) > 0:
            duration = times[-1] + float(tail_duration)
            basis = "ffprobe final decoded frame timestamp plus its declared duration"
        else:
            raise ValueError("ffprobe has no video stream duration or final frame duration; container/audio duration cannot define video_end")
    end = float(duration)
    if not math.isfinite(end) or end <= 0 or end <= times[-1]:
        raise ValueError("ffprobe did not provide a positive media_end beyond the last decoded frame")
    return {"times": times, "source_timestamps": raw_times, "media_start": 0.0, "media_end": end,
            "media_end_basis": basis, "time_origin_source_timestamp": raw_times[0],
            "probe_sha256": hashlib.sha256(result.stdout).hexdigest()}


def select_frames(times, start, end, sample_fps=4.0, max_frames=16):
    # Match ms_visual_observe.prepare: first/last frame, timestamp-rate selection,
    # then evenly reduce the selected list. Indices stay global in stream 0:v:0.
    available = [i for i, t in enumerate(times) if t >= start and t < end]
    if len(available) < 2:
        raise ValueError("requested window contains fewer than two frames")
    selected = [available[0]]
    for index in available[1:]:
        if times[index] - times[selected[-1]] >= 1 / sample_fps - 1e-9:
            selected.append(index)
    if selected[-1] != available[-1]:
        selected.append(available[-1])
    if len(selected) > max_frames:
        selected = [selected[round(k * (len(selected) - 1) / (max_frames - 1))] for k in range(max_frames)]
    return selected


def params_identity(args, code):
    value = {"chunk_seconds": 2.0, "min_tail_seconds": 1.5, "sample_fps": 4.0, "max_frames": 16,
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(), "code_sha256": code["sha256"],
        "model_path": str(Path(args.model).resolve()), "tensor_parallel_size": args.tp,
        "batch_size": args.batch_size, "max_num_seqs": args.max_num_seqs,
        "max_model_len": 16384, "max_pixels": args.max_pixels, "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": 0.0, "max_tokens": 1200, "seed": 0, "trust_remote_code": False,
        "image_limit": 16, "ffmpeg": str(Path(args.ffmpeg).resolve()),
        "ffprobe": str(Path(args.ffprobe).resolve())}
    return {"values": value, "sha256": digest(value)}


def binding_for(row, source, model, params, window):
    return {"id": row["id"], "source_sha256": source["sha256"], "model_sha256": model["sha256"],
            "params_sha256": params["sha256"], "window": window}


@contextmanager
def bundle_lock(bundle):
    bundle.mkdir(parents=True, exist_ok=True)
    with (bundle / ".lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"bundle is active in another managed process: {bundle}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prepare_window(bundle, row, source, probe, window, binding, args):
    manifest_path = bundle / "manifest.json"
    if manifest_path.exists():
        return verify_evidence(bundle, binding, source)
    bundle.mkdir(parents=True, exist_ok=True)
    selected = select_frames(probe["times"], window["start"], window["end"])
    # Each attempt gets new frame files: interrupted extraction never overwrites
    # evidence that may already be referenced by a raw response.
    import tempfile
    frame_dir = Path(tempfile.mkdtemp(prefix="frames-", dir=bundle))
    expression = "+".join(f"eq(n\\,{index})" for index in selected)
    subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-i", row["video_path"], "-map", "0:v:0",
        "-vf", f"select={expression},scale=640:640:force_original_aspect_ratio=decrease",
        "-fps_mode", "vfr", "-q:v", "3", str(frame_dir / "%06d.jpg")], check=True, capture_output=True)
    images = sorted(frame_dir.glob("*.jpg"))
    verify_source(source)
    if len(images) != len(selected):
        raise ValueError("frame extraction count mismatch")
    evidence, content = [], [{"type": "text", "text":
        "Observe this ordered frame sequence. Only describe supported body movements."}]
    for index, image in zip(selected, images):
        evidence.append({"index": index, "time": probe["times"][index],
            "source_timestamp": probe["source_timestamps"][index], "file": image.relative_to(bundle).as_posix(),
            "sha256": sha256(image)})
        content.extend([{"type": "text", "text": f"Source frame {index}, time {probe['times'][index]:.6f} seconds."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                base64.b64encode(image.read_bytes()).decode("ascii")}}])
    request = {"model": str(Path(args.model).resolve()), "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        "temperature": 0.0, "max_tokens": 1200}
    atomic_write(bundle / "request.json", json_bytes(request))
    manifest = {"schema": VISUAL_SCHEMA, "batch_schema": SCHEMA, "binding": binding, "input": row,
        "video_path": row["video_path"], "video_sha256": source["files"]["video_path"]["sha256"],
        "source_identity": source, "frames": evidence, "request_sha256": sha256(bundle / "request.json"),
        "window": window, "media_end": probe["media_end"], "media_end_basis": probe["media_end_basis"],
        "probe_sha256": probe["probe_sha256"], "video_stream_selector": "0:v:0 (ffprobe v:0)",
        "sampling_policy": {"requested_fps": 4.0, "max_frames": 16, "half_open_window_bounds": True,
            "full_continuous_video_observed_by_model": False},
        "window_boundaries_are_action_boundaries": False, "rule_labels_supplied_to_observer": False,
        "semantic_approval": False, "human_review_completed": False}
    raw = json_bytes(manifest)
    atomic_write(bundle / "manifest.receipt.json", json_bytes({"manifest_sha256": hashlib.sha256(raw).hexdigest()}))
    atomic_write(manifest_path, raw)
    return manifest


def verify_evidence(bundle, binding, source):
    raw = (bundle / "manifest.json").read_bytes()
    receipt = strict_json((bundle / "manifest.receipt.json").read_text())
    manifest = strict_json(raw)
    if receipt.get("manifest_sha256") != hashlib.sha256(raw).hexdigest() or manifest.get("binding") != binding:
        raise ValueError("prepared evidence binding or manifest SHA mismatch")
    if manifest.get("video_sha256") != source["files"]["video_path"]["sha256"]:
        raise ValueError("prepared video SHA mismatch")
    verify_source(source)
    if sha256(bundle / "request.json") != manifest["request_sha256"]:
        raise ValueError("prepared request was altered")
    for frame in manifest["frames"]:
        image = (bundle / frame["file"]).resolve()
        if not image.is_relative_to(bundle.resolve()) or sha256(image) != frame["sha256"]:
            raise ValueError("frame evidence is missing, changed or outside bundle")
    return manifest


def validate_generation(bundle, *, required=False):
    path = bundle / "generation.json"
    if not path.is_file():
        if required:
            raise ValueError("production raw response lacks generation/EOS statistics; interrupted response is not complete")
        return  # Validator-only CPU fixtures do not impersonate model calls.
    value = strict_json(path.read_text())
    if required and value.get("raw_response_sha256") != sha256(bundle / "response.raw.txt"):
        raise ValueError("generation statistics do not match raw response SHA")
    if value["finish_reason"] != "stop" or value["completion_tokens"] >= value["max_tokens"]:
        raise ValueError(f"local VLM completion truncated or not normally stopped: {value['finish_reason']}")


def finish_response(bundle, binding, manifest, raw):
    raw_path = bundle / "response.raw.txt"
    if raw_path.exists() and raw_path.read_bytes() != raw:
        raise ValueError("raw response already exists with different content; refusing overwrite")
    atomic_write(raw_path, raw)
    try:
        validate_generation(bundle, required=manifest.get("batch_schema") == SCHEMA)
        # The original parser accepts JSON and an encoded assistant string. This
        # wrapper preserves raw model text, including fenced JSON, unchanged.
        candidate = validate_candidate(parse_response(json_bytes(raw.decode("utf-8"))), manifest)
        candidate.update(video_sha256=manifest["video_sha256"], request_sha256=manifest["request_sha256"],
                         raw_response_sha256=hashlib.sha256(raw).hexdigest(), binding=binding)
        atomic_write(bundle / "candidate.json", json_bytes(candidate))
        status, error = ("abstain" if candidate["abstain"] else "candidate"), None
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as failure:
        status, error = "failed", str(failure)
    return write_receipt(bundle, binding, status, error=error)


def write_receipt(bundle, binding, status, *, error=None):
    files = {}
    for name in ("manifest.json", "request.json", "response.raw.txt", "generation.json", "candidate.json"):
        if (bundle / name).is_file():
            files[name] = sha256(bundle / name)
    value = {"schema": SCHEMA, "binding": binding, "status": status, "error": error, "files_sha256": files,
        "semantic_approval": False, "human_review_completed": False, "training_entry_updated": False}
    atomic_write(bundle / "receipt.json", json_bytes(value))
    return value


def resume_receipt(bundle, binding, source):
    if not (bundle / "receipt.json").is_file():
        return None
    value = strict_json((bundle / "receipt.json").read_text())
    if value.get("binding") != binding or value.get("status") not in ("candidate", "abstain", "failed", "prepared"):
        raise ValueError("window receipt binding or status mismatch")
    for name, expected in value["files_sha256"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path) != expected:
            raise ValueError("window artifact changed; cannot resume")
    if value["status"] in ("candidate", "abstain"):
        manifest = verify_evidence(bundle, binding, source)
        validate_generation(bundle, required=manifest.get("batch_schema") == SCHEMA)
        raw = (bundle / "response.raw.txt").read_bytes()
        checked = validate_candidate(parse_response(json_bytes(raw.decode("utf-8"))), manifest)
        checked.update(video_sha256=manifest["video_sha256"], request_sha256=manifest["request_sha256"],
                       raw_response_sha256=hashlib.sha256(raw).hexdigest(), binding=binding)
        if strict_json((bundle / "candidate.json").read_text()) != checked:
            raise ValueError("candidate differs from original response")
        if (value["status"] == "abstain") != checked["abstain"]:
            raise ValueError("window receipt status disagrees with candidate")
    if value["status"] == "prepared":
        verify_evidence(bundle, binding, source)
        return None
    return value


def select_attempt(window_root, binding, source, retry_failed=False):
    attempts = sorted(path for path in window_root.glob("attempt-*")
                      if path.is_dir() and re.fullmatch(r"attempt-[0-9]{6}", path.name))
    bundle = attempts[-1] if attempts else window_root
    receipt = resume_receipt(bundle, binding, source)
    if retry_failed and receipt is not None and receipt["status"] == "failed":
        number = int(attempts[-1].name.split("-")[1]) + 1 if attempts else 1
        bundle = window_root / f"attempt-{number:06d}"
        bundle.mkdir()
        receipt = None
    return bundle, receipt


def prepare_clip(row, out, model, params, args):
    source = source_identity(row)
    verify_source(source)
    probe = probe_video(row["video_path"], args.ffprobe)
    windows = fixed_windows(0.0, probe["media_end"], 2.0, 1.5)
    clip_key = digest({"source_sha256": source["sha256"], "model_sha256": model["sha256"],
                       "params_sha256": params["sha256"]})
    clip = out / "ids" / row["id"] / clip_key
    work = []
    for index, window in enumerate(windows):
        binding = binding_for(row, source, model, params, window)
        window_root = clip / f"w{index:04d}"
        with bundle_lock(window_root):
            bundle, receipt = select_attempt(window_root, binding, source, args.retry_failed)
            try:
                if receipt is None:
                    manifest = prepare_window(bundle, row, source, probe, window, binding, args)
                    if (bundle / "response.raw.txt").is_file():
                        receipt = finish_response(bundle, binding, manifest, (bundle / "response.raw.txt").read_bytes())
                    else:
                        write_receipt(bundle, binding, "prepared")
            except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
                # Existing receipts are not rewritten when integrity validation
                # fails. The enclosing row is a failure rather than a resume.
                if (bundle / "receipt.json").exists():
                    raise
                receipt = write_receipt(bundle, binding, "failed", error=f"prepare failed: {error}")
            work.append({"bundle": str(bundle), "window_root": str(window_root), "binding": binding, "receipt": receipt})
    return {"row": row, "clip": str(clip), "source": source, "windows": work,
            "media_end": probe["media_end"], "media_end_basis": probe["media_end_basis"]}


def final_caption(clip):
    segments, counts = [], {"candidate": 0, "abstain": 0, "failed": 0, "prepared": 0}
    for item in clip["windows"]:
        bundle, receipt = Path(item["bundle"]), item["receipt"]
        status = receipt["status"] if receipt else "prepared"
        counts[status] += 1
        window = item["binding"]["window"]
        candidate = strict_json((bundle / "candidate.json").read_text()) if status == "candidate" else None
        segments.append({**window, "status": status, "caption": candidate["motion_caption"] if candidate else "",
            "events": candidate["events"] if candidate else [], "bundle": str(bundle),
            "receipt_sha256": sha256(bundle / "receipt.json") if (bundle / "receipt.json").is_file() else None})
    # Only verbatim window captions are concatenated. No second LLM, summary,
    # inferred transition, gap filling or new whole-clip facts are introduced.
    text = "\n".join(f"[{s['start']:.3f}-{s['end']:.3f}s] {s['caption']}" for s in segments if s["caption"])
    complete = counts["prepared"] == 0 and counts["failed"] == 0
    status = ("failed" if counts["failed"] else "prepared" if counts["prepared"] else
              "abstain" if not counts["candidate"] else "candidate")
    value = {"schema": SCHEMA, "id": clip["row"]["id"], "input": clip["row"], "status": status,
        "source_sha256": clip["source"]["sha256"],
        "model_sha256": clip["windows"][0]["binding"]["model_sha256"],
        "params_sha256": clip["windows"][0]["binding"]["params_sha256"], "media_end": clip["media_end"],
        "media_end_basis": clip["media_end_basis"], "counts": counts, "segments": segments,
        "timed_caption": text, "observation_complete_without_failures": complete,
        "whole_clip_summary_generated": False, "semantic_approval": False, "human_review_completed": False,
        "training_entry_updated": False, "caption_policy": "verbatim_candidate_windows_only; abstain_and_failed_gaps_not_filled"}
    destination = Path(clip["clip"]) / "final.json"
    atomic_write(destination, json_bytes(value))
    return {"id": value["id"], "status": status, "counts": counts, "final_path": str(destination),
            "final_sha256": sha256(destination), "observation_complete_without_failures": complete}


def load_local_model(args, model, code, out):
    # Full model SHA was computed once at startup, never once per SID/window.
    # Recheck the exact file tree and unchanged resolved stat snapshot directly
    # before loading; source Python is additionally rehashed at this boundary.
    verify_current(model, model=True)
    verify_current(code, rehash=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    from vllm import LLM, SamplingParams
    import vllm
    receipt = {"schema": SCHEMA, "model_identity": model, "code_identity": code,
        "model_content_sha_verified_at_startup": True, "current_model_inventory_and_stat_verified_before_load": True,
        "current_code_sha_verified_before_load": True, "runtime_vllm_version": vllm.__version__,
        "transport": "local_vllm_LLM.chat", "third_party_endpoint_used": False, "model_download_allowed": False,
        "runtime_python": sys.executable, "explicit_cpu_sitepath_appended": _cpu_site,
        "cpu_sitepath_is_not_prepend": True}
    atomic_write(out / "load.receipt.json", json_bytes(receipt))
    llm = LLM(model=str(Path(args.model).resolve()), tensor_parallel_size=args.tp, max_model_len=16384,
        max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 16, "video": 0}, mm_processor_kwargs={"max_pixels": args.max_pixels},
        enable_prefix_caching=True, trust_remote_code=False, seed=0)
    return llm, SamplingParams(temperature=0.0, max_tokens=1200)


def observe_batch(items, llm, sampling, source):
    # Hold nonblocking file locks through chat and atomic writes. Overlapping
    # shards cannot submit or overwrite a window concurrently.
    from contextlib import ExitStack
    with ExitStack() as locks:
        pending, messages = [], []
        for item in items:
            bundle = Path(item["bundle"])
            locks.enter_context(bundle_lock(Path(item["window_root"])))
            receipt = resume_receipt(bundle, item["binding"], source)
            if receipt:
                item["receipt"] = receipt
                continue
            manifest = verify_evidence(bundle, item["binding"], source)
            request = strict_json((bundle / "request.json").read_text())
            messages.append(request["messages"])
            pending.append((item, manifest))
        if not pending:
            return
        try:
            outputs = llm.chat(messages, sampling, use_tqdm=False)
            if len(outputs) != len(pending):
                raise ValueError("vLLM output count differs from submitted windows")
        except Exception as error:
            for item, _ in pending:
                item["receipt"] = write_receipt(Path(item["bundle"]), item["binding"], "failed",
                                                 error=f"local vLLM chat failed: {type(error).__name__}: {error}")
            return
        for (item, manifest), output in zip(pending, outputs):
            bundle = Path(item["bundle"])
            try:
                if len(output.outputs) != 1:
                    raise ValueError("expected exactly one local model completion")
                completion = output.outputs[0]
                raw = completion.text.encode("utf-8")
                tokenizer = llm.get_tokenizer()
                eos = tokenizer.eos_token_id
                tokens = list(completion.token_ids)
                stats = {"finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                    "prompt_tokens": len(output.prompt_token_ids or []), "completion_tokens": len(tokens),
                    "max_tokens": 1200, "eos_token_id": eos, "last_output_token_id": tokens[-1] if tokens else None,
                    "last_output_token_is_eos": bool(tokens) and tokens[-1] == eos,
                    "eos_is_not_required_in_returned_token_ids": True, "model_call_performed": True,
                    "raw_response_sha256": hashlib.sha256(raw).hexdigest()}
                atomic_write(bundle / "generation.json", json_bytes(stats))
                atomic_write(bundle / "response.raw.txt", raw)
                verify_source(source)
                item["receipt"] = finish_response(bundle, item["binding"], manifest, raw)
            except (ValueError, OSError, KeyError, TypeError) as error:
                item["receipt"] = write_receipt(bundle, item["binding"], "failed", error=str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), default="run")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True, help="Existing absolute local model path, never a hub ID")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="IDs on this shard, not a successful-output target")
    parser.add_argument("--batch-size", type=int, default=1, help="Window conversations per local LLM.chat call")
    parser.add_argument("--prepare-workers", type=int, default=1, help="Bounded joined CPU workers; no detached children")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only failed windows in new attempt directories; preserve old raw/receipts")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    if (args.nshard < 1 or not 0 <= args.shard < args.nshard or args.limit < 0 or args.batch_size < 1
            or args.prepare_workers < 1 or args.tp < 1 or args.max_num_seqs < 1
            or args.batch_size > args.max_num_seqs or args.max_pixels < 28 * 28
            or not 0 < args.gpu_memory_utilization < 1):
        parser.error("invalid shard, limit, batching, worker, TP, pixels or GPU memory arguments")
    if not Path(args.model).is_absolute():
        parser.error("--model must be an absolute local directory, not a remote model identifier")
    import shutil
    for name in ("ffmpeg", "ffprobe"):
        resolved = shutil.which(getattr(args, name))
        if resolved is None:
            parser.error(f"{name} executable unavailable; supply its installed absolute path")
        setattr(args, name, resolved)
    out = Path(args.out).resolve()
    rows, manifest_sha = read_manifest(args.manifest, args.shard, args.nshard, args.limit)
    for row in rows:
        for key in SOURCE_FIELDS:
            if overlaps(out, Path(row[key]).parent):
                parser.error(f"--out must be separate from each source asset directory: {row[key]}")
    if out == Path(args.manifest).resolve() or out in Path(args.manifest).resolve().parents:
        parser.error("--out must not contain or overwrite the input manifest")
    out.mkdir(parents=True, exist_ok=True)
    # Per-shard managed-process locks allow independent nodes to share --out.
    # Evidence has separate SID/window locks; no shared JSONL is appended.
    control = out / "shards" / f"shard-{args.shard:05d}-of-{args.nshard:05d}"
    with bundle_lock(control):
        model, code = model_identity(args.model), code_identity()
        params = params_identity(args, code)
        run = {"schema": SCHEMA, "stage": args.stage, "manifest_path": str(Path(args.manifest).resolve()),
            "manifest_sha256": manifest_sha, "model_identity": model, "code_identity": code,
            "params_identity": params, "shard": args.shard, "nshard": args.nshard, "limit": args.limit,
            "selected_ids": len(rows), "retry_failed": args.retry_failed,
            "semantic_approval": False, "human_review_completed": False}
        atomic_write(control / "run.receipt.json", json_bytes(run))
        llm = sampling = None
        load_error = None
        results = []
        # CPU futures are bounded by the worker count, and joined by the context
        # manager. GPU execution stays in this managed foreground process.
        with ThreadPoolExecutor(max_workers=args.prepare_workers) as pool:
            for offset in range(0, len(rows), args.prepare_workers):
                group = rows[offset:offset + args.prepare_workers]
                futures = [pool.submit(prepare_clip, row, out, model, params, args) for row in group]
                for row, future in zip(group, futures):
                    try:
                        clip = future.result()
                        if args.stage == "run":
                            pending = [item for item in clip["windows"] if item["receipt"] is None]
                            if pending and llm is None:
                                if load_error is not None:
                                    raise RuntimeError(f"local model load previously failed; no repeated GPU attempts: {load_error}")
                                try:
                                    llm, sampling = load_local_model(args, model, code, control)
                                except Exception as error:
                                    load_error = f"{type(error).__name__}: {error}"
                                    raise
                            for start in range(0, len(pending), args.batch_size):
                                observe_batch(pending[start:start + args.batch_size], llm, sampling, clip["source"])
                        result = final_caption(clip)
                    except Exception as error:
                        result = {"id": row["id"], "status": "failed", "error": f"{type(error).__name__}: {error}",
                                  "observation_complete_without_failures": False}
                        errors = control / "failures"
                        errors.mkdir(exist_ok=True)
                        atomic_write(errors / f"{row['id']}.json", json_bytes({**result, "input": row,
                            "model_sha256": model["sha256"], "params_sha256": params["sha256"],
                            "semantic_approval": False, "human_review_completed": False}))
                    results.append(result)
                    row_dir = control / "rows"
                    row_dir.mkdir(exist_ok=True)
                    atomic_write(row_dir / f"{row['id']}.json", json_bytes(result))
                    atomic_write(control / "progress.json", json_bytes({"processed_ids": len(results), "selected_ids": len(rows),
                        "stage": args.stage, "semantic_approval": False}))
                    print(json.dumps({"id": row["id"], "status": result["status"], "processed_ids": len(results),
                                      "selected_ids": len(rows)}, ensure_ascii=False), flush=True)
        atomic_write(control / "index.jsonl", b"".join((json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n").encode() for item in results))
        counts = {name: sum(item["status"] == name for item in results) for name in ("candidate", "abstain", "failed", "prepared")}
        summary = {"schema": SCHEMA, "stage": args.stage, "selected_ids": len(rows), "processed_ids": len(results),
            "counts": counts, "all_selected_ids_observed_without_failures":
                args.stage == "run" and bool(rows) and not counts["failed"] and not counts["prepared"],
            "semantic_approval": False, "human_review_completed": False, "training_entry_updated": False}
        atomic_write(control / "summary.json", json_bytes(summary))
        print(json.dumps(summary), flush=True)
        return 1 if counts["failed"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
