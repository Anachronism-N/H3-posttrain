#!/usr/bin/env python3
"""Run local, hash-bound BeatThis inference; retain every success and failure."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

SCHEMA = "beatthis_audio_detections_v1"
PACKAGES = ("beat-this", "rotary-embedding-torch", "torch", "torchaudio",
            "einops", "soxr", "numpy", "soundfile")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path, expected_count):
    raw = Path(path).read_bytes()
    rows = []
    seen = set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Invalid or duplicate id at input line {number}")
        if not isinstance(row.get("audio_path"), str) or not row["audio_path"]:
            raise ValueError(f"Missing audio_path at input line {number}")
        seen.add(sample_id)
        rows.append((row, number, hashlib.sha256(line).hexdigest()))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} input rows, got {len(rows)}")
    return rows, hashlib.sha256(raw).hexdigest()


def validate_times(values, duration):
    times = list(values)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v < 0 or v > duration + 0.02
           for v in times):
        raise ValueError("Invalid model timestamps (20ms terminal-frame tolerance)")
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Model timestamps must be strictly increasing")
    return times


def detection_summary(beats, downbeats, duration):
    beats = validate_times(beats, duration)
    downbeats = validate_times(downbeats, duration)
    if any(t not in beats for t in downbeats):
        raise ValueError("Minimal postprocessor downbeat did not snap to a beat")
    return {
        "beats": beats, "downbeats": downbeats,
        "beat_count": len(beats), "downbeat_count": len(downbeats),
        "complete_downbeat_interval_count": max(0, len(downbeats) - 1),
        "insufficient_downbeats_for_complete_bar": len(downbeats) < 2,
        "short_context": duration < 30.0,
        "short_context_basis": "observed audio duration below 1500 / 50 = 30s inference chunk",
        "timestamp_reference": "seconds from decoded normalized audio sample zero",
        "audio_video_offset_seconds": "unknown",
        "meter": "unknown", "bar_phase_accuracy_verified": False,
        "beat_accuracy_verified": False, "downbeat_accuracy_verified": False,
        "audio_video_alignment_verified": False,
    }


def source_identity(source_path, checkpoint_path):
    source = json.loads(Path(source_path).read_text())
    observed = sha256_file(checkpoint_path)
    if observed != source.get("checkpoint_sha256"):
        raise ValueError("Checkpoint bytes differ from source manifest")
    if Path(source["checkpoint_path"]).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("Checkpoint path differs from source manifest")
    if not source.get("source_url") or source.get("license") != "MIT":
        raise ValueError("Source URL and MIT license declaration required")
    license_path = Path(source["license_path"])
    if sha256_file(license_path) != source.get("license_sha256"):
        raise ValueError("License bytes differ from source manifest")
    return {"checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": observed, "source": source,
            "source_manifest_path": str(Path(source_path).resolve()),
            "source_manifest_sha256": sha256_file(source_path)}


def run(args):
    rows, manifest_hash = load_manifest(args.manifest, args.expected_count)
    model_identity = source_identity(args.checkpoint_source, args.checkpoint)
    out_dir = Path(args.output_dir)
    # Exclusive creation prevents replacing old measurements or partial runs.
    out_dir.mkdir(parents=True, exist_ok=False)
    import torch
    import torchaudio
    import beat_this.inference
    import beat_this.preprocessing
    import soundfile

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    params = {"device": args.device, "float16": False, "dbn": False,
              "postprocessor": "minimal", "postprocessor_fps": 50,
              "cpu_threads": args.cpu_threads, "chunk_size_frames": 1500,
              "border_size_frames": 6, "overlap_mode": "keep_first",
              "audio_cropping": "none; full audio_path", "checkpoint_is_local": True}
    runtime = {
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "sys_prefix": sys.prefix, "pid": os.getpid(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "torch_module_path": torch.__file__, "torchaudio_module_path": torchaudio.__file__,
        "beatthis_inference_path": beat_this.inference.__file__,
        "beatthis_inference_sha256": sha256_file(beat_this.inference.__file__),
        "beatthis_preprocessing_sha256": sha256_file(beat_this.preprocessing.__file__),
        "helper_path": str(Path(__file__).resolve()), "helper_sha256": sha256_file(__file__),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }
    receipt = {"schema": SCHEMA, "status": "running", "started_unix": time.time(),
               "input_manifest_path": str(Path(args.manifest).resolve()),
               "input_manifest_sha256": manifest_hash, "expected_count": args.expected_count,
               "model": model_identity, "params": params, "runtime": runtime}
    receipt_path = out_dir / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    try:
        tracker = beat_this.inference.File2Beats(
            checkpoint_path=str(Path(args.checkpoint).resolve()),
            device=args.device, float16=False, dbn=False)
    except Exception as error:
        receipt.update(status="model_load_failed", error_type=type(error).__name__,
                       error_message=str(error), traceback=traceback.format_exc(), exit_code=2)
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        raise
    receipt["model_load_completed_unix"] = time.time()
    parameters = list(tracker.model.parameters())
    receipt["model_runtime"] = {"parameter_devices": sorted({str(p.device) for p in parameters}),
                                "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
                                "parameter_count": sum(p.numel() for p in parameters)}
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    failures = 0
    insufficient = 0
    output_path = out_dir / "detections.jsonl"
    with output_path.open("x") as output:
        for row, number, line_hash in rows:
            started = time.time()
            result = {
                "schema": SCHEMA, "id": row["id"], "sample_id": row["id"],
                "split": row.get("split"), "dataset_role": row.get("dataset_role"),
                "audio_path": row["audio_path"], "audio_sha256": None,
                "duration_seconds": None, "sample_rate": None, "channels": None,
                "input_manifest_path": receipt["input_manifest_path"],
                "input_manifest_sha256": manifest_hash,
                "input_manifest_line_number": number, "input_manifest_line_sha256": line_hash,
                "source_row": row, "model_sha256": model_identity["checkpoint_sha256"],
                "model_source_manifest_sha256": model_identity["source_manifest_sha256"],
                "helper_sha256": runtime["helper_sha256"], "library_versions": runtime["packages"],
                "params": params, "status": "error", "beats": None, "downbeats": None,
                "inference_called": False, "model_prediction_returned": False,
                "started_unix": started,
            }
            try:
                result["audio_sha256"] = sha256_file(row["audio_path"])
                info = soundfile.info(row["audio_path"])
                duration = info.frames / info.samplerate
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Audio is empty or has invalid duration")
                result.update(duration_seconds=duration, sample_rate=info.samplerate,
                              channels=info.channels, audio_frames=info.frames)
                result["inference_called"] = True
                beats, downbeats = tracker(row["audio_path"])
                result["model_prediction_returned"] = True
                raw_beats, raw_downbeats = beats.tolist(), downbeats.tolist()
                # A repr retains even malformed NaN/Inf without generating invalid JSON.
                result["raw_model_predictions_repr"] = {"beats": repr(raw_beats),
                                                        "downbeats": repr(raw_downbeats)}
                result.update(detection_summary(raw_beats, raw_downbeats, duration))
                if sha256_file(row["audio_path"]) != result["audio_sha256"]:
                    raise ValueError("Audio changed during inference")
                result["status"] = "ok"
                insufficient += result["insufficient_downbeats_for_complete_bar"]
            except Exception as error:
                failures += 1
                result.update(error_type=type(error).__name__, error_message=str(error),
                              traceback=traceback.format_exc())
            result["elapsed_seconds"] = time.time() - started
            output.write(json.dumps(result, allow_nan=False) + "\n")
            output.flush()
            print(json.dumps({"id": row["id"], "status": result["status"],
                              "beat_count": result.get("beat_count"),
                              "downbeat_count": result.get("downbeat_count"),
                              "error": result.get("error_message"),
                              "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    receipt.update(status="completed" if failures == 0 else "completed_with_failures",
                   completed_unix=time.time(), attempted_count=len(rows),
                   successful_count=len(rows) - failures, failed_count=failures,
                   insufficient_downbeats_count=insufficient,
                   output_path=str(output_path.resolve()), output_sha256=sha256_file(output_path),
                   exit_code=0 if failures == 0 else 2)
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt_path": str(receipt_path), "status": receipt["status"],
                      "successful_count": receipt["successful_count"], "failed_count": failures}), flush=True)
    return receipt["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.expected_count < 1 or args.cpu_threads < 1:
        parser.error("expected-count and cpu-threads must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
