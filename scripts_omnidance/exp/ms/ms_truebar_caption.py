"""Consume actual BeatThis receipts and run frozen original-MS bar captions.

No new beats are constructed. The normalized input retains the complete raw
downbeat sequence and all original detection/receipt/source references.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import ms_music_windows as windows

FINAL0_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
FINAL0_SHA1 = "e1506282faf66ca10e8ab50ee26bd542b7b9ff0a"


def decode(payload):
    return json.loads(payload, parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + v)))


def checked_file(path, expected):
    windows.digest(expected, "file SHA")
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("source file SHA mismatch: " + str(path))
    return payload


def normalize_detections(detections, manifest):
    detections, manifest = Path(detections).resolve(), Path(manifest).resolve()
    receipt_path = detections.parent / "run_receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = decode(receipt_bytes)
    if receipt.get("schema") != "beatthis_audio_detections_v1" or receipt.get("status") != "completed":
        raise ValueError("actual completed BeatThis receipt required")
    if type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0 or type(receipt.get("failed_count")) is not int or receipt["failed_count"] != 0:
        raise ValueError("BeatThis run failed or receipt count is not typed")
    if Path(receipt["output_path"]).resolve() != detections:
        raise ValueError("receipt output path differs")
    payload = checked_file(detections, receipt["output_sha256"])
    manifest_bytes = checked_file(manifest, receipt["input_manifest_sha256"])
    if Path(receipt["input_manifest_path"]).resolve() != manifest:
        raise ValueError("BeatThis source manifest path differs")
    lines = [line for line in manifest_bytes.splitlines() if line.strip()]
    source_rows = [decode(line) for line in lines]
    raw_lines = [line for line in payload.splitlines() if line.strip()]
    rows = [decode(line) for line in raw_lines]
    count = len(source_rows)
    for key in ("expected_count", "attempted_count", "successful_count"):
        if type(receipt.get(key)) is not int or receipt[key] != count:
            raise ValueError("BeatThis run count differs from actual manifest")
    if len(rows) != count or [r.get("id") for r in rows] != [r.get("id") for r in source_rows]:
        raise ValueError("BeatThis row order/count differs from source manifest")
    params, model, runtime = receipt["params"], receipt["model"], receipt["runtime"]
    if params.get("device") != "cpu" or params.get("dbn") is not False or params.get("float16") is not False:
        raise ValueError("real bar run requires typed CPU FP32 dbn=false")
    if params.get("postprocessor") != "minimal" or params.get("postprocessor_fps") != 50 or params.get("audio_cropping") != "none; full audio_path":
        raise ValueError("BeatThis timestamps must come from minimal full-audio inference")
    if receipt.get("model_runtime", {}).get("parameter_devices") != ["cpu"] or receipt["model_runtime"].get("parameter_dtypes") != ["torch.float32"]:
        raise ValueError("actual model runtime must be CPU FP32")
    source = decode(checked_file(model["source_manifest_path"], model["source_manifest_sha256"]))
    if source != model["source"] or source.get("schema") != "beatthis_checkpoint_source_v1":
        raise ValueError("source manifest differs from consumed model")
    if source.get("checkpoint_variant") != "final0" or source.get("official_sha1_match") is not True or source.get("license") != "MIT":
        raise ValueError("official final0 provenance required")
    if model.get("checkpoint_sha256") != FINAL0_SHA256 or source.get("checkpoint_sha256") != FINAL0_SHA256:
        raise ValueError("not verified official final0 checkpoint")
    checkpoint = checked_file(model["checkpoint_path"], FINAL0_SHA256)
    if hashlib.sha1(checkpoint).hexdigest() != FINAL0_SHA1 or source.get("official_response_sha1") != FINAL0_SHA1:
        raise ValueError("official checkpoint SHA1 mismatch")
    if Path(source["checkpoint_path"]).resolve() != Path(model["checkpoint_path"]).resolve():
        raise ValueError("checkpoint source path differs")
    checked_file(source["license_path"], source["license_sha256"])
    checked_file(source["package_metadata_path"], source["package_metadata_sha256"])
    inference_bytes = checked_file(runtime["beatthis_inference_path"], runtime["beatthis_inference_sha256"])
    inference_path = Path(runtime["beatthis_inference_path"])
    checked_file(inference_path.with_name("preprocessing.py"), runtime["beatthis_preprocessing_sha256"])
    checked_file(runtime["helper_path"], runtime["helper_sha256"])
    if b"class File2Beats" not in inference_bytes or not isinstance(runtime.get("packages", {}).get("beat-this"), str):
        raise ValueError("File2Beats library implementation/version missing")
    normalized = []
    for index, (row, source_row, raw_line, source_line) in enumerate(zip(rows, source_rows, raw_lines, lines), 1):
        if row.get("schema") != "beatthis_audio_detections_v1" or row.get("status") != "ok" or row.get("inference_called") is not True or row.get("model_prediction_returned") is not True:
            raise ValueError("actual model inference success missing")
        if row.get("source_row") != source_row or row.get("input_manifest_line_number") != index or row.get("input_manifest_line_sha256") != hashlib.sha256(source_line).hexdigest():
            raise ValueError("detection source row/line identity differs")
        if row.get("model_sha256") != FINAL0_SHA256 or row.get("model_source_manifest_sha256") != model["source_manifest_sha256"]:
            raise ValueError("detection checkpoint/source identity differs")
        if row.get("helper_sha256") != runtime["helper_sha256"] or row.get("library_versions") != runtime["packages"] or row.get("params") != params:
            raise ValueError("detection library/helper/params differ")
        if row.get("input_manifest_sha256") != receipt["input_manifest_sha256"] or row.get("audio_path") != source_row["audio_path"]:
            raise ValueError("detection audio source differs")
        audio = checked_file(row["audio_path"], row["audio_sha256"])
        duration = windows.number(row.get("duration_seconds"), "audio duration", positive=True)
        sample_rate = windows.number(row.get("sample_rate"), "sample_rate", positive=True)
        if type(row.get("audio_frames")) is not int or row["audio_frames"] / sample_rate != duration:
            raise ValueError("decoded audio frame duration differs")
        downbeats, beats = row.get("downbeats"), row.get("beats")
        if not isinstance(downbeats, list) or not isinstance(beats, list):
            raise ValueError("actual beat/downbeat arrays missing")
        for timestamps in (beats, downbeats):
            values = [windows.number(t, "timestamp") for t in timestamps]
            if any(t < 0 or t > duration for t in values) or any(b <= a for a,b in zip(values,values[1:])):
                raise ValueError("raw timestamps outside media; no implicit clipping/fallback allowed")
        if any(t not in beats for t in downbeats):
            raise ValueError("minimal model downbeats are not beat-snapped")
        if type(row.get("complete_downbeat_interval_count")) is not int or row["complete_downbeat_interval_count"] != max(0,len(downbeats)-1):
            raise ValueError("actual bar interval count differs")
        provenance = {"method": "beat_this.inference.File2Beats", "checkpoint_name": "final0",
            "dbn": False, "device": "cpu", "synthetic": False,
            "downbeats_origin": "model_downbeat_head_postprocessed", "checkpoint_sha256": FINAL0_SHA256,
            "audio_sha256": row["audio_sha256"], "audio_path": row["audio_path"],
            "audio_duration_seconds": duration, "library_source_sha256": runtime["beatthis_inference_sha256"],
            "library_source_sha256_basis": "installed beat_this/inference.py, not entire distribution",
            "library_version": runtime["packages"]["beat-this"], "library_versions": runtime["packages"],
            "params": params, "raw_detections_path": str(detections), "raw_detections_sha256": receipt["output_sha256"],
            "raw_detection_line": index, "raw_detection_line_sha256": hashlib.sha256(raw_line).hexdigest(),
            "run_receipt_path": str(receipt_path), "run_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "checkpoint_source_path": model["source_manifest_path"], "checkpoint_source_sha256": model["source_manifest_sha256"],
            "helper_path": runtime["helper_path"], "helper_sha256": runtime["helper_sha256"],
            "adapter_path": str(Path(__file__).resolve()), "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "raw_beats_seconds": beats, "raw_downbeats_seconds": downbeats,
            "bar_phase_accuracy_verified": False, "audio_video_alignment_verified": False}
        normalized.append({"id": row["id"], "downbeats_seconds": downbeats, "source": provenance})
    return normalized


def run(args):
    if not args.bar_input:
        raise ValueError("--bar-input must name actual BeatThis detections.jsonl")
    args.strategy = "bar"
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError("output must be a new independent directory")
    normalized_path = out.with_name(out.name + ".bar_input.jsonl")
    if normalized_path.exists():
        raise ValueError("normalized bar input already exists; never overwrite")
    records = normalize_detections(args.bar_input, args.manifest)
    payload = b"".join((json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n").encode() for r in records)
    with normalized_path.open("xb") as handle:
        handle.write(payload)
    args.bar_input = str(normalized_path)
    return windows.run(args)


if __name__ == "__main__":
    result = run(windows.parser().parse_args())
    print(json.dumps({k:v for k,v in result.items() if k != "records"}), flush=True)
    sys.exit(0 if result["complete"] else 1)
