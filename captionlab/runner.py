"""Run a frozen probe through an explicitly supplied original training runtime.

No built-in checkpoint migration, network service, generation, or training. The
factory is trusted local Python supplied by the operator, not code from a dataset.
"""
import importlib.util
import hashlib
from pathlib import Path
import sys
import time
import json

from .io import file_sha, object_sha, read_json, write_json
from .scoring import summarize_scores
from .targets import convert_record, validate_source_records


def audit_runtime(config, base_dir):
    if config.get("schema") != "captionlab.runtime.v1":
        raise ValueError("Unsupported runtime config")
    if not isinstance(config.get("protocol_id"), str) or not config["protocol_id"].strip():
        raise ValueError("Explicit original input/target protocol_id required")
    factory = config["factory"]
    base = Path(base_dir).resolve()

    def verify(item):
        path = Path(item["path"]).expanduser()
        path = path.resolve() if path.is_absolute() else (base / path).resolve()
        before = path.stat()
        digest = file_sha(path)
        after = path.stat()
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or digest != item["sha256"]):
            raise ValueError("Runtime artifact changed: " + str(path))
        return {"path": str(path), "sha256": digest, "size_bytes": after.st_size}

    factory_ref = verify(factory)
    if not isinstance(factory.get("kwargs", {}), dict):
        raise ValueError("factory kwargs must be an object")
    artifacts, seen = [], set()
    for item in config["artifacts"]:
        role = item["role"]
        if role not in ("model", "tokenizer", "checkpoint", "runtime"):
            raise ValueError("Unknown runtime artifact role")
        reference = verify(item)
        if reference["path"] in seen:
            raise ValueError("Duplicate runtime artifact path")
        seen.add(reference["path"])
        artifacts.append(dict(reference, role=role))
    if not {"model", "tokenizer", "runtime"}.issubset({r["role"] for r in artifacts}):
        raise ValueError("Declare model, tokenizer and original runtime files")
    mode = config.get("checkpoint_mode")
    if mode not in ("base_only", "portable"):
        raise ValueError("checkpoint_mode must be base_only or portable")
    has_checkpoint = any(r["role"] == "checkpoint" for r in artifacts)
    if has_checkpoint != (mode == "portable"):
        raise ValueError("Checkpoint mode and declared artifacts differ")
    content = lambda roles: sorted(({"role": r["role"], "sha256": r["sha256"],
                                    "size_bytes": r["size_bytes"]} for r in artifacts if r["role"] in roles),
                                  key=lambda r: (r["role"], r["sha256"]))
    implementation = {p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"factory": factory_ref, "artifacts": artifacts, "diagnostic_source_sha256": implementation,
            "runtime_config_sha256": object_sha(config), "protocol_id": config["protocol_id"],
            "model_identity": object_sha({"files": content({"model", "checkpoint", "runtime"}),
                                           "factory": factory_ref["sha256"], "config": config,
                                           "diagnostic_source": implementation}),
            "tokenizer_identity": object_sha(content({"tokenizer"})),
            "declared_artifact_hashes_verified": True,
            "artifact_inventory_completeness": "operator_responsibility_not_automatically_verified",
            "cache_parity": "not_verified", "zero_portable_parity": "not_verified"}


def _load_runtime(config, audit):
    path = audit["factory"]["path"]
    name = "_captionlab_factory_" + audit["factory"]["sha256"]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Factory must be an importable Python source file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Execute the exact hashed source snapshot. SourceFileLoader can otherwise
    # reuse a stale same-size/same-mtime .pyc despite a changed source SHA.
    code = Path(path).read_bytes()
    if hashlib.sha256(code).hexdigest() != audit["factory"]["sha256"]:
        raise ValueError("Factory changed after runtime audit")
    exec(compile(code, path, "exec"), module.__dict__)
    runtime = module.load_runtime(**config["factory"].get("kwargs", {}))
    if not isinstance(runtime, dict) or set(runtime) != {"adapter", "receipt"}:
        raise ValueError("Factory must return adapter and receipt")
    receipt = runtime["receipt"]
    # This is a runtime-specific attestation, distinct from byte identity above.
    if (receipt.get("protocol_id") != config["protocol_id"] or
            receipt.get("checkpoint_mode") != config["checkpoint_mode"] or
            receipt.get("target_contract_verified") is not True or
            (config["checkpoint_mode"] == "portable" and
             receipt.get("portable_tensors_verified") is not True)):
        raise ValueError("Factory did not attest original target contract / exact portable tensor restore")
    return runtime


def _audio_paths(manifest, base_dir):
    import soundfile as sf
    result = {}
    for case in manifest["cases"]:
        for cond in case["conditions"]:
            audio = cond["audio"]
            path = Path(audio["path"]).expanduser()
            path = path.resolve() if path.is_absolute() else (Path(base_dir) / path).resolve()
            key = (str(path), audio["sha256"])
            if key not in result:
                if file_sha(path) != audio["sha256"]:
                    raise ValueError("Probe audio bytes changed: " + str(path))
                info = sf.info(path)
                if (info.frames != audio["num_samples"] or info.samplerate != audio["sample_rate"]
                        or info.channels != 1):
                    raise ValueError("Probe audio geometry differs from declaration")
                result[key] = path
    return result


def validate_probe(manifest):
    if manifest.get("schema") != "captionlab.probe.v1":
        raise ValueError("Unsupported probe manifest")
    summarize_scores(manifest, [], n_resamples=1)
    validate_source_records([case["target_source"] for case in manifest["cases"]])
    mode = manifest["target_mode"]
    if mode not in ("raw", "actions_timed", "actions_untimed"):
        raise ValueError("Unsupported target mode")
    for case in manifest["cases"]:
        source = case["target_source"]
        if source["id"] != case["case_id"] or source["split"] != "val" or case["split"] != "val":
            raise ValueError("Probe target source ID/split differs")
        converted = convert_record(source, keep_timestamps=mode != "actions_untimed",
                                   genre_classes=manifest["genre_classes"])
        expected = source["raw_caption"] if mode == "raw" else converted["target_text"]
        if case["target_mode"] != mode or case["target_text"] != expected:
            raise ValueError("Probe target text differs from declared source conversion")
        expected_windows = [{"start": w["start"], "end": w["end"], "caption": w["action_text"].strip()}
                            for w in converted["windows"]]
        if case["target_windows"] != expected_windows:
            raise ValueError("Probe target-window metadata differs from original action spans")
        for condition in case["conditions"]:
            audio, kind = condition["audio"], condition["kind"]
            transform = "zero_waveform_same_geometry" if kind == "silence" else "identity"
            if condition["transform"] != transform:
                raise ValueError("Unexpected audio transform")
            if any(audio[k] != source["audio"][k] for k in ("sample_rate", "num_samples", "pad_intervals")):
                raise ValueError("Conditions must share declared rate/sample/padding geometry")
            if kind in ("true", "silence"):
                if audio != source["audio"] or condition["audio_id"] != source["id"]:
                    raise ValueError("True/silence source differs from target source")
            elif (condition["donor_source_group_id"] == case["source_group_id"] or
                  any(audio[k] == source["audio"][k] for k in ("sha256", "pcm_sha256"))):
                raise ValueError("Known related audio is not a mismatch candidate")


def _source_row(case, cond, base_dir, out_dir, local_files):
    import numpy as np
    import soundfile as sf
    audio = cond["audio"]
    path = Path(audio["path"]).expanduser()
    path = path.resolve() if path.is_absolute() else (Path(base_dir) / path).resolve()
    # Read/hash once and hand the original collator an immutable run-local copy,
    # instead of asking it to re-open a mutable external path after our audit.
    key = ("original", audio["sha256"])
    if key not in local_files:
        snapshot = path.read_bytes()
        if hashlib.sha256(snapshot).hexdigest() != audio["sha256"]:
            raise ValueError("Audio changed after preflight")
        local_path = out_dir / ("audio-" + audio["sha256"] + path.suffix)
        with local_path.open("xb") as stream:
            stream.write(snapshot)
        local_files[key] = (local_path, audio["sha256"])
    path, _ = local_files[key]
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if (waveform.shape != (audio["num_samples"], 1) or rate != audio["sample_rate"] or
            not np.isfinite(waveform).all()):
        raise ValueError("Require declared finite, mono, complete waveform geometry")
    digest = audio["sha256"]
    if cond["kind"] == "silence":
        key = ("silence", digest, rate, len(waveform))
        if key not in local_files:
            silence_path = out_dir / ("silence-" + object_sha(list(key)) + ".wav")
            sf.write(silence_path, np.zeros_like(waveform), rate, subtype="FLOAT")
            local_files[key] = (silence_path, file_sha(silence_path))
        path, digest = local_files[key]
    source = case["target_source"]
    # Target provenance stays separate from donor identity; every condition uses
    # the same raw target, labels and authoritative target-window metadata.
    return {"id": case["case_id"], "source_group_id": source["source_group_id"],
            "audio_path": str(path), "audio_sha256": digest,
            "audio_duration_seconds": audio["num_samples"] / audio["sample_rate"],
            "audio_window_seconds": [0.0, audio["num_samples"] / audio["sample_rate"]],
            "target_text": case["target_text"],
            "target_windows": case["target_windows"]}


def run_probe(manifest_path, runtime_config_path, out_dir):
    from .runtime import score_target
    manifest_path, runtime_config_path = Path(manifest_path).resolve(), Path(runtime_config_path).resolve()
    manifest, config = read_json(manifest_path), read_json(runtime_config_path)
    # The scoring validator checks duplicate cases/conditions and target digests
    # before any loading/calls. Model protocols are not inferred from a name.
    validate_probe(manifest)
    if config.get("target_mode") != manifest["target_mode"]:
        raise ValueError("Runtime target_mode does not match frozen probe")
    _audio_paths(manifest, manifest_path.parent)
    audit = audit_runtime(config, runtime_config_path.parent)
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    write_json(output / "runtime_audit.json", audit)
    manifest_sha = object_sha(manifest)
    records, silences, start = [], {}, time.monotonic()
    current = {"stage": "load_runtime"}
    try:
        runtime = _load_runtime(config, audit)
        write_json(output / "load_receipt.json", runtime["receipt"])
        with (output / "scores.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for case in manifest["cases"]:
                case_records = []
                for condition in case["conditions"]:
                    current = {"stage": "score", "case_id": case["case_id"],
                               "condition_id": condition["condition_id"]}
                    row = _source_row(case, condition, manifest_path.parent, output, silences)
                    measured = score_target(runtime["adapter"], row)
                    record = {"case_id": case["case_id"], "condition_id": condition["condition_id"],
                              "target_sha256": case["target_sha256"],
                              "model_identity": audit["model_identity"],
                              "tokenizer_identity": audit["tokenizer_identity"],
                              "manifest_sha256": manifest_sha,
                              "runtime_config_sha256": audit["runtime_config_sha256"],
                              **{k: measured[k] for k in ("target_token_ids", "token_nll", "eos_token_id")},
                              "runtime_details": {k: v for k, v in measured.items() if k not in
                                                  ("target_token_ids", "token_nll", "eos_token_id")}}
                    record["runtime_details"]["audio"] = {
                        "path": row["audio_path"], "sha256": row["audio_sha256"],
                        "original_sha256": condition["audio"]["sha256"],
                        "transform": condition["transform"],
                        "declared_pcm_identity_independently_verified": False}
                    # Catch condition/target drift before persisting a success.
                    summarize_scores({"cases": [case]}, case_records + [record], n_resamples=1)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    case_records.append(record)
        summary = summarize_scores(manifest, records)
        summary["elapsed_seconds"] = time.monotonic() - start
        summary["new_training_updates"] = 0
        summary["new_generation_calls"] = 0
        write_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        # Fail closed; keep successful records and an explicit missing denominator.
        write_json(output / "failure.json", dict(current, error_type=type(error).__name__,
                                                message=str(error), completed_records=len(records)))
        partial = summarize_scores(manifest, records)
        partial["run_status"] = "failed"
        write_json(output / "failure_summary.json", partial)
        raise
