#!/usr/bin/env python3
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
SYSTEM = (
    "Design a solo choreography unit using the supplied text suggestions, tentative timing cues, "
    "outline and history. You have no audio or video access. Engineering fixture values are not "
    "observations of a real song. Return exactly one JSON object with unit_id and events. Each event "
    "has exactly start, end, track, action. Use global seconds, only the body track, at most three "
    "sequential events, a 0.25-second grid, minimum duration 0.5 seconds, and exact full-unit coverage. "
    "Do not add partners, props or scenery; do not invent physical entry states. No prose or fences."
)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def validate_unit(value, unit):
    if not isinstance(value, dict) or set(value) != {"unit_id", "events"} or value["unit_id"] != unit["unit_id"]:
        raise ValueError("Unit output violates the exact JSON contract")
    events = value["events"]
    if not isinstance(events, list) or not 1 <= len(events) <= 3:
        raise ValueError("Invalid number of events")
    cursor = unit["start"]
    for event in events:
        if not isinstance(event, dict) or set(event) != {"start", "end", "track", "action"}:
            raise ValueError("Event output violates the exact JSON contract")
        start, end = event["start"], event["end"]
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end)):
            raise ValueError("Nonfinite or nonnumeric event clock")
        if start != cursor or end > unit["end"] or end - start < 0.5:
            raise ValueError("Gap, overlap, out-of-bounds or short event")
        if any(abs(t * 4 - round(t * 4)) > 1e-7 for t in (start, end)):
            raise ValueError("Event is off the 0.25-second grid")
        if event["track"] != "body" or not isinstance(event["action"], str) or not event["action"].strip():
            raise ValueError("Invalid track or empty action")
        if len(event["action"]) > 600:
            raise ValueError("Action exceeds the text budget")
        cursor = end
    if cursor != unit["end"]:
        raise ValueError("Events do not fully cover the unit")


def stage_model(config, output):
    source = Path(config["mounted_model_dir"])
    manifest_path = ROOT / "source_manifest.json"
    if sha_file(manifest_path) != config["source_manifest_sha256"]:
        raise ValueError("Packaged source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    marker = json.loads((source / "transfer_complete.json").read_text())
    if (marker.get("complete") is not True or marker["model"] != config["model_repository"]
            or marker["revision"] != config["model_revision"]
            or marker["source_manifest_sha256"] != config["source_manifest_sha256"]):
        raise ValueError("OSS model completion marker differs")
    if (manifest["model"] != config["model_repository"] or manifest["revision"] != config["model_revision"]
            or len(manifest["files"]) != config["expected_files"]
            or manifest["total_bytes"] != config["expected_bytes"]):
        raise ValueError("Model inventory differs from the reviewed package")
    parent = next((p for p in map(Path, ("/checkpoint/binary", "/checkpoint", "/tmp"))
                   if p.is_dir() and shutil.disk_usage(p).free >= config["expected_bytes"] + (8 << 30)), None)
    if parent is None:
        raise OSError("Insufficient scratch for the complete model; existing data will not be deleted")
    local = parent / config["run_id"]
    local.mkdir()
    receipts = []
    for item in manifest["files"]:
        name = PurePosixPath(item["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Model path escapes the snapshot")
        original, target = source / item["path"], local / item["path"]
        if original.stat().st_size != item["bytes"]:
            raise ValueError("OSS file size differs: " + item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if sha_file(target) != item["sha256"]:
            raise ValueError("Downloaded model SHA256 differs: " + item["path"])
        receipts.append(dict(path=item["path"], sha256=item["sha256"], bytes=item["bytes"]))
        print(json.dumps(dict(stage="model_copy", checked=len(receipts), total=len(manifest["files"]))), flush=True)
    write_new(output / "model_verification.json", dict(files=receipts, all_sha256_verified=True))
    return local


def main():
    if os.environ.get("RANK") not in (None, "", "0"):
        print("Only rank 0 executes the inference smoke test", flush=True)
        return
    config = json.loads((ROOT / "job_config.json").read_text())
    report = dict(schema="choreobench.native_text_smoke_result.v1", run_id=config["run_id"],
                  engineering_only=True, input_provenance=config["input_provenance"],
                  started_unix=time.time(), model_calls_attempted=0, validated_units=0,
                  sft_model_validated=False, choreography_quality_evaluated=False,
                  remote_persistence_verified=False, status="started")
    output = None
    try:
        source = Path(config["mounted_model_dir"])
        parent = Path(config["result_parent"])
        if not source.is_dir() or not parent.is_dir():
            raise FileNotFoundError("The declared OSS source/result mount is unavailable")
        destination = parent / config["run_id"]
        destination.mkdir()
        output = destination
        write_new(output / "job_config.json", config)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        report["runtime"] = dict(python=sys.version, libc=platform.libc_ver(), torch=torch.__version__,
                                 transformers=transformers.__version__, cuda=torch.version.cuda,
                                 visible_gpus=torch.cuda.device_count(), inference_device=config["model_device"])
        if torch.cuda.device_count() != config["allocated_gpus"]:
            raise ValueError("Expected exactly eight allocated visible GPUs")
        report["gpus"] = [dict(ordinal=i, name=torch.cuda.get_device_properties(i).name,
                              bytes=torch.cuda.get_device_properties(i).total_memory)
                          for i in range(torch.cuda.device_count())]
        model_path = stage_model(config, output)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                device_map={"": config["model_device"]}, local_files_only=True, trust_remote_code=False).eval()
        eos = model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        if not eos_ids or any(type(t) is not int for t in eos_ids):
            raise ValueError("Native EOS token identity is unavailable")
        generation = GenerationConfig(do_sample=False, num_beams=1,
            max_new_tokens=config["max_new_tokens"], eos_token_id=eos,
            pad_token_id=model.generation_config.pad_token_id)
        report["generation_config"] = generation.to_dict()
        history = []
        for index in range(config["probes"]):
            unit = dict(unit_id="probe-u" + str(index), start=index * 5, end=(index + 1) * 5)
            payload = dict(engineering_fixture=True, target_unit=unit,
                bpm=dict(value=96, source="engineering_fixture", observed=False),
                bars=dict(count=2, meter="4/4", observed=False), style=dict(status="unavailable"),
                motion_candidates=["Step forward and lift one arm, then settle the weight.",
                                   "Bend the knees and open both arms, then rise with an overhead reach."],
                candidate_source="manual engineering controls, not SFT outputs",
                acoustic_summary=dict(trend="rise" if index == 0 else "steady", observed=False),
                outline=dict(intent="Introduce travel and vary the return"), history=history)
            messages = [dict(role="system", content=SYSTEM), dict(role="user", content=json.dumps(payload))]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(config["model_device"])
            write_new(output / ("%02d.started.json" % index), dict(messages=messages, unit=unit,
                seed=config["seed"], generation_config=generation.to_dict()))
            report["model_calls_attempted"] += 1
            torch.manual_seed(config["seed"])
            started = time.monotonic()
            prefix_len = inputs["input_ids"].shape[1]
            try:
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=generation)
            except Exception as error:
                write_new(output / ("%02d.raw.json" % index), dict(stop_reason="infrastructure_error",
                    usage=dict(input_tokens=prefix_len, output_tokens=None), new_token_ids=None,
                    error=dict(type=type(error).__name__, message=str(error))))
                raise
            prefix_matches = torch.equal(generated[:, :prefix_len], inputs["input_ids"])
            ids = generated[0, prefix_len:].cpu().tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            stop = "eos" if ids and ids[-1] in eos_ids else "incomplete_output"
            raw = dict(text=text, new_token_ids=ids, stop_reason=stop,
                prefix_matches=prefix_matches, generated_sequence_ids=generated[0].cpu().tolist(),
                usage=dict(input_tokens=prefix_len, output_tokens=len(ids)),
                latency_seconds=time.monotonic() - started)
            write_new(output / ("%02d.raw.json" % index), raw)
            if not prefix_matches:
                raise ValueError("Generation changed the input prefix")
            if stop != "eos":
                raise ValueError("Native generation did not finish at EOS")
            decision = json.loads(text)
            validate_unit(decision, unit)
            write_new(output / ("%02d.validated.json" % index), decision)
            history.append(decision)
            report["validated_units"] += 1
        report["status"] = "engineering_smoke_passed"
    except Exception as error:
        report.update(status="engineering_smoke_failed", error=dict(type=type(error).__name__, message=str(error)))
    report["finished_unix"] = time.time()
    if output is not None:
        try:
            write_new(output / "result.json", report)
        except Exception as error:
            report.update(status="engineering_smoke_failed",
                          persistence_error=dict(type=type(error).__name__, message=str(error)))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["status"] == "engineering_smoke_passed" else 2)


if __name__ == "__main__":
    main()
