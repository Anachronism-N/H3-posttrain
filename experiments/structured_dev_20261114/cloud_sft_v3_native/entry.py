#!/usr/bin/env python3
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import urllib.parse
import urllib.request
import wave

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

from experiments.structured_dev_20261114.cloud_qwen32_native.entry import OssTransport, sha_file, write_new


class SftTransport(OssTransport):
    def download_asset(self, item, target, config):
        source = urllib.parse.urlsplit(item["source_oss"])
        name = PurePosixPath(item["path"])
        if (source.scheme != "oss" or source.netloc != "futurelab-dpro-hz"
                or ".." in PurePosixPath(source.path).parts or name.is_absolute() or ".." in name.parts
                or not any(item["source_oss"].startswith(config[k]) for k in ("model_oss_prefix", "adapter_oss_prefix"))):
            raise ValueError("SFT asset violates its declared source scope")
        url = self.capabilities["ASSET_GET"][item["path"]]
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != "https" or parsed.netloc != "futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com"
                or urllib.parse.unquote(parsed.path) != source.path or parsed.fragment):
            raise ValueError("SFT capability violates its HTTPS object scope")
        with self.opener.open(url, timeout=600) as response, target.open("xb") as stream:
            remaining = item["bytes"]
            while remaining:
                block = response.read(min(8 << 20, remaining))
                if not block:
                    raise ValueError("SFT asset ended before its declared size")
                stream.write(block)
                remaining -= len(block)
            if response.read(1):
                raise ValueError("SFT asset exceeds its declared size")
        if sha_file(target) != item["sha256"]:
            raise ValueError("SFT asset SHA256 differs from the frozen manifest")


def requests(config):
    if (config["source_split"] != "dev" or config["source_song_id"] != "fma-1881"
            or config["source_interval_samples"] != [0, 81000] or config["max_new_tokens"] != 1024
            or config["candidates_per_window"] != 2 or config["task_prompt"] is not None
            or config["allow_full_268_case_collection_in_this_smoke"] is not False):
        raise ValueError("This smoke permits only the frozen two-case development window")
    audio = dict(config["audio"], path=str(ROOT / "dev1881-first-window.wav"))
    if audio["sample_rate"] != 16000 or audio["num_samples"] != 81000 or sha_file(audio["path"]) != audio["sha256"]:
        raise ValueError("SFT waveform identity or clock differs")
    with wave.open(audio["path"], "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getnframes()) != (16000, 1, 2, 81000):
            raise ValueError("SFT smoke requires exactly one canonical PCM16 observation")
    result = []
    if len(config["cases"]) != 2:
        raise ValueError("This smoke permits exactly two model calls")
    for index, case in enumerate(config["cases"]):
        expected = "fma-1881/untimed/0000/%02d" % index
        seed = int(hashlib.sha256(("sol61-select:" + expected).encode()).hexdigest()[:8], 16)
        if case != dict(case_id=expected, seed=seed):
            raise ValueError("SFT canonical case ID or seed differs")
        result.append(dict(call_id=expected, case_id=expected, response_kind="local_caption",
                           max_output_tokens=1024, audio=dict(audio)))
    return result


def stage_assets(config, store, output):
    manifest_path = ROOT / "source_assets_manifest.json"
    if sha_file(manifest_path) != config["source_assets_manifest_sha256"]:
        raise ValueError("SFT source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["files"]) != 35 or len({r["path"] for r in manifest["files"]}) != 35:
        raise ValueError("SFT source inventory differs from the frozen package")
    parent = next((p for p in map(Path, ("/checkpoint/binary", "/checkpoint", "/tmp"))
        if p.is_dir() and shutil.disk_usage(p).free >= manifest["total_bytes"] + (8 << 30)), None)
    if parent is None:
        raise OSError("Insufficient scratch; no existing data will be removed")
    local = parent / (config["run_id"] + "-model")
    local.mkdir()
    receipts = []
    for item in manifest["files"]:
        name = PurePosixPath(item["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("SFT asset path escapes the new run")
        target = local / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        store.download_asset(item, target, config)
        receipts.append(dict(path=item["path"], bytes=item["bytes"], sha256=item["sha256"]))
        print(json.dumps(dict(stage="sft_asset_verification", checked=len(receipts), total=35)), flush=True)
    write_new(store, output / "asset_verification.json", dict(files=receipts, all_sha256_verified=True))
    return local


def verify_result(result, request, expected_seed):
    raw, costs = result["raw"], result["costs"]
    if any(type(costs.get(k)) is not int or costs[k] < 0 for k in ("input_tokens", "output_tokens")):
        raise ValueError("SFT call usage is unknown or invalid")
    prefix, tokens, sequence = raw["input_token_ids"], raw["new_token_ids"], raw["generated_sequence_ids"]
    if (len(prefix) != costs["input_tokens"] or len(tokens) != costs["output_tokens"]
            or sequence != prefix + tokens or costs["input_tokens"] != 76):
        raise ValueError("SFT raw IDs do not match the measured native prefix and usage")
    if (not tokens or tokens[-1] not in (151645, 151643) or len(tokens) > 1024
            or result["stop_reason"] != "eos" or raw["stop_reason"] != "eos"):
        raise ValueError("SFT native output did not complete within the frozen budget")
    if raw["canonical_case_id"] != request["case_id"] or raw["seed"] != expected_seed:
        raise ValueError("SFT raw case ID or seed differs")
    if result.get("synthetic") is not False or result["text"] != raw["raw_text"]:
        raise ValueError("SFT output is synthetic or differs from its raw response")
    if raw.get("processor_export_verified") is not True or not result["text"].strip():
        raise ValueError("SFT returned no usable text or lacks verified portable processor")
    return dict(case_id=request["case_id"], seed=expected_seed, input_tokens=costs["input_tokens"],
                output_tokens=costs["output_tokens"], native_eos_token_id=tokens[-1], text=result["text"])


def runtime_preflight(config):
    import torch
    from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
    packages = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "accelerate", "numpy", "librosa", "soundfile")}
    if (sys.prefix == sys.base_prefix or packages != config["runtime_packages"]
            or torch.__version__ != "2.8.0+cu128" or torch.cuda.device_count() != 8):
        raise ValueError("SFT requires its pinned private environment and eight visible GPUs")
    with torch.inference_mode():
        value = torch.ones((8, 8), device=config["model_device"], dtype=torch.bfloat16)
        if not torch.all(value @ value == 8).item():
            raise ValueError("SFT BF16 kernel preflight failed")
    return dict(packages=packages, python=sys.version, python_prefix=sys.prefix,
                visible_gpus=8, model_device=config["model_device"], bf16_kernel_verified=True)


def error_message(error):
    return re.sub(r"https?://\S+", "[HTTP location omitted]", str(error))


def main():
    if os.environ.get("RANK") not in (None, "", "0"):
        return
    config = json.loads((ROOT / "job_config.json").read_text())
    report = dict(schema="choreobench.best_sft_native_smoke.v1", run_id=config["run_id"],
        started_unix=time.time(), model_calls_attempted=0, validated_cases=0, status="started",
        full_collection_started=False, choreography_quality_evaluated=False,
        benchmark_first_line_protocol_equivalent=False, remote_persistence_verified=False)
    store, output = None, None
    try:
        store = SftTransport(config, json.loads((ROOT / "signed_io.json").read_text()))
        output = ROOT / (config["run_id"] + "-records")
        output.mkdir()
        write_new(store, output / "job_config.json", config)
        declared = requests(config)
        report["runtime"] = runtime_preflight(config)
        write_new(store, output / "runtime_preflight.json", report["runtime"])
        local = stage_assets(config, store, output)
        from experiments.structured_dev_20261114 import checkpoint_provider
        if sha_file(checkpoint_provider.__file__) != config["provider_source_sha256"]:
            raise ValueError("SFT provider implementation changed")
        loader = ROOT / "music_planner_adapters.py"
        if sha_file(loader) != config["expected_loader_sha256"]:
            raise ValueError("SFT native loader changed")
        provider = checkpoint_provider.create(dict(kind="native", family="omni_captioner", portable_checkpoint=True,
            model=str(local / "base"), checkpoint=str(local / "adapter"), adapter_source=str(loader),
            device_map={"": config["model_device"]}, max_length=8192))
        write_new(store, output / "provider_identity.json", provider.identity)
        for index, request in enumerate(declared):
            write_new(store, output / ("%02d.started.json" % index), dict(request=request, seed=config["cases"][index]["seed"],
                provider_identity=provider.identity))
            report["model_calls_attempted"] += 1
            try:
                result = provider.generate(request)
            except Exception as error:
                write_new(store, output / ("%02d.error.json" % index), dict(status="inference_error",
                    error_type=type(error).__name__, error_message=error_message(error),
                    usage=dict(input_tokens=None, output_tokens=None),
                    new_token_ids=None, model_call_outcome="not_retried"))
                raise
            write_new(store, output / ("%02d.raw.json" % index), result)
            verified = verify_result(result, request, config["cases"][index]["seed"])
            write_new(store, output / ("%02d.validated.json" % index), verified)
            report["validated_cases"] += 1
        report["status"] = "engineering_smoke_passed"
    except Exception as error:
        report.update(status="engineering_smoke_failed", error_type=type(error).__name__,
                      error_message=error_message(error))
    report["finished_unix"] = time.time()
    if output is not None:
        write_new(store, output / "result.json", report)
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["status"] == "engineering_smoke_passed" else 2)


if __name__ == "__main__":
    main()
