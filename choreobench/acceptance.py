"""One real native base/checkpoint window with immutable output and measurements."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

try:
    import resource
except ImportError:  # Windows: even --help must work without Unix resource APIs.
    resource = None

from .audio import WINDOW_SECONDS, inspect_audio
from .baseline import _local_actions, _recorded_result
from .cli import doctor
from .io import file_sha, object_sha, read_json_snapshot, write_json
from .providers import NativeProvider


_CUDA_MEASUREMENTS = {
    "allocated_bytes": "memory_allocated", "reserved_bytes": "memory_reserved",
    "peak_allocated_bytes": "max_memory_allocated", "peak_reserved_bytes": "max_memory_reserved",
}


def resource_snapshot(torch_module=None, device=None):
    """Unavailable measurements are null with a reason, never guessed zeros.

    Linux keeps ru_maxrss (KiB), /proc RSS, CUDA peaks and nvidia-smi. Darwin
    ru_maxrss is already bytes. Unsupported units/APIs remain unknown.
    """
    values = dict(host_peak_rss_bytes=None, host_rss_bytes=None, nvidia_smi=None,
                  **{key: None for key in _CUDA_MEASUREMENTS})
    status = dict(host_peak_rss_bytes="resource_module_unavailable", host_rss_bytes="procfs_unavailable")
    if resource is not None:
        try:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if not math.isfinite(peak) or peak < 0:
                raise ValueError("Invalid ru_maxrss")
            multiplier = 1024 if sys.platform.startswith("linux") else 1 if sys.platform == "darwin" else None
            if multiplier is None:
                status["host_peak_rss_bytes"] = "ru_maxrss_units_unknown"
            else:
                values["host_peak_rss_bytes"] = int(peak * multiplier)
                status["host_peak_rss_bytes"] = "measured"
        except (OSError, AttributeError, TypeError, ValueError, OverflowError) as error:
            status["host_peak_rss_bytes"] = "unavailable_" + type(error).__name__
    if sys.platform.startswith("linux"):
        try:
            pages = int(Path("/proc/self/statm").read_text().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages < 0 or page_size <= 0:
                raise ValueError("Invalid procfs RSS")
            values["host_rss_bytes"] = pages * page_size
            status["host_rss_bytes"] = "measured"
        except (OSError, AttributeError, ValueError, IndexError) as error:
            status["host_rss_bytes"] = "unavailable_" + type(error).__name__
    for key, method in _CUDA_MEASUREMENTS.items():
        status[key] = "cuda_not_initialized"
        if torch_module is not None and device is not None and getattr(device, "type", None) == "cuda":
            try:
                value = getattr(torch_module.cuda, method)(device)
                if type(value) is not int or value < 0:
                    raise ValueError("Invalid CUDA measurement")
                values[key], status[key] = value, "measured"
            except Exception as error:
                status[key] = "unavailable_" + type(error).__name__
    try:
        process = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,driver_version",
            "--format=csv,noheader"], capture_output=True, text=True, check=False, timeout=10)
        values["nvidia_smi_returncode"] = process.returncode
        if process.returncode == 0 and isinstance(process.stdout, str) and process.stdout.strip():
            values["nvidia_smi"] = process.stdout
            status["nvidia_smi"] = "measured"
        else:
            status["nvidia_smi"] = "nonzero_exit" if process.returncode else "no_output"
    except (OSError, subprocess.SubprocessError) as error:
        status["nvidia_smi"] = "unavailable_" + type(error).__name__
    values["measurement_status"] = status
    return values


def _snapshot(torch_module=None, device=None):
    # Measurements are advisory evidence. Preserve the primary load/generation
    # error even if an unexpected platform/monitoring failure occurs as well.
    try:
        return resource_snapshot(torch_module, device)
    except Exception as error:
        return dict(host_peak_rss_bytes=None, host_rss_bytes=None, nvidia_smi=None,
            **{key: None for key in _CUDA_MEASUREMENTS},
            measurement_status={"snapshot": "unavailable_" + type(error).__name__})


def _artifact_type(spec):
    if not isinstance(spec, dict) or spec.get("kind") != "native":
        raise ValueError("Acceptance requires an explicit native provider")
    checkpoint = spec.get("checkpoint")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("checkpoint must be a nonempty portable path or null")
    if checkpoint is not None and "task_prompt" in spec:
        raise ValueError("task_prompt cannot override a portable checkpoint's trained prefix")
    return "portable-checkpoint" if checkpoint is not None else "base-only"


def _loaded_verification(provider, artifact, artifact_type):
    verification = provider.verification
    if (not isinstance(verification, dict)
            or verification.get("mode") != artifact["mode"]
            or verification.get("artifact_identity_sha256") != object_sha(artifact)):
        raise ValueError("Loaded verification is not bound to the actual audited artifacts")
    expected = artifact_type == "portable-checkpoint"
    if verification.get("loaded_portable_state_verified") is not expected:
        raise ValueError("Portable checkpoint learned tensors were not verified exactly" if expected else
                         "Base-only verification unexpectedly claims a portable checkpoint")
    return verification


def run_acceptance(provider_path, audio_path, output_dir, *, seed=7, max_output_tokens=512, artifact_type=None):
    """Audit/load/generate once, preserving failures; no training or retries.

    artifact_type optionally pins base-only vs portable-checkpoint. The provider
    config selects the artifact; the audit and actual loaded tensor receipt must
    agree. This is a GPU format/EOS acceptance, never a musical quality score.
    """
    if type(seed) is not int or not 0 <= seed < 2 ** 63 or type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("seed must be in [0,2**63) and max_output_tokens must be positive")
    if artifact_type not in (None, "base-only", "portable-checkpoint"):
        raise ValueError("Unknown artifact_type")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="choreobench.local_acceptance.v1", accepted=False, artifact_type=None,
        expected_artifact_type=artifact_type, checkpoint=None, quality_verified=False,
        retries=0, stage="input", training_performed=False)
    torch_module, device = None, None
    try:
        spec, config_sha = read_json_snapshot(provider_path)
        report.update(provider=spec, provider_config_sha256=config_sha)
        actual_type = _artifact_type(spec)
        report["artifact_type"] = actual_type
        if artifact_type is not None and artifact_type != actual_type:
            raise ValueError("Provider artifact type differs from explicit --artifact-type")
        audio = inspect_audio(audio_path)
        audio.pop("pcm")
        if audio["num_samples"] != int(math.floor(WINDOW_SECONDS * audio["sample_rate"] + .5)):
            raise ValueError("Exactly one real 5.0625s window is required; no cropping or padding")
        report.update(environment=doctor(), audio=audio, before_audit=_snapshot())
        write_json(output / "input.json", report)
        report["stage"] = "audit"
        start = time.monotonic()
        provider = NativeProvider(spec, "local", Path(provider_path).resolve().parent)
        artifact = provider.identity["artifact_identity"]
        expected_mode = "fair_portable_checkpoint" if actual_type == "portable-checkpoint" else "native_base_only"
        if artifact.get("mode") != expected_mode or (actual_type == "portable-checkpoint" and not isinstance(artifact.get("checkpoint"), dict)):
            raise ValueError("Audited artifact mode differs from the requested base/checkpoint type")
        report.update(identity=copy.deepcopy(provider.identity), identity_sha256=object_sha(provider.identity),
            audited_artifact_mode=artifact["mode"], checkpoint=copy.deepcopy(artifact.get("checkpoint")),
            audit_seconds=time.monotonic() - start)
        write_json(output / "identity.json", provider.identity)
        import torch
        torch_module = torch
        device = torch.device(spec.get("device", "cpu"))
        if device.type != "cuda":
            raise ValueError("Real GPU acceptance requires an explicit CUDA device")
        report["stage"] = "cuda_initialization"
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        report.update(cuda_runtime=torch.version.cuda, before_load=_snapshot(torch, device))
        report["stage"] = "load"
        start = time.monotonic()
        provider.load()
        torch.cuda.synchronize(device)
        report.update(load_seconds=time.monotonic() - start, after_load=_snapshot(torch, device),
                      loaded_artifact_verification=copy.deepcopy(provider.verification))
        if object_sha(provider.identity) != report["identity_sha256"]:
            raise ValueError("Native provider identity changed while loading")
        _loaded_verification(provider, artifact, actual_type)
        write_json(output / "loaded.json", report)
        request = dict(call_id="acceptance/local/0000/00", response_kind="local_caption", audio=audio,
                       seed=seed, max_output_tokens=max_output_tokens)
        write_json(output / "request.json", request)
        report["stage"] = "generate"
        start = time.monotonic()
        response = provider.generate(request)
        try:
            torch.cuda.synchronize(device)
        finally:
            report["inference_seconds"] = time.monotonic() - start
            write_json(output / "response.raw.json", response)
        report["after_inference"] = _snapshot(torch, device)
        report["stage"] = "validate_output"
        if object_sha(provider.identity) != report["identity_sha256"]:
            raise ValueError("Native provider identity changed while generating")
        _recorded_result(dict(success=True, result=response), provider.identity, request)
        raw = response.get("raw")
        if not isinstance(raw, dict) or raw.get("loaded_artifact_verification") != provider.verification:
            raise ValueError("Response loaded verification differs from accepted checkpoint/base identity")
        report["parsed"] = _local_actions(response["text"])
        if response["stop_reason"] != "eos":
            raise ValueError("No genuine EOS stop; raw output retained")
        report.update(accepted=True, stage="accepted", actual_tokens=response["costs"],
            stop_reason=response["stop_reason"], response_sha256=file_sha(output / "response.raw.json"))
        return report
    except Exception as error:
        report.update(failed_stage=report["stage"], error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        report["failure_resources"] = _snapshot(torch_module, device)
        raise
    finally:
        write_json(output / "acceptance.json", report)
        print(json.dumps(dict(accepted=report["accepted"], artifact_type=report["artifact_type"],
            out_dir=str(output), error=report.get("error")), ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--artifact-type", choices=("base-only", "portable-checkpoint"),
                        help="Optional explicit expectation; otherwise selected by the provider config")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2 ** 63 or args.max_output_tokens < 1:
        parser.error("seed must be in [0,2**63) and max-output-tokens must be positive")
    return run_acceptance(args.provider, args.audio, args.out_dir, seed=args.seed,
        max_output_tokens=args.max_output_tokens, artifact_type=args.artifact_type)


if __name__ == "__main__":
    main()
