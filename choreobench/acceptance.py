"""One real local window with immutable raw output and measured resource evidence."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

from .audio import inspect_audio
from .baseline import _local_actions
from .cli import doctor
from .io import file_sha, object_sha, read_json, write_json
from .providers import NativeProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 63 or args.max_output_tokens < 1:
        parser.error("seed must be in [0,2**63) and max-output-tokens must be positive")
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="choreobench.local_acceptance.v1", accepted=False,
                  checkpoint=None, quality_verified=False, retries=0)
    torch = None
    device = None
    def snapshot():
        result = dict(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        result["host_rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        if torch is not None and device is not None:
            result.update(allocated_bytes=torch.cuda.memory_allocated(device),
                          reserved_bytes=torch.cuda.memory_reserved(device),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        result["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,driver_version",
            "--format=csv,noheader"], capture_output=True, text=True, check=False, timeout=10).stdout
        return result
    try:
        spec = read_json(args.provider)
        if spec.get("kind") != "native" or spec.get("checkpoint"):
            raise ValueError("This acceptance entry requires an explicit native base-only provider")
        audio = inspect_audio(args.audio)
        audio.pop("pcm")
        if audio["num_samples"] != round(5.0625 * audio["sample_rate"]):
            raise ValueError("Exactly one real 5.0625s window is required; no cropping or padding")
        report.update(environment=doctor(), provider=spec, provider_config_sha256=file_sha(args.provider),
                      audio=audio, before_audit=snapshot())
        write_json(output / "input.json", report)
        start = time.monotonic()
        provider = NativeProvider(spec, "local", Path(args.provider).resolve().parent)
        report.update(identity=provider.identity, identity_sha256=object_sha(provider.identity),
                      audit_seconds=time.monotonic() - start)
        write_json(output / "identity.json", provider.identity)
        import torch as torch_module
        torch = torch_module
        device = torch.device(spec.get("device", "cpu"))
        if device.type != "cuda":
            raise ValueError("Real GPU acceptance requires an explicit CUDA device")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        report.update(cuda_runtime=torch.version.cuda, before_load=snapshot())
        start = time.monotonic()
        provider.load()
        torch.cuda.synchronize(device)
        report.update(load_seconds=time.monotonic() - start, after_load=snapshot())
        write_json(output / "loaded.json", report)
        request = dict(call_id="acceptance/local/0000/00", response_kind="local_caption", audio=audio,
                       seed=args.seed, max_output_tokens=args.max_output_tokens)
        write_json(output / "request.json", request)
        start = time.monotonic()
        response = provider.generate(request)
        torch.cuda.synchronize(device)
        report.update(inference_seconds=time.monotonic() - start, after_inference=snapshot())
        write_json(output / "response.raw.json", response)
        report["parsed"] = _local_actions(response["text"])
        if response["stop_reason"] != "eos":
            raise ValueError("No genuine EOS stop; raw output retained")
        report.update(accepted=True, actual_tokens=response["costs"],
                      stop_reason=response["stop_reason"], response_sha256=file_sha(output / "response.raw.json"))
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        try:
            report["failure_resources"] = snapshot()
        except Exception as snapshot_error:
            report["snapshot_error"] = str(snapshot_error)
        raise
    finally:
        write_json(output / "acceptance.json", report)
        print(json.dumps(dict(accepted=report["accepted"], out_dir=str(output), error=report.get("error")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
