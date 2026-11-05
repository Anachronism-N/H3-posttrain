"""Control-flow fixtures only; no real model, CUDA, training or external API."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
import types
import urllib.request
import wave

import pytest

from choreobench import acceptance, providers, resident
from choreobench.io import file_sha, object_sha, read_json, write_json


@pytest.fixture
def native_runtime(monkeypatch):
    state = dict(audits=[], loads=[], checks=[], samples=[], load_error=None, verify_error=None,
        verification_change=None, mode_override=None, text="[0-2s] Step left. [2-5.0625s] Turn right.", stop="eos")
    prefix = "scripts_omnidance.exp.ms."
    identity = types.ModuleType(prefix + "music_planner_search_identity")
    def audit(family, model, checkpoint, source):
        state["audits"].append((family, model, checkpoint, source))
        return dict(schema="fixture_artifacts_not_real_weights", family=family,
            mode=state["mode_override"] or ("fair_portable_checkpoint" if checkpoint else "native_base_only"),
            base_files=[dict(path=str(model), sha256="a" * 64)],
            checkpoint=dict(root=str(checkpoint), training_fingerprint="fixture-not-training",
                adapter_metadata={"max_length": 8192}) if checkpoint else None)
    def verify(adapter, artifact):
        state["checks"].append(adapter)
        if "provider" in state:
            assert state["provider"].adapter is None, "Do not publish an adapter before verification"
        if state["verify_error"]:
            raise state["verify_error"]
        receipt = dict(schema="music_planner_loaded_artifact_verification_v1", mode=artifact["mode"],
            artifact_identity_sha256=object_sha(artifact),
            loaded_portable_state_verified=artifact["mode"] == "fair_portable_checkpoint")
        if state["verification_change"]:
            receipt.update(state["verification_change"])
        return receipt
    identity.audit_model_artifacts, identity.verify_loaded_portable = audit, verify
    adapters = types.ModuleType(prefix + "music_planner_adapters")
    def load(*args, **kwargs):
        state["loads"].append((args, kwargs))
        if state["load_error"]:
            raise state["load_error"]
        return types.SimpleNamespace(processor=types.SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: json.dumps(messages)))
    adapters.load_planner_adapter = load
    search = types.ModuleType(prefix + "music_planner_search")
    search.validate_sampling = lambda *args: None
    def sample(adapter, row, **kwargs):
        state["samples"].append(copy.deepcopy(row))
        return dict(raw_text=state["text"], stop_reason=state["stop"], new_token_ids=[1, 2], input_tokens=12)
    search.sample_native = sample
    for module in (identity, adapters, search):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    torch = types.ModuleType("torch")
    torch.device = lambda name: types.SimpleNamespace(type=name.split(":")[0])
    torch.version = types.SimpleNamespace(cuda="fixture-no-CUDA-runtime")
    torch.cuda = types.SimpleNamespace(set_device=lambda device: None, reset_peak_memory_stats=lambda device: None,
        synchronize=lambda device: None, memory_allocated=lambda device: 101, memory_reserved=lambda device: 202,
        max_memory_allocated=lambda device: 303, max_memory_reserved=lambda device: 404)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(acceptance, "doctor", lambda: {"fixture_only": True, "native_inference_verified": False})
    monkeypatch.setattr(acceptance, "resource_snapshot", lambda *args: {
        "host_peak_rss_bytes": None, "measurement_status": {"fixture_only": "not_real_measurements"}})
    return state


def provider_spec(tmp_path, checkpoint=False, **extra):
    return dict(kind="native", model=str(tmp_path / "fixture-no-model"), device="cuda:0",
                **({"checkpoint": str(tmp_path / "fixture-no-checkpoint")} if checkpoint else {}), **extra)


def inputs(tmp_path, *, checkpoint=False, rate=16000, count=81000, **extra):
    spec = provider_spec(tmp_path, checkpoint, **extra)
    config = tmp_path / "provider.json"
    write_json(config, spec)
    wav = tmp_path / "window.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x01\x00" * count)
    return config, wav, tmp_path / "acceptance"


def request():
    return dict(call_id="fixture", response_kind="local_caption", seed=7, max_output_tokens=32,
        audio={"path": "fixture.wav", "sha256": "a" * 64})


@pytest.mark.parametrize("failure", ["verify", "load"])
def test_failed_native_initialization_is_atomic_and_poisoned(native_runtime, tmp_path, failure):
    state = native_runtime
    state[failure + "_error"] = ValueError("fixture " + failure + " failure")
    provider = providers.NativeProvider(provider_spec(tmp_path, checkpoint=True), "local")
    state["provider"] = provider
    with pytest.raises(ValueError, match="fixture"):
        provider.generate(request())
    assert provider.adapter is None and provider.verification is None and provider.requires_restart
    for action in (provider.load, lambda: provider.generate(request())):
        with pytest.raises(RuntimeError, match="restart"):
            action()
    assert len(state["loads"]) == 1 and state["samples"] == []
    assert len(state["checks"]) == (1 if failure == "verify" else 0)


def test_successful_native_load_publishes_verified_adapter_once(native_runtime, tmp_path):
    state = native_runtime
    provider = providers.NativeProvider(provider_spec(tmp_path, checkpoint=True), "local")
    state["provider"] = provider
    provider.load()
    assert provider.adapter is not None and provider.verification["loaded_portable_state_verified"]
    provider.load()
    output = provider.generate(request())
    assert len(state["loads"]) == len(state["checks"]) == 1
    assert not provider.requires_restart
    assert output["raw"]["loaded_artifact_verification"] == provider.verification


def test_resident_poison_flag_requires_restart_without_publishing_adapter(native_runtime, tmp_path):
    native_runtime["verify_error"] = ValueError("fixture loaded tensor mismatch")
    config, wav, _ = inputs(tmp_path)
    server = resident.create_resident_server(read_json(config), tmp_path)
    native_runtime["provider"] = server.provider
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    url = "http://127.0.0.1:" + str(server.server_address[1])
    try:
        client = resident.NativeServiceProvider(dict(kind="native_service", base_url=url,
            expected_identity_sha256=server.identity_sha256, timeout_seconds=3))
        audio = acceptance.inspect_audio(wav)
        audio.pop("pcm")
        req = dict(request(), audio=audio)
        assert client.generate(req)["raw"]["status"] == 500
        assert server.provider.adapter is None and server.provider.requires_restart
        with urllib.request.urlopen(url + "/health", timeout=3) as response:
            assert json.loads(response.read())["requires_restart"] is True
        req["call_id"] = "another-explicit-request"
        assert client.generate(req)["raw"]["status"] == 503
        assert len(native_runtime["loads"]) == len(native_runtime["checks"]) == 1
        assert not native_runtime["samples"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_acceptance_distinguishes_base_and_exact_portable_identity(native_runtime, tmp_path, checkpoint):
    config, wav, output = inputs(tmp_path, checkpoint=checkpoint)
    kind = "portable-checkpoint" if checkpoint else "base-only"
    report = acceptance.run_acceptance(config, wav, output, artifact_type=kind)
    assert report["accepted"] and report["artifact_type"] == kind
    assert report["provider_config_sha256"] == file_sha(config)
    assert report["identity_sha256"] == object_sha(report["identity"])
    assert report["loaded_artifact_verification"]["loaded_portable_state_verified"] is checkpoint
    assert (report["checkpoint"] is not None) is checkpoint
    assert not report["training_performed"] and not report["quality_verified"] and report["retries"] == 0
    assert report == read_json(output / "acceptance.json")
    assert len(native_runtime["loads"]) == len(native_runtime["samples"]) == 1
    assert set(native_runtime["samples"][0]) == {"id", "audio_path", "audio_sha256", "audio_window_seconds"}
    with pytest.raises(FileExistsError):
        acceptance.run_acceptance(config, wav, output)


def test_acceptance_keeps_base_task_prompt_out_of_checkpoint_mode(native_runtime, tmp_path):
    config, wav, output = inputs(tmp_path, checkpoint=True, task_prompt="Change the trained task.")
    with pytest.raises(ValueError, match="cannot override"):
        acceptance.run_acceptance(config, wav, output)
    assert native_runtime["audits"] == []
    assert not read_json(output / "acceptance.json")["accepted"]


def test_acceptance_prompted_base_preserves_actual_template(native_runtime, tmp_path):
    config, wav, output = inputs(tmp_path, task_prompt="Compose two timed movement proposals.")
    report = acceptance.run_acceptance(config, wav, output)
    response = read_json(output / "response.raw.json")
    assert report["artifact_type"] == "base-only"
    assert report["identity"]["protocol"] == "choreobench-prompted-omni-base-v1"
    assert "Compose two timed movement proposals." in response["raw"]["formatted_task_prompt"]


def test_explicit_artifact_expectation_cannot_silently_fall_back_to_base(native_runtime, tmp_path):
    config, wav, output = inputs(tmp_path)
    with pytest.raises(ValueError, match="artifact type differs"):
        acceptance.run_acceptance(config, wav, output, artifact_type="portable-checkpoint")
    assert not native_runtime["audits"]
    report = read_json(output / "acceptance.json")
    assert not report["accepted"] and report["failed_stage"] == "input"


@pytest.mark.parametrize("change", [{"loaded_portable_state_verified": False}, {"artifact_identity_sha256": "b" * 64}])
def test_checkpoint_acceptance_requires_exact_loaded_verification(native_runtime, tmp_path, change):
    config, wav, output = inputs(tmp_path, checkpoint=True)
    native_runtime["verification_change"] = change
    with pytest.raises(ValueError, match="verified exactly|not bound"):
        acceptance.run_acceptance(config, wav, output)
    report = read_json(output / "acceptance.json")
    assert not report["accepted"] and report["failed_stage"] == "load"
    assert native_runtime["samples"] == []


@pytest.mark.parametrize("mode", ["wrong_audit", "bad_format", "not_eos", "over_budget"])
def test_acceptance_preserves_failed_output_and_never_retries(native_runtime, tmp_path, mode):
    config, wav, output = inputs(tmp_path)
    if mode == "wrong_audit":
        native_runtime["mode_override"] = "unknown-protocol"
    elif mode == "bad_format":
        native_runtime["text"] = "Untimed proposed movement."
    elif mode == "not_eos":
        native_runtime["stop"] = "stop"
    with pytest.raises(ValueError):
        acceptance.run_acceptance(config, wav, output, max_output_tokens=1 if mode == "over_budget" else 512)
    report = read_json(output / "acceptance.json")
    assert not report["accepted"] and report["retries"] == 0
    if mode == "wrong_audit":
        assert native_runtime["loads"] == []
    else:
        assert (output / "response.raw.json").exists()
        assert len(native_runtime["samples"]) == 1


def test_discrete_window_rounding_matches_canonical_half_up_contract(native_runtime, tmp_path):
    config, wav, output = inputs(tmp_path, rate=8, count=41)  # 5.0625*8=40.5, half-up is 41.
    assert acceptance.run_acceptance(config, wav, output)["accepted"]


def test_short_audio_does_not_load_or_pad(native_runtime, tmp_path):
    config, wav, output = inputs(tmp_path, count=80000)
    with pytest.raises(ValueError, match="Exactly one"):
        acceptance.run_acceptance(config, wav, output)
    assert native_runtime["loads"] == native_runtime["audits"] == []


def test_measurement_failure_does_not_replace_model_failure(native_runtime, monkeypatch, tmp_path):
    config, wav, output = inputs(tmp_path)
    native_runtime["verify_error"] = ValueError("fixture primary loaded-state mismatch")
    def broken_monitor(*args):
        raise RuntimeError("fixture unavailable monitor")
    monkeypatch.setattr(acceptance, "resource_snapshot", broken_monitor)
    with pytest.raises(ValueError, match="primary loaded-state mismatch"):
        acceptance.run_acceptance(config, wav, output)
    report = read_json(output / "acceptance.json")
    assert report["error_type"] == "ValueError" and report["failed_stage"] == "load"
    assert report["failure_resources"]["peak_reserved_bytes"] is None
    assert report["failure_resources"]["measurement_status"]["snapshot"] == "unavailable_RuntimeError"


def test_response_is_saved_when_post_generation_cuda_errors(native_runtime, monkeypatch, tmp_path):
    config, wav, output = inputs(tmp_path)
    calls = []
    def synchronize(device):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("fixture asynchronous generation error")
    monkeypatch.setattr(sys.modules["torch"].cuda, "synchronize", synchronize)
    with pytest.raises(RuntimeError, match="asynchronous"):
        acceptance.run_acceptance(config, wav, output)
    assert (output / "response.raw.json").exists()
    assert not read_json(output / "acceptance.json")["accepted"]


def test_generation_timing_excludes_response_publication_io(native_runtime, monkeypatch, tmp_path):
    config, wav, output = inputs(tmp_path)
    clock = [0.]
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: clock[0])
    module = sys.modules["scripts_omnidance.exp.ms.music_planner_search"]
    sample, publish = module.sample_native, acceptance.write_json
    def delayed_sample(*args, **kwargs):
        clock[0] += 3
        return sample(*args, **kwargs)
    def delayed_publish(path, value):
        if path.name == "response.raw.json":
            clock[0] += 100
        return publish(path, value)
    monkeypatch.setattr(module, "sample_native", delayed_sample)
    monkeypatch.setattr(acceptance, "write_json", delayed_publish)
    report = acceptance.run_acceptance(config, wav, output)
    assert report["inference_seconds"] == 3


def test_windows_unknown_host_measurements_do_not_hide_available_cuda(monkeypatch):
    monkeypatch.setattr(acceptance, "resource", None)
    monkeypatch.setattr(acceptance, "sys", types.SimpleNamespace(platform="win32"))
    def missing(*args, **kwargs):
        raise FileNotFoundError("fixture no nvidia-smi")
    monkeypatch.setattr(acceptance.subprocess, "run", missing)
    cuda = types.SimpleNamespace(**{name: lambda device: 123 for name in acceptance._CUDA_MEASUREMENTS.values()})
    result = acceptance.resource_snapshot(types.SimpleNamespace(cuda=cuda), types.SimpleNamespace(type="cuda"))
    assert result["host_peak_rss_bytes"] is None and result["host_rss_bytes"] is None and result["nvidia_smi"] is None
    assert all(result[key] == 123 for key in acceptance._CUDA_MEASUREMENTS)
    assert result["measurement_status"]["host_peak_rss_bytes"] == "resource_module_unavailable"


@pytest.mark.parametrize("platform,multiplier", [("linux", 1024), ("darwin", 1), ("win32", None)])
def test_rss_units_are_explicit_and_linux_measurements_remain_available(monkeypatch, platform, multiplier):
    monkeypatch.setattr(acceptance, "sys", types.SimpleNamespace(platform=platform))
    monkeypatch.setattr(acceptance, "resource", types.SimpleNamespace(RUSAGE_SELF=0,
        getrusage=lambda who: types.SimpleNamespace(ru_maxrss=1234)))
    monkeypatch.setattr(acceptance, "Path", lambda path: types.SimpleNamespace(read_text=lambda: "8 3 0"))
    monkeypatch.setattr(acceptance.os, "sysconf", lambda name: 4096, raising=False)
    monkeypatch.setattr(acceptance.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="fixture inventory"))
    result = acceptance.resource_snapshot()
    assert result["host_peak_rss_bytes"] == (1234 * multiplier if multiplier else None)
    assert result["host_rss_bytes"] == (12288 if platform == "linux" else None)
    assert result["nvidia_smi"] == "fixture inventory"
    assert result["allocated_bytes"] is None


def test_help_does_not_require_resource_torch_or_real_gpu():
    code = """
import builtins, runpy, sys
original = builtins.__import__
def no_heavy_or_unix(name, *args, **kwargs):
    if name in ('resource', 'torch'):
        raise ImportError('fixture unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = no_heavy_or_unix
sys.argv = ['choreobench.acceptance', '--help']
runpy.run_module('choreobench.acceptance', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
        cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--artifact-type" in result.stdout and "portable-checkpoint" in result.stdout
