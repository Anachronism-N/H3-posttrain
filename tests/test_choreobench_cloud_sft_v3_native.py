import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest


@pytest.fixture
def entry():
    path = Path(__file__).parents[1] / "experiments/structured_dev_20261114/cloud_sft_v3_native/entry.py"
    spec = importlib.util.spec_from_file_location("sft_cloud_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(entry, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    audio = tmp_path / "dev1881-first-window.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 81000)
    cases = []
    for index in range(2):
        case_id = "fma-1881/untimed/0000/%02d" % index
        cases.append(dict(case_id=case_id, seed=int(hashlib.sha256(("sol61-select:" + case_id).encode()).hexdigest()[:8], 16)))
    return dict(run_id="sft-test", source_split="dev", source_song_id="fma-1881",
        source_interval_samples=[0, 81000], max_new_tokens=1024, candidates_per_window=2,
        task_prompt=None, allow_full_268_case_collection_in_this_smoke=False, cases=cases,
        audio=dict(sample_rate=16000, num_samples=81000, sha256=hashlib.sha256(audio.read_bytes()).hexdigest()),
        model_oss_prefix="oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/models/test-base/",
        adapter_oss_prefix="oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/runs/test-checkpoint/adapter/",
        result_oss_prefix="oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/runs/sft-test/",
        model_device="cuda:0")


def response(request, seed):
    prefix, tokens = [10] * 76, [77, 151645]
    text = "Step and settle."
    return dict(text=text, synthetic=False, stop_reason="eos", identity={},
        costs=dict(input_tokens=76, output_tokens=2, latency_seconds=0.1),
        raw=dict(input_token_ids=prefix, new_token_ids=tokens, generated_sequence_ids=prefix + tokens,
                 canonical_case_id=request["case_id"], seed=seed, stop_reason="eos",
                 raw_text=text, processor_export_verified=True))


def test_requests_bind_exact_waveform_cases_and_seeds(entry, config):
    requests = entry.requests(config)
    assert len(requests) == 2
    assert [r["case_id"] for r in requests] == [r["case_id"] for r in config["cases"]]
    assert all(r["max_output_tokens"] == 1024 and r["response_kind"] == "local_caption" for r in requests)
    assert requests[0]["audio"] == requests[1]["audio"]
    assert all("task_prompt" not in r for r in requests)


@pytest.mark.parametrize("field,value", [
    ("source_split", "test"), ("source_interval_samples", [0, 80000]),
    ("max_new_tokens", 512), ("candidates_per_window", 1), ("task_prompt", "Generate timestamps"),
    ("allow_full_268_case_collection_in_this_smoke", True),
])
def test_cannot_expand_or_change_smoke_contract(entry, config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        entry.requests(config)


def test_changed_case_seed_or_waveform_is_rejected(entry, config):
    changed = copy.deepcopy(config)
    changed["cases"][0]["seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        entry.requests(changed)
    changed = copy.deepcopy(config)
    changed["audio"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        entry.requests(changed)


def test_complete_native_result_has_real_usage_and_ids(entry, config):
    request = entry.requests(config)[0]
    result = response(request, config["cases"][0]["seed"])
    assert entry.verify_result(result, request, config["cases"][0]["seed"])["output_tokens"] == 2
    result["raw"]["new_token_ids"][-1] = 151643
    result["raw"]["generated_sequence_ids"][-1] = 151643
    assert entry.verify_result(result, request, config["cases"][0]["seed"])["native_eos_token_id"] == 151643


@pytest.mark.parametrize("mutation", ["unknown_usage", "non_eos", "truncated", "prefix", "empty", "processor", "synthetic", "seed"])
def test_invalid_return_is_not_reclassified_as_success(entry, config, mutation):
    request = entry.requests(config)[0]
    seed = config["cases"][0]["seed"]
    result = response(request, seed)
    if mutation == "unknown_usage":
        result["costs"]["output_tokens"] = None
    elif mutation == "non_eos":
        result["raw"]["new_token_ids"][-1] = 77
        result["raw"]["generated_sequence_ids"][-1] = 77
    elif mutation == "truncated":
        result["stop_reason"] = result["raw"]["stop_reason"] = "budget_truncation"
    elif mutation == "prefix":
        result["raw"]["generated_sequence_ids"][0] = 1
    elif mutation == "empty":
        result["text"] = result["raw"]["raw_text"] = ""
    elif mutation == "processor":
        result["raw"]["processor_export_verified"] = False
    elif mutation == "synthetic":
        result["synthetic"] = True
    else:
        result["raw"]["seed"] += 1
    with pytest.raises(ValueError):
        entry.verify_result(result, request, seed)


@pytest.mark.parametrize("mutation", ["scheme", "host", "object", "source", "path"])
def test_dual_source_capabilities_cannot_escape_scope(entry, config, tmp_path, mutation):
    item = dict(path="base/config.bin", source_oss=config["model_oss_prefix"] + "config.bin", bytes=4,
                sha256=hashlib.sha256(b"data").hexdigest())
    url = "https://futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com/niezichao.nzc/m2d_train_v1/models/test-base/config.bin?Signature=fixture"
    if mutation == "scheme":
        url = url.replace("https:", "http:")
    elif mutation == "host":
        url = url.replace("futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com", "wrong.invalid")
    elif mutation == "object":
        url = url.replace("config.bin", "other.bin")
    elif mutation == "source":
        item["source_oss"] = "oss://futurelab-dpro-hz/unapproved/config.bin"
    else:
        item["path"] = "../escape"
    store = entry.SftTransport(config, dict(ASSET_GET={item["path"]: url}))
    with pytest.raises(ValueError) as error:
        store.download_asset(item, tmp_path / "asset", config)
    assert "Signature=" not in str(error.value)
    assert not (tmp_path / "asset").exists()


def test_valid_adapter_source_download_is_size_and_hash_checked(entry, config, tmp_path):
    item = dict(path="adapter/config.bin", source_oss=config["adapter_oss_prefix"] + "config.bin", bytes=4,
                sha256=hashlib.sha256(b"data").hexdigest())
    url = "https://futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com/niezichao.nzc/m2d_train_v1/runs/test-checkpoint/adapter/config.bin?Signature=fixture"
    store = entry.SftTransport(config, dict(ASSET_GET={item["path"]: url}))
    store.opener = SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(b"data"))
    target = tmp_path / "asset"
    store.download_asset(item, target, config)
    assert target.read_bytes() == b"data"
    with pytest.raises(FileExistsError):
        store.download_asset(item, target, config)


@pytest.mark.parametrize("mode", ["success", "truncated", "exception"])
def test_smoke_starts_before_calls_and_preserves_failure_raw(entry, config, tmp_path, monkeypatch, capsys, mode):
    from experiments.structured_dev_20261114 import checkpoint_provider as cp
    (tmp_path / "job_config.json").write_text(json.dumps(config))
    (tmp_path / "signed_io.json").write_text("{}")
    (tmp_path / "music_planner_adapters.py").write_bytes(b"reviewed loader fixture")
    config["expected_loader_sha256"] = hashlib.sha256((tmp_path / "music_planner_adapters.py").read_bytes()).hexdigest()
    config["provider_source_sha256"] = entry.sha_file(cp.__file__)
    (tmp_path / "job_config.json").write_text(json.dumps(config))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(entry.SftTransport, "put", lambda *args: None)
    monkeypatch.setattr(entry, "runtime_preflight", lambda *args: {"runtime": "test fixture"})
    monkeypatch.setattr(entry, "stage_assets", lambda *args: tmp_path)
    calls = []
    output = tmp_path / "sft-test-records"

    class Provider:
        identity = {"provider": "fixture, not actual SFT"}

        def generate(self, request):
            index = len(calls)
            assert (output / ("%02d.started.json" % index)).exists()
            calls.append(request)
            if mode == "exception":
                raise RuntimeError("inference unavailable")
            value = response(request, config["cases"][index]["seed"])
            if mode == "truncated":
                value["stop_reason"] = value["raw"]["stop_reason"] = "budget_truncation"
            return value

    monkeypatch.setattr(cp, "create", lambda *args: Provider())
    with pytest.raises(SystemExit) as error:
        entry.main()
    report = json.loads(capsys.readouterr().out)
    assert report["full_collection_started"] is False
    if mode == "success":
        assert error.value.code == 0 and report["validated_cases"] == 2 and len(calls) == 2
    else:
        assert error.value.code == 2 and report["validated_cases"] == 0 and len(calls) == 1
        assert not (output / "00.validated.json").exists()
    if mode == "truncated":
        assert json.loads((output / "00.raw.json").read_text())["stop_reason"] == "budget_truncation"
    elif mode == "exception":
        failure = json.loads((output / "00.error.json").read_text())
        assert failure["usage"] == dict(input_tokens=None, output_tokens=None)
        assert failure["new_token_ids"] is None


def test_packaged_frozen_loader_is_byte_identical():
    path = Path(__file__).parents[1] / "experiments/structured_dev_20261114/cloud_sft_v3_native/music_planner_adapters.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "01dcd73b7a4ba19dca26f9dc4e30142f539b27ecfc7979ae2600be8ca8c6fae5"


def test_package_bootstrap_import_does_not_use_bare_entry(monkeypatch):
    import importlib
    import sys
    monkeypatch.setitem(sys.modules, "entry", SimpleNamespace())
    module = importlib.import_module("experiments.structured_dev_20261114.cloud_qwen32_native.bootstrap")
    assert module.NoRedirect.__module__ == "experiments.structured_dev_20261114.cloud_qwen32_native.entry"
    sft = importlib.import_module("experiments.structured_dev_20261114.cloud_sft_v3_native.bootstrap")
    assert sft.SftTransport.__module__ == "experiments.structured_dev_20261114.cloud_sft_v3_native.entry"


def test_error_messages_do_not_expose_signed_urls(entry):
    message = entry.error_message(RuntimeError("failed https://example.invalid/x?Signature=private&Expires=123"))
    assert "Signature=" not in message and "private" not in message
