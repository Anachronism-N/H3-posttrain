import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import urllib.error
import urllib.parse

import pytest


@pytest.fixture
def entry():
    source = Path(__file__).parents[1] / "experiments/structured_dev_20261114/cloud_qwen32_native/entry.py"
    spec = importlib.util.spec_from_file_location("cloud_oss_transport_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    return {
        "run_id": "transport-test",
        "model_oss_prefix": "oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/models/qwen-test/",
        "result_oss_prefix": "oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/runs/transport-test/",
    }


def capabilities(config):
    result = {}
    for method, field in (("GET", "model_oss_prefix"), ("PUT", "result_oss_prefix"), ("HEAD", "result_oss_prefix")):
        prefix = urllib.parse.urlsplit(config[field]).path
        result[method] = {
            name: "https://futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com" + prefix + name + "?Signature=unit-only"
            for name in ("config.json", "result.json", "raw.json", "transfer_complete.json")
        }
    return result


@pytest.mark.parametrize("mutation", ["http", "host", "path", "fragment"])
def test_rejects_capability_outside_https_object_scope(entry, config, mutation):
    links = capabilities(config)
    url = links["GET"]["config.json"]
    if mutation == "http":
        url = url.replace("https:", "http:")
    elif mutation == "host":
        url = url.replace("futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com", "wrong-bucket.invalid")
    elif mutation == "path":
        url = url.replace("config.json", "another.json")
    else:
        url += "#fragment"
    links["GET"]["config.json"] = url
    store = entry.OssTransport(config, links)
    with pytest.raises(ValueError, match="scope") as error:
        store.open("GET", "config.json")
    assert "Signature=" not in str(error.value)


def test_result_prefix_cannot_target_another_run(entry, config):
    links = capabilities(config)
    config["result_oss_prefix"] = config["result_oss_prefix"].replace("transport-test", "another-run")
    with pytest.raises(ValueError, match="this new run"):
        entry.OssTransport(config, links)


def test_redirects_are_rejected_without_exposing_url(entry):
    with pytest.raises(RuntimeError, match="redirects"):
        entry.NoRedirect().redirect_request(None, None, 302, "", {}, "https://wrong.invalid/?Signature=unit-only")


def test_download_uses_exact_size_and_never_overwrites_local_file(entry, config, tmp_path):
    store = entry.OssTransport(config, capabilities(config))
    store.open = lambda *args: io.BytesIO(b"data")
    target = tmp_path / "model"
    store.download("config.json", target, 4)
    assert target.read_bytes() == b"data"
    with pytest.raises(FileExistsError):
        store.download("config.json", target, 4)
    assert target.read_bytes() == b"data"


@pytest.mark.parametrize("body", [b"dat", b"datax"])
def test_download_rejects_short_or_oversized_objects(entry, config, tmp_path, body):
    store = entry.OssTransport(config, capabilities(config))
    store.open = lambda *args: io.BytesIO(body)
    target = tmp_path / "model"
    with pytest.raises(ValueError, match="declared size"):
        store.download("config.json", target, 4)
    assert target.stat().st_size <= 4


def test_existing_remote_record_blocks_put(entry, config, tmp_path):
    store = entry.OssTransport(config, capabilities(config))
    calls = []

    def open_object(method, *args):
        calls.append(method)
        return io.BytesIO()

    store.open = open_object
    target = tmp_path / "raw.json"
    target.write_text("{}")
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        store.put(target)
    assert calls == ["HEAD"]


def test_remote_auth_failure_is_not_treated_as_missing(entry, config, tmp_path):
    store = entry.OssTransport(config, capabilities(config))
    calls = []

    def open_object(method, *args):
        calls.append(method)
        raise urllib.error.HTTPError("", 403, "Forbidden", {}, None)

    store.open = open_object
    target = tmp_path / "raw.json"
    target.write_text("{}")
    with pytest.raises(urllib.error.HTTPError):
        store.put(target)
    assert calls == ["HEAD"]


def test_missing_remote_record_allows_exact_put_payload(entry, config, tmp_path):
    store = entry.OssTransport(config, capabilities(config))
    calls = []

    def open_object(method, name, data=None):
        calls.append((method, name, data))
        if method == "HEAD":
            raise urllib.error.HTTPError("", 404, "Not Found", {}, None)
        response = io.BytesIO()
        response.status = 200
        return response

    store.open = open_object
    target = tmp_path / "raw.json"
    target.write_bytes(b'{"tokens":[1,2]}')
    store.put(target)
    assert calls == [("HEAD", "raw.json", None), ("PUT", "raw.json", target.read_bytes())]


def test_failed_remote_upload_keeps_durable_local_raw(entry, tmp_path):
    target = tmp_path / "raw.json"

    def fail(path):
        assert json.loads(path.read_text()) == {"tokens": [1, 2], "stop_reason": "eos"}
        raise RuntimeError("upload unavailable")

    with pytest.raises(RuntimeError, match="upload unavailable"):
        entry.write_new(SimpleNamespace(put=fail), target, {"tokens": [1, 2], "stop_reason": "eos"})
    assert target.exists()
    with pytest.raises(FileExistsError):
        entry.write_new(SimpleNamespace(put=fail), target, {})


def test_initial_upload_failure_stops_before_runtime_or_model_calls(entry, config, tmp_path, monkeypatch, capsys):
    config["input_provenance"] = "test controls only"
    (tmp_path / "job_config.json").write_text(json.dumps(config))
    (tmp_path / "signed_io.json").write_text(json.dumps(capabilities(config)))
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setenv("RANK", "0")

    def fail(*args):
        raise RuntimeError("upload unavailable")

    monkeypatch.setattr(entry.OssTransport, "put", fail)
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["model_calls_attempted"] == 0
    assert report["validated_units"] == 0
    assert "runtime" not in report
    assert "Signature=" not in json.dumps(report)
    assert (tmp_path / "transport-test-records/job_config.json").exists()


def test_valid_full_unit_clock_is_accepted(entry):
    value = {"unit_id": "u0", "events": [{"start": 0, "end": 5, "track": "body", "action": "Step and settle."}]}
    entry.validate_unit(value, {"unit_id": "u0", "start": 0, "end": 5})


@pytest.mark.parametrize("mutation", ["boolean", "nonfinite", "gap", "offgrid", "short", "overrun", "track", "coverage"])
def test_clock_contract_still_rejects_invalid_outputs(entry, mutation):
    event = {"start": 0, "end": 5, "track": "body", "action": "Step and settle."}
    field, value = {
        "boolean": ("start", False),
        "nonfinite": ("end", float("nan")),
        "gap": ("start", 0.25),
        "offgrid": ("end", 4.9),
        "short": ("end", 0.25),
        "overrun": ("end", 5.25),
        "track": ("track", "head"),
        "coverage": ("end", 4.75),
    }[mutation]
    event[field] = value
    with pytest.raises(ValueError):
        entry.validate_unit({"unit_id": "u0", "events": [event]}, {"unit_id": "u0", "start": 0, "end": 5})


def test_loader_uses_native_to_without_device_map(entry, config, tmp_path, monkeypatch, capsys):
    config.update(input_provenance="test controls only", allocated_gpus=8, model_device="cuda:0")
    (tmp_path / "job_config.json").write_text(json.dumps(config))
    (tmp_path / "signed_io.json").write_text(json.dumps(capabilities(config)))
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(entry.OssTransport, "put", lambda *args: None)
    monkeypatch.setattr(entry, "stage_model", lambda *args: "fixture-model")
    calls = []

    class Model:
        def to(self, device):
            calls.append(("to", device))
            raise RuntimeError("fixture stops before generation")

    def from_pretrained(path, **kwargs):
        assert "device_map" not in kwargs
        assert kwargs == {"torch_dtype": "bf16-fixture", "local_files_only": True, "trust_remote_code": False}
        calls.append(("from_pretrained", path))
        return Model()

    torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda="fixture"),
        bfloat16="bf16-fixture", cuda=SimpleNamespace(device_count=lambda: 8,
        get_device_properties=lambda i: SimpleNamespace(name="fixture", total_memory=1)))
    transformers = SimpleNamespace(__version__="fixture", GenerationConfig=object,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: None),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["model_calls_attempted"] == 0
    assert report["error"]["message"] == "fixture stops before generation"
    assert calls == [("from_pretrained", "fixture-model"), ("to", "cuda:0")]
