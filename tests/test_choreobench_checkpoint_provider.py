import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import wave

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "experiments/structured_dev_20261114/checkpoint_provider.py"
SPEC = importlib.util.spec_from_file_location("tested_sft_checkpoint_provider", SOURCE)
provider_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provider_module)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    (model / "config.json").write_text("{}")
    (model / "weights.safetensors").write_bytes(b"base weights")
    (model / "model.safetensors.index.json").write_text('{"weight_map":{"w":"weights.safetensors"}}')
    (adapter / "adapter_model.safetensors").write_bytes(b"selected adapter")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "processor_config.json").write_text("{}")
    loader = tmp_path / "loader.py"
    loader.write_text("def load_planner_adapter(*args, **kwargs):\n    return _load_impl(*args, **kwargs)\n"
                      "def load_native_processor(*args, **kwargs):\n    return _processor_impl(*args, **kwargs)\n")
    monkeypatch.setattr(provider_module, "EXPECTED_ADAPTER_SHA256",
                        provider_module._sha(adapter / "adapter_model.safetensors"))
    monkeypatch.setattr(provider_module, "EXPECTED_LOADER_SHA256", provider_module._sha(loader))
    spec = dict(model=str(model), checkpoint=str(adapter), adapter_source=str(loader), device_map="cpu")
    provider = provider_module.create(spec)
    state = SimpleNamespace(tokens=[77, 151645], calls=[], rows=[], change_prefix=False, structured=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def generate(self, input_ids, **kwargs):
            state.calls.append(copy.deepcopy(kwargs))
            prefix = input_ids.clone()
            if state.change_prefix:
                prefix[0, 0] += 1
            output = torch.cat((prefix, torch.tensor([state.tokens], dtype=torch.long)), dim=1)
            return SimpleNamespace(sequences=output) if state.structured else output

    def prefix(row):
        state.rows.append(copy.deepcopy(row))
        return dict(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    native = SimpleNamespace(model=Model(), max_length=8192, prefix=prefix, _move=lambda batch: batch,
                             tokenizer=SimpleNamespace(decode=lambda tokens, **kwargs: "The arm sweeps left."))
    provider.adapter = native
    audio = tmp_path / "window.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 81000)
    state.request = dict(call_id="song-A/untimed/0000/00", response_kind="local_caption", seed=42,
                         max_output_tokens=1024, audio=dict(path=str(audio), sample_rate=16000,
                         num_samples=81000, sha256=provider_module._sha(audio)))
    state.provider, state.spec, state.native = provider, spec, native
    state.adapter_path, state.loader = adapter, loader
    return state


def test_measures_actual_tokens_and_uses_audio_only_prefix(fixture):
    result = fixture.provider.generate(fixture.request)
    assert result["costs"]["input_tokens"] == 3
    assert result["costs"]["output_tokens"] == 2
    assert result["raw"]["new_token_ids"] == [77, 151645]
    assert result["stop_reason"] == "eos"
    case_id = fixture.request["call_id"]
    assert result["raw"]["canonical_case_id"] == case_id
    assert result["raw"]["case_id_source"] == "predeclared_song_window_candidate_id"
    expected_seed = int(hashlib.sha256(("sol61-select:" + case_id).encode()).hexdigest()[:8], 16)
    assert result["raw"]["seed"] == expected_seed
    assert fixture.rows[0] == dict(id=case_id, audio_path=fixture.request["audio"]["path"],
                                  audio_sha256=fixture.request["audio"]["sha256"],
                                  audio_window_seconds=[0.0, 5.0625])
    assert set(fixture.calls[0]) == {"generation_config"}
    config = fixture.calls[0]["generation_config"]
    assert config.max_new_tokens == 1024
    assert config.to_dict() == fixture.provider.identity["resolved_generation_config"]
    assert config.forced_eos_token_id is None
    for name, value in provider_module.DECODING.items():
        assert getattr(config, name) == value


@pytest.mark.parametrize("tokens,stop", [
    ([77, 151643], "eos"),
    ([77] * 1023 + [151645], "eos"),
    ([77] * 1024, "budget_truncation"),
    ([77, 78], "incomplete_output"),
    ([], "incomplete_output"),
])
def test_observed_stops_are_not_guessed_from_text(fixture, tokens, stop):
    fixture.tokens = tokens
    result = fixture.provider.generate(fixture.request)
    assert result["stop_reason"] == stop
    assert result["costs"]["output_tokens"] == len(tokens)
    assert result["raw"]["new_token_ids"] == tokens


@pytest.mark.parametrize("budget", [512, 2048, True, 1024.0])
def test_wrong_decode_budget_rejected_before_model_call(fixture, budget):
    request = dict(fixture.request, max_output_tokens=budget)
    with pytest.raises(ValueError, match="exactly 1024"):
        fixture.provider.generate(request)
    assert fixture.calls == [] and fixture.rows == []


def test_explicit_case_seed_is_independent_of_method_name(fixture):
    first = dict(fixture.request, call_id="method-A/clip-0", case_id="clip-0/window-0/candidate-0")
    second = dict(first, call_id="method-B/clip-0", seed=999)
    a = fixture.provider.generate(first)
    b = fixture.provider.generate(second)
    assert a["raw"]["seed"] == b["raw"]["seed"]
    assert a["raw"]["canonical_case_id"] == first["case_id"]
    assert a["raw"]["case_id_source"] == "explicit_request_case_id"


@pytest.mark.parametrize("case_id", ["", " ", "x" * 513, False])
def test_invalid_explicit_case_is_rejected(fixture, case_id):
    with pytest.raises(ValueError, match="case_id"):
        fixture.provider.generate(dict(fixture.request, case_id=case_id))
    assert fixture.calls == []


def test_ambiguous_case_mapping_is_rejected(fixture):
    with pytest.raises(ValueError, match="Supply case_id"):
        fixture.provider.generate(dict(fixture.request, call_id="method-A/case-A"))
    assert fixture.calls == []


def test_waveform_identity_is_checked_before_loading(fixture):
    request = copy.deepcopy(fixture.request)
    request["audio"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Audio SHA256"):
        fixture.provider.generate(request)
    request["audio"] = dict(fixture.request["audio"], num_samples=81001)
    with pytest.raises(ValueError, match="5.0625"):
        fixture.provider.generate(request)
    assert fixture.calls == []


def test_prefix_mutation_and_excess_output_are_rejected(fixture):
    fixture.change_prefix = True
    with pytest.raises(ValueError, match="changed its input prefix"):
        fixture.provider.generate(fixture.request)
    fixture.change_prefix = False
    fixture.tokens = [77] * 1025
    with pytest.raises(ValueError, match="more tokens"):
        fixture.provider.generate(fixture.request)


def test_structured_generation_output_is_supported(fixture):
    fixture.structured = True
    assert fixture.provider.generate(fixture.request)["stop_reason"] == "eos"


def test_identity_cannot_be_mutated_by_caller(fixture):
    identity = fixture.provider.identity
    identity["sampling"]["top_k"] = 999
    identity["sampling"]["eos_token_id"].clear()
    assert fixture.provider.identity["sampling"]["top_k"] == 20
    assert fixture.provider.identity["sampling"]["eos_token_id"] == [151645, 151643]
    assert fixture.provider.identity["adapter_source_sha256"] == provider_module._sha(fixture.loader)
    assert fixture.provider.identity["base_weight_shards"][0]["sha256"] == provider_module._sha(
        Path(fixture.spec["model"]) / "weights.safetensors")


def test_artifact_drift_blocks_later_calls(fixture):
    (fixture.adapter_path / "adapter_config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="artifact changed"):
        fixture.provider.generate(fixture.request)
    assert fixture.calls == []


def test_explicit_device_map_is_forwarded_to_loader(fixture, monkeypatch):
    calls = []
    fixture.provider.adapter = None
    fixture.provider.spec["device_map"] = "cuda:0"

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return fixture.native

    name = "_sol61_adapters_" + provider_module.EXPECTED_LOADER_SHA256[:12]
    module = ModuleType(name)
    module.__file__ = str(fixture.loader)
    module.__spec__ = SimpleNamespace(origin=str(fixture.loader))
    module.__dict__["_load_impl"] = load
    processor = SimpleNamespace(tokenizer=fixture.native.tokenizer)
    module.__dict__["_processor_impl"] = lambda family, path: processor
    exec(compile(fixture.loader.read_text(), str(fixture.loader), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, name, module)
    fixture.provider._load()
    assert calls[0][1]["device_map"] == "cuda:0"
    assert calls[0][1]["dtype"] == torch.bfloat16
    assert fixture.provider.adapter.processor is processor
    assert fixture.provider._processor_export_verified is True


def test_unselected_adapter_is_rejected_before_model_loading(fixture):
    (fixture.adapter_path / "adapter_model.safetensors").write_bytes(b"different adapter")
    with pytest.raises(ValueError, match="selected SFT v3"):
        provider_module.create(fixture.spec)


def test_missing_processor_export_has_no_fallback(fixture):
    (fixture.adapter_path / "processor_config.json").unlink()
    with pytest.raises(ValueError, match="Portable processor"):
        provider_module.create(fixture.spec)


def test_bfloat16_activations_with_float32_projection_are_supported(fixture):
    class MixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.projection = torch.nn.Linear(8, 8, dtype=torch.float32)

        def generate(self, input_ids, **kwargs):
            projected = self.projection(torch.ones((1, 8), dtype=torch.bfloat16))
            assert projected.dtype == torch.bfloat16
            return torch.cat((input_ids, torch.tensor([[77, 151645]])), dim=1)

    fixture.native.model = MixedModel()
    assert fixture.provider.generate(fixture.request)["stop_reason"] == "eos"


def test_model_default_generation_controls_do_not_override_contract(fixture):
    path = Path(fixture.spec["model"]) / "generation_config.json"
    path.write_text('{"forced_eos_token_id":151643,"min_length":500}')
    provider = provider_module.create(fixture.spec)
    provider.adapter = fixture.native
    result = provider.generate(fixture.request)
    config = fixture.calls[0]["generation_config"]
    assert config.forced_eos_token_id is None and config.min_length == 0
    assert result["raw"]["resolved_generation_config"]["forced_eos_token_id"] is None
    assert provider.identity["base_generation_config_sha256"] == provider_module._sha(path)


def test_generation_config_mutation_blocks_dispatch(fixture):
    fixture.provider.generation_config.forced_eos_token_id = 151643
    with pytest.raises(ValueError, match="generation contract changed"):
        fixture.provider.generate(fixture.request)
    assert fixture.calls == []


def test_new_model_generation_file_is_artifact_drift(fixture):
    (Path(fixture.spec["model"]) / "generation_config.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        fixture.provider.generate(fixture.request)
    assert fixture.calls == []


def test_unknown_cached_loader_is_rejected(fixture, monkeypatch):
    fixture.provider.adapter = None
    name = "_sol61_adapters_" + provider_module.EXPECTED_LOADER_SHA256[:12]
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(load_planner_adapter=lambda *a, **k: fixture.native))
    with pytest.raises(ValueError, match="loader provenance"):
        fixture.provider._load()


def test_noncanonical_sample_clock_is_explicitly_rejected(fixture):
    request = copy.deepcopy(fixture.request)
    request["audio"].update(sample_rate=44100, num_samples=223256)
    with pytest.raises(ValueError, match="16000 Hz"):
        fixture.provider.generate(request)
    assert fixture.calls == []


def test_journal_retains_real_usage_and_replays_without_dispatch(fixture, tmp_path):
    from choreobench.baseline import _Journal

    journal = _Journal(tmp_path, "a" * 64, 2)
    result = journal.call(fixture.provider, fixture.request)
    assert result["costs"]["input_tokens"] == 3
    assert result["costs"]["output_tokens"] == 2
    assert journal.call(fixture.provider, fixture.request) == result
    assert len(fixture.calls) == 1


def test_journal_records_truncation_without_successful_caption(fixture, tmp_path):
    from choreobench.baseline import _Journal, ModelOutputError

    fixture.tokens = [77] * 1024
    journal = _Journal(tmp_path, "a" * 64, 2)
    with pytest.raises(ModelOutputError, match="finish normally"):
        journal.call(fixture.provider, fixture.request)
    record = journal.records[fixture.request["call_id"]]["result"]
    assert record["stop_reason"] == "budget_truncation"
    assert record["costs"]["input_tokens"] == 3
    assert record["costs"]["output_tokens"] == 1024


def test_catalog_preflight_declares_case_ids_and_seeds():
    path = SOURCE.parent / "collect_sft_v3.py"
    spec = importlib.util.spec_from_file_location("tested_sft_collector", path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    dataset = dict(songs=[dict(song_id="song-A", audio=dict(sample_rate=16000, num_samples=81000))])
    cases = collector.validate_clock(dataset)
    assert [c["case_id"] for c in cases] == ["song-A/untimed/0000/00", "song-A/untimed/0000/01"]
    assert cases[0]["seed"] == int(hashlib.sha256(
        b"sol61-select:song-A/untimed/0000/00").hexdigest()[:8], 16)
    dataset["songs"][0]["audio"]["sample_rate"] = 44100
    with pytest.raises(ValueError, match="Preflight requires"):
        collector.validate_clock(dataset)


def test_processor_load_failure_does_not_fall_back_to_base(fixture, monkeypatch):
    fixture.provider.adapter = None
    name = "_sol61_adapters_" + provider_module.EXPECTED_LOADER_SHA256[:12]
    module = ModuleType(name)
    module.__file__ = str(fixture.loader)
    module.__spec__ = SimpleNamespace(origin=str(fixture.loader))

    def rejected_processor(*args, **kwargs):
        raise ValueError("Selected processor export failed to load")

    def unexpected_model(*args, **kwargs):
        pytest.fail("Do not load weights after processor validation fails")

    module.__dict__.update(_processor_impl=rejected_processor, _load_impl=unexpected_model)
    exec(compile(fixture.loader.read_text(), str(fixture.loader), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError, match="processor export failed"):
        fixture.provider._load()
    assert fixture.provider.adapter is None
