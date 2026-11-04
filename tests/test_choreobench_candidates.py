import copy
import json

import pytest

from choreobench import baseline, candidates
from choreobench.io import object_sha, read_json
from choreobench.providers import MockProvider
from test_choreobench_baseline import dataset_file, plan_from_result


def collected(tmp_path, *, seconds=12, config=None):
    source, song = dataset_file(tmp_path, seconds=seconds)
    origin = tmp_path / "origin"
    result = baseline.run_baseline(source, origin, config or {})
    bank_path = tmp_path / "local-bank.json"
    summary = candidates.export_candidates(origin, bank_path)
    return source, song, origin, result, bank_path, summary


@pytest.mark.parametrize("mode,new_calls", [("concat", 0), ("global", 1), ("hierarchical", 3)])
def test_exact_local_samples_reused_across_modes_without_local_provider(tmp_path, monkeypatch, mode, new_calls):
    source, _, origin, original, bank_path, summary = collected(tmp_path)
    outer_requests = []
    create = baseline.create_provider
    def only_outer(spec, role, base_dir):
        assert role == "planner", "reuse must not instantiate or audit the local/native provider"
        provider = create(spec, role, base_dir)
        generate = provider.generate
        def observed(request):
            outer_requests.append(copy.deepcopy(request))
            return generate(request)
        provider.generate = observed
        return provider
    monkeypatch.setattr(baseline, "create_provider", only_outer)
    out = tmp_path / mode
    config = dict(mode=mode, local_candidates_path=str(bank_path), max_calls=max(1, new_calls),
                  planner_max_output_tokens=8192)
    result = baseline.run_baseline(source, out, config)
    assert result["complete"] and result["synthetic"]
    assert result["calls_used"] == result["new_calls_used"] == new_calls
    assert result["local_calls_reused"] == summary["recorded_local_calls"] == 3
    assert result["original_collection_calls"] == 3
    assert result["original_collection_costs"] == summary["original_collection_costs"]
    assert len(result["costs"]) == new_calls
    assert len(list((out / "calls").glob("*.started.json"))) == new_calls
    expected_texts = {entry["call_id"]: entry["raw"]["result"]["text"] for entry in read_json(bank_path)["entries"]}
    for request in outer_requests:
        for proposal in request["proposals"]:
            assert proposal["caption"] == expected_texts[proposal["call_id"]]
    if mode == "concat":
        assert plan_from_result(out, result)["events"] == plan_from_result(origin, original)["events"]
    # Resume validates bank/current inputs and consumes neither local nor outer calls.
    assert baseline.run_baseline(source, out, config, resume=True) == result
    assert len(outer_requests) == new_calls


def test_original_native_assets_need_not_exist_when_reusing_a_verified_recording(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    native_config = dict(kind="native", family="omni_captioner", model="missing-native-weights",
                         checkpoint="missing-portable-checkpoint")
    def synthetic_recording(spec, role, base_dir):
        assert role == "local" and spec == native_config
        fixture = MockProvider({"kind": "mock"}, "local")
        fixture.identity = dict(kind="native", synthetic=False,
            artifact_identity={"fixture": "recording-boundary-test-not-real-model-verification"})
        generate = fixture.generate
        def recorded(request):
            result = generate(request)
            result["synthetic"] = False
            return result
        fixture.generate = recorded
        return fixture
    monkeypatch.setattr(baseline, "create_provider", synthetic_recording)
    origin = tmp_path / "origin"
    baseline.run_baseline(source, origin, {"local_provider": native_config})
    bank = tmp_path / "bank.json"
    candidates.export_candidates(origin, bank)
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not open current native weights/checkpoint")
    monkeypatch.setattr(baseline, "create_provider", forbidden)
    result = baseline.run_baseline(source, tmp_path / "reuse", dict(
        local_provider=native_config, local_candidates_path=str(bank), max_calls=1))
    assert result["calls_used"] == 0 and result["local_calls_reused"] == 3
    assert result["synthetic"]  # Synthetic dataset can never become a real benchmark.


@pytest.mark.parametrize("change", [
    {"seed": 7}, {"max_output_tokens": 1000}, {"stride_seconds": 2.},
    {"candidates_per_window": 2}, {"local_provider": {"kind": "native", "model": "different"}}])
def test_local_configuration_mismatch_rejected_before_any_provider_creation(tmp_path, monkeypatch, change):
    source, _, _, _, bank, _ = collected(tmp_path)
    monkeypatch.setattr(baseline, "create_provider", lambda *args, **kwargs: pytest.fail("unexpected provider"))
    with pytest.raises(ValueError, match="identity differs"):
        baseline.run_baseline(source, tmp_path / "reuse", dict(change, local_candidates_path=str(bank)))


def test_dataset_bytes_and_actual_audio_are_bound(tmp_path):
    source, _, _, _, bank, _ = collected(tmp_path)
    # Same JSON meaning, different bytes is a different immutable dataset identity.
    source.write_text(json.dumps(read_json(source)), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        baseline.run_baseline(source, tmp_path / "reuse", {"local_candidates_path": str(bank)})


@pytest.mark.parametrize("field", ["text", "request", "provider", "order", "cost", "binding", "synthetic"])
def test_resealed_bank_cannot_break_original_raw_request_and_result_bindings(tmp_path, field):
    _, _, _, _, path, _ = collected(tmp_path)
    bank = read_json(path)
    if field == "text":
        bank["entries"][0]["raw"]["result"]["text"] = "altered"
    elif field == "request":
        bank["entries"][0]["started"]["request"]["audio"]["parent_start"] = 1
    elif field == "provider":
        bank["entries"][0]["raw"]["result"]["identity"]["kind"] = "different-model"
    elif field == "order":
        bank["entries"].reverse()
    elif field == "cost":
        bank["original_collection_costs"][0]["output_tokens"] = 0
    elif field == "binding":
        bank["binding"]["local_config"]["seed"] = 0
    else:
        bank["synthetic"] = False
    bank.pop("record_sha256")
    path.write_text(json.dumps(baseline._sealed(bank)), encoding="utf-8")
    with pytest.raises(ValueError):
        candidates.load_candidates(path)


def test_bank_publication_is_exclusive_and_original_derived_audio_is_checked(tmp_path):
    _, _, origin, _, bank, _ = collected(tmp_path)
    with pytest.raises(FileExistsError):
        candidates.export_candidates(origin, bank)
    first = read_json(bank)["entries"][0]
    from pathlib import Path
    derived = Path(first["started"]["request"]["audio"]["path"])
    contents = bytearray(derived.read_bytes())
    contents[-1] ^= 1
    derived.write_bytes(bytes(contents))
    with pytest.raises(ValueError, match="expert observation bytes"):
        candidates.export_candidates(origin, tmp_path / "changed.json")


def test_recorded_local_model_failure_is_reused_not_filtered_or_resampled(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    create = baseline.create_provider
    def bad_provider(spec, role, base_dir):
        provider = create(spec, role, base_dir)
        generate = provider.generate
        def bad(request):
            result = generate(request)
            result["text"] = "Original untimed model answer"
            return result
        provider.generate = bad
        return provider
    monkeypatch.setattr(baseline, "create_provider", bad_provider)
    origin = tmp_path / "origin"
    result = baseline.run_baseline(source, origin, {"continue_on_model_error": True})
    assert result["failed_songs"] and result["calls_used"] == 1
    bank = tmp_path / "failed-bank.json"
    summary = candidates.export_candidates(origin, bank)
    assert summary["expected_local_calls"] == 3 and summary["recorded_local_calls"] == 1
    assert summary["failed_songs"][0]["status"] == "local_model_error"
    monkeypatch.setattr(baseline, "create_provider", lambda *args, **kwargs: pytest.fail("must reuse recorded failure"))
    reused = baseline.run_baseline(source, tmp_path / "reuse", dict(
        local_candidates_path=str(bank), continue_on_model_error=True, max_calls=1))
    assert reused["complete"] and reused["success_count"] == 0 and reused["selected_count"] == 1
    assert reused["calls_used"] == 0 and reused["local_calls_reused"] == 1
    assert reused["failed_songs"][0]["retained_in_evaluation_denominator"]


def test_infrastructure_failure_cannot_be_exported_as_a_model_candidate(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    create = baseline.create_provider
    seen = []
    def provider(spec, role, base_dir):
        value = create(spec, role, base_dir)
        generate = value.generate
        def fail_second(request):
            seen.append(1)
            if len(seen) == 2:
                raise OSError("unknown provider outcome")
            result = generate(request)
            result["costs"].update(input_tokens=8, output_tokens=12)
            return result
        value.generate = fail_second
        return value
    monkeypatch.setattr(baseline, "create_provider", provider)
    origin = tmp_path / "origin"
    with pytest.raises(ValueError, match="Recorded provider failure"):
        baseline.run_baseline(source, origin, {})
    failed = read_json(origin / "results.json")
    assert failed["costs"][0]["input_tokens"] == 8 and failed["costs"][0]["output_tokens"] == 12
    assert failed["unknown_outcomes_count"] == 1 and failed["unknown_usage_count"] == 1
    with pytest.raises(ValueError, match="Recorded provider failure"):
        candidates.export_candidates(origin, tmp_path / "bank.json")


def test_bank_hashing_is_per_song_boundary_not_per_window(tmp_path, monkeypatch):
    source, _, _, _, bank, _ = collected(tmp_path, seconds=24)
    hashes = []
    original = candidates.file_sha
    def counted(path):
        if str(path) == str(bank):
            hashes.append(1)
        return original(path)
    monkeypatch.setattr(candidates, "file_sha", counted)
    result = baseline.run_baseline(source, tmp_path / "reuse", {"local_candidates_path": str(bank), "max_calls": 1})
    assert result["local_calls_reused"] >= 5
    assert len(hashes) == 2  # Before song and before final publication; O(N) bytes overall.


def test_changed_bank_during_outer_generation_is_rejected_and_known_cost_retained(tmp_path, monkeypatch):
    source, _, _, _, bank, _ = collected(tmp_path)
    create = baseline.create_provider
    def modifying(spec, role, base_dir):
        assert role == "planner"
        provider = create(spec, role, base_dir)
        generate = provider.generate
        def change(request):
            result = generate(request)
            result["costs"].update(input_tokens=100, output_tokens=100)
            bank.write_text(bank.read_text(encoding="utf-8") + " ", encoding="utf-8")
            return result
        provider.generate = change
        return provider
    monkeypatch.setattr(baseline, "create_provider", modifying)
    out = tmp_path / "reuse"
    with pytest.raises(ValueError, match="bank bytes/metadata changed"):
        baseline.run_baseline(source, out, dict(mode="global", local_candidates_path=str(bank)))
    failed = read_json(out / "results.json")
    assert not failed["complete"] and failed["costs"][0]["output_tokens"] == 100


def test_local_request_comparison_ignores_only_derived_path(tmp_path):
    _, _, _, _, path, _ = collected(tmp_path)
    bank = candidates.load_candidates(path)
    request = copy.deepcopy(next(iter(bank.entries.values()))["started"]["request"])
    request["audio"]["path"] = "/another-run/derived.wav"
    assert bank.call(request)["text"]
    for key, value in (("parent_start", 1), ("sha256", "1" * 64), ("num_samples", 1)):
        changed = copy.deepcopy(request)
        changed["audio"][key] = value
        with pytest.raises(ValueError, match="Reused local request differs"):
            bank.call(changed)
