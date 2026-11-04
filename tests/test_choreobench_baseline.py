import copy
import io
import json
from pathlib import Path
import sys
import types
import urllib.error
import wave

import pytest

from choreobench import baseline
from choreobench.audio import inspect_audio, make_windows, write_window
from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.providers import ChatCompletionsProvider, NativeProvider, ReplayProvider, parse_json_response
from choreobench.schema import validate_plan


def dataset_file(tmp_path, seconds=12, *, split="dev", rate=16000):
    count = int(seconds * rate)
    wav = tmp_path / "song.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x01\x00" * count)
    song = dict(song_id="song1", split=split, song_group_id="private-song-group",
        artist_group_id="private-artist-group", near_duplicate_group_id="private-duplicate-group",
        brief="A synthetic clock test, not musical evidence.",
        audio=dict(path="song.wav", sha256=file_sha(wav), sample_rate=rate, num_samples=count),
        music=dict(source="synthetic fixture, not a detector", beats=[0], downbeats=[0], sections=[
            dict(section_id="a", start=0, end=count // 2, label="unknown", repeat_group=None),
            dict(section_id="b", start=count // 2, end=count, label="unknown", repeat_group=None)]))
    path = tmp_path / "dataset.json"
    write_json(path, dict(schema="choreobench.dataset.v1", dataset_id="fixture", synthetic=True,
        grouping=dict(status="unverified", evidence=""), songs=[song]))
    return path, song


def plan_from_result(out, result):
    return read_json(out / result["plans"][0]["path"])


def test_tail_ownership_partitions_clock_and_keeps_full_observations():
    windows = make_windows(192000, 16000)
    assert [w["observation_start"] for w in windows] == [0, 81000, 111000]
    assert all(w["observation_end"] - w["observation_start"] == 81000 for w in windows)
    assert [(w["ownership_start"], w["ownership_end"]) for w in windows] == [
        (0, 81000), (81000, 111000), (111000, 192000)]
    assert windows[-1]["observation_start"] < windows[-2]["observation_end"]
    assert not windows[-1]["boundaries_are_action_boundaries"]


@pytest.mark.parametrize("count,rate,window,stride", [
    (80000, 16000, 5.0625, None), (81000, 16000, 5, None),
    (81000, 16000, 5.0625, 6), (81000, 16000, 5.0625, 0),
    (True, 16000, 5.0625, None), (81000, 16000, float("nan"), None)])
def test_invalid_window_contract_does_not_pad(count, rate, window, stride):
    with pytest.raises(ValueError):
        make_windows(count, rate, window, stride)


def test_nonintegral_sample_clock_and_dense_stride():
    windows = make_windows(600000, 44100, stride_seconds=2.)
    assert all(abs(w["sample_rounding_error"]) <= .5 for w in windows)
    assert windows[0]["ownership_start"] == 0 and windows[-1]["ownership_end"] == 600000
    assert all(a["ownership_end"] == b["ownership_start"] for a, b in zip(windows, windows[1:]))


def test_audio_slice_is_exact_and_cannot_be_silently_replaced(tmp_path):
    path, song = dataset_file(tmp_path)
    audio = inspect_audio(tmp_path / "song.wav")
    window = make_windows(song["audio"]["num_samples"], 16000)[-1]
    target = tmp_path / "derived.wav"
    receipt = write_window(audio, window, target)
    assert inspect_audio(target)["pcm"] == audio["pcm"][2 * window["observation_start"]:2 * window["observation_end"]]
    assert receipt["parent_audio_sha256"] == song["audio"]["sha256"]
    assert write_window(audio, window, target) == receipt
    target.write_bytes(target.read_bytes()[:-2])
    with pytest.raises(ValueError, match="bytes changed"):
        write_window(audio, window, target)


@pytest.mark.parametrize("mode,expected_calls", [("concat", 3), ("global", 4), ("hierarchical", 6)])
def test_all_three_modes_make_valid_synthetic_whole_clock_plans(tmp_path, mode, expected_calls):
    source, song = dataset_file(tmp_path)
    out = tmp_path / mode
    result = baseline.run_baseline(source, out, dict(mode=mode))
    assert result["complete"] and result["synthetic"]
    assert result["calls_used"] == expected_calls
    plan = plan_from_result(out, result)
    validate_plan(plan, song)
    assert plan["events"][0]["start"] == 0
    assert plan["events"][-1]["end"] == song["audio"]["num_samples"]
    assert all(a["end"] == b["start"] for a, b in zip(plan["events"], plan["events"][1:]))
    assert not result["music_or_physical_quality_verified"]
    assert not plan["provenance"]["baseline"]["global_context_consumed_by_native_local_model"]


def test_three_minute_schedule_is_a_synthetic_clock_smoke_only(tmp_path):
    source, song = dataset_file(tmp_path, seconds=181)
    out = tmp_path / "long"
    result = baseline.run_baseline(source, out, dict(mode="concat", max_calls=100))
    plan = plan_from_result(out, result)
    assert plan["events"][-1]["end"] == 181 * 16000
    assert result["calls_used"] == len(make_windows(181 * 16000, 16000))
    assert result["synthetic"] and not result["native_large_model_server_verified"]


def test_completed_resume_does_not_generate_and_checks_published_bytes(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    out = tmp_path / "out"
    result = baseline.run_baseline(source, out, {})
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached calls must not generate")
    monkeypatch.setattr("choreobench.providers.MockProvider.generate", forbidden)
    assert baseline.run_baseline(source, out, {}, resume=True) == result
    plan_path = out / result["plans"][0]["path"]
    plan_path.write_text(plan_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="plan bytes changed"):
        baseline.run_baseline(source, out, {}, resume=True)


def test_resume_refuses_changed_config_and_input(tmp_path):
    source, _ = dataset_file(tmp_path)
    out = tmp_path / "out"
    baseline.run_baseline(source, out, {})
    with pytest.raises(ValueError, match="identity changed"):
        baseline.run_baseline(source, out, {"seed": 1}, resume=True)
    data = read_json(source)
    data["songs"][0]["brief"] += " changed"
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        baseline.run_baseline(source, out, {}, resume=True)


def test_max_calls_is_entire_run_budget_not_per_song_or_resume(tmp_path):
    source, _ = dataset_file(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="insufficient"):
        baseline.run_baseline(source, out, {"max_calls": 2})
    assert not out.exists()  # No partial paid run when the known total is too large.


def test_pending_outcome_blocks_resume_instead_of_sending_again(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    out = tmp_path / "out"
    count = []
    def crashed(*args):
        count.append(1)
        raise KeyboardInterrupt("simulated process death after request start")
    monkeypatch.setattr("choreobench.providers.MockProvider.generate", crashed)
    with pytest.raises(KeyboardInterrupt):
        baseline.run_baseline(source, out, {})
    with pytest.raises(ValueError, match="Pending/unknown"):
        baseline.run_baseline(source, out, {}, resume=True)
    assert len(count) == 1


def test_bad_raw_caption_is_preserved_and_never_resampled(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    out = tmp_path / "out"
    original = baseline.create_provider
    calls = []
    def create(*args, **kwargs):
        provider = original(*args, **kwargs)
        generate = provider.generate
        def invalid(request):
            calls.append(1)
            response = generate(request)
            response["text"] = "bad untimed original caption"
            response["raw"] = {"full_original": response["text"]}
            return response
        provider.generate = invalid
        return provider
    monkeypatch.setattr(baseline, "create_provider", create)
    for resume in (False, True):
        with pytest.raises(ValueError, match="fixed-two-window"):
            baseline.run_baseline(source, out, {}, resume=resume)
    assert len(calls) == 1
    raw = read_json(next((out / "calls").glob("*.raw.json")))
    assert raw["result"]["text"] == "bad untimed original caption"


def test_outer_model_receives_no_split_group_path_or_private_answers(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    received = []
    original = baseline.create_provider
    def create(spec, role, base_dir):
        provider = original(spec, role, base_dir)
        if role == "planner":
            generate = provider.generate
            def observe(request):
                received.append(copy.deepcopy(request))
                return generate(request)
            provider.generate = observe
        return provider
    monkeypatch.setattr(baseline, "create_provider", create)
    baseline.run_baseline(source, tmp_path / "out", {"mode": "hierarchical"})
    assert len(received) == 3
    for request in received:
        text = json.dumps(request)
        assert "private-song-group" not in text and "private-artist-group" not in text
        assert "private-duplicate-group" not in text and "song.wav" not in text
        assert "target_text" not in text and "semantic_goals" not in text
        assert set(request["song"]) == {"song_id", "brief", "sample_rate", "num_samples", "music"}
    assert received[0]["response_kind"] == "outline"
    assert received[-1]["previous_section_tail"]


def test_malformed_outer_json_fails_once_and_keeps_exact_raw(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    original = baseline.create_provider
    calls = []
    def create(spec, role, base_dir):
        provider = original(spec, role, base_dir)
        if role == "planner":
            generate = provider.generate
            def bad(request):
                calls.append(1)
                response = generate(request)
                response["text"] = '{"schema":"wrong","schema":"duplicate"}'
                return response
            provider.generate = bad
        return provider
    monkeypatch.setattr(baseline, "create_provider", create)
    out = tmp_path / "out"
    for resume in (False, True):
        with pytest.raises(ValueError, match="Duplicate"):
            baseline.run_baseline(source, out, {"mode": "global"}, resume=resume)
    assert len(calls) == 1


def test_replay_lookup_is_by_stable_call_id_and_binds_file_bytes(tmp_path):
    source, _ = dataset_file(tmp_path, seconds=5.0625)
    replay = tmp_path / "replay.json"
    write_json(replay, dict(schema="choreobench.replay.v1", synthetic=True, source={"fixture": "unit"},
        responses={"song1/local/0000/00": {"text": "[0-2s] Steps left. [2-5.0625s] Turns right."}}))
    out = tmp_path / "out"
    config = {"local_provider": {"kind": "replay", "path": str(replay)}}
    result = baseline.run_baseline(source, out, config)
    assert result["calls_used"] == 1 and result["synthetic"]
    payload = read_json(replay)
    payload["source"]["fixture"] = "changed"
    replay.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        baseline.run_baseline(source, out, config, resume=True)


def test_chat_records_exact_response_and_token_budget_without_secret(monkeypatch):
    captured = []
    monkeypatch.setenv("CHOREOBENCH_TEST_KEY", "fake-sensitive-test-key")
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return b'{"model":"actual-server-model","choices":[{"message":{"content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":2}}'
    def send(request, timeout):
        captured.append((request, timeout))
        return Response()
    monkeypatch.setattr("urllib.request.urlopen", send)
    provider = ChatCompletionsProvider(dict(kind="chat_completions", base_url="https://example.test/v1",
        model="requested-model", api_key_env="CHOREOBENCH_TEST_KEY"))
    result = provider.generate(dict(messages=[dict(role="user", content="public plan")], seed=1, max_output_tokens=16))
    body = json.loads(captured[0][0].data)
    assert body["max_tokens"] == 16 and body["seed"] == 1
    assert result["costs"]["output_tokens"] == 2
    assert "actual-server-model" in result["raw"]["body"]
    assert "fake-sensitive-test-key" not in json.dumps(result)
    assert not result["identity"]["remote_weight_identity_verified"]


def test_native_audits_portable_then_consumes_only_audio(monkeypatch, tmp_path):
    audit_calls, loaded, verification, seen_rows = [], [], [], []
    artifact = {"checkpoint": {"adapter_metadata": {"max_length": 8192}}}
    identity_module = types.ModuleType("native_identity_fixture")
    def audit(*args):
        audit_calls.append(args)
        return artifact
    identity_module.audit_model_artifacts = audit
    def verify(adapter, receipt):
        assert receipt == artifact
        verification.append(1)
        return {"verified": True}
    identity_module.verify_loaded_portable = verify
    search = types.ModuleType("native_search_fixture")
    search.validate_sampling = lambda value: None
    def sample(adapter, row, **kwargs):
        seen_rows.append(row)
        assert verification
        return dict(raw_text="[0-2s] Steps. [2-5.0625s] Turns.", new_token_ids=[1, 2],
                    stop_reason="eos", input_tokens=8)
    search.sample_native = sample
    adapters = types.ModuleType("native_adapters_fixture")
    def load(*args, **kwargs):
        loaded.append((args, kwargs))
        return types.SimpleNamespace()
    adapters.load_planner_adapter = load
    prefix = "scripts_omnidance.exp.ms."
    monkeypatch.setitem(sys.modules, prefix + "music_planner_search_identity", identity_module)
    monkeypatch.setitem(sys.modules, prefix + "music_planner_search", search)
    monkeypatch.setitem(sys.modules, prefix + "music_planner_adapters", adapters)
    provider = NativeProvider(dict(kind="native", model=str(tmp_path / "model"), checkpoint=str(tmp_path / "checkpoint")), "local")
    assert audit_calls and not loaded
    request = dict(call_id="local0", response_kind="local_caption", seed=42, max_output_tokens=10,
        audio={"path": "derived.wav", "sha256": "a" * 64}, song={"private": "not used"},
        messages=[{"role": "user", "content": "forbidden editing instruction"}])
    result = provider.generate(request)
    assert set(seen_rows[0]) == {"id", "audio_path", "audio_sha256", "audio_window_seconds"}
    assert seen_rows[0]["audio_window_seconds"] == [0., 5.0625]
    assert result["raw"]["loaded_artifact_verification"]["verified"]


@pytest.mark.parametrize("text", ['{"x":NaN}', '{"x":1,"x":2}', '```json\n{}\n```'])
def test_model_json_has_no_silent_cleaning(text):
    with pytest.raises(ValueError):
        parse_json_response(text)


def test_split_filter_and_no_synthetic_fallback(tmp_path):
    source, _ = dataset_file(tmp_path)
    with pytest.raises(ValueError, match="No songs"):
        baseline.run_baseline(source, tmp_path / "empty", {"splits": ["train"]})
    with pytest.raises(ValueError, match="no fallback"):
        baseline.run_baseline(source, tmp_path / "bad", {"local_provider": {"kind": "unavailable"}})


def test_failed_song_remains_in_denominator_and_next_song_runs_without_retry(tmp_path, monkeypatch):
    source, song = dataset_file(tmp_path)
    data = read_json(source)
    second = copy.deepcopy(song)
    second["song_id"] = "song2"
    data["songs"].append(second)
    source.write_text(json.dumps(data), encoding="utf-8")
    calls = []
    original = baseline.create_provider
    def create(spec, role, base_dir):
        provider = original(spec, role, base_dir)
        generate = provider.generate
        def observed(request):
            calls.append(request["call_id"])
            result = generate(request)
            if request["call_id"].startswith("song1/"):
                result["text"] = "Original malformed model output"
            return result
        provider.generate = observed
        return provider
    monkeypatch.setattr(baseline, "create_provider", create)
    out = tmp_path / "out"
    config = {"continue_on_model_error": True}
    result = baseline.run_baseline(source, out, config)
    assert result["complete"] and not result["all_songs_succeeded"]
    assert result["selected_count"] == 2 and result["success_count"] == 1
    assert result["failed_songs"][0]["song_id"] == "song1"
    assert result["failed_songs"][0]["retained_in_evaluation_denominator"]
    assert result["plans"][0]["song_id"] == "song2"
    assert len(calls) == 4  # One bad first-song call and all3 second-song windows.
    assert baseline.run_baseline(source, out, config, resume=True) == result
    assert len(calls) == 4


def test_continue_on_model_error_does_not_hide_infrastructure_failure(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    original = baseline.create_provider
    calls = []
    def create(spec, role, base_dir):
        provider = original(spec, role, base_dir)
        def failed(request):
            calls.append(request["call_id"])
            raise OSError("connection outcome unknown")
        provider.generate = failed
        return provider
    monkeypatch.setattr(baseline, "create_provider", create)
    config, out = {"continue_on_model_error": True}, tmp_path / "out"
    for resume in (False, True):
        with pytest.raises(ValueError, match="Recorded provider failure"):
            baseline.run_baseline(source, out, config, resume=resume)
    assert len(calls) == 1
    assert not read_json(out / "results.json")["complete"]


def test_manifest_replacement_between_snapshot_and_audio_preflight_is_rejected(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    original = baseline.inspect_audio
    def replacing(path):
        audio = original(path)
        data = read_json(source)
        data["songs"][0]["brief"] = "replaced while preflight read audio"
        source.write_text(json.dumps(data), encoding="utf-8")
        return audio
    monkeypatch.setattr(baseline, "inspect_audio", replacing)
    with pytest.raises(ValueError, match="immutable JSON snapshot"):
        baseline.run_baseline(source, tmp_path / "out", {})
    assert not (tmp_path / "out").exists()


def test_http_network_error_redacts_secret_and_never_retries(monkeypatch):
    monkeypatch.setenv("CHOREOBENCH_TEST_KEY", "fake-key-must-not-leak")
    count = []
    def fail(request, timeout):
        count.append(1)
        raise urllib.error.URLError("server echoed fake-key-must-not-leak")
    monkeypatch.setattr("urllib.request.urlopen", fail)
    provider = ChatCompletionsProvider(dict(kind="chat_completions", model="model", api_key_env="CHOREOBENCH_TEST_KEY"))
    result = provider.generate(dict(messages=[dict(role="user", content="public")], seed=1, max_output_tokens=16))
    assert result["stop_reason"] == "infrastructure_error" and len(count) == 1
    assert "fake-key-must-not-leak" not in json.dumps(result)


@pytest.mark.parametrize("budget,seed", [(0, 1), (-1, 1), (True, 1), (10, -1), (10, True)])
def test_api_budget_validation_precedes_network_request(monkeypatch, budget, seed):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid budget must not hit endpoint")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    provider = ChatCompletionsProvider(dict(kind="chat_completions", model="model"))
    with pytest.raises(ValueError, match="budget"):
        provider.generate(dict(messages=[dict(role="user", content="public")], seed=seed, max_output_tokens=budget))


def test_non_synthetic_replay_is_exactly_request_bound_and_not_a_model_authentication(tmp_path):
    request = dict(call_id="source/local/0000/00", response_kind="local_caption", seed=42,
                   max_output_tokens=64, audio={"sha256": "1" * 64, "path": "original-reference.wav"})
    path = tmp_path / "recording.json"
    write_json(path, dict(schema="choreobench.replay.v1", synthetic=False,
        source=dict(recording_id="unit-test-recording", recording_sha256="2" * 64,
                    provider_identity={"kind": "declared-unit-fixture-origin"}),
        responses={request["call_id"]: dict(text="Recorded original caption.", raw={"original": "envelope"},
            request_sha256=object_sha(request), stop_reason="stop")}))
    provider = ReplayProvider(dict(kind="replay", path=str(path)), "local")
    result = provider.generate(request)
    assert not result["synthetic"]
    assert not result["identity"]["source_model_identity_independently_verified"]
    changed = copy.deepcopy(request)
    changed["audio"]["sha256"] = "3" * 64
    with pytest.raises(ValueError, match="another request"):
        provider.generate(changed)


@pytest.mark.parametrize("usage", [None, [], "invalid"])
def test_invalid_api_usage_preserves_completed_original_envelope(monkeypatch, usage):
    payload = dict(choices=[dict(message={"content": "{}"}, finish_reason="stop")], usage=usage)
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps(payload).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    provider = ChatCompletionsProvider(dict(kind="chat_completions", model="test", api_key_env=""))
    result = provider.generate(dict(messages=[dict(role="user", content="public")], seed=42, max_output_tokens=64))
    assert result["text"] == "{}" and result["stop_reason"] == "stop"
    assert result["costs"]["input_tokens"] is None and result["costs"]["output_tokens"] is None
    assert result["costs"]["usage_metadata_status"] == "unknown_or_invalid"
    assert json.loads(result["raw"]["body"]) == payload


@pytest.mark.parametrize("mode", ["global", "hierarchical"])
def test_planner_budget_is_separate_from_local_native_window_budget(tmp_path, monkeypatch, mode):
    source, _ = dataset_file(tmp_path)
    calls = []
    original = baseline.create_provider
    def create(spec, role, base_dir):
        provider = original(spec, role, base_dir)
        generate = provider.generate
        def observed(request):
            calls.append((role, request["response_kind"], request["max_output_tokens"]))
            return generate(request)
        provider.generate = observed
        return provider
    monkeypatch.setattr(baseline, "create_provider", create)
    result = baseline.run_baseline(source, tmp_path / "out", dict(
        mode=mode, max_output_tokens=64, planner_max_output_tokens=8192))
    assert result["complete"]
    assert all(budget == 64 for role, _, budget in calls if role == "local")
    assert all(budget == 8192 for role, _, budget in calls if role == "planner")
    assert {kind for role, kind, _ in calls if role == "planner"} == (
        {"plan"} if mode == "global" else {"outline", "section_plan"})


def test_180_second_global_mock_has_large_plan_budget_without_expanding_local_budget(tmp_path):
    source, song = dataset_file(tmp_path, seconds=180)
    out = tmp_path / "global180"
    result = baseline.run_baseline(source, out, dict(mode="global", max_output_tokens=2048,
                                                   planner_max_output_tokens=32768))
    assert result["complete"] and result["synthetic"]
    assert plan_from_result(out, result)["events"][-1]["end"] == song["audio"]["num_samples"]
    contract = read_json(out / "contract.json")
    assert contract["config"]["max_output_tokens"] == 2048
    assert contract["config"]["planner_max_output_tokens"] == 32768
    for started in (out / "calls").glob("*.started.json"):
        request = read_json(started)["request"]
        assert request["max_output_tokens"] == (2048 if request["response_kind"] == "local_caption" else 32768)


def test_planner_budget_defaults_to_local_and_is_positive_integer():
    assert baseline._config({"max_output_tokens": 123})["planner_max_output_tokens"] == 123
    for bad in (0, -1, True, None):
        with pytest.raises(ValueError, match="planner_max_output_tokens"):
            baseline._config({"planner_max_output_tokens": bad})
