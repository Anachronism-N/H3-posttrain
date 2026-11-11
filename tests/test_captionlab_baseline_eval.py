"""CPU-only raw baseline audit; synthetic prose is never model evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from captionlab import baseline_eval as module
from choreobench.baseline import _sealed
from choreobench.io import file_sha, object_sha, read_json, write_json


def wav(path, *, rate=16000, count=81000, channels=1, width=2, value=0):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels); handle.setsampwidth(width); handle.setframerate(rate)
        handle.writeframes(bytes([value]) * count * width * channels)


@pytest.fixture
def source(tmp_path):
    cases = []
    for n in range(2):
        path = tmp_path / f"audio{n}.wav"
        wav(path, value=n)
        cases.append(dict(case_id=f"case{n}", source_group_id=f"source{n}", split="validation",
            audio=dict(path=path.name, sha256=file_sha(path), sample_rate=16000, num_samples=81000), duration_seconds=5.0625))
    manifest = tmp_path / "manifest.json"
    write_json(manifest, dict(schema=module.MANIFEST_SCHEMA, dataset=dict(revision="frozen-source", audit="source identities are declarations"),
                             selection=dict(split="validation", seed=7), cases=cases))
    config = tmp_path / "provider.json"
    write_json(config, dict(kind="native", family="omni_captioner", model="/unloaded/model", task_prompt="Fixed choreography task."))
    return manifest, config, tmp_path / "output"


def fake(behavior=None):
    calls = []
    identity = dict(kind="native", synthetic=True, fixture="not-a-model", nested=dict(version=1))
    def generate(request):
        calls.append(copy.deepcopy(request))
        if behavior == "network" and len(calls) == 1:
            raise OSError("Synthetic unknown outcome")
        if behavior == "drift":
            identity["nested"]["version"] = 2
        if callable(behavior):
            behavior(request)
        return dict(text="A freely phrased step, then a turn.", raw=dict(fixture=True),
            identity=copy.deepcopy(identity) if behavior != "wrong_identity" else dict(kind="different", synthetic=True),
            synthetic=True, stop_reason="budget_truncation" if behavior == "truncated" else "stop",
            costs=dict(input_tokens=10, output_tokens=9, latency_seconds=0.))
    provider = SimpleNamespace(identity=identity, generate=generate)
    return calls, lambda *args, **kwargs: provider


def execute(source, factory=None, **kwargs):
    return module.run_baseline(*source[:2], "omni-instruct", source[2], provider_factory=factory, **kwargs)


def forbidden(*args, **kwargs):
    pytest.fail("Provider was created before preflight/audit finished")


def modify(path, function):
    value = read_json(path); function(value); write_json(path, value, overwrite=True)


def test_raw_generation_audio_whitelist_and_completed_resume_without_provider(source):
    calls, factory = fake()
    result = execute(source, factory)
    assert len(calls) == result["calls_used"] == result["case_count"] == 2
    assert result["complete_count"] == 2 and result["failed_count"] == result["unknown_count"] == 0
    assert result["synthetic"] and not result["quality_verified"] and not result["source_group_independence_verified"]
    assert result["unique_audio_count"] == 2
    for request in calls:
        assert set(request) == {"call_id", "response_kind", "audio", "seed", "max_output_tokens"}
        assert set(request["audio"]) == {"path", "sha256", "sample_rate", "num_samples"}
        assert request["response_kind"] == "local_caption" and request["seed"] == 42
    assert all(row["text"] == "A freely phrased step, then a turn." for row in result["rows"])
    lines = [json.loads(line) for line in (source[2] / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines == result["rows"]
    assert execute(source, forbidden, resume=True) == result
    with pytest.raises(FileExistsError):
        execute(source, forbidden)


@pytest.mark.parametrize("behavior,status", [("network", "unknown"), ("truncated", "failed")])
def test_failed_or_unknown_is_retained_not_retried(source, behavior, status):
    calls, factory = fake(behavior)
    result = execute(source, factory)
    assert len(calls) == 2 and result["rows"][0]["status"] == status
    assert result["case_count"] == 2
    if behavior == "network":
        assert result["unknown_outcomes_count"] == 1
        assert result["rows"][1]["status"] == "complete"
    else:
        assert result["rows"][0]["text"] and result["rows"][0]["stop_reason"] == "budget_truncation"
    assert execute(source, forbidden, resume=True) == result


@pytest.mark.parametrize("behavior", ["drift", "wrong_identity"])
def test_identity_is_deepcopied_and_stops_subsequent_dispatch(source, behavior):
    calls, factory = fake(behavior)
    with pytest.raises(ValueError, match="identity"):
        execute(source, factory)
    assert len(calls) == 1 and not (source[2] / "results.json").exists()
    frozen = read_json(source[2] / "contract.json")["provider_identity"]
    assert frozen["nested"]["version"] == 1


def test_changed_manifest_after_dispatch_stops_next_case(source):
    def change(_):
        source[0].write_bytes(source[0].read_bytes() + b" ")
    calls, factory = fake(change)
    with pytest.raises(ValueError, match="Input bytes changed"):
        execute(source, factory)
    assert len(calls) == 1


@pytest.mark.parametrize("bad", ["pending", "extra_case_field", "split", "duplicate", "duration_bool", "clock_bool", "sha", "empty"])
def test_bad_manifests_fail_before_provider_or_output(source, bad):
    def change(doc):
        case = doc["cases"][1]
        if bad == "pending": case["audio"] = None
        elif bad == "extra_case_field": case["reference"] = "A hidden GPT answer."
        elif bad == "split": case["split"] = "test"
        elif bad == "duplicate": case["case_id"] = doc["cases"][0]["case_id"]
        elif bad == "duration_bool": case["duration_seconds"] = True
        elif bad == "clock_bool": case["audio"]["sample_rate"] = True
        elif bad == "sha": case["audio"]["sha256"] = "0" * 64
        else: doc["cases"] = []
    modify(source[0], change)
    with pytest.raises(ValueError):
        execute(source, forbidden)
    assert not source[2].exists()


@pytest.mark.parametrize("channels,width,count", [(2, 2, 81000), (1, 1, 81000), (1, 2, 80000)])
def test_all_waveforms_preflight_before_any_call(source, channels, width, count):
    path = source[0].parent / "audio1.wav"
    wav(path, channels=channels, width=width, count=count)
    def change(doc):
        doc["cases"][1]["audio"].update(sha256=file_sha(path), num_samples=count)
    modify(source[0], change)
    with pytest.raises(ValueError):
        execute(source, forbidden)
    assert not source[2].exists()


@pytest.mark.parametrize("kind", ["mock", "chat_completions", "unknown"])
def test_text_only_or_mock_providers_rejected(source, kind):
    write_json(source[1], dict(kind=kind), overwrite=True)
    with pytest.raises(ValueError, match="no mock/text fallback"):
        execute(source, forbidden)


@pytest.mark.parametrize("artifact", ["raw_missing", "case_missing", "case_changed", "records_changed", "results_changed", "calls_missing", "extra_call_file"])
def test_completed_tampering_rejected_without_provider(source, artifact):
    _, factory = fake(); execute(source, factory)
    out = source[2]
    if artifact == "raw_missing": next((out / "calls").glob("*.raw.json")).unlink()
    elif artifact == "case_missing": next((out / "cases").glob("*.json")).unlink()
    elif artifact == "case_changed":
        path = next((out / "cases").glob("*.json"))
        modify(path, lambda doc: doc.update(text="Injected text."))
    elif artifact == "records_changed": (out / "records.jsonl").write_text("{}\n", encoding="utf-8")
    elif artifact == "results_changed": modify(out / "results.json", lambda doc: doc.update(complete_count=99))
    elif artifact == "calls_missing":
        for path in (out / "calls").iterdir(): path.unlink()
        (out / "calls").rmdir()
    else: write_json(out / "calls" / "unbound.json", {})
    with pytest.raises(ValueError):
        execute(source, forbidden, resume=True)


def test_resealed_request_with_hidden_label_rejected_before_provider(source):
    _, factory = fake(); execute(source, factory)
    start_path = next((source[2] / "calls").glob("*.started.json"))
    start = read_json(start_path)
    start["request"]["reference"] = "Hidden label"
    start["request_sha256"] = object_sha(start["request"])
    start.pop("record_sha256"); start = _sealed(start)
    write_json(start_path, start, overwrite=True)
    raw_path = start_path.with_name(start_path.name.replace(".started.json", ".raw.json"))
    raw = read_json(raw_path); raw["started_sha256"] = start["record_sha256"]
    raw.pop("record_sha256"); write_json(raw_path, _sealed(raw), overwrite=True)
    with pytest.raises(ValueError, match="Historical request"):
        execute(source, forbidden, resume=True)


@pytest.mark.parametrize("input_index", [0, 1])
def test_changed_source_bytes_reject_complete_resume_before_provider(source, input_index):
    _, factory = fake(); execute(source, factory)
    source[input_index].write_bytes(source[input_index].read_bytes() + b" ")
    with pytest.raises(ValueError, match="identity changed"):
        execute(source, forbidden, resume=True)


def test_pending_call_without_raw_never_reissued_even_in_incomplete_run(source):
    _, factory = fake(); execute(source, factory)
    (source[2] / "results.json").unlink(); (source[2] / "records.jsonl").unlink()
    next((source[2] / "calls").glob("*.raw.json")).unlink()
    with pytest.raises(ValueError, match="automatic retry is forbidden"):
        execute(source, forbidden, resume=True)


def test_replay_and_cli_need_no_torch_or_native_model(source, capsys):
    replay = source[0].parent / "replay.json"
    write_json(replay, dict(schema="choreobench.replay.v1", synthetic=True,
        source=dict(note="CPU fixture"), responses={f"omni-instruct/case{n}": dict(text="A synthetic free-form movement paragraph.") for n in range(2)}))
    write_json(source[1], dict(kind="replay", path=replay.name), overwrite=True)
    module.main(["--manifest", str(source[0]), "--provider-config", str(source[1]), "--method-id", "omni-instruct",
                 "--out-dir", str(source[2]), "--seed", "42", "--max-output-tokens", "1024"])
    assert json.loads(capsys.readouterr().out)["complete_count"] == 2
    result = read_json(source[2] / "results.json")
    assert result["scientific_result_status"] == "synthetic_development_only"
    assert result["unknown_usage_count"] == 2


def test_real_replay_source_changed_rejected_on_complete_resume(source):
    replay = source[0].parent / "replay.json"
    write_json(replay, dict(schema="choreobench.replay.v1", synthetic=True, source={},
        responses={f"omni-instruct/case{n}": dict(text="Synthetic action paragraph.") for n in range(2)}))
    write_json(source[1], dict(kind="replay", path=replay.name), overwrite=True)
    execute(source)
    replay.write_bytes(replay.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Replay source bytes changed"):
        execute(source, forbidden, resume=True)


def test_bound_partial_run_uses_existing_calls_without_resampling(source):
    calls, factory = fake(); result = execute(source, factory)
    # A process can die before final publication. Retain only the first fully
    # sealed case, with no pending call or published whole-run result.
    (source[2] / "results.json").unlink(); (source[2] / "records.jsonl").unlink()
    name = module._Journal._name("omni-instruct/case1")
    for path in (source[2] / "calls").glob(name + ".*"): path.unlink()
    (source[2] / "cases" / (name + ".json")).unlink()
    calls.clear()
    restored = execute(source, factory, resume=True)
    assert [request["call_id"] for request in calls] == ["omni-instruct/case1"]
    assert restored["case_count"] == 2 and restored["complete_count"] == result["complete_count"]


def test_same_provider_output_empty_text_is_a_failed_case(source):
    def factory(*args, **kwargs):
        identity = dict(kind="native", synthetic=True)
        return SimpleNamespace(identity=identity, generate=lambda request: dict(
            text="", identity=identity, synthetic=True, stop_reason="stop", raw={}, costs={}))
    result = execute(source, factory)
    assert result["failed_count"] == result["case_count"] == 2
    assert execute(source, forbidden, resume=True) == result
