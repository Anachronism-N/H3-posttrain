"""Independent synthetic harness checks; no real model or benchmark claims."""
import copy
import json

import pytest

from choreobench import baseline, candidates, harness
from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.planning_units import build_planning_units
from choreobench.selection import candidate_options
from test_choreobench_baseline import dataset_file


def source_bank(tmp_path, *, stride=5.0625):
    source, song = dataset_file(tmp_path)
    baseline.run_baseline(source, tmp_path / "collection", {
        "candidates_per_window": 2, "stride_seconds": stride})
    path = tmp_path / "candidates.json"
    candidates.export_candidates(tmp_path / "collection", path)
    return source, song, path


class SyntheticRecordingSelector:
    """A monkeypatched provider with replay identity; never a real model."""
    identity = {"kind": "replay", "role": "planner", "synthetic": True,
                "source": "test-only fabricated replay fixture"}

    def __init__(self, mutate=None, interrupt=False):
        self.requests = []
        self.mutate, self.interrupt = mutate, interrupt

    def generate(self, request):
        self.requests.append(copy.deepcopy(request))
        if self.interrupt:
            raise KeyboardInterrupt("Synthetic unknown-outcome fixture")
        payload = json.loads(request["messages"][-1]["content"])
        choices = [{"window_id": wid, "candidate_index": 1} for wid in payload["authoritative_window_ids"]]
        answer = {"choices": choices}
        if self.mutate:
            self.mutate(answer)
        return {"identity": copy.deepcopy(self.identity), "synthetic": True, "text": json.dumps(answer),
                "raw": {"synthetic": True}, "stop_reason": "stop",
                "costs": {"input_tokens": None, "output_tokens": None, "latency_seconds": 0.}}


def llm_config():
    return {"mode": "llm", "selector_provider": {"kind": "replay", "path": "unused-test-fixture.json"},
            "unit_config": {"fallback_seconds": 2}}


def test_index0_uses_no_provider_and_keeps_cross_block_decisions_unique(tmp_path, monkeypatch):
    source, song, bank = source_bank(tmp_path)
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: pytest.fail("index0 must not load a model"))
    result = harness.run_harness(source, bank, tmp_path / "harness", {"unit_config": {"fallback_seconds": 1}})
    assert result["complete"] and result["synthetic"] and result["success_count"] == 1
    assert result["calls_used"] == result["new_local_model_calls"] == 0
    assert not result["music_or_physical_quality_verified"]
    refs = [r for r in result["artifacts"] if r["path"].startswith("selections/")]
    selection = read_json(tmp_path / "harness" / refs[0]["path"])
    assert len(selection["choices"]) == 3
    assert len({c["window_id"] for c in selection["choices"]}) == 3
    assert all(c["candidate_index"] == 0 for c in selection["choices"])


def test_llm_model_facing_request_whitelist_and_one_authority_per_window(tmp_path, monkeypatch):
    source, song, bank = source_bank(tmp_path)
    selector = SyntheticRecordingSelector()
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: selector)
    result = harness.run_harness(source, bank, tmp_path / "harness", llm_config())
    assert result["complete"] and result["all_songs_succeeded"]
    owned = []
    for request in selector.requests:
        assert set(request) == {"call_id", "seed", "max_output_tokens", "response_kind", "messages"}
        assert request["response_kind"] == "window_selection"
        raw = json.dumps(request)
        for hidden in ("private-song-group", "private-artist-group", "private-duplicate-group",
                       "song.wav", "dataset_snapshot", "private_goals", "semantic_goals", "source_plan"):
            assert hidden not in raw
        payload = json.loads(request["messages"][-1]["content"])
        assert "binding" not in payload and "split" not in payload and "audio" not in payload
        owned.extend(payload["authoritative_window_ids"])
        assert all(ref["window_id"] in {w["window_id"] for w in payload["windows"]}
                   for ref in payload["response_template"]["choices"])
    assert owned == ["w0000", "w0001", "w0002"]
    assert len(selector.requests) == result["calls_used"] == result["expected_calls"]


def test_completed_resume_has_no_new_calls_and_rejects_changed_artifact(tmp_path, monkeypatch):
    source, _, bank = source_bank(tmp_path)
    selector = SyntheticRecordingSelector()
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: selector)
    output, cfg = tmp_path / "harness", llm_config()
    result = harness.run_harness(source, bank, output, cfg)
    count = len(selector.requests)
    assert harness.run_harness(source, bank, output, cfg, resume=True) == result
    assert len(selector.requests) == count
    path = output / result["plans"][0]["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact changed"):
        harness.run_harness(source, bank, output, cfg, resume=True)
    assert len(selector.requests) == count


@pytest.mark.parametrize("mutate", [
    lambda answer: answer.update(action="forbidden rewrite"),
    lambda answer: answer["choices"][0].update(action="forbidden rewrite"),
    lambda answer: answer["choices"][0].update(candidate_index=True),
    lambda answer: answer["choices"][0].update(window_id="another-window"),
    lambda answer: answer["choices"].clear(),
])
def test_invalid_model_scope_is_retained_as_failure_without_replacement(tmp_path, monkeypatch, mutate):
    source, _, bank = source_bank(tmp_path)
    selector = SyntheticRecordingSelector(mutate=mutate)
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: selector)
    output, cfg = tmp_path / "harness", llm_config()
    result = harness.run_harness(source, bank, output, cfg)
    assert result["complete"] and not result["all_songs_succeeded"]
    assert result["success_count"] == 0 and result["plans"] == []
    assert result["failed_songs"][0]["retained_in_evaluation_denominator"]
    assert len(selector.requests) == result["calls_used"] == 1
    assert len(list((output / "calls").glob("*.raw.json"))) == 1
    assert harness.run_harness(source, bank, output, cfg, resume=True) == result
    assert len(selector.requests) == 1


def test_unknown_outcome_resume_never_resends_the_same_billable_request(tmp_path, monkeypatch):
    source, _, bank = source_bank(tmp_path)
    selector = SyntheticRecordingSelector(interrupt=True)
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: selector)
    output, cfg = tmp_path / "harness", llm_config()
    with pytest.raises(KeyboardInterrupt):
        harness.run_harness(source, bank, output, cfg)
    assert len(list((output / "calls").glob("*.started.json"))) == 1
    assert not list((output / "calls").glob("*.raw.json"))
    selector.interrupt = False
    with pytest.raises(ValueError, match="automatic retry is forbidden"):
        harness.run_harness(source, bank, output, cfg, resume=True)
    assert len(selector.requests) == 1


def test_dataset_source_change_fails_before_any_outer_generation(tmp_path, monkeypatch):
    source, _, bank = source_bank(tmp_path)
    dataset = read_json(source)
    dataset["songs"][0]["brief"] += " changed source identity"
    write_json(source, dataset, overwrite=True)
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: pytest.fail("changed source must fail first"))
    with pytest.raises(ValueError, match="identity differs"):
        harness.run_harness(source, bank, tmp_path / "harness", llm_config())


def test_dense_stride_uses_source_windows_and_only_committed_prior_fragments(tmp_path, monkeypatch):
    source, _, bank = source_bank(tmp_path, stride=2)
    selector = SyntheticRecordingSelector()
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: selector)
    output = tmp_path / "harness"
    result = harness.run_harness(source, bank, output, llm_config())
    assert result["complete"] and result["success_count"] == 1
    options_ref = next(ref for ref in result["artifacts"] if ref["path"].startswith("options/"))
    options = read_json(output / options_ref["path"])
    windows = {row["window_id"]: row["window"] for row in options["windows"]}
    assert [w["observation_start"] for w in windows.values()] == [0, 32000, 64000, 96000, 111000]
    assert sum(len(json.loads(r["messages"][-1]["content"])["authoritative_window_ids"])
               for r in selector.requests) == len(windows)
    contexts = [prior for request in selector.requests
                for prior in json.loads(request["messages"][-1]["content"])["prior_selected_text"]]
    assert contexts and any(prior["omitted_fragment_count"] for prior in contexts)
    for prior in contexts:
        assert "raw_text" not in prior and prior["timing_policy"] == "strict"
        window = windows[prior["window_id"]]
        for fragment in prior["scheduled_fragments"]:
            assert window["ownership_start"] <= fragment["scheduled_interval"][0]
            assert fragment["scheduled_interval"][1] <= window["ownership_end"]
            assert fragment["source_interval"] == fragment["scheduled_interval"]
            assert not fragment["partial_source"] and not fragment["semantic_timing_verified"]


def test_legal_all_null_scores_are_explicit_song_abstention_with_exact_input_snapshot(tmp_path, monkeypatch):
    source, song, bank_path = source_bank(tmp_path)
    bank = candidates.load_candidates(bank_path)
    options = candidate_options(bank, song, build_planning_units(song))
    n = len(options["windows"])
    scores = {"schema": "choreobench.selection_scores.v1", "dataset_sha256": file_sha(source),
        "bank_sha256": file_sha(bank_path), "songs": {song["song_id"]: {
            "options_sha256": object_sha(options), "unary": [[None, None] for _ in range(n)],
            "transitions": [[[0, 0], [0, 0]] for _ in range(n - 1)],
            "score_provenance": {"source": "synthetic missing-score fixture", "synthetic": True}}}}
    score_path = tmp_path / "scores.json"
    write_json(score_path, scores)
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: pytest.fail("scores mode must not load a model"))
    output = tmp_path / "harness"
    config = {"mode": "scores", "scores_path": str(score_path)}
    result = harness.run_harness(source, bank_path, output, config)
    assert result["complete"] and not result["all_songs_succeeded"]
    assert result["selected_count"] == 1 and result["success_count"] == 0
    assert result["calls_used"] == 0 and result["plans"] == []
    assert result["failed_songs"][0]["retained_in_evaluation_denominator"]
    assert read_json(output / "contract.json")["scores_snapshot"] == scores
    assert harness.run_harness(source, bank_path, output, config, resume=True) == result
