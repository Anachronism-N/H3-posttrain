import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from choreobench import harness
from choreobench.audio import make_windows
from choreobench.baseline import run_baseline
from choreobench.candidates import export_candidates, load_candidates
from choreobench.cli import main
from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.planning_units import build_planning_units
from choreobench.selection import candidate_options
from test_choreobench_baseline import dataset_file


def fixture(tmp_path, **changes):
    source, song = dataset_file(tmp_path)
    config = dict(candidates_per_window=2, **changes)
    run_baseline(source, tmp_path / "collection", config)
    bank_path = tmp_path / "bank.json"
    export_candidates(tmp_path / "collection", bank_path)
    return source, song, bank_path


def fake_selector(monkeypatch, requests, *, malformed=False, failure=False):
    identity = dict(kind="test_text_provider", synthetic=True, version="frozen")
    def generate(request):
        requests.append(copy.deepcopy(request))
        if failure:
            raise OSError("unknown service outcome")
        payload = json.loads(request["messages"][1]["content"])
        choices = [{"window_id": wid, "candidate_index": 1} for wid in payload["authoritative_window_ids"]]
        if malformed:
            choices[0]["window_id"] = "outside-scope"
        return dict(text=json.dumps({"choices": choices}), raw={"fixture": True}, identity=identity,
                    synthetic=True, stop_reason="stop", costs=dict(input_tokens=100, output_tokens=50))
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: SimpleNamespace(identity=identity, generate=generate))


def test_index0_strict_has_no_calls_and_preserves_real_gaps_and_resume(tmp_path):
    source, song, bank = fixture(tmp_path)
    out = tmp_path / "harness"
    result = harness.run_harness(source, bank, out, {"max_calls": 0})
    assert result["all_songs_succeeded"] and result["calls_used"] == 0
    assert result["new_local_model_calls"] == 0 and not result["model_weights_updated"]
    assert result["original_collection_calls"] == 6
    reports = [read_json(out / r["path"]) for r in result["artifacts"] if r["path"].startswith("compilation/")]
    assert reports[0]["omitted_intervals"] == [{"start": 81000, "end": 111000}]
    assert reports[0]["coverage_fraction"] < 1
    assert harness.run_harness(source, bank, out, {"max_calls": 0}, resume=True) == result
    changed = out / next(r["path"] for r in result["artifacts"] if r["path"].startswith("compilation/"))
    changed.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        harness.run_harness(source, bank, out, {"max_calls": 0}, resume=True)


def test_llm_chooses_each_whole_window_once_and_never_sees_private_groups(tmp_path, monkeypatch):
    source, song, bank = fixture(tmp_path)
    requests = []
    fake_selector(monkeypatch, requests)
    config = dict(mode="llm", selector_provider={"kind": "test-only"}, max_calls=2)
    out = tmp_path / "harness"
    result = harness.run_harness(source, bank, out, config)
    assert result["calls_used"] == 2 and result["all_songs_succeeded"]
    authoritative = []
    for request in requests:
        text = request["messages"][1]["content"]
        assert "private-song-group" not in text and "private-artist-group" not in text
        payload = json.loads(text)
        authoritative.extend(payload["authoritative_window_ids"])
        assert not {"goals", "semantic_goals", "source_plan"} & set(payload)
    assert authoritative == ["w0000", "w0001", "w0002"]
    selection = read_json(out / next(r["path"] for r in result["artifacts"] if r["path"].startswith("selections/")))
    assert [c["candidate_index"] for c in selection["choices"]] == [1, 1, 1]
    assert selection["synthetic"]
    assert harness.run_harness(source, bank, out, config, resume=True) == result
    assert len(requests) == 2


def test_prior_context_contains_retained_schedule_not_dropped_caption(tmp_path, monkeypatch):
    source, song, bank = fixture(tmp_path, stride_seconds=1)
    requests = []
    fake_selector(monkeypatch, requests)
    harness.run_harness(source, bank, tmp_path / "harness", dict(
        mode="llm", selector_provider={"kind": "test-only"}, max_calls=2))
    later = json.loads(requests[1]["messages"][1]["content"])
    assert later["prior_selected_text"]
    for row in later["prior_selected_text"]:
        assert "raw_text" not in row
        assert row["scheduled_fragments"] == []
        assert row["omitted_fragment_count"] == 2


def test_budget_rejected_before_provider_initialization(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    monkeypatch.setattr(harness, "create_provider", lambda *a, **k: pytest.fail("no provider before budget preflight"))
    with pytest.raises(ValueError, match="max_calls"):
        harness.run_harness(source, bank, tmp_path / "harness", dict(
            mode="llm", selector_provider={"kind": "test-only"}, max_calls=0))
    assert not (tmp_path / "harness").exists()


def test_bad_selector_scope_fails_song_without_replacement(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    requests = []
    fake_selector(monkeypatch, requests, malformed=True)
    result = harness.run_harness(source, bank, tmp_path / "harness", dict(
        mode="llm", selector_provider={"kind": "test-only"}, max_calls=2))
    assert result["complete"] and result["success_count"] == 0
    assert result["failed_songs"][0]["retained_in_evaluation_denominator"]
    assert result["calls_used"] == len(requests) == 1
    assert result["costs"][0]["output_tokens"] == 50


def test_unknown_outcome_is_not_retried_on_resume(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    requests = []
    fake_selector(monkeypatch, requests, failure=True)
    config = dict(mode="llm", selector_provider={"kind": "test-only"}, max_calls=2)
    out = tmp_path / "harness"
    for resume in (False, True):
        with pytest.raises(ValueError, match="provider failure"):
            harness.run_harness(source, bank, out, config, resume=resume)
    assert len(requests) == 1
    result = read_json(out / "results.json")
    assert not result["complete"] and result["unknown_outcomes_count"] == 1


def test_scores_mode_binds_raw_scores_and_runs_without_model(tmp_path):
    source, song, bank_path = fixture(tmp_path)
    bank = load_candidates(bank_path)
    options = candidate_options(bank, song, build_planning_units(song))
    n = len(options["windows"])
    scores = {"schema": "choreobench.selection_scores.v1", "dataset_sha256": file_sha(source),
              "bank_sha256": file_sha(bank_path), "songs": {song["song_id"]: {
                  "options_sha256": object_sha(options), "unary": [[0, 1] for _ in range(n)],
                  "transitions": [[[0, 0], [0, 0]] for _ in range(n-1)],
                  "score_provenance": {"source": "synthetic-score-fixture", "synthetic": True}}}}
    write_json(tmp_path / "scores.json", scores)
    result = harness.run_harness(source, bank_path, tmp_path / "harness", dict(
        mode="scores", scores_path="scores.json", max_calls=0))
    assert result["all_songs_succeeded"] and result["calls_used"] == 0
    contract = read_json(tmp_path / "harness" / "contract.json")
    assert contract["scores_snapshot"] == scores
    assert contract["scores_sha256"] == file_sha(tmp_path / "scores.json")


def test_custom_stride_units_match_bank_and_cli_plan_units(tmp_path, capsys):
    source, song, bank = fixture(tmp_path, stride_seconds=4)
    config = tmp_path / "harness.json"
    write_json(config, {"max_calls": 0})
    assert main(["run-harness", "--dataset", str(source), "--candidates", str(bank),
                 "--config", str(config), "--out-dir", str(tmp_path / "harness")]) == 0
    result = read_json(tmp_path / "harness" / "results.json")
    units = read_json(tmp_path / "harness" / next(r["path"] for r in result["artifacts"] if r["path"].startswith("units/")))
    assert units["windows_sha256"] == object_sha(make_windows(192000, 16000, stride_seconds=4))
    assert main(["plan-units", "--dataset", str(source), "--song-id", song["song_id"],
                 "--stride-seconds", "4", "--out", str(tmp_path / "units.json")]) == 0
    assert read_json(tmp_path / "units.json") == units


def test_request_character_guard_fails_explicitly_without_sending_or_truncating(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    requests = []
    fake_selector(monkeypatch, requests)
    result = harness.run_harness(source, bank, tmp_path / "harness", dict(
        mode="llm", selector_provider={"kind": "test-only"}, max_calls=2, max_request_characters=10))
    assert not requests and result["calls_used"] == 0
    assert result["complete"] and result["success_count"] == 0
    assert "max_request_characters" in result["failed_songs"][0]["message"]


def test_resident_collection_bank_and_cpu_harness_share_one_model_instance(tmp_path, monkeypatch):
    import threading
    from choreobench import providers, resident
    from test_choreobench_resident import FakeNativeProvider
    class CaptionProvider(FakeNativeProvider):
        def generate(self, request):
            response = super().generate(request)
            response["text"] = "[0-2s] Steps left. [2-5.0625s] Raises both arms."
            return response
    monkeypatch.setattr(providers, "NativeProvider", CaptionProvider)
    source, _ = dataset_file(tmp_path)
    server = resident.create_resident_server({"kind": "native", "model": "fixture"}, tmp_path)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    spec = dict(kind="native_service", base_url="http://127.0.0.1:" + str(server.server_address[1]),
                expected_identity_sha256=server.identity_sha256, timeout_seconds=3, api_key_env="")
    try:
        assert server.provider.adapter is None
        result = run_baseline(source, tmp_path / "collection", dict(local_provider=spec, candidates_per_window=2))
        assert result["all_songs_succeeded"] and result["calls_used"] == 6
        assert server.provider.loads == 1 and len(server.provider.calls) == 6
        bank = tmp_path / "resident-bank.json"
        export_candidates(tmp_path / "collection", bank)
        selected = harness.run_harness(source, bank, tmp_path / "harness", {"max_calls": 0})
        assert selected["all_songs_succeeded"] and selected["new_local_model_calls"] == 0
        assert server.provider.loads == 1 and len(server.provider.calls) == 6
        assert run_baseline(source, tmp_path / "collection", dict(local_provider=spec, candidates_per_window=2), resume=True) == result
        assert len(server.provider.calls) == 6
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
