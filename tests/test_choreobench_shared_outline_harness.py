"""Shared intermediate content is actually consumed, with no extra outline call."""
import copy
import json
from types import SimpleNamespace

import pytest

from choreobench import coordination, harness
from choreobench.io import read_json, object_sha
from test_choreobench_harness import fixture
from test_choreobench_ablations import ablation_planner, cfg


def source_outline(tmp_path, monkeypatch):
    from choreobench.outlines import export_shared_outlines
    source, _, bank = fixture(tmp_path)
    requests = []
    ablation_planner(monkeypatch, requests)
    producer = tmp_path / "producer"
    result = harness.run_harness(source, bank, producer, cfg())
    assert result["all_songs_succeeded"]
    path = tmp_path / "shared.json"
    export_shared_outlines(producer, path)
    return source, bank, path


def test_two_runs_consume_identical_outline_and_skip_generation(tmp_path, monkeypatch):
    source, bank, shared_path = source_outline(tmp_path, monkeypatch)
    outlines, unit_payloads = [], []
    for name, history in (("main", True), ("no_history", False)):
        requests = []
        ablation_planner(monkeypatch, requests)
        config = cfg(use_history=history)
        config.update(shared_outline_path="shared.json", max_calls=2)
        out = tmp_path / name
        result = harness.run_harness(source, bank, out, config)
        assert result["all_songs_succeeded"] and result["calls_used"] == result["expected_calls"] == 2
        assert all(r["response_kind"] == "coordination_unit" for r in requests)
        assert result["shared_outline_source_costs"]["recorded_calls"] == 1
        assert result["new_local_model_calls"] == 0 and result["synthetic"]
        metadata = next(read_json(out / r["path"]) for r in result["artifacts"] if r["path"].startswith("outline_metadata/"))
        assert metadata["source"] == "shared_import" and metadata["generated_by_model"]
        assert not metadata["generated_in_this_run"]
        outline = next(read_json(out / r["path"]) for r in result["artifacts"] if r["path"].startswith("outlines/"))
        assert object_sha(outline) == metadata["outline_sha256"]
        plan = read_json(out / result["plans"][0]["path"])
        assert plan["provenance"]["outline_sha256"] == object_sha(outline)
        assert not plan["provenance"]["outline_generated_in_this_run"]
        outlines.append(outline)
        unit_payloads.append([json.loads(r["messages"][-1]["content"]) for r in requests])
        assert harness.run_harness(source, bank, out, config, resume=True) == result
        assert len(requests) == 2
    assert outlines[0] == outlines[1]
    for first, second in zip(*unit_payloads):
        for key in ("whole_song_design", "current_section_design", "current_motif_definitions", "local_proposals", "event_clock"):
            assert first[key] == second[key]
        assert "boundary_context" in first and "boundary_context" not in second


def test_invalid_shared_input_rejected_before_provider_initialization(tmp_path, monkeypatch):
    source, bank, shared_path = source_outline(tmp_path, monkeypatch)
    shared_path.write_text("{}", encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid shared outline reached provider initialization")
    monkeypatch.setattr(coordination, "create_provider", forbidden)
    with pytest.raises(ValueError):
        harness.run_harness(source, bank, tmp_path / "out", cfg(shared_outline_path=str(shared_path)))
    assert not (tmp_path / "out").exists()


def test_shared_file_mutation_stops_before_next_call_and_preserves_resume(tmp_path, monkeypatch):
    source, bank, shared_path = source_outline(tmp_path, monkeypatch)
    original_bytes = shared_path.read_bytes()
    requests = []
    ablation_planner(monkeypatch, requests)
    provider = coordination.create_provider()
    def generate(request):
        result = provider.generate(request)
        if len(requests) == 1:
            shared_path.write_bytes(original_bytes + b"\n")
        return result
    monkeypatch.setattr(coordination, "create_provider", lambda *a, **k:
        SimpleNamespace(identity=provider.identity, generate=generate))
    config = cfg(shared_outline_path=str(shared_path))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="source bytes changed"):
        harness.run_harness(source, bank, out, config)
    assert len(requests) == 1
    interrupted = read_json(out / "results.json")
    assert not interrupted["complete"] and interrupted["calls_used"] == 1
    shared_path.write_bytes(original_bytes)
    recovered = harness.run_harness(source, bank, out, config, resume=True)
    assert recovered["all_songs_succeeded"] and recovered["calls_used"] == 2
    assert len(requests) == 2  # First completed call is reused.
    shared_path.write_bytes(original_bytes + b"\n")
    with pytest.raises(ValueError):
        harness.run_harness(source, bank, out, config, resume=True)
    assert len(requests) == 2


@pytest.mark.parametrize("value", [True, 1, [], {}, "", "  "])
def test_shared_path_is_strict(value):
    with pytest.raises(ValueError, match="shared_outline_path"):
        harness._config(cfg(shared_outline_path=value))


def test_shared_outline_is_not_no_outline_or_selector_mode():
    with pytest.raises(ValueError, match="use_outline"):
        harness._config(cfg(shared_outline_path="asset.json", use_outline=False))
    with pytest.raises(ValueError, match="only consumed"):
        harness._config(dict(mode="index0", shared_outline_path=None))
    assert harness._config(harness._config(dict(mode="index0")))["mode"] == "index0"
