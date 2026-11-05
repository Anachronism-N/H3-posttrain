"""Controlled inference ablations use synthetic providers only."""
import copy
import json
from types import SimpleNamespace

import pytest

from choreobench import coordination, harness
from choreobench.baseline import ModelOutputError
from choreobench.io import read_json
from test_choreobench_harness import fixture


def ablation_planner(monkeypatch, requests):
    identity = dict(kind="ablation_fixture", synthetic=True, version=1)
    def generate(request):
        requests.append(copy.deepcopy(request))
        payload = json.loads(request["messages"][-1]["content"])
        if request["response_kind"] == "coordination_outline":
            answer = dict(motifs=[dict(motif_id="m1", description="Travel then open arms")],
                sections=[dict(section_id=s["section_id"], intent="Introduce and vary travel", motif_ids=["m1"])
                          for s in payload["song"]["music"]["sections"]])
        else:
            unit, clock = payload["planning_unit"], payload["event_clock"]
            event = dict(track="body", action="COMMITTED_ACTION_CANARY_" + unit["unit_id"],
                         motif_id="m1" if payload["outline_available"] else None,
                         source_call_ids=[payload["local_proposals"][0]["call_id"]])
            if clock["format"] == "boundary_ids":
                event.update(start_boundary=0, end_boundary=len(clock["boundary_samples"]) - 1)
            else:
                event.update(start=unit["start"], end=unit["end"])
            answer = dict(unit_id=unit["unit_id"], events=[event])
        return dict(text=json.dumps(answer), raw={"synthetic_fixture": True}, identity=identity, synthetic=True,
                    stop_reason="stop", costs=dict(input_tokens=100, output_tokens=50))
    monkeypatch.setattr(coordination, "create_provider", lambda *a, **k: SimpleNamespace(identity=identity, generate=generate))


def cfg(**changes):
    return dict(mode="coordinated", planner_provider={"kind": "fixture"}, max_calls=10,
                event_time_format="boundary_ids", event_schedule_policy="balanced_full_unit",
                structured_unit_output=True, min_event_seconds=.5, **changes)


@pytest.mark.parametrize("outline,history", [(True, True), (False, True), (True, False), (False, False)])
def test_ablation_inputs_calls_and_resume_are_explicit(tmp_path, monkeypatch, outline, history):
    source, _, bank = fixture(tmp_path)
    requests = []
    ablation_planner(monkeypatch, requests)
    config = cfg(use_outline=outline, use_history=history)
    result = harness.run_harness(source, bank, tmp_path / "out", config)
    assert result["all_songs_succeeded"] and result["calls_used"] == 2 + int(outline)
    assert result["expected_calls"] == result["calls_used"] and result["new_local_model_calls"] == 0
    assert result["ablation"] == dict(use_outline=outline, use_history=history)
    units = [r for r in requests if r["response_kind"] == "coordination_unit"]
    first, second = [json.loads(r["messages"][-1]["content"]) for r in units]
    for payload in (first, second):
        assert payload["outline_available"] is outline and payload["history_available"] is history
        assert ("whole_song_design" in payload) is outline
        assert ("current_section_design" in payload) is outline
        assert ("current_motif_definitions" in payload) is outline
        assert ("boundary_context" in payload) is history
        assert ("recent_committed_events" in payload) is history
    assert ("COMMITTED_ACTION_CANARY_pu0000" in units[1]["messages"][-1]["content"]) is history
    if not history:
        assert "source_prefix_sha256" not in units[1]["messages"][-1]["content"]
        assert "prior_event_count" not in units[1]["messages"][-1]["content"]
    plan = read_json(tmp_path / "out" / result["plans"][0]["path"])
    if not outline:
        assert plan["motifs"] == [] and all(e["motif_id"] is None for e in plan["events"])
        assert first["output_contract"]["allowed_motif_ids"] == []
    metadata = next(read_json(tmp_path / "out" / r["path"]) for r in result["artifacts"] if r["path"].startswith("outline_metadata/"))
    assert metadata["generated_by_model"] is outline
    report = next(read_json(tmp_path / "out" / r["path"]) for r in result["artifacts"] if r["path"].startswith("diagnostics/"))
    assert report["schedule_diagnostics"]["event_schedule_policy"] == "balanced_full_unit"
    assert report["event_timing_policy"]["event_schedule_policy"] == "balanced_full_unit"
    assert (report["full_song_design_sha256"] is not None) is outline
    assert harness.run_harness(source, bank, tmp_path / "out", config, resume=True) == result
    assert len(requests) == result["calls_used"]
    with pytest.raises(ValueError, match="Immutable"):
        harness.run_harness(source, bank, tmp_path / "out", dict(config, use_history=not history), resume=True)
    assert len(requests) == result["calls_used"]


def test_same_units_candidates_clocks_and_per_call_seeds_across_four_arms(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    common = None
    for index, (outline, history) in enumerate(((True, True), (False, True), (True, False), (False, False))):
        requests = []
        ablation_planner(monkeypatch, requests)
        result = harness.run_harness(source, bank, tmp_path / str(index), cfg(use_outline=outline, use_history=history))
        assert result["all_songs_succeeded"]
        values = []
        for r in requests:
            if r["response_kind"] == "coordination_unit":
                p = json.loads(r["messages"][-1]["content"])
                values.append(dict(call_id=r["call_id"], seed=r["seed"], clock=p["event_clock"],
                    local_proposals=p["local_proposals"], music=p["music_context"], unit=p["planning_unit"]))
        if common is None:
            common = values
        else:
            assert values == common


def test_outline_disabled_changes_call_preflight_without_increasing_budget(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    requests = []
    ablation_planner(monkeypatch, requests)
    config = cfg(use_outline=False)
    config["max_calls"] = 2
    result = harness.run_harness(source, bank, tmp_path / "no-outline", config)
    assert result["calls_used"] == 2
    before = len(requests)
    with pytest.raises(ValueError, match="max_calls"):
        harness.run_harness(source, bank, tmp_path / "outline", dict(config, use_outline=True))
    assert len(requests) == before


def test_schema_lower_bound_rejects_dense_free_request_before_any_model_call(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    requests = []
    ablation_planner(monkeypatch, requests)
    config = cfg(use_outline=False)
    config.update(event_schedule_policy="free", event_grid_seconds=.005, max_request_characters=120000)
    result = harness.run_harness(source, bank, tmp_path / "out", config)
    assert result["complete"] and not result["all_songs_succeeded"]
    assert result["calls_used"] == 0 and requests == []
    assert "schema alone exceeds" in result["failed_songs"][0]["message"]
    assert result["failed_songs"][0]["retained_in_evaluation_denominator"]


@pytest.mark.parametrize("name", ["use_outline", "use_history"])
def test_ablation_flags_are_strict_and_only_consumed_in_coordinated(name):
    for invalid in (0, 1, None, "false"):
        with pytest.raises(ValueError, match="boolean"):
            harness._config(dict(cfg(), **{name: invalid}))
    with pytest.raises(ValueError, match="only consumed"):
        harness._config(dict(mode="index0", **{name: True}))


def test_complete_request_guard_is_applied_after_attaching_schema(tmp_path):
    source, song, bank_path = fixture(tmp_path)
    from choreobench.candidates import load_candidates
    from choreobench.planning_units import build_planning_units
    from choreobench.selection import candidate_options
    from choreobench.baseline import _empty_plan
    units = build_planning_units(song)
    options = candidate_options(load_candidates(bank_path), song, units)
    config = harness._config(cfg(use_outline=False))
    request = coordination._unit_request(song, options, units, units["units"][0], coordination._neutral_outline(song),
                                         _empty_plan(song), config)
    size = len(json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    assert size > sum(len(m["content"]) for m in request["messages"])
    with pytest.raises(ModelOutputError, match="Complete serialized"):
        coordination._unit_request(song, options, units, units["units"][0], coordination._neutral_outline(song),
                                  _empty_plan(song), dict(config, max_request_characters=size - 1))
