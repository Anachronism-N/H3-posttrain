import json

import pytest

from choreobench import coordination, harness
from choreobench.io import read_json
from choreobench.timing import TimingInfeasibleError, event_clock, event_interval, literal_source_copies
from test_choreobench_coordination import config, fake_planner
from test_choreobench_harness import fixture


def test_sample_rounding_real_endpoint_and_duration_ceiling():
    clock = event_clock(dict(start=100, end=110), 10,
                        dict(event_time_format="boundary_ids", event_grid_seconds=.25, min_event_seconds=.31))
    assert clock["boundary_samples"] == [100, 103, 106, 109, 110]
    assert clock["minimum_duration_samples"] == 4
    assert event_interval(dict(start_boundary=0, end_boundary=2), clock) == (100, 106)
    with pytest.raises(ValueError, match="shorter"):
        event_interval(dict(start_boundary=3, end_boundary=4), clock)


@pytest.mark.parametrize("start,end", [(True, 2), (0, False), (-1, 2), (0, 99), (2, 1), (1, 1), (0., 2)])
def test_boundary_ids_reject_coercion_reversal_and_out_of_range(start, end):
    clock = event_clock(dict(start=0, end=16000), 16000, dict(event_time_format="boundary_ids"))
    with pytest.raises(ValueError):
        event_interval(dict(start_boundary=start, end_boundary=end), clock)


@pytest.mark.parametrize("start,end,rate,step,minimum", [(100, 110, 10, .25, .31),
    (324000, 472960, 16000, .25, .5), (0, 16000, 16000, .25, 0)])
def test_decoder_branches_cover_exactly_feasible_intervals(start, end, rate, step, minimum):
    unit = dict(unit_id="u", start=start, end=end)
    clock = event_clock(unit, rate, dict(event_time_format="boundary_ids", event_grid_seconds=step, min_event_seconds=minimum))
    item = coordination.unit_response_schema(unit, ["m"], ["call"], clock=clock)["properties"]["events"]["items"]
    samples = clock["boundary_samples"]
    permitted = set()
    for branch in item["anyOf"]:
        assert branch["type"] == "object" and branch["additionalProperties"] is False
        fields = branch["properties"]
        assert branch["required"] == list(fields)
        a = fields["start_boundary"]["enum"][0]
        assert a not in {pair[0] for pair in permitted}
        for b in range(fields["end_boundary"]["minimum"], fields["end_boundary"]["maximum"] + 1):
            permitted.add((a, b))
    expected = {(a, b) for a, sa in enumerate(samples) for b, sb in enumerate(samples)
                if sb - sa >= clock["minimum_duration_samples"]}
    assert permitted == expected
    assert all(event_interval(dict(start_boundary=a, end_boundary=b), clock) == (samples[a], samples[b])
               for a, b in permitted)


def grid_reply(answer, request, payload):
    if request["response_kind"] == "coordination_unit":
        for event in answer["events"]:
            event.pop("start")
            event.pop("end")
            event.update(start_boundary=0, end_boundary=len(payload["event_clock"]["boundary_samples"]) - 1)


def test_grid_composition_compiles_to_exact_original_clock_and_resumes(tmp_path, monkeypatch):
    source, song, bank = fixture(tmp_path)
    requests = []
    fake_planner(monkeypatch, requests, grid_reply)
    cfg = dict(config(), event_time_format="boundary_ids", event_grid_seconds=.25,
               min_event_seconds=.5, structured_unit_output=True)
    result = harness.run_harness(source, bank, tmp_path / "out", cfg)
    assert result["all_songs_succeeded"]
    plan = read_json(tmp_path / "out" / result["plans"][0]["path"])
    assert plan["events"][0]["start"] == 0
    assert plan["events"][-1]["end"] == song["audio"]["num_samples"]
    for request in requests[1:]:
        payload = json.loads(request["messages"][-1]["content"])
        item = request["response_schema"]["properties"]["events"]["items"]
        assert set(item) == {"anyOf"}
        fields = item["anyOf"][0]["properties"]
        assert all(list(branch["properties"]) == payload["output_contract"]["fields"] for branch in item["anyOf"])
        assert list(fields) == payload["output_contract"]["fields"]
        assert list(fields)[:2] == ["start_boundary", "end_boundary"]
        assert "start" not in fields
        assert "the next start_boundary" in request["messages"][0]["content"]
        assert "whole-block body phrase must be the sole event" in request["messages"][0]["content"]
        assert payload["event_clock"]["minimum_duration_samples"] == 8000
        assert not payload["event_clock"]["beat_alignment_verified"]
    assert harness.run_harness(source, bank, tmp_path / "out", cfg, resume=True) == result
    assert len(requests) == result["calls_used"]
    with pytest.raises(ValueError, match="Immutable"):
        harness.run_harness(source, bank, tmp_path / "out", dict(cfg, event_grid_seconds=.5), resume=True)


@pytest.mark.parametrize("overlap", [False, True])
def test_multiple_grid_phrases_accept_adjacent_intervals_but_reject_body_overlay(tmp_path, monkeypatch, overlap):
    source, _, bank = fixture(tmp_path)
    requests = []
    def phrases(answer, request, payload):
        if request["response_kind"] != "coordination_unit":
            return
        event = answer["events"][0]
        event.pop("start")
        event.pop("end")
        last = len(payload["event_clock"]["boundary_samples"]) - 1
        cuts = [0, last // 3, 2 * last // 3, last]
        answer["events"] = [dict(event, start_boundary=a, end_boundary=b) for a, b in zip(cuts, cuts[1:])]
        if overlap:
            answer["events"][0]["end_boundary"] = last
    fake_planner(monkeypatch, requests, phrases)
    result = harness.run_harness(source, bank, tmp_path / "out",
        dict(config(), event_time_format="boundary_ids", min_event_seconds=.5, structured_unit_output=True))
    assert result["complete"]
    if overlap:
        assert result["success_count"] == 0 and result["calls_used"] == 2
        assert "Overlapping incompatible body scopes" in result["failed_songs"][0]["message"]
        assert result["failed_songs"][0]["retained_in_evaluation_denominator"]
        assert not list((tmp_path / "out").glob("blocks/*/*.json"))
    else:
        assert result["all_songs_succeeded"]
        plan = read_json(tmp_path / "out" / result["plans"][0]["path"])
        assert len(plan["events"]) == 6
        assert all(a["end"] == b["start"] for a, b in zip(plan["events"], plan["events"][1:]))


@pytest.mark.parametrize("duration,maximum_count", [(358, 8), (318, 7)])
def test_balanced_schedule_reserves_grid_duration_for_remaining_internal_phrases(duration, maximum_count):
    clock = event_clock(dict(start=0, end=duration), 100, dict(event_time_format="boundary_ids",
        event_grid_seconds=.4, min_event_seconds=.39, event_schedule_policy="balanced_full_unit", max_events_per_unit=8))
    assert [len(s) for s in clock["permitted_schedules"]] == list(range(1, maximum_count + 1))
    for schedule in clock["permitted_schedules"]:
        intervals = [event_interval(e, clock) for e in schedule]
        assert intervals[0][0] == 0 and intervals[-1][1] == duration
        assert all(b - a >= 39 for a, b in intervals)
        assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))


@pytest.mark.parametrize("end,rate,step,minimum", [(162000, 16000, .25, .5), (110, 10, .25, .31), (25, 10, 1, .6)])
def test_balanced_public_schedules_preserve_real_end_and_duration(end, rate, step, minimum):
    unit = dict(start=0, end=end, unit_id="u")
    clock = event_clock(unit, rate, dict(event_time_format="boundary_ids", event_grid_seconds=step,
        min_event_seconds=minimum, event_schedule_policy="balanced_full_unit"))
    assert clock["permitted_schedules"] and not clock["schedule_is_musical_or_physical_evidence"]
    schema = coordination.unit_response_schema(unit, [], ["call"], clock=clock)
    alternatives = schema["properties"]["events"]["anyOf"]
    assert len(alternatives) == len(clock["permitted_schedules"])
    for schedule, array in zip(clock["permitted_schedules"], alternatives):
        assert 1 <= len(schedule) <= 3
        intervals = [event_interval(event, clock) for event in schedule]
        assert intervals[0][0] == 0 and intervals[-1][1] == end
        assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
        assert array["minItems"] == array["maxItems"] == len(schedule) and array["items"] is False
        for pair, slot in zip(schedule, array["prefixItems"]):
            assert slot["additionalProperties"] is False
            assert all(slot["properties"][name]["enum"] == [value] for name, value in pair.items())


@pytest.mark.parametrize("change_schedule", [False, True])
def test_complete_public_schedule_compiles_or_rejects_without_filling(tmp_path, monkeypatch, change_schedule):
    source, song, bank = fixture(tmp_path)
    requests = []
    def compose(answer, request, payload):
        if request["response_kind"] != "coordination_unit":
            return
        original = answer["events"][0]
        original.pop("start"); original.pop("end")
        answer["events"] = [dict(original, **pair) for pair in payload["event_clock"]["permitted_schedules"][-1]]
        if change_schedule:
            answer["events"][0]["start_boundary"] += 1
    fake_planner(monkeypatch, requests, compose)
    cfg = dict(config(), event_time_format="boundary_ids", min_event_seconds=.5,
               event_schedule_policy="balanced_full_unit", structured_unit_output=True)
    result = harness.run_harness(source, bank, tmp_path / "out", cfg)
    if change_schedule:
        assert result["complete"] and not result["all_songs_succeeded"] and result["calls_used"] == 2
        assert "publicly permitted full-unit schedule" in result["failed_songs"][0]["message"]
        assert not list((tmp_path / "out").glob("blocks/*/*.json"))
    else:
        assert result["all_songs_succeeded"]
        plan = read_json(tmp_path / "out" / result["plans"][0]["path"])
        assert sum(e["end"] - e["start"] for e in plan["events"]) == song["audio"]["num_samples"]
        assert all(not e["entry"] and not e["exit"] for e in plan["events"])
        assert harness.run_harness(source, bank, tmp_path / "out", cfg, resume=True) == result
    for request in requests[1:]:
        assert "PUBLIC complete-unit scheduling constraint" in request["messages"][0]["content"]


@pytest.mark.parametrize("change", [dict(event_schedule_policy="unknown"),
    dict(event_schedule_policy="balanced_full_unit", event_time_format="global_samples")])
def test_invalid_public_schedule_config(change):
    with pytest.raises(ValueError):
        harness._config(dict(config(), **change))
    with pytest.raises(ValueError, match="only consumed"):
        harness._config(dict(mode="index0", event_schedule_policy="free"))


@pytest.mark.parametrize("mode", ["global_samples", "boundary_ids"])
def test_short_action_rejected_without_stretch_retry_or_commit(tmp_path, monkeypatch, mode):
    source, _, bank = fixture(tmp_path)
    requests = []
    def shorten(answer, request, payload):
        if request["response_kind"] != "coordination_unit":
            return
        event = answer["events"][0]
        if mode == "boundary_ids":
            event.pop("start")
            event.pop("end")
            event.update(start_boundary=0, end_boundary=1)
        else:
            event["end"] = event["start"] + 1
    fake_planner(monkeypatch, requests, shorten)
    result = harness.run_harness(source, bank, tmp_path / "out",
        dict(config(), event_time_format=mode, min_event_seconds=.5))
    assert result["complete"] and not result["all_songs_succeeded"] and result["calls_used"] == 2
    assert "shorter" in result["failed_songs"][0]["message"]
    assert result["failed_songs"][0]["retained_in_evaluation_denominator"]
    assert not list((tmp_path / "out").glob("blocks/*/*.json"))


def test_infeasible_real_unit_abstains_before_any_model_call(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    requests = []
    fake_planner(monkeypatch, requests)
    result = harness.run_harness(source, bank, tmp_path / "out", dict(config(), min_event_seconds=100))
    assert result["complete"] and result["success_count"] == 0 and result["calls_used"] == 0
    assert result["failed_songs"][0]["retained_in_evaluation_denominator"] and requests == []


def test_literal_copy_flag_is_computed_including_cited_fragment(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    requests = []
    def copy_source(answer, request, payload):
        if request["response_kind"] == "coordination_unit":
            answer["events"][0]["action"] = payload["local_proposals"][0]["text"]
    fake_planner(monkeypatch, requests, copy_source)
    result = harness.run_harness(source, bank, tmp_path / "out", config())
    report = next(read_json(tmp_path / "out" / r["path"]) for r in result["artifacts"] if r["path"].startswith("diagnostics/"))
    assert "new_actions_not_verbatim" not in report
    assert report["literal_source_copies"]["exact_match_fraction"] == 1
    assert not report["literal_source_copies"]["semantic_novelty_verified"]


def test_literal_copy_compares_fragments_only_against_cited_sources():
    options = {"windows": [{"candidates": [dict(call_id="a", raw_text="Two captions", fragments=[dict(text="Raise arms")]),
                                          dict(call_id="b", raw_text="Turn", fragments=[])]}]}
    plan = {"events": [dict(event_id="e1", action="Raise arms", evidence=["a"]),
                       dict(event_id="e2", action="Turn", evidence=["a"])]}
    result = literal_source_copies(plan, options)
    assert result["matching_event_ids"] == ["e1"] and result["exact_match_fraction"] == .5


def test_grid_capacity_and_subsample_steps_fail_before_provider(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    def no_provider(*a, **k):
        pytest.fail("invalid grid must fail before provider initialization")
    monkeypatch.setattr("choreobench.coordination.create_provider", no_provider)
    for step in (.000001, .0001):
        with pytest.raises(ValueError, match="at least one sample|4097"):
            harness.run_harness(source, bank, tmp_path / str(step),
                dict(config(), event_time_format="boundary_ids", event_grid_seconds=step))


@pytest.mark.parametrize("change", [dict(min_event_seconds=True), dict(min_event_seconds=-1),
    dict(event_time_format="seconds"), dict(event_grid_seconds=0), dict(min_event_seconds=float("inf"))])
def test_invalid_clock_config(change):
    with pytest.raises(ValueError):
        harness._config(dict(config(), **change))


def test_clock_config_is_not_silently_ignored_by_selection_modes():
    with pytest.raises(ValueError, match="only consumed"):
        harness._config(dict(mode="index0", min_event_seconds=.5))
    with pytest.raises(TimingInfeasibleError):
        event_clock(dict(start=0, end=1), 16000, dict(min_event_seconds=.5))


def test_normalized_configs_can_be_saved_and_read_without_unused_clock_fields():
    for cfg in ({"mode": "index0"}, config()):
        normalized = harness._config(cfg)
        assert harness._config(normalized) == normalized
    assert "min_event_seconds" not in harness._config({"mode": "index0"})
