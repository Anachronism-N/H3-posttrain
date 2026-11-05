"""Coordination uses synthetic providers only; no musical effectiveness is implied."""
import copy
import json
from types import SimpleNamespace

import pytest

from choreobench import coordination, harness
from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.public_structure import load_public_structure
from test_choreobench_harness import fixture


def config(**changes):
    return dict(mode="coordinated", planner_provider={"kind": "fixture"}, max_calls=20,
                max_output_tokens=512, outline_max_output_tokens=512, **changes)


def fake_planner(monkeypatch, requests, mutate=None, infrastructure_error=False):
    identity = dict(kind="synthetic_test", synthetic=True, version=1)
    def generate(request):
        requests.append(copy.deepcopy(request))
        if infrastructure_error:
            raise OSError("unknown network outcome")
        payload = json.loads(request["messages"][-1]["content"])
        if request["response_kind"] == "coordination_outline":
            answer = dict(motifs=[dict(motif_id="m1", description="Open arms then travel softly.")],
                sections=[dict(section_id=s["section_id"], intent="Introduce then vary an open-arm travel phrase.",
                    motif_ids=["m1"]) for s in payload["song"]["music"]["sections"]])
        else:
            unit = payload["planning_unit"]
            answer = dict(unit_id=unit["unit_id"], events=[dict(start=unit["start"], end=unit["end"],
                track="body", action="Develop a connected travel phrase for " + unit["unit_id"],
                motif_id="m1", source_call_ids=[payload["local_proposals"][0]["call_id"]])])
        if mutate:
            mutate(answer, request, payload)
        return dict(text=json.dumps(answer), raw={"fixture": True}, identity=identity, synthetic=True,
                    stop_reason="stop", costs=dict(input_tokens=100, output_tokens=50, latency_seconds=0))
    monkeypatch.setattr(coordination, "create_provider", lambda *a, **k: SimpleNamespace(identity=identity, generate=generate))


def test_outline_then_composition_not_two_second_selection_and_causal_memory(tmp_path, monkeypatch):
    source, song, bank = fixture(tmp_path)
    requests = []
    fake_planner(monkeypatch, requests)
    result = harness.run_harness(source, bank, tmp_path / "out", config())
    assert result["all_songs_succeeded"] and result["calls_used"] == 3
    assert result["original_collection_calls"] == 6 and result["new_local_model_calls"] == 0
    assert result["generated_actions_are_new_proposals"] and not result["model_weights_updated"]
    assert [r["response_kind"] for r in requests] == ["coordination_outline", "coordination_unit", "coordination_unit"]
    first, second = [json.loads(r["messages"][-1]["content"]) for r in requests[1:]]
    assert first["boundary_context"]["body_parts"]["torso"]["latest_event"] is None
    assert second["boundary_context"]["motif_memory"][0]["prior_event_count"] == 1
    assert second["recent_committed_events"][0]["event_id"] == "pu0000-e00"
    assert second["current_motif_definitions"] == [{"motif_id": "m1", "description": "Open arms then travel softly."}]
    assert "motifs" not in second["whole_song_design"]
    assert second["unit_clock"]["duration_seconds"] == 6
    assert second["boundary_context"]["body_parts"]["torso"]["latest_event"]["exit"] == {}
    assert not second["boundary_context"]["body_parts"]["torso"]["state_at_boundary_verified"]
    for request in requests:
        messages = request["messages"][-1]["content"]
        assert "private-song-group" not in messages and "private-artist-group" not in messages
        assert "song.wav" not in messages and "semantic_goals" not in messages and '"goals"' not in messages
        assert "proposed_plan" not in messages and "source_prefix" not in messages.replace("source_prefix_sha256", "")
    plan = read_json(tmp_path / "out" / result["plans"][0]["path"])
    assert len(plan["events"]) == 2
    assert [(e["start"], e["end"]) for e in plan["events"]] == [(0, 96000), (96000, 192000)]
    assert all(e["attributes"] == e["entry"] == e["exit"] == {} for e in plan["events"])
    assert plan["provenance"]["generated_actions_are_new_proposals"]
    assert "not verbatim" in plan["provenance"]["citation_semantics"]
    assert harness.run_harness(source, bank, tmp_path / "out", config(), resume=True) == result
    assert len(requests) == 3


def sidecar(tmp_path, source, song):
    structure = dict(schema="choreobench.public_structure.v1", dataset_sha256=file_sha(source), songs=[dict(
        song_id=song["song_id"], audio_sha256=song["audio"]["sha256"], sample_rate=16000, num_samples=192000,
        synthetic=True, music=dict(source="synthetic detector", beats=[0, 32000, 64000, 96000, 128000, 160000],
            downbeats=[0, 64000, 128000], sections=[dict(section_id="new", start=0, end=192000,
                label="unknown", repeat_group=None)]))])
    path = tmp_path / "structure.json"
    write_json(path, structure)
    return path, structure


def test_public_structure_updates_planning_without_loosening_bank_identity(tmp_path, monkeypatch):
    source, song, bank = fixture(tmp_path)
    path, structure = sidecar(tmp_path, source, song)
    requests = []
    fake_planner(monkeypatch, requests)
    result = harness.run_harness(source, bank, tmp_path / "out", config(public_structure_path=str(path)))
    assert result["all_songs_succeeded"] and result["synthetic"]
    original = read_json(source)["songs"][0]
    assert original == song and original["music"]["sections"][0]["section_id"] == "a"
    outline_input = json.loads(requests[0]["messages"][-1]["content"])
    assert outline_input["song"]["music"] == structure["songs"][0]["music"]
    contract = read_json(tmp_path / "out/contract.json")
    assert contract["public_structure_sha256"] == file_sha(path)
    assert contract["dataset_snapshot"] == read_json(source)
    changed = read_json(path)
    changed["songs"][0]["music"]["source"] += " changed"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Immutable"):
        harness.run_harness(source, bank, tmp_path / "out", config(public_structure_path=str(path)), resume=True)
    assert len(requests) == result["calls_used"]


@pytest.mark.parametrize("mutation", ["private", "clock", "hash", "dataset", "missing", "duplicate", "bad_music"])
def test_structure_rejects_private_or_unbound_input_before_any_provider(tmp_path, monkeypatch, mutation):
    source, song, bank = fixture(tmp_path)
    path, value = sidecar(tmp_path, source, song)
    if mutation == "private":
        value["songs"][0]["semantic_goals"] = ["private"]
    elif mutation == "clock":
        value["songs"][0]["sample_rate"] = True
    elif mutation == "hash":
        value["songs"][0]["audio_sha256"] = "0" * 64
    elif mutation == "dataset":
        value["dataset_sha256"] = "0" * 64
    elif mutation == "missing":
        value["songs"] = []
    elif mutation == "duplicate":
        value["songs"].append(copy.deepcopy(value["songs"][0]))
    else:
        value["songs"][0]["music"]["sections"][0]["end"] -= 1
    path.write_text(json.dumps(value))
    monkeypatch.setattr(coordination, "create_provider", lambda *a, **k: pytest.fail("No provider before input validation"))
    with pytest.raises(ValueError):
        harness.run_harness(source, bank, tmp_path / "out", config(public_structure_path=str(path)))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("problem", ["unit_id", "outside_time", "unknown_motif", "unknown_source", "old_source",
    "duplicate_source", "invented_state", "overlap", "reorder", "empty", "too_many"])
def test_bad_unit_keeps_raw_and_never_commits_or_retries(tmp_path, monkeypatch, problem):
    source, _, bank = fixture(tmp_path)
    requests = []
    def mutate(answer, request, payload):
        if request["response_kind"] != "coordination_unit":
            return
        event = answer["events"][0]
        if problem == "unit_id": answer["unit_id"] = "wrong"
        elif problem == "outside_time": event["end"] += 1
        elif problem == "unknown_motif": event["motif_id"] = "wrong"
        elif problem == "unknown_source": event["source_call_ids"] = ["private-reference"]
        elif problem == "old_source": event["source_call_ids"] = ["song1/local/9999/00"]
        elif problem == "duplicate_source": event["source_call_ids"] *= 2
        elif problem == "invented_state": event["exit"] = {"support_foot": "left"}
        elif problem == "empty": answer["events"] = []
        elif problem == "too_many": answer["events"] *= 9
        elif problem == "overlap": answer["events"].append(dict(event, track="upper"))
        else:
            middle = (event["start"] + event["end"]) // 2
            answer["events"] = [dict(event, start=middle), dict(event, end=middle)]
    fake_planner(monkeypatch, requests, mutate)
    result = harness.run_harness(source, bank, tmp_path / "out", config())
    assert result["complete"] and not result["all_songs_succeeded"]
    assert result["calls_used"] == 2 and len(requests) == 2
    assert result["failed_songs"][0]["retained_in_evaluation_denominator"]
    assert not list((tmp_path / "out").glob("blocks/*/*.json"))
    assert harness.run_harness(source, bank, tmp_path / "out", config(), resume=True) == result
    assert len(requests) == 2


def test_unknown_network_outcome_and_budget_do_not_retry(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    requests = []
    fake_planner(monkeypatch, requests, infrastructure_error=True)
    with pytest.raises(ValueError, match="max_calls"):
        harness.run_harness(source, bank, tmp_path / "budget", dict(config(), max_calls=2))
    assert requests == [] and not (tmp_path / "budget").exists()
    for resume in (False, True):
        with pytest.raises(ValueError, match="provider failure"):
            harness.run_harness(source, bank, tmp_path / "out", config(), resume=resume)
    assert len(requests) == 1
    result = read_json(tmp_path / "out/results.json")
    assert not result["complete"] and result["unknown_outcomes_count"] == 1


def test_block_exposes_only_current_section_motif_definitions_and_memory(tmp_path):
    source, song, bank_path = fixture(tmp_path)
    from choreobench.candidates import load_candidates
    from choreobench.planning_units import build_planning_units
    from choreobench.selection import candidate_options
    from choreobench.baseline import _empty_plan
    units = build_planning_units(song)
    options = candidate_options(load_candidates(bank_path), song, units)
    outline = {"motifs": [{"motif_id": "m1", "description": "Early travel"},
                          {"motif_id": "m2", "description": "Later open arms"}],
               "sections": [{"section_id": "a", "intent": "Introduce travel", "motif_ids": ["m1"]},
                            {"section_id": "b", "intent": "Develop the ending", "motif_ids": ["m2"]}]}
    plan = _empty_plan(song)
    plan["motifs"] = outline["motifs"]
    plan["events"] = [{"event_id": "e0", "start": 0, "end": 96000, "track": "body", "action": "Travel early",
        "attributes": {}, "entry": {}, "exit": {}, "motif_id": "m1", "evidence": []}]
    request = coordination._unit_request(song, options, units, units["units"][1], outline, plan, harness._config(config()))
    payload = json.loads(request["messages"][1]["content"])
    assert payload["current_motif_definitions"] == [outline["motifs"][1]]
    assert [m["motif_id"] for m in payload["boundary_context"]["motif_memory"]] == ["m2"]
    assert "motifs" not in payload["whole_song_design"]
    assert payload["recent_committed_events"][0]["motif_id"] == "m1"
    assert payload["output_contract"]["allowed_motif_ids"] == ["m2"]
    assert "SAFETY CEILING" in request["messages"][0]["content"]


def test_invalid_second_block_does_not_enter_future_history(tmp_path, monkeypatch):
    source, _, bank = fixture(tmp_path)
    requests = []
    def mutate(answer, request, payload):
        if request["response_kind"] == "coordination_unit" and payload["planning_unit"]["unit_id"] == "pu0001":
            answer["events"][0]["end"] += 1
    fake_planner(monkeypatch, requests, mutate)
    result = harness.run_harness(source, bank, tmp_path / "out", config())
    assert not result["plans"] and len(requests) == 3
    blocks = list((tmp_path / "out").glob("blocks/*/*.json"))
    assert len(blocks) == 1
    assert len(read_json(blocks[0])["events"]) == 1


def test_same_structure_can_be_used_by_select_only_without_local_resampling(tmp_path):
    source, song, bank = fixture(tmp_path)
    path, _ = sidecar(tmp_path, source, song)
    result = harness.run_harness(source, bank, tmp_path / "out", dict(mode="index0", max_calls=0, public_structure_path=str(path)))
    assert result["all_songs_succeeded"] and result["new_local_model_calls"] == 0
    units = read_json(tmp_path / "out" / next(r["path"] for r in result["artifacts"] if r["path"].startswith("units/")))
    assert all(u["section_id"] == "new" for u in units["units"])


def test_structured_unit_schema_contains_only_current_public_bounds_and_ids(tmp_path):
    source, song, bank_path = fixture(tmp_path)
    from choreobench.candidates import load_candidates
    from choreobench.planning_units import build_planning_units
    from choreobench.selection import candidate_options
    from choreobench.baseline import _empty_plan
    units = build_planning_units(song)
    options = candidate_options(load_candidates(bank_path), song, units)
    outline = dict(motifs=[dict(motif_id="m1", description="Travel")],
        sections=[dict(section_id=s["section_id"], intent="Vary travel", motif_ids=["m1"]) for s in song["music"]["sections"]])
    plan = _empty_plan(song); plan["motifs"] = outline["motifs"]
    request = coordination._unit_request(song, options, units, units["units"][0], outline, plan,
                                         harness._config(config(structured_unit_output=True)))
    schema = request["response_schema"]
    fields = schema["properties"]["events"]["items"]["properties"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["unit_id"]["enum"] == ["pu0000"]
    assert schema["properties"]["events"]["maxItems"] == 3
    assert fields["motif_id"]["enum"] == ["m1", None]
    assert fields["start"]["minimum"] == 0 and fields["end"]["maximum"] == 96000
    assert "private-song-group" not in json.dumps(schema) and "semantic_goals" not in json.dumps(schema)
    assert "support_foot" not in json.dumps(schema)


@pytest.mark.parametrize("budget", [0, 9, True, 1.5])
def test_phrase_budget_is_bounded_before_any_provider(budget):
    with pytest.raises(ValueError, match="max_events_per_unit"):
        harness._config(config(max_events_per_unit=budget))


@pytest.mark.parametrize("module", [harness, coordination])
def test_changed_source_process_refuses_before_any_data_or_provider(monkeypatch, module):
    monkeypatch.setattr(module, "_IMPORT_SIGNATURE", None)
    with pytest.raises(ValueError, match="source changed"):
        if module is harness:
            harness.run_harness("unused-dataset", "unused-bank", "unused-out", config())
        else:
            coordination.run_coordinated("unused-dataset", "unused-bank", "unused-out", config())


@pytest.mark.parametrize("time_format,policy", [("global_samples", "free"), ("boundary_ids", "free"),
                                                   ("boundary_ids", "balanced_full_unit")])
def test_schema_response_format_is_journalable_not_silent_fallback(monkeypatch, time_format, policy):
    from choreobench.providers import ChatCompletionsProvider
    captured = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}'
    def send(request, timeout):
        captured.append(json.loads(request.data)); return Response()
    monkeypatch.setattr("urllib.request.urlopen", send)
    provider = ChatCompletionsProvider(dict(kind="chat_completions", base_url="https://example.test/v1", model="test",
                                           api_key_env="", json_mode=True))
    from choreobench.timing import event_clock
    unit = dict(unit_id="u0", start=0, end=100)
    clock = event_clock(unit, 100, dict(event_time_format=time_format, event_grid_seconds=.25, min_event_seconds=.5,
                                       event_schedule_policy=policy))
    schema = coordination.unit_response_schema(unit, ["m1"], ["call1"], clock=clock)
    request = dict(seed=7, max_output_tokens=32, messages=[dict(role="user", content="public")], response_schema=schema)
    response = provider.generate(request)
    assert len(captured) == 1 and captured[0]["response_format"]["type"] == "json_schema"
    transported = captured[0]["response_format"]["json_schema"]["schema"]
    assert transported == schema
    if policy == "balanced_full_unit":
        arrays = transported["properties"]["events"]["anyOf"]
        assert all(array["items"] is False and len(array["prefixItems"]) == array["maxItems"] for array in arrays)
        assert all(list(slot["properties"])[:2] == ["start_boundary", "end_boundary"]
                   for array in arrays for slot in array["prefixItems"])
    elif time_format == "boundary_ids":
        item = transported["properties"]["events"]["items"]
        assert set(item) == {"anyOf"}
        assert all(list(branch["properties"])[:2] == ["start_boundary", "end_boundary"] for branch in item["anyOf"])
    assert response["raw"]["request"] == captured[0]
    request["response_schema"] = {"type": "object"}
    with pytest.raises(ValueError, match="closed JSON"):
        provider.generate(request)
    assert len(captured) == 1
