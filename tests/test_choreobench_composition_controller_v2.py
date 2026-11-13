"""Bounded-controller regressions; synthetic trajectories are not quality results."""
import copy
import json

import pytest

from choreobench import composition_workflows as m
from choreobench.io import read_json
from test_choreobench_composition_workflows import (
    decision,
    material,
    proposal,
    state as legacy_state,
)


def state(tmp_path, monkeypatch):
    from choreobench.composition_controller_v2 import CompositionStateV2

    original = legacy_state(tmp_path, monkeypatch)
    return CompositionStateV2(original.source, original.outline)


def run(path, sid, out, method="choreography", **kwargs):
    return m.run_composition(
        path, sid, {"kind": "replay"}, out,
        method=method, controller_version="v2", **kwargs,
    )


def steps(out):
    return [read_json(p) for p in sorted((out / "steps").glob("*.json"))]


@pytest.mark.parametrize("flags,expected", [([], "v1"), (["--controller-version", "v2"], "v2")])
def test_cli_controller_is_opt_in_and_forwarded(tmp_path, monkeypatch, flags, expected):
    from choreobench.cli import main

    provider = tmp_path / "provider.json"
    provider.write_text(json.dumps({"kind": "replay"}), encoding="utf-8")
    received = []

    def capture(*args, **kwargs):
        received.append(kwargs)
        return {"controller_version": kwargs["controller_version"]}

    monkeypatch.setattr(m, "run_composition", capture)
    assert main([
        "run-composition", "--source", "frozen-source", "--song-id", "song-01",
        "--provider", str(provider), "--out-dir", str(tmp_path / "run"),
        "--method", "choreography",
    ] + flags) == 0
    assert len(received) == 1 and received[0]["controller_version"] == expected
    assert received[0]["max_input_tokens"] == 2000000
    assert received[0]["max_total_output_tokens"] == 150000


@pytest.mark.parametrize(
    "method,expected_calls",
    [("workflow", 3), ("self_refine", 7), ("react", 5), ("choreography", 5)],
)
def test_v2_finishes_after_last_commit_and_replays_without_new_calls(
    tmp_path, monkeypatch, method, expected_calls,
):
    path, sid, out, calls = material(tmp_path, monkeypatch, method)
    result = run(path, sid, out, method)

    assert result["full_song_succeeded"]
    assert result["committed_units"] == result["total_units"] == 2
    assert result["calls_used"] == len(calls) == expected_calls
    assert result["stop_reason"] == "finished"
    assert all(s["decision"].get("tool") != "finish" for s in steps(out))
    contract = read_json(out / "contract.json")
    assert contract["config"]["controller_version"] == "v2"
    assert contract["config"]["max_calls"] == 9
    assert not result["quality_verified"]
    assert not result["formal_E3_completed"]
    assert run(path, sid, out, method, resume=True) == result
    assert len(calls) == expected_calls


@pytest.mark.parametrize("method", ["react", "choreography"])
def test_submit_is_one_model_call_per_unit_and_exposes_remaining_budgets(
    tmp_path, monkeypatch, method,
):
    def submit(answer, req, payload, index):
        if answer.get("tool") == "propose_block":
            answer["tool"] = "submit_block"

    path, sid, out, calls = material(tmp_path, monkeypatch, method, submit)
    result = run(path, sid, out, method)
    assert result["full_song_succeeded"] and len(calls) == 3
    payloads = [json.loads(c["messages"][-1]["content"]) for c in calls[1:]]
    assert [p["committed_units"] for p in payloads] == [0, 1]
    assert [p["remaining_model_calls"] for p in payloads] == [8, 7]
    assert [p["remaining_input_tokens"] for p in payloads] == [1999900, 1999800]
    assert [p["remaining_output_tokens"] for p in payloads] == [149950, 149900]
    assert all("finish" not in p["available_tools"] for p in payloads)
    assert all("submit_block" in p["available_tools"] for p in payloads)
    for request, payload in zip(calls[1:], payloads):
        schema = request["response_schema"]
        assert schema["properties"]["tool"]["enum"] == payload["available_tools"]
        branches = {b["properties"]["tool"]["enum"][0]: b for b in schema["anyOf"]}
        assert branches["submit_block"]["properties"]["proposal"]["type"] == "object"
        assert branches["read_history"]["properties"]["proposal"]["type"] == "null"


def test_same_draft_revision_commits_only_valid_content(tmp_path, monkeypatch):
    s = state(tmp_path, monkeypatch)
    draft = proposal(s)
    s.execute(decision("propose_block", draft))
    same = decision("revise_block", copy.deepcopy(draft))
    same["reason"] = "Different prose must not conceal an identical draft."
    observed = s.execute(same)
    assert observed["status"] == "committed"
    assert observed["automatic_commit_reason"] == "unchanged_revision"
    assert s.index == 1 and s.draft is None
    assert s.plan["events"][0]["action"] == draft["events"][0]["action"]
    before = copy.deepcopy(s.plan)
    with pytest.raises(ValueError):
        s.execute(same)  # A stale proposal cannot rewrite the committed prefix.
    assert s.plan == before and s.index == 1


def test_invalid_unchanged_revision_never_auto_commits(tmp_path, monkeypatch):
    s = state(tmp_path, monkeypatch)
    bad = proposal(s)
    bad["events"][0]["source_call_ids"] = []
    s.execute(decision("propose_block", bad))
    with pytest.raises(ValueError):
        s.execute(decision("revise_block", copy.deepcopy(bad)))
    assert s.index == 0 and s.plan["events"] == []
    assert s.revisions_used == 1
    with pytest.raises(ValueError):
        s.execute(decision("revise_block", proposal(s, "Another attempt")))
    assert s.plan["events"] == []


def test_invalid_submit_is_uncommitted_and_can_be_repaired_once(tmp_path, monkeypatch):
    s = state(tmp_path, monkeypatch)
    bad = proposal(s)
    bad["events"][0]["source_call_ids"] = []
    with pytest.raises(ValueError):
        s.execute(decision("submit_block", bad))
    assert s.draft == bad and s.plan["events"] == [] and s.index == 0
    s.execute(decision("revise_block", proposal(s, "Repaired content")))
    s.execute(decision("commit_block"))
    assert s.index == 1 and s.plan["events"][0]["action"] == "Repaired content"


def test_submit_cannot_bypass_spent_revision_limit(tmp_path, monkeypatch):
    s = state(tmp_path, monkeypatch)
    s.execute(decision("propose_block", proposal(s, "Initial draft")))
    s.execute(decision("revise_block", proposal(s, "The one allowed revision")))
    retained = copy.deepcopy(s.draft)
    with pytest.raises(ValueError):
        s.execute(decision("submit_block", proposal(s, "Hidden second revision")))
    assert s.index == 0 and s.draft == retained and not s.plan["events"]
    s.execute(decision("commit_block"))
    assert s.plan["events"][0]["action"] == "The one allowed revision"


def test_server_rejects_hidden_read_when_only_commit_calls_remain(tmp_path, monkeypatch):
    s = state(tmp_path, monkeypatch)
    remaining = len(s.source["units"]["units"])
    s.remaining_calls = remaining
    assert s.available_tools(remaining_calls=remaining) == ["submit_block"]
    with pytest.raises(ValueError):
        s.execute(decision("read_history"))
    with pytest.raises(ValueError):
        s.execute(decision("propose_block", proposal(s)))
    assert s.plan["events"] == [] and s.draft is None
    s.execute(decision("submit_block", proposal(s)))
    assert s.index == 1


def test_server_rejects_early_finish_without_discarding_work(tmp_path, monkeypatch):
    s = state(tmp_path, monkeypatch)
    assert "finish" not in s.available_tools()
    with pytest.raises(ValueError):
        s.execute(decision("finish"))
    assert s.index == 0 and not s.plan["events"]


@pytest.mark.parametrize("method", ["react", "choreography"])
def test_changing_reason_cannot_hide_three_idle_decisions(tmp_path, monkeypatch, method):
    def idle(answer, request, payload, index):
        if request["response_kind"] == "coordination_outline":
            return
        answer.clear()
        answer.update(decision("read_history"))
        answer["reason"] = "A fresh explanation %d with no actual progress" % index

    path, sid, out, calls = material(tmp_path, monkeypatch, method, idle)
    result = run(path, sid, out, method)
    assert not result["full_song_succeeded"]
    assert result["stop_reason"] == "stalled_no_content_progress"
    assert result["calls_used"] == len(calls) == 4
    assert result["committed_units"] == 0
    assert not (out / "plan.json").exists()
    assert result["input_tokens"] == 400 and result["output_tokens"] == 200


def test_idle_valid_draft_advances_without_an_extra_finish_call(tmp_path, monkeypatch):
    def idle_after_draft(answer, request, payload, index):
        if request["response_kind"] != "coordination_outline" and payload["current_draft"]:
            answer.clear()
            answer.update(decision("read_history"))
            answer["reason"] = "Same draft with a new reason %d" % index

    path, sid, out, calls = material(tmp_path, monkeypatch, "choreography", idle_after_draft)
    result = run(path, sid, out)
    assert result["full_song_succeeded"] and result["committed_units"] == 2
    assert result["calls_used"] == len(calls) <= 9
    assert result["stop_reason"] == "finished"
    assert all(s["decision"].get("tool") != "finish" for s in steps(out))
    assert read_json(out / "plan.json")["events"]


def test_read_context_is_available_once_without_repeated_snapshot_copies(tmp_path, monkeypatch):
    def inspect_then_submit(answer, request, payload, index):
        if request["response_kind"] == "coordination_outline":
            return
        if index in (2, 3, 5):
            answer.clear()
            answer.update(decision("read_evidence" if index == 2 else "read_history"))
        elif answer.get("tool") == "propose_block":
            answer["tool"] = "submit_block"

    path, sid, out, calls = material(tmp_path, monkeypatch, "choreography", inspect_then_submit)
    result = run(path, sid, out)
    assert result["full_song_succeeded"] and len(calls) == 6
    payloads = [json.loads(c["messages"][-1]["content"]) for c in calls[1:]]
    assert all(len(p["tool_observations"]) <= 2 for p in payloads)
    # Evidence beyond the current unit must reach the planner on the next turn.
    assert any(o.get("candidates") for o in payloads[1]["tool_observations"])
    # Later requests retain a compact record, not repeated full candidate banks.
    assert all("candidates" not in o for o in payloads[2]["tool_observations"])
    assert all("committed_plan" not in o for p in payloads for o in p["tool_observations"])
    assert all("recent_committed_events" not in p for p in payloads)
    assert payloads[-1]["committed_plan"]["events"]
    # Original observations remain inspectable in the immutable audit trail.
    original_steps = steps(out)
    assert original_steps[0]["observation"]["candidates"]
    assert original_steps[3]["observation"]["committed_plan"]["events"]


def test_v2_token_budget_cannot_be_evaded_by_auto_commit(tmp_path, monkeypatch):
    path, sid, out, calls = material(tmp_path, monkeypatch, "choreography")
    result = run(path, sid, out, max_input_tokens=150)
    assert not result["full_song_succeeded"] and result["calls_used"] == len(calls) == 0
    assert read_json(out / "partial-plan.not-success.json")["events"] == []


@pytest.mark.parametrize("usage_field,cap", [("input_tokens", 100000), ("output_tokens", 150000)])
def test_actual_usage_over_budget_cannot_trigger_a_successful_fallback(
    tmp_path, monkeypatch, usage_field, cap,
):
    def submit_then_idle(answer, request, payload, index):
        if index == 2:
            answer["tool"] = "submit_block"
        elif index >= 4:
            answer.clear()
            answer.update(decision("read_history"))

    path, sid, out, calls = material(tmp_path, monkeypatch, "choreography", submit_then_idle)
    provider = m.create_provider({}, "planner")
    generate = provider.generate

    def too_expensive(request):
        response = generate(request)
        if len(calls) == 6:
            response["costs"][usage_field] = cap
        return response

    provider.generate = too_expensive
    monkeypatch.setattr(m, "create_provider", lambda *a, **k: provider)
    result = run(path, sid, out, max_input_tokens=100000, max_total_output_tokens=150000)
    assert result[usage_field] > cap
    assert not result["full_song_succeeded"] and result["committed_units"] == 1
    assert result["calls_used"] == len(calls) == 6
    assert not (out / "plan.json").exists()
    assert run(path, sid, out, max_input_tokens=100000, max_total_output_tokens=150000, resume=True) == result
    assert len(calls) == 6


def test_v2_infrastructure_unknown_aborts_and_cannot_be_retried(tmp_path, monkeypatch):
    path, sid, out, calls = material(tmp_path, monkeypatch, "choreography")
    provider = m.create_provider({}, "planner")
    generate = provider.generate

    def failed(request):
        response = generate(request)
        if len(calls) == 2:
            response["stop_reason"] = "infrastructure_error"
        return response

    provider.generate = failed
    monkeypatch.setattr(m, "create_provider", lambda *a, **k: provider)
    result = run(path, sid, out)
    assert not result["complete"] and not result["full_song_succeeded"]
    assert result["calls_used"] == len(calls) == 2
    with pytest.raises(ValueError):
        run(path, sid, out, resume=True)
    assert len(calls) == 2


def test_v2_contract_cannot_be_resumed_as_v1_or_replace_missing_raw(tmp_path, monkeypatch):
    path, sid, out, calls = material(tmp_path, monkeypatch, "choreography")
    run(path, sid, out)
    original_count = len(calls)
    with pytest.raises(ValueError):
        m.run_composition(path, sid, {"kind": "replay"}, out, method="choreography", resume=True)
    assert len(calls) == original_count
    next((out / "calls").glob("*.raw.json")).unlink()
    with pytest.raises(ValueError):
        run(path, sid, out, resume=True)
    assert len(calls) == original_count
