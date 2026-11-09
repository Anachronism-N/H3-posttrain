"""Revision protocol negatives: frozen sources, matched controls and edit scopes."""

import copy
import json
import random

import pytest

from choreobench.editing import apply_task_patch, public_edit_task, validate_edit_task
from choreobench.evaluation import evaluate_edit
from choreobench.io import object_sha
from choreobench.revision_protocol import (
    DIAGNOSIS_SCHEMA, VERDICT_SCHEMA, RANDOM_REVIEW_REASON,
    action_task, parse_diagnosis, parse_verdict, public_plan, select_events,
)
from choreobench.schema import ValidationError, identifier, validate_plan


def source():
    ids = ["z-first", "a-second", "x-third", "b-fourth", "q-fifth", "c-last"]
    return {"schema": "choreobench.plan.v1", "song_id": "revision-fixture", "sample_rate": 100,
            "num_samples": 6000,
            "motifs": [{"motif_id": "m", "description": "A returning sway."}],
            "provenance": {"method_id": "PRIVATE_METHOD", "private_path": "/private/artifact.json",
                           "nested": {"score": 100, "evaluation_goal": "PRIVATE_TARGET"}},
            "events": [{"event_id": event_id, "start": i * 1000, "end": (i + 1) * 1000,
                        "track": "body", "action": "Sway gently.", "motif_id": "m",
                        "attributes": {"amplitude": 1, "side": "left"},
                        "entry": {"facing": "front"}, "exit": {"facing": "front"},
                        "evidence": ["PRIVATE_CALL_" + str(i)]} for i, event_id in enumerate(ids)]}


def diagnosis(plan=None, issues=None):
    plan = source() if plan is None else plan
    return {"schema": DIAGNOSIS_SCHEMA, "source_plan_sha256": object_sha(plan),
            "issues": [{"event_id": "a-second", "priority": 2, "reason": "Concern A"},
                       {"event_id": "q-fifth", "priority": 3, "reason": "Concern Q"},
                       {"event_id": "z-first", "priority": 2, "reason": "Concern Z"}]
                      if issues is None else issues}


def patch(task, operations):
    return {"schema": "choreobench.patch.v1", "base_plan_sha256": task["source_plan_sha256"],
            "instruction": task["instruction"], "allowed_event_ids": list(task["allowed_event_ids"]),
            "repair_event_ids": list(task["repair_event_ids"]), "protected": copy.deepcopy(task["protected"]),
            "operations": operations}


def verdict(before, after, event_id="z-first", decision="prefer_revision"):
    return {"schema": VERDICT_SCHEMA, "before_plan_sha256": object_sha(before),
            "after_plan_sha256": object_sha(after), "event_id": event_id,
            "decision": decision, "reason": "Internal model preference, not independent quality."}


def test_diagnosis_strict_binds_original_plan_and_does_not_mutate():
    plan = source(); value = diagnosis(plan); snapshot = copy.deepcopy((plan, value))
    parsed = parse_diagnosis(text=json.dumps(value), plan=plan)
    assert parsed == value and (plan, value) == snapshot
    parsed["issues"][0]["reason"] = "Changed"
    assert value["issues"][0]["reason"] == "Concern A"


def test_targeted_set_uses_priority_ties_by_source_order_then_executes_in_source_order():
    plan = source(); value = diagnosis(plan); before = copy.deepcopy(value)
    selected = select_events(plan, value, "targeted", 3, 0)
    assert [row["event_id"] for row in selected] == ["z-first", "a-second", "q-fifth"]
    assert [row["priority"] for row in selected] == [2, 2, 3]
    assert [row["event_id"] for row in select_events(plan, value, "targeted", 2, 999)] == ["z-first", "q-fifth"]
    selected[0]["reason"] = "Changed"
    assert value == before


def test_random_matched_count_uses_all_events_and_never_diagnosis_reasons():
    plan = source(); value = diagnosis(plan)
    selected = select_events(plan, value, "random", 2, 7)
    assert len(selected) == len(select_events(plan, value, "targeted", 2, 7)) == 2
    expected = sorted(random.Random(7).sample(range(len(plan["events"])), 2))
    assert [row["event_id"] for row in selected] == [plan["events"][i]["event_id"] for i in expected]
    assert all(row["priority"] is None and row["reason"] == RANDOM_REVIEW_REASON for row in selected)
    assert select_events(plan, value, "random", 2, 7) == selected
    # For this seed, at least one selected event was never diagnosed.
    assert any(row["event_id"] not in {issue["event_id"] for issue in value["issues"]} for row in selected)
    assert "Concern" not in json.dumps(selected)


def test_random_locations_ignore_diagnosed_identity_priority_and_reason():
    plan = source(); one = diagnosis(plan)
    two = diagnosis(plan, [{"event_id": plan["events"][i]["event_id"], "priority": 1,
                            "reason": "OTHER_PRIVATE_DIAGNOSIS"} for i in (2, 3, 5)])
    assert select_events(plan, one, "random", 3, 22) == select_events(plan, two, "random", 3, 22)


def test_selections_are_frozen_to_source_not_a_changed_intermediate_draft():
    plan = source(); value = diagnosis(plan)
    changed = copy.deepcopy(plan); changed["events"][0]["action"] = "Step instead."
    with pytest.raises(ValidationError, match="frozen source plan"):
        select_events(changed, value, "targeted", 2, 7)
    with pytest.raises(ValidationError, match="frozen source plan"):
        select_events(changed, value, "random", 2, 7)


@pytest.mark.parametrize("policy", ["targeted", "random"])
def test_empty_diagnosis_yields_no_invented_repairs(policy):
    plan = source(); value = diagnosis(plan, [])
    assert parse_diagnosis(json.dumps(value), plan)["issues"] == []
    assert select_events(plan, value, policy, 8, 0) == []


def test_selected_budget_cannot_exceed_issue_count_and_global_rng_is_unchanged():
    plan = source(); value = diagnosis(plan)
    state = random.getstate()
    assert len(select_events(plan, value, "random", 8, 2 ** 63 - 1)) == 3
    assert len(select_events(plan, value, "targeted", 8, 0)) == 3
    assert random.getstate() == state


@pytest.mark.parametrize("change", [
    lambda value: value.update(schema="other"),
    lambda value: value.update(source_plan_sha256="0" * 64),
    lambda value: value.update(extra="PRIVATE_GOAL"),
    lambda value: value.pop("issues"),
    lambda value: value.update(issues=None),
    lambda value: value.update(issues=value["issues"] * 3),
    lambda value: value["issues"].append(copy.deepcopy(value["issues"][0])),
    lambda value: value["issues"][0].update(event_id="unknown"),
    lambda value: value["issues"][0].update(event_id=[]),
    lambda value: value["issues"][0].update(priority=True),
    lambda value: value["issues"][0].update(priority=0),
    lambda value: value["issues"][0].update(priority=4),
    lambda value: value["issues"][0].update(priority=2.0),
    lambda value: value["issues"][0].update(priority="2"),
    lambda value: value["issues"][0].update(reason=""),
    lambda value: value["issues"][0].update(reason="  "),
    lambda value: value["issues"][0].update(reason=None),
    lambda value: value["issues"][0].update(reason="x" * 2001),
    lambda value: value["issues"][0].update(reason="\ud800"),
    lambda value: value["issues"][0].update(new_goal="private"),
    lambda value: value["issues"][0].pop("priority"),
])
def test_invalid_diagnosis_rejected_by_text_and_direct_dict_selection(change):
    plan = source(); value = diagnosis(plan); change(value)
    with pytest.raises(ValueError):
        parse_diagnosis(json.dumps(value), plan)
    with pytest.raises(ValueError):
        select_events(plan, value, "targeted", 2, 7)


@pytest.mark.parametrize("value", ["```json\n{}\n```", "{} commentary", "[]", "null", "1",
                                   '{"schema":"a","schema":"b"}',
                                   '{"issues":NaN}', '{"issues":1e999}'])
def test_diagnosis_uses_strict_json_without_repair(value):
    with pytest.raises(ValueError):
        parse_diagnosis(value, source())


@pytest.mark.parametrize("options", [
    {"policy": "unknown"}, {"policy": []}, {"max_repairs": True}, {"max_repairs": 0},
    {"max_repairs": 9}, {"max_repairs": 1.0}, {"seed": True}, {"seed": -1},
    {"seed": 2 ** 63}, {"seed": 1.0},
])
def test_selection_options_are_strict(options):
    arguments = dict(policy="targeted", max_repairs=2, seed=7); arguments.update(options)
    plan = source()
    with pytest.raises(ValidationError):
        select_events(plan, diagnosis(plan), **arguments)


def test_public_plan_drops_private_provenance_and_evidence_and_is_deeply_detached():
    plan = source(); snapshot = copy.deepcopy(plan)
    public = public_plan(plan)
    validate_plan(public)
    assert public["provenance"] == {} and all(event["evidence"] == [] for event in public["events"])
    assert "PRIVATE" not in json.dumps(public) and plan == snapshot
    public["events"][0]["attributes"]["amplitude"] = 999
    public["motifs"][0]["description"] = "Changed"
    assert plan == snapshot


def test_action_task_preserves_private_fact_fields_and_permits_only_action():
    plan = source(); snapshot = copy.deepcopy(plan)
    task = action_task(plan, "z-first", "Make this action smaller; preserve all other fields.")
    validate_edit_task(task, plan); identifier(task["task_id"], "task_id")
    assert task["allowed_event_ids"] == ["z-first"] and task["repair_event_ids"] == []
    assert task["goals"] == [] and task["semantic_goals"] == []
    assert {row["field"] for row in task["protected"]} == {
        "start", "end", "track", "motif_id", "attributes", "entry", "exit"}
    assert all(row["event_id"] == "*" for row in task["protected"])
    operation = {"event_id": "z-first", "field": "action", "value": "A small sway."}
    after = apply_task_patch(plan, patch(task, [operation]), task)
    assert plan == snapshot and after["events"][0]["action"] == "A small sway."
    for before_event, after_event in zip(plan["events"], after["events"]):
        for field in before_event:
            if (before_event["event_id"], field) != ("z-first", "action"):
                assert after_event[field] == before_event[field]
    assert evaluate_edit(plan, after, task)["overall_status"] == "not_evaluable"
    assert "PRIVATE" not in json.dumps(public_edit_task(task))


@pytest.mark.parametrize("field,value", [
    ("start", 1), ("end", 999), ("track", "upper"), ("motif_id", None),
    ("attributes.side", "right"), ("entry.facing", "back"), ("exit.facing", "back"),
    ("evidence", []), ("event_id", "renamed"),
])
def test_action_scope_rejects_any_other_field(field, value):
    plan = source(); task = action_task(plan, "z-first", "Revise only action.")
    with pytest.raises(ValidationError):
        apply_task_patch(plan, patch(task, [{"event_id": "z-first", "field": field, "value": value}]), task)


def test_action_scope_cannot_expand_or_rewrite_task_and_noop_remains_legal():
    plan = source(); task = action_task(plan, "z-first", "Revise only action.")
    proposed = patch(task, [{"event_id": "a-second", "field": "action", "value": "Changed."}])
    with pytest.raises(ValidationError, match="outside"):
        apply_task_patch(plan, proposed, task)
    proposed["allowed_event_ids"].append("a-second")
    with pytest.raises(ValidationError, match="frozen"):
        apply_task_patch(plan, proposed, task)
    noop = apply_task_patch(plan, patch(task, []), task)
    assert noop["events"] == plan["events"]
    assert evaluate_edit(plan, noop, task)["overall_status"] == "not_evaluable"


def test_task_id_is_safe_stable_and_bound_to_plan_event_and_instruction():
    plan = source(); task = action_task(plan, "z-first", "A / ../ unicode 动作 instruction")
    assert task == action_task(copy.deepcopy(plan), "z-first", task["instruction"])
    assert "/" not in task["task_id"] and len(task["task_id"]) <= 128
    assert task["task_id"] != action_task(plan, "a-second", task["instruction"])["task_id"]
    assert task["task_id"] != action_task(plan, "z-first", "Different")["task_id"]
    changed = copy.deepcopy(plan); changed["events"][1]["action"] = "Changed."
    assert task["task_id"] != action_task(changed, "z-first", task["instruction"])["task_id"]


@pytest.mark.parametrize("event_id,instruction", [("unknown", "Review"), ([], "Review"),
                                                 ("z-first", ""), ("z-first", None)])
def test_invalid_action_task_rejected(event_id, instruction):
    with pytest.raises(ValidationError):
        action_task(source(), event_id, instruction)


@pytest.mark.parametrize("decision", ["prefer_revision", "keep_original", "uncertain"])
def test_exact_verdict_is_an_internal_choice_only(decision):
    before = source(); after = copy.deepcopy(before); after["events"][0]["action"] = "A smaller sway."
    snapshot = copy.deepcopy((before, after)); value = verdict(before, after, decision=decision)
    parsed = parse_verdict(text=json.dumps(value), before=before, after=after, event_id="z-first")
    assert parsed == value and (before, after) == snapshot
    assert set(parsed) == {"schema", "before_plan_sha256", "after_plan_sha256", "event_id", "decision", "reason"}
    assert "quality" not in parsed and "score" not in parsed


@pytest.mark.parametrize("change", [
    lambda value: value.update(schema="other"),
    lambda value: value.update(before_plan_sha256="0" * 64),
    lambda value: value.update(after_plan_sha256="0" * 64),
    lambda value: value.update(event_id="a-second"),
    lambda value: value.update(decision="pass"),
    lambda value: value.update(decision=True),
    lambda value: value.update(decision=[]),
    lambda value: value.update(reason=""),
    lambda value: value.update(reason=None),
    lambda value: value.update(reason="x" * 2001),
    lambda value: value.update(quality_score=5),
    lambda value: value.pop("reason"),
])
def test_verdict_bad_binding_fields_and_fake_scores_rejected(change):
    before = source(); after = copy.deepcopy(before)
    value = verdict(before, after); change(value)
    with pytest.raises(ValueError):
        parse_verdict(json.dumps(value), before, after, "z-first")


def test_verdict_cannot_mix_other_song_clock_missing_event_or_duplicate_json_keys():
    before = source(); after = copy.deepcopy(before)
    with pytest.raises(ValidationError, match="both bound plans"):
        parse_verdict(json.dumps(verdict(before, after, "unknown")), before, after, "unknown")
    after["song_id"] = "another-song"
    with pytest.raises(ValidationError, match="same song and clock"):
        parse_verdict(json.dumps(verdict(before, after)), before, after, "z-first")
    raw = json.dumps(verdict(before, before))[:-1] + ',"decision":"keep_original"}'
    with pytest.raises(ValueError, match="Duplicate"):
        parse_verdict(raw, before, before, "z-first")


def test_parsers_reject_invalid_source_plan_before_using_model_fields():
    before = source(); invalid = copy.deepcopy(before); invalid["events"][0]["end"] = -1
    with pytest.raises(ValidationError):
        parse_diagnosis(json.dumps(diagnosis(before)), invalid)
    with pytest.raises(ValidationError):
        public_plan(invalid)
    with pytest.raises(ValidationError):
        parse_verdict(json.dumps(verdict(before, before)), before, invalid, "z-first")
