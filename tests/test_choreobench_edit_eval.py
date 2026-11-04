"""Negative contracts for editing, private goals, clustered evaluation and reviews."""

import copy
import json

import pytest

from choreobench.editing import (
    apply_patch, apply_task_patch, public_edit_task, validate_edit_task,
)
from choreobench.evaluation import evaluate_edit, evaluate_plan, song_bootstrap
from choreobench.io import object_sha
from choreobench.reviews import blank_review_template, build_review_pack, import_reviews
from choreobench.schema import ValidationError


def song():
    return {
        "song_id": "original_song", "split": "test", "song_group_id": "group_song",
        "artist_group_id": "group_artist", "near_duplicate_group_id": "group_recording",
        "brief": "Develop a dance motif, then finish calmly.",
        "audio": {"path": "private/source_audio.wav", "sha256": "a" * 64,
                  "sample_rate": 100, "num_samples": 1000},
        "music": {"source": "private_structure_tool", "beats": [0, 250, 500, 750],
                  "downbeats": [0, 500], "sections": [
                      {"section_id": "original_section", "start": 0, "end": 1000,
                       "label": "unknown", "repeat_group": None}]},
    }


def plan():
    def event(event_id, start, end, action):
        return {"event_id": event_id, "start": start, "end": end, "track": "body",
                "action": action, "attributes": {"side": "right", "amplitude": 1.0, "enabled": True},
                "entry": {"facing": "front"}, "exit": {"facing": "front"},
                "motif_id": "original_motif", "evidence": ["private_call:77"]}
    return {"schema": "choreobench.plan.v1", "song_id": "original_song",
            "sample_rate": 100, "num_samples": 1000,
            "motifs": [{"motif_id": "original_motif", "description": "A broad arm sweep."}],
            "events": [event("event_a", 0, 500, "Wave the right arm."),
                       event("event_b", 500, 1000, "Return to the starting posture.")],
            "provenance": {"method_id": "secret_method_alpha", "candidate_file": "private/answer.json"}}


def task(source=None):
    source = plan() if source is None else source
    return {"schema": "choreobench.edit_task.v1", "task_id": "original_task",
            "song_id": source["song_id"], "source_plan_sha256": object_sha(source),
            "instruction": "Change the first arm action to the left; preserve the exit facing.",
            "allowed_event_ids": ["event_a"], "repair_event_ids": [],
            "protected": [{"event_id": "*", "field": "exit"}],
            "goals": [{"event_id": "event_a", "field": "attributes.side", "op": "eq", "value": "left"}],
            "semantic_goals": ["PRIVATE_SEMANTIC_CHECK: the arm instruction is coherent and the feet are preserved."]}


def patch(source=None):
    source = plan() if source is None else source
    contract = task(source)
    return {"schema": "choreobench.patch.v1", "base_plan_sha256": object_sha(source),
            "instruction": contract["instruction"], "allowed_event_ids": ["event_a"],
            "repair_event_ids": [], "protected": copy.deepcopy(contract["protected"]),
            "operations": [{"event_id": "event_a", "field": "attributes.side", "value": "left"}]}


def test_patch_is_transactional_and_records_exact_binding():
    before, proposal = plan(), patch()
    snapshot = copy.deepcopy((before, proposal))
    after = apply_patch(before, proposal)
    assert (before, proposal) == snapshot
    assert after["events"][0]["attributes"]["side"] == "left"
    assert after["provenance"]["method_id"] == before["provenance"]["method_id"]
    assert after["provenance"]["choreobench_edit"] == {
        "base_plan_sha256": object_sha(before), "patch_sha256": object_sha(proposal)}
    after["events"][0]["entry"]["facing"] = "away"
    assert before == snapshot[0]


def test_failed_second_operation_leaves_all_inputs_unchanged():
    before, proposal = plan(), patch()
    proposal["operations"].append({"event_id": "event_b", "field": "action", "value": "Unexpected change."})
    snapshot = copy.deepcopy((before, proposal))
    with pytest.raises(ValidationError):
        apply_patch(before, proposal)
    assert (before, proposal) == snapshot


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(base_plan_sha256="0" * 64),
    lambda p: p.update(private_answer="do not silently ignore"),
    lambda p: p["operations"][0].update(extra="do not silently ignore"),
    lambda p: p["operations"][0].update(field="song_id", value="other"),
    lambda p: p["operations"][0].update(field="attributes.new_key", value="new"),
    lambda p: p["operations"][0].update(field="attributes", value={}),
    lambda p: p["operations"][0].update(field="action.__class__", value="unsafe"),
    lambda p: p["operations"].append(copy.deepcopy(p["operations"][0])),
    lambda p: p.update(allowed_event_ids=["event_a", "event_a"]),
    lambda p: p.update(repair_event_ids=["unknown_event"]),
])
def test_strict_patch_rejects_bad_bindings_paths_and_ambiguous_operations(mutation):
    proposal = patch()
    mutation(proposal)
    with pytest.raises(ValidationError):
        apply_patch(plan(), proposal)


@pytest.mark.parametrize("field,value", [
    ("start", False), ("start", 0.5), ("start", {}), ("end", 1001),
    ("end", 0), ("track", []), ("track", "finger"), ("motif_id", []),
    ("motif_id", "unknown_motif"), ("action", ""), ("attributes.side", {}),
    ("attributes.amplitude", float("nan")),
])
def test_patch_rejects_clock_type_and_schema_violations(field, value):
    proposal = patch()
    proposal["operations"] = [{"event_id": "event_a", "field": field, "value": value}]
    with pytest.raises(ValidationError):
        apply_patch(plan(), proposal)


def test_wildcard_whole_object_protection_rejects_child_change():
    proposal = patch()
    proposal["protected"] = [{"event_id": "*", "field": "attributes"}]
    with pytest.raises(ValidationError, match="protected"):
        apply_patch(plan(), proposal)


def test_noop_is_kept_but_does_not_meet_an_unmet_target():
    source, proposal = plan(), patch()
    proposal["operations"] = []
    after = apply_patch(source, proposal)
    report = evaluate_edit(source, after, task(source))
    assert report["no_op"] and not report["literal_joint_success"]
    assert report["literal_goal_satisfaction_rate"] == 0
    assert report["overall_status"] == "literal_fail"


def test_protected_noop_does_not_raise_and_time_reordering_is_canonical():
    source, proposal = plan(), patch()
    proposal["operations"] = [{"event_id": "event_a", "field": "exit.facing", "value": "front"}]
    assert apply_patch(source, proposal)["events"] == source["events"]
    proposal["allowed_event_ids"] = ["event_a", "event_b"]
    proposal["operations"] = [{"event_id": event_id, "field": field, "value": value}
                              for event_id, field, value in [("event_a", "start", 500), ("event_a", "end", 1000),
                                                             ("event_b", "start", 0), ("event_b", "end", 500)]]
    assert [event["event_id"] for event in apply_patch(source, proposal)["events"]] == ["event_b", "event_a"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(allowed_event_ids=["event_a", "event_b"]),
    lambda p: p.update(repair_event_ids=["event_b"]),
    lambda p: p.update(protected=[]),
    lambda p: p.update(instruction="A new unrelated instruction."),
])
def test_model_cannot_redefine_task_authority(mutation):
    proposal = patch()
    mutation(proposal)
    with pytest.raises(ValidationError, match="frozen"):
        apply_task_patch(plan(), proposal, task())


def test_public_task_never_contains_private_answers_and_is_detached():
    private = task()
    private["goals"][0]["value"] = "PRIVATE_LITERAL_ANSWER"
    view = public_edit_task(private)
    serialized = json.dumps(view)
    assert "PRIVATE_LITERAL_ANSWER" not in serialized
    assert "PRIVATE_SEMANTIC_CHECK" not in serialized
    assert "goals" not in view and "semantic_goals" not in view
    view["allowed_event_ids"].append("new")
    assert private["allowed_event_ids"] == ["event_a"]


def test_task_rejects_wrong_source_and_unsupported_goal_operator():
    wrong = task()
    wrong["source_plan_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        validate_edit_task(wrong, plan())
    wrong = task()
    wrong["goals"][0]["op"] = "python_eval"
    with pytest.raises(ValidationError):
        validate_edit_task(wrong)


def test_literal_pass_still_has_pending_semantics_and_no_music_score():
    source = plan()
    after = apply_task_patch(source, patch(source), task(source))
    report = evaluate_edit(source, after, task(source))
    assert report["literal_joint_success"]
    assert report["overall_status"] == "pending_independent_review"
    assert report["semantic_goals"][0]["status"] == "pending_independent_review"
    assert report["music_alignment"]["status"] == "not_measured"
    assert report["physical_quality"]["status"] == "not_measured"


def test_contains_check_is_literal_and_never_claims_semantic_truth():
    source, contract = plan(), task()
    contract["goals"] = [{"event_id": "event_a", "field": "action", "op": "contains", "value": "wave"}]
    after = copy.deepcopy(source)
    after["events"][0]["action"] = "Do not wave."
    report = evaluate_edit(source, after, contract)
    assert report["literal_goals"][0]["satisfied"]
    assert report["overall_status"] == "pending_independent_review"


def test_boolean_does_not_satisfy_numeric_one_and_empty_protections_are_na():
    source, contract = plan(), task()
    contract["goals"] = [{"event_id": "event_a", "field": "attributes.enabled", "op": "eq", "value": 1}]
    contract["protected"] = []
    report = evaluate_edit(source, source, contract)
    assert not report["literal_goals"][0]["satisfied"]
    assert report["protection_preservation_rate"] is None


@pytest.mark.parametrize("field,value", [("song_id", "other_song"), ("sample_rate", 200)])
def test_evaluate_edit_catches_immutable_changes_in_valid_external_plans(field, value):
    source, after = plan(), plan()
    after[field] = value
    report = evaluate_edit(source, after, task(source))
    assert report["after_valid"] and not report["literal_joint_success"]
    assert field in report["changed_immutable_fields"]


def test_external_scope_and_evidence_changes_cannot_receive_joint_success():
    source, after = plan(), plan()
    after["events"][1]["action"] = "Unrequested whole-body change."
    report = evaluate_edit(source, after, task(source))
    assert report["out_of_scope_event_ids"] == ["event_b"]
    after = plan()
    after["events"][0]["attributes"]["side"] = "left"
    after["events"][0]["evidence"] = ["fabricated_source"]
    report = evaluate_edit(source, after, task(source))
    assert not report["literal_joint_success"]
    assert report["changed_evidence_event_ids"] == ["event_a"]


def test_invalid_after_plan_gets_no_partial_success():
    source, after = plan(), plan()
    after["events"][0]["end"] = 700
    report = evaluate_edit(source, after, task(source))
    assert not report["after_valid"] and not report["literal_joint_success"]
    assert report["literal_goal_satisfaction_rate"] is None


def test_plan_coverage_counts_parallel_tracks_once_and_reports_gaps():
    source = plan()
    source["events"][0].update(track="left_arm", start=0, end=400)
    source["events"][1].update(track="right_arm", start=0, end=400)
    source["events"].sort(key=lambda e: (e["start"], e["track"], e["event_id"]))
    report = evaluate_plan(source, song())
    assert report["valid"]
    assert report["coverage"]["covered_samples"] == 400
    assert report["coverage"]["gaps"] == [[400, 1000]]
    assert report["music_alignment"] == {"status": "not_measured"}


def test_invalid_plan_never_gets_coverage_score():
    source = plan()
    source["num_samples"] = 1001
    report = evaluate_plan(source, song())
    assert not report["valid"] and report["coverage"] is None


def test_song_bootstrap_weights_songs_not_number_of_windows():
    rows = [{"song_id": "many_windows", "value": 0.0}] * 100 + [{"song_id": "one_window", "value": 10.0}]
    report = song_bootstrap(rows, n_resamples=100, seed=41)
    assert report["estimate"] == 5 and report["n_songs"] == 2
    assert report == song_bootstrap(rows, n_resamples=100, seed=41)


@pytest.mark.parametrize("rows", [[], [{"song_id": "only_one_song", "value": 0.9}]])
def test_bootstrap_insufficient_songs_does_not_invent_confidence(rows):
    report = song_bootstrap(rows)
    assert report["interval"] is None and report["status"] == "insufficient_songs"


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), "0.5"])
def test_bootstrap_cannot_silently_drop_or_coerce_missing_values(bad):
    with pytest.raises(ValidationError):
        song_bootstrap([{"song_id": "s", "value": bad}])


def test_finite_large_values_do_not_produce_nonfinite_bootstrap_json():
    report = song_bootstrap([{"song_id": "s1", "value": 1e308},
                             {"song_id": "s2", "value": 1e308}], n_resamples=20)
    assert report["estimate"] == 1e308
    assert report["interval"] == [1e308, 1e308]
    json.dumps(report, allow_nan=False)


def review_bundle(edit=True):
    original = plan()
    item = {"song": song(), "plan": original, "method_id": "secret_method_alpha"}
    if edit:
        item["source_plan"] = original
        item["edit_task"] = task(original)
        item["plan"] = apply_task_patch(original, patch(original), item["edit_task"])
    return build_review_pack([item], seed=13)


def review_form(bundle, origin="human"):
    return blank_review_template(bundle["public"], reviewer_id="rater_1", origin=origin,
                                 model_id="independent_model_v1" if origin == "model_judge" else None)


def test_blind_pack_strips_methods_paths_provenance_and_private_answers():
    original = {"song": song(), "plan": plan(), "method_id": "secret_method_alpha",
                "source_plan": plan(), "edit_task": task()}
    original["edit_task"]["goals"][0]["value"] = "PRIVATE_LITERAL_ANSWER"
    snapshot = copy.deepcopy(original)
    bundle = build_review_pack([original])
    serialized = json.dumps(bundle["public"])
    for secret in ("secret_method_alpha", "private/", "private_call", "original_song",
                   "PRIVATE_LITERAL_ANSWER", "PRIVATE_SEMANTIC_CHECK", "candidate_file"):
        assert secret not in serialized
    assert original == snapshot
    public_item = bundle["public"]["items"][0]
    assert public_item["song"]["audio"]["path"] == "assets/" + "a" * 64 + ".wav"
    assert bundle["private"]["items"][0]["audio_source_path"] == "private/source_audio.wav"


def test_visible_method_name_in_action_fails_instead_of_silently_rewriting_it():
    item = {"song": song(), "plan": plan(), "method_id": "secret_method_alpha"}
    item["plan"]["events"][0]["action"] += " secret_method_alpha"
    with pytest.raises(ValidationError, match="method identifier"):
        build_review_pack([item])


def test_missing_reviews_and_null_forms_remain_pending():
    bundle = review_bundle()
    empty = {"schema": "choreobench.reviews.v1", "pack_sha256": object_sha(bundle["public"]), "reviews": []}
    report = import_reviews(bundle["public"], bundle["private"], empty)
    assert report["pending_item_count"] == 1 and report["items"][0]["status"] == "pending"
    report = import_reviews(bundle["public"], bundle["private"], review_form(bundle))
    assert report["items"][0]["human_review_count"] == 0


def test_partial_review_does_not_claim_music_or_full_completion():
    bundle = review_bundle()
    form = review_form(bundle)
    form["reviews"][0]["ratings"]["long_range_structure"] = 4
    report = import_reviews(bundle["public"], bundle["private"], form)
    assert report["items"][0]["status"] == "partial"
    assert report["items"][0]["criteria_status"]["musical_fit"] == "pending"
    assert report["pending_item_count"] == 1


def test_full_model_review_does_not_count_as_human_review():
    bundle = review_bundle()
    form = review_form(bundle, "model_judge")
    form["reviews"][0]["audio_seen"] = True
    form["reviews"][0]["ratings"] = {key: 4 for key in form["reviews"][0]["ratings"]}
    report = import_reviews(bundle["public"], bundle["private"], form)
    item = report["items"][0]
    assert item["status"] == "reviewed" and item["model_judge_status"] == "reviewed"
    assert item["human_status"] == "pending" and item["human_review_count"] == 0


@pytest.mark.parametrize("mutation", [
    lambda form: form.update(pack_sha256="0" * 64),
    lambda form: form["reviews"][0].update(item_sha256="0" * 64),
    lambda form: form["reviews"][0].update(review_id="unknown"),
    lambda form: form["reviews"][0].update(origin="assistant_delegated"),
    lambda form: form["reviews"][0].update(model_id="model_mislabeled_as_human"),
    lambda form: form["reviews"][0].update(audio_seen="true"),
    lambda form: form["reviews"][0]["ratings"].update(musical_fit=4),
    lambda form: form["reviews"][0]["ratings"].update(long_range_structure=True),
    lambda form: form["reviews"][0]["ratings"].update(long_range_structure=6),
    lambda form: form["reviews"][0]["ratings"].update(motion_sync=5),
    lambda form: form["reviews"].append(copy.deepcopy(form["reviews"][0])),
])
def test_review_import_rejects_misbinding_false_human_claims_and_bad_ratings(mutation):
    bundle = review_bundle()
    form = review_form(bundle)
    mutation(form)
    with pytest.raises(ValidationError):
        import_reviews(bundle["public"], bundle["private"], form)


def test_rehashing_tampered_public_content_cannot_rebind_review_to_original_plan():
    bundle = review_bundle()
    public_item = bundle["public"]["items"][0]
    public_item["plan"]["events"][0]["action"] = "An easier, unrelated plan."
    public_item["item_sha256"] = object_sha({key: value for key, value in public_item.items() if key != "item_sha256"})
    bundle["private"]["items"][0]["item_sha256"] = public_item["item_sha256"]
    bundle["private"]["public_pack_sha256"] = object_sha(bundle["public"])
    with pytest.raises(ValidationError, match="original plan"):
        import_reviews(bundle["public"], bundle["private"], review_form(bundle))


def test_private_original_tampering_is_rejected():
    bundle = review_bundle()
    bundle["private"]["items"][0]["original"]["plan"]["events"][0]["action"] = "Changed."
    with pytest.raises(ValidationError, match="hash mismatch"):
        import_reviews(bundle["public"], bundle["private"], review_form(bundle))


def test_non_edit_pack_rejects_edit_rating_and_marks_edit_fields_na():
    bundle = review_bundle(edit=False)
    form = review_form(bundle)
    form["reviews"][0]["ratings"]["edit_target_satisfaction"] = 4
    with pytest.raises(ValidationError, match="edit task"):
        import_reviews(bundle["public"], bundle["private"], form)
    form["reviews"][0]["ratings"]["edit_target_satisfaction"] = None
    report = import_reviews(bundle["public"], bundle["private"], form)
    assert report["items"][0]["criteria_status"]["edit_target_satisfaction"] == "not_applicable"
