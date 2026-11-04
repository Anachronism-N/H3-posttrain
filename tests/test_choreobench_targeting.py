"""Deterministic authoring fixtures, not real benchmark samples or results."""

import copy
import json
import unittest

from choreobench.editing import public_edit_task, validate_edit_task
from choreobench.io import object_sha
from choreobench.schema import ValidationError, validate_plan
from choreobench.targeting import TargetResolutionError, build_edit_task, resolve_targets


def event(event_id, start, end, track="body"):
    return {"event_id": event_id, "start": start, "end": end, "track": track,
            "action": "An explicitly authored fixture action.", "attributes": {"side": "left"},
            "entry": {}, "exit": {}, "motif_id": None, "evidence": []}


def fixture():
    song = {"song_id": "fixture-song", "split": "dev", "song_group_id": None,
            "artist_group_id": None, "near_duplicate_group_id": None, "brief": "",
            "audio": {"path": "fixture.wav", "sha256": "0" * 64,
                      "sample_rate": 16000, "num_samples": 120000},
            "music": {"source": "synthetic fixture only", "beats": [], "downbeats": [], "sections": [
                {"section_id": "verse", "start": 0, "end": 40000, "label": "verse", "repeat_group": None},
                {"section_id": "chorus-1", "start": 40000, "end": 80000, "label": "chorus", "repeat_group": "chorus"},
                {"section_id": "chorus-2", "start": 80000, "end": 120000, "label": "chorus", "repeat_group": "chorus"}]}}
    events = [event("verse-body", 0, 40000), event("c1-arm", 40000, 80000, "left_arm"),
              event("c1-torso", 40000, 80000, "torso"), event("c1-leg", 40000, 80000, "right_leg"),
              event("c2-upper", 80000, 120000, "upper"), event("c2-lower", 80000, 120000, "lower")]
    events.sort(key=lambda row: (row["start"], row["track"], row["event_id"]))
    plan = {"schema": "choreobench.plan.v1", "song_id": song["song_id"], "sample_rate": 16000,
            "num_samples": 120000, "motifs": [], "events": events, "provenance": {"fixture": True}}
    validate_plan(plan, song)
    return plan, song


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.song = fixture()

    def test_section_id_is_exact_and_does_not_select_returning_chorus(self):
        result = resolve_targets(self.plan, self.song, {"section_id": "chorus-1"})
        self.assertEqual(result["allowed_event_ids"], ["c1-arm", "c1-leg", "c1-torso"])
        self.assertEqual((result["selection"]["start"], result["selection"]["end"]), (40000, 80000))
        self.assertTrue(result["can_build_task"])

    def test_label_occurrence_is_one_based_in_section_time_order(self):
        result = resolve_targets(self.plan, self.song, {"section_label": "chorus", "occurrence": 2})
        self.assertEqual(result["selection"]["section_id"], "chorus-2")
        self.assertEqual(result["allowed_event_ids"], ["c2-lower", "c2-upper"])

    def test_body_scope_upper_includes_only_descendants(self):
        result = resolve_targets(self.plan, self.song, {"section_id": "chorus-1", "body_scope": "upper"})
        self.assertEqual(result["allowed_event_ids"], ["c1-arm", "c1-torso"])
        all_body = resolve_targets(self.plan, self.song, {"section_id": "chorus-2", "body_scope": "body"})
        self.assertEqual(all_body["allowed_event_ids"], ["c2-lower", "c2-upper"])

    def test_narrow_arm_does_not_authorize_whole_body_event(self):
        selector = {"section_id": "verse", "body_scope": "left_arm"}
        with self.assertRaises(TargetResolutionError) as caught:
            resolve_targets(self.plan, self.song, selector)
        self.assertEqual(caught.exception.code, "cross_body_scope")
        self.assertIn("outside", str(caught.exception))
        preview = resolve_targets(self.plan, self.song, selector, strict=False)
        self.assertEqual(preview["allowed_event_ids"], [])
        self.assertEqual(preview["cross_body_scope_event_ids"], ["verse-body"])
        self.assertFalse(preview["can_build_task"])

    def test_exact_track_differs_from_hierarchical_body_scope(self):
        result = resolve_targets(self.plan, self.song, {"section_id": "chorus-1", "track": "left_arm"})
        self.assertEqual(result["allowed_event_ids"], ["c1-arm"])
        with self.assertRaises(TargetResolutionError) as caught:
            resolve_targets(self.plan, self.song, {"section_id": "chorus-1", "track": "upper"})
        self.assertEqual(caught.exception.code, "no_matches")

    def test_integer_interval_obeys_half_open_boundaries(self):
        result = resolve_targets(self.plan, self.song, {"start": 40000, "end": 80000})
        self.assertEqual(result["matched_event_ids"], ["c1-arm", "c1-leg", "c1-torso"])
        self.assertEqual(result["cross_boundary_event_ids"], [])

    def test_crossing_event_preview_keeps_original_and_intersection_times(self):
        selector = {"start": 50000, "end": 70000, "track": "left_arm"}
        with self.assertRaises(TargetResolutionError) as caught:
            resolve_targets(self.plan, self.song, selector)
        row = caught.exception.preview["matched_events"][0]
        self.assertEqual((row["start"], row["end"]), (40000, 80000))
        self.assertEqual(row["overlap"], {"start": 50000, "end": 70000})
        self.assertEqual(row["temporal_relation"], "spans_interval")
        self.assertFalse(row["targetable"])

    def test_both_crossing_directions_and_contained_events_are_auditable(self):
        self.plan["events"] = [event("a", 0, 10000), event("b", 10000, 20000), event("c", 20000, 30000)]
        result = resolve_targets(self.plan, self.song, {"start": 5000, "end": 25000}, strict=False)
        self.assertEqual([row["temporal_relation"] for row in result["matched_events"]],
                         ["crosses_start", "contained", "crosses_end"])
        self.assertEqual(result["allowed_event_ids"], ["b"])
        self.assertFalse(result["can_build_task"])
        with self.assertRaises(TargetResolutionError):
            resolve_targets(self.plan, self.song, {"start": 5000, "end": 25000})

    def test_combined_temporal_and_body_conflict_has_both_explanations(self):
        with self.assertRaises(TargetResolutionError) as caught:
            resolve_targets(self.plan, self.song, {"start": 1000, "end": 2000, "body_scope": "left_arm"})
        self.assertEqual(caught.exception.code, "cross_boundary_and_scope")
        self.assertEqual(caught.exception.preview["cross_boundary_event_ids"], ["verse-body"])
        self.assertEqual(caught.exception.preview["cross_body_scope_event_ids"], ["verse-body"])

    def test_no_matches_is_an_error_even_for_preview_mode(self):
        self.plan["events"] = [event("a", 0, 10000)]
        for strict in (False, True):
            with self.subTest(strict=strict), self.assertRaises(TargetResolutionError) as caught:
                resolve_targets(self.plan, self.song, {"start": 10000, "end": 20000}, strict=strict)
            self.assertEqual(caught.exception.code, "no_matches")
            self.assertEqual(caught.exception.preview["matched_event_ids"], [])

    def test_unknown_section_and_label_do_not_infer_hidden_structure(self):
        for selector in ({"section_id": "missing"}, {"section_label": "bridge", "occurrence": 1},
                         {"section_label": "CHORUS", "occurrence": 1},
                         {"section_label": "chorus", "occurrence": 3}):
            with self.subTest(selector=selector), self.assertRaises(ValidationError):
                resolve_targets(self.plan, self.song, selector)

    def test_explicit_unknown_section_can_be_targeted_without_becoming_a_chorus(self):
        self.song["music"]["sections"][0]["label"] = "unknown"
        result = resolve_targets(self.plan, self.song, {"section_label": "unknown", "occurrence": 1})
        self.assertEqual(result["selection"]["section_label"], "unknown")
        self.assertEqual(result["allowed_event_ids"], ["verse-body"])

    def test_missing_ambiguous_extra_and_natural_language_selectors_fail(self):
        invalid = [None, "the second chorus", {}, {"section_label": "chorus"},
                   {"section_id": "verse", "start": 0, "end": 40000},
                   {"start": 0}, {"occurrence": 1},
                   {"section_id": "verse", "query": "left hand"},
                   {"section_id": "verse", "track": "body", "body_scope": "body"}]
        for selector in invalid:
            with self.subTest(selector=selector), self.assertRaises(ValidationError):
                resolve_targets(self.plan, self.song, selector)

    def test_occurrence_requires_positive_integer_not_boolean_or_float(self):
        for value in (0, -1, True, 1.0, None, "2"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                resolve_targets(self.plan, self.song, {"section_label": "chorus", "occurrence": value})

    def test_samples_and_scopes_are_strictly_validated(self):
        invalid = [{"start": False, "end": 40000}, {"start": 0.0, "end": 40000},
                   {"start": -1, "end": 40000}, {"start": 0, "end": 120001},
                   {"start": 40000, "end": 40000}, {"start": 0, "end": True},
                   {"section_id": "verse", "body_scope": []},
                   {"section_id": "verse", "body_scope": None},
                   {"section_id": "verse", "track": "fingers"}]
        for selector in invalid:
            with self.subTest(selector=selector), self.assertRaises(ValidationError):
                resolve_targets(self.plan, self.song, selector)
        with self.assertRaises(ValidationError):
            resolve_targets(self.plan, self.song, {"section_id": "verse"}, strict=0)

    def test_source_hashes_and_detached_preview(self):
        old_plan, old_song = copy.deepcopy(self.plan), copy.deepcopy(self.song)
        selector = {"section_id": "chorus-1"}
        result = resolve_targets(self.plan, self.song, selector)
        self.assertEqual(result["source_plan_sha256"], object_sha(self.plan))
        self.assertEqual(result["song_sha256"], object_sha(self.song))
        self.assertEqual(result["audio_sha256"], self.song["audio"]["sha256"])
        result["matched_events"][0]["action"] = "changed preview"
        result["selector"]["section_id"] = "changed preview"
        self.assertEqual(self.plan, old_plan)
        self.assertEqual(self.song, old_song)
        self.assertEqual(selector, {"section_id": "chorus-1"})

    def test_failed_resolution_also_does_not_mutate_sources(self):
        old_plan, old_song = copy.deepcopy(self.plan), copy.deepcopy(self.song)
        with self.assertRaises(TargetResolutionError):
            resolve_targets(self.plan, self.song, {"start": 1, "end": 20})
        self.assertEqual(self.plan, old_plan)
        self.assertEqual(self.song, old_song)

    def test_plan_must_bind_the_validated_song_and_clock(self):
        for key, value in (("song_id", "different"), ("sample_rate", 24000), ("num_samples", 120001)):
            invalid = copy.deepcopy(self.plan)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                resolve_targets(invalid, self.song, {"section_id": "verse"})


class TaskAuthoringTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.song = fixture()

    def build(self, selector=None, **kwargs):
        return build_edit_task(self.plan, self.song, selector or {"section_id": "chorus-1", "track": "left_arm"},
                               task_id="fixture-edit", instruction="Change the selected arm action.", **kwargs)

    def test_build_matches_existing_private_and_public_task_contracts(self):
        task = self.build()
        self.assertIs(validate_edit_task(task, self.plan), task)
        self.assertEqual(task["allowed_event_ids"], ["c1-arm"])
        self.assertEqual(task["repair_event_ids"], [])
        self.assertEqual(task["source_plan_sha256"], object_sha(self.plan))
        self.assertEqual(task["goals"], [])
        self.assertNotIn("selector", task)
        public = public_edit_task(task)
        self.assertNotIn("goals", public)
        self.assertNotIn("semantic_goals", public)

    def test_private_constraints_are_detached_and_never_appear_in_preview(self):
        goals = [{"event_id": "c1-arm", "field": "action", "op": "contains", "value": "SECRET_PRIVATE_ANSWER"}]
        semantic = ["SECRET_PRIVATE_SEMANTIC_GOAL"]
        protected = [{"event_id": "*", "field": "start"}]
        repairs = ["verse-body"]
        task = self.build(goals=goals, semantic_goals=semantic, protected=protected, repair_event_ids=repairs)
        preview = resolve_targets(self.plan, self.song, {"section_id": "chorus-1", "track": "left_arm"})
        self.assertNotIn("SECRET_PRIVATE", json.dumps(preview))
        self.assertNotIn("SECRET_PRIVATE", json.dumps(public_edit_task(task)))
        goals[0]["value"] = "mutated"
        semantic.append("mutated")
        protected[0]["field"] = "end"
        repairs.clear()
        self.assertEqual(task["goals"][0]["value"], "SECRET_PRIVATE_ANSWER")
        self.assertEqual(len(task["semantic_goals"]), 1)
        self.assertEqual(task["protected"][0]["field"], "start")
        self.assertEqual(task["repair_event_ids"], ["verse-body"])

    def test_repair_events_are_explicit_known_and_disjoint(self):
        task = self.build(repair_event_ids=["verse-body"])
        self.assertEqual(task["repair_event_ids"], ["verse-body"])
        for repairs in (["missing"], ["c1-arm"], ["verse-body", "verse-body"], "verse-body"):
            with self.subTest(repairs=repairs), self.assertRaises(ValidationError):
                self.build(repair_event_ids=repairs)

    def test_private_goal_and_protection_validation_is_reused(self):
        for kwargs in ({"protected": [{"event_id": "missing", "field": "action"}]},
                       {"goals": [{"event_id": "missing", "field": "action", "op": "eq", "value": "x"}]},
                       {"semantic_goals": "not a list"},
                       {"goals": [{"event_id": "c1-arm", "field": "start", "op": "gte", "value": True}]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                self.build(**kwargs)

    def test_preview_mode_cannot_authorize_a_cross_boundary_task(self):
        selector = {"start": 50000, "end": 70000, "track": "left_arm"}
        self.assertFalse(resolve_targets(self.plan, self.song, selector, strict=False)["can_build_task"])
        with self.assertRaises(TargetResolutionError):
            self.build(selector, repair_event_ids=["c1-arm"])

    def test_build_has_no_automatic_natural_language_or_input_mutation(self):
        original_plan, original_song = copy.deepcopy(self.plan), copy.deepcopy(self.song)
        task = build_edit_task(self.plan, self.song, {"section_id": "verse"},
                               task_id="fixture-not-understanding", instruction="Ambiguous creative instruction.")
        self.assertEqual(task["allowed_event_ids"], ["verse-body"])
        self.assertEqual(task["goals"], [])
        self.assertEqual(task["semantic_goals"], [])
        self.assertEqual(self.plan, original_plan)
        self.assertEqual(self.song, original_song)


if __name__ == "__main__":
    unittest.main()
