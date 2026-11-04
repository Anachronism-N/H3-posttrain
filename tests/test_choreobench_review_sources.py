"""Review source declarations stay private and never authenticate a result."""
import copy
import json
import unittest

from choreobench.editing import validate_edit_task
from choreobench.io import object_sha
from choreobench.reviews import blank_review_template, build_review_pack, import_reviews
from choreobench.schema import ValidationError


def fixture(binding=None):
    song = {"song_id": "review-source-fixture", "split": "dev", "song_group_id": None,
            "artist_group_id": None, "near_duplicate_group_id": None, "brief": "",
            "audio": {"path": "private/audio-fixture.wav", "sha256": "0" * 64,
                      "sample_rate": 16000, "num_samples": 81000},
            "music": {"source": "fixture", "beats": [], "downbeats": [], "sections": [
                {"section_id": "s", "start": 0, "end": 81000, "label": "unknown", "repeat_group": None}]}}
    plan = {"schema": "choreobench.plan.v1", "song_id": song["song_id"], "sample_rate": 16000,
            "num_samples": 81000, "motifs": [], "provenance": {}, "events": [
                {"event_id": "e", "start": 0, "end": 81000, "track": "body", "action": "Step left.",
                 "attributes": {}, "entry": {}, "exit": {}, "motif_id": None, "evidence": []}]}
    result = {"song": song, "plan": plan, "method_id": "source-fixture-method"}
    if binding is not None:
        result["source_binding"] = copy.deepcopy(binding)
    return result


def source_binding(kind="submission", synthetic=False):
    return {"kind": kind, "sha256": "a" * 64, "dataset_sha256": "b" * 64, "synthetic": synthetic}


def imported(bundle):
    template = blank_review_template(bundle["public"], reviewer_id="fixture-declared-reviewer")
    return import_reviews(bundle["public"], bundle["private"], template)


class ReviewSourceTests(unittest.TestCase):
    def test_missing_binding_is_explicitly_unknown(self):
        bundle = build_review_pack([fixture()])
        self.assertNotIn("source_binding_sha256", bundle["private"]["items"][0])
        result = imported(bundle)["items"][0]
        self.assertIsNone(result["source_binding"])
        self.assertEqual(result["scientific_result_status"], "unknown")

    def test_run_and_submission_declarations_remain_unaudited(self):
        for kind in ("run", "submission"):
            with self.subTest(kind=kind):
                binding = source_binding(kind)
                bundle = build_review_pack([fixture(binding)])
                private = bundle["private"]["items"][0]
                self.assertEqual(private["source_binding_sha256"], object_sha(binding))
                self.assertEqual(private["original"]["source_binding"], binding)
                result = imported(bundle)["items"][0]
                self.assertEqual(result["source_binding"], binding)
                self.assertEqual(result["scientific_result_status"], "declared_real_not_audited")
                self.assertEqual(result["status"], "pending")

    def test_synthetic_source_stays_synthetic_after_review_import(self):
        bundle = build_review_pack([fixture(source_binding(synthetic=True))])
        result = imported(bundle)["items"][0]
        self.assertEqual(result["scientific_result_status"], "synthetic")
        self.assertTrue(result["source_binding"]["synthetic"])

    def test_binding_never_changes_or_leaks_into_public_review_pack(self):
        unbound = build_review_pack([fixture()], seed=17)
        bound = build_review_pack([fixture(source_binding())], seed=17)
        self.assertEqual(bound["public"], unbound["public"])
        public = json.dumps(bound["public"])
        self.assertNotIn("source_binding", public)
        self.assertNotIn("a" * 64, public)
        self.assertNotIn("b" * 64, public)
        self.assertNotIn(bound["private"]["items"][0]["source_binding_sha256"], public)

    def test_binding_is_detached_from_callers_and_imported_reports(self):
        item = fixture(source_binding())
        original = copy.deepcopy(item)
        bundle = build_review_pack([item])
        item["source_binding"]["sha256"] = "c" * 64
        self.assertEqual(bundle["private"]["items"][0]["original"], original)
        result = imported(bundle)["items"][0]
        result["source_binding"]["sha256"] = "d" * 64
        self.assertEqual(bundle["private"]["items"][0]["original"], original)

    def test_source_binding_requires_exact_fields_and_finite_typed_identity(self):
        invalid = [None, {}, dict(source_binding(), kind="other"), dict(source_binding(), kind=[]),
                   dict(source_binding(), sha256="short"), dict(source_binding(), sha256="A" * 64),
                   dict(source_binding(), dataset_sha256=None), dict(source_binding(), synthetic=1),
                   dict(source_binding(), extra="not allowed")]
        for binding in invalid:
            with self.subTest(binding=binding), self.assertRaises(ValidationError):
                item = fixture()
                item["source_binding"] = binding
                build_review_pack([item])

    def test_changed_private_source_declaration_is_rejected(self):
        bundle = build_review_pack([fixture(source_binding())])
        bundle["private"]["items"][0]["original"]["source_binding"]["sha256"] = "c" * 64
        with self.assertRaises(ValidationError):
            imported(bundle)

    def test_missing_or_dangling_private_source_hash_is_rejected(self):
        bound = build_review_pack([fixture(source_binding())])
        del bound["private"]["items"][0]["source_binding_sha256"]
        with self.assertRaises(ValidationError):
            imported(bound)
        unbound = build_review_pack([fixture()])
        unbound["private"]["items"][0]["source_binding_sha256"] = "c" * 64
        with self.assertRaises(ValidationError):
            imported(unbound)

    def test_forged_binding_type_is_revalidated_even_if_hash_is_recomputed(self):
        bundle = build_review_pack([fixture(source_binding())])
        row = bundle["private"]["items"][0]
        row["original"]["source_binding"]["synthetic"] = "false"
        row["source_binding_sha256"] = object_sha(row["original"]["source_binding"])
        with self.assertRaises(ValidationError):
            imported(bundle)

    def test_rebound_declarations_do_not_become_authenticated_results(self):
        bundle = build_review_pack([fixture(source_binding())])
        row = bundle["private"]["items"][0]
        row["original"]["source_binding"]["sha256"] = "c" * 64
        row["source_binding_sha256"] = object_sha(row["original"]["source_binding"])
        result = imported(bundle)["items"][0]
        self.assertEqual(result["scientific_result_status"], "declared_real_not_audited")
        self.assertIn("not authenticated", imported(bundle)["note"])

    def test_private_edit_goals_still_do_not_enter_public_source_metadata(self):
        item = fixture(source_binding())
        source = copy.deepcopy(item["plan"])
        task = {"schema": "choreobench.edit_task.v1", "task_id": "fixture-edit", "song_id": source["song_id"],
                "source_plan_sha256": object_sha(source), "instruction": "Change the selected action.",
                "allowed_event_ids": ["e"], "repair_event_ids": [], "protected": [],
                "goals": [{"event_id": "e", "field": "action", "op": "contains", "value": "SECRET_PRIVATE_ANSWER"}],
                "semantic_goals": ["SECRET_PRIVATE_SEMANTIC_GOAL"]}
        validate_edit_task(task, source)
        item["source_plan"], item["edit_task"] = source, task
        item["plan"]["events"][0]["action"] = "A different proposed movement."
        bundle = build_review_pack([item])
        self.assertNotIn("SECRET_PRIVATE", json.dumps(bundle["public"]))
        self.assertNotIn("source_binding", json.dumps(bundle["public"]))
        self.assertEqual(imported(bundle)["items"][0]["scientific_result_status"], "declared_real_not_audited")


if __name__ == "__main__":
    unittest.main()
