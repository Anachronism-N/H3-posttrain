"""Synthetic format/count fixtures only; never actual scores, media, or human verdicts."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms/music_planner_human_review_result.py"
spec = importlib.util.spec_from_file_location("human_review_result_test_subject", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class HumanReviewResultTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pack = self.root / "synthetic_public"
        self.pack.mkdir()
        self.public_path = self.pack / "manifest.json"
        self.private_path = self.root / "synthetic_private_key.json"
        self.scores_path = self.root / "synthetic_scores.json"
        self.out = self.root / "synthetic_report.json"
        self.public = {"schema": "anonymous-offline-video-review-v1", "bundle_id": "0" * 32,
                       "axes": list(m.AXES), "cases": []}
        self.private = {"schema": "private-offline-video-review-key-v1", "bundle_id": "0" * 32,
                        "receipts": [{"path": "/synthetic/private-only/receipt"}], "cases": []}
        for index, alias in enumerate(m.CASES):
            sha = digest(("synthetic audio " + alias).encode())
            public_case = {"case": alias, "original_audio": {"file": alias + "/music.wav", "sha256": sha},
                           "videos": []}
            private_case = {"case": alias, "id": "SYNTHETIC-ORIGINAL-ID-" + alias,
                            "audio_ref": {"path": "/synthetic/private-only/original.wav", "sha256": sha},
                            "original_audio_copy": {"path": str(self.pack / alias / "music.wav"), "sha256": sha},
                            "videos": []}
            for offset, label in enumerate("ABC"):
                sha = digest(("synthetic video " + alias + label).encode())
                public_case["videos"].append({"label": label, "file": alias + "/" + label + ".mp4", "sha256": sha})
                private_case["videos"].append({"label": label, "family": m.FAMILIES[(offset + index) % 3],
                    "video": {"path": "/synthetic/private-only/original.mp4"},
                    "copy": {"path": str(self.pack / alias / (label + ".mp4")), "sha256": sha}})
            self.public["cases"].append(public_case)
            self.private["cases"].append(private_case)
        self.scores = {"schema": "anonymous-offline-human-scores-v1", "bundle_id": "0" * 32,
                       "reviewer": "SYNTHETIC-REVIEWER-DO-NOT-EXPORT", "downloaded_at": "2026-10-25T00:00:00.000Z",
                       "cases": [{"case": alias, "axes": dict(zip(m.AXES, ("A", "B", "C", "similar", "unknown"))),
                                  "reason": "Synthetic format fixture, not an actual review."} for alias in m.CASES]}
        self.write(self.public_path, self.public)
        self.write(self.private_path, self.private)
        constants = {"PUBLIC_MANIFEST": self.public_path, "PRIVATE_KEY": self.private_path,
                     "PUBLIC_MANIFEST_SHA": digest(self.public_path.read_bytes()),
                     "PRIVATE_KEY_SHA": digest(self.private_path.read_bytes()), "BUNDLE_ID": "0" * 32}
        for name, value in constants.items():
            item = patch.object(m, name, value)
            item.start()
            self.addCleanup(item.stop)

    def write(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def refreeze_fixture(self):
        self.write(self.public_path, self.public)
        self.write(self.private_path, self.private)
        m.PUBLIC_MANIFEST_SHA = digest(self.public_path.read_bytes())
        m.PRIVATE_KEY_SHA = digest(self.private_path.read_bytes())

    def run_score(self, scores=None, out=None):
        self.write(self.scores_path, self.scores if scores is None else scores)
        return m.ingest(self.scores_path, self.out if out is None else out)

    def test_complete_three_cases_five_axes_are_only_independent_preferences(self):
        report = self.run_score()
        self.assertTrue(report["human_review_complete"])
        self.assertEqual(report["answered_axis_choices"], 15)
        self.assertEqual(report["expected_axis_choices"], 15)
        self.assertFalse(report["human_quality_accepted"])
        self.assertTrue(report["requires_manual_synthesis"])
        self.assertEqual(report["choice_semantics"], "independent_three_candidate_preferences_not_pairwise_or_accuracy")
        for case, private_case in zip(report["cases"], self.private["cases"]):
            self.assertEqual(set(case["axes"]), set(m.AXES))
            for axis, video in zip(m.AXES[:3], private_case["videos"]):
                self.assertEqual(case["axes"][axis], {"choice": "prefer_named_family", "prefer_named_family": video["family"]})
        for axis in m.AXES:
            counts = report["axis_choice_counts"][axis]
            self.assertEqual(sum(counts["prefer_named_family"].values())
                             + sum(counts[name] for name in ("similar", "all_unusable", "unknown", "unanswered")), 3)
        text = json.dumps(report)
        for key in ("accuracy", "pairwise", "winner", "overall_winner", "ranking", "votes"):
            self.assertNotIn('"' + key + '":', text)
        self.assertEqual(json.loads(self.out.read_text()), report)
        self.assertEqual(self.out.stat().st_mode & 0o777, 0o600)

    def test_unknown_all_unusable_and_similar_remain_explicit_human_choices(self):
        for index, case in enumerate(self.scores["cases"]):
            case["axes"] = dict.fromkeys(m.AXES, ("unknown", "all_unusable", "similar")[index])
        report = self.run_score()
        self.assertTrue(report["human_review_complete"])
        for index, case in enumerate(report["cases"]):
            for axis in m.AXES:
                self.assertEqual(case["axes"][axis], {"choice": ("unknown", "all_unusable", "similar")[index],
                                                       "prefer_named_family": None})
                self.assertEqual(report["axis_choice_counts"][axis][("unknown", "all_unusable", "similar")[index]], 1)
        self.assertFalse(report["human_quality_accepted"])

    def test_partial_blank_and_null_stay_unanswered_not_unknown(self):
        self.scores["cases"][0]["axes"]["clarity"] = ""
        self.scores["cases"][1]["axes"]["continuity"] = None
        report = self.run_score()
        self.assertFalse(report["human_review_complete"])
        self.assertEqual(report["answered_axis_choices"], 13)
        for index, axis in ((0, "clarity"), (1, "continuity")):
            self.assertEqual(report["cases"][index]["axes"][axis], {"choice": None, "prefer_named_family": None})
            self.assertEqual(report["axis_choice_counts"][axis]["unanswered"], 1)
            self.assertEqual(report["axis_choice_counts"][axis]["unknown"], 0)
        self.assertFalse(report["human_quality_accepted"])

    def test_all_null_or_empty_template_refused_even_with_reason(self):
        for empty in (None, ""):
            for case in self.scores["cases"]:
                case["axes"] = dict.fromkeys(m.AXES, empty)
            with self.subTest(empty=empty), self.assertRaisesRegex(ValueError, "unfilled"):
                self.run_score()
            self.assertFalse(self.out.exists())

    def test_exact_cases_and_axes_required_no_missing_duplicate_or_extra(self):
        mutations = []
        scores = copy.deepcopy(self.scores); scores["cases"].pop(); mutations.append(scores)
        scores = copy.deepcopy(self.scores); scores["cases"][2]["case"] = "case1"; mutations.append(scores)
        scores = copy.deepcopy(self.scores); scores["cases"][2]["case"] = "case4"; mutations.append(scores)
        scores = copy.deepcopy(self.scores); scores["cases"][0]["axes"].pop("clarity"); mutations.append(scores)
        scores = copy.deepcopy(self.scores); scores["cases"][0]["axes"]["extra"] = "A"; mutations.append(scores)
        scores = copy.deepcopy(self.scores); scores["cases"][0].pop("reason"); mutations.append(scores)
        for index, scores in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.run_score(scores)
            self.assertFalse(self.out.exists())

    def test_illegal_types_choices_unknown_schema_and_bundle_refused(self):
        for value in (0, 1, True, False, [], {}, "D", "a", " A", "winner"):
            scores = copy.deepcopy(self.scores)
            scores["cases"][0]["axes"]["clarity"] = value
            with self.subTest(choice=value), self.assertRaises(ValueError):
                self.run_score(scores)
        for field, value in (("schema", "wrong"), ("bundle_id", "1" * 32), ("downloaded_at", None),
                             ("downloaded_at", ""), ("reviewer", []), ("cases", {})):
            scores = copy.deepcopy(self.scores); scores[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.run_score(scores)
        for value in (0, [], {}):
            scores = copy.deepcopy(self.scores); scores["cases"][0]["reason"] = value
            with self.subTest(reason=value), self.assertRaises(ValueError):
                self.run_score(scores)
        self.assertFalse(self.out.exists())

    def test_duplicate_json_keys_at_root_and_nested_and_nonfinite_refused(self):
        text = json.dumps(self.scores)
        variants = (text.replace('"schema":', '"schema":"duplicate", "schema":', 1),
                    text.replace('"clarity":', '"clarity":"A", "clarity":', 1),
                    text.replace('"clarity": "A"', '"clarity": NaN', 1))
        for index, variant in enumerate(variants):
            self.scores_path.write_text(variant)
            with self.subTest(index=index), self.assertRaises(ValueError):
                m.ingest(self.scores_path, self.out)
        self.assertFalse(self.out.exists())

    def test_public_private_bundle_schema_cases_and_copy_sha_must_match(self):
        public, private = copy.deepcopy(self.public), copy.deepcopy(self.private)
        def invalid_pair(change):
            self.public, self.private = copy.deepcopy(public), copy.deepcopy(private)
            change()
            self.refreeze_fixture()
            with self.assertRaises(ValueError):
                self.run_score()
            self.assertFalse(self.out.exists())
        invalid_pair(lambda: self.private.update(bundle_id="1" * 32))
        invalid_pair(lambda: self.public.update(bundle_id="1" * 32))
        invalid_pair(lambda: self.private.update(schema="wrong"))
        invalid_pair(lambda: self.public.update(axes=["clarity"]))
        invalid_pair(lambda: self.private["cases"].pop())
        invalid_pair(lambda: self.private["cases"][2].update(case="case1"))
        invalid_pair(lambda: self.private["cases"][0]["videos"][0]["copy"].update(sha256="f" * 64))
        invalid_pair(lambda: self.private["cases"][0]["videos"][0].update(family="invalid"))
        invalid_pair(lambda: self.private["cases"][0]["videos"][0].update(label="B"))
        invalid_pair(lambda: self.private["cases"][0]["audio_ref"].update(sha256="f" * 64))

    def test_frozen_sha_drift_and_duplicate_key_in_private_refused(self):
        for path in (self.public_path, self.private_path):
            before = path.read_bytes()
            path.write_bytes(before + b" ")
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "frozen .* SHA"):
                self.run_score()
            path.write_bytes(before)
        text = self.private_path.read_text()
        self.private_path.write_text(text.replace('"bundle_id":', '"bundle_id":"duplicate", "bundle_id":', 1))
        m.PRIVATE_KEY_SHA = digest(self.private_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.run_score()
        self.assertFalse(self.out.exists())

    def test_public_report_omits_private_id_paths_mapping_and_reviewer_preserves_reason(self):
        self.scores.pop("reviewer")
        self.scores["cases"][0]["reason"] = '<script>not executed</script>; "quoted"; 中文理由'
        self.scores["cases"][1]["reason"] = None
        report = self.run_score()
        text = self.out.read_text()
        for token in ("reviewer", "SYNTHETIC-REVIEWER", str(self.private_path), str(self.root),
                      "/synthetic/private-only", "SYNTHETIC-ORIGINAL-ID", '"mapping"', '"videos"', '"audio_ref"'):
            self.assertNotIn(token, text)
        self.assertEqual(report["source_sha256"], {"scores": digest(self.scores_path.read_bytes()),
                         "public_manifest": digest(self.public_path.read_bytes()), "private_key": digest(self.private_path.read_bytes())})
        self.assertEqual(json.loads(text)["cases"][0]["reason"], self.scores["cases"][0]["reason"])
        self.assertIsNone(report["cases"][1]["reason"])

    def test_reviewer_present_is_never_exported_and_majority_is_not_a_winner(self):
        for case in self.scores["cases"]:
            case["axes"] = dict.fromkeys(m.AXES, "A")
        report = self.run_score()
        self.assertNotIn(self.scores["reviewer"], self.out.read_text())
        self.assertNotIn("reviewer", report)
        self.assertNotIn("winner", report)
        self.assertFalse(report["human_quality_accepted"])
        self.assertTrue(report["requires_manual_synthesis"])

    def test_existing_output_input_and_frozen_public_never_overwritten(self):
        self.out.write_text("KEEP EXISTING")
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            self.run_score()
        self.assertEqual(self.out.read_text(), "KEEP EXISTING")
        self.write(self.scores_path, self.scores)
        original = self.scores_path.read_bytes()
        with self.assertRaises(ValueError):
            m.ingest(self.scores_path, self.scores_path)
        self.assertEqual(self.scores_path.read_bytes(), original)
        destination = self.pack / "new_report.json"
        with self.assertRaisesRegex(ValueError, "frozen public"):
            m.ingest(self.scores_path, destination)
        self.assertFalse(destination.exists())

    def test_symlink_input_output_and_output_creation_race_refused(self):
        self.write(self.scores_path, self.scores)
        linked = self.root / "linked_scores.json"; linked.symlink_to(self.scores_path)
        with self.assertRaises(ValueError):
            m.ingest(linked, self.out)
        linked_out = self.root / "linked_out.json"; linked_out.symlink_to(self.root / "missing")
        with self.assertRaises(ValueError):
            m.ingest(self.scores_path, linked_out)
        real_open = m.os.open
        def race(path, flags, *args):
            if Path(path) == self.out:
                self.out.write_text("RACE KEEP")
            return real_open(path, flags, *args)
        with patch.object(m.os, "open", side_effect=race), self.assertRaises(FileExistsError):
            m.ingest(self.scores_path, self.out)
        self.assertEqual(self.out.read_text(), "RACE KEEP")

    def test_each_input_sha_is_rechecked_before_any_output(self):
        self.write(self.scores_path, self.scores)
        real_read = m.read_snapshot
        for changing in (self.scores_path, self.public_path, self.private_path):
            before = changing.read_bytes()
            reads = {}
            def change(path):
                reads[path] = reads.get(path, 0) + 1
                if path == changing and reads[path] == 2:
                    changing.write_bytes(before + b" ")
                return real_read(path)
            with self.subTest(input=changing.name), patch.object(m, "read_snapshot", side_effect=change), \
                 self.assertRaisesRegex(ValueError, "snapshot SHA changed"):
                m.ingest(self.scores_path, self.out)
            self.assertFalse(self.out.exists())
            changing.write_bytes(before)

    def test_main_gate_error_does_not_echo_private_paths_or_choices(self):
        with patch.object(m.sys, "argv", [str(SCRIPT), "--scores", str(self.scores_path), "--out", str(self.out)]), \
             patch.object(m, "ingest", side_effect=ValueError(str(self.private_path))), \
             patch.object(m.sys, "stderr", new_callable=io.StringIO) as err:
            self.assertEqual(m.main(), 1)
            self.assertNotIn(str(self.private_path), err.getvalue())
            self.assertIn("No human quality verdict inferred", err.getvalue())


if __name__ == "__main__":
    unittest.main()
