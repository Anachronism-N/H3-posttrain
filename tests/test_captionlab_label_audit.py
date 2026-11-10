"""Standard-library CPU probes, not caption-quality or model evaluations."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from captionlab import label_audit


def record(identifier="sample", split="train", group="source", caption=None):
    caption = "One wrist moves left." if caption is None else caption
    return dict(id=identifier, split=split, source_group_id=group,
                caption_version="original-version", motion_caption=caption)


class LabelAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, rows, name="input.jsonl"):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def audit(self, rows):
        return label_audit.audit_labels([self.write(rows)])

    def test_empty_header_only_and_empty_windows_are_not_approved(self):
        rows = [record(str(index), caption=caption) for index, caption in enumerate((
            "", "Dance style: contemporary", " Dance style: contemporary\n[0-2s]\n[2-5.0625s]"))]
        report = self.audit(rows)
        self.assertEqual(report["body_character_status_counts"], {"empty_body": 3})
        self.assertEqual(report["whole_caption_proxies"]["texts"], 3)
        self.assertIsNone(report["whole_caption_proxies"]["action_word_hits_per_1000_words"])
        self.assertTrue(report["audit_complete"])
        self.assertFalse(report["quality_verified"])
        self.assertIsNone(report["training_approved"])
        self.assertIsNone(report["selected_training_records"])
        self.assertNotIn("tiers", report)

    def test_character_ranges_do_not_pretend_to_identify_english_or_quality(self):
        report = self.audit([record("cn", caption="一侧手腕向左移动。"),
                             record("mixed", caption="Turn and 左移。"),
                             record("latin", caption="Bonjour le monde."),
                             record("digits", caption="123 !")])
        self.assertEqual(report["body_character_status_counts"], dict(non_latin_letters_only=1,
            mixed_latin_and_non_latin_letters=1, latin_letters_only=1, no_alphabetic_characters=1))
        self.assertIn("not verified English", report["definitions"]["character_status"])

    def test_one_wrist_and_one_side_do_not_create_quality_tiers(self):
        report = self.audit([record("one", caption="One wrist moves left."),
                             record("side", caption="One side wrist moves left.")])
        self.assertEqual(report["whole_caption_proxies"]["action_word_hits"], 2)
        self.assertEqual(report["body_character_status_counts"], {"latin_letters_only": 2})
        self.assertNotIn("A_narrative", json.dumps(report))

    def test_per_thousand_density_is_correct_for_short_text(self):
        proxy = self.audit([record(caption="Turn and lift.")])["whole_caption_proxies"]
        self.assertEqual(proxy["word_count"], 3)
        self.assertEqual(proxy["action_word_hits"], 2)
        self.assertAlmostEqual(proxy["action_word_hits_per_1000_words"], 2000 / 3)

    def test_repeated_actions_within_and_across_windows_are_separate_proxies(self):
        row = record(caption="Dance style: unknown\n[0-2s] One wrist moves left.\n[2-5s] One wrist moves left.")
        row["windows"] = [dict(start=0, end=2, caption="One wrist moves left."),
                          dict(start=2, end=5, caption="One wrist moves left.")]
        report = self.audit([row])
        self.assertEqual(report["whole_caption_proxies"]["pooled_sentence_duplicate_rate"], .5)
        self.assertEqual(report["structured_window_proxies"]["pooled_sentence_duplicate_rate"], 0)
        self.assertEqual(report["structured_window_proxies"]["texts"], 2)
        self.assertEqual(report["counts"]["cross_window_duplicate_sentence_excess"], 1)
        self.assertEqual(report["counts"]["structured_windows_available_records"], 1)
        self.assertIn("Real periodic actions", " ".join(report["limitations"]))

    def test_sentence_matching_preserves_case_and_punctuation(self):
        report = self.audit([record(caption="Turn. turn. Turn;\nTurn.")])
        proxy = report["whole_caption_proxies"]
        self.assertEqual(proxy["sentence_count"], 4)
        self.assertEqual(proxy["duplicate_sentence_excess"], 1)
        self.assertEqual(proxy["pooled_sentence_duplicate_rate"], .25)

    def test_rep8_stays_within_each_text(self):
        body = "one two three four five six seven eight"
        report = self.audit([record("one", caption=body), record("two", caption=body)])
        self.assertEqual(report["whole_caption_proxies"]["eight_gram_count"], 2)
        self.assertEqual(report["whole_caption_proxies"]["pooled_rep8"], 0)
        repeated = self.audit([record(caption=body + " " + body)])
        self.assertEqual(repeated["whole_caption_proxies"]["duplicate_eight_gram_excess"], 1)
        self.assertAlmostEqual(repeated["whole_caption_proxies"]["pooled_rep8"], 1 / 9)

    def test_duplicate_ids_and_cross_split_groups_across_files(self):
        one = self.write([record("x", "train", "g")], "train.jsonl")
        two = self.write([record("x", "val", "g"), record("y", "val", "h"), record("x", "val", "h")], "val.jsonl")
        counts = label_audit.audit_labels([one, two])["counts"]
        self.assertEqual(counts["line_count"], 4)
        self.assertEqual(counts["unique_ids"], 2)
        self.assertEqual(counts["duplicate_id_occurrences"], 2)
        self.assertEqual(counts["duplicate_distinct_ids"], 1)
        self.assertEqual(counts["ids_spanning_multiple_declared_splits"], 1)
        self.assertEqual(counts["source_groups_spanning_multiple_declared_splits"], 1)

    def test_missing_fields_and_invalid_types_are_counted_without_aliases(self):
        report = self.audit([{}, dict(id=1, split=[], source_group_id="", motion_caption=5, windows={}),
                             dict(id="alias", source_group="not-an-alias", motion_caption="Move.")])
        counts = report["counts"]
        self.assertEqual(counts["json_object_records"], 3)
        self.assertEqual((counts["missing_id"], counts["invalid_id"]), (1, 1))
        self.assertEqual((counts["missing_split"], counts["invalid_split"]), (2, 1))
        self.assertEqual((counts["missing_source_group_id"], counts["invalid_source_group_id"]), (2, 1))
        self.assertEqual(counts["invalid_motion_caption_records"], 2)
        self.assertEqual(counts["malformed_structured_windows_records"], 1)
        self.assertEqual(report["whole_caption_proxies"]["texts"], 1)

    def test_bad_lines_remain_in_denominator_and_input_hash(self):
        path = self.write([record()])
        suffix = b'not json\n\n[]\n{"id":"duplicate","id":"key"}\n{"value":1e999}\n{"value":NaN}\n\xff\n'
        with path.open("ab") as stream:
            stream.write(suffix)
        raw = path.read_bytes()
        report = label_audit.audit_labels([path]); counts = report["counts"]
        self.assertEqual((counts["line_count"], counts["json_object_records"]), (8, 1))
        self.assertEqual((counts["invalid_json_lines"], counts["non_object_json_lines"], counts["invalid_utf8_lines"]), (5, 1, 1))
        self.assertEqual(report["inputs"][0]["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["inputs"][0]["size_bytes"], len(raw))
        self.assertEqual(len(report["error_line_examples"]), 7)

    def test_no_inferred_windows_and_original_version_counts(self):
        report = self.audit([record(caption="[0-2s] Turn.\n[2-5s] Lift.")])
        self.assertEqual(report["structured_window_proxies"]["texts"], 0)
        self.assertEqual(report["counts"]["structured_windows_unavailable_records"], 1)
        self.assertEqual(report["caption_version_counts"], {"original-version": 1})
        self.assertEqual(report["whole_caption_proxies"]["sentence_count"], 2)
        self.assertEqual(report["counts"]["missing_caption_version"], 0)

    def test_explicit_empty_and_non_latin_windows_keep_their_denominators(self):
        row = record(caption="Turn.")
        row["windows"] = [dict(caption=""), dict(caption="左手上移。")]
        report = self.audit([row])
        self.assertEqual(report["structured_window_proxies"]["texts"], 2)
        self.assertEqual(report["structured_window_character_status_counts"],
                         dict(empty_body=1, non_latin_letters_only=1))

    def test_escaped_lone_surrogate_is_a_counted_bad_record(self):
        path = self.root / "surrogate.jsonl"
        path.write_bytes(b'{"split":"\\ud800"}\n')
        report = label_audit.audit_labels([path])
        self.assertEqual(report["counts"]["invalid_json_lines"], 1)
        self.assertEqual(report["counts"]["json_object_records"], 0)

    def test_error_examples_are_bounded_and_input_is_not_loaded_whole(self):
        path = self.root / "bad.jsonl"; path.write_bytes(b"bad\n" * 50)
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("whole-file load")), \
             mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file load")):
            report = label_audit.audit_labels([path])
        self.assertEqual(report["counts"]["invalid_json_lines"], 50)
        self.assertEqual(len(report["error_line_examples"]), 20)

    def test_empty_input_is_a_complete_audit_not_valid_training_data(self):
        path = self.root / "empty.jsonl"; path.write_bytes(b"")
        report = label_audit.audit_labels([path])
        self.assertEqual(report["counts"]["line_count"], 0)
        self.assertEqual(report["inputs"][0]["sha256"], hashlib.sha256(b"").hexdigest())
        self.assertIsNone(report["whole_caption_proxies"]["pooled_rep8"])
        self.assertIsNone(report["training_approved"])

    def test_input_mutation_during_scan_is_rejected(self):
        path = self.write([record()]); original = label_audit._body; changed = False
        def body(text):
            nonlocal changed
            if not changed:
                changed = True
                with path.open("ab") as stream:
                    stream.write(b" \n")
            return original(text)
        with mock.patch.object(label_audit, "_body", side_effect=body):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                label_audit.audit_labels([path])

    def test_duplicate_input_path_is_rejected(self):
        path = self.write([record()])
        with self.assertRaisesRegex(ValueError, "Duplicate input path"):
            label_audit.audit_labels([path, path])

    def test_cli_is_utf8_exclusive_and_aggregates_without_caption_text(self):
        path = self.write([record(caption="Unmistakable source caption to keep private.")])
        out = self.root / "reports/audit.json"
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            label_audit.main(["--input", str(path), "--out", str(out)])
        report_text = out.read_text(encoding="utf-8")
        self.assertNotIn("Unmistakable source caption", report_text)
        self.assertEqual(json.loads(report_text)["inputs"][0]["sha256"], source_hash)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source_hash)
        previous = out.read_bytes()
        with mock.patch.object(label_audit, "audit_labels", side_effect=AssertionError("must not scan")):
            with self.assertRaises(FileExistsError):
                label_audit.main(["--input", str(path), "--out", str(out)])
        self.assertEqual(out.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
