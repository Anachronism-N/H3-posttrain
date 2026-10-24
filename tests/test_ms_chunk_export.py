"""Synthetic terminal shards test exact unions, receipts and independent A/B."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms"
sys.path.insert(0, str(MS))
import ms_chunk_caption as producer
import ms_chunk_export as exporter
from test_motionscript_v2_cli import write_fixture
from test_ms_chunk_caption import video_info


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nlf, self.source, self.template = write_fixture(self.root)
        self.media = self.root / "media"
        self.media.mkdir()
        self.video, self.audio = self.media / "synthetic.mp4", self.media / "synthetic.wav"
        self.video.write_bytes(b"synthetic video fixture, not real media")
        self.audio.write_bytes(b"synthetic audio fixture")
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        (self.upstream / "captioning.py").write_text("# synthetic source hash fixture\n")
        self.selection = self.root / "selection.jsonl"
        self.rows = [{"id": sid, "nlf_path": str(self.source), "video_path": str(self.video),
                      "audio_path": str(self.audio), "source_group": "synthetic-source", "split": "train",
                      "source_record": {"synthetic": True}, "normalized_audio_offset_seconds": "unknown"}
                     for sid in ("a", "b", "c", "d")]
        self.write_selection()
        self.run_root = self.root / "run"
        self.run_root.mkdir()
        self.blank = False
        self.args = exporter.parser().parse_args(["--run-root", str(self.run_root), "--out", str(self.root / "export")])
        self.make_shards()

    def write_selection(self):
        self.selection.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def fake_captioner(self, arrays, src, shim):
        valid = arrays["valid"] & arrays["joint_valid"].all(axis=1)
        segments = []
        first = None
        for index in range(len(valid) + 1):
            current = index < len(valid) and valid[index]
            boundary = index and index < len(valid) and (arrays["segment_id"][index] != arrays["segment_id"][index-1]
                        or not np.array_equal(arrays["joint_segment_id"][index], arrays["joint_segment_id"][index-1]))
            if first is not None and (not current or boundary):
                if index-first >= 4:
                    segments.append({"start": float(arrays["time"][first]), "end": float(arrays["time"][index-1]),
                        "source": "original_motionscript_rules_on_v2_coordinates", "ms": ["" if self.blank else "She raises her left arm twice. A second later she lowers her arm."],
                        "ms_nonagg": ["synthetic raw nonaggregated text"]})
                first = None
            if current and first is None:
                first = index
        return segments

    def make_shards(self, *, create=True):
        jobs = []
        for shard in range(2):
            argv = [sys.executable, str(MS / "ms_chunk_caption.py"), "--manifest", str(self.selection),
                    "--template", str(self.template), "--upstream-src", str(self.upstream),
                    "--out", str(self.run_root / f"shard{shard}"), "--workers", "1", "--shard", str(shard), "--nshard", "2"]
            jobs.append({"shard": shard, "host": "synthetic-cpu", "argv": argv})
            if create:
                args = producer.parser().parse_args(argv[2:])
                def init(context):
                    producer._WORKER = context
                with mock.patch.object(producer, "init_worker", side_effect=init), mock.patch.object(producer, "original_captions", side_effect=self.fake_captioner), mock.patch.object(producer, "probe_video", side_effect=lambda *a, **kw: video_info()), contextlib.redirect_stdout(io.StringIO()):
                    producer.run(args)
        launch = {"input_manifest": str(self.selection), "input_manifest_sha256": exporter.digest(self.selection.read_bytes()),
                  "source_sha256": exporter.digest((MS / "ms_chunk_caption.py").read_bytes()),
                  "target_ids": 4, "shards": 2, "actual_workers": 2, "semantic_approval": False,
                  "human_review_completed": False, "jobs": jobs, "synthetic_fixture": True}
        (self.run_root / "launch.json").write_bytes(producer.json_bytes(launch))

    def read(self, name, shard=0):
        return json.loads((self.run_root / f"shard{shard}" / name).read_bytes())

    def replace_report(self, sid, transform):
        shard = ("a", "b", "c", "d").index(sid) % 2
        directory = self.run_root / f"shard{shard}"
        path = directory / f"{sid}.json"
        value = json.loads(path.read_bytes())
        transform(value)
        content = producer.json_bytes(value)
        path.write_bytes(content)
        (directory / f"{sid}.receipt.json").write_bytes(producer.json_bytes({"report_sha256": exporter.digest(content)}))
        manifest = self.read("manifest.json", shard)
        for record in manifest["records"]:
            if record["id"] == sid:
                record["report_sha256"] = exporter.digest(content)
        (directory / "manifest.json").write_bytes(producer.json_bytes(manifest))
        self.reseal_terminal(shard)

    def reseal_terminal(self, shard):
        directory = self.run_root / f"shard{shard}"
        (directory / "manifest.receipt.json").write_bytes(producer.json_bytes({
            "manifest.json_sha256": exporter.digest((directory / "manifest.json").read_bytes()),
            "chunks.jsonl_sha256": exporter.digest((directory / "chunks.jsonl").read_bytes())}))

    def export(self):
        return exporter.run(self.args)

    def test_exact_union_independent_A_B_and_completed_resume(self):
        frozen = exporter.digest((MS / "ms_chunk_caption.py").read_bytes())
        result = self.export()
        self.assertTrue(result["complete"])
        self.assertEqual(result["unique_id_count"], 4)
        self.assertEqual(result["successfully_processed_count"], 4)
        self.assertEqual(result["B_window_status_counts"], {"candidate": 8})
        self.assertEqual(result["A_clip_status_counts"], {"candidate": 4})
        self.assertEqual(result["caption_coverage"]["A_raw_original_nonempty_clip_count"], 4)
        self.assertEqual(result["caption_coverage"]["B_two_windows_both_candidate_clip_count"], 4)
        self.assertEqual(result["caption_coverage"]["B_incomplete_candidate_coverage_clip_count"], 0)
        self.assertFalse(result["formal_approval"])
        self.assertFalse(result["accepted"])
        out = Path(self.args.out)
        captions = [json.loads(line) for line in (out / "captions.jsonl").read_text().splitlines()]
        self.assertEqual([row["id"] for row in captions], ["a", "b", "c", "d"])
        self.assertEqual(captions[0]["A_whole_clip"]["mode"], "A_original_rules_whole_clip_same_coordinates")
        self.assertTrue(captions[0]["B_windows"][0]["candidate_text_eligible"])
        self.assertFalse(captions[0]["B_windows"][0]["formal_training_eligible"])
        self.assertEqual(captions[0]["B_windows"][0]["window_end"], 2)
        self.assertIn("twice", captions[0]["B_windows"][0]["motion_caption"])
        self.assertEqual(captions[0]["B_complete_clip_motion_caption"], "\n".join(window["motion_caption"] for window in captions[0]["B_windows"]))
        self.assertEqual((out / "errors.jsonl").read_bytes(), b"")
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        self.args.resume = True
        self.assertEqual(self.export(), result)
        self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})
        self.assertEqual(frozen, exporter.digest((MS / "ms_chunk_caption.py").read_bytes()))

    def test_no_original_media_or_raw_NLF_read_by_exporter(self):
        original = Path.read_bytes
        def read_bytes(path):
            if path in (self.source, self.video, self.audio):
                raise AssertionError("postverify must not re-read original media or raw NLF")
            return original(path)
        with mock.patch.object(Path, "read_bytes", read_bytes):
            self.assertTrue(self.export()["complete"])

    def test_running_lock_is_pending_without_coords_reads_or_output(self):
        (self.run_root / "shard0" / ".run.lock").write_text("synthetic active producer")
        original = exporter.read_artifact
        def read(path, **kwargs):
            if path.name.endswith(".coords.npz"):
                raise AssertionError("must not read running producer coords")
            return original(path, **kwargs)
        with mock.patch.object(exporter, "read_artifact", side_effect=read):
            result = self.export()
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["complete"])
        self.assertFalse(result["exported"])
        self.assertFalse(Path(self.args.out).exists())
        self.assertTrue((self.run_root / "shard0" / ".run.lock").exists())

    def test_missing_terminal_manifest_or_resume_rejection_is_pending(self):
        rejected = self.run_root / "shard1" / "resume.errors.json"
        rejected.write_text("synthetic prior rejected cache")
        result = self.export()
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["exported"])

    def test_check_only_full_verification_creates_no_export(self):
        self.args.check_only = True
        result = self.export()
        self.assertTrue(result["complete"])
        self.assertFalse(result["exported"])
        self.assertFalse(Path(self.args.out).exists())

    def test_tampered_per_report_coords_or_aggregate_receipt_rejected(self):
        for shard, name in ((0, "a.json"), (0, "a.coords.npz"), (1, "chunks.jsonl"), (1, "manifest.json")):
            with self.subTest(name=name):
                path = self.run_root / f"shard{shard}" / name
                before = path.read_bytes()
                path.write_bytes(before + b"altered")
                with self.assertRaisesRegex(ValueError, "altered"):
                    self.export()
                self.assertFalse(Path(self.args.out).exists())
                path.write_bytes(before)

    def test_resealed_report_false_approval_sourcegroup_and_added_action_refused(self):
        def approved(report):
            report["semantic_approval"] = True
        def wrong_group(report):
            report["source_group"] = "invented-group"
        def invented(report):
            report["chunks"][0]["cleaned_motion_text"] += " The dancer jumps."
        for transform, message in ((approved, "approval"), (wrong_group, "source_group"), (invented, "candidate text")):
            with self.subTest(message=message):
                # Independent fixture per negative avoids resealed tampering from
                # an earlier subcase masking the subsequent intended gate.
                original = {p: p.read_bytes() for d in (self.run_root / "shard0",) for p in d.iterdir()}
                self.replace_report("a", transform)
                with self.assertRaisesRegex(ValueError, message):
                    self.export()
                for path, content in original.items():
                    path.write_bytes(content)

    def test_exact_expected_union_rejects_duplicate_missing_wrong_split(self):
        path = self.run_root / "shard1" / "manifest.json"
        manifest = self.read("manifest.json", 1)
        manifest["records"][1]["id"] = "b"
        path.write_bytes(producer.json_bytes(manifest))
        self.reseal_terminal(1)
        with self.assertRaisesRegex(ValueError, "exact expected IDs"):
            self.export()

    def test_changed_input_manifest_or_frozen_source_bind_refused(self):
        before = self.selection.read_bytes()
        self.rows[0]["split"] = "test"
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "manifest SHA/count"):
            self.export()
        self.selection.write_bytes(before)
        path = self.run_root / "launch.json"
        launch = json.loads(path.read_bytes())
        launch["source_sha256"] = "0" * 64
        path.write_bytes(producer.json_bytes(launch))
        with self.assertRaisesRegex(ValueError, "frozen producer"):
            self.export()

    def test_resealed_different_shard_config_or_upstream_context_refused(self):
        path = self.run_root / "shard1" / "protocol.json"
        before = path.read_bytes()
        marker = json.loads(before)
        marker["protocol"]["config"]["workers"] = 99
        marker["fingerprint"] = exporter.digest(producer.json_bytes(marker["protocol"]))
        path.write_bytes(producer.json_bytes(marker))
        with self.assertRaisesRegex(ValueError, "config differs"):
            self.export()
        path.write_bytes(before)
        marker = json.loads(before)
        marker["protocol"]["upstream_sha256"]["source/captioning.py"] = "1" * 64
        marker["fingerprint"] = exporter.digest(producer.json_bytes(marker["protocol"]))
        path.write_bytes(producer.json_bytes(marker))
        with self.assertRaisesRegex(ValueError, "contexts differ"):
            self.export()

    def test_raw_no_output_is_exported_null_never_static_or_A_fallback(self):
        self.args.out = str(self.root / "blank_export")
        self.run_root = self.root / "blank_run"
        self.run_root.mkdir()
        self.args.run_root = str(self.run_root)
        self.blank = True
        self.make_shards()
        result = self.export()
        self.assertTrue(result["complete"])
        self.assertEqual(result["B_window_abstain_reasons"], {"no_output": 8})
        rows = [json.loads(line) for line in (Path(self.args.out) / "captions.jsonl").read_text().splitlines()]
        self.assertEqual(result["caption_coverage"]["A_raw_original_nonempty_clip_count"], 0)
        self.assertEqual(result["caption_coverage"]["B_two_windows_all_abstain_clip_count"], 4)
        for row in rows:
            self.assertIsNone(row["B_complete_clip_motion_caption"])
            self.assertIsNone(row["A_whole_clip"]["motion_caption"])
            for window in row["B_windows"]:
                self.assertIsNone(window["motion_caption"])
                self.assertFalse(window["candidate_text_eligible"])
                self.assertEqual(window["abstain_reason"], "no_output")

    def test_partial_gap_support_remains_abstain_with_raw_lineage(self):
        with np.load(self.source) as data:
            nlf = {key: data[key] for key in data.files}
        nlf["valid"][20:35] = False
        np.savez(self.source, **nlf)
        self.run_root = self.root / "partial_run"
        self.run_root.mkdir()
        self.args.run_root = str(self.run_root)
        self.make_shards()
        result = self.export()
        self.assertTrue(result["complete"])
        self.assertGreater(result["B_window_abstain_reasons"].get("partial_support", 0), 0)
        captions = [json.loads(line) for line in (Path(self.args.out) / "captions.jsonl").read_text().splitlines()]
        for row in captions:
            self.assertEqual(row["A_whole_clip"]["abstain_reason"], "partial_support")
            self.assertIsNone(row["A_whole_clip"]["motion_caption"])
        for row in (json.loads(line) for line in (Path(self.args.out) / "lineage.jsonl").read_text().splitlines()):
            self.assertTrue(row["raw_MS_and_support_indices_retained_in_report"])
            self.assertFalse(row["raw_NLF_reloaded_by_exporter"])

    def test_source_failure_accounted_exactly_without_claiming_all_processed(self):
        self.rows[2]["nlf_path"] = str(self.nlf / "missing.npz")
        self.write_selection()
        self.run_root = self.root / "failed_run"
        self.run_root.mkdir()
        self.args.run_root = str(self.run_root)
        self.make_shards()
        result = self.export()
        self.assertFalse(result["complete"])
        self.assertTrue(result["all_shards_terminal"])
        self.assertTrue(result["all_expected_ids_accounted"])
        self.assertFalse(result["all_expected_successfully_processed"])
        self.assertEqual(result["failed_source_count"], 1)
        self.assertEqual(result["caption_coverage"]["target_clip_denominator"], 4)
        self.assertEqual(result["caption_coverage"]["processed_clip_denominator"], 3)
        self.assertIsNone(result["caption_coverage"]["failed_source_window_count"])
        rows = [json.loads(line) for line in (Path(self.args.out) / "captions.jsonl").read_text().splitlines()]
        failed = next(row for row in rows if row["id"] == "c")
        self.assertIsNone(failed["A_whole_clip"])
        self.assertIsNone(failed["B_windows"])
        self.assertEqual(failed["status"], "failed")

    def test_export_resume_tamper_and_alias_refused(self):
        self.export()
        self.args.resume = True
        path = Path(self.args.out) / "captions.jsonl"
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "altered"):
            self.export()
        self.args.out = str(self.run_root / "unsafe_final")
        with self.assertRaisesRegex(ValueError, "separate"):
            self.export()

    def test_resealed_quality_count_mismatch_refused(self):
        self.replace_report("a", lambda report: report["quality"].update(source_valid_frames=0))
        with self.assertRaisesRegex(ValueError, "quality/missing count"):
            self.export()

    def test_resealed_cross_gap_or_window_support_indices_refused(self):
        self.replace_report("a", lambda report: report["chunks"][0]["supported_descriptions"][0]["target_indices"].append(40))
        with self.assertRaisesRegex(ValueError, "support indices"):
            self.export()

    def test_resealed_completed_export_cannot_claim_formal_approval(self):
        self.export()
        out = Path(self.args.out)
        final = json.loads((out / "complete.json").read_bytes())
        final["formal_approval"] = True
        content = producer.json_bytes(final)
        (out / "complete.json").write_bytes(content)
        (out / "complete.receipt.json").write_bytes(producer.json_bytes({"complete_sha256": exporter.digest(content)}))
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "summary/approval"):
            self.export()

    def test_coverage_mixed_partial_empty_and_error_never_full_pool_caption(self):
        candidate = {"status": "candidate", "abstain_reason": None, "raw_original_description_nonempty": True}
        partial = {"status": "abstain", "abstain_reason": "partial_support", "raw_original_description_nonempty": True}
        empty = {"status": "abstain", "abstain_reason": "no_output", "raw_original_description_nonempty": False}
        def clip(A, B, complete):
            return {"status": "processed", "A_whole_clip": A, "B_windows": B,
                    "B_complete_clip_motion_caption": complete}
        outputs = {"a": clip(candidate, [candidate, candidate], "two independently eligible windows"),
                   "b": clip(candidate, [candidate, partial], None),
                   "c": clip(candidate, [empty, candidate], None),
                   "d": clip(empty, [empty, empty], None),
                   "e": {"status": "failed", "error": {"error_type": "ValueError"}}}
        coverage = exporter.coverage_summary(outputs)
        self.assertEqual(coverage["target_clip_denominator"], 5)
        self.assertEqual(coverage["processed_clip_denominator"], 4)
        self.assertEqual(coverage["A_raw_original_nonempty_clip_count"], 3)
        self.assertEqual(coverage["B_two_windows_both_candidate_clip_count"], 1)
        self.assertEqual(coverage["B_two_windows_only_one_candidate_clip_count"], 2)
        self.assertEqual(coverage["B_two_windows_all_abstain_clip_count"], 1)
        self.assertEqual(coverage["B_at_least_one_partial_support_clip_count"], 1)
        self.assertEqual(coverage["B_at_least_one_no_output_clip_count"], 2)
        self.assertEqual(coverage["execution_error_type_counts"], {"ValueError": 1})
        self.assertIsNone(coverage["failed_source_window_count"])
        self.assertFalse(coverage["formal_training_eligible"])


if __name__ == "__main__":
    unittest.main()
