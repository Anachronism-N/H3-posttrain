"""Rebuilt canonical byte checks for v2, without modifying historical exports."""

import json
from pathlib import Path
from unittest import mock

import test_ms_chunk_export as fixtures
import ms_chunk_export as first
import ms_chunk_export_v2 as verifier


class ExportV2Tests(fixtures.ExportTests):
    def setUp(self):
        patch = mock.patch.object(fixtures, "exporter", verifier)
        patch.start()
        self.addCleanup(patch.stop)
        super().setUp()

    def reseal_export(self, transform):
        out = Path(self.args.out)
        rows = [json.loads(line) for line in (out / "captions.jsonl").read_text().splitlines()]
        transform(rows)
        content = b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for row in rows)
        (out / "captions.jsonl").write_bytes(content)
        saved = json.loads((out / "complete.json").read_bytes())
        saved["file_sha256"]["captions.jsonl"] = verifier.digest(content)
        final = verifier.json_bytes(saved)
        (out / "complete.json").write_bytes(final)
        (out / "complete.receipt.json").write_bytes(verifier.json_bytes({"complete_sha256": verifier.digest(final)}))

    def test_three_resealed_final_resume_semantic_mutations_rejected(self):
        self.export()
        out = Path(self.args.out)
        original = {path: path.read_bytes() for path in out.iterdir()}
        def partial_filled(rows):
            rows[0]["B_windows"][0]["status"] = "abstain"
            rows[0]["B_windows"][0]["abstain_reason"] = "partial_support"
            rows[0]["B_complete_clip_motion_caption"] = "invented complete body despite partial"
        def empty_eligible(rows):
            rows[0]["B_windows"][0]["motion_caption"] = ""
            rows[0]["B_windows"][0]["candidate_text_eligible"] = True
        def A_fallback(rows):
            rows[0]["B_windows"][0]["motion_caption"] = rows[0]["A_whole_clip"]["motion_caption"]
        self.args.resume = True
        for mutation in (partial_filled, empty_eligible, A_fallback):
            with self.subTest(mutation=mutation.__name__):
                self.reseal_export(mutation)
                with self.assertRaisesRegex(ValueError, "canonical producer reconstruction"):
                    self.export()
                for path, content in original.items():
                    path.write_bytes(content)

    def test_three_resealed_producer_source_path_or_B_mode_rejected(self):
        def input_path(report):
            report["input_path"] = "/different/source.npz"
        def source(report):
            report["source"] = "visual_or_music_fake_source"
        def B_mode(report):
            report["chunks"][0]["mode"] = "A_original_rules_whole_clip_same_coordinates"
        directory = self.run_root / "shard0"
        original = {path: path.read_bytes() for path in directory.iterdir()}
        for mutation, message in ((input_path, "input_path"), (source, "original MS source"), (B_mode, "B window mode")):
            with self.subTest(mutation=mutation.__name__):
                self.replace_report("a", mutation)
                with self.assertRaisesRegex(ValueError, message):
                    self.export()
                for path, content in original.items():
                    path.write_bytes(content)

    def test_historical_first_export_readonly_verification_preserves_original_fingerprint_and_files(self):
        result = first.run(self.args)
        out = Path(self.args.out)
        before = {path.name: path.read_bytes() for path in out.iterdir()}
        self.args.verify_existing = True
        self.args.verification_receipt = str(self.root / "independent-v2-verification.json")
        original = Path.read_bytes
        def read_bytes(path):
            if path in (self.source, self.video, self.audio):
                raise AssertionError("v2 must not read original media or raw NLF")
            return original(path)
        with mock.patch.object(Path, "read_bytes", read_bytes):
            verified = self.export()
        self.assertTrue(verified["complete"])
        self.assertTrue(verified["exact_expected_bytes_verified"])
        self.assertFalse(verified["existing_export_modified"])
        self.assertFalse(verified["formal_approval"])
        self.assertEqual(verified["first_export_source_sha256"], verifier.V1_SOURCE_SHA256)
        self.assertEqual(verified["verifier_source_sha256"], verifier.digest(Path(verifier.__file__).read_bytes()))
        self.assertEqual(verified["existing_complete_sha256"], verifier.digest(before["complete.json"]))
        self.assertEqual(before, {path.name: path.read_bytes() for path in out.iterdir()})
        self.assertEqual(json.loads((out / "complete.json").read_bytes())["fingerprint"], result["fingerprint"])
        self.assertTrue(Path(self.args.verification_receipt).is_file())

    def test_historical_first_export_resealed_caption_is_rejected_without_rewriting_it(self):
        first.run(self.args)
        self.reseal_export(lambda rows: rows[0]["B_windows"][0].update(motion_caption=rows[0]["A_whole_clip"]["motion_caption"]))
        out = Path(self.args.out)
        before = {path.name: path.read_bytes() for path in out.iterdir()}
        self.args.verify_existing = True
        with self.assertRaisesRegex(ValueError, "canonical producer reconstruction"):
            self.export()
        self.assertEqual(before, {path.name: path.read_bytes() for path in out.iterdir()})

    def test_resealed_final_summary_numeric_bool_rejected_on_resume(self):
        self.export()
        self.args.resume = True
        out = Path(self.args.out)
        original = {path: path.read_bytes() for path in out.iterdir()}
        for key, value in (("complete", 1), ("semantic_approval", 0), ("formal_approval", 0), ("all_expected_successfully_processed", 1)):
            with self.subTest(key=key):
                saved = json.loads((out / "complete.json").read_bytes())
                saved[key] = value
                content = verifier.json_bytes(saved)
                (out / "complete.json").write_bytes(content)
                (out / "complete.receipt.json").write_bytes(verifier.json_bytes({"complete_sha256": verifier.digest(content)}))
                with self.assertRaisesRegex(ValueError, "canonical producer reconstruction"):
                    self.export()
                for path, content in original.items():
                    path.write_bytes(content)

    def test_resealed_historical_summary_numeric_bool_rejected_readonly(self):
        first.run(self.args)
        self.args.verify_existing = True
        out = Path(self.args.out)
        original = {path: path.read_bytes() for path in out.iterdir()}
        for key, value in (("complete", 1), ("semantic_approval", 0), ("formal_approval", 0), ("all_expected_successfully_processed", 1)):
            with self.subTest(key=key):
                saved = json.loads((out / "complete.json").read_bytes())
                saved[key] = value
                content = verifier.json_bytes(saved)
                (out / "complete.json").write_bytes(content)
                (out / "complete.receipt.json").write_bytes(verifier.json_bytes({"complete_sha256": verifier.digest(content)}))
                tampered = {path: path.read_bytes() for path in out.iterdir()}
                with self.assertRaisesRegex(ValueError, "canonical producer reconstruction"):
                    self.export()
                self.assertEqual(tampered, {path: path.read_bytes() for path in out.iterdir()})
                for path, content in original.items():
                    path.write_bytes(content)

    def test_resealed_lineage_or_error_bytes_not_trusted_from_saved_self_hash(self):
        self.export()
        out = Path(self.args.out)
        original = {path: path.read_bytes() for path in out.iterdir()}
        self.args.resume = True
        for name in ("lineage.jsonl", "errors.jsonl"):
            with self.subTest(name=name):
                path = out / name
                path.write_bytes(path.read_bytes() + b"{}\n")
                saved = json.loads((out / "complete.json").read_bytes())
                saved["file_sha256"][name] = verifier.digest(path.read_bytes())
                content = verifier.json_bytes(saved)
                (out / "complete.json").write_bytes(content)
                (out / "complete.receipt.json").write_bytes(verifier.json_bytes({"complete_sha256": verifier.digest(content)}))
                with self.assertRaisesRegex(ValueError, "canonical producer reconstruction"):
                    self.export()
                for path, content in original.items():
                    path.write_bytes(content)
