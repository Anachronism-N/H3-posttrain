import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HELPER = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms/runtime_jobs/beatthis_audio_v1.py"
spec = importlib.util.spec_from_file_location("beatthis_audio_v1", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BeatThisAudioContractTests(unittest.TestCase):
    def test_predictions_remain_model_outputs_not_periodic_bpm(self):
        result = module.detection_summary([0.32, 0.86, 1.42, 1.92, 2.50], [0.86, 2.50], 5.0625)
        self.assertEqual(result["beats"], [0.32, 0.86, 1.42, 1.92, 2.50])
        self.assertEqual(result["downbeats"], [0.86, 2.50])
        self.assertEqual(result["complete_downbeat_interval_count"], 1)
        self.assertTrue(result["short_context"])
        self.assertEqual(result["meter"], "unknown")
        self.assertEqual(result["audio_video_offset_seconds"], "unknown")
        for key in ("beat_accuracy_verified", "downbeat_accuracy_verified",
                    "bar_phase_accuracy_verified", "audio_video_alignment_verified"):
            self.assertIs(result[key], False)

    def test_zero_and_one_downbeat_are_not_fabricated_bars(self):
        for downbeats in ([], [0.32]):
            result = module.detection_summary([0.32], downbeats, 5.0625)
            self.assertTrue(result["insufficient_downbeats_for_complete_bar"])
            self.assertEqual(result["complete_downbeat_interval_count"], 0)
            self.assertEqual(result["downbeats"], downbeats)
        empty = module.detection_summary([], [], 5.0625)
        self.assertEqual(empty["beats"], [])
        self.assertEqual(empty["beat_count"], 0)

    def test_invalid_timestamp_contract(self):
        for times in ([float("nan")], [float("inf")], [True], [-0.1],
                      [5.1], [0.4, 0.4], [0.4, 0.3], ["0.4"]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                module.validate_times(times, 5.0625)
        self.assertEqual(module.validate_times([5.08], 5.0625), [5.08])

    def test_minimal_postprocessor_downbeat_subset(self):
        with self.assertRaises(ValueError):
            module.detection_summary([0.3, 0.7], [0.5], 5.0625)

    def test_input_hashes_match_exact_bytes_and_holdout_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl"
            row = {"id": "test_holdout", "audio_path": "/example/audio.wav",
                   "split": "test", "dataset_role": "fixed3"}
            line = json.dumps(row).encode()
            raw = line + b"\n"
            path.write_bytes(raw)
            rows, digest = module.load_manifest(path, 1)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            self.assertEqual(rows[0], (row, 1, hashlib.sha256(line).hexdigest()))
            with self.assertRaises(ValueError):
                module.load_manifest(path, 50)
            path.write_bytes(raw + raw)
            with self.assertRaises(ValueError):
                module.load_manifest(path, 2)

    def test_source_model_license_bytes_and_path_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "final0.ckpt"
            license_file = root / "LICENSE"
            source_file = root / "source.json"
            checkpoint.write_bytes(b"test-only-checkpoint")
            license_file.write_text("MIT fixture")
            source = {"checkpoint_path": str(checkpoint),
                      "checkpoint_sha256": module.sha256_file(checkpoint),
                      "source_url": "provided-source-label", "license": "MIT",
                      "license_path": str(license_file),
                      "license_sha256": module.sha256_file(license_file)}
            source_file.write_text(json.dumps(source))
            identity = module.source_identity(source_file, checkpoint)
            self.assertEqual(identity["checkpoint_sha256"], source["checkpoint_sha256"])
            checkpoint.write_bytes(b"different")
            with self.assertRaises(ValueError):
                module.source_identity(source_file, checkpoint)
            checkpoint.write_bytes(b"test-only-checkpoint")
            license_file.write_text("changed license")
            with self.assertRaises(ValueError):
                module.source_identity(source_file, checkpoint)

    def test_run_preserves_missing_audio_failure_not_empty_detections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "synthetic.ckpt"
            license_file = root / "LICENSE"
            source_path = root / "source.json"
            manifest = root / "manifest.jsonl"
            checkpoint.write_bytes(b"unit-test-not-a-model")
            license_file.write_text("synthetic MIT fixture")
            source_path.write_text(json.dumps({
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": module.sha256_file(checkpoint),
                "source_url": "synthetic-unit-test-source", "license": "MIT",
                "license_path": str(license_file),
                "license_sha256": module.sha256_file(license_file)}))
            manifest.write_text(json.dumps({"id": "missing", "audio_path": str(root / "missing.wav"),
                                            "split": "test", "dataset_role": "fixed3"}) + "\n")
            args = SimpleNamespace(manifest=manifest, expected_count=1, checkpoint_source=source_path,
                                   checkpoint=checkpoint, output_dir=root / "output",
                                   cpu_threads=1, device="cpu")
            with mock.patch("beat_this.inference.File2Beats") as loader, \
                 mock.patch("torch.set_num_interop_threads"):
                loader.return_value.model.parameters.return_value = []
                self.assertEqual(module.run(args), 2)
            result = json.loads((root / "output/detections.jsonl").read_text())
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["beats"])
            self.assertIsNone(result["downbeats"])
            self.assertIs(result["inference_called"], False)
            receipt = json.loads((root / "output/run_receipt.json").read_text())
            self.assertEqual(receipt["failed_count"], 1)
            self.assertEqual(receipt["successful_count"], 0)
            self.assertEqual(receipt["exit_code"], 2)
            before = (root / "output/detections.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                module.run(args)
            self.assertEqual((root / "output/detections.jsonl").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
