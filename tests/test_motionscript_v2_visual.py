"""Open-vocabulary workflow tests; no VLM or external API is called."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms"
sys.path.insert(0, str(MS))
import ms_visual_observe as visual


def candidate():
    return {"motion_caption": "The dancer performs a cross-behind step and pivots the torso.",
            "events": [{"id": "v1", "description": "Cross-behind step followed by a torso pivot",
                        "frame_indices": [0, 8]}], "uncertainties": [], "abstain": False}


class VisualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        # Hash-binding fixture only. The separate FFmpeg test decodes a real
        # synthetic video; these bytes are not claimed to be real observations.
        self.video = self.root / "fixture.video"
        self.video.write_bytes(b"synthetic hash fixture")
        frames = []
        for index in (0, 8):
            image = self.bundle / f"frame_{index}.jpg"
            image.write_bytes(f"synthetic frame {index}".encode())
            frames.append({"index": index, "time": index / 16, "file": image.name,
                           "sha256": visual.sha256(image)})
        request = self.bundle / "request.json"
        request.write_bytes(visual.json_bytes({"model": "test-model", "messages": []}))
        self.manifest = {"schema": visual.SCHEMA, "video_path": str(self.video),
            "video_sha256": visual.sha256(self.video), "request_sha256": visual.sha256(request), "frames": frames}
        raw = visual.json_bytes(self.manifest)
        (self.bundle / "manifest.json").write_bytes(raw)
        (self.bundle / "manifest.receipt.json").write_bytes(visual.json_bytes({
            "manifest_sha256": hashlib.sha256(raw).hexdigest()}))

    def test_new_action_names_are_not_restricted_to_rule_vocabulary(self):
        value = visual.validate_candidate(candidate(), self.manifest)
        self.assertIn("cross-behind", value["motion_caption"])
        self.assertFalse(value["semantic_approval"])
        self.assertFalse(value["events"][0]["time_range_is_exact_event_boundary"])
        self.assertFalse(value["rule_labels_supplied_to_observer"])

    def test_nonexistent_frame_evidence_and_looping_caption_rejected(self):
        value = candidate()
        value["events"][0]["frame_indices"] = [0, 99]
        with self.assertRaisesRegex(ValueError, "supplied frames"):
            visual.validate_candidate(value, self.manifest)
        value = candidate()
        value["motion_caption"] = "The elbow bends. The elbow bends."
        with self.assertRaisesRegex(ValueError, "identical clause"):
            visual.validate_candidate(value, self.manifest)

    def test_real_repetition_can_be_described_once(self):
        value = candidate()
        value["motion_caption"] = "The dancer repeatedly crosses one foot behind the other."
        self.assertTrue(visual.validate_candidate(value, self.manifest)["format_valid"])

    def test_format_pass_and_self_claimed_approval_do_not_authorize_export(self):
        visual.import_response(self.bundle, visual.json_bytes(candidate()))
        review = self.root / "review.json"
        review.write_bytes(visual.json_bytes({"approved": True, "reviewer": "test"}))
        with self.assertRaisesRegex(ValueError, "explicit review"):
            visual.export(self.bundle, review, self.root / "export.json")
        path = self.bundle / "candidate.json"
        value = json.loads(path.read_text())
        value["semantic_approval"] = True
        path.write_bytes(visual.json_bytes(value))
        with self.assertRaisesRegex(ValueError, "differs from its original"):
            visual.export(self.bundle, review, self.root / "export.json")

    def test_explicit_hash_bound_review_exports_without_touching_training(self):
        visual.import_response(self.bundle, visual.json_bytes(candidate()))
        review = self.root / "review.json"
        review.write_bytes(visual.json_bytes({"approved": True, "video_observed": True,
            "reviewer": "synthetic-test-reviewer", "reviewer_type": "human",
            "candidate_sha256": visual.sha256(self.bundle / "candidate.json"),
            "video_sha256": self.manifest["video_sha256"], "request_sha256": self.manifest["request_sha256"]}))
        value = visual.export(self.bundle, review, self.root / "export.json")
        self.assertTrue(value["semantic_approval"])
        self.assertFalse(value["training_entry_updated"])
        self.assertEqual(value["source"], "independent_video_open_vocabulary_reviewed")

    def test_source_video_or_frame_changes_invalidate_review_evidence(self):
        frame = self.bundle / "frame_0.jpg"
        old = frame.read_bytes()
        frame.write_bytes(b"changed frame")
        with self.assertRaisesRegex(ValueError, "frame evidence"):
            visual.verify_bundle(self.bundle)
        frame.write_bytes(old)
        self.video.write_bytes(b"new video")
        with self.assertRaisesRegex(ValueError, "source video"):
            visual.verify_bundle(self.bundle)

    def test_abstention_does_not_force_action_or_minimum_word_count(self):
        value = {"motion_caption": "", "events": [], "uncertainties": ["No supported movement"], "abstain": True}
        self.assertTrue(visual.validate_candidate(value, self.manifest)["abstain"])

    def test_ambiguous_duplicate_keys_in_model_response_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            visual.parse_response(b'{"motion_caption":"first","motion_caption":"second"}')

    def test_identical_raw_response_can_resume_interrupted_candidate_write(self):
        raw = visual.json_bytes(candidate())
        original = visual.atomic_write

        def interrupt(path, data):
            if path.name == "candidate.json":
                raise OSError("interrupted candidate")
            original(path, data)

        with mock.patch.object(visual, "atomic_write", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "interrupted"):
                visual.import_response(self.bundle, raw)
        self.assertFalse((self.bundle / "candidate.json").exists())
        self.assertFalse(visual.import_response(self.bundle, raw)["semantic_approval"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe unavailable")
    def test_prepare_real_synthetic_video_and_actual_frame_indices(self):
        source_dir = self.root / "videos"
        source_dir.mkdir()
        video = source_dir / "synthetic.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc=size=160x120:rate=16", "-t", "1", "-pix_fmt", "yuv420p", str(video)],
                       check=True, capture_output=True)
        out = self.root / "prepared"
        args = argparse.Namespace(video=str(video), out=str(out), model="test-model", sample_fps=8.,
                                  max_frames=5, start=0., end=None, ffmpeg="ffmpeg", ffprobe="ffprobe")
        value = visual.prepare(args)
        self.assertEqual(len(value["frames"]), 5)
        self.assertEqual(value["frames"][0]["index"], 0)
        self.assertEqual(value["frames"][-1]["index"], 15)
        self.assertEqual(visual.verify_bundle(out), value)
        request = json.loads((out / "request.json").read_text())
        self.assertEqual(request["model"], "test-model")
        self.assertFalse(value["rule_labels_supplied_to_observer"])
        self.assertNotIn("source_facts", request)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe unavailable")
    def test_multi_stream_video_sampling_matches_probed_stream(self):
        source_dir = self.root / "videos"
        source_dir.mkdir()
        video = source_dir / "two_streams.mkv"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=red:size=160x120:rate=16", "-f", "lavfi", "-i", "color=blue:size=320x240:rate=16",
            "-t", "1", "-map", "0:v", "-map", "1:v", "-c:v", "libx264",
            "-disposition:v:0", "0", "-disposition:v:1", "0", str(video)], check=True, capture_output=True)
        out = self.root / "multi_prepared"
        args = argparse.Namespace(video=str(video), out=str(out), model="test-model", sample_fps=8.,
                                  max_frames=3, start=0., end=None, ffmpeg="ffmpeg", ffprobe="ffprobe")
        value = visual.prepare(args)
        image = out / value["frames"][0]["file"]
        result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(image),
            "-vf", "scale=1:1", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            check=True, capture_output=True)
        self.assertGreater(result.stdout[0], result.stdout[2] + 100)  # Red stream, not larger blue stream.
        self.assertIn("0:v:0", value["video_stream_selector"])


if __name__ == "__main__":
    unittest.main()
