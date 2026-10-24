"""CPU schema/hash fixtures only; no NLF detector or VLM is impersonated.

All synthetic response fixtures test the validator and receipt machinery, not
real visual correctness or model inference. Optional FFmpeg tests decode actual
synthetic video. No GPU is touched, model downloaded, or endpoint contacted.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms"
sys.path.insert(0, str(MS))
import ms_visual_batch as batch

FFMPEG = os.environ.get("MS_VISUAL_FFMPEG") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("MS_VISUAL_FFPROBE") or shutil.which("ffprobe")


def response_fixture(indices=(0, 8), abstain=False):
    return {"motion_caption": "" if abstain else "The arm rises and the torso pivots.",
            "events": [] if abstain else [{"id": "fixture-event", "description": "An arm rise with a torso pivot",
                                          "frame_indices": list(indices)}],
            "uncertainties": [], "abstain": abstain}


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binding = {"id": "fixture", "source_sha256": "s" * 64, "model_sha256": "m" * 64,
                        "params_sha256": "p" * 64, "window": {"start": 0.0, "end": 2.0}}

    def manifest_row(self, sid="fixture"):
        return {"id": sid, "video_path": str(self.root / "missing-video.mp4"),
                "audio_path": str(self.root / "missing-audio.wav"),
                "nlf_path": str(self.root / "missing-nlf.npz"), "source_group": "unit-fixture", "split": "train"}

    def write_input(self, rows):
        path = self.root / "input.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def hash_bundle(self):
        # Hash fixtures, not decoded frames, model weights or real observations.
        bundle = self.root / "bundle"
        bundle.mkdir()
        video = self.root / "hash-fixture.video"
        video.write_bytes(b"synthetic hash-binding fixture; not a video")
        source = {"files": {"video_path": batch.file_identity(video)}, "sha256": "s" * 64}
        frames = []
        for index in (0, 8):
            path = bundle / f"hash-frame-{index}.jpg"
            path.write_bytes(f"synthetic hash fixture {index}; not a decoded image".encode())
            frames.append({"index": index, "time": index / 16, "file": path.name, "sha256": batch.sha256(path)})
        request = {"model": "fixture-not-loaded", "messages": [{"role": "system", "content": batch.SYSTEM}]}
        batch.atomic_write(bundle / "request.json", batch.json_bytes(request))
        manifest = {"schema": batch.VISUAL_SCHEMA, "binding": self.binding, "video_path": str(video),
                    "video_sha256": source["files"]["video_path"]["sha256"], "frames": frames,
                    "request_sha256": batch.sha256(bundle / "request.json")}
        raw = batch.json_bytes(manifest)
        batch.atomic_write(bundle / "manifest.json", raw)
        batch.atomic_write(bundle / "manifest.receipt.json", batch.json_bytes({
            "manifest_sha256": hashlib.sha256(raw).hexdigest()}))
        return bundle, manifest, source

    def test_manifest_is_explicit_and_diagnostic_splits_remain_declared(self):
        row = self.manifest_row()
        row["rich_caption"] = "never sent to observer"
        rows, sha = batch.read_manifest(self.write_input([row]))
        self.assertEqual(rows[0]["source_group"], "unit-fixture")
        self.assertNotIn("rich_caption", rows[0])
        self.assertEqual(len(sha), 64)
        for field in batch.FIELDS:
            broken = self.manifest_row()
            del broken[field]
            with self.assertRaisesRegex(ValueError, "explicitly contain"):
                batch.read_manifest(self.write_input([broken]))
        for split in ("test", "val"):
            declared = self.manifest_row()
            declared["split"] = split
            diagnostic, _ = batch.read_manifest(self.write_input([declared]))
            self.assertEqual(diagnostic[0]["split"], split)
        for split in (None, "", "guessed"):
            broken = self.manifest_row()
            broken["split"] = split
            with self.assertRaisesRegex(ValueError, "split must explicitly"):
                batch.read_manifest(self.write_input([broken]))

    def test_manifest_rejects_duplicate_unsafe_and_relative_inputs(self):
        for rows in ([self.manifest_row(), self.manifest_row("FIXTURE")], [self.manifest_row("../escape")]):
            with self.assertRaises(ValueError):
                batch.read_manifest(self.write_input(rows))
        row = self.manifest_row()
        row["nlf_path"] = "relative.npz"
        with self.assertRaisesRegex(ValueError, "absolute path"):
            batch.read_manifest(self.write_input([row]))

    def test_manifest_shard_limit_and_duplicate_keys(self):
        path = self.write_input([self.manifest_row(f"id{i}") for i in range(6)])
        rows, _ = batch.read_manifest(path, 1, 2, 2)
        self.assertEqual([row["id"] for row in rows], ["id1", "id3"])
        path.write_text('{"id":"a","id":"b"}\n')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            batch.read_manifest(path)

    def test_request_clarifies_real_validator_schema_without_action_prior(self):
        self.assertIn("unique nonempty JSON string", batch.OBSERVE_TEXT)
        self.assertIn("at least two distinct", batch.OBSERVE_TEXT)
        self.assertIn("both events and motion_caption", batch.OBSERVE_TEXT)
        self.assertNotIn("cross-behind", batch.OBSERVE_TEXT)
        self.assertNotIn("genre", batch.OBSERVE_TEXT)

    def test_missing_assets_fail_without_creating_nlf_placeholders(self):
        row = self.manifest_row()
        with self.assertRaisesRegex(ValueError, "missing.*no placeholder"):
            batch.source_identity(row)
        self.assertFalse(Path(row["nlf_path"]).exists())

    def test_fixed_windows_merge_real_short_tail(self):
        self.assertEqual(batch.fixed_windows(0, 4.9, 2, 1.5),
                         [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 4.9}])
        self.assertEqual(batch.fixed_windows(0, 1.1, 2, 1.5), [{"start": 0.0, "end": 1.1}])

    def test_selection_uses_true_global_indices_and_prepare_semantics(self):
        times = [i / 16 for i in range(80)]
        selected = batch.select_frames(times, 2, 4.9)
        self.assertEqual(selected[0], 32)
        self.assertEqual(selected[-1], 78)
        self.assertLessEqual(len(selected), 16)
        self.assertTrue(all(b > a for a, b in zip(selected, selected[1:])))
        self.assertEqual(batch.select_frames(times, 0, 2), list(range(0, 32, 4)) + [31])
        self.assertNotIn(32, batch.select_frames(times, 0, 2))
        budgeted = batch.select_frames(times, 0, 4.9, max_frames=3)
        self.assertEqual(len(budgeted), 3)
        self.assertEqual((budgeted[0], budgeted[-1]), (0, 78))
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            batch.select_frames(times, 0, 0.01)

    def test_probe_gets_video_duration_not_nlf_or_nominal_fps(self):
        value = {"frames": [{"best_effort_timestamp_time": str(10 + i * 0.25)} for i in range(9)],
                 "streams": [{"duration": "2.25"}], "format": {"duration": "99"}}
        with mock.patch.object(batch.subprocess, "run", return_value=argparse.Namespace(stdout=json.dumps(value).encode())) as run:
            probe = batch.probe_video("explicit-video.mp4", "installed-ffprobe")
        self.assertEqual(probe["media_end"], 2.25)
        self.assertEqual(probe["times"], [i * 0.25 for i in range(9)])
        self.assertEqual(probe["time_origin_source_timestamp"], 10)
        self.assertIn("v:0", run.call_args.args[0])
        value["frames"][1]["best_effort_timestamp_time"] = "10"
        with mock.patch.object(batch.subprocess, "run", return_value=argparse.Namespace(stdout=json.dumps(value).encode())):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                batch.probe_video("explicit-video.mp4", "installed-ffprobe")

    def test_probe_can_use_actual_last_frame_declared_duration(self):
        value = {"frames": [{"best_effort_timestamp_time": "0"},
                            {"best_effort_timestamp_time": "0.5", "pkt_duration_time": "0.5"}], "streams": [{}]}
        with mock.patch.object(batch.subprocess, "run", return_value=argparse.Namespace(stdout=json.dumps(value).encode())):
            self.assertEqual(batch.probe_video("video", "ffprobe")["media_end"], 1)

    def test_container_duration_never_substitutes_video_end(self):
        value = {"frames": [{"best_effort_timestamp_time": "0"}, {"best_effort_timestamp_time": "0.5"}],
                 "streams": [{}], "format": {"duration": "99"}}
        with mock.patch.object(batch.subprocess, "run", return_value=argparse.Namespace(stdout=json.dumps(value).encode())):
            with self.assertRaisesRegex(ValueError, "container/audio duration cannot"):
                batch.probe_video("video", "ffprobe")

    def test_production_raw_without_generation_stats_never_resumes_as_complete(self):
        bundle, manifest, source = self.hash_bundle()
        manifest["batch_schema"] = batch.SCHEMA
        raw = batch.json_bytes(response_fixture())
        receipt = batch.finish_response(bundle, self.binding, manifest, raw)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("lacks generation", receipt["error"])
        self.assertFalse((bundle / "candidate.json").exists())

    def test_length_finish_retains_raw_but_is_failed_even_if_json_valid(self):
        bundle, manifest, source = self.hash_bundle()
        raw = batch.json_bytes(response_fixture())
        batch.atomic_write(bundle / "generation.json", batch.json_bytes({"finish_reason": "length",
            "completion_tokens": 1200, "max_tokens": 1200, "fixture_not_actual_model_call": True}))
        receipt = batch.finish_response(bundle, self.binding, manifest, raw)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("truncated", receipt["error"])
        self.assertEqual((bundle / "response.raw.txt").read_bytes(), raw)
        self.assertFalse((bundle / "candidate.json").exists())
        self.assertIn("generation.json", receipt["files_sha256"])

    def test_model_same_basename_is_not_identity_and_mutation_is_detected(self):
        identities = []
        for parent in ("a", "b"):
            root = self.root / parent / "same-basename"
            root.mkdir(parents=True)
            (root / "config.json").write_text("{}")
            (root / "fixture.safetensors").write_bytes(b"hash fixture only; not loadable model weights")
            identities.append(batch.model_identity(root))
        self.assertNotEqual(identities[0]["sha256"], identities[1]["sha256"])
        batch.verify_current(identities[0], model=True)
        (Path(identities[0]["root"]) / "fixture.safetensors").write_bytes(b"changed hash fixture")
        with self.assertRaisesRegex(ValueError, "changed before loading"):
            batch.verify_current(identities[0], model=True)
        (Path(identities[1]["root"]) / "new-config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            batch.verify_current(identities[1], model=True)

    def test_terminal_fixture_receipts_resume_only_exact_bindings(self):
        bundle, manifest, source = self.hash_bundle()
        raw = batch.json_bytes(response_fixture())
        receipt = batch.finish_response(bundle, self.binding, manifest, raw)
        self.assertEqual(receipt["status"], "candidate")
        self.assertFalse(receipt["human_review_completed"])
        self.assertEqual(batch.resume_receipt(bundle, self.binding, source), receipt)
        for field in ("source_sha256", "model_sha256", "params_sha256"):
            changed = {**self.binding, field: "changed"}
            with self.assertRaisesRegex(ValueError, "binding"):
                batch.resume_receipt(bundle, changed, source)
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            batch.finish_response(bundle, self.binding, manifest, b"different raw text")

    def test_raw_invalid_json_and_hallucinated_frame_are_failed_not_complete(self):
        bundle, manifest, source = self.hash_bundle()
        raw = b"not JSON at all"
        receipt = batch.finish_response(bundle, self.binding, manifest, raw)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual((bundle / "response.raw.txt").read_bytes(), raw)
        self.assertFalse((bundle / "candidate.json").exists())
        self.assertEqual(batch.resume_receipt(bundle, self.binding, source)["status"], "failed")
        value = response_fixture((0, 999))
        with self.assertRaisesRegex(ValueError, "supplied frames"):
            batch.validate_candidate(value, manifest)

    def test_validator_rejects_repeat_words_one_frame_and_nonempty_abstain(self):
        _, manifest, _ = self.hash_bundle()
        value = response_fixture()
        value["motion_caption"] = "The arm rises. The arm rises."
        with self.assertRaisesRegex(ValueError, "identical clause"):
            batch.validate_candidate(value, manifest)
        value["motion_caption"] = "word " * 141
        with self.assertRaisesRegex(ValueError, "140 words"):
            batch.validate_candidate(value, manifest)
        value = response_fixture((0,))
        with self.assertRaisesRegex(ValueError, "at least two"):
            batch.validate_candidate(value, manifest)
        value = response_fixture(abstain=True)
        value["motion_caption"] = "guessed motion"
        with self.assertRaisesRegex(ValueError, "empty caption"):
            batch.validate_candidate(value, manifest)

    def test_abstain_fixture_has_no_caption_and_no_approval(self):
        bundle, manifest, source = self.hash_bundle()
        receipt = batch.finish_response(bundle, self.binding, manifest, batch.json_bytes(response_fixture(abstain=True)))
        self.assertEqual(receipt["status"], "abstain")
        self.assertEqual(batch.resume_receipt(bundle, self.binding, source)["status"], "abstain")
        candidate = json.loads((bundle / "candidate.json").read_text())
        self.assertEqual(candidate["motion_caption"], "")
        self.assertFalse(candidate["semantic_approval"])

    def test_interrupted_raw_can_be_finalized_without_model_call(self):
        bundle, manifest, source = self.hash_bundle()
        raw = batch.json_bytes(response_fixture())
        original = batch.atomic_write

        def interrupt(path, data):
            if path.name == "candidate.json":
                raise OSError("simulated interrupted candidate write")
            original(path, data)

        with mock.patch.object(batch, "atomic_write", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "interrupted"):
                batch.finish_response(bundle, self.binding, manifest, raw)
        self.assertEqual((bundle / "response.raw.txt").read_bytes(), raw)
        self.assertFalse((bundle / "receipt.json").exists())
        self.assertEqual(batch.finish_response(bundle, self.binding, manifest, raw)["status"], "candidate")

    def test_tampered_frame_request_raw_or_candidate_blocks_resume(self):
        bundle, manifest, source = self.hash_bundle()
        batch.finish_response(bundle, self.binding, manifest, batch.json_bytes(response_fixture()))
        for name in ("hash-frame-0.jpg", "request.json", "response.raw.txt", "candidate.json"):
            path = bundle / name
            original = path.read_bytes()
            path.write_bytes(b"tampered fixture")
            with self.assertRaises(ValueError):
                batch.resume_receipt(bundle, self.binding, source)
            path.write_bytes(original)

    def test_failed_retry_uses_new_attempt_preserves_old_and_does_not_repeat_candidate(self):
        root, manifest, source = self.hash_bundle()
        old_raw = b"invalid old response fixture"
        failed = batch.finish_response(root, self.binding, manifest, old_raw)
        old_receipt = (root / "receipt.json").read_bytes()
        bundle, receipt = batch.select_attempt(root, self.binding, source, False)
        self.assertEqual(bundle, root)
        self.assertEqual(receipt["status"], "failed")
        with batch.bundle_lock(root):
            retry, receipt = batch.select_attempt(root, self.binding, source, True)
        self.assertEqual(retry.name, "attempt-000001")
        self.assertIsNone(receipt)
        self.assertEqual((root / "response.raw.txt").read_bytes(), old_raw)
        self.assertEqual((root / "receipt.json").read_bytes(), old_receipt)
        # Pure receipt fixture only, not a fake successful LLM.chat invocation.
        for name in ("manifest.json", "manifest.receipt.json", "request.json", "hash-frame-0.jpg", "hash-frame-8.jpg"):
            shutil.copyfile(root / name, retry / name)
        result = batch.finish_response(retry, self.binding, manifest, batch.json_bytes(response_fixture()))
        self.assertEqual(result["status"], "candidate")
        selected, resumed = batch.select_attempt(root, self.binding, source, True)
        self.assertEqual(selected, retry)
        self.assertEqual(resumed["status"], "candidate")
        self.assertFalse((root / "attempt-000002").exists())
        self.assertEqual((root / "receipt.json").read_bytes(), old_receipt)

    def test_prepared_receipt_never_counts_as_observation(self):
        bundle, manifest, source = self.hash_bundle()
        batch.write_receipt(bundle, self.binding, "prepared")
        self.assertIsNone(batch.resume_receipt(bundle, self.binding, source))
        clip = {"row": self.manifest_row(), "clip": str(bundle.parent), "source": source,
                "windows": [{"bundle": str(bundle), "binding": self.binding, "receipt": None}],
                "media_end": 2, "media_end_basis": "unit fixture, not media validation"}
        result = batch.final_caption(clip)
        self.assertEqual(result["status"], "prepared")
        self.assertFalse(result["observation_complete_without_failures"])
        self.assertEqual(json.loads(Path(result["final_path"]).read_text())["timed_caption"], "")

    def test_final_concat_is_verbatim_and_failed_windows_remain_gaps(self):
        bundle, manifest, source = self.hash_bundle()
        receipt = batch.finish_response(bundle, self.binding, manifest, batch.json_bytes(response_fixture()))
        failed = self.root / "failed-window"
        failed.mkdir()
        binding = {**self.binding, "window": {"start": 2.0, "end": 4.0}}
        failure = batch.write_receipt(failed, binding, "failed", error="fixture failure, no model call")
        clip = {"row": self.manifest_row(), "clip": str(bundle.parent), "source": source,
                "windows": [{"bundle": str(bundle), "binding": self.binding, "receipt": receipt},
                            {"bundle": str(failed), "binding": binding, "receipt": failure}],
                "media_end": 4, "media_end_basis": "unit fixture"}
        result = batch.final_caption(clip)
        self.assertEqual(result["status"], "failed")
        value = json.loads(Path(result["final_path"]).read_text())
        self.assertEqual(value["timed_caption"], "[0.000-2.000s] The arm rises and the torso pivots.")
        self.assertFalse(value["observation_complete_without_failures"])
        self.assertFalse(value["whole_clip_summary_generated"])
        self.assertFalse(value["semantic_approval"])

    def test_window_locks_are_nonblocking_and_no_detached_process_is_created(self):
        path = self.root / "locked"
        with batch.bundle_lock(path):
            with self.assertRaisesRegex(ValueError, "active in another managed process"):
                with batch.bundle_lock(path):
                    self.fail("lock incorrectly acquired")
        with batch.bundle_lock(path):
            pass

    def test_cli_rejects_remote_model_and_bad_batch_before_vllm_import(self):
        for extra in (["--model", "remote/hub-id"], ["--model", str(self.root), "--batch-size", "2"]):
            with self.assertRaises(SystemExit):
                batch.main(["--manifest", str(self.root / "missing.jsonl"), "--out", str(self.root / "out"), *extra])
        self.assertNotIn("vllm", sys.modules)

    @unittest.skipUnless(FFMPEG and FFPROBE, "installed FFmpeg/FFprobe unavailable; no installation attempted")
    def test_real_synthetic_video_sampling_and_video_only_multiframe_request(self):
        source_dir = self.root / "videos"
        source_dir.mkdir()
        video = source_dir / "synthetic.mp4"
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "testsrc=size=160x120:rate=16", "-t", "4.9", "-pix_fmt", "yuv420p", str(video)],
            check=True, capture_output=True)
        source = {"files": {"video_path": batch.file_identity(video)}, "sha256": "s" * 64}
        probe = batch.probe_video(video, FFPROBE)
        windows = batch.fixed_windows(0, probe["media_end"], 2, 1.5)
        self.assertEqual(len(windows), 2)
        row = {"id": "synthetic-video-only-test", "video_path": str(video),
               "rich_caption": "THIS RICH TEXT MUST NOT ENTER THE REQUEST", "genre": "NOT AN OBSERVER INPUT"}
        args = argparse.Namespace(ffmpeg=FFMPEG, model=str(self.root / "not-loaded-test-model"))
        for index, window in enumerate(windows):
            bundle = self.root / f"prepared-{index}"
            binding = {**self.binding, "window": window}
            manifest = batch.prepare_window(bundle, row, source, probe, window, binding, args)
            self.assertEqual(batch.verify_evidence(bundle, binding, source), manifest)
            self.assertTrue(all(window["start"] <= frame["time"] < window["end"] for frame in manifest["frames"]))
            self.assertLessEqual(len(manifest["frames"]), 16)
            if index == 1:
                self.assertEqual(manifest["frames"][0]["index"], 32)
                self.assertEqual(manifest["frames"][0]["time"], 2)
            request = json.loads((bundle / "request.json").read_text())
            self.assertEqual(request["messages"][0]["content"], batch.SYSTEM)
            content = request["messages"][1]["content"]
            expected = json.dumps([frame["index"] for frame in manifest["frames"]])
            self.assertIn("Allowed Source frame indices for this window: " + expected, content[0]["text"])
            self.assertIn(batch.OBSERVE_TEXT, content[0]["text"])
            self.assertEqual(sum(item["type"] == "image_url" for item in content), len(manifest["frames"]))
            self.assertNotIn("THIS RICH TEXT", json.dumps(request))
            self.assertNotIn("NOT AN OBSERVER INPUT", json.dumps(request))
            self.assertFalse((bundle / "candidate.json").exists())
            self.assertFalse(manifest["human_review_completed"])


if __name__ == "__main__":
    unittest.main()
