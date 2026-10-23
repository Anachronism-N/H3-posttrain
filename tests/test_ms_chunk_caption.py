"""Synthetic contracts: strict matched-media input and unapproved original MS."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms"
sys.path.insert(0, str(MS))
import ms_chunk_caption as cli
from test_motionscript_v2_cli import write_fixture


def fake_captions(arrays, src, shim):
    if not len(arrays["time"]):
        return []
    return [{"start": float(arrays["time"][0]), "end": float(arrays["time"][-1]),
             "source": "synthetic_stub_not_motion_evidence",
             "ms": ["She raises her left arm twice. A second later she raises her left arm twice."],
             "ms_nonagg": ["His left arm moves."]}]


def video_info():
    return {"time": np.arange(81) / 16, "fps": 16., "media_end": 5.0625,
            "media_end_source": "synthetic decoded final frame duration", "pts_offset": 3.,
            "width": 704, "height": 1280}


class SliceTests(unittest.TestCase):
    def arrays(self):
        time = np.arange(81) / 20
        return {"time": time, "translation": np.column_stack((time + 3, time, time)),
                "coords": np.zeros((81, 54, 3)), "valid": np.ones(81, bool),
                "joint_valid": np.ones((81, 22), bool), "segment_id": np.zeros(81, int),
                "joint_segment_id": np.zeros((81, 22), int),
                "source_time": time.copy(), "source_valid": np.ones(81, bool),
                "source_joint_valid": np.ones((81, 22), bool),
                "source_uncertainty": np.ones((81, 22)), "unknown_metadata": np.arange(81)}

    def test_half_open_global_slice_and_equal_length_raw_retention(self):
        arrays = self.arrays()
        sliced, selected = cli.slice_arrays(arrays, 2, 4)
        np.testing.assert_array_equal(selected, np.arange(40, 80))
        np.testing.assert_allclose(sliced["translation"][0], [5, 2, 2])
        self.assertEqual(sliced["time"][0], 2)
        self.assertLess(sliced["time"][-1], 4)
        for name in ("source_time", "source_valid", "source_uncertainty", "source_joint_valid", "unknown_metadata"):
            self.assertIs(sliced[name], arrays[name])
        np.testing.assert_array_equal(arrays["time"], np.arange(81) / 20)

    def test_actual_window_header_support_separate_and_counts_unchanged(self):
        result = cli.describe_windows(self.arrays(), cli.fixed_windows(0, 4.1), None, None, captioner=fake_captions)
        self.assertEqual([(r["window_start"], r["window_end"]) for r in result], [(0, 2), (2, 4.1)])
        self.assertEqual(result[1]["supported_descriptions"][0]["target_indices"], list(range(40, 81)))
        self.assertIn("[2.000000, 4.100000) seconds:", result[1]["cleaned_motion_text"])
        self.assertEqual(result[0]["supported_descriptions"][0]["end"], 1.95)
        self.assertEqual(result[0]["cleaned_motion_text"].count("twice"), 2)
        self.assertNotIn("second later", result[0]["cleaned_motion_text"])
        self.assertIn("the dancer raises the dancer's left arm", result[0]["cleaned_motion_text"])
        self.assertIn("A second later", result[0]["original_captions"][0]["ms"][0])
        for row in result:
            self.assertEqual(row["status"], "candidate")
            self.assertFalse(row["semantic_approval"])
            self.assertFalse(row["human_review_completed"])
            self.assertFalse(row["coordinate_reset_applied"])
            self.assertFalse(row["sample_span_is_complete_window_evidence"])

    def test_empty_no_output_and_partial_support_abstention(self):
        empty = cli.describe_windows(self.arrays(), [{"start": 8, "end": 10}], None, None, captioner=fake_captions)[0]
        self.assertEqual(empty["status"], "abstain")
        self.assertEqual(empty["abstain_reason"], "no_samples_in_window")
        self.assertEqual(empty["selected_target_indices"], [])
        def blank(arrays, src, shim):
            return [{"start": 0., "end": 1.95, "ms": ["", "  "], "ms_nonagg": []}]
        row = cli.describe_windows(self.arrays(), [{"start": 0, "end": 2}], None, None, captioner=blank)[0]
        self.assertEqual(row["abstain_reason"], "no_output")
        arrays = self.arrays()
        arrays["joint_valid"][10, 3] = False
        partial = cli.describe_windows(arrays, [{"start": 0, "end": 2}], None, None, captioner=fake_captions)[0]
        self.assertEqual(partial["abstain_reason"], "partial_support")
        self.assertEqual(partial["cleaned_motion_text"], "")
        self.assertIn("twice", partial["partial_supported_motion_text"])
        self.assertEqual(partial["support_status"], "partial")

    def test_raw_source_gap_is_partial_even_when_target_grid_misses_it(self):
        arrays = self.arrays()
        arrays["source_joint_valid"][15, 4] = False
        row = cli.describe_windows(arrays, [{"start": 0, "end": 2}], None, None, captioner=fake_captions)[0]
        self.assertEqual(row["status"], "abstain")
        self.assertEqual(row["abstain_reason"], "partial_support")

    def test_original_call_does_not_stitch_root_or_joint_gaps_or_reset(self):
        arrays = self.arrays()
        arrays = {key: value[:12] if key in cli.TEMPORAL_FIELDS else value for key, value in arrays.items()}
        arrays["segment_id"] = np.repeat([4, 9], 6)
        arrays["joint_segment_id"][8:, 1] = 2
        arrays["valid"][5] = False
        arrays["translation"][:, 0] = np.arange(12) + 10
        arrays["coords"][:, 53, 0] = arrays["translation"][:, 0]
        captures = []
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory)
            captioning = types.ModuleType("text2pose.posescript.captioning")
            captioning.__file__ = str(src / "captioning.py")
            def main(coords, **kwargs):
                captures.append(coords.copy())
                return None, None, ["raw"], ["He turns his body."]
            captioning.main = main
            posescript = types.ModuleType("text2pose.posescript")
            posescript.captioning = captioning
            text2pose = types.ModuleType("text2pose")
            text2pose.posescript = posescript
            torch = types.SimpleNamespace(manual_seed=lambda seed: None, as_tensor=lambda value, **kwargs: value, float32=np.float32)
            with mock.patch.dict(sys.modules, {"torch": torch, "text2pose": text2pose,
                       "text2pose.posescript": posescript, "text2pose.posescript.captioning": captioning}):
                old_path = sys.path.copy()
                try:
                    rows = cli.describe_windows(arrays, [{"start": 0, "end": 2}], src, None)
                finally:
                    sys.path[:] = old_path
        self.assertEqual([len(x) for x in captures], [5, 4])
        np.testing.assert_array_equal(captures[0][:, 53, 0], np.arange(5) + 10)
        np.testing.assert_array_equal(captures[1][:, 53, 0], np.arange(8, 12) + 10)
        self.assertEqual(rows[0]["supported_descriptions"][1]["target_indices"], list(range(8, 12)))
        self.assertEqual(rows[0]["status"], "abstain")


class ManifestAndCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nlf, self.source, self.template = write_fixture(self.root)
        media = self.root / "media"
        media.mkdir()
        self.video, self.audio = media / "synthetic.mp4", media / "synthetic.wav"
        self.video.write_bytes(b"synthetic video hash fixture")
        self.audio.write_bytes(b"synthetic audio hash fixture")
        self.manifest = self.root / "selection.jsonl"
        self.row = {"id": "synthetic", "nlf_path": "nlf/synthetic.npz", "video_path": str(self.video),
                    "audio_path": str(self.audio), "source_group": "synthetic-source", "split": "train",
                    "video_end_seconds": 5.0625, "normalized_audio_offset_seconds": "unknown", "source_record": {"synthetic": True}}
        self.write_manifest([self.row])
        self.src = self.root / "upstream"
        self.src.mkdir()
        (self.src / "captioning.py").write_text("# synthetic-only hash fixture\n")
        self.args = cli.parser().parse_args(["--manifest", str(self.manifest), "--template", str(self.template),
                   "--upstream-src", str(self.src), "--out", str(self.root / "chunks_out")])

    def write_manifest(self, rows):
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def run_cli(self):
        def init(context):
            cli._WORKER = context
        with mock.patch.object(cli, "init_worker", side_effect=init), mock.patch.object(cli, "original_captions", side_effect=fake_captions), mock.patch.object(cli, "probe_video", side_effect=lambda *a, **k: video_info()), contextlib.redirect_stdout(io.StringIO()):
            return cli.run(self.args)

    def test_required_metadata_missing_and_declared_end_cannot_bypass_video(self):
        for key in ("id", "nlf_path", "video_path", "audio_path", "source_group", "split"):
            with self.subTest(key=key):
                row = {k: v for k, v in self.row.items() if k != key}
                row["media_end"] = 5.0625
                self.write_manifest([row])
                with self.assertRaises(ValueError):
                    cli.load_manifest(self.manifest)
        for changed in ({"source_group": ""}, {"split": "all"}, {"id": "../bad"}):
            self.write_manifest([{**self.row, **changed}])
            with self.assertRaises(ValueError):
                cli.load_manifest(self.manifest)

    def test_id_sid_compatibility_additional_metadata_preserved(self):
        row = dict(self.row)
        row["sid"] = row.pop("id")
        self.write_manifest([row])
        records, digest = cli.load_manifest(self.manifest)
        self.assertEqual(records[0]["id"], "synthetic")
        self.assertNotIn("sid", records[0])
        self.assertEqual(records[0]["source_record"], {"synthetic": True})
        self.assertEqual(records[0]["normalized_audio_offset_seconds"], "unknown")
        self.assertEqual(len(digest), 64)
        self.write_manifest([{**self.row, "sid": "different"}])
        with self.assertRaisesRegex(ValueError, "disagree"):
            cli.load_manifest(self.manifest)

    def test_parse_entire_manifest_without_file_stats_or_missing_unselected_failure(self):
        self.write_manifest([self.row, {**self.row, "id": "missing", "nlf_path": "/not-existing/nonselected.npz"}])
        with mock.patch.object(Path, "is_file", side_effect=AssertionError("parser must not stat")):
            rows, _ = cli.load_manifest(self.manifest)
        self.assertEqual(len(rows), 2)
        self.args.limit = 1
        result = self.run_cli()
        self.assertTrue(result["complete"])
        self.assertFalse(result["all_input_manifest_processed"])

    def test_duplicate_casefold_ids_rejected(self):
        self.write_manifest([self.row, {**self.row, "id": "SYNTHETIC"}])
        with self.assertRaisesRegex(ValueError, "colliding"):
            cli.load_manifest(self.manifest)

    def test_video_nlf_exact_count_fps_and_timestamps_gates(self):
        with np.load(self.source) as data:
            nlf = {key: data[key] for key in data.files}
        with mock.patch.object(cli, "probe_video", return_value=video_info()):
            end, metadata = cli.media_bounds(self.row, nlf, ffprobe="synthetic")
            self.assertEqual(end, 5.0625)
            self.assertTrue(metadata["nlf_video_timestamps_matched"])
            self.assertFalse(metadata["cache_video_identity_verified"])
            for changed, message in (({"pose": nlf["pose"][:-1]}, "count"),
                                     ({"fps": 15.}, "fps"), ({"timestamps": np.arange(81) / 16 + .1}, "timestamps")):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    cli.media_bounds(self.row, {**nlf, **changed}, ffprobe="synthetic")
            times_only = {key: value for key, value in nlf.items() if key != "fps"}
            times_only["timestamps"] = np.arange(81) / 16
            self.assertTrue(cli.media_bounds(self.row, times_only, ffprobe="synthetic")[1]["nlf_video_fps_matched"])
            with self.assertRaisesRegex(ValueError, "actual video end"):
                cli.media_bounds({**self.row, "media_end": 100}, nlf, ffprobe="synthetic")
        with mock.patch.object(cli, "probe_video", return_value={**video_info(), "media_end": None}):
            with self.assertRaisesRegex(ValueError, "cannot bypass"):
                cli.media_bounds({**self.row, "media_end": 5.0625}, nlf, ffprobe="synthetic")

    def test_complete_preprocess_once_whole_then_windows_raw_not_copied_and_resume(self):
        source_bytes = self.source.read_bytes()
        with mock.patch.object(cli, "preprocess", wraps=cli.preprocess) as preprocessing:
            result = self.run_cli()
            self.assertEqual(preprocessing.call_count, 1)
        self.assertTrue(result["complete"])
        self.assertEqual(result["chunk_count"], 2)
        out = Path(self.args.out)
        report = json.loads((out / "synthetic.json").read_bytes())
        self.assertFalse((out / "synthetic.raw.npz").exists())
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertTrue(report["media"]["media_end_verified_from_video"])
        self.assertEqual(report["id"], "synthetic")
        self.assertNotIn("sid", report)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["whole_clip_original"]["original_captions"][0]["start"], 0.)
        self.assertEqual(report["whole_clip_original"]["original_captions"][0]["end"], 5.)
        with np.load(out / "synthetic.coords.npz") as arrays:
            self.assertEqual(len(arrays["time"]), 101)
            self.assertEqual(len(arrays["source_time"]), 81)
        self.assertEqual(report["chunks"][1]["window_end"], 5.0625)
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        self.args.resume = True
        with mock.patch.object(cli, "preprocess", side_effect=AssertionError("resume must not recompute")):
            self.assertEqual(result, self.run_cli())
        self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_changed_manifest_or_worker_config_resume_refused(self):
        self.run_cli()
        self.args.resume = True
        self.write_manifest([{**self.row, "source_group": "different"}])
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            self.run_cli()
        self.write_manifest([self.row])
        self.args.workers = 2
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            self.run_cli()

    def test_tampered_artifacts_rejected_without_overwriting_valid_manifest(self):
        self.run_cli()
        self.args.resume = True
        out = Path(self.args.out)
        final_before = (out / "manifest.json").read_bytes()
        for name in ("synthetic.coords.npz", "synthetic.json", "chunks.jsonl", "manifest.json"):
            with self.subTest(name=name):
                path = out / name
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                try:
                    result = self.run_cli()
                    self.assertFalse(result["complete"])
                    self.assertEqual(result["completed_count"], 0)
                    self.assertIn("altered", result["errors"][0]["error"])
                    self.assertEqual((out / "manifest.json").read_bytes(), final_before)
                except ValueError as error:
                    self.assertIn("altered", str(error))
                finally:
                    path.write_bytes(before)

    def test_changed_nlf_video_or_audio_resume_fails_without_recompute(self):
        self.run_cli()
        self.args.resume = True
        final_before = (Path(self.args.out) / "manifest.json").read_bytes()
        for path in (self.source, self.video, self.audio):
            with self.subTest(path=path):
                before = path.read_bytes()
                path.write_bytes(before + b"changed source")
                with mock.patch.object(cli, "preprocess", side_effect=AssertionError("must refuse")):
                    result = self.run_cli()
                self.assertFalse(result["complete"])
                self.assertIn("resume input/protocol mismatch", result["errors"][0]["error"])
                self.assertEqual((Path(self.args.out) / "manifest.json").read_bytes(), final_before)
                path.write_bytes(before)

    def test_per_id_failure_isolated_and_other_records_continue(self):
        self.write_manifest([{**self.row, "id": "bad", "nlf_path": "nlf/missing.npz"}, self.row])
        result = self.run_cli()
        self.assertFalse(result["complete"])
        self.assertFalse(result["dataset_processed"])
        self.assertEqual(result["completed_count"], 1)
        self.assertEqual(result["errors"][0]["id"], "bad")
        self.assertTrue((Path(self.args.out) / "bad.error.json").exists())
        self.assertTrue((Path(self.args.out) / "synthetic.json").exists())

    def test_output_alias_and_legacy_reuse_refused(self):
        self.args.out = str(self.nlf / "bad")
        with self.assertRaisesRegex(ValueError, "separate"):
            self.run_cli()
        self.args.out = str(self.root / "legacy")
        Path(self.args.out).mkdir()
        marker = Path(self.args.out) / "existing.json"
        marker.write_text("legacy payload")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.run_cli()
        self.assertEqual(marker.read_text(), "legacy payload")

    def test_live_lock_and_symlink_outputs_refused(self):
        out = Path(self.args.out)
        out.mkdir()
        lock = out / ".run.lock"
        lock.write_text("other worker")
        with self.assertRaises(FileExistsError):
            self.run_cli()
        self.assertEqual(lock.read_text(), "other worker")
        self.args.out = str(self.root / "symlink_out")
        Path(self.args.out).mkdir()
        (Path(self.args.out) / "protocol.json").symlink_to(self.manifest)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_cli()

    def test_shard_exact_selection(self):
        self.write_manifest([{**self.row, "id": "a"}, {**self.row, "id": "b"}, {**self.row, "id": "c"}])
        self.args.shard = 1
        self.args.nshard = 2
        result = self.run_cli()
        self.assertEqual([r["id"] for r in result["records"]], ["b"])
        self.assertFalse(result["all_input_manifest_processed"])
        self.args.shard = 2
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.run_cli()


if __name__ == "__main__":
    unittest.main()
