"""CPU-only v2 contract tests, using explicitly synthetic fullframe fixtures."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ms_fullframe_pose2d_draft_v1 as v1
import ms_fullframe_pose2d_draft_v2 as adapter
import test_ms_fullframe_pose2d_draft_v1 as fixtures_v1


class FullframeV2Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.row, self.manifest, self.data = self.fixture("case81")

    def fixture(self, sid, split="train"):
        row, manifest, data = fixtures_v1.FullframeDraftTest.fixture(self, sid, split)
        data["boxes"] = data.pop("bbox")
        data["schema"] = np.array(adapter.BOXES_SCHEMA)
        data.update(width=np.array(manifest["width"]), height=np.array(manifest["height"]),
                    duration=np.array(manifest["duration_seconds"]),
                    source_pts=np.arange(manifest["frame_count"], dtype=np.int64) * 1024,
                    source_time_base=np.array("1/16384"))
        data["decoder_frame_timestamps"] = np.column_stack((data["source_timestamps"], data["source_timestamps"] + 0.0625))
        manifest["input_schema"] = adapter.BOXES_SCHEMA
        manifest["metadata_provenance"]["actual_probe"] = {"frames": [
            {"best_effort_timestamp": int(pts), "best_effort_timestamp_time": str(t), "pkt_duration_time": "0.0625"}
            for pts, t in zip(data["source_pts"], data["source_timestamps"])]}
        return row, manifest, data

    write_fixture = fixtures_v1.FullframeDraftTest.write_fixture

    def draft(self):
        mask = adapter.validate_observations(self.row, self.manifest, self.data)
        return adapter.generate_draft(self.row, self.manifest, self.data, mask)

    def test_boxes_only_mapping_and_original_provenance_binding(self):
        before = {key: (value.dtype.str, value.shape, value.tobytes()) for key, value in self.data.items()}
        manifest_before = copy.deepcopy(self.manifest)
        with patch.object(v1, "validate_observations", wraps=v1.validate_observations) as validate:
            draft = self.draft()
        bridged_manifest, bridged_data = validate.call_args.args[1:]
        self.assertEqual(bridged_manifest["input_schema"], adapter.VALIDATION_SCHEMA)
        self.assertEqual(adapter.rules.scalar_text(bridged_data, "schema"), adapter.VALIDATION_SCHEMA)
        np.testing.assert_array_equal(bridged_data["bbox"], self.data["boxes"])
        self.assertFalse(np.shares_memory(bridged_data["bbox"], self.data["boxes"]))
        self.assertNotIn("boxes", bridged_data)
        self.assertEqual(self.manifest, manifest_before)
        self.assertEqual(before, {key: (value.dtype.str, value.shape, value.tobytes()) for key, value in self.data.items()})
        self.assertNotIn("bbox", self.data)
        self.assertEqual(draft["schema"], adapter.SCHEMA)
        source = draft["source"]
        self.assertEqual(source["input_schema"], adapter.BOXES_SCHEMA)
        self.assertEqual(source["fullframe_binding"], v1.fullframe_binding(self.manifest, self.data))
        self.assertNotEqual(source["fullframe_binding"]["observations_sha256"], v1.fullframe_binding(bridged_manifest, bridged_data)["observations_sha256"])
        self.assertEqual(source["adapter"]["mapping"], {"boxes": "bbox", "schema": adapter.VALIDATION_SCHEMA})
        self.assertFalse(source["adapter"]["original_bbox_field_claimed"])
        self.assertFalse(source["adapter"]["extraction_recipe_equivalence_claimed"])
        self.assertFalse(source["adapter"]["original_cache_rewritten"])
        self.assertFalse(source["provenance"]["media_or_weights_accessed"])
        self.assertNotIn("roi_manifest_path", source)
        self.assertNotIn("bar_plan_path", source)
        self.assertEqual(draft["status"], "partial")
        self.assertFalse(draft["beat_alignment_claimed"])
        for key, expected in adapter.FLAGS.items():
            self.assertIs(draft[key], expected)

    def test_boxes_shape_dtype_and_tracked_finite(self):
        for boxes in (self.data["boxes"][:-1], self.data["boxes"][:, :3],
                      self.data["boxes"].reshape(81, 1, 4), self.data["boxes"].astype(np.int32)):
            with self.subTest(shape=boxes.shape, dtype=boxes.dtype):
                data = copy.deepcopy(self.data)
                data["boxes"] = boxes
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)
        self.data["boxes"][4] = np.nan
        with self.assertRaises(ValueError):
            self.draft()
        self.data["tracked"][4] = False
        mask = adapter.validate_observations(self.row, self.manifest, self.data)
        self.assertFalse(mask[4].any())
        self.assertTrue(np.isnan(self.data["boxes"][4]).all())

    def test_unknown_schema_and_arbitrary_bbox_alias_rejected(self):
        for schema in ("unknown", "ms_fullframe_pose2d_npz_v2", "ms_pose2d_cache_v1"):
            with self.subTest(schema=schema):
                data = copy.deepcopy(self.data)
                data["schema"] = np.array(schema)
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)
        self.data["bbox"] = self.data.pop("boxes")
        with self.assertRaises(ValueError):
            self.draft()

    def test_original_manifest_schema_mismatch_and_optional_match(self):
        self.manifest["input_schema"] = adapter.VALIDATION_SCHEMA
        with self.assertRaisesRegex(ValueError, "original input_schema mismatch"):
            self.draft()
        del self.manifest["input_schema"]
        self.assertEqual(self.draft()["source"]["input_schema"], adapter.BOXES_SCHEMA)

    def test_width_height_video_duration_metadata_mismatch(self):
        for key in ("width", "height", "duration"):
            for value in (np.array(float(self.data[key]) + 1), np.array(np.nan), np.array([float(self.data[key])])):
                with self.subTest(key=key, value=value):
                    data = copy.deepcopy(self.data)
                    data[key] = value
                    with self.assertRaises(ValueError):
                        adapter.validate_observations(self.row, self.manifest, data)
        self.manifest["audio_duration"] = 20.0
        draft = self.draft()
        self.assertEqual(draft["observation"]["unknown_intervals"][-1]["end"], 5.0625)

    def test_source_pts_time_base_wrong_and_unpaired(self):
        changes = [("source_pts", self.data["source_pts"] + 1),
                   ("source_pts", self.data["source_pts"].astype(float)),
                   ("source_pts", self.data["source_pts"][:-1]),
                   ("source_time_base", np.array("1/16000")),
                   ("source_time_base", np.array("0/16384")),
                   ("source_time_base", np.array("1/0"))]
        for key, value in changes:
            with self.subTest(key=key, value=str(value)[:40]):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)
        for key in ("source_pts", "source_time_base"):
            data = copy.deepcopy(self.data)
            del data[key]
            with self.assertRaises(ValueError):
                adapter.validate_observations(self.row, self.manifest, data)

    def test_recorded_pts_ticks_and_optional_manifest_pts_mismatch(self):
        self.manifest["metadata_provenance"]["actual_probe"]["frames"][3]["best_effort_timestamp"] += 1
        with self.assertRaises(ValueError):
            self.draft()
        self.manifest["metadata_provenance"]["actual_probe"]["frames"][3]["best_effort_timestamp"] -= 1
        for key, value in (("source_pts", (self.data["source_pts"] + 1).tolist()), ("source_time_base", "1/16000")):
            manifest = copy.deepcopy(self.manifest)
            manifest[key] = value
            with self.assertRaises(ValueError):
                adapter.validate_observations(self.row, manifest, self.data)

    def test_decoder_shape_start_end_and_real_packet_duration(self):
        changes = [self.data["decoder_frame_timestamps"][:, 0],
                   self.data["decoder_frame_timestamps"][:-1],
                   self.data["decoder_frame_timestamps"].astype(int)]
        for index, value in (((4, 0), 0.26), ((4, 1), 0.24), ((4, 1), 0.25), ((4, 1), 0.32), ((4, 1), 0.31), ((80, 1), 6.0), ((4, 0), np.nan)):
            bad = self.data["decoder_frame_timestamps"].copy()
            bad[index] = value
            changes.append(bad)
        for value in changes:
            with self.subTest(shape=value.shape):
                data = copy.deepcopy(self.data)
                data["decoder_frame_timestamps"] = value
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)
        self.manifest["metadata_provenance"]["actual_probe"]["frames"][3]["pkt_duration_time"] = "0.06"
        with self.assertRaises(ValueError):
            self.draft()

    def test_decoder_evidence_required_no_guessed_fps(self):
        del self.manifest["metadata_provenance"]["actual_probe"]
        with self.assertRaisesRegex(ValueError, "actual_probe.frames"):
            self.draft()
        del self.data["decoder_frame_timestamps"]
        self.assertEqual(self.draft()["source"]["input_schema"], adapter.BOXES_SCHEMA)

    def test_nonzero_source_origin_not_relative_decoder_pts(self):
        self.data["source_timestamps"] += 2.0
        self.data["source_pts"] += 32768
        self.data["decoder_frame_timestamps"] += 2.0
        self.manifest["source_timestamps"] = self.data["source_timestamps"].tolist()
        self.manifest["pts_origin"] = 2.0
        for frame in self.manifest["metadata_provenance"]["actual_probe"]["frames"]:
            frame["best_effort_timestamp"] += 32768
            frame["best_effort_timestamp_time"] = str(float(frame["best_effort_timestamp_time"]) + 2.0)
        self.assertEqual(self.draft()["observation"]["native_pts_origin"], 2.0)
        self.data["decoder_frame_timestamps"] -= 2.0
        with self.assertRaises(ValueError):
            self.draft()

    def test_effective_valid_score_finite_threshold_not_relaxed(self):
        for key, value in (("score", np.nan), ("score", 0.3), ("xy", np.nan)):
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data[key][4, 4] = value
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)
        self.manifest["score_threshold"] = 0.2
        with self.assertRaises(ValueError):
            self.draft()

    def test_oob_toes_keep_arm_events_and_original_bytes_valid_binding(self):
        toe = adapter.rules.joint_layout()["names"].index("left_big_toe")
        self.data["xy"][4:6, toe, 1] = 1282.5
        self.write_fixture()
        paths = [Path(self.row[key]) for key in ("npz_path", "fullframe_manifest_path")]
        before_disk = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
        before_arrays = {key: value.tobytes() for key, value in self.data.items()}
        manifest, data, mask = adapter.load_observations(self.row)
        self.assertFalse(mask[4:6, toe].any())
        self.assertTrue(data["valid"][4:6, toe].all())
        draft = adapter.generate_draft(self.row, manifest, data, mask)
        self.assertTrue(any(e["support"]["signal"].startswith("right_wrist") for e in draft["events"]))
        self.assertEqual(draft["status"], "partial")
        evidence = draft["observation"]["spatial_evidence"]
        self.assertEqual(evidence["excluded_point_count"], 2)
        self.assertEqual(evidence["affected_frame_indices"], [4, 5])
        self.assertEqual(evidence["affected_joint_indices"], [toe])
        self.assertFalse(evidence["whole_source_unusable_claimed"])
        self.assertTrue(evidence["local_evidence_limited"])
        self.assertTrue(draft["source"]["provenance"]["local_spatial_evidence_limited"])
        self.assertEqual(draft["source"]["fullframe_binding"], v1.fullframe_binding(manifest, data))
        self.assertEqual(draft["source"]["input_schema"], adapter.BOXES_SCHEMA)
        self.assertEqual(before_arrays, {key: value.tobytes() for key, value in data.items()})
        self.assertEqual(before_disk, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])
        self.assertEqual(draft["observation"]["unknown_intervals"][-1],
                         {"start": 5.0, "end": 5.0625, "reason": "outside_cached_observations"})

    def test_oob_wrist_never_event_support_and_signal_support_reported(self):
        self.data["xy"][40, 4, 1] = 1280.0
        mask = adapter.validate_observations(self.row, self.manifest, self.data)
        draft = adapter.generate_draft(self.row, self.manifest, self.data, mask)
        for event in draft["events"]:
            if 4 in event["support"]["joint_indices"]:
                self.assertNotIn(40, event["support"]["frame_indices"])
        support = draft["observation"]["spatial_evidence"]["per_signal_support"]
        wrist_support = next(s for s in support if s["signal"] == "right_wrist_relative_shoulder_horizontal")
        self.assertEqual(wrist_support["excluded_support_frame_indices"], [40])
        self.assertEqual(wrist_support["usable_frame_count"], 80)

    def test_raw_nan_or_low_score_cannot_hide_behind_out_of_frame(self):
        for key, index, value in (("xy", (4, 4, 0), np.nan), ("score", (4, 4), np.nan),
                                  ("score", (4, 4), 0.3), ("score", (4, 4), 0.2)):
            data = copy.deepcopy(self.data)
            data["xy"][4, 4, 1] = 1282.0
            data[key][index] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                adapter.validate_observations(self.row, self.manifest, data)

    def test_all_oob_needs_regeneration_and_no_rule_support(self):
        self.data["xy"][:, :, 1] = 1280.0
        mask = adapter.validate_observations(self.row, self.manifest, self.data)
        self.assertFalse(mask.any())
        draft = adapter.generate_draft(self.row, self.manifest, self.data, mask)
        self.assertEqual(draft["status"], "needs_regeneration")
        self.assertFalse(draft["events"])
        evidence = draft["observation"]["spatial_evidence"]
        self.assertEqual(evidence["excluded_point_count"], 81 * 134)
        self.assertTrue(all(s["below_minimum_total_frames"] for s in evidence["per_signal_support"]))
        self.assertTrue(self.data["valid"].all())
        self.assertEqual(draft["source"]["fullframe_binding"], v1.fullframe_binding(self.manifest, self.data))

    def test_oob_does_not_skip_bbox_identity_layout_pts_or_flags_validation(self):
        self.data["xy"][4, 4, 1] = 1282.0
        for key, value in (("boxes", np.full((81, 4), np.nan)), ("source_pts", self.data["source_pts"] + 1),
                           ("id", np.array("wrong")), ("joint_layout", np.array("{}")),
                           ("quality_approved", np.array(True))):
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.validate_observations(self.row, self.manifest, data)

    def test_oob_policy_applies_only_to_explicit_supported_schemas(self):
        for schema in v1.INPUT_SCHEMAS:
            row, manifest, data = fixtures_v1.FullframeDraftTest.fixture(self, schema)
            data["schema"] = np.array(schema)
            data["xy"][40, 4, 0] = -1.0
            mask = adapter.validate_observations(row, manifest, data)
            self.assertFalse(mask[40, 4])
            draft = adapter.generate_draft(row, manifest, data, mask)
            self.assertEqual(draft["source"]["input_schema"], schema)
            self.assertEqual(draft["source"]["adapter"]["mapping"], {})
            self.assertEqual(draft["status"], "partial")
            self.assertEqual(draft["observation"]["spatial_evidence"]["excluded_point_count"], 1)
            self.assertEqual(draft["source"]["fullframe_binding"], v1.fullframe_binding(manifest, data))
            with self.assertRaises(ValueError):
                v1.validate_observations(row, manifest, data)

    def test_raw_score_above_one_and_invalid_nan_preserved(self):
        self.assertGreater(self.data["score"].max(), 1)
        self.data["valid"][40, 4] = False
        self.data["xy"][40, 4] = np.nan
        self.data["score"][40, 4] = np.nan
        self.write_fixture()
        manifest, data, mask = adapter.load_observations(self.row)
        self.assertFalse(mask[40, 4])
        self.assertTrue(np.isnan(data["xy"][40, 4]).all())
        self.assertTrue(np.isnan(data["score"][40, 4]))
        self.assertEqual(float(data["score"][0, 0]), 2.0)
        draft = adapter.generate_draft(self.row, manifest, data, mask)
        for event in draft["events"]:
            if event["support"]["signal"].startswith("right_wrist"):
                self.assertNotIn(40, event["support"]["frame_indices"])
        self.assertEqual(draft["status"], "partial")

    def test_frozen_v1_identity_layout_native_coverage_time_and_flags(self):
        mutations = {"id": np.array("wrong"), "split": np.array("val"), "video_sha256": np.array("b" * 64),
                     "joint_layout": np.array("{}"), "quality_approved": np.array(True),
                     "frame_indices": self.data["frame_indices"] + 1,
                     "times": self.data["times"] + 0.01, "source_timestamps": self.data["source_timestamps"] + 1}
        for key, value in mutations.items():
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = value
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)
        row = dict(self.row, split="val")
        with self.assertRaises(ValueError):
            adapter.validate_observations(row, self.manifest, self.data)

    def test_03_and_both_v1_schemas_still_work_without_new_fields(self):
        row, manifest, data = fixtures_v1.FullframeDraftTest.fixture(self, "old03")
        for schema in v1.INPUT_SCHEMAS:
            with self.subTest(schema=schema):
                data["schema"] = np.array(schema)
                manifest["input_schema"] = schema
                self.write_fixture(row, manifest, data)
                loaded_manifest, loaded_data, mask = adapter.load_observations(row)
                np.testing.assert_array_equal(mask, v1.validate_observations(row, manifest, data))
                draft = adapter.generate_draft(row, loaded_manifest, loaded_data, mask)
                old = v1.generate_draft(row, manifest, data, mask)
                self.assertEqual(draft["events"], old["events"])
                self.assertEqual(draft["source"]["fullframe_binding"], old["source"]["fullframe_binding"])
                self.assertEqual(draft["source"]["input_schema"], schema)
                self.assertEqual(draft["source"]["adapter"]["mapping"], {})
                self.assertEqual(draft["schema"], adapter.SCHEMA)

    def test_no_disk_rewrite_old_roi_access_and_write_once(self):
        self.write_fixture()
        inputs = [Path(self.row[key]) for key in ("npz_path", "fullframe_manifest_path")]
        before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs]
        with patch.object(adapter.rules, "load_observations", side_effect=AssertionError("Old ROI loader called")), \
             patch.object(adapter.rules, "load_bar_windows", side_effect=AssertionError("Old bars called")):
            first = adapter.process_row(self.row, self.root / "out")
        self.assertGreater(first["event_count"], 0)
        self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in inputs])
        output = Path(first["path"])
        output_before = output.read_bytes()
        with patch.object(adapter, "load_observations", side_effect=AssertionError("Existing output must skip reload")):
            result = adapter.process_row(self.row, self.root / "out")
        self.assertEqual(result["status"], "existing_not_overwritten")
        self.assertEqual(output.read_bytes(), output_before)

    def test_npz_recorded_hash_and_no_pickle(self):
        self.write_fixture()
        self.manifest["npz_sha256"] = hashlib.sha256(Path(self.row["npz_path"]).read_bytes()).hexdigest()
        self.write_fixture()
        adapter.load_observations(self.row)
        self.manifest["npz_sha256"] = "b" * 64
        self.write_fixture()
        with self.assertRaises(ValueError):
            adapter.load_observations(self.row)
        del self.manifest["npz_sha256"]
        self.data["object_trap"] = np.array({"not": "allowed"}, dtype=object)
        self.write_fixture()
        with self.assertRaises(ValueError):
            adapter.load_observations(self.row)

    def test_cli_default_three_rejection_separate_and_no_all_mode(self):
        rows = []
        for sid in ("first", "second", "third"):
            row, manifest, data = self.fixture(sid)
            self.write_fixture(row, manifest, data)
            rows.append(row)
        path = self.root / "input.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "not JSON\n", encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = adapter.main(["--manifest", str(path), "--out", str(self.root / "cli_out")])
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        summary = json.loads(stdout.getvalue().splitlines()[-1])
        self.assertEqual(summary["sources_attempted"], 3)
        self.assertEqual(summary["sources_written"], 3)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = adapter.main(["--manifest", str(path), "--out", str(self.root / "cli_out"), "--limit", "4"])
        self.assertEqual(code, 1)
        rejection = json.loads(stderr.getvalue())
        self.assertEqual(rejection["status"], "rejected")
        self.assertGreaterEqual(rejection["profile_seconds"]["failed_e2e"], 0)
        self.assertNotIn('"status": "rejected"', stdout.getvalue())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            adapter.main(["--manifest", str(path), "--out", str(self.root / "cli_out"), "--limit", "0"])


if __name__ == "__main__":
    unittest.main()
