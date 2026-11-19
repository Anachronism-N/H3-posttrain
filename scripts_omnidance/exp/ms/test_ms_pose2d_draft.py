"""Synthetic real-layout contract tests; no GPU, video, weights or production writes."""
import contextlib
import copy
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import ms_pose2d_draft as ms
from dwpose_full_extract import joint_layout


class PoseDraftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.names = {name: i for i, name in enumerate(joint_layout()["names"])}

    def fixture(self, times=None, sid="sample"):
        times = np.asarray(times if times is not None else np.arange(7) * .1, dtype=np.float64)
        n = len(times)
        xy = np.zeros((n, 134, 2), dtype=np.float32)
        for name, point in {"right_shoulder": (100, 100), "left_shoulder": (200, 100),
                            "right_hip": (100, 200), "left_hip": (200, 200),
                            "right_elbow": (80, 150), "left_elbow": (220, 150),
                            "right_wrist": (60, 200), "left_wrist": (240, 200),
                            "right_knee": (100, 300), "left_knee": (200, 300),
                            "right_ankle": (100, 400), "left_ankle": (200, 400)}.items():
            xy[:, self.names[name]] = point
        directory = self.root / sid
        directory.mkdir()
        row = {"id": sid, "split": "train", "npz_path": str(directory / "raw_wholebody134.npz"),
               "roi_manifest_path": str(directory / "roi_manifest.json")}
        data = {"xy": xy, "score": np.full((n, 134), 2.0, np.float32),
                "valid": np.ones((n, 134), dtype=bool), "tracked": np.ones(n, dtype=bool),
                "times": times, "source_timestamps": times + 10,
                "frame_indices": np.arange(n, dtype=np.int64),
                "joint_layout": np.array(json.dumps(joint_layout())),
                "binding_sha256": np.array("fixture-binding"), "video_sha256": np.array("fixture-video")}
        roi = {"id": sid, "source": {"path": str(directory / f"{sid}.mp4"), "sha256": "fixture-video"},
               "raw": {"path": row["npz_path"]}, "binding_sha256": "fixture-binding", "video_sha256": "fixture-video",
               "model_ref": {"joint_layout": joint_layout()}, "anatomical_side_guaranteed": False,
               "selected_frames": [{"frame_index": i, "time": float(t), "source_timestamp": float(t + 10),
                                    "tracked": True} for i, t in enumerate(times)],
               "probe": {"time_origin": 10, "frame_times": times.tolist(),
                         "actual_video_pts": (times + 10).tolist(), "duration": float(times[-1])}}
        return row, roi, data

    def persist(self, row, roi, data):
        np.savez_compressed(row["npz_path"], **data)
        Path(row["roi_manifest_path"]).write_text(json.dumps(roi), encoding="utf-8")

    def draft(self, row, roi, data, bars=None):
        mask = ms.validate_observations(row, roi, data)
        return ms.generate_draft(row, roi, data, mask, bars)

    def wrist_events(self, draft, direction=None):
        return [e for e in draft["events"] if e["support"]["signal"] == "right_wrist_relative_shoulder_vertical"
                and (direction is None or e["direction"] == direction)]

    def test_upward_dynamic_and_raw_score_above_one(self):
        row, roi, data = self.fixture()
        data["xy"][:, self.names["right_wrist"], 1] -= np.arange(7) * 10
        self.persist(row, roi, data)
        roi2, loaded, mask = ms.load_observations(row)
        draft = ms.generate_draft(row, roi2, loaded, mask)
        events = self.wrist_events(draft, -1)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertAlmostEqual(event["start"], 0)
        self.assertAlmostEqual(event["end"], .6)
        self.assertIn("一侧", event["text"])
        self.assertIn("上移", event["text"])
        self.assertEqual(event["support"]["frame_indices"], list(range(7)))
        self.assertEqual(event["support"]["dt_seconds"], np.diff(data["times"]).tolist())
        self.assertEqual(draft["status"], "draft_generated")
        for key, value in ms.FLAGS.items():
            self.assertIs(draft[key], value)
        self.assertFalse(draft["anatomical_side_guaranteed"])
        self.assertFalse(draft["beat_alignment_claimed"])

    def test_reversal_is_two_separate_observed_events(self):
        row, roi, data = self.fixture(np.arange(9) * .1)
        data["xy"][:, self.names["right_wrist"], 1] += np.array([0, -10, -20, -30, -40, -30, -20, -10, 0])
        events = self.wrist_events(self.draft(row, roi, data))
        self.assertEqual([e["direction"] for e in events], [-1, 1])
        self.assertAlmostEqual(events[0]["end"], .4)
        self.assertAlmostEqual(events[1]["start"], .4)

    def test_stationary_and_single_frame_not_dynamic(self):
        for sid, times in (("static", None), ("single", [0])):
            row, roi, data = self.fixture(times, sid)
            draft = self.draft(row, roi, data)
            self.assertEqual(draft["events"], [])
            self.assertEqual(draft["status"], "needs_regeneration")
            self.assertIn("unknown", draft["ms_text"])

    def test_invalid_nan_frame_breaks_events(self):
        row, roi, data = self.fixture(np.arange(9) * .1)
        data["xy"][:, self.names["right_wrist"], 1] -= np.arange(9) * 10
        data["valid"][4] = False
        data["tracked"][4] = False
        data["xy"][4] = np.nan
        data["score"][4] = np.nan
        roi["selected_frames"][4]["tracked"] = False
        draft = self.draft(row, roi, data)
        self.assertEqual(draft["status"], "partial")
        for event in draft["events"]:
            self.assertNotIn(4, event["support"]["frame_indices"])
            self.assertFalse(event["start"] < .4 < event["end"])
        self.assertTrue(draft["observation"]["unknown_intervals"])
        json.dumps(draft, allow_nan=False)

    def test_all_invalid_nan_stays_unknown_not_zero_fact(self):
        row, roi, data = self.fixture()
        data["valid"][:] = False
        data["tracked"][:] = False
        data["xy"][:] = np.nan
        data["score"][:] = np.nan
        for frame in roi["selected_frames"]:
            frame["tracked"] = False
        draft = self.draft(row, roi, data)
        self.assertEqual(draft["status"], "needs_regeneration")
        self.assertEqual(draft["events"], [])
        self.assertEqual(draft["observation"]["observed_spans"], [])
        self.assertTrue(all(w["unknown"] for w in draft["windows"]))
        self.assertTrue(np.isnan(data["xy"]).all())
        json.dumps(draft, allow_nan=False)

    def test_effective_valid_nan_rejected(self):
        row, roi, data = self.fixture()
        for key in ("xy", "score"):
            bad = copy.deepcopy(data)
            bad[key][0, self.names["right_wrist"]] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                self.draft(row, roi, bad)

    def test_camera_translation_not_dancer_motion(self):
        row, roi, data = self.fixture()
        data["xy"] += np.arange(7, dtype=np.float32)[:, None, None] * np.array([120, -80], dtype=np.float32)
        draft = self.draft(row, roi, data)
        self.assertEqual(draft["events"], [])
        self.assertEqual(draft["status"], "needs_regeneration")

    def test_uniform_camera_zoom_not_relative_motion(self):
        row, roi, data = self.fixture()
        data["xy"] *= (1 + np.arange(7, dtype=np.float32) * .1)[:, None, None]
        self.assertEqual(self.draft(row, roi, data)["events"], [])

    def test_variable_dt_uses_real_seconds(self):
        row, roi, data = self.fixture([0, .04, .12, .21, .35, .50, .69])
        data["xy"][:, self.names["right_wrist"], 1] -= data["times"] * 100
        event = self.wrist_events(self.draft(row, roi, data), -1)[0]
        self.assertAlmostEqual(event["support"]["mean_speed_per_s"], -1, places=5)
        self.assertEqual(event["support"]["dt_seconds"], np.diff(data["times"]).tolist())

    def test_large_time_hole_not_spliced(self):
        row, roi, data = self.fixture([0, .1, .2, 2, 2.1, 2.2])
        data["xy"][3:, self.names["right_wrist"], 1] -= 100
        draft = self.draft(row, roi, data)
        self.assertEqual(self.wrist_events(draft), [])
        self.assertTrue(any(w["reason"] == "time_gap" for w in draft["observation"]["unknown_intervals"]))

    def test_degenerate_body_scale_unknown(self):
        row, roi, data = self.fixture()
        data["xy"][:, self.names["left_shoulder"]] = data["xy"][:, self.names["right_shoulder"]]
        data["xy"][:, self.names["right_wrist"], 1] -= np.arange(7) * 10
        draft = self.draft(row, roi, data)
        self.assertEqual(draft["events"], [])
        self.assertEqual(draft["observation"]["body_scale_valid_frames"], 0)

    def test_projected_elbow_and_knee_angle_change(self):
        row, roi, data = self.fixture()
        angles = np.radians(np.linspace(100, 160, 7))
        # A=(100,100), B=(100,150); moving C opens the image-plane angle.
        data["xy"][:, self.names["right_elbow"]] = [100, 150]
        data["xy"][:, self.names["right_wrist"], 0] = 100 + 50 * np.sin(angles)
        data["xy"][:, self.names["right_wrist"], 1] = 150 - 50 * np.cos(angles)
        data["xy"][:, self.names["right_ankle"], 0] = 100 + 80 * np.sin(angles)
        data["xy"][:, self.names["right_ankle"], 1] = 300 - 80 * np.cos(angles)
        events = self.draft(row, roi, data)["events"]
        for joint in ("elbow", "knee"):
            found = [e for e in events if e["support"]["signal"] == f"right_{joint}_projected_angle"]
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["direction"], 1)
            self.assertIn("二维投影", found[0]["text"])

    def test_identity_and_pts_mismatch_rejected(self):
        row, roi, data = self.fixture()
        variants = []
        for key in ("id", "binding_sha256", "video_sha256"):
            bad_roi = copy.deepcopy(roi)
            bad_roi[key] = "wrong"
            variants.append((row, bad_roi, data))
        bad = copy.deepcopy(data)
        bad["times"][2] += .01
        variants.append((row, roi, bad))
        bad = copy.deepcopy(data)
        bad["source_timestamps"][2] += .01
        variants.append((row, roi, bad))
        bad_roi = copy.deepcopy(roi)
        bad_roi["selected_frames"][2]["frame_index"] = 12
        variants.append((row, bad_roi, data))
        bad_roi = copy.deepcopy(roi)
        bad_roi["probe"]["actual_video_pts"][2] += .01
        variants.append((row, bad_roi, data))
        bad = copy.deepcopy(data)
        bad["times"][2] = bad["times"][1]
        variants.append((row, roi, bad))
        for args in variants:
            with self.subTest(), self.assertRaises(ValueError):
                self.draft(*args)

    def test_3d_and_fake_mapping_rejected(self):
        row, roi, data = self.fixture()
        bad = copy.deepcopy(data)
        bad["xy"] = np.zeros((7, 134, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "2D"):
            self.draft(row, roi, bad)
        bad = copy.deepcopy(data)
        layout = joint_layout()
        layout["names"][2], layout["names"][5] = layout["names"][5], layout["names"][2]
        bad["joint_layout"] = np.array(json.dumps(layout))
        with self.assertRaisesRegex(ValueError, "134"):
            self.draft(row, roi, bad)

    def test_bar_windows_only_map_observed_event_and_outside_unknown(self):
        row, roi, data = self.fixture()
        data["xy"][:, self.names["right_wrist"], 1] -= np.arange(7) * 10
        plan = self.root / "plan.json"
        binding = {"id": row["id"], "video_sha256": roi["video_sha256"],
                   "video_pts_offset": 10, "media_start": 0, "media_end": 3}
        roi["probe"]["duration"] = 3
        raw_windows = [{"start": 0, "end": .3}, {"start": 2, "end": 3}]
        plan.write_text(json.dumps({"id": row["id"], "binding": binding, "windows": raw_windows,
                                   "window_manifest": {"id": row["id"], "binding": binding,
                                                       "time_basis": "global_media_seconds", "windows": raw_windows}}))
        row["bar_plan_path"] = str(plan)
        bars = ms.load_bar_windows(row, roi)
        draft = self.draft(row, roi, data, bars)
        self.assertEqual(draft["status"], "partial")
        self.assertTrue(draft["windows"][0]["event_refs"])
        self.assertEqual(draft["windows"][1]["event_refs"], [])
        self.assertTrue(draft["windows"][1]["unknown"])
        self.assertAlmostEqual(self.wrist_events(draft)[0]["end"], .6)
        self.assertFalse(draft["beat_alignment_claimed"])
        plan.write_text(json.dumps({"id": "wrong", "windows": [{"start": 0, "end": 1}]}))
        with self.assertRaisesRegex(ValueError, "ID"):
            ms.load_bar_windows(row, roi)

    def test_real_window_manifest_binding_and_pts_offset_rejected(self):
        row, roi, data = self.fixture()
        roi["probe"]["actual_video_pts_sha256"] = "roi-hash-is-an-absolute-times-and-ticks-object"
        relative_hash = hashlib.sha256((json.dumps(roi["probe"]["frame_times"], ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
        binding = {"id": row["id"], "video_sha256": roi["video_sha256"],
                   "video_pts_offset": 10, "media_start": 0, "media_end": .6,
                   "actual_video_pts_sha256": relative_hash}
        windows = [{"start": 0, "end": .6}]
        plan = {"id": row["id"], "binding": binding, "windows": windows,
                "window_manifest": {"id": row["id"], "binding": binding,
                                    "time_basis": "global_media_seconds", "windows": windows}}
        path = self.root / "real-plan.json"
        row["bar_plan_path"] = str(path)
        path.write_text(json.dumps(plan))
        loaded = ms.load_bar_windows(row, roi)
        self.assertEqual(loaded[0]["basis"], "provided_media_window")
        for key in ("audio_video_alignment_verified", "beat_phase_verified", "motions_on_beats_verified",
                    "one_window_is_exactly_one_bar", "window_boundaries_are_action_boundaries"):
            self.assertFalse(loaded[0][key])
        variants = []
        bad = copy.deepcopy(plan)
        bad["window_manifest"]["binding"]["video_sha256"] = "wrong"
        variants.append(bad)
        bad = copy.deepcopy(plan)
        bad["binding"]["video_pts_offset"] = 0
        variants.append(bad)
        bad = copy.deepcopy(plan)
        bad["window_manifest"]["windows"] = [{"start": 0, "end": .5}]
        variants.append(bad)
        bad = copy.deepcopy(plan)
        bad["binding"]["actual_video_pts_sha256"] = "wrong-relative-pts-hash"
        bad["window_manifest"]["binding"] = copy.deepcopy(bad["binding"])
        variants.append(bad)
        for bad in variants:
            path.write_text(json.dumps(bad))
            with self.subTest(), self.assertRaises(ValueError):
                ms.load_bar_windows(row, roi)

    def test_short_cache_does_not_extrapolate_video(self):
        row, roi, data = self.fixture()
        roi["probe"]["duration"] = 30
        data["xy"][:, self.names["right_wrist"], 1] -= np.arange(7) * 10
        draft = self.draft(row, roi, data)
        self.assertTrue(all(e["end"] <= .600001 for e in draft["events"]))
        self.assertEqual(draft["observation"]["unknown_intervals"][-1],
                         {"start": float(data["times"][-1]), "end": 30.0, "reason": "outside_cached_observations"})

    def test_write_once_never_overwrites(self):
        row, roi, data = self.fixture()
        self.persist(row, roi, data)
        out = self.root / "out"
        row["diagnostic_not_production"] = True
        first = ms.process_row(row, out)
        self.assertTrue(first["diagnostic_not_production"])
        self.assertGreater(first["profile_seconds"]["read_and_validate"], 0)
        self.assertGreater(first["profile_seconds"]["generate"], 0)
        self.assertGreater(first["profile_seconds"]["write_back"], 0)
        path = Path(first["path"])
        before = path.read_bytes()
        data["xy"][:, self.names["right_wrist"], 1] -= np.arange(7) * 10
        self.persist(row, roi, data)
        second = ms.process_row(row, out)
        self.assertEqual(second["status"], "existing_not_overwritten")
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            ms.write_new(path, {"different": True})
        self.assertEqual(path.read_bytes(), before)

    def test_cli_default_three_sources_and_e2e_counts(self):
        rows = []
        for i in range(4):
            row, roi, data = self.fixture(sid=f"sample{i}")
            self.persist(row, roi, data)
            rows.append(row)
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        output = io.StringIO()
        out = self.root / "out"
        with contextlib.redirect_stdout(output):
            code = ms.main(["--manifest", str(manifest), "--out", str(out)])
        self.assertEqual(code, 0)
        summary = json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(summary["sources_attempted"], 3)
        self.assertEqual(summary["sources_written"], 3)
        self.assertEqual(summary["event_count"], 0)
        self.assertGreater(summary["window_count"], 0)
        self.assertGreater(summary["elapsed_e2e_seconds"], 0)
        self.assertFalse((out / "train" / "sample3" / "draft.json").exists())

    def test_npz_pickle_fields_rejected(self):
        row, roi, data = self.fixture()
        data["joint_layout"] = np.array(joint_layout(), dtype=object)
        self.persist(row, roi, data)
        with self.assertRaisesRegex(ValueError, "allow_pickle=False"):
            ms.load_observations(row)

    def test_recorded_pts_ticks_mismatch_rejected(self):
        row, roi, data = self.fixture()
        ticks = list(range(len(data["times"])))
        roi["probe"]["actual_video_pts_ticks"] = ticks
        for frame, tick in zip(roi["selected_frames"], ticks):
            frame["source_pts"] = tick
        self.draft(row, roi, data)
        roi["selected_frames"][2]["source_pts"] = 999
        with self.assertRaisesRegex(ValueError, "source_pts"):
            self.draft(row, roi, data)

    def test_missing_hand_evidence_is_not_window_success(self):
        row, roi, data = self.fixture()
        data["valid"][:, [self.names["left_wrist"], self.names["right_wrist"]]] = False
        draft = self.draft(row, roi, data)
        self.assertEqual(draft["status"], "needs_regeneration")
        self.assertEqual(draft["events"], [])
        self.assertEqual(draft["windows"][0]["status"], "needs_regeneration")
        self.assertIn("unknown", draft["windows"][0]["text"])

    def test_unsafe_identifiers_and_no_implicit_all_rejected(self):
        row, roi, data = self.fixture()
        row["split"] = "../elsewhere"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            ms.process_row(row, self.root / "out")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ms.main(["--manifest", "unused", "--out", "unused", "--limit", "0"])


if __name__ == "__main__":
    unittest.main()
