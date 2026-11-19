"""CPU-only fullframe contract tests; synthetic fixtures are explicitly test-only."""
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

import ms_fullframe_pose2d_draft_v1 as adapter


class FullframeDraftTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.row, self.manifest, self.data = self.fixture("case81")

    def fixture(self, sid, split="train"):
        folder = self.root / split / sid
        folder.mkdir(parents=True)
        row = {"id": sid, "split": split, "npz_path": str(folder / "wholebody.npz"),
               "fullframe_manifest_path": str(folder / "fullframe_manifest.json")}
        n = 81
        xy = np.full((n, 134, 2), [120, 250], dtype=np.float32)
        index = {name: i for i, name in enumerate(adapter.rules.joint_layout()["names"])}
        for name, position in {"right_shoulder": [100, 200], "left_shoulder": [200, 200],
                               "right_hip": [100, 300], "left_hip": [200, 300],
                               "right_knee": [100, 400], "left_knee": [200, 400],
                               "right_ankle": [100, 500], "left_ankle": [200, 500],
                               "left_wrist": [210, 220]}.items():
            xy[:, index[name]] = position
        xy[:, index["right_wrist"], 0] = np.linspace(100, 200, n)
        xy[:, index["right_wrist"], 1] = 220
        times = np.arange(n, dtype=np.float64) / 16
        data = {"xy": xy, "score": np.full((n, 134), 2.0, dtype=np.float32),
                "valid": np.ones((n, 134), dtype=bool), "tracked": np.ones(n, dtype=bool),
                "bbox": np.tile(np.array([50, 100, 250, 550], dtype=np.float32), (n, 1)),
                "frame_indices": np.arange(n, dtype=np.int64), "times": times,
                "source_timestamps": times.copy(), "schema": np.array("ms_fullvideo_pose2d_diagnostic_v1"),
                "id": np.array(sid), "split": np.array(split), "video_sha256": np.array("a" * 64),
                "joint_layout": np.array(json.dumps(adapter.rules.joint_layout())),
                **{key: np.array(value) for key, value in adapter.FLAGS.items()}}
        manifest = {"schema": adapter.MANIFEST_SCHEMA, "id": sid, "split": split,
                    "npz_path": row["npz_path"], "video_path": str(self.root / "videos" / (sid + ".mp4")),
                    "video_sha256": "a" * 64, "frame_count": n, "width": 704, "height": 1280,
                    "pts_origin": 0.0, "duration_seconds": 5.0625, "frame_indices": data["frame_indices"].tolist(),
                    "times": times.tolist(), "source_timestamps": times.tolist(),
                    "joint_layout": adapter.rules.joint_layout(), "score_threshold": 0.3,
                    "metadata_provenance": {"scope": "synthetic unit-test fixture, not real media evidence"},
                    **adapter.FLAGS}
        return row, manifest, data

    def write_fixture(self, row=None, manifest=None, data=None):
        row = self.row if row is None else row
        manifest = self.manifest if manifest is None else manifest
        data = self.data if data is None else data
        np.savez_compressed(row["npz_path"], **data)
        Path(row["fullframe_manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")

    def draft(self):
        mask = adapter.validate_observations(self.row, self.manifest, self.data)
        return adapter.generate_draft(self.row, self.manifest, self.data, mask)

    def test_81_frames_and_unobserved_video_tail(self):
        draft = self.draft()
        self.assertEqual(draft["observation"]["frame_count"], 81)
        self.assertTrue(draft["observation"]["all_native_frames_present"])
        self.assertEqual(draft["observation"]["unknown_intervals"][-1],
                         {"start": 5.0, "end": 5.0625, "reason": "outside_cached_observations"})
        self.assertTrue(all(w["basis"] in ("observed_time", "unobserved") for w in draft["windows"]))
        self.assertNotIn("roi_manifest_path", draft["source"])
        self.assertNotIn("bar_plan_path", draft["source"])
        self.assertEqual(draft["source"]["fullframe_binding"]["schema"], adapter.BINDING_SCHEMA)

    def test_native_pts_reverse_and_manifest_disagreement_rejected(self):
        for key in ("times", "source_timestamps"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key][2] = data[key][1]
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)
        self.manifest["times"][5] += 0.01
        with self.assertRaises(ValueError):
            self.draft()

    def test_wrong_2d_shape_and_actual_frame_count_rejected(self):
        for key in ("xy", "score", "valid", "tracked", "bbox"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = data[key][:-1]
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)
        self.data["xy"] = np.pad(self.data["xy"], ((0, 0), (0, 0), (0, 1)))
        with self.assertRaises(ValueError):
            self.draft()

    def test_source_id_and_split_binding_rejected(self):
        for key in ("id", "split", "video_sha256"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key] = np.array("wrong")
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)
        self.manifest["split"] = "val"
        with self.assertRaises(ValueError):
            self.draft()

    def test_wrong_layout_rejected(self):
        for target in ("manifest", "npz"):
            with self.subTest(target=target):
                manifest, data = copy.deepcopy(self.manifest), copy.deepcopy(self.data)
                layout = adapter.rules.joint_layout()
                layout["coordinate_system"] = "xyz_3d"
                if target == "manifest":
                    manifest["joint_layout"] = layout
                else:
                    data["joint_layout"] = np.array(json.dumps(layout))
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, manifest, data)

    def test_effective_valid_nan_rejected(self):
        for key in ("xy", "score"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key][4, 4] = np.nan
                with self.assertRaises(ValueError):
                    adapter.validate_observations(self.row, self.manifest, data)

    def test_invalid_nan_preserved_and_not_event_support(self):
        self.data["valid"][40, 4] = False
        self.data["xy"][40, 4] = np.nan
        self.data["score"][40, 4] = np.nan
        self.write_fixture()
        manifest, data, mask = adapter.load_observations(self.row)
        self.assertTrue(np.isnan(data["xy"][40, 4]).all())
        self.assertFalse(mask[40, 4])
        draft = adapter.generate_draft(self.row, manifest, data, mask)
        for event in draft["events"]:
            if event["support"]["signal"].startswith("right_wrist"):
                self.assertNotIn(40, event["support"]["frame_indices"])
        self.assertEqual(draft["status"], "partial")

    def test_scores_above_one_are_raw_and_threshold_is_enforced(self):
        self.assertGreater(self.data["score"].max(), 1)
        self.assertTrue(self.draft()["events"])
        self.data["score"][1, 4] = 0.2
        with self.assertRaises(ValueError):
            self.draft()
        self.data["valid"][1, 4] = False
        self.assertFalse(adapter.validate_observations(self.row, self.manifest, self.data)[1, 4])

    def test_omitted_native_frames_rejected(self):
        self.data["frame_indices"][40:] += 1
        with self.assertRaises(ValueError):
            self.draft()
        self.manifest["frame_indices"] = self.data["frame_indices"].tolist()
        with self.assertRaises(ValueError):
            self.draft()

    def test_native_time_hole_stays_unknown_without_fill(self):
        self.data["times"][40:] += 0.6
        self.data["source_timestamps"] = self.data["times"].copy()
        self.manifest["times"] = self.data["times"].tolist()
        self.manifest["source_timestamps"] = self.data["times"].tolist()
        self.manifest["duration_seconds"] += 0.6
        draft = self.draft()
        holes = [v for v in draft["observation"]["unknown_intervals"] if v["reason"] == "time_gap"]
        self.assertEqual(len(holes), 1)
        self.assertEqual(holes[0]["start"], self.data["times"][39])
        self.assertEqual(holes[0]["end"], self.data["times"][40])
        self.assertTrue(all(not (39 in e["support"]["frame_indices"] and 40 in e["support"]["frame_indices"]) for e in draft["events"]))

    def test_new_npz_alone_determines_events_never_reads_old_roi(self):
        self.write_fixture()
        with patch.object(adapter.rules, "load_observations", side_effect=AssertionError("Old ROI loader called")), \
             patch.object(adapter.rules, "load_bar_windows", side_effect=AssertionError("Old bars called")):
            moving = adapter.process_row(self.row, self.root / "moving_out")
            self.data["xy"][:] = self.data["xy"][0]
            self.write_fixture()
            static = adapter.process_row(self.row, self.root / "static_out")
        self.assertGreater(moving["event_count"], 0)
        self.assertEqual(static["event_count"], 0)
        self.assertNotEqual(moving["ms_text"], static["ms_text"])

    def test_write_once(self):
        self.write_fixture()
        first = adapter.process_row(self.row, self.root / "out")
        path = Path(first["path"])
        before = path.read_bytes()
        self.data["xy"][:] = self.data["xy"][0]
        self.write_fixture()
        self.assertEqual(adapter.process_row(self.row, self.root / "out")["status"], "existing_not_overwritten")
        self.assertEqual(path.read_bytes(), before)

    def test_partial_is_always_unapproved_in_both_input_schemas(self):
        for schema in adapter.INPUT_SCHEMAS:
            with self.subTest(schema=schema):
                self.data["schema"] = np.array(schema)
                draft = self.draft()
                self.assertEqual(draft["status"], "partial")
                for key, value in adapter.FLAGS.items():
                    self.assertIs(draft[key], value)
                self.assertFalse(draft["beat_alignment_claimed"])
        self.data["quality_approved"] = np.array(True)
        with self.assertRaises(ValueError):
            self.draft()

    def test_npz_hash_and_nonpickle(self):
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

    def test_offsets_and_cross_split_paths(self):
        self.data["source_timestamps"] += 2.0
        self.manifest["source_timestamps"] = self.data["source_timestamps"].tolist()
        with self.assertRaises(ValueError):
            self.draft()
        self.manifest["pts_origin"] = 2.0
        self.assertEqual(self.draft()["observation"]["native_pts_origin"], 2.0)
        self.row["split"] = "val"
        with self.assertRaises(ValueError):
            adapter.check_row(self.row)

    def test_cli_default_three_and_failure_is_separate(self):
        rows = []
        for sid in ("first", "second", "third"):
            row, manifest, data = self.fixture(sid)
            self.write_fixture(row, manifest, data)
            rows.append(row)
        list_path = self.root / "input.jsonl"
        list_path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "not JSON\n", encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = adapter.main(["--manifest", str(list_path), "--out", str(self.root / "cli_out")])
        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue().splitlines()[-1])["sources_attempted"], 3)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = adapter.main(["--manifest", str(list_path), "--out", str(self.root / "cli_out"), "--limit", "4"])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(stderr.getvalue())["status"], "rejected")
        self.assertNotIn('"status": "rejected"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
