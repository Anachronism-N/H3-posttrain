"""CPU geometry/provenance tests; fake Wholebody is not a GPU or semantic audit."""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_omnidance/exp/ms"))
import ms_hand_roi as roi

FFMPEG = os.environ.get("MS_HAND_ROI_TEST_FFMPEG") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("MS_HAND_ROI_TEST_FFPROBE") or shutil.which("ffprobe")


def observations():
    xy = np.full((134, 2), np.nan, np.float32)
    score = np.full(134, .1, np.float32)
    grid = np.array([[x, y] for y in (0, 7, 14, 21, 28) for x in (0, 7, 14, 21, 28)], np.float32)[:21]
    for part, offset in (("left_hand", [40, 45]), ("right_hand", [165, 55])):
        start, end, wrist = roi.HANDS[part]
        xy[start:end] = grid + offset
        score[start:end] = .9
        xy[wrist], score[wrist] = xy[start], .9
    return xy, score


def image():
    return np.random.default_rng(24).integers(0, 256, (128, 256, 3), dtype=np.uint8)


def fake_probe(n=5, width=256, height=128, offset=.125):
    times = [offset + i / 16. for i in range(n)]
    ticks = list(range(n))
    return {"width": width, "height": height, "decoded_video_frames": n,
            "actual_video_pts": times, "actual_video_pts_ticks": ticks, "time_base": "1/16",
            "actual_video_pts_sha256": roi.digest({"times": times, "ticks": ticks}),
            "frame_times": [i / 16. for i in range(n)], "time_origin": offset,
            "duration": str(n / 16.), "avg_frame_rate": "16/1", "pts_source": "CPU fixture"}


class FakeArray:
    def __init__(self, value):
        self.value = value

    def asnumpy(self):
        return self.value.copy()


class FakeReader:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def __getitem__(self, index):
        self.calls.append(index)
        return FakeArray(self.frames[index])


class HandROITest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ms-hand-roi-cpu-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rgb = image()
        self.xy, self.scores = observations()

    def source(self, explicit=(0, 2, 4)):
        video = self.root / "source.mp4"
        if not video.exists():
            video.write_bytes(b"synthetic source bytes; not a real video")
        manifest = self.root / "manifest.jsonl"
        row = {"id": "fixture", "video_path": str(video), "source_group": "cpu_fixture", "split": "test"}
        if not manifest.exists():
            manifest.write_text(json.dumps(row) + "\n")
        rows, identity = roi.read_manifest(manifest)
        with patch.object(roi, "probe_video", return_value=fake_probe()):
            return roi.prepare_source(rows[0], identity, {"sha256": "a" * 64, "files": {}},
                                      dict(roi.DEFAULT_PARAMS), "/fixture/ffprobe", "/fixture/ffmpeg", list(explicit))

    def extracted(self):
        source = self.source()
        out = self.root / "out"
        out.mkdir()
        directory = roi.ensure_prepared(out, source)
        reader = FakeReader([self.rgb for _ in range(5)])
        calls = []

        def fake_model(bgr):
            calls.append(bgr.copy())
            return self.xy[None].copy(), self.scores[None].copy(), np.array([[20, 20, 230, 115]], np.float32)

        with patch.object(roi, "open_native_reader", return_value=(reader, "CPU fake reader")):
            manifest = roi.extract_source(directory, source, fake_model)
        self.assertEqual(reader.calls, [0, 2, 4])
        self.assertEqual(len(calls), 3)
        np.testing.assert_array_equal(calls[0], self.rgb[..., ::-1])
        return directory, source, manifest

    def test_budgeted_sampling_keeps_complete_video_tail(self):
        probe = {"frame_times": [i / 16 for i in range(81)]}
        selected = roi.select_frame_indices(probe, roi.DEFAULT_PARAMS)
        self.assertEqual(len(selected), 32)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 80)
        self.assertTrue(all(b > a for a, b in zip(selected, selected[1:])))
        self.assertTrue(all(type(i) is int for i in selected))

    def test_verified_134_layout_and_wrist_indices(self):
        layout = roi.joint_layout()
        self.assertEqual(layout["names"][7], "left_wrist")
        self.assertEqual(layout["names"][4], "right_wrist")
        self.assertEqual(layout["names"][92], "left_hand_wrist")
        self.assertEqual(layout["names"][93], "left_thumb_1")
        self.assertEqual(layout["names"][113], "right_hand_wrist")
        self.assertEqual(layout["parts_half_open"]["left_hand"], [92, 113])
        with self.assertRaises(ValueError):
            roi.hand_roi(self.rgb, self.xy[:133], self.scores[:133], "left_hand")

    def test_valid_left_and_right_raw_threshold_not_probability(self):
        for part in roi.HANDS:
            with self.subTest(part=part):
                proposal = roi.hand_roi(self.rgb, self.xy, self.scores, part)
                self.assertEqual(proposal["status"], "supported")
                start, end, body = roi.HANDS[part]
                self.assertEqual(proposal["supportpointindices"], list(range(start, end)))
                self.assertEqual(proposal["body_wrist_index"], body)
                self.assertEqual(proposal["wrist_alignment_px"], 0.)
                self.assertIs(proposal["quality_is_probability"], False)
                self.assertIs(proposal["scores_summary"]["is_probability"], False)
                self.assertFalse(proposal["anatomical_side_guaranteed"])
                self.assertGreaterEqual(proposal["crop_pixel_size"][0], 32)
                self.assertNotIn("hand_shape", proposal)

    def test_outside_six_points_cannot_crop_other_hand_still_valid(self):
        self.scores[92:113] = .1
        self.scores[92:99] = .99
        self.xy[93:99] = [300, 150]
        left = roi.hand_roi(self.rgb, self.xy, self.scores, "left_hand")
        right = roi.hand_roi(self.rgb, self.xy, self.scores, "right_hand")
        self.assertEqual(left["status"], "unknown")
        self.assertEqual(left["valid_hand_points"], 1)
        self.assertIsNone(left["crop_xyxy"])
        self.assertEqual(right["status"], "supported")
        with self.assertRaises(ValueError):
            roi.save_crop(self.rgb, left, self.root / "must-not-exist.jpg")
        self.assertFalse((self.root / "must-not-exist.jpg").exists())

    def test_strict_score_finite_and_frame_bounds_mask(self):
        xy = np.array([[0, 0], [255, 127], [256, 10], [-1, 10], [10, 128], [np.nan, 10], [10, np.inf], [10, 10], [10, 10]])
        scores = np.array([.9, .9, .99, .99, .99, .99, .99, .3, np.nan])
        np.testing.assert_array_equal(roi.point_valid(xy, scores, 256, 128), [True, True] + [False] * 7)
        self.assertFalse(roi.point_valid(xy, scores, 256, 128, tracked=False).any())

    def test_wrist_missing_or_wrong_index_unknown(self):
        for wrist in (7, 92):
            with self.subTest(wrist=wrist):
                score = self.scores.copy()
                score[wrist] = .3
                proposal = roi.hand_roi(self.rgb, self.xy, score, "left_hand")
                self.assertEqual(proposal["reason"], "missing_body_or_hand_wrist")
        xy = self.xy.copy()
        xy[92] = np.nan
        self.assertEqual(roi.hand_roi(self.rgb, xy, self.scores, "left_hand")["status"], "unknown")
        self.assertEqual(roi.hand_roi(self.rgb, xy, self.scores, "right_hand")["status"], "supported")

    def test_nonfinite_points_retained_by_subject_but_invalid_locally(self):
        points = self.xy.copy()
        points[93] = np.nan
        scores = self.scores.copy()
        scores[94] = np.inf
        selected, raw_scores, _, reason = roi.select_raw_subject(
            (points[None], scores[None], np.array([[0, 0, 256, 128]], np.float32)), None, .3)
        self.assertEqual(reason, "tracked")
        self.assertTrue(np.isnan(selected[93]).all())
        valid = roi.point_valid(selected, raw_scores, 256, 128)
        self.assertFalse(valid[93])
        self.assertFalse(valid[94])
        self.assertTrue(valid[113])

    def test_low_scores_are_not_invented_and_raw_values_are_unchanged(self):
        scores = self.scores.copy()
        scores[92:113] = .29
        before = self.xy.copy()
        proposal = roi.hand_roi(self.rgb, self.xy, scores, "left_hand")
        self.assertEqual(proposal["status"], "unknown")
        np.testing.assert_array_equal(self.xy, before)
        np.testing.assert_array_equal(scores[92:113], np.full(21, .29, np.float32))

    def test_native_extent_and_blur_unknown(self):
        xy = self.xy.copy()
        xy[92:113] = (xy[92:113] - xy[92]) * .1 + xy[92]
        tiny = roi.hand_roi(self.rgb, xy, self.scores, "left_hand")
        self.assertEqual(tiny["reason"], "native_hand_extent_too_small")
        self.assertIsNone(tiny["crop_xyxy"])
        flat = roi.hand_roi(np.full_like(self.rgb, 127), self.xy, self.scores, "left_hand")
        self.assertEqual(flat["reason"], "native_hand_blur_heuristic_unknown")
        self.assertEqual(flat["blur_laplacian_variance"], 0.)
        self.assertTrue(flat["blur_is_heuristic_not_truth"])

    def test_bbox_clipping_and_affine_inverse(self):
        xy = self.xy.copy()
        xy[92:113] -= [40, 45]
        xy[7] = xy[92]
        proposal = roi.hand_roi(self.rgb, xy, self.scores, "left_hand")
        self.assertEqual(proposal["status"], "supported")
        self.assertEqual(proposal["crop_xyxy"][:2], [0, 0])
        for point in xy[92:113]:
            local = np.array(proposal["source_to_ROI_affine"]) @ np.r_[point, 1]
            recovered = np.array(proposal["ROI_to_source_affine"]) @ np.r_[local, 1]
            np.testing.assert_allclose(recovered, point)
            self.assertTrue((local >= 0).all())
            self.assertTrue((local < proposal["crop_pixel_size"]).all())
        self.assertFalse(proposal["rotated"])
        self.assertFalse(proposal["mirrored"])
        self.assertFalse(proposal["resized"])

    def test_wrist_misalignment_and_box_jump_unknown(self):
        xy = self.xy.copy()
        xy[7] = [125, 45]
        proposal = roi.hand_roi(self.rgb, xy, self.scores, "left_hand")
        self.assertEqual(proposal["reason"], "body_hand_wrist_misaligned")
        self.assertGreater(proposal["wrist_alignment_px"], 70)
        jump = roi.hand_roi(self.rgb, self.xy, self.scores, "right_hand", previous_box=[0, 0, 32, 32])
        self.assertEqual(jump["reason"], "hand_bbox_jump_not_supported")
        self.assertIsNone(jump["crop_xyxy"])
        self.assertGreater(jump["hand_bbox_jump_normalized"], 2.)

    def test_subject_iou_rejection_no_reinitialization(self):
        result = (self.xy[None], self.scores[None], np.array([[100, 50, 230, 120]], np.float32))
        points, scores, box, reason = roi.select_raw_subject(result, np.array([0, 0, 40, 40]), .3)
        self.assertEqual(reason, "low_subject_iou")
        self.assertIsNone(points)
        self.assertIsNone(box)
        proposal = roi.hand_roi(self.rgb, self.xy, self.scores, "left_hand", tracked=False)
        self.assertEqual(proposal["reason"], "subject_not_continuously_tracked")

    def test_real_synthetic_rgb_crop_hash_and_source_not_modified(self):
        before = self.rgb.copy()
        proposal = roi.hand_roi(self.rgb, self.xy, self.scores, "left_hand")
        path = self.root / "crop.jpg"
        asset = roi.save_crop(self.rgb, proposal, path)
        self.assertEqual(asset["sha256"], roi.sha256(path))
        x0, y0, x1, y1 = proposal["crop_xyxy"]
        self.assertEqual(asset["crop_rgb_sha256"], hashlib.sha256(self.rgb[y0:y1, x0:x1].tobytes()).hexdigest())
        np.testing.assert_array_equal(self.rgb, before)
        import cv2
        decoded = cv2.imread(str(path))
        self.assertEqual(decoded.shape, (y1 - y0, x1 - x0, 3))
        with self.assertRaises(FileExistsError):
            roi.save_crop(self.rgb, proposal, path)
        self.assertEqual(roi.sha256(path), asset["sha256"])

    def test_native_sampling_real_indices_no_interpolation(self):
        probe = fake_probe(n=81)
        chosen = roi.select_frame_indices(probe, roi.DEFAULT_PARAMS)
        self.assertEqual(len(chosen), 32)
        self.assertEqual((chosen[0], chosen[-1]), (0, 80))
        self.assertTrue(all(i % 2 == 0 for i in chosen))
        self.assertEqual(roi.select_frame_indices(probe, roi.DEFAULT_PARAMS, [16, 18, 22, 26]), [16, 18, 22, 26])
        for invalid in ([1, 1], [2, 1], [-1], [81], [], [1.5], [True]):
            with self.subTest(indices=invalid), self.assertRaises(ValueError):
                roi.select_frame_indices(probe, roi.DEFAULT_PARAMS, invalid)
        probe["frame_times"][1] = probe["frame_times"][0]
        with self.assertRaises(ValueError):
            roi.select_frame_indices(probe, roi.DEFAULT_PARAMS)

    def test_prepare_resume_source_sha_pts_model_parameter_mismatch(self):
        source = self.source()
        out = self.root / "out"
        out.mkdir()
        directory = roi.ensure_prepared(out, source)
        original = (directory / "source_manifest.json").read_bytes()
        self.assertEqual(roi.ensure_prepared(out, source), directory)
        for key in ("video_sha256", "binding_sha256", "source_group"):
            changed = copy.deepcopy(source)
            changed[key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                roi.ensure_prepared(out, changed)
        changed = copy.deepcopy(source)
        changed["selected_frames"][0]["source_timestamp"] += .01
        with self.assertRaises(ValueError):
            roi.ensure_prepared(out, changed)
        changed = copy.deepcopy(source)
        changed["model_ref"]["sha256"] = "different model"
        with self.assertRaises(ValueError):
            roi.ensure_prepared(out, changed)
        changed = copy.deepcopy(source)
        changed["params"]["padding"] += .01
        with self.assertRaises(ValueError):
            roi.ensure_prepared(out, changed)
        self.assertEqual((directory / "source_manifest.json").read_bytes(), original)
        video = Path(source["source"]["path"])
        video.write_bytes(b"changed video bytes")
        with self.assertRaises(ValueError):
            roi.verify_identity(source["source"])
        with self.assertRaises(ValueError):
            roi.ensure_prepared(out, self.source())

    def test_extract_raw21_receipt_and_exact_resume_without_decoding(self):
        directory, source, manifest = self.extracted()
        self.assertEqual(len(manifest["crops"]), 6)
        self.assertEqual(len(manifest["selected_frames"]), 3)
        self.assertFalse(manifest["semantic_approval"])
        with np.load(manifest["raw"]["path"], allow_pickle=False) as raw:
            self.assertEqual(raw["left_hand_xy"].shape, (3, 21, 2))
            self.assertEqual(raw["right_hand_score"].shape, (3, 21))
            np.testing.assert_array_equal(raw["frame_indices"], [0, 2, 4])
            np.testing.assert_array_equal(raw["left_hand_xy"][0], self.xy[92:113])
            self.assertEqual(str(raw["video_sha256"]), source["video_sha256"])
            self.assertFalse(raw["valid"][0, 0])
            self.assertTrue(np.isnan(raw["xy"][0, 0]).all())
        with patch.object(roi, "open_native_reader", side_effect=AssertionError("resume must not decode")):
            self.assertEqual(roi.extract_source(directory, source, None), manifest)
        self.assertEqual(roi.sha256(source["source"]["path"]), source["video_sha256"])

    def test_evidence_two_hands_one_frame_not_two_observations(self):
        _, _, manifest = self.extracted()
        one = roi.select_roi_evidence(manifest, 0, .1, [0])
        for part in roi.HANDS:
            self.assertEqual(one[part]["status"], "unknown")
            self.assertEqual(one[part]["crops"], [])
        self.assertEqual(one["distinct_source_frame_indices"], [])
        selected = roi.select_roi_evidence(manifest, 0, .3, [0, 2, 4], max_per_hand=2)
        self.assertEqual(selected["left_hand"]["frame_indices"], [0, 4])
        self.assertEqual(selected["right_hand"]["frame_indices"], [0, 4])
        self.assertEqual(selected["distinct_source_frame_indices"], [0, 4])
        self.assertTrue(selected["same_frame_two_hands_is_one_temporal_observation"])

    def test_evidence_rejects_duplicate_pts_frames_crop_and_hash(self):
        _, _, manifest = self.extracted()
        bad = copy.deepcopy(manifest)
        bad["selected_frames"][1]["frame_index"] = 0
        with self.assertRaises(ValueError):
            roi.select_roi_evidence(bad, 0, 1)
        bad = copy.deepcopy(manifest)
        bad["selected_frames"][1]["time"] = 0
        with self.assertRaises(ValueError):
            roi.select_roi_evidence(bad, 0, 1)
        bad = copy.deepcopy(manifest)
        bad["crops"].append(copy.deepcopy(bad["crops"][0]))
        with self.assertRaises(ValueError):
            roi.select_roi_evidence(bad, 0, 1)
        bad = copy.deepcopy(manifest)
        bad["crops"][0]["time"] += .01
        with self.assertRaises(ValueError):
            roi.select_roi_evidence(bad, 0, 1)
        bad = copy.deepcopy(manifest)
        bad["crops"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            roi.select_roi_evidence(bad, 0, 1)
        with self.assertRaises(ValueError):
            roi.select_roi_evidence(manifest, 0, 1, [0, 0])

    def test_completed_cannot_reuse_different_source_or_modified_crop(self):
        directory, source, manifest = self.extracted()
        changed = copy.deepcopy(source)
        changed["video_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            roi.validate_completed(directory, changed)
        Path(manifest["crops"][0]["path"]).write_bytes(b"bad JPEG")
        with self.assertRaises(ValueError):
            roi.validate_completed(directory, source)

    def test_prepare_cli_no_cuda_import_no_download_no_env_changes(self):
        source = self.source()
        original_import = builtins.__import__
        env_before = dict(os.environ)

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "onnxruntime", "decord"):
                raise AssertionError(f"prepare cannot import {name}")
            return original_import(name, *args, **kwargs)

        with patch.object(roi, "model_identity", return_value=source["model_ref"]) as identify, \
                patch.object(roi, "probe_video", return_value=fake_probe()), \
                patch.object(roi, "load_cuda_model", side_effect=AssertionError("prepare cannot load CUDA")), \
                patch("builtins.__import__", side_effect=guarded_import):
            result = roi.main(["prepare", "--manifest", source["source_manifest"]["path"],
                               "--out", str(self.root / "cli"), "--ffprobe", "/fixture/ffprobe",
                               "--ffmpeg", "/fixture/ffmpeg"])
            self.assertEqual(result, 0)
            identify.assert_called_once()
        self.assertEqual(dict(os.environ), env_before)
        receipt = json.loads((self.root / "cli/fixture/prepare_receipt.json").read_text())
        self.assertFalse(receipt["cuda_loaded"])
        self.assertFalse((self.root / "cli/fixture/raw_wholebody134.npz").exists())

    def test_model_hash_once_then_stat_guard(self):
        scail = self.root / "scail"
        weights = scail / "pretrained_weights/DWPose"
        code = scail / "DWPoseProcess/dwpose"
        weights.mkdir(parents=True)
        code.mkdir(parents=True)
        for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
            (weights / name).write_bytes(b"CPU test placeholder model; never loaded")
        for name in ("wholebody.py", "onnxdet.py", "onnxpose.py", "__init__.py"):
            (code / name).write_text("# CPU fixture, never imported\n")
        identity = roi.model_identity(scail)
        with patch.object(roi, "sha256", side_effect=AssertionError("stat checks must not rehash weights")):
            roi.verify_model(identity)
            roi.verify_model(identity)
        (weights / "yolox_l.onnx").write_bytes(b"modified")
        with self.assertRaises(ValueError):
            roi.verify_model(identity)

    def test_isolated_import_skips_upstream_environment_mutation(self):
        code = self.root / "isolated"
        code.mkdir()
        (code / "__init__.py").write_text('import os\nos.environ["ROI_BAD_GLOBAL"] = "mutated"\n')
        (code / "wholebody.py").write_text("from . import onnxdet, onnxpose\nclass Wholebody: pass\n")
        (code / "onnxdet.py").write_text("VALUE = 1\n")
        (code / "onnxpose.py").write_text("VALUE = 2\n")
        identity = {"files": {key: roi.file_identity(code / (key + ".py"))
                              for key in ("wholebody", "onnxdet", "onnxpose")}}
        before = dict(os.environ)
        with patch.dict(sys.modules):
            module = roi.load_verified_wholebody_module(identity)
            self.assertEqual(Path(module.__file__), code / "wholebody.py")
            self.assertEqual(module.onnxdet.VALUE, 1)
            self.assertEqual(module.onnxpose.VALUE, 2)
        self.assertEqual(dict(os.environ), before)

    def test_plan_manifest_no_nlf_required_duplicate_ids_rejected(self):
        source = self.source()
        plan = self.root / "plan.jsonl"
        record = source["input_row"]["manifest_record"]
        plan.write_text(json.dumps({"manifest_record": record, "plan_status": "ready"}) + "\n")
        rows, _ = roi.read_manifest(plan, ids=["fixture"], limit=0)
        self.assertEqual(rows[0]["manifest_record"], record)
        with self.assertRaises(ValueError):
            roi.read_manifest(plan, ids=["missing"])
        plan.write_text((json.dumps(record) + "\n") * 2)
        with self.assertRaises(ValueError):
            roi.read_manifest(plan)

    @unittest.skipUnless(FFMPEG and FFPROBE, "existing ffmpeg/ffprobe required for real CPU media smoke")
    def test_real_cpu_video_pts_native_decode_and_source_hash_unchanged(self):
        video = self.root / "actual.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=256x128:rate=16",
                        "-t", "0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
        before = roi.sha256(video)
        probe = roi.probe_video(video, FFPROBE)
        self.assertEqual((probe["width"], probe["height"]), (256, 128))
        self.assertEqual(probe["decoded_video_frames"], 8)
        source = {"source": roi.file_identity(video), "probe": probe}
        reader, basis = roi.open_native_reader(source, 1)
        rgb = reader[2].asnumpy()
        self.assertEqual(rgb.shape, (128, 256, 3))
        self.assertIn("ffprobe", basis)
        proposal = roi.hand_roi(rgb, self.xy, self.scores, "left_hand", params={**roi.DEFAULT_PARAMS, "min_blur_variance": 0.})
        self.assertEqual(proposal["status"], "supported")
        roi.save_crop(rgb, proposal, self.root / "actual_crop.jpg")
        self.assertEqual(roi.sha256(video), before)


if __name__ == "__main__":
    unittest.main()
