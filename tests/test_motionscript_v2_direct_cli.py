"""Direct-point contracts. All model/canonical values are explicit test fixtures.

The video tests execute FFmpeg. The injected predictor is not a trained NLF;
these tests do not establish actual palm/foot accuracy on RGB dance videos.
"""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms"
sys.path.insert(0, str(MS))
import ms_landmarks as cli


def spec_fixture():
    # Deliberately artificial coordinates; never supplied as real canonical assets.
    return cli.build_smplx_spec(np.arange(165).reshape(55, 3) * .001,
        np.arange(10475 * 3).reshape(10475, 3) * .00001,
        source={"fixture": "synthetic indexed arrays; not an anatomical model"})


class Tensor(np.ndarray):
    def to(self, *args, **kwargs):
        return self

    def permute(self, *axes):
        return self.transpose(axes)


class Method:
    schema = "TEST DOUBLE, not a serialized NLF method"

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class DirectContractTests(unittest.TestCase):
    def test_default_32_queries_include_palm_and_foot_without_finger_chains(self):
        spec = spec_fixture()
        names = [p["name"] for p in spec["points"]]
        self.assertEqual(len(names), 32)
        self.assertEqual(names[:22], list(cli.BODY_NAMES))
        self.assertEqual(names[22:26], ["left_index_mcp", "left_little_mcp", "right_index_mcp", "right_little_mcp"])
        self.assertEqual(names[-6:], ["left_big_toe", "left_little_toe", "left_heel", "right_big_toe", "right_little_toe", "right_heel"])
        self.assertFalse(any("tip" in x or "pip" in x or "dip" in x for x in names))
        origins = {p["name"]: p for p in spec["source"]["point_origins"]}
        self.assertEqual(origins["left_little_mcp"]["index"], 31)
        self.assertEqual(origins["right_little_mcp"]["index"], 46)
        self.assertEqual(origins["left_heel"]["index"], 8846)
        np.testing.assert_allclose(spec["points"][25]["canonical"], np.arange(165).reshape(55, 3)[46] * .001)

    def test_wrong_topology_and_nonfinite_canonical_assets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "canonical joints"):
            cli.build_smplx_spec(np.zeros((52, 3)), np.zeros((10475, 3)))
        vertices = np.zeros((10475, 3))
        vertices[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "canonical joints"):
            cli.build_smplx_spec(np.zeros((55, 3)), vertices)

    def test_direct_adapter_passes_rgb_boxes_queries_and_never_calls_fit(self):
        queried, received = [], []

        def queries(points):
            queried.append(points.copy())
            return {"TEST_WEIGHTS": points}

        def predict(images, boxes, weights, *, num_aug):
            received.append((images.copy(), boxes, num_aug))
            self.assertIs(weights["TEST_WEIGHTS"], queried_weights[0])
            k = len(weights["TEST_WEIGHTS"])
            return {"poses3d": [np.full((len(box), k, 3), images[i].mean()) for i, box in enumerate(boxes)],
                    "poses2d": [np.full((len(box), k, 2), 7.) for box in boxes],
                    "uncertainties": [np.full((len(box), k), .07) for box in boxes]}

        model = types.SimpleNamespace(eval=lambda: model,
            get_weights_for_canonical_points=Method(queries), estimate_poses_batched=Method(predict))
        fake = types.SimpleNamespace(jit=types.SimpleNamespace(load=lambda *a, **kw: model),
            float32=np.float32, inference_mode=contextlib.nullcontext,
            tensor=lambda data, dtype, device: np.asarray(data, dtype=dtype).view(Tensor),
            as_tensor=lambda data, dtype, device: np.asarray(data, dtype=dtype).view(Tensor),
            from_numpy=lambda data: data.view(Tensor))
        with mock.patch.dict(sys.modules, {"torch": fake, "torchvision": types.ModuleType("torchvision")}):
            predictor = cli.NLFDirectPredictor("TEST_MODEL", spec_fixture(), "cpu", 3)
            queried_weights = [predictor.weights["TEST_WEIGHTS"]]
            frames = np.stack([np.full((4, 5, 3), 80, np.uint8), np.full((4, 5, 3), 160, np.uint8)])
            boxes = np.array([[0, 0, 5, 4], [0, 0, 5, 4]], np.float32)
            result = predictor(frames, boxes, np.array([True, True]))
            points, uncertainty, valid, pixels = cli.unpack_predictions(result, [1, 1], 32)
        self.assertEqual(queried[0].shape, (32, 3))
        self.assertEqual(received[0][0].shape, (2, 3, 4, 5))
        self.assertEqual(received[0][0].dtype, np.uint8)
        self.assertEqual(received[0][2], 3)
        np.testing.assert_allclose(points[:, 0, 0], [.08, .16])
        np.testing.assert_allclose(uncertainty, .07)
        np.testing.assert_allclose(pixels, 7.)
        self.assertTrue(valid.all())

    def test_missing_checkpoint_direct_api_does_not_fall_back_to_smpl(self):
        model = types.SimpleNamespace(eval=lambda: model, detect_smpl_batched=lambda *a: self.fail("SMPL fallback called"))
        fake = types.SimpleNamespace(jit=types.SimpleNamespace(load=lambda *a, **kw: model))
        with mock.patch.dict(sys.modules, {"torch": fake, "torchvision": types.ModuleType("torchvision")}):
            with self.assertRaisesRegex(ValueError, "no SMPL fallback"):
                cli.NLFDirectPredictor("TEST_MODEL", spec_fixture(), "cpu", 1)

    def test_empty_people_mm_conversion_and_native_uncertainty(self):
        points, uncertainty, valid, pixels = cli.unpack_predictions({
            "poses3d": [np.array([[[10, 20, 3000], [np.nan, 0, 0]]]), np.empty((0, 2, 3))],
            "poses2d": [np.array([[[123, 456], [0, 0]]]), np.empty((0, 2, 2))],
            "uncertainties": [np.array([[.05, .2]]), np.empty((0, 2))]}, [1, 0], 2)
        np.testing.assert_allclose(points[0, 0], [.01, .02, 3])
        np.testing.assert_allclose(pixels[0, 0], [123, 456])
        self.assertEqual(float(uncertainty[0, 0]), np.float32(.05))
        self.assertEqual(valid.tolist(), [[True, False], [False, False]])
        self.assertTrue(np.isnan(points[1]).all())
        with self.assertRaisesRegex(ValueError, "point order/count"):
            cli.unpack_predictions({"poses3d": [np.zeros((1, 24, 3))], "poses2d": [np.zeros((1, 24, 2))],
                                    "uncertainties": [np.zeros((1, 24))]}, [1], 32)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg needed for real video I/O tests")
class DirectVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "inputs"
        source.mkdir()
        self.video, self.boxes = source / "test.mp4", source / "boxes.npz"
        self.spec, self.model = source / "spec.json", source / "TEST_MODEL"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=red:size=64x96:rate=10:duration=1", "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(self.video)], check=True)
        np.savez(self.boxes, box=np.tile([0, 0, 64, 96, .9], (10, 1)), valid=np.ones(10, bool), fps=10.)
        self.spec.write_bytes(cli.json_bytes(spec_fixture()))
        self.model.write_bytes(b"TEST DOUBLE; NOT REAL NLF")
        self.args = cli.parser().parse_args(["extract", "--video", str(self.video), "--boxes-nlf", str(self.boxes),
            "--spec", str(self.spec), "--model", str(self.model), "--out", str(self.root / "extracted"), "--bs", "4"])

    def factory(self, *args):
        class Predictor:
            method_schemas = {"test_double": "not a trained NLF"}

            def __call__(self, frames, boxes, valid):
                # An RGB-conditioned test function, not an image-understanding model.
                assert frames.shape[1:] == (96, 64, 3)
                assert np.mean(frames[..., 0]) > 200 and np.mean(frames[..., 2]) < 5
                return {"poses3d": [np.full((int(ok), 32, 3), float(frame.mean())) for frame, ok in zip(frames, valid)],
                        "poses2d": [np.full((int(ok), 32, 2), 30.) for ok in valid],
                        "uncertainties": [np.full((int(ok), 32), .08) for ok in valid]}
        return Predictor()

    def test_real_decode_query_cache_and_description_end_to_end(self):
        info = cli.probe_video(self.video)
        self.assertEqual(len(info["time"]), 10)
        np.testing.assert_allclose(info["time"], np.arange(10) * .1)
        self.assertAlmostEqual(info["media_end"], 1.)
        manifest = cli.extract(self.args, predictor_factory=self.factory)
        self.assertFalse(manifest["smpl_fitting_used"])
        self.assertFalse(manifest["template_fk_used"])
        arrays, _, _ = cli.load_bundle(Path(self.args.out))
        self.assertEqual(arrays["points_m"].shape, (10, 32, 3))
        np.testing.assert_allclose(arrays["points2d_px"], 30.)
        self.assertTrue(arrays["valid"].all())
        args = cli.parser().parse_args(["describe", "--bundle", self.args.out, "--out", str(self.root / "described")])
        report = cli.describe(args)
        self.assertEqual(report["point_count"], 32)
        self.assertFalse(report["semantic_approval"])
        self.assertEqual(report["events"], [])  # Degenerate fixture coordinates cannot produce palm/foot facts.
        self.assertTrue((Path(args.out) / "coords.npz").is_file())
        with np.load(Path(args.out) / "coords.npz", allow_pickle=False) as saved:
            self.assertEqual(saved["landmark_signal_values"].shape[1], 8)
            self.assertTrue(np.isnan(saved["landmark_signal_values"]).all())
            self.assertFalse(saved["landmark_signal_valid"].any())
            self.assertEqual(saved["raw_points2d_px"].shape, (10, 32, 2))
        self.assertEqual(len(report["continuous_signals"]["signals"]), 8)

    def test_unobserved_cached_box_does_not_receive_invented_prediction(self):
        valid = np.ones(10, bool)
        valid[3:7] = False
        np.savez(self.boxes, box=np.tile([0, 0, 64, 96, .9], (10, 1)), valid=valid, fps=10.)
        cli.extract(self.args, predictor_factory=self.factory)
        arrays, _, _ = cli.load_bundle(Path(self.args.out))
        self.assertTrue(np.isnan(arrays["points_m"][3:7]).all())
        self.assertFalse(arrays["valid"][3:7].any())

    def test_wrong_box_timing_fails_before_model_loading(self):
        np.savez(self.boxes, box=np.tile([0, 0, 64, 96, .9], (10, 1)), valid=np.ones(10, bool), fps=16.)
        with self.assertRaisesRegex(ValueError, "frame rate differs"):
            cli.extract(self.args, predictor_factory=lambda *a: self.fail("model loaded before alignment checks"))
        self.assertFalse(Path(self.args.out).exists())

    def test_cache_edit_invalidates_bundle(self):
        cli.extract(self.args, predictor_factory=self.factory)
        path = Path(self.args.out) / "raw.npz"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "cache changed"):
            cli.load_bundle(path.parent)

    def test_description_binds_manifest_snapshot_used_to_read_points(self):
        from motionscript_v2 import landmarks

        cli.extract(self.args, predictor_factory=self.factory)
        path = Path(self.args.out) / "manifest.json"
        original_sha = cli.sha256(path)
        original_preprocess = landmarks.preprocess_landmarks

        def change_after_load(*args, **kwargs):
            path.write_bytes(path.read_bytes() + b" ")
            return original_preprocess(*args, **kwargs)

        args = cli.parser().parse_args(["describe", "--bundle", self.args.out, "--out", str(self.root / "description")])
        with mock.patch.object(landmarks, "preprocess_landmarks", side_effect=change_after_load):
            report = cli.describe(args)
        self.assertEqual(report["source_manifest_sha256"], original_sha)
        self.assertNotEqual(report["source_manifest_sha256"], cli.sha256(path))

    def test_fixed_chunks_keep_media_end_distinct_from_last_point_time(self):
        cli.extract(self.args, predictor_factory=self.factory)
        args = cli.parser().parse_args(["describe", "--bundle", self.args.out, "--out", str(self.root / "chunks"),
            "--chunk-seconds", ".4", "--min-tail-seconds", ".3"])
        report = cli.describe(args)
        self.assertEqual(report["window_spec"]["windows"], [{"start": 0., "end": .4}, {"start": .4, "end": 1.}])
        self.assertEqual(len(report["chunks"]), 2)
        self.assertTrue(all(row["caption_is_training_target"] is False for row in report["chunks"]))
        self.assertTrue(all(row["semantic_approval"] is False for row in report["chunks"]))
        self.assertEqual(report["window_time_meaning"], "annotation window, not exact action boundary")
        with np.load(Path(args.out) / "coords.npz", allow_pickle=False) as saved:
            self.assertAlmostEqual(saved["time"][-1], .9)
        self.assertEqual(report["media_bounds"]["end"], 1.)

    def test_external_windows_require_the_same_video_identity(self):
        cli.extract(self.args, predictor_factory=self.factory)
        windows = self.root / "windows.json"
        windows.write_bytes(cli.json_bytes({"schema": "motionscript_windows_v1", "strategy": "explicit",
            "video_sha256": "different video", "windows": [{"start": 0., "end": 1.}]}))
        args = cli.parser().parse_args(["describe", "--bundle", self.args.out, "--out", str(self.root / "chunks"),
            "--windows", str(windows)])
        with self.assertRaisesRegex(ValueError, "same source video SHA256"):
            cli.describe(args)
        self.assertFalse(Path(args.out).exists())

    def test_local_palm_changes_keep_chunk_times_and_full_track_indices(self):
        lookup = {row["name"]: i for i, row in enumerate(spec_fixture()["points"])}

        class MovingPalmFixture:
            method_schemas = {"test_double": "synthetic palm changes, not trained NLF"}
            seen = 0

            def __call__(self, frames, boxes, valid):
                result = {"poses3d": [], "poses2d": [], "uncertainties": []}
                for _ in frames:
                    p = np.zeros((32, 3))
                    p[lookup["left_shoulder"]] = [-.3, 0, 3]
                    p[lookup["left_elbow"]] = [0, 0, 3]
                    wrist = np.array([0, .3, 3])
                    p[lookup["left_wrist"]] = wrist
                    angle = np.radians(self.seen * 10.)
                    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                                         [-np.sin(angle), 0, np.cos(angle)]])
                    p[lookup["left_index_mcp"]] = wrist + rotation @ [.04, .07, 0]
                    p[lookup["left_little_mcp"]] = wrist + rotation @ [-.04, .07, 0]
                    result["poses3d"].append(p[None] * 1000.)
                    result["poses2d"].append(np.full((1, 32, 2), 20.))
                    result["uncertainties"].append(np.full((1, 32), .01))
                    self.seen += 1
                return result

        cli.extract(self.args, predictor_factory=lambda *a: MovingPalmFixture())
        args = cli.parser().parse_args(["describe", "--bundle", self.args.out, "--out", str(self.root / "changes"),
            "--chunk-seconds", ".4", "--min-tail-seconds", ".3", "--smooth-seconds", "0"])
        report = cli.describe(args)
        ids = []
        with np.load(Path(args.out) / "coords.npz", allow_pickle=False) as saved:
            time = saved["time"]
            for chunk in report["chunks"]:
                self.assertTrue(chunk["events"])
                for event in chunk["events"]:
                    self.assertGreaterEqual(event["start"], chunk["window"]["start"])
                    self.assertLess(event["end"], chunk["window"]["end"])
                    self.assertAlmostEqual(time[event["evidence"]["global_target_sample_start"]], event["start"])
                    self.assertAlmostEqual(time[event["evidence"]["global_target_sample_end"]], event["end"])
                    ids.append(event["id"])
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
