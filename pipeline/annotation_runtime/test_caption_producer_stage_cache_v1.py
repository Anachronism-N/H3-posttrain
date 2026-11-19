"""Synthetic CPU-only gate tests; no SSH, leases, media assets or model loading."""
from collections import Counter
from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np

sys.dont_write_bytecode = True
R = Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime")
launch = json.loads((R / "curated_collaborative3_n2p0_v1_launch.json").read_bytes())

def argv_in(value):
    if isinstance(value, list) and "--repo" in value:
        return value
    if isinstance(value, dict):
        for item in value.values():
            result = argv_in(item)
            if result is not None:
                return result
    return None

argv = argv_in(launch)
D = Path(argv[argv.index("--repo") + 1]) / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(D))
sys.path.insert(0, str(R))
import ms_hand_roi as roi
import curated_producer_io_v4 as io4
from caption_producer_stage_cache_v1 import SourceCache, NativeStageCache, canonical


def load_pipeline():
    return io4.base.load_pipeline()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="caption-stage-cache-cpu-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / "shared"
        self.local = self.root / "local"
        self.out.mkdir()
        self.local.mkdir()
        self.video, self.audio, self.weights = [self.root / name for name in ("source.mp4", "source.wav", "cpu-model.ref")]
        self.video.write_bytes(b"synthetic-video-input-not-decoded")
        self.audio.write_bytes(b"synthetic-audio-input-not-inferred")
        self.weights.write_bytes(b"cpu-only-model-reference-no-real-model")
        self.row = {"id": "cpu_source", "split": "train", "batch_id": "train-b000000",
                    "source_ordinal": 0, "video_path": str(self.video), "audio_path": str(self.audio),
                    "nlf_path": None, "source_group": "cpu_group"}
        files = {"cpu_reference": roi.file_identity(self.weights)}
        self.model = {"files": files, "sha256": roi.digest(files)}
        self.frames = [{"frame_index": i, "time": float(i) / 8, "source_timestamp": 2. + float(i) / 8,
                        "source_pts": 16 + i} for i in range(2)]
        self.prepared = {"id": self.row["id"], "input_row": {"id": self.row["id"], "manifest_record": self.row},
                         "source": roi.file_identity(self.video), "video_sha256": roi.sha256(self.video),
                         "model_ref": self.model, "params": dict(roi.DEFAULT_PARAMS),
                         "probe": {"width": 32, "height": 32}, "selected_frames": self.frames,
                         "source_group": "cpu_group"}
        self.prepared["binding_sha256"] = roi.digest(self.prepared)
        self.cache = SourceCache(self.row, self.out / "roi" / self.row["id"])
        self.addCleanup(lambda: self.cache.close() if self.cache.active else None)
        self.calls = Counter()

    def hash_read(self, path):
        self.calls[str(Path(path).resolve())] += 1
        return roi.sha256(path)

    def complete(self, parent=None, receipt=True, wrong_pts=False):
        parent = parent or self.local
        parent.mkdir(parents=True, exist_ok=True)
        directory = roi.ensure_prepared(parent, self.prepared)
        (directory / "crops").mkdir()
        raw = directory / "raw_wholebody134.npz"
        points = np.arange(2 * 134 * 2, dtype=np.float32).reshape(2, 134, 2) / 10
        scores = np.full((2, 134), .2, dtype=np.float32)
        with raw.open("xb") as stream:
            np.savez_compressed(stream, binding_sha256=np.array(self.prepared["binding_sha256"]),
                                video_sha256=np.array(self.prepared["video_sha256"]),
                                frame_indices=np.array([0, 1]), source_timestamps=np.array([2., 2.125 + (1 if wrong_pts else 0)]),
                                xy=points, score=scores)
        crop = directory / "crops/f000000_left_hand.jpg"
        # Opaque synthetic asset; native validator's SHA gate, not JPEG quality evidence.
        crop.write_bytes(b"synthetic-crop-bytes-for-native-SHA-gate")
        manifest = {"id": self.row["id"], "source": self.prepared["source"],
                    "binding_sha256": self.prepared["binding_sha256"], "video_sha256": self.prepared["video_sha256"],
                    "model_ref": self.model, "probe": self.prepared["probe"], "selected_frames": self.frames,
                    "source_manifest": {"path": str(directory / "source_manifest.json"), "sha256": roi.sha256(directory / "source_manifest.json")},
                    "raw": {"path": str(raw), "sha256": roi.sha256(raw)},
                    "crops": [{"path": str(crop), "sha256": roi.sha256(crop)}],
                    "semantic_approval": False}
        roi.write_json_new(directory / "roi_manifest.json", manifest)
        payload = {"binding_sha256": self.prepared["binding_sha256"], "roi_manifest_sha256": roi.sha256(directory / "roi_manifest.json")}
        if receipt:
            roi.write_json_new(directory / "extract_receipt.json", payload)
        return directory, payload

    def validate(self, directory, prepared):
        self.calls["validate"] += 1
        return roi.validate_completed(directory, prepared)

    def test_same_source_sha_cross_function_and_changed_file_refusal(self):
        expected = self.cache.sha(self.video, self.hash_read)
        self.assertEqual(expected, self.cache.sha(str(self.video), self.hash_read))
        self.assertEqual(1, self.calls[str(self.video)])
        self.video.write_bytes(b"changed-input")
        with self.assertRaises(ValueError):
            self.cache.sha(self.video, self.hash_read)

    def test_source_release_discards_media_and_result_cache(self):
        self.cache.sha(self.video, self.hash_read)
        self.cache.close()
        self.assertFalse(self.cache.hashes)
        with self.assertRaises(RuntimeError):
            self.cache.sha(self.video, self.hash_read)
        other = SourceCache({**self.row, "id": "cpu_next", "source_ordinal": 1}, self.out / "roi/cpu_next")
        try:
            other.sha(self.video, self.hash_read)
            self.assertEqual(2, self.calls[str(self.video)])
        finally:
            other.close()

    def test_probe_same_protocol_arguments_only(self):
        def probe():
            self.calls["probe"] += 1
            return {"times": [0., .125], "raw_pts": [2., 2.125], "shape": [32, 32, 3]}
        first = self.cache.probe("roi", self.video, ["ffprobe"], probe)
        second = self.cache.probe("roi", self.video, ["ffprobe"], probe)
        second["times"][0] = 999
        self.assertEqual([0., .125], first["times"])
        self.cache.probe("bar", self.video, ["ffprobe"], probe)
        self.cache.probe("roi", self.video, ["different-ffprobe"], probe)
        self.assertEqual(3, self.calls["probe"])

    def test_probe_changed_source_fails(self):
        self.cache.probe("roi", self.video, [], lambda: {"times": [0., 1.]})
        self.video.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.cache.probe("roi", self.video, [], lambda: self.fail("must not reprobe changed input"))

    def test_completed_exact_binding_copies_results_and_only_validates_once(self):
        directory, _ = self.complete()
        self.cache.register_root(directory, self.prepared)
        first = self.cache.validate(directory, self.prepared, self.validate)
        second = self.cache.validate(directory, copy.deepcopy(self.prepared), self.validate)
        second["crops"].clear()
        self.assertEqual(1, len(first["crops"]))
        self.assertEqual(1, len(self.cache.validate(directory, self.prepared, self.validate)["crops"]))
        self.assertEqual(1, self.calls["validate"])

    def test_changed_prepared_id_split_and_params_rejected(self):
        directory, _ = self.complete()
        self.cache.register_root(directory, self.prepared)
        self.cache.validate(directory, self.prepared, self.validate)
        for change in ("id", "split", "params"):
            value = copy.deepcopy(self.prepared)
            if change == "id":
                value["id"] = "wrong"
            elif change == "split":
                value["input_row"]["manifest_record"]["split"] = "test"
            else:
                value["params"]["sample_fps"] = 4.
            with self.assertRaises(ValueError, msg=change):
                self.cache.validate(directory, value, self.validate)
        self.assertEqual(1, self.calls["validate"])

    def test_unregistered_root_and_symlink_refused(self):
        directory, _ = self.complete()
        with self.assertRaises(ValueError):
            self.cache.validate(directory, self.prepared, self.validate)
        alias = self.root / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.cache.register_root(alias, self.prepared)

    def test_partial_receipt_never_cached_as_complete(self):
        directory, receipt = self.complete(receipt=False)
        self.cache.register_root(directory, self.prepared)
        with self.assertRaises(ValueError):
            self.cache.validate(directory, self.prepared, self.validate)
        self.assertFalse(self.cache.validated)
        roi.write_json_new(directory / "extract_receipt.json", receipt)
        self.cache.validate(directory, self.prepared, self.validate)
        self.assertEqual(2, self.calls["validate"])

    def test_actual_npz_pts_gate_still_runs(self):
        directory, _ = self.complete(wrong_pts=True)
        self.cache.register_root(directory, self.prepared)
        with self.assertRaisesRegex(ValueError, "Raw source/frame/PTS"):
            self.cache.validate(directory, self.prepared, self.validate)
        self.assertFalse(self.cache.validated)

    def test_source_marker_modification_and_receipt_removal_invalidate(self):
        directory, _ = self.complete()
        self.cache.register_root(directory, self.prepared)
        self.cache.validate(directory, self.prepared, self.validate)
        (directory / "source_manifest.json").write_bytes(b"modified-source-marker")
        with self.assertRaises(ValueError):
            self.cache.validate(directory, self.prepared, self.validate)
        (directory / "extract_receipt.json").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.cache.validate(directory, self.prepared, self.validate)

    def test_completed_root_added_io_receipts_do_not_invalidate(self):
        directory, _ = self.complete()
        self.cache.register_root(directory, self.prepared)
        expected = self.cache.validate(directory, self.prepared, self.validate)
        roi.write_json_new(directory / "io_publish_receipt.json", {"cpu_only": True})
        roi.write_json_new(directory / "io_publish_ownership.json", {"cpu_only": True})
        self.assertEqual(expected, self.cache.validate(directory, self.prepared, self.validate))
        self.assertEqual(1, self.calls["validate"])

    def test_completed_cache_hit_zero_jpeg_npz_stat_or_read(self):
        import caption_producer_stage_cache_v1 as cache_module
        directory, _ = self.complete()
        self.cache.register_root(directory, self.prepared)
        expected = self.cache.validate(directory, self.prepared, self.validate)
        observed = []
        original = cache_module.fingerprint
        def tracked(path):
            observed.append(str(path))
            self.assertIn(Path(path).name, ("source_manifest.json", "extract_receipt.json"))
            return original(path)
        with mock.patch.object(cache_module, "fingerprint", tracked), mock.patch.object(Path, "read_bytes", side_effect=AssertionError("cache hit must not read assets")), mock.patch.object(Path, "read_text", side_effect=AssertionError("cache hit must not read assets")):
            self.assertEqual(expected, self.cache.validate(directory, self.prepared, self.validate))
        self.assertEqual(2, len(observed))
        self.assertEqual(1, self.calls["validate"])

    def test_native_writeonce_and_partial_extract_refusal(self):
        directory, _ = self.complete()
        original = (directory / "source_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            roi.write_new(directory / "source_manifest.json", b"replacement")
        self.assertEqual(original, (directory / "source_manifest.json").read_bytes())
        partial_parent = self.root / "partial"
        partial_parent.mkdir()
        partial = roi.ensure_prepared(partial_parent, self.prepared)
        (partial / "orphan.bin").write_bytes(b"retain-me")
        with self.assertRaisesRegex(ValueError, "Partial/orphan"):
            roi.extract_source(partial, self.prepared, lambda _: self.fail("model must not run"))
        self.assertEqual(b"retain-me", (partial / "orphan.bin").read_bytes())

    def test_cross_thread_owner_refused(self):
        failures = []
        def run():
            try:
                self.cache.sha(self.video, self.hash_read)
            except RuntimeError as error:
                failures.append(str(error))
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        self.assertEqual(1, len(failures))

    def stage(self):
        pipeline = load_pipeline()
        pipeline.iter_rows = lambda *a, **k: iter([(self.row, {})])
        pipeline.run_producer = lambda *a, **k: None
        pipeline.audio_probe = lambda *a, **k: {"synthetic_audio_probe": True}
        pipeline.load_beat_runtime = lambda *a, **k: (lambda _: ([0.], [0.]), {"cpu_fake": True})
        bar = types.SimpleNamespace(probe_actual=lambda *a, **k: {"times": [0., .125]})
        @contextmanager
        def lock(directory):
            self.calls["lock_acquired"] += 1
            yield
            self.calls["lock_released"] += 1
        visual = types.SimpleNamespace(sha256=roi.sha256, bundle_lock=lock, source_identity=lambda row: {"files": {}})
        self.reports = []
        return NativeStageCache(pipeline, bar, roi, visual, self.out, self.reports.append)

    def startup(self, stage):
        roi.verify_model(self.model)
        stage.pipeline.save(self.root / "protocol.json", {"roi_model_ref": self.model, "roi_params": self.prepared["params"]})

    def test_startup_model_once_stage_zero_repeat_restore_restart(self):
        stage = self.stage()
        original = roi.verify_model
        def check(value):
            self.calls["model"] += 1
            return original(value)
        with mock.patch.object(roi, "verify_model", check):
            with stage.installed():
                self.startup(stage)
                rows = stage.pipeline.iter_rows()
                next(rows)
                with stage.visual.bundle_lock(stage.pipeline.source_dir(self.out, self.row)):
                    for _ in range(4):
                        roi.verify_model(copy.deepcopy(self.model))
                    changed = copy.deepcopy(self.model)
                    changed["sha256"] = "wrong"
                    with self.assertRaises(ValueError):
                        roi.verify_model(changed)
                rows.close()
                self.assertEqual(1, self.calls["model"])
                # Resume/outside source scope must keep native model gate.
                roi.verify_model(self.model)
                self.assertEqual(2, self.calls["model"])
            self.assertIs(roi.verify_model, check)
            with self.stage().installed():
                roi.verify_model(self.model)
            self.assertEqual(3, self.calls["model"])
        self.assertIs(roi.verify_model, original)
        self.assertEqual(1, self.calls["lock_released"])
        self.assertIsNone(stage.source)

    def test_io4_first_shared_gate_mapping_writeonce_and_final_cache(self):
        directory, _ = self.complete()
        stage = self.stage()
        original = roi.validate_completed
        def checked(root, prepared):
            self.calls["validate:" + str(root)] += 1
            return original(root, prepared)
        with mock.patch.object(roi, "validate_completed", checked):
            with stage.installed(publishers=(io4,)):
                self.startup(stage)
                rows = stage.pipeline.iter_rows()
                next(rows)
                with stage.visual.bundle_lock(stage.pipeline.source_dir(self.out, self.row)):
                    stage.source.register_root(directory, self.prepared)
                    local = roi.validate_completed(directory, self.prepared)
                    final = self.out / "roi" / self.row["id"]
                    shared = io4.publish_roi(roi, directory, final, self.prepared, .001)
                    self.assertEqual(io4.base.remap_paths(local, directory, final), shared)
                    self.assertEqual(1, self.calls["validate:" + str(directory)])
                    self.assertEqual(1, self.calls["validate:" + str(final)])
                    sha_reads = stage.source.counts["sha_reads"]
                    self.assertEqual(shared, roi.validate_completed(final, self.prepared))
                    self.assertEqual(sha_reads, stage.source.counts["sha_reads"])
                    self.assertEqual(1, self.calls["validate:" + str(final)])
                    self.assertTrue((final / "extract_receipt.json").is_file())
                    self.assertFalse(shared["semantic_approval"])
                    with self.assertRaises(FileExistsError):
                        io4.publish_roi(roi, directory, final, self.prepared, .001)
                rows.close()
        self.assertEqual(1, len(self.reports))
        self.assertGreater(self.reports[0]["seconds"]["shared_publish"], 0)
        self.assertGreater(self.reports[0]["seconds"]["shared_validation"], 0)

    def test_native_cpu_decode_points_scores_and_unknown_roi_unchanged(self):
        # Fake RGB reader/model input only; native extraction, NPZ and ROI policies run.
        rgb = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        class Frame:
            def asnumpy(self):
                return rgb.copy()
        class Reader:
            def __getitem__(self, index):
                return Frame()
        def model(bgr):
            self.calls["cpu_fake_model"] += 1
            np.testing.assert_array_equal(bgr, rgb[:, :, ::-1])
            return (np.full((1, 134, 2), 10., np.float32), np.full((1, 134), .2, np.float32),
                    np.array([[1, 1, 30, 30]], np.float32))
        cv2_stub = types.SimpleNamespace(COLOR_RGB2BGR=1, cvtColor=lambda value, _: value[:, :, ::-1])
        with mock.patch.dict(sys.modules, {"cv2": cv2_stub}), mock.patch.object(roi, "open_native_reader", lambda *a: (Reader(), "synthetic_cpu_reader")):
            baseline_parent = self.root / "baseline"
            baseline_parent.mkdir()
            baseline_root = roi.ensure_prepared(baseline_parent, self.prepared)
            baseline = roi.extract_source(baseline_root, self.prepared, model)
            stage = self.stage()
            with stage.installed():
                self.startup(stage)
                rows = stage.pipeline.iter_rows()
                next(rows)
                with stage.visual.bundle_lock(stage.pipeline.source_dir(self.out, self.row)):
                    cached_root = roi.ensure_prepared(self.local, self.prepared)
                    cached = roi.extract_source(cached_root, self.prepared, model)
                    self.assertEqual(io4.base.remap_paths(baseline, baseline_root, cached_root), cached)
                    self.assertEqual(cached, roi.validate_completed(cached_root, self.prepared))
                    self.assertTrue(all(part["status"] == "unknown" for part in cached["hand_observations"]))
                    with np.load(baseline["raw"]["path"], allow_pickle=False) as first, np.load(cached["raw"]["path"], allow_pickle=False) as second:
                        self.assertEqual(first.files, second.files)
                        for key in first.files:
                            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
                    self.assertEqual(1, stage.source.counts["validation_calls"])
                    self.assertGreater(stage.source.counts["sha_hits"], 0)
                rows.close()
        self.assertEqual(4, self.calls["cpu_fake_model"])
        self.assertGreater(self.reports[0]["seconds"]["roi_extraction"], 0)

    def test_native_decode_wrong_shape_still_fails(self):
        cv2_stub = types.SimpleNamespace(COLOR_RGB2BGR=1, cvtColor=lambda value, _: value[:, :, ::-1])
        bad_frame = types.SimpleNamespace(asnumpy=lambda: np.zeros((16, 16, 3), np.uint8))
        reader = {0: bad_frame, 1: bad_frame}
        with mock.patch.dict(sys.modules, {"cv2": cv2_stub}), mock.patch.object(roi, "open_native_reader", lambda *a: (reader, "synthetic_cpu_reader")):
            stage = self.stage()
            with stage.installed():
                self.startup(stage)
                rows = stage.pipeline.iter_rows()
                next(rows)
                with stage.visual.bundle_lock(stage.pipeline.source_dir(self.out, self.row)):
                    directory = roi.ensure_prepared(self.local, self.prepared)
                    with self.assertRaisesRegex(ValueError, "Decoded image is not native RGB"):
                        roi.extract_source(directory, self.prepared, lambda _: self.fail("wrong shape must fail before model"))
                    self.assertFalse((directory / "extract_receipt.json").exists())
                    self.assertFalse(stage.source.validated)
                rows.close()

    def test_native_probe_gate_preserved_and_true_repeat_cached(self):
        stage = self.stage()
        payload = {"streams": [{"width": 32, "height": 32, "avg_frame_rate": "8/1", "time_base": "1/8", "duration": "1"}],
                   "frames": [{"best_effort_timestamp": 16 + i, "best_effort_timestamp_time": str(2 + i / 8), "width": 32, "height": 32} for i in range(2)]}
        def run(*a, **k):
            self.calls["ffprobe"] += 1
            return types.SimpleNamespace(stdout=json.dumps(payload).encode())
        with mock.patch.object(roi.subprocess, "run", run):
            expected = roi.probe_video(str(self.video), "/tmp/cpu-only-ffprobe")
            with stage.installed():
                self.startup(stage)
                rows = stage.pipeline.iter_rows()
                next(rows)
                with stage.visual.bundle_lock(stage.pipeline.source_dir(self.out, self.row)):
                    first = roi.probe_video(str(self.video), "/tmp/cpu-only-ffprobe")
                    self.assertEqual(expected, first)
                    self.assertEqual(first, roi.probe_video(str(self.video), "/tmp/cpu-only-ffprobe"))
                    self.assertEqual(2, self.calls["ffprobe"])
                    payload["frames"][1]["width"] = 16
                    with self.assertRaisesRegex(ValueError, "dimensions"):
                        roi.probe_video(str(self.video), "/tmp/cpu-only-other-ffprobe")
                    self.assertEqual(1, stage.source.counts["probe_calls"])
                    self.assertEqual(1, stage.source.counts["probe_hits"])
                rows.close()

    def test_restart_changed_model_fails_real_startup_gate(self):
        with self.stage().installed():
            roi.verify_model(self.model)
        self.weights.write_bytes(b"changed-startup-model-reference")
        with self.stage().installed():
            with self.assertRaisesRegex(ValueError, "Identity file changed"):
                roi.verify_model(self.model)

    def test_failure_restores_functions_clears_scope_and_logs_exception(self):
        stage = self.stage()
        original_sha = roi.sha256
        with self.assertRaisesRegex(RuntimeError, "cpu-test-failure"):
            with stage.installed():
                self.startup(stage)
                rows = stage.pipeline.iter_rows()
                next(rows)
                with stage.visual.bundle_lock(stage.pipeline.source_dir(self.out, self.row)):
                    roi.sha256(self.video)
                    raise RuntimeError("cpu-test-failure")
        self.assertIs(roi.sha256, original_sha)
        self.assertIsNone(stage.source)
        self.assertFalse(stage.patches)
        self.assertEqual("raised_RuntimeError", self.reports[0]["status"])
        self.assertEqual(1, len(self.reports))
        self.assertFalse(self.reports[0]["formal_training"])

    def test_io2_run_hook_order_and_full_final_timer(self):
        stage = self.stage()
        pipeline = stage.pipeline
        appended = []
        pipeline.append = lambda path, value: appended.append(copy.deepcopy(value))
        def captured_native_run(args):
            self.startup(stage)
            for row, _ in pipeline.iter_rows():
                with stage.visual.bundle_lock(pipeline.source_dir(self.out, row)):
                    pipeline.append(self.out / "ready.jsonl", {"status": "ready", "source_ordinal": row["source_ordinal"]})
            return "cpu-native-run-result"
        with mock.patch.object(io4.base, "publish_roi", io4.publish_roi):
            before = io4.base.publish_roi
            with stage.installed(publishers=(io4.base,)):
                result = stage.run_wrapper(captured_native_run)(types.SimpleNamespace())
            self.assertIs(io4.base.publish_roi, before)
        self.assertEqual("cpu-native-run-result", result)
        self.assertEqual(1, len(appended))
        self.assertGreater(self.reports[0]["seconds"]["full_final_check"], 0)
        self.assertIsNone(stage.source)

    def test_wrong_source_lock_refused_before_native_lock(self):
        stage = self.stage()
        with stage.installed():
            rows = stage.pipeline.iter_rows()
            next(rows)
            with self.assertRaises(ValueError):
                with stage.visual.bundle_lock(self.out / "sources/test" / self.row["id"]):
                    self.fail("wrong split lock admitted")
            rows.close()
        self.assertEqual(0, self.calls["lock_acquired"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
