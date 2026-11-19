"""Real CPU filesystem/NPZ/JPEG tests; no mocked-success ROI or inference.

Optional --replay-inputs accepts exactly three original completed ROI directories
and --result-json writes isolated A(v3)/B(v4) publication measurements.
"""
import argparse
import builtins
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import io as python_io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import cv2
import numpy as np

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
sys.path.insert(0, str(RT / "source_curated_producer_v1/repo/scripts_omnidance/exp/ms"))
import curated_producer_io_v3 as previous
import curated_producer_io_v4 as publication

_spec = importlib.util.spec_from_file_location("_native_roi_publication_test_v4", RT / "source_hand_roi_v3/ms_hand_roi.py")
roi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roi)


def write_bundle(root):
    """Explicit publication fixture, not detector output or an annotation candidate."""
    source_file = root / "source_fixture.bin"
    source_file.write_bytes(b"publication-test-source-only")
    model_file = root / "not_a_model.bin"
    model_file.write_bytes(b"publication-test-identity-only")
    files = {"test_fixture": roi.file_identity(model_file)}
    frames = [{"frame_index": 0, "source_timestamp": 0., "time": 0., "source_pts": 0},
              {"frame_index": 8, "source_timestamp": .5, "time": .5, "source_pts": 8}]
    prepared = {"id": "cpu_publication_fixture", "source": roi.file_identity(source_file),
                "video_sha256": roi.sha256(source_file),
                "model_ref": {"files": files, "sha256": roi.digest(files), "actual_inference_model": False},
                "params": dict(roi.DEFAULT_PARAMS), "probe": {"test_fixture_only": True},
                "selected_frames": frames, "input_row": {"manifest_record": {"source_ordinal": 500}},
                "publication_fixture_only": True}
    prepared["binding_sha256"] = roi.digest(prepared)
    local = root / "local"
    local.mkdir()
    (local / "crops").mkdir()
    (local / "source_manifest.json").write_bytes(roi.json_bytes(prepared))
    (local / "prepare_receipt.json").write_bytes(roi.json_bytes({"binding_sha256": prepared["binding_sha256"]}))
    raw = local / "raw_wholebody134.npz"
    np.savez_compressed(raw, binding_sha256=prepared["binding_sha256"], video_sha256=prepared["video_sha256"],
                        frame_indices=np.array([0, 8]), source_timestamps=np.array([0., .5]))
    crops = []
    for frame in frames:
        for part in ("left_hand", "right_hand"):
            path = local / "crops" / ("f%06d_%s.jpg" % (frame["frame_index"], part))
            image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
            success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise RuntimeError("Real JPEG fixture encoding failed")
            path.write_bytes(encoded.tobytes())
            crops.append({**frame, "part": part, "path": str(path), "sha256": roi.sha256(path),
                          "ROI_to_source_affine": [[1., 0., 16.], [0., 1., 32.]],
                          "crop_xyxy": [16, 32, 48, 64], "test_fixture_only": True})
    manifest = {"source": prepared["source"], "video_sha256": prepared["video_sha256"],
                "binding_sha256": prepared["binding_sha256"], "model_ref": prepared["model_ref"],
                "probe": prepared["probe"], "selected_frames": copy.deepcopy(frames),
                "source_manifest": {"path": str(local / "source_manifest.json"),
                                    "sha256": roi.sha256(local / "source_manifest.json")},
                "raw": {"path": str(raw), "sha256": roi.sha256(raw)}, "crops": crops}
    (local / "roi_manifest.json").write_bytes(roi.json_bytes(manifest))
    (local / "extract_receipt.json").write_bytes(roi.json_bytes({"binding_sha256": prepared["binding_sha256"],
        "roi_manifest_sha256": roi.sha256(local / "roi_manifest.json")}))
    roi.validate_completed(local, prepared)
    return local, prepared


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.local_tmp = tempfile.TemporaryDirectory(prefix="cpu-roi-v4-local-")
        self.root = Path(self.local_tmp.name)
        self.shared_tmp = tempfile.TemporaryDirectory(prefix="cpu-roi-v4-shared-", dir=RT / "smalltests")
        self.shared = Path(self.shared_tmp.name)
        self.local, self.prepared = write_bundle(self.root)
        self.final = self.shared / self.prepared["id"]
        self.assertNotEqual(self.local.stat().st_dev, self.shared.stat().st_dev)

    def tearDown(self):
        self.shared_tmp.cleanup()
        self.local_tmp.cleanup()

    def publish(self):
        return publication.publish_roi(roi, self.local, self.final, self.prepared, 1.25)

    def test_native_validation_and_jpeg_npz_bytes_parameters_pts_affines(self):
        original = roi.validate_completed(self.local, self.prepared)
        result = self.publish()
        expected = publication.base.remap_paths(original, self.local, self.final)
        self.assertEqual(result, expected)
        self.assertEqual(roi.validate_completed(self.final, self.prepared), expected)
        for relative in ["source_manifest.json", "prepare_receipt.json", "raw_wholebody134.npz"]:
            self.assertEqual((self.local / relative).read_bytes(), (self.final / relative).read_bytes())
        for crop in result["crops"]:
            self.assertEqual(roi.sha256(crop["path"]), crop["sha256"])
            self.assertEqual(cv2.imread(crop["path"]).shape[:2], (32, 32))
        ownership = json.loads((self.final / "io_publish_ownership.json").read_bytes())
        self.assertEqual(len(ownership["owner_token"]), 32)
        self.assertEqual(ownership["source_binding_sha256"], self.prepared["binding_sha256"])
        for asset in ownership["expected_assets"]:
            self.assertEqual(roi.sha256(self.final / asset["relative_path"]), asset["sha256"])
        evidence = json.loads((self.final / "io_publish_receipt.json").read_bytes())
        self.assertEqual(evidence["owner_token"], ownership["owner_token"])
        self.assertFalse(evidence["whole_directory_atomic"])
        self.assertFalse(evidence["complete_roi_validated"])
        self.assertTrue(evidence["per_file_shared_fsync"])
        self.assertEqual(evidence["publisher_pre_receipt_counters"]["shared_rehash_read_bytes"], 0)
        self.assertEqual(evidence["local_extract_seconds"], 1.25)
        self.assertFalse((self.shared / "producer.terminal.json").exists())

    def test_receipt_is_last_atomic_file_only_after_assets_and_directory_fsync(self):
        real_link, real_fsync = os.link, os.fsync
        synced, linked = set(), []
        def sync(fd):
            synced.add(os.readlink("/proc/self/fd/" + str(fd)))
            return real_fsync(fd)
        def link(source, target, *args, **kwargs):
            self.assertEqual(Path(target), self.final / "extract_receipt.json")
            self.assertFalse(Path(target).exists())
            self.assertTrue((self.final / "io_publish_receipt.json").is_file())
            for relative in ["raw_wholebody134.npz", "source_manifest.json", "prepare_receipt.json", "roi_manifest.json"]:
                self.assertIn(str(self.final / relative), synced)
            for crop in roi.validate_completed(self.local, self.prepared)["crops"]:
                self.assertIn(str(self.final / "crops" / Path(crop["path"]).name), synced)
            self.assertIn(str(self.final / "crops"), synced)
            self.assertIn(str(self.final), synced)
            linked.append(str(target))
            return real_link(source, target, *args, **kwargs)
        with patch.object(os, "fsync", side_effect=sync), patch.object(os, "link", side_effect=link):
            self.publish()
        self.assertEqual(linked, [str(self.final / "extract_receipt.json")])

    def assert_failed_final_refused(self):
        self.assertTrue(self.final.is_dir())
        self.assertFalse((self.final / "extract_receipt.json").exists())
        retained = {p.relative_to(self.final).as_posix(): p.read_bytes() for p in self.final.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(retained, {p.relative_to(self.final).as_posix(): p.read_bytes() for p in self.final.rglob("*") if p.is_file()})

    def test_interrupted_partial_stream_retained_no_receipt_no_retry(self):
        real_open = Path.open
        target = self.final / "raw_wholebody134.npz"
        class InterruptedWriter:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def write(self, data):
                self.stream.write(data[:17])
                self.stream.flush()
                raise OSError("injected interrupted asset write")
        def opened(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            return InterruptedWriter(stream) if path == target else stream
        with patch.object(Path, "open", opened):
            with self.assertRaisesRegex(OSError, "interrupted asset write"):
                self.publish()
        self.assertEqual(target.stat().st_size, 17)
        self.assert_failed_final_refused()

    def test_local_bytes_changed_after_prevalidation_rejected_by_in_write_sha(self):
        real_open = Path.open
        def opened(path, *args, **kwargs):
            if path == self.final / "raw_wholebody134.npz":
                with real_open(self.local / "raw_wholebody134.npz", "ab") as stream:
                    stream.write(b"injected changed bytes after local fullvalidation")
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", opened):
            with self.assertRaisesRegex(ValueError, "streamed content SHA mismatch"):
                self.publish()
        self.assert_failed_final_refused()

    def test_file_fsync_failure_prevents_receipt(self):
        real_sync = os.fsync
        def sync(fd):
            if os.readlink("/proc/self/fd/" + str(fd)) == str(self.final / "raw_wholebody134.npz"):
                raise OSError("injected asset fsync failure")
            return real_sync(fd)
        with patch.object(os, "fsync", side_effect=sync):
            with self.assertRaisesRegex(OSError, "asset fsync failure"):
                self.publish()
        self.assert_failed_final_refused()

    def test_receipt_link_failure_no_receipt_final_retained(self):
        def failed_link(*args, **kwargs):
            raise OSError("injected commit link failure")
        with patch.object(os, "link", side_effect=failed_link):
            with self.assertRaisesRegex(OSError, "commit link failure"):
                self.publish()
        self.assert_failed_final_refused()
        self.assertFalse(any(p.name.startswith(".extract_receipt") for p in self.final.iterdir()))

    def test_existing_empty_identical_partial_and_symlink_are_never_reused(self):
        self.final.mkdir()
        with self.assertRaises(FileExistsError): self.publish()
        self.assertEqual(list(self.final.iterdir()), [])
        shutil.copytree(self.local, self.final, dirs_exist_ok=True)
        before = {p.relative_to(self.final).as_posix(): p.read_bytes() for p in self.final.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError): self.publish()
        for p in self.final.rglob("*"):
            if p.is_file(): self.assertEqual(p.read_bytes(), before[p.relative_to(self.final).as_posix()])
        adapter = publication.LocalROI(roi, self.root)
        self.assertEqual(adapter.ensure_prepared(self.shared, self.prepared), self.final)
        with patch.object(roi, "extract_source", side_effect=AssertionError("Unexpected inference")):
            self.assertEqual(adapter.extract_source(self.final, self.prepared, None),
                             roi.validate_completed(self.final, self.prepared))
        for name, target in (("dangling", self.shared / "absent"), ("alias", self.final)):
            link = self.shared / name
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError): publication.publish_roi(roi, self.local, link, self.prepared, 0.)
            self.assertTrue(link.is_symlink())
            self.assertFalse((self.shared / "absent").exists())

    def test_completed_resume_reuses_without_writes_or_inference(self):
        expected = self.publish()
        before = {p.relative_to(self.final).as_posix(): p.read_bytes()
                  for p in self.final.rglob("*") if p.is_file()}
        adapter = publication.LocalROI(roi, self.root)
        with patch.object(roi, "extract_source", side_effect=AssertionError("Unexpected inference")), \
             ActualIOCounts(self.shared) as operations:
            self.assertEqual(adapter.ensure_prepared(self.shared, self.prepared), self.final)
            self.assertEqual(adapter.extract_source(self.final, self.prepared, None), expected)
        self.assertEqual(operations.counts["shared_link"], 0)
        self.assertEqual(operations.counts["shared_unlink"], 0)
        self.assertEqual(operations.counts["shared_file_fsync"], 0)
        self.assertEqual(before, {p.relative_to(self.final).as_posix(): p.read_bytes()
                                 for p in self.final.rglob("*") if p.is_file()})

    def test_partial_resume_remains_rejected_and_retained(self):
        self.final.mkdir()
        (self.final / "source_manifest.json").write_bytes(roi.json_bytes(self.prepared))
        before = (self.final / "source_manifest.json").read_bytes()
        adapter = publication.LocalROI(roi, self.root)
        with self.assertRaisesRegex(ValueError, "Incomplete extraction"):
            adapter.ensure_prepared(self.shared, self.prepared)
        self.assertEqual((self.final / "source_manifest.json").read_bytes(), before)
        self.assertFalse((self.final / "extract_receipt.json").exists())

    def test_crop_escape_and_local_symlink_and_unexpected_file_refused(self):
        path = self.local / "roi_manifest.json"
        original = json.loads(path.read_bytes())
        escaped = copy.deepcopy(original)
        escaped["crops"][0]["path"] = str(self.root / "escaped.jpg")
        (self.root / "escaped.jpg").write_bytes(Path(original["crops"][0]["path"]).read_bytes())
        path.write_bytes(roi.json_bytes(escaped))
        receipt_path = self.local / "extract_receipt.json"
        receipt = json.loads(receipt_path.read_bytes())
        receipt["roi_manifest_sha256"] = roi.sha256(path)
        receipt_path.write_bytes(roi.json_bytes(receipt))
        with self.assertRaises(ValueError): self.publish()
        self.assertFalse(self.final.exists())
        path.write_bytes(roi.json_bytes(original))
        receipt["roi_manifest_sha256"] = roi.sha256(path)
        receipt_path.write_bytes(roi.json_bytes(receipt))
        link = self.local / "unexpected_symlink"
        link.symlink_to(self.root / "source_fixture.bin")
        with self.assertRaisesRegex(ValueError, "directory/symlink"): self.publish()
        link.unlink()
        (self.local / "unknown_user_file").write_bytes(b"retain user bytes")
        with self.assertRaisesRegex(ValueError, "Unexpected/missing"): self.publish()
        self.assertFalse(self.final.exists())

    def test_actual_operation_counter_wraps_real_a_b_publication(self):
        final_a = self.shared / "A_v3" / self.prepared["id"]
        with ActualIOCounts(self.shared) as count_a:
            previous.publish_roi(roi, self.local, final_a, self.prepared, None)
        with ActualIOCounts(self.shared) as count_b:
            self.publish()
        self.assertEqual(count_b.counts["shared_link"], 1)
        self.assertEqual(count_b.counts["shared_unlink"], 1)
        self.assertLess(count_b.counts["shared_file_open"], count_a.counts["shared_file_open"])
        self.assertLess(count_b.counts["shared_hash_read_bytes"], count_a.counts["shared_hash_read_bytes"])
        self.assertEqual(count_b.counts["shared_file_fsync"], count_a.counts["shared_file_fsync"])
        self.assertEqual(count_b.counts["shared_file_fsync"], 11)
        roi.validate_completed(final_a, self.prepared)
        roi.validate_completed(self.final, self.prepared)

    def test_referenced_crop_symlink_refused_without_ownership(self):
        manifest = roi.validate_completed(self.local, self.prepared)
        crop = Path(manifest["crops"][0]["path"])
        external = self.root / "copied_crop.jpg"
        shutil.copyfile(crop, external)
        crop.unlink()
        crop.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "not canonical"):
            self.publish()
        self.assertFalse(self.final.exists())
        self.assertTrue(crop.is_symlink())
        self.assertEqual(roi.sha256(external), manifest["crops"][0]["sha256"])

    def test_source_identity_change_during_publish_stops_before_manifest_and_receipt(self):
        real_open = Path.open
        def opened(path, *args, **kwargs):
            if path == self.final / "raw_wholebody134.npz":
                with real_open(Path(self.prepared["source"]["path"]), "ab") as stream:
                    stream.write(b"injected source identity change")
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", opened):
            with self.assertRaisesRegex(ValueError, "Identity file changed"):
                self.publish()
        self.assertFalse((self.final / "roi_manifest.json").exists())
        self.assert_failed_final_refused()

    def test_model_identity_changed_before_publish_creates_no_final(self):
        model = Path(next(iter(self.prepared["model_ref"]["files"].values()))["path"])
        with model.open("ab") as stream: stream.write(b"injected model identity change")
        with self.assertRaisesRegex(ValueError, "Identity file changed"):
            self.publish()
        self.assertFalse(self.final.exists())

    def test_shared_corruption_at_commit_is_rejected_by_real_final_native_validator(self):
        real_link = os.link
        def link(source, target, *args, **kwargs):
            with (self.final / "raw_wholebody134.npz").open("ab") as stream:
                stream.write(b"injected shared corruption at commit")
            return real_link(source, target, *args, **kwargs)
        with patch.object(os, "link", side_effect=link):
            with self.assertRaisesRegex(ValueError, "content SHA"):
                self.publish()
        self.assertTrue((self.final / "extract_receipt.json").exists())
        self.assertFalse((self.shared / "producer.terminal.json").exists())
        with self.assertRaises(FileExistsError): self.publish()

    def test_incomplete_or_bad_crop_hash_refused_before_ownership(self):
        crop = next((self.local / "crops").iterdir())
        with crop.open("ab") as stream: stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "content SHA"): self.publish()
        self.assertFalse(self.final.exists())
        (self.local / "extract_receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete extraction"): self.publish()
        self.assertFalse(self.final.exists())


class ActualIOCounts:
    """Wrap actual Python file-operation entrypoints; never replace outcomes."""
    def __init__(self, shared):
        self.shared = str(shared) + os.sep
        self.root = str(shared)
        self.counts = dict(shared_file_open=0, shared_directory_open=0, shared_link=0, shared_unlink=0,
                           shared_file_fsync=0, shared_directory_fsync=0, shared_hash_read_bytes=0,
                           local_hash_read_bytes=0, shared_stat=0, shared_lstat=0, shared_scandir=0)
        self.hash_depth = 0
        self.stack = ExitStack()

    def is_shared(self, path):
        if not isinstance(path, (str, bytes, os.PathLike)): return False
        text = os.fsdecode(path)
        return text == self.root or text.startswith(self.shared)

    def __enter__(self):
        real_io_open, real_builtin_open, real_os_open = python_io.open, builtins.open, os.open
        real_fsync, real_link, real_unlink = os.fsync, os.link, os.unlink
        real_sha = roi.sha256
        owner = self
        class HashReader:
            def __init__(self, stream, shared): self.stream, self.shared = stream, shared
            def __enter__(self): self.stream.__enter__(); return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def read(self, *args, **kwargs):
                chunk = self.stream.read(*args, **kwargs)
                owner.counts["shared_hash_read_bytes" if self.shared else "local_hash_read_bytes"] += len(chunk)
                return chunk
            def __getattr__(self, name): return getattr(self.stream, name)
        def opened(real, path, *args, **kwargs):
            stream = real(path, *args, **kwargs)
            if self.is_shared(path): self.counts["shared_file_open"] += 1
            return HashReader(stream, self.is_shared(path)) if self.hash_depth else stream
        def low_open(path, flags, *args, **kwargs):
            fd = real_os_open(path, flags, *args, **kwargs)
            if self.is_shared(path):
                self.counts["shared_directory_open" if flags & os.O_DIRECTORY else "shared_file_open"] += 1
            return fd
        def fsync(fd):
            real_fsync(fd)
            if self.is_shared(os.readlink("/proc/self/fd/" + str(fd))):
                self.counts["shared_directory_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "shared_file_fsync"] += 1
        def link(source, target, *args, **kwargs):
            result = real_link(source, target, *args, **kwargs)
            if self.is_shared(target): self.counts["shared_link"] += 1
            return result
        def unlink(path, *args, **kwargs):
            result = real_unlink(path, *args, **kwargs)
            if self.is_shared(path): self.counts["shared_unlink"] += 1
            return result
        def sha(path):
            self.hash_depth += 1
            try: return real_sha(path)
            finally: self.hash_depth -= 1
        self.stack.enter_context(patch.object(python_io, "open", lambda path, *a, **kw: opened(real_io_open, path, *a, **kw)))
        self.stack.enter_context(patch.object(builtins, "open", lambda path, *a, **kw: opened(real_builtin_open, path, *a, **kw)))
        for name, wrapper in (("open", low_open), ("fsync", fsync), ("link", link), ("unlink", unlink)):
            self.stack.enter_context(patch.object(os, name, wrapper))
        self.stack.enter_context(patch.object(roi, "sha256", sha))
        for name in ("stat", "lstat", "scandir"):
            real = getattr(os, name)
            def measured(path, *args, _real=real, _name=name, **kwargs):
                result = _real(path, *args, **kwargs)
                if self.is_shared(path): self.counts["shared_" + _name] += 1
                return result
            self.stack.enter_context(patch.object(os, name, measured))
        return self

    def __exit__(self, *args): return self.stack.__exit__(*args)


def replay_three(inputs, result_path, optimized_first=False):
    if len(inputs) != 3 or len(set(inputs)) != 3:
        raise ValueError("Replay requires exactly three distinct completed native ROI directories")
    records = []
    with tempfile.TemporaryDirectory(prefix="native-roi-v4-replay-local-") as local_root:
        shared_root = Path(tempfile.mkdtemp(prefix="native-roi-v4-replay-shared-", dir=RT / "smalltests"))
        try:
            for index, item in enumerate(inputs):
                original = Path(item).resolve(strict=True)
                prepared = json.loads((original / "source_manifest.json").read_bytes())
                manifest = roi.validate_completed(original, prepared)
                # No repeated weight hashing, media copying, detector, decoder or inference.
                roi.verify_identity(prepared["source"])
                roi.verify_model(prepared["model_ref"])
                local = Path(local_root) / prepared["id"]
                copied_started = time.monotonic()
                local.mkdir()
                (local / "crops").mkdir()
                for relative in ("source_manifest.json", "prepare_receipt.json", "raw_wholebody134.npz"):
                    shutil.copyfile(original / relative, local / relative)
                for crop in manifest["crops"]:
                    shutil.copyfile(crop["path"], local / "crops" / Path(crop["path"]).name)
                mapped = publication.base.remap_paths(manifest, original, local)
                (local / "roi_manifest.json").write_bytes(roi.json_bytes(mapped))
                receipt = json.loads((original / "extract_receipt.json").read_bytes())
                receipt["roi_manifest_sha256"] = roi.sha256(local / "roi_manifest.json")
                (local / "extract_receipt.json").write_bytes(roi.json_bytes(receipt))
                roi.validate_completed(local, prepared)
                copy_seconds = time.monotonic() - copied_started
                pair = {}
                variants = [("A_v3", previous), ("B_v4", publication)]
                if optimized_first:
                    variants.reverse()
                for variant, module in variants:
                    parent = shared_root / variant
                    parent.mkdir(exist_ok=True)
                    final = parent / prepared["id"]
                    started = time.monotonic()
                    with ActualIOCounts(shared_root) as observed:
                        result = module.publish_roi(roi, local, final, prepared, None)
                    seconds = time.monotonic() - started
                    expected = publication.base.remap_paths(manifest, original, final)
                    if result != expected or roi.validate_completed(final, prepared) != expected:
                        raise ValueError("Native replay content/params/PTS/affines changed")
                    for relative in ("source_manifest.json", "prepare_receipt.json", "raw_wholebody134.npz"):
                        if (original / relative).read_bytes() != (final / relative).read_bytes():
                            raise ValueError("Native replay source/raw bytes changed: " + relative)
                    for crop in result["crops"]:
                        if roi.sha256(crop["path"]) != crop["sha256"]:
                            raise ValueError("Native replay JPEG SHA changed")
                        if cv2.imread(crop["path"]) is None:
                            raise ValueError("Native replay JPEG cannot decode")
                    pair[variant] = {"actual_operations": observed.counts, "full_publish_wall_seconds": seconds,
                                     "native_validate_completed": True, "all_metadata_equal_after_only_output_path_remap": True,
                                     "source_prepare_raw_jpeg_bytes_unchanged": True, "output_directory": str(final)}
                records.append({"id": prepared["id"], "original_directory": str(original),
                                "source_binding_sha256": prepared["binding_sha256"],
                                "model_sha256": prepared["model_ref"]["sha256"],
                                "num_crops": len(manifest["crops"]),
                                "local_extract_seconds": None, "local_extract_rerun": False,
                                "local_byte_replay_copy_and_verify_seconds": copy_seconds, **pair})
            totals = {}
            for variant in ("A_v3", "B_v4"):
                totals[variant] = {"full_publish_wall_seconds": sum(r[variant]["full_publish_wall_seconds"] for r in records),
                    "actual_operations": {key: sum(r[variant]["actual_operations"][key] for r in records)
                                          for key in records[0][variant]["actual_operations"]}}
            result = {"schema": "native_roi_publication_v4_cpu_replay3", "records": records, "totals": totals,
                      "variant_order": [name for name, _ in variants], "node": os.uname().nodename,
                      "measurement_scope": "actual Python filesystem entrypoints and SHA stream reads during complete publish_roi including local preflight and final native validation; not strace/kernel syscall counts",
                      "no_detector_or_inference_or_gpu": True, "no_production_database_writes": True,
                      "no_source_media_or_model_weight_hash_replay": True, "fsync_retained_per_shared_file": True,
                      "directory_visibility_atomic": False, "temporary_output_removed_after_test": True,
                      "speedup_claim": None, "semantic_approval": False, "formal_training": False,
                      "source_sha256": {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                        for module in (previous, publication)},
                      "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            with Path(result_path).open("x") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            return result
        finally:
            shutil.rmtree(shared_root)  # This function owns only this unique test tree.


if __name__ == "__main__":
    if "--replay-inputs" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--replay-inputs", nargs=3, required=True)
        parser.add_argument("--result-json", required=True)
        parser.add_argument("--optimized-first", action="store_true")
        args = parser.parse_args()
        result = replay_three(args.replay_inputs, args.result_json, args.optimized_first)
        print(json.dumps({"result_json": args.result_json, "totals": result["totals"]}, ensure_ascii=False))
    else:
        unittest.main()
