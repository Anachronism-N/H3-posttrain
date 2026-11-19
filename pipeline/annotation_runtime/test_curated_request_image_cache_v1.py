"""CPU-only real frozen-function replay: exactly three real source cases.

Run with /opt/venv_eval/bin/python -B and the explicitly configured CPU sitepath.
All mutation/race/capacity tests copy real JPEGs into temporary directories.
No model/GPU call, source edits, request writes, run integration or installation.
"""
from __future__ import annotations

from argparse import Namespace
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
BAR = RT.parent
FROZEN = RT / "source_curated_consumer_v1" / "ms_bar_annotation.py"
FROZEN_SHA = "df29139651345ebd4e743657f3553f2e5165c8ab327bfd66ac84f225dc63005e"
IDS = ("hwrlCvOCYUKpljlV", "KMcaVVwQmswUfTuP", "qKgmlBaDdOeqvUZZ")
if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA:
    raise RuntimeError("frozen source SHA mismatch")
sys.path.insert(0, str(FROZEN.parent))
import ms_bar_annotation as bar
from curated_request_image_cache_v1 import SourceImageCache
if Path(bar.__file__).resolve() != FROZEN:
    raise RuntimeError("not using the actual frozen bar module")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def source_roots():
    rows = [json.loads(line) for line in (RT / "oldbar3_explicit_alternative_candidates_v1.jsonl")
            .read_text().splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    roots = []
    for index, sid in enumerate(IDS):
        root = BAR / "full12000_stable_roi_v1" / f"shard{index}" / "sources" / sid
        if not (root / "prepared.json").is_file():
            root = Path(by_id[sid]["source_hint"])
        if root.name != sid or not (root / "prepared.json").is_file():
            raise RuntimeError("missing fixed real source " + sid)
        roots.append(root)
    return roots


@contextmanager
def jpeg_reads(rgb_root):
    original = Path.open
    counts = Counter()
    by_path = Counter()

    def observed(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if mode == "rb" and path.suffix.lower() in (".jpg", ".jpeg"):
            kind = "body" if path.absolute().is_relative_to(rgb_root.absolute()) else "crop"
            counts[kind] += 1
            by_path[str(path.absolute())] += 1
        return stream

    with patch.object(Path, "open", observed):
        yield counts, by_path


def request_set(root, prepared, args):
    forbidden = {p: set(v) for p, v in prepared["forbidden"].items()}
    requests, locations = [], []
    for manifest in prepared["windows"]:
        if len(manifest["frames"]) < 2:
            continue
        requests.append(bar.visual_request(args, root / "rgb", manifest, forbidden))
        locations.append(root / "windows" / manifest["window"]["window_id"] / "attempt-0" / "request.json")
    for spec in prepared["boundaries"]:
        item = bar.boundary_item(spec, prepared, root, args, prepared["binding"])
        if "request" in item:
            requests.append(item["request"])
            locations.append(root / "boundaries" / spec["id"] / "attempt-0" / "request.json")
    return requests, locations


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots = source_roots()
        cls.prepared = bar.load(cls.roots[0] / "prepared.json")
        cls.original_rgb_content = bar.rgb_content

    def setUp(self):
        self.assertIs(bar.rgb_content, type(self).original_rgb_content)

    def tearDown(self):
        self.assertIs(bar.rgb_content, type(self).original_rgb_content)

    def copied_sample(self, directory):
        root = Path(directory)
        rgb_root = root / "rgb"
        rgb_root.mkdir(parents=True)
        manifest = self.prepared["windows"][0]
        real_crop = next(c for hand in ("left_hand", "right_hand")
                         for c in manifest["hand_roi_evidence"][hand]["crops"])
        frame = copy.deepcopy(next(f for f in manifest["frames"] if f["index"] == real_crop["frame_index"]))
        body = rgb_root / "body.jpg"
        shutil.copy2(self.roots[0] / "rgb" / frame["file"], body)
        frame["file"] = body.name
        crop = copy.deepcopy(real_crop)
        crop_path = root / "crop.jpg"
        shutil.copy2(crop["path"], crop_path)
        crop["path"] = str(crop_path)
        evidence = {p: {"crops": [crop] if p == crop["part"] else []}
                    for p in ("left_hand", "right_hand")}
        return rgb_root, frame, evidence

    def content(self, rgb_root, frame, evidence=None):
        return bar.rgb_content(rgb_root, [frame], {"fixture": "ordered 元数据"}, evidence)

    def test_fixed_three_real_request_sets(self):
        for sid, root in zip(IDS, self.roots):
            with self.subTest(source=sid):
                prepared = bar.load(root / "prepared.json")
                first = prepared["windows"][0]["window"]["window_id"]
                recorded = bar.load(root / "windows" / first / "attempt-0" / "request.json")
                args = Namespace(model=recorded["model"], visual_max_tokens=recorded["sampling"]["max_tokens"],
                                 retry_failed=False)
                with jpeg_reads(root / "rgb") as (before_reads, before_paths):
                    before, locations = request_set(root, prepared, args)
                for request, location in zip(before, locations):
                    self.assertEqual(canonical(request), canonical(bar.load(location)))
                with jpeg_reads(root / "rgb") as (after_reads, after_paths):
                    with SourceImageCache(bar, root / "rgb") as cache:
                        after, after_locations = request_set(root, prepared, args)
                        self.assertEqual(cache.stats["read_count"], sum(after_reads.values()))
                        self.assertEqual(cache.stats["body_read_count"], after_reads["body"])
                        self.assertEqual(cache.stats["crop_read_count"], after_reads["crop"])
                        stats = cache.stats
                        self.assertLessEqual(stats["peak_body_entries"], 32)
                        self.assertLessEqual(stats["peak_crop_entries"], 64)
                        self.assertEqual(stats["stat_checks"], stats["read_count"] * 4 + stats["cache_hits"] * 2)
                self.assertEqual(locations, after_locations)
                self.assertEqual(len(before), len(after))
                hashes, total_bytes = [], 0
                for old, new in zip(before, after):
                    self.assertEqual(canonical(old), canonical(new))
                    self.assertEqual(bar.core.batch.json_bytes(old), bar.core.batch.json_bytes(new))
                    old_bytes, new_bytes = canonical(old), canonical(new)
                    old_sha, new_sha = hashlib.sha256(old_bytes).hexdigest(), hashlib.sha256(new_bytes).hexdigest()
                    self.assertEqual(old_sha, new_sha)
                    hashes.append(old_sha)
                    total_bytes += len(old_bytes)
                self.assertLess(after_reads["body"], before_reads["body"])
                self.assertEqual(before_reads["crop"], 2 * after_reads["crop"])
                for path, count in after_paths.items():
                    if not Path(path).is_relative_to(root / "rgb"):
                        self.assertEqual(count, 1)
                        self.assertEqual(before_paths[path], 2)
                self.assertLess(sum(after_reads.values()), sum(before_reads.values()))
                print(json.dumps({"source": sid, "window_count": len(prepared["windows"]),
                    "boundary_count": len(prepared["boundaries"]), "request_count": len(before),
                    "request_canonical_bytes": total_bytes, "request_sha256s": hashes,
                    "request_set_sha256": hashlib.sha256(canonical(before)).hexdigest(),
                    "canonical_and_order_preserving_bytes_equal": True,
                    "recorded_original_requests_equal": True,
                    "original_body_read_count": before_reads["body"],
                    "original_crop_read_count": before_reads["crop"],
                    "original_read_count": sum(before_reads.values()), **stats}, sort_keys=True), flush=True)

    def test_wrong_body_sha(self):
        with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
            root, frame, evidence = self.copied_sample(tmp)
            frame["sha256"] = "0" * 64
            with SourceImageCache(bar, root) as cache:
                with self.assertRaisesRegex(ValueError, "SHA differs"):
                    self.content(root, frame)
                self.assertEqual(cache.stats["read_count"], 1)

    def test_wrong_roi_sha(self):
        with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
            root, frame, evidence = self.copied_sample(tmp)
            next(c for p in evidence.values() for c in p["crops"])["sha256"] = "0" * 64
            with SourceImageCache(bar, root):
                with self.assertRaisesRegex(ValueError, "SHA differs"):
                    self.content(root, frame, evidence)

    def test_same_path_bytes_change_and_pseudo_expected_sha_update(self):
        for update_sha in (False, True):
            for kind in ("body", "crop"):
                with self.subTest(update_sha=update_sha, kind=kind):
                    with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
                        root, frame, evidence = self.copied_sample(tmp)
                        crop = next(c for p in evidence.values() for c in p["crops"])
                        record = frame if kind == "body" else crop
                        path = root / frame["file"] if kind == "body" else Path(crop["path"])
                        with SourceImageCache(bar, root) as cache:
                            self.content(root, frame, evidence)
                            reads = cache.stats["read_count"]
                            data = path.read_bytes()
                            changed = bytes([data[0] ^ 1]) + data[1:]
                            path.write_bytes(changed)
                            if update_sha:
                                record["sha256"] = hashlib.sha256(changed).hexdigest()
                            with self.assertRaisesRegex(ValueError, "expected SHA changed|stat changed"):
                                self.content(root, frame, evidence)
                            self.assertEqual(cache.stats["read_count"], reads)

    def test_stat_only_changes_and_all_snapshot_fields(self):
        for field in ("device", "inode", "size", "mtime", "ctime"):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
                    root, frame, evidence = self.copied_sample(tmp)
                    path = root / frame["file"]
                    with SourceImageCache(bar, root) as cache:
                        self.content(root, frame)
                        original_stat = os.stat
                        fields = {"device": "st_dev", "inode": "st_ino", "size": "st_size",
                                  "mtime": "st_mtime_ns", "ctime": "st_ctime_ns"}
                        def changed_stat(value, *args, **kwargs):
                            result = original_stat(value, *args, **kwargs)
                            if Path(value) == path:
                                data = {k: getattr(result, k) for k in
                                        ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}
                                data[fields[field]] += 1
                                return Namespace(**data)
                            return result
                        with patch.object(os, "stat", changed_stat):
                            with self.assertRaisesRegex(ValueError, "stat changed"):
                                self.content(root, frame)
                        self.assertEqual(cache.stats["read_count"], 1)
        with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
            root, frame, evidence = self.copied_sample(tmp)
            path = root / frame["file"]
            with SourceImageCache(bar, root):
                self.content(root, frame)
                value = path.stat()
                os.utime(path, ns=(value.st_atime_ns, value.st_mtime_ns + 1_000_000))
                with self.assertRaisesRegex(ValueError, "stat changed"):
                    self.content(root, frame)

    def test_stat_openfd_and_after_races(self):
        for stage in ("opened", "after_fd", "after_path", "cache_after_path"):
            with self.subTest(stage=stage):
                with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
                    root, frame, evidence = self.copied_sample(tmp)
                    with SourceImageCache(bar, root) as cache:
                        if stage == "cache_after_path":
                            self.content(root, frame)
                        snapshot = cache._snapshot
                        calls = 0
                        change_at = {"opened": 2, "after_fd": 3, "after_path": 4,
                                     "cache_after_path": 2}[stage]
                        def racing_snapshot(*args, **kwargs):
                            nonlocal calls
                            result = snapshot(*args, **kwargs)
                            calls += 1
                            if calls == change_at:
                                return (*result[:4], result[4] + 1)
                            return result
                        with patch.object(cache, "_snapshot", racing_snapshot):
                            with self.assertRaisesRegex(ValueError, "changed"):
                                self.content(root, frame)

    def test_original_budget_over_16_still_rejected(self):
        with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
            root, frame, evidence = self.copied_sample(tmp)
            with self.assertRaisesRegex(ValueError, "16 image budget"):
                bar.rgb_content(root, [frame] * 17, {})
            with SourceImageCache(bar, root):
                with self.assertRaisesRegex(ValueError, "16 image budget"):
                    bar.rgb_content(root, [frame] * 17, {})

    def test_direct_original_content_equal_and_repeat_read_reduction(self):
        with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
            root, frame, evidence = self.copied_sample(tmp)
            with jpeg_reads(root) as (counts, paths):
                original = self.content(root, frame, evidence)
                self.assertEqual(counts, {"body": 1, "crop": 2})
            with SourceImageCache(bar, root) as cache:
                self.assertEqual(canonical(original), canonical(self.content(root, frame, evidence)))
                self.assertEqual(canonical(original), canonical(self.content(root, frame, evidence)))
                self.assertEqual(cache.stats["read_count"], 2)
                self.assertEqual(cache.stats["cache_hits"], 2)
                self.assertEqual(cache.stats["stat_checks"], 12)
                self.assertEqual(cache.stats["hashed_bytes"], sum(Path(path).stat().st_size for path in paths))

    def test_context_restore_reentry_exit_exception_and_different_source(self):
        with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
            root, frame, evidence = self.copied_sample(tmp)
            cache = SourceImageCache(bar, root)
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                with cache:
                    self.content(root, frame, evidence)
                    with self.assertRaisesRegex(ValueError, "reentry"):
                        with cache:
                            pass
                    with self.assertRaisesRegex(ValueError, "reentry"):
                        with SourceImageCache(bar, root):
                            pass
                    with self.assertRaisesRegex(ValueError, "different RGB root"):
                        self.content(Path(tmp) / "another-rgb", frame)
                    raise RuntimeError("sentinel")
            self.assertIs(bar.rgb_content, type(self).original_rgb_content)
            self.assertFalse(cache._bindings)
            self.assertFalse(cache._images["body"])
            self.assertFalse(cache._images["crop"])
            with self.assertRaisesRegex(ValueError, "reentry"):
                with cache:
                    pass
            with self.assertRaisesRegex(ValueError, "outside its context"):
                cache._rgb_content(root, [frame], {})
            with SourceImageCache(bar, root) as new_cache:
                self.content(root, frame, evidence)
                self.assertEqual(new_cache.stats["read_count"], 2)
                self.assertEqual(new_cache.stats["cache_hits"], 0)
            other_root, other_frame, other_evidence = self.copied_sample(Path(tmp) / "new-source")
            with SourceImageCache(bar, other_root) as other_cache:
                self.content(other_root, other_frame, other_evidence)
                self.assertEqual(other_cache.stats["read_count"], 2)
                self.assertEqual(other_cache.stats["cache_hits"], 0)

    def test_lru_limits_and_evicted_bindings_do_not_accept_changed_bytes(self):
        with tempfile.TemporaryDirectory(prefix="image-cache-negative-", dir="/tmp") as tmp:
            root, frame, evidence = self.copied_sample(tmp)
            originals = {"body": root / frame["file"], "crop": Path(next(c for p in evidence.values() for c in p["crops"])["path"])}
            with SourceImageCache(bar, root) as cache:
                for kind, limit in (("body", 32), ("crop", 64)):
                    sha = hashlib.sha256(originals[kind].read_bytes()).hexdigest()
                    paths = []
                    for i in range(limit + 1):
                        path = root / f"capacity-{kind}-{i}.jpg"
                        shutil.copy2(originals[kind], path)
                        paths.append(path)
                        cache._data_url(path, sha, kind)
                    self.assertEqual(len(cache._images[kind]), limit)
                    self.assertEqual(cache.stats[f"peak_{kind}_entries"], limit)
                    reads = cache.stats["read_count"]
                    cache._data_url(paths[0], sha, kind)
                    self.assertEqual(cache.stats["read_count"], reads + 1)
                    data = paths[1].read_bytes()
                    changed = bytes([data[0] ^ 1]) + data[1:]
                    paths[1].write_bytes(changed)
                    for expected in (sha, hashlib.sha256(changed).hexdigest()):
                        with self.assertRaisesRegex(ValueError, "stat changed|expected SHA changed"):
                            cache._data_url(paths[1], expected, kind)


if __name__ == "__main__":
    unittest.main(verbosity=2)
