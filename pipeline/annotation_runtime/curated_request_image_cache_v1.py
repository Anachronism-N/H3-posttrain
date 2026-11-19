"""Explicit, synchronous single-source cache for the frozen bar.rgb_content.

Usage: with SourceImageCache(bar, source_root / "rgb") as cache: ...
Only rgb_content is temporarily replaced; window/boundary callers stay original.
This is NOT connected to any existing run. A future caller must bind this file's
SHA256 into a NEW protocol; an old code_identity cannot attest cache activation.
Image dataURLs use separate 32-body/64-ROI LRUs. Lightweight first-seen SHA/stat
bindings survive eviction until context exit, so eviction cannot accept changes.
Stats are out-of-band and must never be added to model request JSON.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import stat


class SourceImageCache:
    def __init__(self, bar, rgb_root):
        self.bar = bar
        self.rgb_root = Path(os.path.abspath(rgb_root))
        self._images = {"body": OrderedDict(), "crop": OrderedDict()}
        self._bindings = {}
        self._original = None
        self._active = False
        self._used = False
        self._reads = {"body": 0, "crop": 0}
        self._hashed_bytes = 0
        self._hits = 0
        self._stat_checks = 0
        self._peaks = {"body": 0, "crop": 0}

    @property
    def stats(self):
        return {"read_count": sum(self._reads.values()),
                "body_read_count": self._reads["body"],
                "crop_read_count": self._reads["crop"],
                "hashed_bytes": self._hashed_bytes, "cache_hits": self._hits,
                "stat_checks": self._stat_checks,
                "peak_body_entries": self._peaks["body"],
                "peak_crop_entries": self._peaks["crop"]}

    def __enter__(self):
        if self._used or getattr(self.bar.rgb_content, "_curated_source_image_cache_v1", False):
            raise ValueError("source image cache reentry is forbidden")
        self._original = self.bar.rgb_content
        self._used = self._active = True

        def replacement(root, frames, header, hand_evidence=None):
            return self._rgb_content(root, frames, header, hand_evidence)

        replacement._curated_source_image_cache_v1 = True
        self.bar.rgb_content = replacement
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.bar.rgb_content = self._original
        finally:
            self._active = False
            self._images["body"].clear()
            self._images["crop"].clear()
            self._bindings.clear()
        return False

    def _snapshot(self, path=None, descriptor=None):
        value = os.stat(path) if descriptor is None else os.fstat(descriptor)
        self._stat_checks += 1
        if not stat.S_ISREG(value.st_mode):
            raise ValueError("request JPEG is not a regular file")
        return (value.st_dev, value.st_ino, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns)

    def _data_url(self, path, expected_sha, kind):
        if not self._active:
            raise ValueError("source image cache is outside its context")
        if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
            raise ValueError("request JPEG requires an explicit SHA256")
        path = Path(os.path.abspath(path))
        binding = self._bindings.get(path)
        if binding is not None and binding[0] != expected_sha:
            raise ValueError("request JPEG expected SHA changed within source")
        before = self._snapshot(path=path)
        if binding is not None and binding[1] != before:
            raise ValueError("request JPEG stat changed within source")
        key = (path, expected_sha)
        images = self._images[kind]
        if key in images:
            if self._snapshot(path=path) != before:
                raise ValueError("request JPEG stat changed during cache reuse")
            images.move_to_end(key)
            self._hits += 1
            return images[key]
        limit = 32 if kind == "body" else 64
        if len(images) == limit:
            images.popitem(last=False)
        with path.open("rb") as stream:
            opened = self._snapshot(descriptor=stream.fileno())
            if opened != before:
                raise ValueError("request JPEG changed between stat and open")
            raw = stream.read()
            self._reads[kind] += 1
            after_fd = self._snapshot(descriptor=stream.fileno())
        after = self._snapshot(path=path)
        if not before == opened == after_fd == after or len(raw) != before[2]:
            raise ValueError("request JPEG changed during read")
        actual_sha = hashlib.sha256(raw).hexdigest()
        self._hashed_bytes += len(raw)
        if actual_sha != expected_sha:
            raise ValueError("request JPEG SHA differs from evidence")
        self._bindings[path] = (expected_sha, before)
        images[key] = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        self._peaks[kind] = max(self._peaks[kind], len(images))
        return images[key]

    def _rgb_content(self, root, frames, header, hand_evidence=None):
        if Path(os.path.abspath(root)) != self.rgb_root:
            raise ValueError("source image cache cannot serve a different RGB root")
        content = [{"type": "text", "text": json.dumps(header, ensure_ascii=False, allow_nan=False)}]
        for frame in frames:
            path = Path(os.path.abspath(Path(root) / frame["file"]))
            if not path.is_relative_to(self.rgb_root):
                raise ValueError("body JPEG escapes the bound source RGB root")
            content.extend([
                {"type": "text", "text": f"Source frame {frame['index']}, time {frame['time']:.9f}s, original PTS {frame['source_timestamp']:.9f}s."},
                {"type": "image_url", "image_url": {
                    "url": self._data_url(path, frame["sha256"], "body")}}])
            if hand_evidence:
                for part in ("left_hand", "right_hand"):
                    for crop in hand_evidence[part]["crops"]:
                        if crop["frame_index"] != frame["index"]:
                            continue
                        url = self._data_url(crop["path"], crop["sha256"], "crop")
                        metadata = {k: crop[k] for k in ("part", "frame_index", "time", "source_timestamp",
                            "crop_xyxy", "source_frame_size", "crop_pixel_size", "ROI_to_source_affine",
                            "source_to_ROI_affine", "supportpointindices", "hand_support_xyxy", "rawframe_sha256")}
                        metadata.update({"same_source_frame_as_body_context": True, "not_another_time_observation": True,
                            "crop_adds_native_detail": False, "anatomical_side_is_unreviewed": True})
                        content.extend([{"type": "text", "text": json.dumps(metadata, allow_nan=False)},
                            {"type": "image_url", "image_url": {"url": url}}])
        self.bar.need(sum(c["type"] == "image_url" for c in content) <= 16,
                      "body plus ROI exceeds 16 image budget")
        return content
