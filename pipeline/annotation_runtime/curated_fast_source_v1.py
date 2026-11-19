"""Single-source immutable input caching and indexed frame selection."""
from __future__ import annotations

import base64
from bisect import bisect_left
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import stat

RT = Path(__file__).resolve().parent
IMAGE_CACHE = RT / "curated_request_image_cache_v1.py"
IMAGE_CACHE_SHA = "8e40c5f784ebd548a2ab7d27450cb75a8d4e145576886ed690330925b51c79e3"
raw = IMAGE_CACHE.read_bytes()
if hashlib.sha256(raw).hexdigest() != IMAGE_CACHE_SHA:
    raise ValueError("Original image request implementation changed")
spec = importlib.util.spec_from_file_location("fast_source_native_images", IMAGE_CACHE)
images = importlib.util.module_from_spec(spec)
exec(compile(raw, str(IMAGE_CACHE), "exec"), images.__dict__)


def fingerprint(path=None, fd=None):
    value = os.stat(path) if fd is None else os.fstat(fd)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Source input must be a regular file")
    return tuple(getattr(value, k) for k in
                 ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


class SourceAssets:
    def __init__(self, root):
        self.root = Path(os.path.abspath(root))
        self.allowed = set()
        self.values = {}
        self.identities = {}
        self.documents = {}
        self.urls = {}
        self.active = False
        self.rgb_resolved = None
        self.stats = {"input_reads": 0, "input_bytes": 0, "cache_hits": 0,
                      "sha_reuses": 0, "json_parses": 0, "boundary_stat_checks": 0}
        for name in ("binding.json", "plan.json", "prepared.json", "prepared.receipt.json",
                     "actual_pts.json", "source.identity.json", "source.refs.json", "rgb/manifest.json"):
            self.allow(self.root / name)

    def key(self, path):
        return Path(os.path.abspath(os.fspath(path)))

    def allow(self, path):
        self.allowed.add(self.key(path))

    def capture(self, path, expected=None):
        if not self.active:
            raise ValueError("Input cache outside source context")
        key = self.key(path)
        if key not in self.allowed:
            raise ValueError("Input outside source cache closure")
        if key in self.values:
            value, sha = self.values[key]
            self.stats["cache_hits"] += 1
        else:
            before = fingerprint(key)
            resolved = key.resolve(strict=True)
            if key.is_relative_to(self.root / "rgb"):
                if self.rgb_resolved is None:
                    self.rgb_resolved = (self.root / "rgb").resolve(strict=True)
                if not resolved.is_relative_to(self.rgb_resolved):
                    raise ValueError("Body JPEG symlink escapes source RGB root")
            with key.open("rb") as stream:
                opened = fingerprint(fd=stream.fileno())
                value = stream.read()
                finished = fingerprint(fd=stream.fileno())
            if not before == opened == finished == fingerprint(key) or len(value) != before[2]:
                raise ValueError("Source input changed during read")
            sha = hashlib.sha256(value).hexdigest()
            self.identities[key] = (before, resolved)
            self.values[key] = (value, sha)
            self.stats["input_reads"] += 1
            self.stats["input_bytes"] += len(value)
        if expected is not None and expected != sha:
            raise ValueError("Source input expected SHA differs")
        return value, sha

    def load(self, path, parser):
        key = self.key(path)
        if key not in self.allowed:
            return parser(path)
        value, _ = self.capture(key)
        if key not in self.documents:
            # Preserve the original strict JSON parser, including duplicate-key rejection.
            self.documents[key] = self.parse_bytes(value)
            self.stats["json_parses"] += 1
        return copy.deepcopy(self.documents[key])

    def bind_prepared(self, prepared):
        for section in ("raw2d_mask", "hand_roi"):
            ref = prepared[section]
            if ref.get("available"):
                self.allow(ref["path"])
        roi = prepared["hand_roi"]
        if roi.get("available"):
            for name in ("source_manifest.json", "extract_receipt.json"):
                self.allow(Path(roi["path"]).with_name(name))
        for window in prepared["windows"]:
            for frame in window["frames"]:
                self.body_path(frame["file"])
            for part in ("left_hand", "right_hand"):
                for crop in (window.get("hand_roi_evidence") or {}).get(part, {}).get("crops", []):
                    self.allow(crop["path"])
        for frame in self.load(self.root / "rgb/manifest.json", self.original_load)["frames"]:
            self.body_path(frame["file"])

    def body_path(self, filename):
        path = self.key(self.root / "rgb" / filename)
        if not path.is_relative_to(self.root / "rgb"):
            raise ValueError("Body JPEG escapes source RGB root")
        self.allow(path)
        return path

    def data_url(self, path, expected, kind):
        key = self.key(path)
        if key not in self.allowed and kind == "body":
            self.body_path(key.relative_to(self.root / "rgb"))
        value, sha = self.capture(key, expected)
        if key not in self.urls:
            self.urls[key] = "data:image/jpeg;base64," + base64.b64encode(value).decode("ascii")
        self.stats["sha_reuses"] += 1
        return self.urls[key]

    def verify_boundary(self):
        if self.rgb_resolved is not None and (self.root / "rgb").resolve(strict=True) != self.rgb_resolved:
            raise ValueError("Source RGB directory changed before result commit")
        for path, (identity, _) in self.identities.items():
            self.stats["boundary_stat_checks"] += 1
            if fingerprint(path) != identity:
                raise ValueError("Source input changed before result commit")


class FrameSelection:
    def __init__(self):
        self.times = None
        self.windows = {}

    def select(self, times, start, end, sample_fps=4.0, max_frames=16):
        if self.times is not times:
            self.times = times
            self.windows.clear()
        key = (start, end, sample_fps, max_frames)
        if key not in self.windows:
            available = range(bisect_left(times, start), bisect_left(times, end))
            if len(available) < 2:
                raise ValueError("requested window contains fewer than two frames")
            selected = [available[0]]
            for index in available[1:]:
                if times[index] - times[selected[-1]] >= 1 / sample_fps - 1e-9:
                    selected.append(index)
            if selected[-1] != available[-1]:
                selected.append(available[-1])
            if len(selected) > max_frames:
                selected = [selected[round(k * (len(selected) - 1) / (max_frames - 1))]
                            for k in range(max_frames)]
            self.windows[key] = selected
        return list(self.windows[key])

    def window(self, times, start, end, sample_fps):
        available = range(bisect_left(times, start), bisect_left(times, end))
        native_short = end - start < 2 / sample_fps
        if len(available) < 2 or native_short:
            selected = list(available)
            if len(selected) > 16:
                selected = [available[round(k * (len(available) - 1) / 15)] for k in range(16)]
        else:
            selected = self.select(times, start, end, sample_fps, 16)
        return selected, {"requested_fps": sample_fps, "max_frames": 16,
                          "native_short_window_fallback": native_short,
                          "native_available_frame_count": len(available),
                          "half_open_window": True, "indices_are_original_global": True}


@contextmanager
def fast_source(bar, root):
    if getattr(bar, "_fast_source_active", False):
        raise ValueError("Concurrent/nested source cache is forbidden")
    assets = SourceAssets(root)
    originals = {"load": bar.load, "sha256": bar.batch.sha256, "verify_prepared": bar.verify_prepared,
                 "prepare_source": bar.prepare_source, "select_frames": bar.batch.select_frames,
                 "select_window_frames": bar.select_window_frames}
    assets.original_load = originals["load"]
    assets.parse_bytes = bar.strict_json
    frame_index = FrameSelection()
    image = images.SourceImageCache(bar, assets.root / "rgb")
    image._data_url = assets.data_url

    def load(path):
        return assets.load(path, originals["load"])

    def sha256(path):
        if assets.key(path) not in assets.allowed:
            return originals["sha256"](path)
        assets.stats["sha_reuses"] += 1
        return assets.capture(path)[1]

    def prepared_call(function, *args, **kwargs):
        value = function(*args, **kwargs)
        assets.bind_prepared(value)
        return value

    def verify_prepared(*args, **kwargs):
        # Bind immutable inputs before the original verifier hashes every frame.
        assets.bind_prepared(load(assets.root / "prepared.json"))
        return prepared_call(originals["verify_prepared"], *args, **kwargs)

    def prepare_source(*args, **kwargs):
        return prepared_call(originals["prepare_source"], *args, **kwargs)

    bar._fast_source_active = assets.active = True
    bar.load, bar.batch.sha256 = load, sha256
    bar.verify_prepared, bar.prepare_source = verify_prepared, prepare_source
    bar.batch.select_frames = frame_index.select
    bar.select_window_frames = frame_index.window
    try:
        with image:
            yield assets
            assets.verify_boundary()
    finally:
        bar.load, bar.batch.sha256 = originals["load"], originals["sha256"]
        bar.verify_prepared, bar.prepare_source = originals["verify_prepared"], originals["prepare_source"]
        bar.batch.select_frames = originals["select_frames"]
        bar.select_window_frames = originals["select_window_frames"]
        bar._fast_source_active = assets.active = False
        assets.values.clear()
        assets.documents.clear()
        assets.urls.clear()
