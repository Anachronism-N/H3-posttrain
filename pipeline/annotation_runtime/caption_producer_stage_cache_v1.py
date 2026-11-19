"""Opt-in, single-coordinator native producer cache; no CLI or resource actions.

Install only around the frozen native run; IOv2 selection and local ROI routing
remain unchanged. Only an acquired native bundle_lock enables source caching. Resume hints run all original checks.
The first completed validation of each local/shared root is always native.
Distinct bar/ROI probe protocols are deliberately not interchanged.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import stat
import threading
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Cache refuses symlink: " + str(path))
    resolved = path.resolve(strict=True)
    value = resolved.stat()
    return str(resolved), (value.st_dev, value.st_ino, value.st_size,
                           value.st_mtime_ns, value.st_ctime_ns, stat.S_IFMT(value.st_mode))


class SourceCache:
    """One source, under its original lock; never persisted or reused on resume."""
    def __init__(self, row, shared_directory):
        self.row = copy.deepcopy(row)
        self.shared_directory = Path(shared_directory).resolve()
        self.thread = threading.get_ident()
        self.active = True
        self.prepared = None
        self.roots = {str(self.shared_directory)}
        self.media = {str(Path(row[key]).resolve()) for key in
                      ("video_path", "audio_path", "nlf_path") if row.get(key) is not None}
        self.hashes, self.probes, self.validated = {}, {}, {}
        self.counts, self.seconds = Counter(), Counter()
        self.started = time.monotonic()
        self.status = "unfinished"

    def check_owner(self):
        if not self.active or threading.get_ident() != self.thread:
            raise RuntimeError("Source cache used outside its owning lock/coordinator")

    def bind_prepared(self, prepared):
        self.check_owner()
        if (prepared.get("id") != self.row["id"]
                or prepared.get("input_row", {}).get("id") != self.row["id"]
                or prepared.get("input_row", {}).get("manifest_record") != self.row
                or str(Path(prepared["source"]["path"]).resolve()) != str(Path(self.row["video_path"]).resolve())):
            raise ValueError("Prepared source ID/split/original row differs from locked source")
        key = canonical(prepared)
        if self.prepared is not None and key != self.prepared:
            raise ValueError("Prepared source/model/params changed within one source")
        self.prepared = key

    def register_root(self, directory, prepared):
        self.bind_prepared(prepared)
        directory = Path(directory)
        if directory.is_symlink() or directory.name != self.row["id"]:
            raise ValueError("ROI root must be canonical and source-ID bound")
        self.roots.add(str(directory.resolve(strict=True)))

    def eligible(self, resolved):
        return resolved in self.media or any(Path(resolved).is_relative_to(root) for root in self.roots)

    def sha(self, path, original):
        self.check_owner()
        resolved, before = fingerprint(path)
        if not self.eligible(resolved):
            return original(path)
        kind = "media_sha" if resolved in self.media else "jpeg_sha" if Path(resolved).suffix == ".jpg" else "roi_artifact_sha"
        cached = self.hashes.get(resolved)
        if cached is not None:
            if before != cached[0]:
                raise ValueError("Write-once/source file changed within cached source: " + resolved)
            self.counts["sha_hits"] += 1
            self.counts[kind + "_hits"] += 1
            return cached[1]
        result = original(path)
        if fingerprint(path) != (resolved, before):
            raise ValueError("File changed during original SHA: " + resolved)
        self.hashes[resolved] = (before, result)
        self.counts["sha_reads"] += 1
        self.counts[kind + "_reads"] += 1
        self.counts[kind + "_read_bytes"] += before[2]
        self.counts["sha_read_bytes"] += before[2]
        return result

    def probe(self, protocol, path, arguments, original):
        self.check_owner()
        resolved, before = fingerprint(path)
        if resolved not in self.media:
            raise ValueError("Probe path differs from locked original source")
        key = (protocol, resolved, canonical(arguments))
        cached = self.probes.get(key)
        if cached is not None:
            if before != cached[0]:
                raise ValueError("Source changed after original probe")
            self.counts["probe_hits"] += 1
            return copy.deepcopy(cached[1])
        result = original()
        if fingerprint(path) != (resolved, before):
            raise ValueError("Source changed during original probe")
        self.probes[key] = (before, copy.deepcopy(result))
        self.counts["probe_calls"] += 1
        return result

    def validate(self, directory, prepared, original):
        self.bind_prepared(prepared)
        directory = Path(directory)
        if directory.is_symlink():
            raise ValueError("ROI root is a symlink")
        root = str(directory.resolve(strict=True))
        if root not in self.roots:
            raise ValueError("ROI validation root was not prepared by this source/owner")
        cached = self.validated.get(root)
        if cached is not None:
            for path, expected in cached[1].items():
                self.counts["validation_marker_checks"] += 1
                if fingerprint(path) != expected:
                    raise ValueError("Completed source/commit marker changed after native validation")
            self.counts["validation_hits"] += 1
            return copy.deepcopy(cached[0])
        # Full native gate, including receipt, asset SHA, actual NPZ/PTS checks.
        result = original(directory, prepared)
        if json.loads((directory / "source_manifest.json").read_bytes()) != prepared:
            raise ValueError("Completed source manifest differs from expected prepared source")
        for asset in [result["raw"]] + result["crops"]:
            asset_path = Path(asset["path"])
            if asset_path.is_symlink() or asset_path.resolve(strict=True) != asset_path or not asset_path.is_relative_to(directory):
                raise ValueError("Validated ROI asset escaped canonical source root")
        # The owning source lock and native write-once publisher guarantee assets
        # are not rewritten. Reuse checks only source binding and last commit,
        # not every JPEG/NPZ or directory mtime (IO receipts may be added later).
        markers = [directory / "source_manifest.json", directory / "extract_receipt.json"]
        signatures = {str(path): fingerprint(path) for path in markers}
        self.validated[root] = (copy.deepcopy(result), signatures)
        self.counts["validation_calls"] += 1
        return result

    def close(self):
        self.check_owner()
        self.active = False
        self.hashes.clear()
        self.probes.clear()
        self.validated.clear()
        self.roots.clear()
        self.media.clear()
        self.prepared = None


class NativeStageCache:
    """Process-local patch set, explicitly restored; never patches Path or IO writes.

    API: with NativeStageCache(pipeline, bar, roi, visual, out, emit).installed(
        publishers=(io_v4.base,)) as cache:
        return cache.run_wrapper(captured_native_run)(args)

    When entered by an IOv2 run hook, roi is the raw module behind LocalROI.roi
    and publishers=(io_v4.base,) targets the actual publisher. Exit this context
    before IOv4.main restores its original publication functions.

    emit receives one compact dict per locked source (use a dedicated log).
    Timers are inclusive and can overlap; CPU tests are not throughput evidence.
    Existing lock, selection, budget, CUDA lease and write-once code are untouched.
    """
    def __init__(self, pipeline, bar, roi, visual, out, emit):
        self.pipeline, self.bar, self.roi, self.visual = pipeline, bar, roi, visual
        self.out, self.emit = Path(out).resolve(), emit
        self.current_row, self.source = None, None
        self.model_ref, self.params = None, None
        self.verified_models = set()
        self.patches = []
        self.thread = threading.get_ident()

    def timed(self, label, original, *args, **kwargs):
        source = self.source
        if source is None:
            return original(*args, **kwargs)
        source.check_owner()
        started = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            source.seconds[label] += time.monotonic() - started

    def patch(self, module, name, wrapper):
        original = getattr(module, name)
        setattr(module, name, wrapper(original))
        self.patches.append((module, name, original))

    def sha_wrapper(self, original):
        def wrapped(path):
            return original(path) if self.source is None else self.source.sha(path, original)
        return wrapped

    def probe_wrapper(self, protocol):
        def wrap(original):
            def wrapped(path, *args, **kwargs):
                def call():
                    return self.timed("probe", original, path, *args, **kwargs)
                if self.source is None:
                    return call()
                return self.source.probe(protocol, path, [args, kwargs], call)
            return wrapped
        return wrap

    def verify_model_wrapper(self, original):
        def wrapped(reference):
            key = canonical(reference)
            if self.source is not None:
                self.source.check_owner()
                if key != self.model_ref:
                    raise ValueError("ROI model reference changed after startup boundary")
                if key in self.verified_models:
                    self.source.counts["model_gate_hits"] += 1
                    return None
            if self.model_ref is None and key in self.verified_models:
                return None
            result = original(reference)
            self.verified_models.add(key)
            return result
        return wrapped

    def save_wrapper(self, original):
        def wrapped(path, value):
            if Path(path).name == "protocol.json" and "roi_model_ref" in value:
                self.roi.verify_model(value["roi_model_ref"])
                self.model_ref, self.params = canonical(value["roi_model_ref"]), canonical(value["roi_params"])
            if self.source is not None and Path(path).name == "producer.terminal.json":
                self.source.status = value.get("status", "unknown")
            return original(path, value)
        return wrapped

    def rows_wrapper(self, original):
        def wrapped(*args, **kwargs):
            try:
                for row, batch in original(*args, **kwargs):
                    if self.source is not None or threading.get_ident() != self.thread:
                        raise RuntimeError("Native source iteration cannot overlap coordinators")
                    self.current_row = copy.deepcopy(row)
                    yield row, batch
            finally:
                self.current_row = None
        return wrapped

    def lock_wrapper(self, original):
        @contextmanager
        def wrapped(directory):
            row = self.current_row
            if row is None or Path(directory).resolve() != self.pipeline.source_dir(self.out, row).resolve():
                raise ValueError("Native source lock differs from selected ID/split output")
            if self.source is not None or threading.get_ident() != self.thread:
                raise RuntimeError("Source lock/cache cannot overlap coordinators")
            with original(directory):
                source = SourceCache(row, self.out / "roi" / row["id"])
                self.source = source
                try:
                    yield
                except BaseException as error:
                    source.status = "raised_" + type(error).__name__
                    raise
                finally:
                    source.seconds["wall"] = time.monotonic() - source.started
                    report = {"schema": "caption_producer_stage_cache_timing_v1",
                              "id": row["id"], "split": row["split"],
                              "source_ordinal": row["source_ordinal"], "status": source.status,
                              "seconds": {key: source.seconds[key] for key in
                                  ("prepare", "probe", "roi_extraction", "local_validation",
                                   "shared_publish", "shared_validation", "beat", "full_final_check", "wall")},
                              "counters": dict(source.counts),
                              "timers_inclusive": True, "wall_excludes_lock_wait": True,
                              "semantic_approval": False, "formal_training": False}
                    source.close()
                    self.source = None
                    self.emit(report)
        return wrapped

    def prepare_wrapper(self, original):
        def wrapped(*args, **kwargs):
            result = self.timed("prepare", original, *args, **kwargs)
            if self.source is not None:
                if canonical(result["model_ref"]) != self.model_ref or canonical(result["params"]) != self.params:
                    raise ValueError("Prepared model/params differ from startup boundary")
                self.source.bind_prepared(result)
            return result
        return wrapped

    def ensure_wrapper(self, original):
        def wrapped(out, prepared):
            if self.source is not None:
                self.source.bind_prepared(prepared)
            result = original(out, prepared)
            if self.source is not None:
                self.source.register_root(result, prepared)
            return result
        return wrapped

    def validate_wrapper(self, original):
        def wrapped(directory, prepared):
            if self.source is None:
                return original(directory, prepared)
            label = "shared_validation" if Path(directory).resolve() == self.source.shared_directory else "local_validation"
            return self.timed(label, self.source.validate, directory, prepared, original)
        return wrapped

    def run_wrapper(self, original):
        def wrapped(*args, **kwargs):
            append = self.pipeline.append
            def timed_append(path, value):
                if value.get("status") == "ready" and value.get("source_ordinal") is not None:
                    return self.timed("full_final_check", append, path, value)
                return append(path, value)
            self.pipeline.append = timed_append
            try:
                return original(*args, **kwargs)
            finally:
                self.pipeline.append = append
        return wrapped

    def beat_wrapper(self, original):
        def wrapped(*args, **kwargs):
            tracker, params = self.timed("beat", original, *args, **kwargs)
            def tracked(*args, **kwargs):
                return self.timed("beat", tracker, *args, **kwargs)
            return tracked, params
        return wrapped

    @contextmanager
    def installed(self, publishers=()):
        if self.patches:
            raise RuntimeError("Native cache already installed")
        try:
            for module, name in ((self.pipeline, "sha"), (self.roi, "sha256"), (self.visual, "sha256")):
                self.patch(module, name, self.sha_wrapper)
            self.patch(self.roi, "verify_model", self.verify_model_wrapper)
            self.patch(self.pipeline, "save", self.save_wrapper)
            self.patch(self.pipeline, "iter_rows", self.rows_wrapper)
            self.patch(self.visual, "bundle_lock", self.lock_wrapper)
            self.patch(self.visual, "source_identity", lambda original: lambda *a, **k: self.timed("prepare", original, *a, **k))
            self.patch(self.roi, "prepare_source", self.prepare_wrapper)
            self.patch(self.roi, "ensure_prepared", self.ensure_wrapper)
            self.patch(self.roi, "validate_completed", self.validate_wrapper)
            self.patch(self.bar, "probe_actual", self.probe_wrapper("bar.probe_actual"))
            self.patch(self.roi, "probe_video", self.probe_wrapper("roi.probe_video"))
            self.patch(self.pipeline, "audio_probe", self.probe_wrapper("pipeline.audio_probe"))
            self.patch(self.roi, "extract_source", lambda original: lambda *a, **k: self.timed("roi_extraction", original, *a, **k))
            self.patch(self.pipeline, "load_beat_runtime", self.beat_wrapper)
            self.patch(self.pipeline, "run_producer", self.run_wrapper)
            for publisher in publishers:
                self.patch(publisher, "publish_roi", lambda original: lambda *a, **k: self.timed("shared_publish", original, *a, **k))
            yield self
        finally:
            for module, name, original in reversed(self.patches):
                setattr(module, name, original)
            self.patches.clear()
            self.verified_models.clear()
            self.model_ref, self.params, self.current_row = None, None, None
