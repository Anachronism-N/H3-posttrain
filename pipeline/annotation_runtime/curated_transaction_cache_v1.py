"""LOCK-scoped dynamic artifact reuse around unchanged native transaction bytecode."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import stat
import time
from types import FunctionType

NATIVE_SHA = {
    "ms_bar_annotation.py": "df29139651345ebd4e743657f3553f2e5165c8ab327bfd66ac84f225dc63005e",
    "ms_caption_rewrite.py": "6052fbb8f799c8a88833b8f564a935dad4562e16b8faf63f3f265143c0c43232",
    "ms_visual_batch.py": "547b005a779357e25845e01a048d2eb6a79d41d01a9d534f4ebc901405a816e1",
    "ms_optimize.py": "d519390c8588a1019674fa15e9de932b89504171ae682e6c0c1cf75ae2378350",
}
ARTIFACTS = frozenset(("request.json", "binding.json", "response.raw.txt",
                       "generation.json", "response.json", "final.json"))
PHASES = ("requestprepare", "provenanceverify", "chatwall", "outputpersist", "finalvalidate")


def clone(function, **replacements):
    namespace = {**function.__globals__, **replacements}
    value = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    value.__kwdefaults__ = function.__kwdefaults__
    return value


class Proxy:
    def __init__(self, original, **overrides):
        self.original = original
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self.original, name)


def fingerprint(path=None, fd=None):
    value = os.stat(path) if fd is None else os.fstat(fd)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Transaction artifact must be a regular file")
    return tuple(getattr(value, name) for name in
                 ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


class TransactionCache:
    def __init__(self, root, parser):
        self.root = Path(os.path.abspath(root))
        self.parser = parser
        self.active = False
        self.values = {}
        self.identities = {}
        self.documents = {}
        self.root_identity = None
        self.stats = {"artifact_reads": 0, "artifact_bytes": 0, "cache_hits": 0,
                      "sha_reuses": 0, "json_parses": 0, "boundary_stat_checks": 0,
                      "successful_save_captures": 0}

    def key(self, path):
        return Path(os.path.abspath(os.fspath(path)))

    def allowed(self, path):
        key = self.key(path)
        return key.parent == self.root and key.name in ARTIFACTS

    def remember(self, path, raw, identity):
        key = self.key(path)
        self.values[key] = (raw, hashlib.sha256(raw).hexdigest())
        self.identities[key] = (identity, key.resolve(strict=True))
        if key.suffix == ".json":
            try:
                self.documents[raw] = self.parser(raw)
                self.stats["json_parses"] += 1
            except (ValueError, TypeError):
                # Invalid on-disk JSON must still fail at the original parser gate.
                pass

    def capture(self, path, expected=None):
        key = self.key(path)
        if not self.active or not self.allowed(key):
            raise ValueError("Transaction artifact outside active LOCK scope/closure")
        if key not in self.values:
            before = fingerprint(key)
            with key.open("rb") as stream:
                opened = fingerprint(fd=stream.fileno())
                raw = stream.read()
                finished = fingerprint(fd=stream.fileno())
            if not before == opened == finished == fingerprint(key) or len(raw) != before[2]:
                raise ValueError("Transaction artifact changed during read")
            self.remember(key, raw, before)
            self.stats["artifact_reads"] += 1
            self.stats["artifact_bytes"] += len(raw)
        else:
            self.stats["cache_hits"] += 1
        raw, sha = self.values[key]
        if expected is not None and expected != sha:
            raise ValueError("Transaction artifact expected SHA differs")
        return raw, sha

    def parse(self, raw):
        if self.active and raw in self.documents:
            return copy.deepcopy(self.documents[raw])
        return self.parser(raw)

    def verify_boundary(self):
        if not self.active:
            return
        root_stat = self.root.stat()
        if (self.root.resolve(strict=True), root_stat.st_dev, root_stat.st_ino) != self.root_identity:
            raise ValueError("Transaction root changed before result commit")
        for path, (identity, resolved) in self.identities.items():
            self.stats["boundary_stat_checks"] += 1
            if fingerprint(path) != identity:
                raise ValueError("Transaction artifact changed before result commit")

    def clear(self):
        self.active = False
        self.values.clear()
        self.identities.clear()
        self.documents.clear()


class CachedPath:
    def __init__(self, path, cache):
        self.path, self.cache = Path(path), cache

    def __fspath__(self):
        return os.fspath(self.path)

    def __str__(self):
        return str(self.path)

    def __truediv__(self, name):
        return CachedPath(self.path / name, self.cache)

    def __getattr__(self, name):
        return getattr(self.path, name)

    def read_bytes(self):
        if self.cache.active and self.cache.allowed(self.path):
            return self.cache.capture(self.path)[0]
        return self.path.read_bytes()

    def read_text(self, encoding=None, errors=None):
        if self.cache.active and self.cache.allowed(self.path):
            with io.TextIOWrapper(io.BytesIO(self.read_bytes()), encoding=encoding, errors=errors) as stream:
                return stream.read()
        return self.path.read_text(encoding=encoding, errors=errors)


class NativeTransactions:
    def __init__(self, bar, emit, source_boundary=None):
        self.bar, self.core = bar, bar.core
        self.emit, self.source_boundary = emit, source_boundary
        self.original = bar.transactions
        self.metrics = None

    def measured(self, phase, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            if self.metrics is not None:
                self.metrics[phase] += time.perf_counter() - started

    def transaction(self, root, request, llm, sampling, validator, binding):
        cache = TransactionCache(root, self.core.strict_json)
        started = time.perf_counter()
        resumed = False

        @contextmanager
        def bundle_lock(path):
            nonlocal resumed
            with self.core.batch.bundle_lock(Path(path)):
                resumed = (Path(path) / "receipt.json").exists()
                # Receipt resume is always the original full read/hash/validator gate.
                cache.active = not resumed
                root_stat = cache.root.stat()
                cache.root_identity = (cache.root.resolve(strict=True), root_stat.st_dev, root_stat.st_ino)
                try:
                    yield
                    self.measured("finalvalidate", cache.verify_boundary)
                finally:
                    cache.clear()

        def sha256(path):
            if cache.active and cache.allowed(path):
                cache.stats["sha_reuses"] += 1
                return cache.capture(path)[1]
            return self.core.batch.sha256(Path(path))

        def save_once(path, raw):
            path = Path(path)
            if path.name == "receipt.json":
                self.measured("finalvalidate", cache.verify_boundary)
                if self.source_boundary is not None:
                    self.measured("provenanceverify", self.source_boundary)
            phase = "requestprepare" if path.name in ("request.json", "binding.json") else "outputpersist"
            before = fingerprint(path) if cache.active and cache.allowed(path) and path.exists() else None
            self.measured(phase, self.core.save_once, path, raw)
            if cache.active and cache.allowed(path):
                after = fingerprint(path)
                if before is not None and before != after:
                    raise ValueError("Transaction artifact changed during write-once validation")
                if len(raw) != after[2]:
                    raise ValueError("Transaction artifact size differs after native save")
                cache.remember(path, raw, after)
                cache.stats["successful_save_captures"] += 1

        save_json = clone(self.core.save_json, save_once=save_once)
        batch = Proxy(self.core.batch, sha256=sha256, bundle_lock=bundle_lock)
        generation = clone(self.core.validate_generation, batch=batch, strict_json=cache.parse)
        function = clone(self.core.transaction, Path=lambda p: CachedPath(p, cache),
                         batch=batch, save_once=save_once, save_json=save_json, strict_json=cache.parse,
                         validate_generation=lambda *a: self.measured("provenanceverify", generation, *a))
        try:
            return function(root, request, self.model(llm) if llm is not None else None, sampling,
                            lambda value: self.measured("finalvalidate", validator, value), binding)
        finally:
            cache.clear()
            self.emit({"event": "native_core_transaction", "root": str(root),
                       "receipt_resume_full_validation": resumed,
                       "core_transactionwall": time.perf_counter() - started, **cache.stats})

    def model(self, llm):
        return Proxy(llm, chat=lambda *a, **kw: self.measured("chatwall", llm.chat, *a, **kw))

    def run(self, items, runtime, args):
        if self.metrics is not None:
            raise ValueError("Concurrent/nested native transactions are forbidden")
        self.metrics = {name: 0.0 for name in PHASES}
        started = time.perf_counter()
        core = Proxy(self.core, transaction=self.transaction,
                     structured_sampling=lambda *a, **kw: self.measured(
                         "requestprepare", self.core.structured_sampling, *a, **kw),
                     save_once=lambda *a: self.measured("outputpersist", self.core.save_once, *a),
                     save_json=lambda *a: self.measured("outputpersist", self.core.save_json, *a))
        persist = clone(self.bar.persist_completion, core=core)
        native = clone(self.original, core=core, persist_completion=persist)

        def measured_runtime():
            llm, sampling_cls = runtime()
            return self.model(llm), sampling_cls

        try:
            return native(items, measured_runtime, args)
        finally:
            value = {"event": "native_transactions_timing", "roots": [str(i["root"]) for i in items],
                     **self.metrics, "transactionwall": time.perf_counter() - started,
                     "scope": "bar.transactions_only_not_full_source",
                     "chatwall_is_model_API_wall_not_CUDA_active_or_prefill_decode": True}
            self.metrics = None
            self.emit(value)


@contextmanager
def transaction_cache(bar, emit, source_boundary=None):
    """Install only bar.transactions; emit is an independent stdout/JSONL callback."""
    for name, expected in NATIVE_SHA.items():
        path = Path(bar.__file__).with_name(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen native transaction source changed: " + name)
    if getattr(bar, "_transaction_cache_v1_active", False):
        raise ValueError("Transaction cache installation reentry is forbidden")
    state = NativeTransactions(bar, emit, source_boundary)
    bar._transaction_cache_v1_active = True
    bar.transactions = state.run
    try:
        yield state
    finally:
        bar.transactions = state.original
        bar._transaction_cache_v1_active = False
