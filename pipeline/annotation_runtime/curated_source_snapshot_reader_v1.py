"""Source-bound immutable reads: validate at entry/exit, not per request."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
PILOT = RT / "curated_local_input_bundle_pilot3_v1.py"
PILOT_SHA = "977e2fcbb8d9c512559eecd36764dda4532708c8edda34c321d48fcb1c6fd26a"
raw = PILOT.read_bytes()
if hashlib.sha256(raw).hexdigest() != PILOT_SHA:
    raise ValueError("Original CPU replay code changed")
spec = importlib.util.spec_from_file_location("snapshot_original_cpu_replay", PILOT)
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
exec(compile(raw, str(PILOT), "exec"), pilot.__dict__)


class SourceSnapshotReader(pilot.Reader):
    def __init__(self, source_key, entries):
        super().__init__(source_key, entries, True)
        self._active = False
        self._used = False
        self._bytes = {}
        self._json = {}
        self._known_hashes = {}
        self.committed = False
        self.boundary_stats = {"files_checked": 0, "content_bytes": 0,
                               "entry_seconds": 0.0, "exit_seconds": 0.0}

    def _verify_closure(self, capture):
        for logical, entry in self.entries.items():
            pilot.need(Path(logical).is_absolute() and entry.logical_path == logical
                       and entry.source_key == self.source_key, "Source closure identity differs")
            resolved = str(Path(logical).resolve(strict=True))
            pilot.need(resolved == entry.original_resolved_path, "Original path identity changed")
            original, original_stat = pilot.stable_bytes(logical)
            pilot.need(original_stat == entry.original_stat
                       and pilot.digest(original) == entry.expected_sha256, "Original content changed")
            local, local_stat = pilot.stable_bytes(entry.local_path)
            pilot.need(local_stat == entry.local_stat and local == original
                       and pilot.digest(local) == entry.local_content_sha256
                       == entry.expected_sha256, "Local snapshot content changed")
            self.boundary_stats["files_checked"] += 2
            self.boundary_stats["content_bytes"] += len(original) + len(local)
            if capture:
                self._bytes[logical] = local
                self._known_hashes[local] = entry.expected_sha256

    def __enter__(self):
        pilot.need(not self._used, "Source snapshot reentry forbidden")
        self._used = True
        start = time.perf_counter()
        try:
            self._verify_closure(True)
            self._active = True
            return self
        except BaseException:
            self._bytes.clear()
            self._known_hashes.clear()
            raise
        finally:
            self.boundary_stats["entry_seconds"] = time.perf_counter() - start

    def __exit__(self, exc_type, exc, traceback):
        start = time.perf_counter()
        try:
            if exc_type is None:
                self._verify_closure(False)
                self.committed = True
        finally:
            self.boundary_stats["exit_seconds"] = time.perf_counter() - start
            self._active = False
            self._bytes.clear()
            self._json.clear()
            self._known_hashes.clear()
        return False

    def read(self, path, expected_sha=None):
        pilot.need(self._active, "Snapshot read outside source context")
        logical = os.fspath(path)
        pilot.need(logical in self.entries, "Path outside source closure")
        entry = self.entries[logical]
        pilot.need(entry.source_key == self.source_key and entry.logical_path == logical,
                   "Source mapping mismatch")
        pilot.need(expected_sha is None or expected_sha == entry.expected_sha256,
                   "Expected SHA mismatch")
        value = self._bytes[logical]
        self.stats["read_count"] += 1
        self.stats["read_bytes"] += len(value)
        self.stats["snapshot_memory_reads"] = self.stats.get("snapshot_memory_reads", 0) + 1
        return value

    def load(self, path):
        logical = os.fspath(path)
        raw = self.read(path)
        if logical not in self._json:
            self._json[logical] = json.loads(raw)
        return copy.deepcopy(self._json[logical])

    def checked(self, reference):
        self.read(reference["path"], reference["sha256"])
        return self.load(reference["path"])

    def hash(self, value):
        pilot.need(self._active, "Snapshot hash outside source context")
        if value in self._known_hashes:
            self.stats["verified_sha_reuses"] = self.stats.get("verified_sha_reuses", 0) + 1
            return self._known_hashes[value]
        return super().hash(value)

    def sha(self, path):
        self.read(path)
        self.stats["verified_sha_reuses"] = self.stats.get("verified_sha_reuses", 0) + 1
        return self.entries[os.fspath(path)].expected_sha256


def run(report_path):
    pilot.need(report_path.parent == RT and not report_path.exists(), "Fresh report path required")
    c = pilot.module("snapshot_fixed_three_scope", RT / "prepare_curated_coarse_empty_n5_v1.py")
    worker = c.worker_module()
    worker.c_targets = c.TARGETS
    worker.consumer_ref = {"path": str(c.CONSUMER), "sha256": c.CONSUMER_SHA}
    launch_raw = pilot.stable_bytes(c.LAUNCH)[0]
    pilot.need(pilot.digest(launch_raw) == "9f5e51a9e2461e864c824add7370ff89fcb1b84db1eae13c59a26b81d77f6806",
               "Original launch changed")
    launch = json.loads(launch_raw)
    worker.expected_model_sha256 = launch["modelref"]["model_identity_sha256"]
    manifest_raw = pilot.stable_bytes(worker.OUT / "jobs.manifest.json")[0]
    pilot.need(launch["jobs_manifest_ref"]["sha256"] == pilot.digest(manifest_raw), "Original scope changed")
    manifest = json.loads(manifest_raw)
    pilot.need([h["source_key"] for h in manifest["jobs"]] == [k for k, _ in c.TARGETS], "Fixed three required")
    original_globals = (Path.open, Path.read_bytes, Path.resolve)
    code_refs = {str(p): pilot.digest(pilot.stable_bytes(p)[0]) for p in
                 (Path(__file__), PILOT, RT / "test_curated_source_snapshot_reader_v1.py")}
    sources = [pilot.discover(worker, hint) for hint in manifest["jobs"]]
    budgets = {str(s["budget"]): pilot.digest(pilot.stable_bytes(s["budget"])[0]) for s in sources}
    local = Path(tempfile.mkdtemp(prefix="source-snapshot-pilot3-", dir="/tmp"))
    results = []
    for source in sources:
        entries, build = pilot.build_bundle(source, local)
        baseline = pilot.Reader(source["job"]["source_key"], entries, False)
        start = time.perf_counter()
        with pilot.MetadataProfile() as a_profile:
            a = pilot.replay(worker, source, baseline)
        a_wall = time.perf_counter() - start
        snapshot = SourceSnapshotReader(source["job"]["source_key"], entries)
        start = time.perf_counter()
        with pilot.MetadataProfile() as c_profile:
            with snapshot:
                stage_start = time.perf_counter()
                optimized = pilot.replay(worker, source, snapshot)
                stage_wall = time.perf_counter() - stage_start
        c_wall = time.perf_counter() - start
        pilot.need(snapshot.committed and all(a[k] == optimized[k] for k in
                   ("request", "filtered", "final", "generation", "receipt")), "Snapshot replay differs")
        results.append({"source_key": source["job"]["source_key"], "bundle_build": build,
                        "baseline_seconds": a_wall, "snapshot_total_seconds": c_wall,
                        "snapshot_stage_seconds": stage_wall, "boundary": snapshot.boundary_stats,
                        "baseline_reads": baseline.stats, "snapshot_reads": snapshot.stats,
                        "baseline_metadata": a_profile.stats, "snapshot_metadata": c_profile.stats,
                        "request_sha256": a["canonical_request_sha256"], "request_and_validators_equal": True})
    pilot.need(budgets == {p: pilot.digest(pilot.stable_bytes(p)[0]) for p in budgets}, "Original budgets changed")
    pilot.need(original_globals == (Path.open, Path.read_bytes, Path.resolve), "Global filesystem methods changed")
    for path, sha in code_refs.items():
        pilot.need(pilot.digest(pilot.stable_bytes(path)[0]) == sha, "Execution changed")
    report = {"schema": "source_snapshot_reader_pilot3_v1", "source_count": 3, "sources": results,
              "code_refs": code_refs, "new_model_calls": 0, "new_budget_claims": 0,
              "original_budget_unchanged": True, "whole_worker_covered": False,
              "production_deployed": False, "scope": "request construction and original response validation",
              "snapshot_contract": "Read immutable verified bytes; reject changed original/local files before commit",
              "cold_cache_claimed": False, "metadata_profile_times_are_nested": True}
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"report": str(report_path), "sources": results}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    run(parser.parse_args().report)
