import contextlib
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import takeover_n6_monitor_v1 as monitor


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archives = self.root / "archives"
        self.archives.mkdir()
        self.worker = self.root / "workers/n6g23"
        (self.worker / "terminals").mkdir(parents=True)
        self.config = dict(node="N6", pool_worker_id="n6g23", draft_out=str(self.archives),
                           plan=dict(sha256="a" * 64))
        self.now = 1000.0
        self.subject = monitor.Monitor(self.config, worker_root=self.worker, clock=lambda: self.now)

    def progress(self, keys):
        value = dict(completed_source_keys=keys, counts={"draft_generated": len(keys)},
                     processed=len(keys), reserved=len(keys) + 3, UTC="2026-10-26T00:00:00Z",
                     worker_id="n6g23", pool_sha256="a" * 64, global_complete=True)
        (self.archives / "progress.json").write_bytes(monitor.encoded(value))

    def terminal_path(self, sid, split="train"):
        return self.worker / "terminals" / (monitor.digest(["a" * 64, sid, split]) + ".json")

    def terminal(self, sid, status="draft_generated", split="train", **overrides):
        value = dict(id=sid, split=split, status=status, pool_sha256="a" * 64,
                     owner=dict(worker_id="n6g23", node="N6"), **monitor.FLAGS)
        value.update(overrides)
        self.terminal_path(sid, split).write_bytes(monitor.encoded(value))

    def baseline(self):
        self.progress([])
        self.subject.sample()

    def test_baseline_excludes_history_and_reads_no_terminal(self):
        self.progress([["old", "train"]])
        original = monitor.read_json
        with mock.patch.object(monitor, "read_json", wraps=original) as reader:
            sample = self.subject.sample()
            self.subject.sample()
        self.assertEqual(sample["baseline_source_count"], 1)
        self.assertEqual(sample["observed_since_start"]["success_generated_or_partial"], 0)
        self.assertEqual(reader.call_count, 2)
        self.assertTrue(all(call.args[0] == self.archives / "progress.json" for call in reader.call_args_list))

    def test_new_sources_exact_digest_success_and_no_double_count(self):
        self.baseline()
        self.progress([["new", "train"]])
        self.terminal("new")
        self.now += 60
        original = monitor.read_json
        with mock.patch.object(monitor, "read_json", wraps=original) as reader:
            sample = self.subject.sample()
        self.assertEqual(reader.call_args_list[1].args[0], self.terminal_path("new"))
        self.assertEqual(sample["observed_since_start"]["draft_generated"], 1)
        self.assertEqual(sample["new_since_previous_observation"]["sources"][0]["id"], "new")
        self.assertEqual(self.subject.sample()["observed_since_start"]["draft_generated"], 1)
        self.assertEqual(self.subject.sample()["new_since_previous_observation"]["sources"], [])

    def test_partial_regeneration_alarm_and_monotonic_rolling(self):
        self.baseline()
        self.progress([["partial", "train"], ["failed", "train"]])
        self.terminal("partial", "draft_partial")
        self.terminal("failed", "draft_needs_regeneration")
        self.now += 60
        sample = self.subject.sample()
        self.assertEqual(sample["observed_since_start"]["success_generated_or_partial"], 1)
        self.assertEqual(sample["observed_since_start"]["regeneration"], 1)
        self.now += 600
        self.progress([["partial", "train"], ["failed", "train"], ["failed2", "train"]])
        self.terminal("failed2", "draft_needs_regeneration")
        sample = self.subject.sample()
        self.assertEqual(sample["seconds_without_new_success"], 600)
        self.assertTrue(sample["alarms"])
        self.now += 301
        sample = self.subject.sample()
        self.assertEqual(sample["rolling_15min"]["success_generated_or_partial"], 0)
        self.assertEqual(sample["rolling_15min"]["regeneration"], 1)

    def test_pending_missing_bad_json_nul_and_owner_flags_retry(self):
        self.baseline()
        self.progress([["retry", "train"]])
        self.assertEqual(self.subject.sample()["pending_count"], 1)
        for raw in (b"{", b"{\x00}"):
            self.terminal_path("retry").write_bytes(raw)
            self.assertEqual(self.subject.sample()["pending_count"], 1)
            self.assertNotIn(("retry", "train"), self.subject.seen)
        for overrides in (dict(owner=dict(worker_id="n6g23", node="N4")),
                          dict(training_eligible=True), dict(needs_visual_revision=False),
                          dict(id="wrong"), dict(split="val")):
            self.terminal("retry", **overrides)
            self.assertEqual(self.subject.sample()["pending_count"], 1)
        self.terminal("retry")
        sample = self.subject.sample()
        self.assertEqual(sample["pending_count"], 0)
        self.assertEqual(sample["observed_since_start"]["draft_generated"], 1)
        (self.archives / "progress.json").write_bytes(b"{\x00}")
        self.assertEqual(self.subject.sample()["state"], "progress_pending")
        self.progress([["retry", "train"]])
        self.assertEqual(self.subject.sample()["state"], "observing")

    def test_progress_bounded_pending_start_and_no_global_scope(self):
        sample = self.subject.sample()
        self.assertEqual(sample["state"], "pending_start")
        self.now += 700
        self.assertEqual(self.subject.sample()["alarms"], [])
        (self.archives / "progress.json").write_bytes(b" " * (monitor.MAX_PROGRESS_BYTES + 1))
        sample = self.subject.sample()
        self.assertEqual(sample["state"], "pending_start")
        self.assertIn("1024 KiB", sample["progress_error"])
        self.progress([["history", "train"]])
        sample = self.subject.sample()
        self.assertEqual(sample["baseline_source_count"], 1)
        self.assertEqual(sample["scope"], "single_lane_observed_since_start")
        text = json.dumps(sample)
        for forbidden in ("global_complete", "global_terminal", "quality_approved", "resource_assessment"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(set(sample["source_progress"]), {"counts", "processed", "reserved", "UTC"})

    def test_delayed_baseline_starts_rate_after_capture(self):
        self.assertIsNone(self.subject.sample()["rolling_15min"]["success_per_minute"])
        self.now += 700
        self.progress([["history", "train"]])
        sample = self.subject.sample()
        self.assertEqual(sample["rolling_15min"]["observed_seconds"], 0)
        self.assertIsNone(sample["rolling_15min"]["success_per_minute"])
        self.progress([["history", "train"], ["new", "train"]])
        self.terminal("new")
        self.now += 60
        sample = self.subject.sample()
        self.assertEqual(sample["rolling_15min"]["observed_seconds"], 60)
        self.assertEqual(sample["rolling_15min"]["success_per_minute"], 1)
        self.assertFalse(sample["rolling_15min"]["full_window"])

    def test_small_raw_reads_and_separate_progress_limit(self):
        path = self.root / "large.json"
        path.write_bytes(monitor.encoded(dict(padding="x" * 300000)))
        with self.assertRaisesRegex(ValueError, "256 KiB"):
            monitor.read_json(path)
        import os
        sizes = []
        original = os.fdopen
        class Reader:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def fileno(self):
                return self.stream.fileno()
            def read(self, size):
                sizes.append(size)
                return self.stream.read(size)
        def raw_open(fd, mode, buffering):
            self.assertEqual(buffering, 0)
            return Reader(original(fd, mode, buffering=buffering))
        with mock.patch.object(monitor.os, "fdopen", side_effect=raw_open):
            value = monitor.read_json(path, monitor.MAX_PROGRESS_BYTES)
        self.assertEqual(len(value["padding"]), 300000)
        self.assertTrue(sizes and max(sizes) <= 4096)

    def test_cli_once_outputs_stop_and_local_lock(self):
        config_path = self.root / "config.json"
        config_path.write_bytes(monitor.encoded(self.config))
        out = self.root / "monitor"
        lock_path = self.root / "local.lock"
        args = ["--config", str(config_path), "--out", str(out), "--once"]
        with mock.patch.object(monitor, "local_lock_path", return_value=lock_path):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(monitor.main(args), 0)
            self.assertIn("pending_start", stdout.getvalue())
            latest = out / "summary_latest.json"
            log = out / "observations.jsonl"
            self.assertEqual(len(log.read_text().splitlines()), 1)
            self.assertEqual(json.loads(latest.read_text())["state"], "pending_start")
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    self.assertEqual(monitor.main(args), 2)
                self.assertIn("already_running", stderr.getvalue())
            (out / "STOP").touch()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(monitor.main(args), 0)
            self.assertEqual(len(log.read_text().splitlines()), 1)
            self.assertEqual(monitor.local_lock_path(out), lock_path)
        self.assertEqual(monitor.local_lock_path(out).parent, Path(tempfile.gettempdir()))


if __name__ == "__main__":
    unittest.main()
