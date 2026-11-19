import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ms_external_incremental_patrol_v2 as patrol


class PatrolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.ticket = {
            "schema": "ms_external_unified_assignment_v1",
            "worker_id": "a100-extra-01", "edition": "current_explicit",
            "assignment_sha256": "a" * 64, "allowed_gpu_indices": [0, 1, 2, 3],
            "source_count": 40000, "publication_sources_per_batch": 20,
            "gpu_partition": "assignment row index modulo 4 equals physical gpu index",
            "no_VLM": True, "semantic_approval": False,
            "quality_approved": False, "training_eligible": False,
            "output_root": str(self.root / "current_results"),
        }
        self.ticket_path = self.root / "current.ticket.json"
        self.queue_path = self.root / "node.queue.json"
        self.write_ticket()
        self.spec = {"queue_path": str(self.queue_path), "current_ticket_path": str(self.ticket_path), "layout": "flat"}
        self.batches = pathlib.Path(self.ticket["output_root"]) / "batches"
        self.batches.mkdir(parents=True)

    def write_ticket(self):
        raw = json.dumps(self.ticket).encode()
        self.ticket_path.write_bytes(raw)
        self.queue_path.write_text(json.dumps({"schema": "ms_external_authorized_rollover_queue_v1", "worker_id": self.ticket["worker_id"], "tickets": [{"path": str(self.ticket_path), "sha256": hashlib.sha256(raw).hexdigest()}, {"path": str(self.root / "not_current_rollover.ticket.json"), "sha256": "f" * 64}]}))

    def monitor(self, interval=900):
        return patrol.Patrol([self.spec], self.root / "new_monitor_output", interval)

    def batch_path(self, number, gpu=0):
        if self.spec["layout"] == "flat":
            return self.batches / ("gpu%d-run-b%04d" % (gpu, number))
        return self.batches / ("gpu%d" % gpu) / ("g%d-1793039558694280242-1f96bdfb-batch%03d" % (gpu, number))

    def batch(self, number, gpu=0, count=1, failure=False, mutate=None):
        directory = self.batch_path(number, gpu)
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "batch.tar"
        archive.write_bytes(b"opaque-not-a-tar")
        records = []
        for i in range(count):
            index = (number * 20 + i) * 4 + gpu
            identity = "source_%d" % index
            source = {"split": "train", "id": identity, "video_path": str(self.root / "videos" / (identity + ".mp4")), "audio_path": str(self.root / "audio" / (identity + ".wav"))}
            r = {"source_row": source, "status": patrol.SUCCESS, "vlm_called": False, "semantic_approval": False, "quality_approved": False, "training_eligible": False}
            if self.spec["layout"] == "flat":
                source["assignment_index"] = index
            else:
                r["assignment_index"] = index
            if failure and i == count - 1:
                r["status"] = patrol.FAILURE
                del r["vlm_called"]
                r["reason"] = "synthetic_decode_failure_sealed_no_retry"
            records.append(r)
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        declaration = {"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha}
        c = {"schema": "ms_unified_novlm_batch_commit_v1", "worker": self.ticket["worker_id"], "edition": self.ticket["edition"], "gpu": gpu, "assignment_sha256": self.ticket["assignment_sha256"], "batch_sha256": sha, "semantic_approval": False, "quality_approved": False, "training_eligible": False, "receipts": records}
        if self.spec["layout"] == "flat":
            c.update(package=declaration, owner_pid=1234, commit_unix=1700000000 + number)
        else:
            c.update(assignment_indices=[r["assignment_index"] for r in records], publications=[{"path": str(archive), "sha256": sha, "receipt": declaration}], run_id=directory.name.rsplit("-batch", 1)[0])
        if mutate:
            mutate(c)
        (directory / "commit.json").write_text(json.dumps(c))
        return directory / "commit.json"

    def events(self):
        path = self.root / "new_monitor_output" / "commit_events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_flat_1500_entries_one_scan_for_four_gpus_and_baseline_not_new(self):
        for number in range(375):
            for gpu in range(4):
                self.batch(number, gpu)
        monitor = self.monitor()
        with mock.patch.object(patrol.os, "scandir", wraps=patrol.os.scandir) as scan:
            first = monitor.sample()
            self.assertEqual(scan.call_count, 1)
        self.assertFalse(first["measurement_incomplete"])
        self.assertEqual(first["io_counters"]["directory_entries"], 1500)
        self.assertEqual(first["io_counters"]["commit_metadata_reads"], 8)
        self.assertEqual(first["new_successful_unique_sources_lower_bound"], 0)
        self.assertTrue(all(e["baseline_not_new_production"] for e in self.events()))
        for gpu in range(4):
            self.batch(375, gpu)
        second = monitor.sample()
        self.assertEqual(second["new_successful_unique_sources_lower_bound"], 4)
        self.assertEqual(second["io_counters"]["commit_metadata_reads"], 4)
        third = monitor.sample()
        self.assertEqual(third["io_counters"]["commit_metadata_reads"], 0)
        self.assertEqual(third["new_successful_unique_sources_lower_bound"], 0)

    def test_current_clusters80k_partition_literal_matches_actual_ticket(self):
        self.ticket["gpu_partition"] = "zero-based assignment row index modulo4 equals GPU index; do not reuse source_ordinal modulo"
        self.write_ticket()
        self.batch(0)
        first = self.monitor().sample()
        self.assertFalse(first["measurement_incomplete"])
        self.assertEqual(first["new_successful_unique_sources_lower_bound"], 0)

    def test_nested_layout_no_owner_or_commit_time_required(self):
        self.spec["layout"] = "nested"
        for gpu in range(4):
            (self.batches / ("gpu%d" % gpu)).mkdir()
            self.batch(0, gpu)
        monitor = self.monitor()
        first = monitor.sample()
        self.assertFalse(first["measurement_incomplete"])
        self.assertEqual(first["io_counters"]["directory_enumerations"], 4)
        self.batch(1)
        second = monitor.sample()
        self.assertEqual(second["new_successful_unique_sources_lower_bound"], 1)
        event = self.events()[-1]
        self.assertNotIn("owner_pid", event)
        self.assertNotIn("commit_unix", event)
        self.assertEqual(event["records"][0]["assignment_index"], 80)

    def test_missing_commit_retry_and_newborn_commit(self):
        self.batch_path(0).mkdir()
        monitor = self.monitor()
        first = monitor.sample()
        self.assertEqual(first["io_counters"]["commit_metadata_reads"], 0)
        second = monitor.sample()
        self.assertEqual(second["new_successful_unique_sources_lower_bound"], 0)
        self.batch(0)
        self.batch_path(1).mkdir()
        third = monitor.sample()
        self.assertEqual(third["new_successful_unique_sources_lower_bound"], 1)
        self.batch(1)
        fourth = monitor.sample()
        self.assertEqual(fourth["new_successful_unique_sources_lower_bound"], 1)
        self.assertEqual(monitor.sample()["new_successful_unique_sources_lower_bound"], 0)

    def test_mutable_commit_is_rejected_without_second_metadata_read(self):
        p = self.batch(0)
        monitor = self.monitor()
        monitor.sample()
        c = json.loads(p.read_text())
        c["owner_pid"] = 99999999
        p.write_text(json.dumps(c))
        second = monitor.sample()
        self.assertTrue(second["measurement_incomplete"])
        self.assertIsNone(second["observed_success_rate_per_hour"])
        self.assertEqual(second["io_counters"]["commit_metadata_reads"], 0)
        self.assertTrue(any("immutable_commit_changed" in e["error"] for e in second["errors"]))

    def test_19_strict_successes_survive_failure_missing_vlm_flag(self):
        monitor = self.monitor()
        monitor.sample()
        self.batch(0, count=20, failure=True)
        sample = monitor.sample()
        self.assertEqual(sample["new_successful_unique_sources_lower_bound"], 19)
        self.assertEqual(sample["new_failed_unique_receipts_lower_bound"], 1)
        self.assertTrue(sample["measurement_incomplete"])
        self.assertIsNone(sample["observed_success_rate_per_hour"])
        failed = self.events()[-1]["records"][-1]
        self.assertNotIn("vlm_called", failed)
        self.assertFalse(failed["no_vlm_proven"])
        self.assertIn("no_vlm_not_proven", failed["evidence_gaps"])
        self.assertIn("reason", failed["failure_evidence"])

    def test_explicit_true_approval_rejects_entire_batch(self):
        monitor = self.monitor()
        monitor.sample()
        self.batch(0, count=20, failure=True, mutate=lambda c: c["receipts"][-1].update(quality_approved=True))
        sample = monitor.sample()
        self.assertEqual(sample["new_successful_unique_sources_lower_bound"], 0)
        self.assertEqual(sample["new_failed_unique_receipts_lower_bound"], 0)
        self.assertTrue(sample["measurement_incomplete"])

    def test_assignment_and_source_path_mismatch_failclosed(self):
        monitor = self.monitor()
        monitor.sample()
        self.batch(0, mutate=lambda c: c["receipts"][0]["source_row"].update(assignment_index=1))
        self.batch(1, mutate=lambda c: c.update(assignment_sha256="b" * 64))
        self.batch(2, mutate=lambda c: c["receipts"][0]["source_row"].update(audio_path=str(self.root / "different.wav")))
        sample = monitor.sample()
        self.assertEqual(sample["new_successful_unique_sources_lower_bound"], 0)
        self.assertTrue(sample["measurement_incomplete"])
        self.assertEqual(len(sample["errors"]), 3)

    def test_queue_ticket_sha_mismatch_and_no_rollover_guess(self):
        self.ticket["source_count"] = 30000
        self.ticket_path.write_text(json.dumps(self.ticket))
        sample = self.monitor().sample()
        self.assertEqual(sample["io_counters"]["directory_enumerations"], 0)
        self.assertTrue(sample["measurement_incomplete"])
        self.assertIn("ticket_sha_mismatch", sample["errors"][0]["error"])

    def test_missing_node_io_still_updates_heartbeat_and_latest_each_sample(self):
        monitor = self.monitor()
        for index in range(2):
            with mock.patch.object(patrol, "paths_for_node", side_effect=OSError("synthetic_node_io_failure")):
                sample = monitor.sample()
            heartbeat = json.loads((monitor.output / "heartbeat.json").read_text())
            latest = json.loads((monitor.output / "latest_summary.json").read_text())
            self.assertEqual(heartbeat["sample_index"], index)
            self.assertEqual(latest["sample_index"], index)
            self.assertTrue(sample["measurement_incomplete"])
            self.assertIsNone(sample["observed_success_rate_per_hour"])

    def test_wall_gap_and_failed_interval_are_never_subtracted(self):
        monitor = self.monitor()
        with mock.patch.object(patrol.time, "monotonic", return_value=100), mock.patch.object(patrol.time, "time", return_value=1000):
            monitor.sample()
        with mock.patch.object(patrol.time, "monotonic", return_value=1000), mock.patch.object(patrol.time, "time", return_value=1900), mock.patch.object(patrol, "paths_for_node", side_effect=OSError("node_down")):
            monitor.sample()
        self.batch(0)
        with mock.patch.object(patrol.time, "monotonic", return_value=2800), mock.patch.object(patrol.time, "time", return_value=3700):
            last = monitor.sample()
        self.assertEqual(last["total_observation_monotonic_seconds"], 2700)
        self.assertEqual(last["window_wall_seconds"], 1800)
        self.assertEqual(last["new_successful_unique_sources_lower_bound"], 1)
        self.assertIsNone(last["observed_success_rate_per_hour"])
        self.assertTrue(any(e["error"] == "wall_clock_or_sampling_event_gap" for e in last["errors"]))

    def test_metadata_budget_12_and_deferred_commits_are_not_lost(self):
        monitor = self.monitor()
        monitor.sample()
        for number in range(15):
            self.batch(number)
        second = monitor.sample()
        self.assertEqual(second["io_counters"]["commit_metadata_reads"], 12)
        self.assertEqual(second["new_successful_unique_sources_lower_bound"], 12)
        self.assertTrue(second["measurement_incomplete"])
        third = monitor.sample()
        self.assertEqual(third["new_successful_unique_sources_lower_bound"], 3)
        self.assertEqual(third["io_counters"]["commit_metadata_reads"], 3)

    def test_metadata_open_scope_excludes_archive_media_assignments_and_claims(self):
        commit = self.batch(0)
        monitor = self.monitor()
        opened = []
        real_open = patrol.os.open

        def tracked_open(path, *args, **kwargs):
            opened.append(str(path))
            return real_open(path, *args, **kwargs)

        with mock.patch.object(patrol.os, "open", side_effect=tracked_open):
            first = monitor.sample()
            second = monitor.sample()
        self.assertFalse(first["measurement_incomplete"])
        self.assertFalse(second["measurement_incomplete"])
        self.assertEqual(set(opened), {str(self.queue_path), str(self.ticket_path), str(commit)})
        self.assertEqual(opened.count(str(commit)), 1)
        self.assertFalse(self.events()[0]["archive_payload_read"])

    def test_archive_size_and_current_batch_path_binding_failclosed(self):
        monitor = self.monitor()
        monitor.sample()
        self.batch(0, mutate=lambda c: c["package"].update(bytes=9999))
        self.batch(1, mutate=lambda c: c["package"].update(path=str(self.root / "old_edition" / "batch.tar")))
        sample = monitor.sample()
        self.assertEqual(sample["new_successful_unique_sources_lower_bound"], 0)
        self.assertTrue(sample["measurement_incomplete"])
        self.assertEqual(len(sample["errors"]), 2)

    def test_nested_assignment_indices_mismatch_failclosed(self):
        self.spec["layout"] = "nested"
        for gpu in range(4):
            (self.batches / ("gpu%d" % gpu)).mkdir()
        monitor = self.monitor()
        monitor.sample()
        self.batch(0, mutate=lambda c: c.update(assignment_indices=[4]))
        sample = monitor.sample()
        self.assertEqual(sample["new_successful_unique_sources_lower_bound"], 0)
        self.assertTrue(sample["measurement_incomplete"])
        self.assertIn("commit_assignment_indices_mismatch", sample["errors"][0]["error"])

    def test_source_unique_key_not_receipt_or_batch_count(self):
        monitor = self.monitor()
        monitor.sample()
        self.batch(0)
        monitor.sample()

        def repeat_source(c):
            source = c["receipts"][0]["source_row"]
            source.update(id="source_0", video_path=str(self.root / "videos/source_0.mp4"), audio_path=str(self.root / "audio/source_0.wav"))

        self.batch(1, mutate=repeat_source)
        second = monitor.sample()
        self.assertEqual(second["new_successful_unique_sources_lower_bound"], 0)
        self.assertEqual(second["cumulative_successful_unique_sources_lower_bound"], 1)
        self.assertFalse(second["measurement_incomplete"])

    def test_directory_enumeration_bound_and_pending_bound(self):
        with mock.patch.object(patrol, "MAX_ENTRIES", 2):
            for number in range(3):
                self.batch(number)
            sample = self.monitor().sample()
        self.assertTrue(sample["measurement_incomplete"])
        self.assertIn("node_directory_entry_bound_exceeded", sample["errors"][0]["error"])

    def test_pending_overflow_explicitly_marks_event_gap(self):
        monitor = self.monitor()
        monitor.sample()
        with mock.patch.object(patrol, "MAX_PENDING", 2):
            for number in range(3):
                self.batch_path(number).mkdir()
            sample = monitor.sample()
        self.assertTrue(sample["measurement_incomplete"])
        self.assertTrue(any(e["error"] == "pending_overflow_event_gap" for e in sample["errors"]))
        self.assertLessEqual(sample["lanes"][0]["pending_commits"], 2)

    def test_metadata_over_1mib_is_rejected(self):
        self.ticket_path.write_bytes(b" " * (patrol.MAX_METADATA + 1))
        sample = self.monitor().sample()
        self.assertTrue(sample["measurement_incomplete"])
        self.assertIn("over_1MiB", sample["errors"][0]["error"])

    def test_once_cli_on_cpu_fixture_and_existing_output_is_rejected(self):
        self.batch(0)
        config = self.root / "config.json"
        config.write_text(json.dumps({"nodes": [self.spec]}))
        output = self.root / "cli_output"
        args = [sys.executable, patrol.__file__, "--config", str(config), "--output-dir", str(output), "--once", "--interval-seconds", ".01"]
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["new_successful_unique_sources_lower_bound"], 0)
        self.assertEqual(summary["monitor_GPU_calls"], 0)
        repeated = subprocess.run(args, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("FileExistsError", repeated.stderr)

    def test_main_interval_uses_real_deadlines(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"nodes": [self.spec]}))
        args = ["monitor", "--config", str(config), "--output-dir", str(self.root / "timed_output"), "--interval-seconds", "900"]
        with mock.patch.object(sys, "argv", args), mock.patch.object(patrol.Patrol, "sample", return_value={}), mock.patch.object(patrol.time, "monotonic", return_value=100), mock.patch.object(patrol.time, "sleep", side_effect=RuntimeError("stop_after_first_sleep")) as sleep, mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "stop_after_first_sleep"):
                patrol.main()
        sleep.assert_called_once_with(900)


if __name__ == "__main__":
    unittest.main()
