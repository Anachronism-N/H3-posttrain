"""CPU-only monitor tests; collectors and SSH are mocked, no production files."""
import copy
import fcntl
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock
from contextlib import redirect_stdout

import ms_novlm_jobs_monitor_v1 as monitor


def fixture(phase="test_phase", sid="source1", published=False):
    job = {"node": "N3", "host": "28.216.19.213", "phase": phase,
           "config_path": "/mock/" + phase + ".json", "task_handle": "mock-task"}
    row = {"split": "train", "id": sid, "video_path": "/mock/" + sid + ".mp4",
           "audio_path": "/mock/" + sid + ".wav"}
    receipt = {"source_row": row, "status": "extracted_rule_draft_unapproved", **monitor.UNREVIEWED}
    cfg = {"node": "N3", "phase": phase, "config_path": job["config_path"],
           "gpus": [0], "targetcount": 1, "selected_source_ids": [sid],
           "out": "/mock/output/" + phase, "queue": "/mock/queue/" + phase}
    result = {"job": job, "reachable": True, "warnings": [], "config": cfg,
              "ticket": {"sources": [row], "ticket_id": "ticket"},
              "worker_process": {"pid": 42, "start_ticks": 7, "alive": True, "identity_verified": True},
              "manager_process": {"alive": True, "wchan": "do_wait"},
              "local_receipts": [], "commit_path": cfg["out"] + "/" + phase + "/batches/ticket/commit.json",
              "commit_sha256": "mock-sha"}
    if published:
        result["commit"] = {"schema": "ms_unified_novlm_batch_commit_v2", "edition": phase,
            "ticket_id": "ticket", "unique_sources_attempted": 1, "receipts": [receipt],
            "batch_sha256": "a"*64, "archive_sha256": "a"*64, "archive_bytes": 10240,
            "archive_path": str(Path(result["commit_path"]).parent / "batch.tar"),
            "prefetch_drained": True, "vlm_called": False, "quality_approved": False,
            "beat_alignment_claimed": False, "production_complete_claimed": False,
            "profile_seconds": {"models_load_once": 1, "pack_publish": 2, "whole_e2e": 3}}
        result["summary"] = {"schema": "ms_unified_native_summary_v1", "selected_source_ids": [sid],
            "selected": 1, "processed": 1, "source_outcomes": [receipt], "prefetch_drained": True,
            "human_review_completed": False, "production_complete_claimed": False,
            "commit_ref": {"path": result["commit_path"], "sha256": "mock-sha"},
            "runtime_gate_profile": {"calls": 1, "seconds": 0.1}}
        result["worker_exit"] = {"phase": phase, "exit_code": 0}
        result["resources_safe"] = {"phase": phase, **{key: True for key in monitor.SAFE_FLAGS}}
        result["worker_process"]["alive"] = False
    return result


class MonitorTests(unittest.TestCase):
    def test_existing_commit_is_baseline_and_global_split_id_dedup(self):
        state = monitor.Monitor()
        first = state.observe([fixture(published=True)], now=0, wall=100)
        self.assertEqual(first["baseline_committed_source_count"], 1)
        self.assertEqual(first["new_committed_source_count"], 0)
        second = state.observe([fixture(published=True), fixture("other_phase", published=True)], now=60)
        self.assertEqual(second["committed_source_count"], 1)
        self.assertEqual(second["new_this_cycle"], [])
        self.assertEqual(second["rolling_15min"]["sources_per_minute"], 0)
        other_split = fixture("other_split", published=True)
        other_split["ticket"]["sources"][0]["split"] = "validation"
        third = state.observe([fixture(published=True), other_split], now=120)
        self.assertEqual(third["committed_source_count"], 2)

    def test_local_receipts_are_never_added_to_published_sources(self):
        observation = fixture()
        observation["local_receipts"] = [{"source_row": observation["ticket"]["sources"][0],
            "root": "/mock/ms-unified-work", "age_seconds": 5, "status": "extracted_rule_draft_unapproved"}]
        summary = monitor.Monitor().observe([observation], now=0)
        self.assertEqual(summary["committed_source_count"], 0)
        job = summary["jobs"][0]
        self.assertEqual(job["local_completed_unpublished_count"], 1)
        self.assertEqual(job["latest_local_receipt_age_seconds"], 5)
        self.assertFalse(job["worker_finished"])

    def test_failed_exit_unsafe_or_wrong_sources_never_finished(self):
        good = fixture(published=True)
        self.assertTrue(monitor.summarize_job(good)["worker_finished"])
        bad_exit = copy.deepcopy(good)
        bad_exit["worker_exit"]["exit_code"] = 1
        bad_safe = copy.deepcopy(good)
        bad_safe["resources_safe"][monitor.SAFE_FLAGS[0]] = False
        bad_rows = copy.deepcopy(good)
        bad_rows["commit"]["receipts"][0]["source_row"]["id"] = "unexpected"
        bad_flags = copy.deepcopy(good)
        bad_flags["commit"]["receipts"][0]["vlm_called"] = True
        bad_failure = copy.deepcopy(good)
        bad_failure["failure"] = {"error": "gate failed"}
        for observation in (bad_exit, bad_safe, bad_rows, bad_flags, bad_failure):
            self.assertFalse(monitor.summarize_job(observation)["worker_finished"])
        self.assertTrue(monitor.summarize_job(bad_exit)["reliable_published"])
        self.assertFalse(monitor.summarize_job(bad_rows)["reliable_published"])
        self.assertFalse(monitor.summarize_job(bad_flags)["reliable_published"])

    def test_mock_ssh_one_call_per_host_timeout_warning_not_done(self):
        one, two = fixture(), fixture("second")
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, json.dumps([one, two]), ""))
        results = monitor.collect_host("28.216.19.213", [one["job"], two["job"]], runner=runner)
        self.assertEqual(len(results), 2)
        runner.assert_called_once()
        self.assertEqual(runner.call_args.kwargs["timeout"], 20)
        command = runner.call_args.args[0]
        self.assertEqual(command[0], "ssh")
        self.assertIn("python3 -B -c", command[-1])
        timeout = Mock(side_effect=subprocess.TimeoutExpired("ssh", 20))
        unreachable = monitor.collect_host("28.216.19.213", [one["job"]], runner=timeout)
        result = monitor.Monitor().observe(unreachable, now=0)
        self.assertFalse(result["observation_complete"])
        self.assertEqual(result["jobs"][0]["status"], "unreachable")
        self.assertFalse(result["jobs"][0]["worker_finished"])
        self.assertTrue(result["warnings"])

    def test_once_heartbeat_precedes_collector_and_output_flock(self):
        with tempfile.TemporaryDirectory(prefix="test-novlm-monitor-") as directory:
            root = Path(directory)
            registry = root / "registry.json"
            observation = fixture()
            registry.write_text(json.dumps([observation["job"]]))
            out = root / "out"
            stream = io.StringIO()
            def collector(jobs):
                self.assertIn("heartbeat_before_ssh", stream.getvalue())
                self.assertEqual(jobs, [observation["job"]])
                return [observation]
            with redirect_stdout(stream):
                self.assertEqual(monitor.run(registry, out, once=True, collector=collector), 0)
            lines = (out / "monitor.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), json.loads((out / "latest_summary.json").read_text()))
            with (out / "monitor.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(monitor.run(registry, out, once=True, collector=collector), 2)
            (out / "STOP").touch()
            not_called = Mock(side_effect=AssertionError("STOP must skip SSH"))
            self.assertEqual(monitor.run(registry, out, once=True, collector=not_called), 0)
            not_called.assert_not_called()

    def test_rolling_rate_keeps_actual_monotonic_wall_window_and_lower_bound(self):
        state = monitor.Monitor()
        state.observe([fixture()], now=0, wall=10000)
        published = state.observe([fixture(published=True)], now=60, wall=1)
        self.assertEqual(published["new_committed_source_count"], 1)
        self.assertEqual(published["rolling_15min"]["window_seconds"], 60)
        self.assertFalse(published["rolling_15min"]["measurement_incomplete"])
        self.assertEqual(published["rolling_15min"]["sources_per_minute"], 1)
        unreachable = {"job": fixture()["job"], "reachable": False, "warnings": ["timeout"]}
        state.observe([unreachable], now=120)
        recovered = state.observe([fixture(published=True)], now=180)
        self.assertEqual(recovered["rolling_15min"]["window_seconds"], 180)
        self.assertTrue(recovered["rolling_15min"]["measurement_incomplete"])
        self.assertIsNone(recovered["rolling_15min"]["sources_per_minute"])
        self.assertAlmostEqual(recovered["rolling_15min"]["observed_lower_bound_sources_per_minute"], 1/3)
        after_window = state.observe([fixture(published=True)], now=1200)
        self.assertEqual(after_window["rolling_15min"]["new_sources"], 0)
        self.assertIsNone(after_window["rolling_15min"]["sources_per_minute"])
        self.assertEqual(after_window["rolling_15min"]["window_seconds"], 900)
        self.assertEqual(after_window["rolling_15min"]["observed_lower_bound_sources_per_minute"], 0)

    def test_idle_only_for_alive_expected_running_not_paused_finished_or_parent_wait(self):
        state = monitor.Monitor()
        running = fixture()
        for now in range(0, 601, 60):
            result = state.observe([running], now=now)
        self.assertTrue(result["jobs"][0]["idle_alarm"])
        recent_receipt = copy.deepcopy(running)
        recent_receipt["local_receipts"] = [{"source_row": recent_receipt["ticket"]["sources"][0], "age_seconds": 30}]
        self.assertFalse(state.observe([recent_receipt], now=660)["jobs"][0]["idle_alarm"])
        recent_receipt["local_receipts"][0]["age_seconds"] = 601
        self.assertTrue(state.observe([recent_receipt], now=720)["jobs"][0]["idle_alarm"])
        for key, value in (("known_paused", True), ("worker_process", {"alive": False})):
            altered = copy.deepcopy(running)
            altered[key] = value
            result = state.observe([altered], now=660)
            self.assertFalse(result["jobs"][0]["idle_alarm"])
            self.assertFalse(result["jobs"][0]["expected_running"])
        result = state.observe([fixture(published=True)], now=720)
        self.assertFalse(result["jobs"][0]["idle_alarm"])
        self.assertTrue(result["jobs"][0]["worker_finished"])

    def test_registry_allowlist_and_bad_ssh_json_are_local_warnings(self):
        job = fixture()["job"]
        self.assertEqual(monitor.validate_registry([job]), [job])
        self.assertEqual(monitor.validate_registry({"schema": "ms_novlm_live_jobs_registry_v1", "jobs": [job]}), [job])
        n2 = {**job, "node": "N2", "host": "28.216.17.70", "gpu": 2}
        self.assertEqual(monitor.validate_registry([n2]), [n2])
        with self.assertRaises(ValueError):
            monitor.validate_registry([{**n2, "gpu": 6}])
        for changed in ({**job, "host": "other-host"}, {**job, "config_path": "relative.json"}):
            with self.assertRaises(ValueError):
                monitor.validate_registry([changed])
        malformed = Mock(return_value=subprocess.CompletedProcess([], 0, "not JSON", ""))
        result = monitor.collect_host(job["host"], [job], runner=malformed)
        self.assertFalse(result[0]["reachable"])
        self.assertTrue(result[0]["warnings"])
        compile(monitor.REMOTE, "<readonly-remote>", "exec")
        bad = fixture("bad_phase")
        bad["local_receipts"] = [{"source_row": {"id": "missing-split"}}]
        partial = monitor.Monitor().observe([bad, fixture(published=True)], now=0)
        self.assertEqual(partial["jobs"][0]["status"], "collection_error")
        self.assertTrue(partial["jobs"][1]["worker_finished"])
        self.assertFalse(partial["observation_complete"])


if __name__ == "__main__":
    unittest.main()
