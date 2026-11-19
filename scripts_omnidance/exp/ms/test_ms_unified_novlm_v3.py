"""Synthetic inputs/models only, exercising frozen real v2 CPU NPZ/rules."""
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import ms_unified_novlm_v2 as core
import ms_unified_novlm_v3 as worker
import test_ms_unified_novlm_v2 as fixtures


class ContinuousTest(unittest.TestCase):
    gate = fixtures.UnifiedTest.gate
    actual_probe = fixtures.UnifiedTest.actual_probe
    decoder = fixtures.UnifiedTest.decoder
    pose_result = fixtures.UnifiedTest.pose_result

    def setUp(self):
        fixtures.UnifiedTest.setUp(self)
        for index in range(3, 43):
            sid = f"fixture{index}"
            video, audio = self.root / f"{sid}.mp4", self.root / f"{sid}.wav"
            video.write_bytes(b"synthetic-video-not-decodable")
            audio.write_bytes(b"synthetic-audio-not-decodable")
            self.rows.append({"id": sid, "split": "train", "video_path": str(video), "audio_path": str(audio)})
        self.ticket["schema"] = worker.SCHEMA
        self.heartbeat = io.StringIO()

    def run_worker(self, limit=3, **kwargs):
        reader = kwargs.pop("resource_reader", lambda root: {"cgroup_headroom": 40 * worker.GIB, "local_free": 32 * worker.GIB})
        with redirect_stdout(self.heartbeat):
            return worker.run_ticket(self.ticket, limit=limit, resource_gate=self.gate, model_loader=self.loader,
                                     probe=self.probe, decoder=self.decoder, audio_info=self.audio_info,
                                     resource_reader=reader, **kwargs)

    def edition(self):
        return self.root / "shared" / self.ticket["edition"]

    def read_commit(self, batch):
        path = Path(batch["commit_path"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), batch["commit_sha256"])
        return json.loads(path.read_text())

    def test_three_batches_models_once_real_v2_npz_bgr97_receipts_progress(self):
        with patch.object(core.v2, "validate_observations", wraps=core.v2.validate_observations) as validate:
            summary = self.run_worker(limit=43)
        self.assertEqual(summary["schema"], worker.SUMMARY_SCHEMA)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["source_count"], 43)
        self.assertEqual(summary["processed"], 43)
        self.assertEqual(summary["selected_source_ids"], [row["id"] for row in self.rows])
        self.assertEqual([row["id"] for row in summary["all_outcomes"]], summary["selected_source_ids"])
        self.assertEqual(summary["model_init_count"], 1)
        self.assertEqual(self.loader.call_count, 1)
        self.assertEqual(self.pose.call_count, 43 * 97)
        self.assertEqual(validate.call_count, 43)
        self.assertEqual(summary["total_frames"], 43 * 97)
        self.assertEqual(summary["counts"]["generated"], 43)
        self.assertFalse(summary["quality_approved"])
        self.assertFalse(summary["vlm_called"])
        self.assertFalse(summary["production_complete_claimed"])
        self.assertTrue(summary["allprefetch_drained"])
        self.assertEqual([batch["source_count"] for batch in summary["completed_batches"]], [20, 20, 3])
        archives = []
        for batch in summary["completed_batches"]:
            commit = self.read_commit(batch)
            self.assertEqual(commit["schema"], "ms_unified_novlm_batch_commit_v3")
            self.assertEqual(commit["parent_ticket_id"], self.ticket["ticket_id"])
            self.assertEqual(commit["unique_sources_attempted"], batch["source_count"])
            self.assertTrue(commit["prefetch_drained"])
            self.assertTrue(Path(batch["commit_path"]).parent.name.endswith(f"-b{batch['batch_index']:06d}"))
            self.assertEqual({path.name for path in Path(batch["commit_path"]).parent.iterdir()}, {"batch.tar", "commit.json"})
            archives.append(commit["archive_path"])
            with tarfile.open(commit["archive_path"]) as archive:
                self.assertEqual(len(archive.getnames()), 5 * batch["source_count"])
                for receipt in commit["receipts"]:
                    sid = receipt["source_row"]["id"]
                    entry = commit["archive_index"][sid]
                    sealed = json.load(archive.extractfile(entry["receipt_member"]))
                    self.assertEqual(sealed, receipt)
                    self.assertEqual(sealed["locator"]["archive_path"], commit["archive_path"])
                    manifest = json.load(archive.extractfile(entry["members"]["fullframe_manifest.json"]))
                    raw = archive.extractfile(entry["members"]["pose.npz"]).read()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["npz_sha256"])
                    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
                        data = {key: saved[key] for key in saved.files}
                    np.testing.assert_array_equal(data["frame_indices"], np.arange(97))
                    np.testing.assert_array_equal(data["source_timestamps"], (np.arange(97) + 8) / 16)
                    self.assertTrue(core.v2.validate_observations(manifest, manifest, data).all())
                    self.assertFalse(Path(manifest["npz_path"]).exists())
        self.assertEqual(len(set(archives)), 3)
        progress = json.loads(Path(summary["progress_path"]).read_text())
        self.assertEqual(progress["schema"], worker.PROGRESS_SCHEMA)
        self.assertEqual(progress["processed"], 43)
        self.assertEqual(progress["completed_batch_count"], 3)
        self.assertEqual(progress["latest_completed_batch"], summary["completed_batches"][-1])
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(len(self.heartbeat.getvalue().splitlines()), 3)
        self.assertFalse(list((self.root / "local").glob("ms-unified-v3-*")))
        self.assertEqual(set(summary["profile_seconds"]), {"models_load_once", "copy", "probe", "decode_pose", "rules", "beat", "write", "pack_publish", "whole_e2e"})

    def test_stop_after_one_batch_no_claim_for_leftovers(self):
        stop = self.root / "STOP"
        self.ticket["stop_file"] = str(stop)
        original = worker.publish_progress
        def request_stop(path, value):
            original(path, value)
            if value["processed"] == 20:
                stop.write_bytes(b"stop-after-current-batch")
        with patch.object(worker, "publish_progress", side_effect=request_stop):
            summary = self.run_worker(limit=43)
        self.assertEqual(summary["status"], "drained_not_complete")
        self.assertEqual(summary["uncommitted_prefetch"], 0)
        self.assertEqual(json.loads(Path(summary["progress_path"]).read_text())["uncommitted_prefetch"], 0)
        self.assertEqual(summary["source_count"], 43)
        self.assertEqual(summary["processed"], 20)
        self.assertEqual(summary["unprocessed_source_ids"], [row["id"] for row in self.rows[20:]])
        self.assertEqual(len(summary["completed_batches"]), 1)
        self.assertTrue(stop.exists())
        for row in self.rows[20:]:
            self.assertFalse((self.edition() / "claims" / "sources" / (row["id"] + ".json")).exists())
        self.assertEqual(json.loads(Path(summary["progress_path"]).read_text())["status"], "drained_not_complete")
        self.assertTrue(summary["allprefetch_drained"])

    def test_stop_symlink_rejected_before_gate_and_no_output(self):
        target = self.root / "stop-target"
        target.write_bytes(b"stop")
        link = self.root / "STOP"
        link.symlink_to(target)
        self.ticket["stop_file"] = str(link)
        gate = Mock(side_effect=AssertionError("must not enter gate"))
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.run_ticket(self.ticket, resource_gate=gate)
        gate.assert_not_called()
        self.assertFalse((self.root / "shared").exists())

    def test_failed_copy_sealed_alone_other_source_survives(self):
        self.rows[0]["video_path"] = str(self.root / "absent" / "fixture0.mp4")
        summary = self.run_worker(limit=2)
        self.assertEqual(summary["counts"]["failed_sealed_no_retry"], 1)
        self.assertEqual(summary["counts"]["generated"], 1)
        self.assertFalse(summary["all_outcomes"][0]["input_copy_completed"])
        commit = self.read_commit(summary["completed_batches"][0])
        with tarfile.open(commit["archive_path"]) as archive:
            self.assertEqual(len(archive.getnames()), 6)
        self.assertEqual(commit["unique_sources_attempted"], 2)
        self.assertEqual(summary["status"], "completed")
        self.assertFalse(summary["quality_approved"])

    def test_source_claim_global_edition_cannot_reassign_cross_split(self):
        first = self.run_worker(limit=2)
        self.ticket["ticket_id"] = "batch2"
        self.ticket["sources"] = [{**row, "split": "val"} for row in self.rows[:2]]
        second = self.run_worker(limit=2)
        self.assertEqual(second["counts"]["existing_claim_not_reassigned"], 2)
        self.assertEqual(second["counts"]["generated"], 0)
        self.assertEqual(self.probe.call_count, 2)
        commit = self.read_commit(second["completed_batches"][0])
        self.assertEqual(commit["unique_sources_attempted"], 0)
        with tarfile.open(commit["archive_path"]) as archive:
            self.assertEqual(len(archive.getnames()), 2)
        self.assertNotEqual(first["completed_batches"][0]["commit_path"], second["completed_batches"][0]["commit_path"])

    def test_ticket_claim_once_no_second_model_init(self):
        self.run_worker(limit=1)
        with self.assertRaises(FileExistsError):
            self.run_worker(limit=1)
        self.assertEqual(self.loader.call_count, 1)

    def test_default_deny_zero_writes_models(self):
        with self.assertRaises(PermissionError):
            worker.run_ticket(self.ticket, model_loader=self.loader)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "shared").exists())

    def test_low_headroom_before_claim_model(self):
        with self.assertRaises(PermissionError):
            self.run_worker(resource_reader=lambda root: {"cgroup_headroom": 31 * worker.GIB, "local_free": 32 * worker.GIB})
        self.loader.assert_not_called()
        self.assertFalse((self.root / "shared").exists())

    def test_duplicate_splitmix_rejected(self):
        self.ticket["sources"] = self.rows[:-1] + [{**self.rows[0], "split": "val"}]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_worker(limit=10000)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "shared").exists())

    def test_limit_batch_validation_and_finite_ten_thousand(self):
        for limit in (0, 10001, True):
            with self.assertRaises(ValueError):
                worker.check_ticket(self.ticket, limit)
        for size in (0, 101, True):
            self.ticket["batch_sources"] = size
            with self.assertRaises(ValueError):
                worker.check_ticket(self.ticket)
        self.ticket["batch_sources"] = 100
        self.assertEqual(len(worker.check_ticket(self.ticket, 10000)[0]), 43)
        self.assertEqual(len(worker.check_ticket(self.ticket)[0]), 3)

    def test_stage_realtime_source_batch_and_context_lifetime_prefetch_bound(self):
        self.ticket["batch_sources"] = 2
        active = [False]
        stages = []
        @contextmanager
        def lease(ticket):
            active[0] = True
            with self.gate(ticket) as evidence:
                yield evidence
            active[0] = False
        def stage(before_load):
            self.assertTrue(active[0])
            stages.append(before_load)
        def load(ticket):
            stage(True)
            return (self.pose, self.beats)
        max_inputs, max_cases = [], []
        copy = worker.copy_source
        extract = worker.extract_and_rule
        def copy_bound(row, root, budget, profile):
            value = copy(row, root, budget, profile)
            max_inputs.append(len(list(root.glob("*/*/inputs"))))
            return value
        def extract_bound(*args, **kwargs):
            max_cases.append(len(list(args[5].parents[2].glob("*/*"))))
            return extract(*args, **kwargs)
        with patch.object(worker, "copy_source", side_effect=copy_bound), patch.object(worker, "extract_and_rule", side_effect=extract_bound), redirect_stdout(self.heartbeat):
            summary = worker.run_ticket(self.ticket, limit=6, resource_gate=lease, model_loader=load,
                                        probe=self.probe, decoder=self.decoder, audio_info=self.audio_info,
                                        resource_reader=lambda root: {"cgroup_headroom": 40 * worker.GIB, "local_free": 32 * worker.GIB}, stage_gate=stage)
        self.assertEqual(summary["processed"], 6)
        self.assertFalse(active[0])
        self.assertEqual(stages, [True] + [False] * (1 + 6 + 3))
        self.assertLessEqual(max(max_inputs), 2)
        self.assertLessEqual(max(max_cases), 2)

    def test_cache_only_own_reclaimed_after_durable_commit(self):
        sentinel = self.root / "local" / "other-job"
        sentinel.mkdir()
        (sentinel / "keep").write_bytes(b"unrelated")
        with patch.object(worker, "write_once", wraps=worker.write_once) as write:
            summary = self.run_worker(limit=1)
        self.assertTrue(Path(summary["completed_batches"][0]["commit_path"]).is_file())
        self.assertTrue((sentinel / "keep").exists())
        self.assertFalse(list((self.root / "local").glob("ms-unified-v3-*")))
        self.ticket["ticket_id"] = "publication_failure"
        self.ticket["sources"] = [self.rows[1]]
        original = worker.write_once
        def fail_commit(path, raw):
            if Path(path).name == "commit.json":
                raise OSError("synthetic commit publication failure")
            return original(path, raw)
        with patch.object(worker, "write_once", side_effect=fail_commit), self.assertRaisesRegex(OSError, "synthetic"):
            self.run_worker(limit=1)
        retained = list((self.root / "local").glob("ms-unified-v3-*"))
        self.assertEqual(len(retained), 1)
        self.assertTrue((retained[0] / "cases" / "train" / "fixture1" / "receipt.json").is_file())
        self.assertTrue((sentinel / "keep").exists())


if __name__ == "__main__":
    unittest.main()
