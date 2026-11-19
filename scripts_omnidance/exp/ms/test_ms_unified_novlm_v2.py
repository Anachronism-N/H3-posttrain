"""Mocked models/decoder + real v2 CPU rules; no production media/GPU calls."""
from contextlib import contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import ms_unified_novlm_v1 as frozen_v1
import ms_unified_novlm_v2 as worker


class UnifiedTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "local").mkdir()
        self.rows = []
        for index in range(3):
            sid = f"fixture{index}"
            video, audio = self.root / f"{sid}.mp4", self.root / f"{sid}.wav"
            video.write_bytes(b"synthetic-video-not-decodable")
            audio.write_bytes(b"synthetic-audio-not-decodable")
            self.rows.append({"id": sid, "split": "train", "video_path": str(video), "audio_path": str(audio)})
        self.ticket = {"schema": worker.SCHEMA, "edition": "fixture_new_novlm", "ticket_id": "batch1", "authorization": "native_parent", "gpu_index": 0, "output_root": str(self.root / "shared"), "local_cache_root": str(self.root / "local"), "beat_checkpoint": str(self.root / "mock.ckpt"), "scail_root": str(self.root / "scail"), "sources": self.rows}
        self.n, self.width, self.height = 97, 32, 64
        self.probe = Mock(side_effect=self.actual_probe)
        self.beats = Mock(return_value=(np.array([0.2, 0.7]), np.array([0.2])))
        self.pose = Mock(side_effect=self.pose_result)
        self.loader = Mock(return_value=(self.pose, self.beats))
        self.audio_info = Mock(return_value={"duration_seconds": 8.0, "sample_rate": 16000, "channels": 1, "frames": 128000})
        self.missing, self.outside = False, False

    @contextmanager
    def gate(self, ticket):
        yield {"granted": True, "owner_pid": os.getpid(), "gpu_index": ticket["gpu_index"], "gpu_uuid": "fixture-not-real-GPU", "gpu_observed_free": True, "native_memory_gate_preserved": True, "observed_unix": time.time()}

    def actual_probe(self, path):
        return {"streams": [{"width": self.width, "height": self.height, "duration": self.n / 16, "nb_frames": str(self.n), "time_base": "1/16"}], "frames": [{"best_effort_timestamp": i + 8, "best_effort_timestamp_time": str((i + 8) / 16), "pkt_duration_time": "0.0625"} for i in range(self.n)]}

    def decoder(self, path, width, height):
        for index in range(self.n):
            frame = np.empty((height, width, 3), np.uint8)
            frame[:] = [7, 13, 231]  # Explicit BGR blue=7/red=231, not RGB.
            frame[0, 1, 0] = index
            yield frame

    def pose_result(self, frame):
        np.testing.assert_array_equal(frame[0, 0], [7, 13, 231])
        index = int(frame[0, 1, 0])
        if self.missing and index == 3:
            return np.empty((0, 134, 2)), np.empty((0, 134)), np.empty((0, 4))
        points = np.full((1, 134, 2), [12, 25], np.float32)
        names = {name: index for index, name in enumerate(worker.pose.joint_layout()["names"])}
        positions = {"right_shoulder": [10, 20], "left_shoulder": [20, 20], "right_hip": [10, 30], "left_hip": [20, 30], "right_knee": [10, 40], "left_knee": [20, 40], "right_ankle": [10, 50], "left_ankle": [20, 50], "left_wrist": [21, 22], "right_wrist": [10 + index / 10, 22]}
        for name, position in positions.items():
            points[0, names[name]] = position
        if self.outside and index == 4:
            points[0, names["right_wrist"], 0] = -2
        box = [5, 10, 25, 55]
        if self.missing and index == 4:
            box = [0, 0, 2, 2]  # IoU rejection, never reinitialize from tiny box.
        return points, np.full((1, 134), 2.0, np.float32), np.array([box], np.float32)

    def run_worker(self, **kwargs):
        with patch.object(worker, "resource_snapshot", return_value={"cgroup_headroom": 40 * worker.GIB, "local_free": 32 * worker.GIB}):
            return worker.run_ticket(self.ticket, resource_gate=self.gate, model_loader=self.loader, probe=self.probe, decoder=self.decoder, audio_info=self.audio_info, **kwargs)

    def case(self, sid="fixture0"):
        return self.root / "shared" / self.ticket["edition"] / "batches" / self.ticket["ticket_id"] / "cases" / "train" / sid

    def member_bytes(self, name, sid="fixture0"):
        with tarfile.open(self.case(sid).parents[2] / "batch.tar") as archive:
            return archive.extractfile(f"cases/train/{sid}/{name}").read()

    def load_json(self, name, sid="fixture0"):
        return json.loads(self.member_bytes(name, sid))

    def load_npz(self, sid="fixture0"):
        return np.load(io.BytesIO(self.member_bytes("pose.npz", sid)), allow_pickle=False)

    def test_three_sources_models_once_native_all_frames_bgr_and_actual_cpu_npz_consumer(self):
        with patch.object(worker.v2, "validate_observations", wraps=worker.v2.validate_observations) as validate, patch.object(worker.v2, "generate_draft", wraps=worker.v2.generate_draft) as generate:
            commit = self.run_worker()
        self.assertEqual(commit["unique_sources_attempted"], 3)
        self.assertEqual(self.loader.call_count, 1)
        self.assertEqual(self.pose.call_count, 3 * 97)
        self.assertEqual(self.probe.call_count, 3)
        self.assertEqual(self.beats.call_count, 3)
        self.assertEqual(validate.call_count, 3)
        self.assertEqual(generate.call_count, 3)
        for row in self.rows:
            path = self.case(row["id"])
            bound = {"id": row["id"], "split": row["split"], "npz_path": str(path / "pose.npz"), "fullframe_manifest_path": str(path / "fullframe_manifest.json")}
            manifest = self.load_json("fullframe_manifest.json", row["id"])
            raw = self.member_bytes("pose.npz", row["id"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["npz_sha256"])
            with self.load_npz(row["id"]) as saved:
                data = {key: saved[key] for key in saved.files}
            mask = worker.v2.validate_observations(bound, manifest, data)
            self.assertEqual(str(data["schema"]), worker.CACHE_SCHEMA)
            self.assertEqual(len(data["frame_indices"]), 97)
            self.assertTrue(mask.all())
            draft = worker.v2.generate_draft(bound, manifest, data, mask)
            self.assertTrue(draft["events"])
            self.assertFalse(draft["quality_approved"])
            self.assertEqual(draft["observation"]["unknown_intervals"][-1]["end"], 97 / 16)
            self.assertEqual(draft["observation"]["unknown_intervals"][-1]["start"], 96 / 16)
            self.assertFalse(path.exists())
            self.assertEqual(set(commit["archive_index"][row["id"]]["members"]), set(worker.FILES))
        self.assertFalse(any((self.root / "local").glob("ms-unified-*")))

    def test_missing_iou_and_scores_above_one_retained(self):
        self.missing = True
        self.run_worker(limit=1)
        with self.load_npz() as data:
            self.assertTrue(np.isnan(data["xy"][[3, 4]]).all())
            self.assertFalse(data["tracked"][[3, 4]].any())
            np.testing.assert_array_equal(data["missing_reason"][[3, 4]], [1, 2])
            self.assertTrue(data["tracked"][5])
            self.assertEqual(float(data["score"][0, 0]), 2.0)
            self.assertTrue(data["valid"][0].all())
        self.assertIn(self.load_json("draft.json")["status"], ("partial", "needs_regeneration"))

    def test_outside_points_only_rule_mask_local_unknown(self):
        self.outside = True
        self.run_worker(limit=1)
        with self.load_npz() as data:
            self.assertTrue(data["valid"][4].all())
            self.assertEqual(data["xy"][4, 4, 0], -2)
        evidence = self.load_json("draft.json")["observation"]["spatial_evidence"]
        self.assertEqual(evidence["excluded_point_count"], 1)
        self.assertTrue(evidence["local_evidence_limited"])
        self.assertFalse(evidence["whole_source_unusable_claimed"])

    def test_empty_beats_recorded_audio_clock_only(self):
        self.beats.return_value = (np.array([]), np.array([]))
        self.run_worker(limit=1)
        beats = self.load_json("beats.json")
        self.assertTrue(beats["emptybeats"])
        self.assertEqual(beats["clock"], "audio_only")
        self.assertFalse(beats["motions_on_beats_verified"])
        self.assertFalse(self.load_json("draft.json")["beat_alignment_claimed"])

    def test_gate_denied_zero_model_calls_zero_claims(self):
        with self.assertRaises(PermissionError):
            worker.run_ticket(self.ticket, model_loader=self.loader)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "shared").exists())

    def test_insufficient_real_headroom_denied_before_claim_or_model(self):
        with patch.object(worker, "resource_snapshot", return_value={"cgroup_headroom": 31 * worker.GIB, "local_free": 32 * worker.GIB}), self.assertRaises(PermissionError):
            worker.run_ticket(self.ticket, resource_gate=self.gate, model_loader=self.loader)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "shared").exists())

    def test_stale_gpu_evidence_denied(self):
        @contextmanager
        def stale(ticket):
            with self.gate(ticket) as evidence:
                evidence["observed_unix"] -= 31
                yield evidence
        with self.assertRaises(PermissionError):
            worker.run_ticket(self.ticket, resource_gate=stale, model_loader=self.loader)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "shared").exists())

    def test_reversed_pts_failed_sealed_budget_and_no_reassignment(self):
        actual = self.actual_probe(None)
        actual["frames"][2]["best_effort_timestamp_time"] = actual["frames"][1]["best_effort_timestamp_time"]
        self.probe.side_effect = None
        self.probe.return_value = actual
        commit = self.run_worker(limit=1)
        self.assertEqual(commit["unique_sources_attempted"], 1)
        self.assertEqual(commit["receipts"][0]["status"], "failed_sealed_no_retry")
        self.assertTrue(commit["receipts"][0]["budget_consumed"])
        self.pose.assert_not_called()
        self.ticket["ticket_id"] = "batch2"
        skipped = self.run_worker(limit=1)
        self.assertEqual(skipped["unique_sources_attempted"], 0)
        self.assertEqual(skipped["receipts"][0]["status"], "existing_claim_not_reassigned")
        self.assertEqual(self.probe.call_count, 1)

    def test_duplicate_id_splitmix_rejected_before_gate(self):
        self.ticket["sources"].append({**self.rows[0], "split": "val"})
        gate = Mock(side_effect=AssertionError("gate should not run"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            worker.run_ticket(self.ticket, resource_gate=gate)
        gate.assert_not_called()

    def test_native_decoder_omission_failed_not_complete(self):
        self.decoder = lambda path, width, height: iter([np.zeros((height, width, 3), np.uint8)])
        self.pose.side_effect = None
        self.pose.return_value = self.pose_result(next(UnifiedTest.decoder(self, None, 32, 64)))
        commit = self.run_worker(limit=1)
        self.assertEqual(commit["receipts"][0]["status"], "failed_sealed_no_retry")
        self.assertIn("omitted", commit["receipts"][0]["error"])
        self.assertFalse(commit["production_complete_claimed"])

    def test_batch_archive_only_two_shared_files_no_cases_and_writeonce(self):
        self.run_worker()
        batch = self.case().parents[2]
        self.assertEqual({path.name for path in batch.iterdir()}, {"batch.tar", "commit.json"})
        self.assertFalse((batch / "cases").exists())
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in batch.rglob("*") if path.is_file()}
        with self.assertRaises(FileExistsError):
            self.run_worker()
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})
        with tarfile.open(batch / "batch.tar") as archive:
            names = archive.getnames()
        self.assertEqual(len(names), 15)
        self.assertTrue(all(Path(name).name in worker.FILES for name in names))
        self.assertTrue(all(not any(word in name.lower() for word in ("crop", "rgb", "vlm", "request", ".mp4", ".wav")) for name in names))
        path = batch / "commit.json"
        with self.assertRaises(FileExistsError):
            worker.write_once(path, b"overwrite")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before[path])

    def test_prefetch_source_budget_rejects_without_retry(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            worker.copy_source(self.rows[0], self.root / "small", 1)

    def test_writeonce_symlink_target_is_not_followed(self):
        target = self.root / "target"
        target.write_bytes(b"original")
        path = self.root / "link"
        path.symlink_to(target)
        with self.assertRaises(FileExistsError):
            worker.write_once(path, b"replacement")
        self.assertEqual(target.read_bytes(), b"original")

    def test_claim_exclusive_nofollow_and_failure_budget_immutable(self):
        path = self.root / "claims" / "source.json"
        with patch.object(worker.os, "open", wraps=os.open) as opened:
            worker.claim_once(path, {"budget_consumed": True, "owner_pid": os.getpid()})
        flags = next(call.args[1] for call in opened.call_args_list if call.args[0] == path)
        self.assertTrue(flags & os.O_EXCL)
        self.assertTrue(flags & os.O_NOFOLLOW)
        with self.assertRaises(FileExistsError):
            worker.claim_once(path, {"budget_consumed": False})
        self.assertTrue(json.loads(path.read_text())["budget_consumed"])

    def test_native_gate_only_no_extra_local_lease_stage_checks(self):
        stages = Mock()
        commit = self.run_worker(limit=1, stage_gate=stages)
        self.assertTrue(commit["prefetch_drained"])
        self.assertEqual(stages.call_args_list, [unittest.mock.call(False)] * 3)
        self.assertFalse(list((self.root / "local").glob("*.lease")))
        self.assertTrue(commit["receipts"][0]["input_copy_completed"])
        self.assertTrue(all(value >= 0 for value in commit["profile_seconds"].values()))

    def test_insufficient_independent_local_reserve_denied(self):
        with patch.object(worker, "resource_snapshot", return_value={"cgroup_headroom": 40 * worker.GIB, "local_free": 15 * worker.GIB}), self.assertRaises(PermissionError):
            worker.run_ticket(self.ticket, resource_gate=self.gate, model_loader=self.loader)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "shared").exists())

    def test_actual_pts_ticks_mismatch_rejected(self):
        actual = self.actual_probe(None)
        actual["frames"][3]["best_effort_timestamp"] = 0
        with self.assertRaisesRegex(ValueError, "PTS ticks"):
            worker.video_metadata(actual)

    def test_bounded_output_artifacts_rejected_before_any_payload_write(self):
        copied = {"video_path": Path(self.rows[0]["video_path"]), "audio_path": Path(self.rows[0]["audio_path"])}
        hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in copied.items()}
        local_case = self.root / "bounded" / "train" / self.rows[0]["id"]
        with self.assertRaisesRegex(ValueError, "output budget"):
            worker.extract_and_rule(self.rows[0], copied, hashes, (self.pose, self.beats), local_case, local_case, self.probe, self.decoder, max_output_bytes=1, audio_info=self.audio_info)
        self.assertFalse(list(local_case.iterdir()))

    def test_actual_cgroup_v1_v2_headroom_and_ceph_fuse_rejection(self):
        from types import SimpleNamespace
        for version in (1, 2):
            group = "4:memory:/fixture\n" if version == 1 else "0::/fixture\n"
            root = "/sys/fs/cgroup/memory" if version == 1 else "/sys/fs/cgroup"
            limit = "memory.limit_in_bytes" if version == 1 else "memory.max"
            usage = "memory.usage_in_bytes" if version == 1 else "memory.current"
            documents = {"/proc/self/cgroup": group, "/proc/meminfo": "MemAvailable: 67108864 kB\n", "/proc/mounts": "local / ext4 rw 0 0\n"}
            for parent in (root, root + "/fixture"):
                documents[parent + "/" + limit] = str(40 * worker.GIB)
                documents[parent + "/" + usage] = str(8 * worker.GIB)
            with patch.object(Path, "read_text", autospec=True, side_effect=lambda path, **kwargs: documents[str(path)]), patch.object(worker.shutil, "disk_usage", return_value=SimpleNamespace(free=24 * worker.GIB)):
                actual = worker.resource_snapshot(self.root / "local")
                self.assertEqual(actual["cgroup_headroom"], 32 * worker.GIB)
                self.assertEqual(actual["local_free"], 24 * worker.GIB)
                documents["/proc/mounts"] += f"ceph {self.root}/local fuse.ceph-fuse rw 0 0\n"
                with self.assertRaises(PermissionError):
                    worker.resource_snapshot(self.root / "local")

    def test_beat_timestamps_checked_against_audio_not_video_duration(self):
        self.audio_info.return_value["duration_seconds"] = 0.5
        commit = self.run_worker(limit=1)
        self.assertEqual(commit["receipts"][0]["status"], "failed_sealed_no_retry")
        self.assertTrue(commit["receipts"][0]["input_copy_completed"])
        self.assertIn("BeatThis", commit["receipts"][0]["error"])

    def test_failed_copy_receipt_marks_not_completed_and_drains(self):
        self.ticket["sources"][0]["video_sha256"] = "0" * 64
        commit = self.run_worker(limit=1)
        receipt = commit["receipts"][0]
        self.assertEqual(receipt["status"], "failed_sealed_no_retry")
        self.assertFalse(receipt["input_copy_completed"])
        self.assertTrue(commit["prefetch_drained"])
        self.pose.assert_not_called()

    def test_explicit_tools_use_ticket_paths_without_model_or_path_guessing(self):
        with patch.object(worker.subprocess, "run", return_value=unittest.mock.Mock(stdout="{}")) as run:
            worker.probe_video(Path("/fixture/video.mp4"), "/fixture/verified-ffprobe")
        self.assertEqual(run.call_args.args[0][0], "/fixture/verified-ffprobe")
        self.assertEqual(self.ticket["scail_root"], str(self.root / "scail"))

    def test_sealed_disk_receipts_equal_commit_after_all_profile_updates(self):
        self.ticket["sources"][1]["video_sha256"] = "0" * 64
        commit = self.run_worker()
        self.assertEqual(len(commit["receipts"]), 3)
        for receipt in commit["receipts"]:
            disk = self.load_json("receipt.json", receipt["source_row"]["id"])
            self.assertEqual(disk, receipt)
            self.assertIn("whole_case_e2e", disk["profile_seconds"])
            self.assertGreaterEqual(disk["profile_seconds"]["whole_case_e2e"], disk["profile_seconds"]["copy"])
        batch = self.case().parents[2]
        with tarfile.open(batch / "batch.tar") as archive:
            for receipt in commit["receipts"]:
                row = receipt["source_row"]
                archived = json.load(archive.extractfile(f"cases/{row['split']}/{row['id']}/receipt.json"))
                self.assertEqual(archived, receipt)

    def test_cpu_test_cannot_accept_production_ticket(self):
        with self.assertRaises(SystemExit), patch.object(worker, "load_models") as load:
            worker.main(["--cpu-test", "--ticket", "forbidden.json"])
        load.assert_not_called()

    def test_archive_members_byte_identical_to_local_sealed_payloads_and_locator_binding(self):
        local_payloads = {}
        original = worker.write_once
        def capture(path, raw):
            result = original(path, raw)
            path = Path(path)
            if "cases" in path.parts and path.name in worker.FILES:
                local_payloads[str(path.relative_to(next(parent for parent in path.parents if parent.name.startswith("ms-unified-"))))] = path.read_bytes()
            return result
        with patch.object(worker, "write_once", side_effect=capture):
            commit = self.run_worker()
        archive_path = Path(commit["archive_path"])
        self.assertEqual(archive_path, self.case().parents[2] / "batch.tar")
        self.assertEqual(commit["archive_bytes"], archive_path.stat().st_size)
        self.assertEqual(commit["archive_sha256"], hashlib.sha256(archive_path.read_bytes()).hexdigest())
        self.assertEqual(commit["batch_sha256"], commit["archive_sha256"])
        self.assertEqual(json.loads((archive_path.parent / "commit.json").read_text()), commit)
        self.assertEqual(commit["publication_scope"], "one archive + commit, no shared per-source files")
        with tarfile.open(archive_path) as archive:
            self.assertEqual(set(archive.getnames()), set(local_payloads))
            for member, raw in local_payloads.items():
                self.assertEqual(archive.extractfile(member).read(), raw)
        for receipt in commit["receipts"]:
            row = receipt["source_row"]
            sid = row["id"]
            self.assertEqual(receipt["schema"], "ms_unified_novlm_case_receipt_v2")
            self.assertFalse(receipt["vlm_called"])
            self.assertFalse(receipt["quality_approved"])
            index = commit["archive_index"][sid]
            self.assertEqual(index["source_row"], row)
            self.assertEqual(index["members"], receipt["artifact_members"])
            self.assertEqual(index["receipt_member"], f"cases/train/{sid}/receipt.json")
            manifest = self.load_json("fullframe_manifest.json", sid)
            draft = self.load_json("draft.json", sid)
            for metadata in (manifest, draft, draft["source"], receipt):
                self.assertEqual(metadata["artifact_storage"], "archive_member")
                self.assertEqual(metadata["locator"], receipt["locator"])
                self.assertEqual(metadata["locator"]["archive_path"], str(archive_path))
            self.assertEqual(manifest["npz_path"], str(self.case(sid) / "pose.npz"))
            self.assertEqual(draft["source"]["npz_path"], manifest["npz_path"])
            self.assertFalse(Path(manifest["npz_path"]).exists())
            with self.load_npz(sid) as saved:
                data = {key: saved[key] for key in saved.files}
            self.assertEqual(draft["source"]["fullframe_binding"], worker.v2.fullframe_binding(manifest, data))

    def test_failed_after_partial_local_write_archives_only_failed_receipt(self):
        original = worker.write_once
        def fail_manifest_once(path, raw):
            if Path(path).name == "fullframe_manifest.json":
                raise OSError("fixture local manifest save failure")
            return original(path, raw)
        with patch.object(worker, "write_once", side_effect=fail_manifest_once):
            commit = self.run_worker(limit=1)
        receipt = commit["receipts"][0]
        self.assertEqual(receipt["status"], "failed_sealed_no_retry")
        self.assertEqual(receipt, self.load_json("receipt.json"))
        self.assertEqual(set(receipt["artifact_members"]), {"receipt.json"})
        self.assertIsNone(receipt["locator"]["npz_member"])
        self.assertIsNone(receipt["locator"]["fullframe_manifest_member"])
        self.assertFalse(receipt["quality_approved"])
        with tarfile.open(commit["archive_path"]) as archive:
            self.assertEqual(archive.getnames(), ["cases/train/fixture0/receipt.json"])
        self.ticket["ticket_id"] = "batch2"
        with patch.object(worker, "copy_source", side_effect=AssertionError("cannot reissue")):
            skipped = self.run_worker(limit=1)
        self.assertEqual(skipped["receipts"][0]["status"], "existing_claim_not_reassigned")
        self.assertEqual(skipped["receipts"][0], self.load_json("receipt.json"))
        self.assertEqual(self.probe.call_count, 1)
        self.assertEqual(self.beats.call_count, 1)

    def test_archive_stream_fsync_link_dir_fsync_once_and_1mib_bounded_reads(self):
        raw = b"fixture" * (300 * 1024)
        with tempfile.TemporaryFile(dir=self.root / "local") as stream:
            stream.write(raw)
            reader = Mock(wraps=stream)
            target = self.root / "shared-transfer" / "batch.tar"
            operations = []
            original_fsync, original_link = os.fsync, os.link
            def fsync(fd):
                operations.append("fsync")
                return original_fsync(fd)
            def link(source, destination):
                operations.append("link")
                return original_link(source, destination)
            with patch.object(worker.os, "fsync", side_effect=fsync), patch.object(worker.os, "link", side_effect=link):
                size, digest = worker.publish_archive_once(target, reader)
            self.assertEqual(operations, ["fsync", "link", "fsync"])
            self.assertTrue(reader.read.call_count >= 3)
            self.assertTrue(all(call.args == (1024 * 1024,) for call in reader.read.call_args_list))
            self.assertEqual(size, len(raw))
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            self.assertEqual(target.read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                worker.publish_archive_once(target, stream)
            self.assertFalse(list(target.parent.glob(".publish-*")))

    def test_archive_output_bound_checked_before_shared_write_and_symlink_not_followed(self):
        with tempfile.TemporaryFile(dir=self.root / "local") as stream:
            stream.write(b"oversize")
            target = self.root / "too-large" / "batch.tar"
            with self.assertRaisesRegex(ValueError, "output budget"):
                worker.publish_archive_once(target, stream, max_bytes=1)
            self.assertFalse(target.parent.exists())
            victim = self.root / "victim"
            victim.write_bytes(b"unchanged")
            target.parent.mkdir()
            target.symlink_to(victim)
            with self.assertRaises(FileExistsError):
                worker.publish_archive_once(target, stream)
            self.assertEqual(victim.read_bytes(), b"unchanged")

    def test_shared_publication_only_two_durable_links_and_local_archive_not_bytesio(self):
        shared_links, local_archive_dirs = [], []
        original_link = os.link
        original_temp = worker.tempfile.TemporaryFile
        def link(source, destination):
            if Path(destination).is_relative_to(self.root / "shared") and "claims" not in Path(destination).parts:
                shared_links.append(Path(destination).name)
            return original_link(source, destination)
        def local_archive(*args, **kwargs):
            local_archive_dirs.append(Path(kwargs["dir"]))
            return original_temp(*args, **kwargs)
        with patch.object(worker.os, "link", side_effect=link), patch.object(worker.tempfile, "TemporaryFile", side_effect=local_archive):
            self.run_worker()
        self.assertEqual(shared_links, ["batch.tar", "commit.json"])
        self.assertEqual(local_archive_dirs, [self.root / "local"])

    def test_unsupported_schema_and_finite_cap_denied_before_gate_models_or_claims(self):
        for schema in ("ms_unified_novlm_ticket_v1", "unapproved", None):
            ticket = {**self.ticket, "schema": schema}
            gate = Mock(side_effect=AssertionError("gate cannot run"))
            with self.assertRaises(ValueError):
                worker.run_ticket(ticket, resource_gate=gate, model_loader=self.loader)
            gate.assert_not_called()
        for limit in (0, 101, True, 1.5):
            with self.assertRaises(ValueError):
                worker.check_ticket(self.ticket, limit)
        ticket = {**self.ticket, "sources": [self.rows[0]] * 101}
        with self.assertRaisesRegex(ValueError, "at most 100"):
            worker.check_ticket(ticket, 100)
        self.loader.assert_not_called()
        self.assertFalse((self.root / "shared").exists())

    def test_old_v1_new_v2_same_fixture_arrays_events_pts_and_native_binding(self):
        row = self.rows[0]
        copied = {key: Path(row[key]) for key in ("video_path", "audio_path")}
        hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in copied.items()}
        bound = self.root / "binding" / "cases" / "train" / row["id"]
        old_case = self.root / "old-local" / "train" / row["id"]
        new_case = self.root / "new-local" / "train" / row["id"]
        for module, local in ((frozen_v1, old_case), (worker, new_case)):
            module.extract_and_rule(row, copied, hashes, (self.pose, self.beats), local, bound,
                                    self.probe, self.decoder, audio_info=self.audio_info)
        self.assertEqual((old_case / "pose.npz").read_bytes(), (new_case / "pose.npz").read_bytes())
        old_manifest = json.loads((old_case / "fullframe_manifest.json").read_text())
        new_manifest = json.loads((new_case / "fullframe_manifest.json").read_text())
        storage_keys = {"artifact_storage", "locator", "artifact_members", "source_path_binding"}
        self.assertEqual(old_manifest, {key: value for key, value in new_manifest.items() if key not in storage_keys})
        with np.load(new_case / "pose.npz", allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        np.testing.assert_array_equal(data["frame_indices"], np.arange(97))
        np.testing.assert_array_equal(data["source_timestamps"], (np.arange(97) + 8) / 16)
        np.testing.assert_array_equal(data["times"], np.arange(97) / 16)
        self.assertEqual(float(data["score"][0, 0]), 2.0)
        old_draft = json.loads((old_case / "draft.json").read_text())
        new_draft = json.loads((new_case / "draft.json").read_text())
        self.assertEqual({key: value for key, value in old_draft.items() if key != "source"},
                         {key: value for key, value in new_draft.items() if key != "source" and key not in storage_keys})
        self.assertTrue(new_draft["events"])
        self.assertEqual(new_draft["source"]["fullframe_binding"], worker.v2.fullframe_binding(new_manifest, data))
        self.assertEqual(old_draft["source"]["fullframe_binding"]["observations_sha256"],
                         new_draft["source"]["fullframe_binding"]["observations_sha256"])
        self.assertEqual(self.pose.call_count, 2 * 97)
        self.assertEqual(self.beats.call_count, 2)

    def test_archive_save_failure_keeps_claim_and_never_reissues(self):
        with patch.object(worker, "publish_archive_once", side_effect=OSError("fixture shared archive save failure")), self.assertRaisesRegex(OSError, "archive save"):
            self.run_worker(limit=1)
        self.assertFalse((self.case().parents[2] / "commit.json").exists())
        self.assertTrue((self.root / "shared" / self.ticket["edition"] / "claims" / "sources" / "fixture0.json").is_file())
        self.ticket["ticket_id"] = "batch2"
        with patch.object(worker, "copy_source", side_effect=AssertionError("cannot reissue")):
            commit = self.run_worker(limit=1)
        self.assertEqual(commit["unique_sources_attempted"], 0)
        self.assertEqual(commit["receipts"][0]["status"], "existing_claim_not_reassigned")
        self.assertEqual(self.pose.call_count, 97)
        self.assertEqual(self.beats.call_count, 1)

    def test_receipt_write_post_link_failure_cannot_mutate_sealed_success_or_retry(self):
        original = worker.write_once
        def fail_after_seal(path, raw):
            result = original(path, raw)
            if Path(path).name == "receipt.json":
                raise OSError("fixture receipt post-link durability failure")
            return result
        with patch.object(worker, "write_once", side_effect=fail_after_seal), self.assertRaises(FileExistsError):
            self.run_worker(limit=1)
        self.assertFalse((self.case().parents[2] / "commit.json").exists())
        self.ticket["ticket_id"] = "batch2"
        commit = self.run_worker(limit=1)
        self.assertEqual(commit["unique_sources_attempted"], 0)
        self.assertEqual(self.pose.call_count, 97)



if __name__ == "__main__":
    unittest.main()
