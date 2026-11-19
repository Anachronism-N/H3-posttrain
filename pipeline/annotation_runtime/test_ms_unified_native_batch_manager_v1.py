"""CPU mocks only: no GPU process, resource signal, native FLOCK or lease writes."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("native_under_test", Path(__file__).with_name("ms_unified_native_batch_manager_v1.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"PYTHONPATH": m.DEPS, "PYTHONHOME": "", "DWPOSE_ALLOW_CPU": "0"}))
        self.stack.enter_context(patch.object(m, "ROOT", self.root))
        self.stack.enter_context(patch.object(m, "dependencies", return_value={str(m.SELF), str(Path(m.base.__file__)), str(m.CORE)}))
        self.owner = {"pid": 700001, "uid": os.getuid(), "argv": ["python", str(m.base.HOLDER)], "start_ticks": 99,
                      "cgroup": "fixture", "CVD": ["CUDA_VISIBLE_DEVICES=2"], "exe": "/fixture/python",
                      "gpu": 2, "uuid": "GPU-fixture", "sig_ign": 0, "sig_blk": 0, "ppid": 1}
        self.rows = [{"id": sid, "split": "train", "video_path": "/fixture/video/" + sid + ".mp4",
                      "audio_path": "/fixture/audio/" + sid + ".wav"} for sid in ("sourceA", "sourceB", "sourceC")]
        self.cfg = {"schema": m.SCHEMA, "node": "N6", "gpus": [2], "holders": [self.owner],
            "release_authorized": True, "phase": "ms_unified_native_v1_diag3", "targetcount": 3, "input_count": 3,
            "out": str(self.root / "ms_unified_native_v1_diag3"), "queue": str(self.root / "queues/ms_unified_native_v1_diag3"),
            "config_path": str(self.root / "config.json"), "cwd": str(m.RT), "selected_source_ids": [r["id"] for r in self.rows],
            "producer": m.ref(m.SELF), "resource_manager_ref": m.ref(m.SELF), "core_ref": m.ref(m.CORE),
            "env": {"CUDA_VISIBLE_DEVICES": "2", "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": m.DEPS}}
        self.cfg["argv"] = [m.PYTHON, "-B", str(m.SELF), "worker", "--config", self.cfg["config_path"]]
        ticket = {"schema": "ms_unified_novlm_ticket_v1", "authorization": "native_parent", "gpu_index": 2,
                  "output_root": self.cfg["out"], "edition": self.cfg["phase"], "ticket_id": "diag3",
                  "local_cache_root": "/tmp", "max_cache_bytes": 8*m.GIB, "beat_checkpoint": "/fixture/beat.ckpt", "scail_root": str(m.SCAIL), "sources": self.rows}
        self.cfg["ticket"] = self.put("ticket.json", ticket)
        plan = self.root / "plan.jsonl"
        plan.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        self.cfg["plan"] = m.ref(plan)
        self.cfg["modelref"] = self.put("models.json", {"schema": "ms_unified_native_model_identity_v1", "files": {}})
        self.cfg["modelref"]["model_identity_sha256"] = self.cfg["modelref"]["sha256"]
        self.cfg["resource_previous_safe"] = self.put("queues/prior/resources_safe.json", {k: True for k in
            ("all_holders_safe", "locks_released", "leases_restored", "managed_parent_wait", "unknown_jobs_not_signaled")})
        self.cfg["holder_restored_refs"] = [self.put("queues/prior/holder_restored_gpu2.json",
            {"uuid": self.owner["uuid"], "holder_identity": self.owner})]
        self.bind()

    def put(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return m.ref(path)

    def bind(self):
        refs = [self.cfg[k] for k in ("ticket", "plan", "core_ref", "modelref", "resource_previous_safe")] + self.cfg["holder_restored_refs"]
        self.cfg["source_sha256"] = {p: m.base.sha(p) for p in m.dependencies() | {r["path"] for r in refs}}

    def resources(self, headroom=160*m.GIB, host=None):
        self.stack.enter_context(patch.object(m.h.subprocess, "check_output", return_value=m.HOST if host is None else host))
        self.stack.enter_context(patch.object(m.base, "identity", return_value=self.owner))
        self.stack.enter_context(patch.object(m.base, "uuid_for", return_value=self.owner["uuid"]))
        self.stack.enter_context(patch.object(m.base, "apps", return_value=[{"pid": self.owner["pid"], "uuid": self.owner["uuid"]}]))
        self.stack.enter_context(patch.object(m.h, "lock_snapshot", return_value={"held": []}))
        self.stack.enter_context(patch.object(m.base, "lease_snapshot", return_value=None))
        self.stack.enter_context(patch.object(m.base, "headroom", return_value={"headroom": headroom, "current": 1, "maximum": "max"}))
        self.stack.enter_context(patch.object(m.shutil, "disk_usage", return_value=types.SimpleNamespace(free=32*m.GIB)))

    def test_default_deny(self):
        with self.assertRaises(PermissionError):
            m.validate_common({})

    def test_wrong_node_denied(self):
        self.cfg["node"] = "N3"
        with self.assertRaises(PermissionError):
            m.validate_common(self.cfg)

    def test_wrong_gpu_and_two_gpus_denied(self):
        for gpus in ([0], [2, 3], [], [True]):
            with self.subTest(gpus=gpus), self.assertRaises(PermissionError):
                m.validate_common(dict(self.cfg, gpus=gpus))

    def test_wrong_worker_argv_denied(self):
        for args in (["/opt/venv_vllm/bin/python"] + self.cfg["argv"][1:], self.cfg["argv"] + ["--limit", "3"]):
            with self.subTest(args=args), self.assertRaises(PermissionError):
                m.validate_common(dict(self.cfg, argv=args))

    def test_extra_holder_and_env_denied(self):
        with self.assertRaises(PermissionError):
            m.validate_common(dict(self.cfg, holders=[self.owner, self.owner]))
        with self.assertRaises(PermissionError):
            m.validate_common(dict(self.cfg, env=dict(self.cfg["env"], DWPOSE_ALLOW_CPU="1")))

    def test_only_exact_offline_pythonpath_authorized(self):
        m.validate_common(self.cfg)
        self.put("config.json", self.cfg)
        for value in ("", "/tmp/arbitrary", m.DEPS + ":/tmp/inject", "/opt/venv_eval/lib/python3.11/site-packages"):
            with self.subTest(value=value), self.assertRaises(PermissionError):
                m.validate_common(dict(self.cfg, env=dict(self.cfg["env"], PYTHONPATH=value)))
            with self.subTest(actual_environment=value), patch.dict(os.environ, {"PYTHONPATH": value}):
                with self.assertRaisesRegex(PermissionError, "exact offline dependency"):
                    m.main(["--config", self.cfg["config_path"], "--config-sha", m.base.sha(self.cfg["config_path"]),
                            "--manager-sha", m.BASE_SHA, "--check-only"])
                with self.assertRaisesRegex(PermissionError, "exact offline dependency"):
                    m.worker(self.cfg["config_path"])
        self.assertFalse(Path(self.cfg["queue"]).exists())

    def test_wrong_physical_host_denied(self):
        self.resources(host="wrong-host")
        with self.assertRaisesRegex(PermissionError, "physical node"):
            m.validate(self.cfg)

    def test_native_ram_budget_never_lowered(self):
        self.resources(headroom=160*m.GIB-1)
        self.assertEqual(m.base.required_budget(self.cfg), 160*m.GIB)
        with self.assertRaisesRegex(PermissionError, "160 GiB"):
            m.validate(self.cfg)

    def test_small_model_only_content_refs(self):
        scail = self.root / "scail"
        beat = self.root / "beat.ckpt"
        paths = {"detector": scail / "pretrained_weights/DWPose/yolox_l.onnx",
                 "pose": scail / "pretrained_weights/DWPose/dw-ll_ucoco_384.onnx", "beat": beat}
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("CPU mock " + name).encode())
        ticket = m.load(self.cfg["ticket"]["path"])
        ticket["beat_checkpoint"] = str(beat)
        ticket["scail_root"] = str(scail)
        self.cfg["ticket"] = self.put("ticket.json", ticket)
        identity = {"schema": "ms_unified_native_model_identity_v1", "files": {name: m.ref(path) for name, path in paths.items()}}
        self.cfg["modelref"] = self.put("models.json", identity)
        self.cfg["modelref"]["model_identity_sha256"] = self.cfg["modelref"]["sha256"]
        with patch.object(m, "SCAIL", scail), patch.object(m, "module", side_effect=AssertionError("models imported")):
            m.check_model(self.cfg)
            beat.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(PermissionError, "checkpoint identity"):
                m.check_model(self.cfg)

    def test_previous_safe_requires_all_flags(self):
        self.resources()
        value = m.load(self.cfg["resource_previous_safe"]["path"])
        value["unknown_jobs_not_signaled"] = False
        self.cfg["resource_previous_safe"] = self.put("queues/prior/resources_safe.json", value)
        self.bind()
        with self.assertRaisesRegex(PermissionError, "every native flag"):
            m.validate(self.cfg)

    def test_check_only_no_queue_model_signal_lock_or_lease_write(self):
        self.resources()
        self.put("config.json", self.cfg)
        before = m.base.sha(m.base.__file__)
        with patch.object(m, "check_model") as model_ref, patch.object(m, "module", side_effect=AssertionError("model/module imported")), \
             patch.object(m.base, "main", side_effect=AssertionError("native lifecycle called")), \
             patch.object(m.base.os, "open", side_effect=AssertionError("lock/lease opened")), \
             patch.object(m.base.os, "pidfd_open", side_effect=AssertionError("holder pidfd opened")), \
             patch.object(m.base.signal, "signal", side_effect=AssertionError("signals configured")), \
             patch.object(m.base.signal, "pidfd_send_signal", side_effect=AssertionError("signal sent")), \
             patch.object(m.base.subprocess, "Popen", side_effect=AssertionError("GPU process started")), \
             patch.dict(os.environ, {"PYTHONPATH": m.DEPS, "PYTHONHOME": "", "DWPOSE_ALLOW_CPU": "0"}), \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            m.main(["--config", self.cfg["config_path"], "--config-sha", m.base.sha(self.cfg["config_path"]),
                    "--manager-sha", m.BASE_SHA, "--check-only"])
        self.assertFalse(Path(self.cfg["queue"]).exists())
        self.assertFalse(Path(self.cfg["out"]).exists())
        self.assertFalse(json.loads(stdout.getvalue())["models_loaded"])
        model_ref.assert_called_once()
        self.assertEqual(before, m.BASE_SHA)
        self.assertEqual(m.base.sha(m.base.__file__), before)

    def sealed_summary(self):
        ticket = m.ticket_for(self.cfg)
        batch = Path(self.cfg["out"]) / ticket["edition"] / "batches" / ticket["ticket_id"]
        batch.mkdir(parents=True)
        archive = batch / "batch.tar"
        archive.write_bytes(b"cpu-fixture-archive")
        receipts = [{"source_row": row, "status": "extracted_rule_draft_unapproved", "input_copy_completed": True} for row in self.rows]
        for receipt in receipts:
            case = batch / "cases" / receipt["source_row"]["split"] / receipt["source_row"]["id"]
            case.mkdir(parents=True)
            (case / "receipt.json").write_text(json.dumps(receipt))
        commit = {"schema": "ms_unified_novlm_batch_commit_v1", "ticket_id": ticket["ticket_id"], "edition": ticket["edition"],
            "unique_sources_attempted": 3, "receipts": receipts, "batch_sha256": m.base.sha(archive), "prefetch_drained": True,
            "quality_approved": False, "beat_alignment_claimed": False, "production_complete_claimed": False}
        (batch / "commit.json").write_text(json.dumps(commit))
        value = {"schema": m.SUMMARY_SCHEMA, "selected_source_ids": self.cfg["selected_source_ids"], "processed": 3, "selected": 3,
            "prefetch_drained": True, "human_review_completed": False, "production_complete_claimed": False,
            "commit_ref": m.ref(batch / "commit.json"), "source_outcomes": receipts,
            "counts": {"extracted_rule_draft_unapproved": 3}, "copied": 3, "partial": 0, "error_sealed": 0}
        summary = Path(self.cfg["out"]) / "summary.json"
        summary.write_text(json.dumps(value))
        return summary, value, batch, commit

    def test_summary_exact_three_sealed_outcomes(self):
        self.sealed_summary()
        result = m.summaries(self.cfg, Path(self.cfg["out"]))
        self.assertEqual(next(iter(result.values()))["summary"]["total_source_count"], 3)

    def test_summary_missing_source_rejected(self):
        summary, value, batch, commit = self.sealed_summary()
        commit["receipts"].pop()
        (batch / "commit.json").write_text(json.dumps(commit))
        value["commit_ref"] = m.ref(batch / "commit.json")
        summary.write_text(json.dumps(value))
        with self.assertRaisesRegex(PermissionError, "exactly cover"):
            m.summaries(self.cfg, Path(self.cfg["out"]))

    def test_summary_fake_copy_counter_rejected(self):
        summary, value, _, _ = self.sealed_summary()
        value["copied"] = 2
        summary.write_text(json.dumps(value))
        with self.assertRaisesRegex(PermissionError, "counters"):
            m.summaries(self.cfg, Path(self.cfg["out"]))

    def test_worker_model_load_uses_true_gate_at_load_boundary(self):
        self.put("config.json", self.cfg)
        events = []
        ownership = {"admitted": True, "caller_owned": True, "gpu_uuids": [self.owner["uuid"]],
                     "child_identity": {"pid": os.getpid()}}
        def fake_gate(cfg, before_load=False):
            events.append(("gate", before_load))
            return ownership
        def fake_loader(ticket):
            events.append(("model_load", None))
            return object()
        def fake_run(ticket, limit, resource_gate, model_loader, stage_gate, resource_reader):
            self.assertEqual(limit, 3)
            self.assertEqual(resource_reader("/tmp")["cgroup_headroom"], 160*m.GIB)
            with resource_gate(ticket) as evidence:
                self.assertTrue(evidence["granted"])
                self.assertTrue(evidence["gpu_observed_free"])
                model_loader(ticket)
                stage_gate(False)
                summary, _, _, commit = self.sealed_summary()
                summary.unlink()
                return commit
        fake_core = types.SimpleNamespace(load_models=fake_loader, run_ticket=fake_run)
        imports = {name: types.SimpleNamespace(__file__=str(m.CORE_DIR / (name + ".py"))) for name in
                   ("dwpose_full_extract", "ms_fullframe_pose2d_draft_v2", "ms_fullframe_pose2d_draft_v1", "ms_pose2d_draft")}
        saved_path = list(sys.path)
        try:
            with patch.object(m, "gate", side_effect=fake_gate), patch.object(m, "check_model"), \
                 patch.object(m, "module", return_value=fake_core), patch.dict(sys.modules, imports), \
                 patch.object(m.base, "apps", return_value=[]), \
                 patch.object(m, "local_budget", return_value={"memory": {"headroom": 160*m.GIB}}):
                m.worker(self.cfg["config_path"])
        finally:
            sys.path[:] = saved_path
        index = events.index(("model_load", None))
        self.assertEqual(events[index-1], ("gate", True))
        self.assertEqual(sum(value for kind, value in events if kind == "gate"), 1)
        self.assertEqual(events[-1], ("gate", False))
        self.assertEqual(m.load(Path(self.cfg["out"]) / "summary.json")["copied"], 3)

    def test_gate_rebind_retains_real_parent_denial(self):
        queue = Path(self.cfg["queue"])
        queue.mkdir(parents=True)
        cfg = dict(self.cfg, _config_path=self.cfg["config_path"], _config_sha="fixture-sha")
        parent = dict(self.owner, pid=101, uid=os.getuid(), cgroup="fixture", argv=["wrong-parent"] )
        child = dict(self.owner, pid=102, uid=os.getuid(), cgroup="fixture", argv=cfg["argv"])
        (queue / "admission.json").write_text(json.dumps({"manager_identity": parent}))
        with patch.object(m.base, "identity", side_effect=lambda pid: parent if pid == 101 else child), \
             patch.object(m.h.os, "getppid", return_value=101), patch.object(m.h.os, "getpid", return_value=102):
            with self.assertRaisesRegex(ValueError, "parent native manager argv"):
                m.gate(cfg, True)

    def test_gate_preserves_full_kernel_ownership_checks(self):
        queue = Path(self.cfg["queue"])
        queue.mkdir(parents=True)
        cfg = dict(self.cfg, _config_path=self.cfg["config_path"], _config_sha="fixture-sha")
        parent = dict(self.owner, pid=101, uid=os.getuid(), cgroup="fixture", argv=[m.PYTHON, "-B", str(m.SELF),
            "--config", cfg["_config_path"], "--config-sha", cfg["_config_sha"], "--manager-sha", m.BASE_SHA])
        child = dict(self.owner, pid=102, uid=os.getuid(), cgroup="fixture", argv=cfg["argv"])
        admission = {"manager_identity": parent, "manager_sha256": m.BASE_SHA, "source_sha256": {str(m.SELF): m.base.sha(m.SELF)},
            "config_sha256": cfg["_config_sha"], "out": cfg["out"], "gpus": [2], "node": "N6", "holders": [self.owner]}
        (queue / "admission.json").write_text(json.dumps(admission))
        (queue / "live.json").write_text(json.dumps({"child_identity": child}))
        lock = types.SimpleNamespace(st_mode=stat.S_IFREG|0o600, st_uid=os.getuid(), st_size=0, st_dev=os.makedev(8,1), st_ino=555)
        lease = types.SimpleNamespace(st_mode=stat.S_IFREG|0o600, st_uid=os.getuid(), st_size=0, st_dev=os.makedev(8,1), st_ino=556, st_mtime=time.time())
        line = "1: FLOCK ADVISORY WRITE 101 08:01:555 0 EOF"
        class ProcPath:
            def __init__(self, path): self.path = str(path)
            def __truediv__(self, name): return ProcPath(self.path + "/" + str(name))
            def __fspath__(self): return self.path
            def __str__(self): return self.path
            def resolve(self): return Path(self.path).resolve()
            def iterdir(self): return [ProcPath("/proc/101/fd/10"), ProcPath("/proc/101/fd/11")]
            def stat(self): return lock if self.path.endswith("/10") else lease
            def lstat(self): return lock if "ms12k_gpu" in self.path else lease
            def read_text(self): return line
        def path_factory(value):
            return ProcPath(value) if str(value).startswith(("/proc", "/tmp/ms12k_gpu", "/tmp/od_reserve")) else Path(value)
        with patch.object(m.base, "identity", side_effect=lambda pid: parent if pid == 101 else child), \
             patch.object(m.base, "uuid_for", return_value=self.owner["uuid"]), patch.object(m.base, "apps", return_value=[]) as apps, \
             patch.object(m.h.os, "getppid", return_value=101), patch.object(m.h.os, "getpid", return_value=102), \
             patch.object(m.h, "Path", side_effect=path_factory), \
             patch.object(m.h.os, "readlink", side_effect=lambda fd: "/tmp/ms12k_gpu_2.lock" if str(fd).endswith("/10") else "/tmp/od_reserve_2"):
            original_code = m.h.parent_gate.__code__
            self.assertTrue(m.gate(cfg, True)["admitted"])
            self.assertIs(m.h.parent_gate.__code__, original_code)
            self.assertEqual(original_code.co_consts.count("/opt/venv_vllm/bin/python"), 1)
            parent["argv"][0] = "/opt/venv_vllm/bin/python"
            (queue / "admission.json").write_text(json.dumps(admission))
            with self.assertRaisesRegex(ValueError, "parent native manager argv"):
                m.gate(cfg, True)
            parent["argv"][0] = m.PYTHON
            (queue / "admission.json").write_text(json.dumps(admission))
            apps.return_value = [{"uuid": self.owner["uuid"], "pid": 999}]
            with self.assertRaisesRegex(ValueError, "occupied before"):
                m.gate(cfg, True)
            lease.st_mtime = time.time()-31
            apps.return_value = []
            with self.assertRaisesRegex(ValueError, "heartbeat"):
                m.gate(cfg, False)


batch = m
class BatchTests(NativeTests):
    def setUp(self):
        super().setUp()
    def select_count(self, n):
        self.rows = [{"id": "batch_%04d" % i, "split": "train", "video_path": "/fixture/video/batch_%04d.mp4" % i,
                      "audio_path": "/fixture/audio/batch_%04d.wav" % i} for i in range(n)]
        ticket = batch.load(self.cfg["ticket"]["path"])
        ticket["sources"] = self.rows
        self.cfg["ticket"] = self.put("ticket.json", ticket)
        plan = self.root / "plan.jsonl"
        plan.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        self.cfg.update(targetcount=n, input_count=n, selected_source_ids=[r["id"] for r in self.rows], plan=batch.ref(plan))
        self.bind()
    def test_exact100_and_denominator_bound(self):
        self.select_count(100)
        batch.validate_common(self.cfg)
        self.assertEqual(batch.selected(self.cfg)[1], 100)
        self.select_count(101)
        with self.assertRaisesRegex(PermissionError, "1..100"):
            batch.validate_common(self.cfg)
    def test_ticket_count_mismatch_and_duplicate_denied(self):
        self.select_count(100)
        with self.assertRaises(PermissionError):
            batch.validate_common(dict(self.cfg, targetcount=99))
        ticket = batch.load(self.cfg["ticket"]["path"])
        ticket["sources"][-1] = ticket["sources"][0]
        self.cfg["ticket"] = self.put("ticket.json", ticket)
        self.bind()
        with self.assertRaises(PermissionError):
            batch.validate_common(self.cfg)

class Core100Tests(unittest.TestCase):
    def test_full100_with_one_model_load_no_truncation(self):
        sys.path.insert(0, str(batch.CORE_DIR))
        import test_ms_unified_novlm_v1 as f
        subject = f.UnifiedTest()
        subject.setUp()
        try:
            rows = []
            for i in range(100):
                sid = "hundred%03d" % i
                v, a = subject.root / (sid + ".mp4"), subject.root / (sid + ".wav")
                v.write_bytes(b"test_video"); a.write_bytes(b"test_audio")
                rows.append(dict(id=sid,split="train",video_path=str(v),audio_path=str(a)))
            subject.rows = rows
            subject.ticket["sources"] = rows
            result = subject.run_worker(limit=100)
            self.assertEqual(result["unique_sources_attempted"], 100)
            self.assertTrue(result["prefetch_drained"])
            self.assertEqual(subject.loader.call_count, 1)
            self.assertEqual(subject.pose.call_count, 9700)
            self.assertEqual(subject.beats.call_count, 100)
            self.assertEqual(len(result["receipts"]), 100)
            self.assertTrue(all(r["status"] == "extracted_rule_draft_unapproved" for r in result["receipts"]))
        finally:
            subject.doCleanups()

if __name__ == "__main__":
    unittest.main()
