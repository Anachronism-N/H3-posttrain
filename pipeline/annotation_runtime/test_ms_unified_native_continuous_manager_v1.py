"""CPU mocks only: no GPU process, resource signal, native FLOCK or lease writes."""
import contextlib
import copy
import importlib.util
import io
import json
import tarfile
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
SPEC = importlib.util.spec_from_file_location("native_under_test", Path(__file__).with_name("ms_unified_native_continuous_manager_v1.py"))
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
        ticket = {"schema": "ms_unified_novlm_ticket_v3", "authorization": "native_parent", "gpu_index": 2,
                  "output_root": self.cfg["out"], "edition": self.cfg["phase"], "ticket_id": "diag3",
                  "batch_sources": 20, "local_cache_root": "/tmp", "max_cache_bytes": 8*m.GIB, "beat_checkpoint": "/fixture/beat.ckpt", "scail_root": str(m.SCAIL), "sources": self.rows}
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
        self.stack.enter_context(patch.object(m.h.subprocess, "check_output", return_value=m.HOSTS[self.cfg["node"]] if host is None else host))
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

    def sealed_summary(self, processed=None, statuses=None):
        ticket = m.ticket_for(self.cfg)
        processed = len(self.rows) if processed is None else processed
        edition = Path(self.cfg["out"]) / ticket["edition"]
        edition.mkdir(parents=True, exist_ok=True)
        batches, outcomes, receipts_all = [], [], []
        for batch_index, offset in enumerate(range(0, processed, 20), 1):
            name = ticket["ticket_id"] + "-b%06d" % batch_index
            batch = edition / "batches" / name
            batch.mkdir(parents=True)
            receipts, index = [], {}
            archive = batch / "batch.tar"
            for i, row in enumerate(self.rows[offset:min(offset + 20, processed)], offset):
                status, draft = (statuses or {}).get(i, ("extracted_rule_draft_unapproved", "draft_generated"))
                good = status == "extracted_rule_draft_unapproved"
                prefix = "cases/" + row["split"] + "/" + row["id"]
                names = ("pose.npz", "fullframe_manifest.json", "beats.json", "draft.json", "receipt.json") if good else ("receipt.json",)
                members = {n: prefix + "/" + n for n in names}
                receipt = {"schema": "ms_unified_novlm_case_receipt_v2", "source_row": row, "status": status,
                    "input_copy_completed": status != "existing_claim_not_reassigned", "semantic_approval": False,
                    "quality_approved": False, "training_eligible": False, "needs_visual_revision": True,
                    "diagnostic_not_production": True, "vlm_called": False, "rgb_or_crops_written": False,
                    "artifact_storage": "archive_member", "artifact_members": members,
                    "locator": {"archive_path": str(archive), "npz_member": members.get("pose.npz"),
                        "fullframe_manifest_member": members.get("fullframe_manifest.json"), "source_case_prefix": prefix}}
                if good:
                    receipt.update(draft_status=draft, frame_count=97, beat_alignment_claimed=False)
                receipts.append(receipt)
                index[row["id"]] = dict(source_row=row, source_case_prefix=prefix, members=members, receipt_member=members["receipt.json"])
                outcomes.append(dict(id=row["id"],split=row["split"],status=status,draft_status=receipt.get("draft_status"),
                    input_copy_completed=receipt["input_copy_completed"],frame_count=receipt.get("frame_count",0),
                    batch_index=batch_index,receipt_member=members["receipt.json"]))
            with tarfile.open(archive, "w") as pack:
                for receipt in receipts:
                    for member in receipt["artifact_members"].values():
                        raw = json.dumps(receipt if member.endswith("/receipt.json") else {"test_only": True}).encode()
                        info = tarfile.TarInfo(member); info.size = len(raw)
                        pack.addfile(info, io.BytesIO(raw))
            sha = m.base.sha(archive)
            counts, statuscounts = m.outcome_counts(receipts)
            commit = {"schema": "ms_unified_novlm_batch_commit_v3", "ticket_id": name, "parent_ticket_id": ticket["ticket_id"],
                "batch_index": batch_index, "selected_range": {"start": offset, "end": offset + len(receipts)}, "edition": ticket["edition"],
                "unique_sources_attempted": len(receipts)-counts["existing_claim_not_reassigned"], "receipts": receipts, "batch_sha256": sha,
                "archive_path": str(archive), "archive_sha256": sha, "archive_bytes": archive.stat().st_size,
                "archive_index": index, "prefetch_drained": True, "quality_approved": False, "beat_alignment_claimed": False,
                "production_complete_claimed": False, "vlm_called": False}
            (batch / "commit.json").write_text(json.dumps(commit))
            batches.append(dict(batch_index=batch_index,commit_path=str(batch/"commit.json"),commit_sha256=m.base.sha(batch/"commit.json"),
                                source_count=len(receipts),counts=counts,statuscounts=statuscounts))
            receipts_all.extend(receipts)
        counts, statuscounts = m.outcome_counts(receipts_all)
        value = {"schema": m.SUMMARY_SCHEMA, "ticket_id": ticket["ticket_id"], "edition": ticket["edition"],
            "status": "completed" if processed == len(self.rows) else "drained_not_complete", "selected_source_ids": self.cfg["selected_source_ids"],
            "processed": processed, "source_count": len(self.rows), "allprefetch_drained": True, "vlm_called": False,
            "quality_approved": False, "beat_alignment_claimed": False, "production_complete_claimed": False,
            "model_init_count": 1, "uncommitted_prefetch": 0, "unprocessed_source_ids": self.cfg["selected_source_ids"][processed:],
            "progress_path": str(edition/"progress.json"), "completed_batches": batches, "all_outcomes": outcomes,
            "counts": counts, "statuscounts": statuscounts, "total_frames": sum(r.get("frame_count",0) for r in receipts_all)}
        summary = Path(self.cfg["out"]) / "summary.json"
        summary.write_text(json.dumps(value))
        return summary, value, batch if batches else None, commit if batches else None

    def test_summary_exact_three_sealed_outcomes(self):
        self.sealed_summary()
        result = m.summaries(self.cfg, Path(self.cfg["out"]))
        self.assertEqual(next(iter(result.values()))["summary"]["total_source_count"], 3)

    def test_summary_missing_source_rejected(self):
        summary, value, batch, commit = self.sealed_summary()
        commit["receipts"].pop()
        (batch / "commit.json").write_text(json.dumps(commit))
        value["completed_batches"][-1]["commit_sha256"] = m.base.sha(batch / "commit.json")
        summary.write_text(json.dumps(value))
        with self.assertRaisesRegex(PermissionError, "exactly cover"):
            m.summaries(self.cfg, Path(self.cfg["out"]))

    def test_summary_fake_copy_counter_rejected(self):
        summary, value, _, _ = self.sealed_summary()
        value["counts"]["generated"] = 2
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
                result = m.load(summary)
                summary.unlink()
                return result
        fake_core = types.SimpleNamespace(load_models=fake_loader, run_ticket=fake_run)
        imports = {name: types.SimpleNamespace(__file__=str(m.CORE_DIR / (name + ".py"))) for name in
                   ("ms_unified_novlm_v2", "dwpose_full_extract", "ms_fullframe_pose2d_draft_v2", "ms_fullframe_pose2d_draft_v1", "ms_pose2d_draft")}
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
        self.assertEqual(m.load(Path(self.cfg["out"]) / "summary.json")["counts"]["generated"], 3)


class CallerGateTests(unittest.TestCase):
    def setUp(self):
        import test_ms_cached_native_gate_v3 as fixture
        from unittest.mock import Mock
        self.subject = fixture.CachedGateTests()
        self.subject.setUp()
        self.addCleanup(self.subject.doCleanups)
        s = self.subject
        self.nvml = types.SimpleNamespace(nvmlInit=Mock(),nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByIndex=Mock(side_effect=lambda gpu: gpu),
            nvmlDeviceGetUUID=Mock(side_effect=lambda gpu: s.uuid(gpu)),
            nvmlDeviceGetComputeRunningProcesses=Mock(side_effect=lambda gpu:
                [types.SimpleNamespace(pid=x["pid"],usedGpuMemory=1024**2) for x in s.apps()]))
        real_queries = m.live_queries.LiveQueries
        for name, value in (("base",fixture.base),("h",s.h),("SELF",s.source),("_NATIVE_GATE",None),
                            ("_LIVE_QUERIES",None),("_GATE_PROFILE",{"calls":0,"seconds":0.0})):
            s.stack.enter_context(patch.object(m,name,value))
        s.stack.enter_context(patch.object(m.live_queries,"LiveQueries",side_effect=lambda base,gpus,uuids:
            real_queries(base,gpus,uuids,module=self.nvml)))
        self.addCleanup(self.close_gate)
    def close_gate(self):
        if m._NATIVE_GATE is not None: m._NATIVE_GATE.close()
        if m._LIVE_QUERIES is not None: m._LIVE_QUERIES.close()
    def test_gate_rebind_retains_real_parent_denial(self):
        s = self.subject
        s.parent["argv"][0] = "/opt/venv_vllm/bin/python"
        s.admission["manager_identity"] = copy.deepcopy(s.parent)
        s.live["manager_identity"] = copy.deepcopy(s.parent)
        s.write_receipts()
        with self.assertRaisesRegex(PermissionError,"parent native manager argv"):
            m.gate(s.cfg,True)
    def test_gate_preserves_full_kernel_ownership_checks(self):
        s = self.subject
        proof = m.gate(s.cfg,True)
        self.assertTrue(proof["caller_owned"])
        reads = [x.read_calls for x in m._NATIVE_GATE._files]
        for _ in range(4): m.gate(s.cfg,False)
        self.assertEqual(reads,[x.read_calls for x in m._NATIVE_GATE._files])
        self.assertEqual(self.nvml.nvmlDeviceGetComputeRunningProcesses.call_count,5)
        self.assertEqual(m._GATE_PROFILE["calls"],5)
        s.lease.st_mtime -= 100
        with self.assertRaisesRegex(PermissionError,"heartbeat"):
            m.gate(s.cfg,False)
    def test_dynamic_worker_uid_denied_after_first_check(self):
        s = self.subject
        m.gate(s.cfg,False)
        s.current["uid"] += 1
        with self.assertRaisesRegex(PermissionError,"identity|uid|UID"):
            m.gate(s.cfg,False)
    def test_before_load_own_worker_gpu_occupancy_denied(self):
        s = self.subject
        s.apps.return_value = [{"pid":s.current["pid"],"uuid":"GPU-fixture"}]
        with self.assertRaisesRegex(PermissionError,"occupied before"):
            m.gate(s.cfg,True)
    def test_gpu_uuid_swap_denied_without_ttl(self):
        s = self.subject
        m.gate(s.cfg,False)
        s.uuid.return_value = "GPU-changed"
        with self.assertRaisesRegex(RuntimeError,"UUID"):
            m.gate(s.cfg,False)


class ContinuousBatchTests(NativeTests):
    def select_count(self,n):
        self.rows = [dict(id="finite%05d"%i,split="train",video_path="/fixture/video/finite%05d.mp4"%i,
                          audio_path="/fixture/audio/finite%05d.wav"%i) for i in range(n)]
        ticket = m.load(self.cfg["ticket"]["path"])
        ticket["sources"] = self.rows
        self.cfg["ticket"] = self.put("ticket.json",ticket)
        plan = self.root/"plan.jsonl"
        plan.write_text("".join(json.dumps(r)+"\n" for r in self.rows))
        self.cfg.update(targetcount=n,input_count=n,selected_source_ids=[r["id"] for r in self.rows],plan=m.ref(plan))
        self.bind()
    def test_exact1000_and_10000_finite_bound(self):
        for n in (1000,10000):
            self.select_count(n)
            m.validate_common(self.cfg)
            self.assertEqual(m.selected(self.cfg)[1],n)
        self.select_count(10001)
        with self.assertRaisesRegex(PermissionError,"1..10000"):
            m.validate_common(self.cfg)
    def test_ticket_schema_and_20_source_batch_required(self):
        for changes in ({"schema":"ms_unified_novlm_ticket_v2"},{"batch_sources":100},{"batch_sources":True},{"stop_file":"relative"}):
            ticket = dict(m.load(self.cfg["ticket"]["path"]),**changes)
            cfg = dict(self.cfg,ticket=self.put("candidate.json",ticket))
            with self.assertRaises(PermissionError): m.ticket_for(cfg)
    def test_ticket_count_mismatch_and_duplicate_denied(self):
        self.select_count(43)
        with self.assertRaises(PermissionError): m.validate_common(dict(self.cfg,targetcount=42))
        ticket = m.load(self.cfg["ticket"]["path"])
        ticket["sources"][-1] = ticket["sources"][0]
        self.cfg["ticket"] = self.put("ticket.json",ticket)
        self.bind()
        with self.assertRaises(PermissionError): m.validate_common(self.cfg)
    def test_multiple_batches_exact_receipts_and_each_archive_hashed_once(self):
        self.select_count(43)
        summary,value,_,_ = self.sealed_summary(statuses={1:("extracted_rule_draft_unapproved","partial"),
            21:("failed_sealed_no_retry",None),22:("existing_claim_not_reassigned",None)})
        real_sha = m.base.sha
        with patch.object(m.base,"sha",wraps=real_sha) as sha, patch.object(m.tarfile,"open",wraps=tarfile.open) as opens:
            result = m.summaries(self.cfg,Path(self.cfg["out"]))
        archives = [str(Path(b["commit_path"]).parent/"batch.tar") for b in value["completed_batches"]]
        self.assertEqual([str(x.args[0]) for x in sha.call_args_list if str(x.args[0]).endswith("batch.tar")],archives)
        self.assertEqual(opens.call_count,3)
        counts = next(iter(result.values()))["counts"]
        self.assertEqual(counts["generated"],41)
        self.assertEqual(counts["partial"],1)
        self.assertEqual(counts["existing_claim_not_reassigned"],1)
    def stop_ticket(self):
        stop = self.root/"STOP"
        ticket = dict(m.load(self.cfg["ticket"]["path"]),stop_file=str(stop))
        self.cfg["ticket"] = self.put("ticket.json",ticket)
        self.bind()
        return stop
    def test_stopped_prefix_drained_without_claiming_full_completion(self):
        self.select_count(43)
        stop = self.stop_ticket(); stop.write_text("stop")
        self.sealed_summary(processed=20)
        result = next(iter(m.summaries(self.cfg,Path(self.cfg["out"])).values()))
        self.assertEqual(result["summary"]["processed_source_count"],20)
        self.assertEqual(result["summary"]["total_source_count"],43)
        self.assertEqual(result["run_status"],"drained_not_complete")
    def test_preexisting_stop_zero_committed_batches(self):
        self.select_count(43)
        self.stop_ticket().write_text("stop")
        self.sealed_summary(processed=0)
        m.summaries(self.cfg,Path(self.cfg["out"]))
    def test_stopped_requires_real_stop_file_and_no_uncommitted_prefetch(self):
        self.select_count(43)
        stop = self.stop_ticket()
        summary,value,_,_ = self.sealed_summary(processed=20)
        with self.assertRaisesRegex(PermissionError,"actual explicit stop"): m.summaries(self.cfg,Path(self.cfg["out"]))
        stop.write_text("stop")
        value["uncommitted_prefetch"] = 1; summary.write_text(json.dumps(value))
        with self.assertRaisesRegex(PermissionError,"uncommitted prefetch"): m.summaries(self.cfg,Path(self.cfg["out"]))
    def test_no_old_batch_or_source_input_rehash(self):
        self.select_count(43)
        self.sealed_summary()
        old = Path(self.cfg["out"])/self.cfg["phase"]/"batches"/"historic"
        old.mkdir(); (old/"commit.json").write_text("not this run"); (old/"batch.tar").write_bytes(b"not a tar")
        real_sha = m.base.sha
        def checked_sha(path):
            self.assertNotIn("historic",str(path))
            self.assertFalse(str(path).startswith("/fixture/video"))
            self.assertFalse(str(path).startswith("/fixture/audio"))
            return real_sha(path)
        with patch.object(m.base,"sha",side_effect=checked_sha): m.summaries(self.cfg,Path(self.cfg["out"]))
    def test_summary_wrong_input_order_model_count_or_completed_flag_denied(self):
        summary,value,_,_ = self.sealed_summary()
        for changes in ({"model_init_count":2},{"processed":2},{"selected_source_ids":list(reversed(self.cfg["selected_source_ids"]))},
                        {"allprefetch_drained":False},{"quality_approved":True},{"counts":dict(value["counts"],generated=4)}):
            with self.subTest(changes=changes):
                summary.write_text(json.dumps(dict(value,**changes)))
                with self.assertRaises(PermissionError): m.summaries(self.cfg,Path(self.cfg["out"]))
    def test_batch_parent_index_path_count_and_sha_denied(self):
        summary,value,batch,commit = self.sealed_summary()
        for changes in ({"parent_ticket_id":"old"},{"batch_index":2},{"selected_range":{"start":1,"end":4}},
                        {"prefetch_drained":False},{"vlm_called":True},{"archive_bytes":1}):
            with self.subTest(changes=changes):
                (batch/"commit.json").write_text(json.dumps(dict(commit,**changes)))
                candidate = copy.deepcopy(value)
                candidate["completed_batches"][0]["commit_sha256"] = m.base.sha(batch/"commit.json")
                summary.write_text(json.dumps(candidate))
                with self.assertRaises(PermissionError): m.summaries(self.cfg,Path(self.cfg["out"]))
    def test_archive_actual_receipt_changed_even_with_new_archive_hash(self):
        summary,value,batch,commit = self.sealed_summary()
        archive = batch/"batch.tar"
        items = []
        with tarfile.open(archive,"r") as pack:
            for info in pack:
                raw = pack.extractfile(info).read()
                if info.name.endswith("/receipt.json"):
                    receipt = json.loads(raw); receipt["quality_approved"] = True
                    raw = json.dumps(receipt).encode()
                items.append((info.name,raw))
        with tarfile.open(archive,"w") as pack:
            for name,raw in items:
                info = tarfile.TarInfo(name); info.size=len(raw); pack.addfile(info,io.BytesIO(raw))
        commit.update(archive_bytes=archive.stat().st_size,archive_sha256=m.base.sha(archive),batch_sha256=m.base.sha(archive))
        (batch/"commit.json").write_text(json.dumps(commit))
        value["completed_batches"][0]["commit_sha256"] = m.base.sha(batch/"commit.json"); summary.write_text(json.dumps(value))
        with self.assertRaisesRegex(PermissionError,"Actual archived"): m.summaries(self.cfg,Path(self.cfg["out"]))


    def test_actual_archive_duplicate_unsafe_and_oversize_receipt_rejected(self):
        summary,value,batch,commit = self.sealed_summary()
        archive = batch/"batch.tar"
        with tarfile.open(archive,"r") as pack:
            original = [(info.name,pack.extractfile(info).read()) for info in pack]
        for kind in ("duplicate","unsafe","oversize"):
            candidate = copy.deepcopy(commit)
            payloads = list(original)
            if kind == "duplicate": payloads.append(original[0])
            elif kind == "unsafe": payloads.append(("cases/../escape",b"bad"))
            else:
                candidate["receipts"][0]["padding"] = "x"*(1024*1024)
                member = candidate["receipts"][0]["artifact_members"]["receipt.json"]
                payloads = [(name,json.dumps(candidate["receipts"][0]).encode() if name == member else raw) for name,raw in payloads]
            with tarfile.open(archive,"w") as pack:
                for name,raw in payloads:
                    info = tarfile.TarInfo(name); info.size=len(raw); pack.addfile(info,io.BytesIO(raw))
            digest = m.base.sha(archive)
            candidate.update(archive_bytes=archive.stat().st_size,archive_sha256=digest,batch_sha256=digest)
            (batch/"commit.json").write_text(json.dumps(candidate))
            final = copy.deepcopy(value)
            final["completed_batches"][0]["commit_sha256"] = m.base.sha(batch/"commit.json")
            summary.write_text(json.dumps(final))
            with self.subTest(kind=kind),self.assertRaisesRegex(PermissionError,"duplicate|unsafe|Oversize"):
                m.summaries(self.cfg,Path(self.cfg["out"]))
    def test_all_explicit_restored_node_gpu_scopes_only(self):
        for node,gpus in m.ALLOWED_GPUS.items():
            for gpu in gpus:
                cfg = copy.deepcopy(self.cfg)
                cfg.update(node=node,gpus=[gpu])
                cfg["holders"][0]["gpu"] = gpu
                cfg["env"].update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH="/opt/curated_music_dependencies_v1" if node == "N2" else m.DEPS)
                ticket = dict(m.load(self.cfg["ticket"]["path"]),gpu_index=gpu)
                cfg["ticket"] = self.put("candidate.json",ticket)
                refs = [cfg[k] for k in ("ticket","plan","core_ref","modelref","resource_previous_safe")]+cfg["holder_restored_refs"]
                cfg["source_sha256"] = {p:m.base.sha(p) for p in m.dependencies()|{r["path"] for r in refs}}
                with self.subTest(node=node,gpu=gpu): m.validate_common(cfg)
        with self.assertRaises(PermissionError): m.validate_common(dict(self.cfg,node="N4"))
    def test_native_budget_reads_v1_and_v2_actual_controller_files(self):
        values = {"/sys/fs/cgroup/memory/memory.usage_in_bytes":"100",
                  "/sys/fs/cgroup/memory/memory.limit_in_bytes":"1000","/proc/meminfo":"MemAvailable: 2 kB\n"}
        with patch.object(Path,"is_file",lambda p:str(p) in values),patch.object(Path,"read_text",lambda p:values[str(p)]):
            self.assertEqual(m.actual_controller_headroom()["headroom"],900)
            values["/sys/fs/cgroup/memory/memory.current"] = "250"
            values["/sys/fs/cgroup/memory/memory.max"] = "900"
            self.assertEqual(m.actual_controller_headroom()["headroom"],650)
            values["/sys/fs/cgroup/memory/memory.max"] = "max"
            self.assertEqual(m.actual_controller_headroom()["headroom"],2048)


class CoreIntegrationTests(NativeTests):
    def integration_fixture(self):
        sys.path.insert(0,str(m.CORE_DIR))
        import test_ms_unified_novlm_v3 as fixtures
        subject = fixtures.ContinuousTest(); subject.setUp()
        self.addCleanup(subject.doCleanups)
        self.rows = subject.rows
        ticket = dict(m.load(self.cfg["ticket"]["path"]),sources=self.rows)
        self.cfg["ticket"] = self.put("ticket.json",ticket)
        plan = self.root/"plan.jsonl"
        plan.write_text("".join(json.dumps(r)+"\n" for r in self.rows))
        self.cfg.update(targetcount=43,input_count=43,selected_source_ids=[r["id"] for r in self.rows],plan=m.ref(plan))
        self.bind()
        return subject,fixtures.worker
    def run_integration(self,stop_after_first=False):
        saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None),saved_path))
        subject,core = self.integration_fixture()
        stop = self.root/"STOP"
        if stop_after_first:
            ticket = dict(m.load(self.cfg["ticket"]["path"]),stop_file=str(stop))
            self.cfg["ticket"] = self.put("ticket.json",ticket); self.bind()
        self.put("config.json",self.cfg)
        events = []
        ownership = dict(admitted=True,caller_owned=True,gpu_uuids=[self.owner["uuid"]],child_identity={"pid":os.getpid()})
        def fake_gate(cfg,before_load=False):
            events.append(before_load); return ownership
        real_run,real_progress,real_sha = core.run_ticket,core.publish_progress,m.base.sha
        def run_actual(ticket,**kwargs):
            self.assertEqual(kwargs["limit"],43)
            return real_run(ticket,probe=subject.probe,decoder=subject.decoder,audio_info=subject.audio_info,**kwargs)
        def progress(path,value):
            real_progress(path,value)
            if stop_after_first and value["processed"] == 20: stop.write_text("STOP")
        def no_old_input_hash(path):
            self.assertNotIn(str(subject.root),str(path))
            return real_sha(path)
        with patch.object(m,"gate",side_effect=fake_gate),patch.object(m,"check_model"), \
             patch.object(m,"module",return_value=core),patch.object(core,"load_models",side_effect=subject.loader), \
             patch.object(core,"run_ticket",side_effect=run_actual) as run,patch.object(core,"publish_progress",side_effect=progress), \
             patch.object(m.base,"apps",return_value=[]),patch.object(m.base,"sha",side_effect=no_old_input_hash), \
             patch.object(m,"local_budget",return_value={"memory":{"headroom":160*m.GIB}}), \
             patch.object(m.shutil,"disk_usage",return_value=types.SimpleNamespace(free=32*m.GIB)), \
             contextlib.redirect_stdout(io.StringIO()):
            m.worker(self.cfg["config_path"])
        result = m.load(Path(self.cfg["out"])/"summary.json")
        self.assertEqual(run.call_count,1)
        self.assertEqual(subject.loader.call_count,1)
        self.assertEqual(events.count(True),1)
        self.assertGreaterEqual(events.count(False),result["processed"])
        self.assertEqual(result["model_init_count"],1)
        self.assertEqual(result["total_frames"],result["processed"]*97)
        self.assertEqual(subject.pose.call_count,result["processed"]*97)
        self.assertTrue(result["allprefetch_drained"])
        self.assertFalse(result["vlm_called"])
        self.assertFalse(result["production_complete_claimed"])
        return result,stop
    def test_real_v3_worker_three_batches_one_model_init_and_dynamic_stages(self):
        result,_ = self.run_integration()
        self.assertEqual(result["status"],"completed")
        self.assertEqual(result["processed"],43)
        self.assertEqual([b["source_count"] for b in result["completed_batches"]],[20,20,3])
        self.assertEqual(result["counts"]["generated"],43)
    def test_real_v3_worker_stop_drained_committed_prefix_only(self):
        result,stop = self.run_integration(stop_after_first=True)
        self.assertTrue(stop.is_file())
        self.assertEqual(result["status"],"drained_not_complete")
        self.assertEqual(result["processed"],20)
        self.assertEqual(result["source_count"],43)
        self.assertEqual(result["uncommitted_prefetch"],0)
        self.assertEqual(result["unprocessed_source_ids"],self.cfg["selected_source_ids"][20:])
        for row in self.rows[20:]:
            claim = Path(self.cfg["out"])/self.cfg["phase"]/"claims"/"sources"/(row["id"]+".json")
            self.assertFalse(claim.exists())


if __name__ == "__main__":
    unittest.main()
