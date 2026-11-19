"""CPU-only /tmp fixtures; no real GPU, process control or production registry I/O."""
import copy
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import ms_vlm_batch_safe_dispatch_v1 as d

v = d.v


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ms-dispatch-cpu-", dir="/tmp")
        self.root = Path(self.tmp.name)
        self.rt = self.root / "runtime"; self.rt.mkdir()
        self.locks = self.root / "locks"; self.locks.mkdir()
        self.stack = ExitStack()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.stack.close)
        self.v4 = self.rt / v.SELF.name; self.v4.write_text("synthetic frozen v4")
        self.dispatcher = self.rt / d.SELF.name; self.dispatcher.write_text("synthetic frozen dispatcher")
        self.basefile = self.rt / "resource_manager.py"; self.basefile.write_text("synthetic lifecycle")
        self.prevproducer = self.rt / "ms_fullframe_vlm_postprocess_v3.py"; self.prevproducer.write_text("synthetic predecessor")
        for obj, name, value in ((d, "RT", self.rt), (d, "ROOT", self.root), (d, "TMP", self.locks),
            (d, "SELF", self.dispatcher), (d, "REGISTRY", self.rt / "registry.json"), (d, "V4_SHA", v.base.sha(self.v4)),
            (v, "RT", self.rt), (v, "ROOT", self.root), (v, "SELF", self.v4),
            (v, "BASE_SHA", v.base.sha(self.basefile)), (v.base, "__file__", str(self.basefile))):
            self.stack.enter_context(patch.object(obj, name, value))
        self.stack.enter_context(patch.object(v, "dependencies", return_value={str(self.v4), str(self.basefile)}))
        self.stack.enter_context(patch.object(v.base, "selected_sources", v.selected))
        self.stack.enter_context(patch.object(v.base, "headroom"))
        self.phase = v.PREFIX + "cpu4"
        self.previous_phase = "ms_fullframe_vlm_postprocess_v3_cpu"
        self.queue = self.root / "queues" / self.previous_phase; self.queue.mkdir(parents=True)
        self.out = self.root / self.previous_phase; self.out.mkdir()
        self.plan = self.rt / "plan.jsonl"
        self.rows = self.make_rows(4)
        self.put_plan(self.plan, self.rows)
        oldplan = self.rt / "oldplan.jsonl"; self.put_plan(oldplan, [dict(id="Old000")])
        self.ffmpeg = self.rt / "ffmpeg"; self.ffmpeg.write_text("synthetic executable"); self.ffmpeg.chmod(0o700)
        self.modelref = self.put(self.rt / "model.json", dict(model_identity={}))
        self.modelref["model_identity_sha256"] = "d" * 64
        self.holders = [self.identity(103 + i, gpu=i, uuid="GPU-synthetic-%d" % i) for i in range(2)]
        self.cp = self.rt / (self.previous_phase + "_config.json")
        self.previous = dict(schema="ms_fullframe_vlm_postprocess_native_v3", phase=self.previous_phase,
            node="N3", gpus=[0, 1], config_path=str(self.cp), cwd="/tmp", env={}, ffmpeg_path=str(self.ffmpeg),
            holders=self.holders, holder_restored_refs=[], resource_safe_refs=[], input_count=1, targetcount=1,
            selected_source_ids=["Old000"], modelref=self.modelref, new_stage_authorized=True, release_authorized=True,
            out=str(self.out), queue=str(self.queue), plan=v.ref(oldplan), producer=v.ref(self.prevproducer),
            resource_manager_ref=v.ref(self.prevproducer), source_sha256={str(self.basefile):v.base.sha(self.basefile), str(self.modelref["path"]):self.modelref["sha256"]},
            argv=[v.PYTHON, "-B", str(self.prevproducer), "worker", "--config", str(self.cp)],
            _stale="forbidden", reuse_existing_results=True, previous_generation_gate={"forbidden": True})
        self.put(self.cp, self.previous)
        self.manager = self.identity(101)
        self.manager["argv"] = [v.PYTHON, "-B", str(self.prevproducer), "--config", str(self.cp),
            "--config-sha", v.base.sha(self.cp), "--manager-sha", v.BASE_SHA]
        self.worker = self.identity(102)
        self.live = dict(phase=self.previous_phase, targetcount=1, argv=self.previous["argv"],
                         manager_identity=self.manager, child_identity=self.worker)
        self.put(self.queue / "live.json", self.live)
        self.replay = {str(self.out / "summary.json"): dict(summary=dict(stage="run"), processed=1)}
        self.put(self.out / "summary.json", self.replay[str(self.out / "summary.json")])
        self.exit_value = dict(exit_code=0, source_unchanged=True, summaries=self.replay,
            **{k:self.previous[k] for k in ("phase", "producer", "plan", "targetcount", "input_count")})
        self.put(self.queue / "worker_exit.json", self.exit_value)
        self.safe = dict(phase=self.previous_phase, **{k:True for k in d.SAFE_FLAGS})
        self.put(self.queue / "resources_safe.json", self.safe)
        for holder in self.holders:
            self.put(self.queue / ("holder_restored_gpu%d.json" % holder["gpu"]),
                     dict(gpu=holder["gpu"], uuid=holder["uuid"], holder_identity=holder))
        self.jobpath = self.rt / (self.phase + "_dispatch.json")
        self.allocationpath = self.rt / "allocation.json"
        self.job = dict(schema="ms_vlm_batch_safe_dispatch_job_v1", job_path=str(self.jobpath),
            previous_config=v.ref(self.cp), previous_live_ref=v.ref(self.queue / "live.json"),
            producer_ref=v.ref(self.v4), dispatcher_ref=v.ref(self.dispatcher), phase=self.phase, node="N3", gpus=[0, 1], host=v.HOSTS["N3"],
            plan=v.ref(self.plan), source_count=4, batch_sources=4, allocation_ref={}, dispatcher_task_handle="SyntheticTask")
        self.freeze()
        self.worker_alive = False
        self.manager_changed = False
        self.stack.enter_context(patch.object(v.base, "identity", side_effect=self.current_identity))
        self.stack.enter_context(patch.object(v.base, "apps", return_value=[dict(pid=h["pid"], uuid=h["uuid"]) for h in self.holders]))
        self.stack.enter_context(patch.object(v.base, "uuid_for", side_effect=lambda g:self.holders[g]["uuid"]))
        self.lock_snapshot = self.stack.enter_context(patch.object(v.h, "lock_snapshot", return_value=dict(held=False)))
        self.lease = self.stack.enter_context(patch.object(v.base, "lease_snapshot", return_value=None))
        self.headroom = self.stack.enter_context(patch.object(v, "actual_controller_headroom", return_value=dict(headroom=160 * 2**30)))
        self.stack.enter_context(patch.object(v.subprocess, "check_output", return_value=v.HOSTS["N3"]))
        self.replay_mock = Mock(side_effect=lambda *args:copy.deepcopy(self.replay))
        self.stack.enter_context(patch.object(v, "module", return_value=SimpleNamespace(summaries=self.replay_mock)))
        self.model = self.stack.enter_context(patch.object(v.base, "check_model"))
        self.exec = self.stack.enter_context(patch.object(d.os, "execv"))
        self.forbidden = [self.stack.enter_context(patch.object(obj, name, side_effect=AssertionError("Forbidden production action")))
            for obj, name in ((v, "runtime"), (v.subprocess, "Popen"), (os, "kill"), (v.base.signal, "pidfd_send_signal"))]

    def identity(self, pid, gpu=None, uuid=None):
        value = dict(pid=pid, uid=os.getuid(), argv=[v.PYTHON], start_ticks=pid * 10,
                     cgroup="synthetic-cgroup", CVD=[], exe="synthetic-python", sig_ign=0, sig_blk=0)
        if gpu is not None:
            value.update(gpu=gpu, uuid=uuid, argv=["/opt/venv_eval/bin/python", "-u", str(v.base.HOLDER)],
                         CVD=["CUDA_VISIBLE_DEVICES=%d" % gpu])
        return value

    def current_identity(self, pid):
        if pid == 101:
            value = copy.deepcopy(self.manager)
            if self.manager_changed: value["start_ticks"] += 1
            return value
        if pid == 102:
            if self.worker_alive: return self.worker
            raise FileNotFoundError("Synthetic predecessor worker has exited")
        if pid in (103, 104): return self.holders[pid - 103]
        return self.identity(pid)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(v.h.bench.encoded(value))
        return v.ref(path)

    def put_plan(self, path, rows):
        path.write_bytes(b"".join(v.h.bench.encoded(r) for r in rows))

    def make_rows(self, count):
        rows = []
        for i in range(count):
            id_ = "Fresh%03d" % i
            manifest_path = self.rt / (id_ + "_manifest.json")
            video = self.root / (id_ + ".mp4")
            manifest = dict(schema="ms_fullframe_pose2d_manifest_v1", id=id_, split="train", video_path=str(video),
                video_sha256="a" * 64, frame_count=2, frame_indices=[0, 1], times=[0., .04],
                source_timestamps=[10., 10.04], pts_origin=10., width=1280, height=704, duration_seconds=.08,
                **dict(list(v.FLAGS.items())[:4]))
            manifest_ref = self.put(manifest_path, manifest)
            binding = dict(schema="ms_fullframe_pose2d_binding_v1", observations_sha256="b" * 64)
            payload = dict(schema=binding["schema"], fullframe_manifest=manifest, observations_sha256=binding["observations_sha256"])
            binding["sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            draft_ref = self.put(self.rt / (id_ + "_draft.json"), dict(schema="ms_fullframe_pose2d_rule_draft_v2",
                id=id_, split="train", ms_text="Observed image-plane arm pose", **dict(list(v.FLAGS.items())[:4]),
                source=dict(fullframe_manifest_path=str(manifest_path), video_path=str(video), video_sha256="a" * 64, fullframe_binding=binding)))
            rows.append(dict(id=id_, split="train", video_path=str(video), rule_draft_ref=draft_ref, fullframe_manifest_ref=manifest_ref))
        return rows

    def freeze(self, allocation_rows=False):
        self.job["plan"] = v.ref(self.plan)
        allocation = {k:self.job[k] for k in ("phase", "node", "gpus", "source_count", "batch_sources")}
        allocation.update(schema="ms_vlm_batch_exclusive_allocation_v1", selected_source_ids=[r["id"] for r in self.rows])
        allocation["rows" if allocation_rows else "plan_refs"] = self.rows if allocation_rows else [self.job["plan"]]
        self.job["allocation_ref"] = self.put(self.allocationpath, allocation)
        self.jobref = self.put(self.jobpath, self.job)
        old = dict(node="N3", gpus=[0, 1], phase=self.previous_phase, config_path=str(self.cp), expected_running=True, task_handle="PreviousTask")
        pending = dict(node="N3", gpus=[0, 1], phase=self.phase, config_path=str(self.rt / (self.phase + "_config.json")),
            expected_running=False, task_handle=self.job["dispatcher_task_handle"], dispatcher_job_ref=self.jobref,
            plan_ref=self.job["plan"], sources=self.job["source_count"], batch_sources=self.job["batch_sources"])
        self.history = dict(node="N3", gpus=[0, 1], phase="historical", expected_running=False, audit="retained")
        self.registry = dict(schema="ms_vlm_postprocess_jobs_registry_v1", jobs=[self.history, old, pending], extra_audit="unchanged")
        self.put(d.REGISTRY, self.registry)

    def run_dispatch(self):
        d.dispatch(str(self.jobpath), self.jobref["sha256"])

    def assert_blocked(self, error=(ValueError, FileNotFoundError, BlockingIOError)):
        before = d.REGISTRY.read_bytes()
        with self.assertRaises(error): self.run_dispatch()
        self.exec.assert_not_called(); self.model.assert_not_called()
        self.assertEqual(before, d.REGISTRY.read_bytes())
        for action in self.forbidden: action.assert_not_called()
        self.assertFalse(any(self.root.rglob("*.claim.json")))

    def test_success_retains_history_two_holders_shared_safe(self):
        self.run_dispatch()
        cp = self.rt / (self.phase + "_config.json"); cfg = v.load(cp)
        self.assertEqual(cfg["holders"], self.holders)
        self.assertEqual(cfg["resource_safe_refs"][0], cfg["resource_safe_refs"][1])
        self.assertEqual([Path(r["path"]).name for r in cfg["holder_restored_refs"]], ["holder_restored_gpu0.json", "holder_restored_gpu1.json"])
        self.assertEqual(set(cfg), set(d.CLONE) | {"batch_sources", "source_sha256"})
        registry = v.load(d.REGISTRY)
        self.assertEqual(registry["jobs"][0], self.history)
        self.assertEqual(registry["extra_audit"], "unchanged")
        self.assertFalse(registry["jobs"][1]["expected_running"])
        self.assertEqual(registry["jobs"][1]["worker_status"], "completed_native_safe_fullreplay")
        self.assertTrue(registry["jobs"][2]["expected_running"])
        self.assertEqual(registry["jobs"][2]["task_handle"], "SyntheticTask")
        self.exec.assert_called_once_with(v.PYTHON, [v.PYTHON, "-B", str(v.SELF), "--config", str(cp), "--config-sha", v.base.sha(cp), "--manager-sha", v.BASE_SHA])
        self.model.assert_called_once(); self.replay_mock.assert_called_once()
        self.headroom.assert_called_once()
        self.assertEqual(self.lease.call_count, 2)
        for action in self.forbidden: action.assert_not_called()

    def test_eight_and_allocation_rows_success(self):
        self.rows = self.make_rows(8); self.put_plan(self.plan, self.rows)
        self.job.update(source_count=8, batch_sources=8)
        self.freeze(allocation_rows=True); self.run_dispatch()
        self.assertEqual(v.load(self.rt / (self.phase + "_config.json"))["targetcount"], 8)

    def test_wrong_job_sha(self):
        self.jobref["sha256"] = "0" * 64; self.assert_blocked()

    def test_wrong_producer_ref(self):
        self.job["producer_ref"] = self.job["previous_config"]; self.freeze(); self.assert_blocked()

    def test_wrong_self_path(self):
        self.job["job_path"] = str(self.rt / "other.json"); self.freeze(); self.assert_blocked()

    def test_unknown_pair(self):
        self.job["gpus"] = [2, 3]; self.freeze(); self.assert_blocked()

    def test_wrong_host(self):
        self.job["host"] = "synthetic-wrong"; self.freeze(); self.assert_blocked()

    def test_wrong_phase(self):
        self.job["phase"] = "old_output"; self.freeze(); self.assert_blocked()

    def test_wrong_count(self):
        self.job["source_count"] = 8; self.freeze(); self.assert_blocked()

    def test_wrong_batch(self):
        self.job["batch_sources"] = 3; self.freeze(); self.assert_blocked()

    def test_plan_count_mismatch(self):
        self.rows = self.rows[:3]; self.put_plan(self.plan, self.rows); self.freeze(); self.assert_blocked()

    def test_casefold_duplicate(self):
        self.rows[1]["id"] = self.rows[0]["id"].lower(); self.put_plan(self.plan, self.rows); self.freeze(); self.assert_blocked()

    def test_plan_changed(self):
        self.plan.write_bytes(b"changed"); self.assert_blocked()

    def test_allocation_changed(self):
        self.allocationpath.write_text("changed"); self.assert_blocked()

    def test_allocation_scope_wrong(self):
        allocation = v.load(self.allocationpath); allocation["node"] = "N2"
        self.job["allocation_ref"] = self.put(self.allocationpath, allocation)
        self.jobref = self.put(self.jobpath, self.job)
        self.registry["jobs"][2]["dispatcher_job_ref"] = self.jobref; self.put(d.REGISTRY, self.registry)
        self.assert_blocked()

    def test_other_plan_collision(self):
        self.registry["jobs"].insert(0, dict(phase="other", expected_running=False, plan_ref=self.job["plan"]))
        self.put(d.REGISTRY, self.registry); self.assert_blocked()

    def test_registry_missing_pending(self):
        self.registry["jobs"].pop(); self.put(d.REGISTRY, self.registry); self.assert_blocked()

    def test_registry_pending_wrong_plan(self):
        self.registry["jobs"][2]["plan_ref"] = self.previous["plan"]; self.put(d.REGISTRY, self.registry); self.assert_blocked()

    def test_registry_duplicate_current(self):
        self.registry["jobs"].append(copy.deepcopy(self.registry["jobs"][1])); self.put(d.REGISTRY, self.registry); self.assert_blocked()

    def test_registry_current_changed(self):
        self.registry["jobs"][1]["config_path"] = str(self.jobpath); self.put(d.REGISTRY, self.registry); self.assert_blocked()

    def test_failure_even_with_safe(self):
        self.put(self.queue / "failure.json", dict(error="synthetic")); self.assert_blocked()

    def test_not_safe_bounded_wait(self):
        (self.queue / "resources_safe.json").unlink()
        with patch.object(d.time, "monotonic", side_effect=[0., 7201.]): self.assert_blocked()

    def test_not_safe_each_flag(self):
        for key in d.SAFE_FLAGS:
            with self.subTest(key=key):
                safe = dict(self.safe, **{key:False}); self.put(self.queue / "resources_safe.json", safe)
                job, _, previous, live, _ = d.check_job(str(self.jobpath), self.jobref["sha256"])
                with self.assertRaises(ValueError): d.wait_safe(job, previous, live)
        self.model.assert_not_called(); self.exec.assert_not_called()
        for action in self.forbidden: action.assert_not_called()

    def test_safe_wrong_phase(self):
        self.put(self.queue / "resources_safe.json", dict(self.safe, phase="wrong")); self.assert_blocked()

    def test_nonzero_exit(self):
        self.put(self.queue / "worker_exit.json", dict(self.exit_value, exit_code=1)); self.assert_blocked()

    def test_wrong_replay(self):
        self.replay = {"wrong": "replay"}; self.assert_blocked()

    def test_source_changed(self):
        self.put(self.queue / "worker_exit.json", dict(self.exit_value, source_unchanged=False)); self.assert_blocked()

    def test_exit_scope_wrong(self):
        self.put(self.queue / "worker_exit.json", dict(self.exit_value, targetcount=2)); self.assert_blocked()

    def test_worker_alive(self):
        self.worker_alive = True; self.assert_blocked()

    def test_manager_changed(self):
        self.manager_changed = True; self.assert_blocked()

    def test_live_receipt_changed(self):
        self.put(self.queue / "live.json", dict(self.live, phase="changed")); self.assert_blocked()

    def test_held_resource_lock(self):
        self.lock_snapshot.return_value = dict(held=True); self.assert_blocked()

    def test_active_lease(self):
        self.lease.side_effect = RuntimeError("active lease"); self.assert_blocked(RuntimeError)

    def test_fresh_headroom_below_original_budget(self):
        self.headroom.return_value = dict(headroom=160 * 2**30 - 1); self.assert_blocked()

    def test_holder_changed(self):
        self.holders[0] = dict(self.holders[0], start_ticks=999); self.assert_blocked(RuntimeError)

    def test_duplicate_once_receipt_prevents_restart(self):
        self.put(self.rt / (self.phase + "_dispatch_started.json"), dict(automatic_restart=False)); self.assert_blocked(FileExistsError)

    def test_local_dispatcher_lock_held(self):
        with (self.locks / "ms-vlm-dispatch-N3-0-1.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); self.assert_blocked(BlockingIOError)

    def test_wrong_previous_ref_sha(self):
        self.job["previous_config"]["sha256"] = "0" * 64; self.freeze(); self.assert_blocked()

    def test_wrong_previous_live_ref_path(self):
        self.job["previous_live_ref"] = self.job["previous_config"]; self.freeze(); self.assert_blocked()

    def test_wrong_physical_node(self):
        with patch.object(v.subprocess, "check_output", return_value="synthetic-other-host"):
            self.assert_blocked()

    def test_new_input_ref_sha_wrong_before_once(self):
        self.rows[0]["rule_draft_ref"]["sha256"] = "0" * 64
        self.put_plan(self.plan, self.rows); self.freeze(); self.assert_blocked()
        self.assertFalse((self.rt / (self.phase + "_dispatch_started.json")).exists())

    def test_frozen_dependency_changed(self):
        self.basefile.write_text("changed lifecycle"); self.assert_blocked()

    def test_manager_zombie(self):
        self.manager["state"] = "Z"
        self.put(self.queue / "live.json", dict(self.live, manager_identity=self.manager))
        self.job["previous_live_ref"] = v.ref(self.queue / "live.json"); self.freeze(); self.assert_blocked()

    def test_full_replay_exception(self):
        self.replay_mock.side_effect = ValueError("Full replay detects altered raw/result")
        self.assert_blocked()

    def test_dispatcher_wrong_self_path(self):
        self.job["dispatcher_ref"] = v.ref(self.v4); self.freeze(); self.assert_blocked()

    def test_dispatcher_changed_before_check(self):
        self.dispatcher.write_text("changed wrapper"); self.assert_blocked()

    def test_dispatcher_changed_at_check_job_tail(self):
        original = d.registry_check
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            self.dispatcher.write_text("changed while checking job")
            return result
        with patch.object(d, "registry_check", side_effect=changed): self.assert_blocked()

    def test_dispatcher_changed_after_model_check(self):
        self.model.side_effect = lambda cfg:self.dispatcher.write_text("changed after model identity check")
        before = d.REGISTRY.read_bytes()
        with self.assertRaises(ValueError): self.run_dispatch()
        self.exec.assert_not_called()
        self.assertEqual(before, d.REGISTRY.read_bytes())

    def other_pending(self, collision=False):
        phase = v.PREFIX + "future_other"
        plan = self.rt / "other_future_plan.jsonl"
        rows = self.rows if collision else [dict(id="Other%03d" % i) for i in range(4)]
        self.put_plan(plan, rows)
        entry = dict(phase=phase, node="N2", gpus=[2, 3], expected_running=False, sources=4, batch_sources=4,
                     config_path=str(self.rt / (phase + "_config.json")), plan_ref=v.ref(plan))
        self.registry["jobs"].append(entry); self.put(d.REGISTRY, self.registry)
        return entry

    def test_other_frozen_pending_future_config_allowed(self):
        pending = copy.deepcopy(self.other_pending())
        self.run_dispatch()
        self.exec.assert_called_once()
        self.assertEqual(v.load(d.REGISTRY)["jobs"][-1], pending)

    def test_other_frozen_pending_source_collision(self):
        self.other_pending(collision=True); self.assert_blocked()

    def test_ordinary_missing_config_fails(self):
        self.registry["jobs"].append(dict(phase="ordinary_history", expected_running=False,
                                         config_path=str(self.rt / "missing_ordinary_config.json")))
        self.put(d.REGISTRY, self.registry); self.assert_blocked()

    def test_other_pending_missing_frozen_plan_fails(self):
        entry = self.other_pending(); entry.pop("plan_ref")
        self.put(d.REGISTRY, self.registry); self.assert_blocked()

    def test_other_pending_expected_running_not_false_fails(self):
        entry = self.other_pending(); entry["expected_running"] = None
        self.put(d.REGISTRY, self.registry); self.assert_blocked()

    def test_other_pending_plan_sha_changed_fails(self):
        entry = self.other_pending(); Path(entry["plan_ref"]["path"]).write_text("changed future plan")
        self.assert_blocked()

    def test_other_pending_wrong_future_path_fails(self):
        entry = self.other_pending(); entry["config_path"] = str(self.rt / "wrong_future_config.json")
        self.put(d.REGISTRY, self.registry); self.assert_blocked()

    def assert_after_replay_blocked(self, mutate):
        before = d.REGISTRY.read_bytes()
        self.model.side_effect = lambda cfg:mutate()
        with self.assertRaises(ValueError): self.run_dispatch()
        self.exec.assert_not_called(); self.replay_mock.assert_called_once()
        self.assertEqual(before, d.REGISTRY.read_bytes())
        for action in self.forbidden: action.assert_not_called()
        self.assertFalse(any(self.root.rglob("*.claim.json")))

    def test_worker_exit_changed_after_replay_fails(self):
        self.assert_after_replay_blocked(lambda:self.put(self.queue / "worker_exit.json", dict(self.exit_value, exit_code=1)))

    def test_summary_changed_after_replay_fails(self):
        self.assert_after_replay_blocked(lambda:self.put(self.out / "summary.json", dict(changed=True)))

    def test_safe_changed_after_replay_fails(self):
        self.assert_after_replay_blocked(lambda:self.put(self.queue / "resources_safe.json", dict(self.safe, leases_restored=False)))

    def test_failure_appears_after_replay_fails(self):
        self.assert_after_replay_blocked(lambda:self.put(self.queue / "failure.json", dict(error="late failure")))

    def test_manager_changes_after_replay_fails(self):
        self.assert_after_replay_blocked(lambda:setattr(self, "manager_changed", True))

    def test_summary_changed_during_replay_fails(self):
        def changed(*args):
            self.put(self.out / "summary.json", dict(changed=True))
            return copy.deepcopy(self.replay)
        self.replay_mock.side_effect = changed; self.assert_blocked()

    def test_final_exec_guard_closure_changed_during_registry_promotion(self):
        original = d.registry_check
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get("promote"):
                self.put(self.queue / "worker_exit.json", dict(self.exit_value, exit_code=1))
            return result
        with patch.object(d, "registry_check", side_effect=changed):
            with self.assertRaises(ValueError): self.run_dispatch()
        self.exec.assert_not_called()
        for action in self.forbidden: action.assert_not_called()

    def test_final_exec_guard_self_changed_during_registry_promotion(self):
        original = d.registry_check
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get("promote"): self.dispatcher.write_text("changed during registry promotion")
            return result
        with patch.object(d, "registry_check", side_effect=changed):
            with self.assertRaises(ValueError): self.run_dispatch()
        self.exec.assert_not_called()
        for action in self.forbidden: action.assert_not_called()

    def test_registry_cas_rechecked_after_validation(self):
        def changed(cfg):
            self.registry["jobs"][1]["expected_running"] = False
            self.put(d.REGISTRY, self.registry)
        self.model.side_effect = changed
        with self.assertRaises(ValueError): self.run_dispatch()
        self.exec.assert_not_called()
        self.assertFalse(v.load(d.REGISTRY)["jobs"][2]["expected_running"])
        self.assertTrue((self.rt / (self.phase + "_dispatch_started.json")).exists())
        self.assertFalse(any(self.root.rglob("*.claim.json")))
        for action in self.forbidden: action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
