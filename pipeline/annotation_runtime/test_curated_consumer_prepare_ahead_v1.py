"""Exactly three isolated /tmp CPU contracts; never loads a GPU/runtime/model."""
from __future__ import annotations

from argparse import Namespace
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import copy
import fcntl
import io
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import curated_consumer_prepare_ahead_v1 as pa

NATIVE = pa.cached.PIPELINE_PATH.parent
native_bar, _, native_visual = pa.pipeline.modules(str(NATIVE), str(NATIVE))


def wait_file(path, process=None):
    deadline = time.monotonic() + 8
    while not Path(path).exists():
        if process is not None and process.poll() is not None:
            raise AssertionError(f"fixture exited before {path}: {process.returncode}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"fixture event missing: {path}")
        time.sleep(.01)


class Fixture:
    def __init__(self, directory):
        self.out = Path(directory).resolve()
        self.args = pa.pipeline.parser().parse_args([
            "consumer", "--manifest", str(self.out / "summary.json"), "--out", str(self.out),
            "--model", str(self.out / "unused-model"), "--ms-source", str(NATIVE),
            "--repo", str(NATIVE), "--limit", "3", "--boundary-review", "off"])
        self.args.hand_roi_root = str(self.out / "roi")
        self.counts, self.verified, self.lock_events = Counter(), [], []
        self.started, self.release = threading.Event(), threading.Event()
        self.bar = Namespace(__file__=str(NATIVE / "ms_bar_annotation.py"),
                             params=lambda args, code: {"unchanged_native_test_parameters": True},
                             code_identity=lambda: {}, validate_row=lambda row: row,
                             runtime_loader=lambda *args: lambda: None,
                             rgb_content=lambda *args: [], need=native_bar.need)
        self.visual = Namespace(bundle_lock=self.lock, digest=pa.pipeline.digest)
        self.bar.prepare_source = self.prepare
        self.bar.verify_prepared = self.verify
        self.bar.process_source = self.process
        self.helper_bar = Namespace(**vars(self.bar))
        self.helper_bar.prepare_source = self.helper_prepare
        self.helper_bar.verify_prepared = self.verify
        self.helper_visual = Namespace(bundle_lock=native_visual.bundle_lock, digest=pa.pipeline.digest)
        self.terminals = []
        # Physical order intentionally differs from source_ordinal order.
        for sid, ordinal in (("case-A", 2), ("case-B", 0), ("case-C", 1)):
            directory = self.out / "sources" / "train" / sid
            roi = self.out / "roi" / sid
            for filename in ("roi_manifest.json", "source_manifest.json", "extract_receipt.json"):
                pa.pipeline.save(roi / filename, {"id": sid, "fixture": filename})
            row = {"id": sid, "split": "train", "batch_id": "test-batch", "source_ordinal": ordinal,
                   "video_path": str(self.out / (sid + ".mp4")), "audio_path": str(self.out / (sid + ".wav"))}
            plan = {"id": sid, "binding": {"id": sid}, "manifest_record": row}
            pa.pipeline.save(directory / "plan.json", plan)
            terminal = {"id": sid, "split": "train", "source_ordinal": ordinal, "status": "ready",
                        "source_binding": plan["binding"], "plan_ref": pa.pipeline.ref(directory / "plan.json"),
                        "roi_ref": pa.pipeline.ref(roi / "roi_manifest.json")}
            pa.pipeline.save(directory / "producer.terminal.json", terminal)
            pa.pipeline.save(directory / "actual_media.json", {"source_identity": {}, "video_probe": {}})
            self.terminals.append(terminal)
        pa.pipeline.save(self.out / "summary.json", {"total": 3, "counts": {"train": 3}, "batches": []})
        (self.out / "ready.jsonl").write_bytes(b"".join(pa.pipeline.raw_json(t) for t in self.terminals))
        style = NATIVE / "ms_dance_style.py"
        self.configuration = {**self.bar.params(self.args, {}),
                              "style_module_sha256": pa.pipeline.sha(style) if style.exists() else None}

    @contextmanager
    def lock(self, directory):
        self.lock_events.append(("before_LOCK_NB", Path(directory).name))
        with native_visual.bundle_lock(directory):
            self.lock_events.append(("locked", Path(directory).name))
            yield

    def prepared(self, row, raw, root, binding, receipt=True):
        root = Path(root)
        native_bar.core.save_once(root / "plan.line.raw.json", raw)
        native_bar.core.save_json(root / "plan.json", row)
        native_bar.core.save_json(root / "binding.json", binding)
        native_bar.core.save_once(root / "rgb" / "frame.jpg", b"unchanged JPEG bytes")
        union = {"binding": binding, "frames": [{"file": "frame.jpg",
                                                "sha256": pa.pipeline.sha(root / "rgb" / "frame.jpg")}]}
        native_bar.core.save_json(root / "rgb" / "manifest.json", union)
        body = {"binding": binding, "source_identity": {"files": {}}, "source_refs": [],
                "raw2d_mask": {"available": False}, "hand_roi": {"available": False},
                "probe": {}, "union_manifest_sha256": pa.pipeline.sha(root / "rgb" / "manifest.json")}
        native_bar.core.save_json(root / "prepared.json", body)
        if receipt:
            native_bar.core.save_json(root / "prepared.receipt.json",
                                      {"binding": binding, "prepared_sha256": pa.pipeline.sha(root / "prepared.json")})
        return body

    def prepare(self, row, raw, root, args, binding, ref_cache):
        self.counts[("main_prepare", row["id"])] += 1
        return self.prepared(row, raw, root, binding)

    def helper_prepare(self, row, raw, root, args, binding, ref_cache):
        self.counts[("helper_prepare", row["id"])] += 1
        if row["id"] == "case-B":
            self.started.set()
            if not self.release.wait(8):
                raise AssertionError("inference/prepare overlap event not reached")
        return self.prepared(row, raw, root, binding)

    def verify(self, root, row, raw, binding):
        self.verified.append(row["id"])
        return native_bar.verify_prepared(root, row, raw, binding)

    def process(self, row, raw, root, args, binding, runtime, ref_cache):
        if (root / "prepared.json").exists():
            self.bar.verify_prepared(root, row, raw, binding)
        else:
            self.bar.prepare_source(row, raw, root, args, binding, ref_cache)
        if row["id"] == "case-A":
            if not self.started.wait(8):
                raise AssertionError("next-source helper did not overlap current source")
            self.lock_events.append(("current_inference_while_future_active", "case-A"))
            self.release.set()
        return {"id": row["id"], "status": "prepared", "binding": binding}

    def job(self, index=1):
        terminal = self.terminals[index]
        return pa.make_job(terminal, self.out / "sources" / "train" / terminal["id"],
                           self.out / "visual" / "shard-0000" / "sources" / terminal["id"],
                           self.args, self.configuration, self.bar, self.visual)

    @contextmanager
    def patches(self):
        with patch.object(pa.pipeline, "modules", return_value=(self.bar, None, self.visual)),              patch.object(pa.pipeline, "load_style_reuse", return_value={}),              patch.object(pa.pipeline, "observe_dance_style", return_value={"status": "pending", "dance_style": "unknown"}),              redirect_stdout(io.StringIO()):
            yield


class EventHelper:
    """Test-only in-process event transport; production always uses exec Popen."""
    def __init__(self, fixture):
        self.fixture, self.pending = fixture, None
        self.responses = queue.Queue()
        self.threads, self.closed = [], False

    def submit(self, job):
        if self.pending is not None:
            raise AssertionError("second future submitted")
        self.pending = job
        def prepare():
            try:
                result = pa.prepare_job(job, self.fixture.helper_bar, self.fixture.helper_visual, {})
            except Exception as error:
                result = {"job_id": job["job_id"], "status": "shared_failure", "error": str(error)}
            self.responses.put(result)
        thread = threading.Thread(target=prepare)
        self.threads.append(thread)
        thread.start()

    def result(self, block):
        try:
            response = self.responses.get(block=block)
        except queue.Empty:
            return None
        if response.get("status") == "shared_failure":
            raise pa.pipeline.SharedFailure(str(response))
        self.fixture.lock_events.append(("helper_locks_released", self.pending["source_id"]))
        return response

    def close(self):
        self.fixture.release.set()
        for thread in self.threads:
            thread.join(8)
            if thread.is_alive():
                raise AssertionError("fixture thread not reclaimed")
        self.closed = True


class PrepareAheadContracts(unittest.TestCase):
    def test_1_native_loop_event_overlap_and_prelock_join(self):
        with tempfile.TemporaryDirectory(prefix="prepare-ahead-case1-", dir="/tmp") as directory:
            fixture = Fixture(directory)
            helper = EventHelper(fixture)
            for stage in ("prepare", "export"):
                other = copy.copy(fixture.args)
                other.stage = stage
                state = pa.PrepareAhead(other, fixture.bar, fixture.visual,
                    helper_factory=lambda: self.fail("Non-run stage started a helper"))
                state.configuration = fixture.configuration
                state.select_next(fixture.out / "sources/train/case-A")
                self.assertIsNone(state.pending)
            unknown = pa.PrepareAhead(fixture.args, fixture.bar, fixture.visual)
            unknown.configuration = fixture.configuration
            unknown.select_next(fixture.out / "sources/train/not-a-real-ready-source")
            self.assertEqual(unknown.ready_seen, set())
            self.assertEqual(unknown.queue_position, 0)
            before = (fixture.visual.bundle_lock, fixture.bar.process_source, fixture.bar.prepare_source,
                      fixture.bar.verify_prepared, pa.pipeline.save, fixture.bar.rgb_content)
            with fixture.patches():
                result = pa.run_consumer(fixture.args, lambda: helper)
            self.assertEqual(result["queue_sources_consumed"], 3)
            self.assertTrue(helper.closed)
            self.assertEqual(fixture.counts[("main_prepare", "case-A")], 1)
            self.assertEqual(fixture.counts[("main_prepare", "case-B")], 0)
            self.assertEqual(fixture.counts[("main_prepare", "case-C")], 0)
            self.assertEqual(fixture.counts[("helper_prepare", "case-B")], 1)
            self.assertEqual(fixture.counts[("helper_prepare", "case-C")], 1)
            self.assertEqual(fixture.verified, ["case-B", "case-C"])
            for sid in ("case-B", "case-C"):
                done = fixture.lock_events.index(("helper_locks_released", sid))
                lock = fixture.lock_events.index(("before_LOCK_NB", sid))
                self.assertLess(done, lock)
            self.assertIn(("current_inference_while_future_active", "case-A"), fixture.lock_events)
            self.assertEqual(before, (fixture.visual.bundle_lock, fixture.bar.process_source, fixture.bar.prepare_source,
                                      fixture.bar.verify_prepared, pa.pipeline.save, fixture.bar.rgb_content))
            protocol = pa.pipeline.load(Path(result["run_root"]) / "protocol.json")
            self.assertEqual(protocol["configuration"], fixture.configuration)
            self.assertEqual(protocol["prepare_ahead_execution_ref"], pa.pipeline.ref(pa.__file__))
            timing = [json.loads(line) for line in (Path(result["run_root"]) / "prepare_ahead.timing.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["event"] == "current_source_wall" for row in timing), 3)
            self.assertEqual(sum(row["event"] == "prepare_ahead_handoff" for row in timing), 2)
            self.assertFalse(list(fixture.out.glob("sources/*/*/*terminal.failed*")))
            self.assertFalse(list(fixture.out.glob("sources/*/*/consumer.terminal.json")))
            # Native limit counts resumed sources and intermediate complete preparations.
            for scenario in ("resume", "complete"):
                bounded = Fixture(Path(directory) / scenario)
                bounded.args.limit = 2
                state = pa.PrepareAhead(bounded.args, bounded.bar, bounded.visual,
                                        lambda: self.fail("prefetched outside native limit"))
                state.configuration = bounded.configuration
                if scenario == "resume":
                    pa.pipeline.save(bounded.out / "sources/train/case-A/consumer.terminal.json", {"status": "resumed"})
                    current = bounded.out / "sources/train/case-B"
                else:
                    job = bounded.job()
                    bounded.prepared(pa.pipeline.load(job["plan_ref"]["path"]),
                                     Path(job["raw_ref"]["path"]).read_bytes(), Path(job["root"]), job["binding"])
                    current = bounded.out / "sources/train/case-A"
                with state.bundle_lock(current):
                    self.assertIsNone(state.pending)
                self.assertIsNone(state.helper)

    def test_2_evidence_mismatch_lock_skip_and_partial_receipt_fidelity(self):
        with tempfile.TemporaryDirectory(prefix="prepare-ahead-case2-", dir="/tmp") as directory:
            fixture = Fixture(directory)
            fixture.release.set()
            job = fixture.job()
            for change in ("source", "binding", "ROI", "raw", "params", "body"):
                bad = copy.deepcopy(job)
                if change == "source":
                    bad["directory"] = str(fixture.out / "sources" / "train" / "case-C")
                elif change == "binding":
                    bad["binding"]["source_binding"] = {"id": "case-C"}
                elif change == "ROI":
                    bad["roi_receipt_refs"][0]["sha256"] = "0" * 64
                elif change == "raw":
                    bad["raw_ref"] = fixture.terminals[2]["plan_ref"]
                elif change == "params":
                    bad["configuration"]["changed"] = True
                else:
                    bad["binding"]["full_source_row_sha256"] = "0" * 64
                bad["job_id"] = pa.pipeline.digest({k: v for k, v in bad.items() if k != "job_id"})
                with self.subTest(change=change), self.assertRaises(ValueError):
                    pa.validate_job(bad, fixture.helper_bar, fixture.helper_visual)
            for lock_path in (job["directory"], job["root"]):
                lock_path = Path(lock_path)
                lock_path.mkdir(parents=True, exist_ok=True)
                with (lock_path / ".lock").open("a+b") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    response = pa.prepare_job(job, fixture.helper_bar, fixture.helper_visual, {})
                    self.assertEqual(response["status"], "skipped_lock")
            self.assertEqual(fixture.counts[("helper_prepare", "case-B")], 0)
            response = pa.prepare_job(job, fixture.helper_bar, fixture.helper_visual, {})
            self.assertEqual(response["status"], "prepared")
            state = pa.PrepareAhead(fixture.args, fixture.bar, fixture.visual)
            state.pending = job
            state.response = {**response, "receipt": {**response["receipt"], "binding": {"wrong": True}}}
            with self.assertRaises(pa.pipeline.SharedFailure):
                with state.bundle_lock(Path(job["directory"])):
                    self.fail("incorrect binding allowed native lock")
            self.assertEqual(fixture.lock_events, [])
            # A matching visual root also joins before its LOCK_NB; unknown paths never join.
            state.response = response
            state.helper = Namespace(pending=job)
            with state.bundle_lock(Path(job["root"])):
                self.assertIsNone(state.pending)
            state.pending = job
            state.helper = Namespace(result=lambda block: self.fail("unknown path consulted future"))
            with state.bundle_lock(fixture.out / "unknown-lock-path"):
                self.assertEqual(state.pending, job)
            # Original verify must still hash union RGB, not trust the helper flag.
            (Path(job["root"]) / "rgb" / "frame.jpg").write_bytes(b"tampered RGB")
            with self.assertRaises(ValueError):
                native_bar.verify_prepared(Path(job["root"]), pa.pipeline.load(job["plan_ref"]["path"]),
                                           Path(job["raw_ref"]["path"]).read_bytes(), job["binding"])
            # Isolated missing-receipt recovery, with the original write-once body.
            partial_job = fixture.job(2)
            root = Path(partial_job["root"])
            plan, raw = pa.pipeline.load(partial_job["plan_ref"]["path"]), Path(partial_job["raw_ref"]["path"]).read_bytes()
            fixture.prepared(plan, raw, root, partial_job["binding"], receipt=False)
            original_body = (root / "prepared.json").read_bytes()
            before = (fixture.bar.prepare_source, fixture.bar.verify_prepared, fixture.bar.process_source, fixture.visual.bundle_lock, pa.pipeline.save)
            def recover(args):
                with fixture.visual.bundle_lock(root):
                    return fixture.bar.verify_prepared(root, plan, raw, partial_job["binding"])
            with fixture.patches(), patch.object(pa.cached, "run_consumer", side_effect=recover):
                pa.run_consumer(fixture.args)
            self.assertEqual((root / "prepared.json").read_bytes(), original_body)
            self.assertTrue((root / "prepared.receipt.json").exists())
            self.assertEqual(fixture.verified[-1], "case-C")
            self.assertEqual(before, (fixture.bar.prepare_source, fixture.bar.verify_prepared, fixture.bar.process_source, fixture.visual.bundle_lock, pa.pipeline.save))
            # A separate partial root whose prepared body disagrees must remain intact.
            bad_root = fixture.out / "partial-mismatch"
            fixture.prepared(plan, raw, bad_root, partial_job["binding"], receipt=False)
            changed = pa.pipeline.load(bad_root / "prepared.json")
            changed["not_original"] = True
            (bad_root / "prepared.json").write_bytes(native_bar.batch.json_bytes(changed))
            snapshot = {p: p.read_bytes() for p in bad_root.rglob("*") if p.is_file()}
            def reject(args):
                return fixture.bar.verify_prepared(bad_root, plan, raw, partial_job["binding"])
            with fixture.patches(), patch.object(pa.cached, "run_consumer", side_effect=reject), self.assertRaises(pa.pipeline.SharedFailure):
                pa.run_consumer(fixture.args)
            self.assertEqual(snapshot, {p: p.read_bytes() for p in bad_root.rglob("*") if p.is_file()})
            self.assertFalse((bad_root / "prepared.receipt.json").exists())
            self.assertFalse(list(fixture.out.glob("sources/*/*/consumer.terminal.json")))
            # Transport rejects an otherwise successful receipt for a different job.
            transport = pa.HelperProcess.__new__(pa.HelperProcess)
            transport.pending, transport.responses = job, queue.Queue()
            transport.responses.put({**response, "job_id": "wrong-source"})
            with self.assertRaises(pa.pipeline.SharedFailure):
                transport.result(False)

    def test_3_exec_helper_sigterm_sigint_reaps_active_subprocess_and_restores(self):
        parent_env = dict(os.environ)
        with patch.dict(os.environ, {"CONDA_PREFIX": "/fixture/miniconda/envs/x",
             "PYTHONPATH": "/fixture/poison", "PYTHONHOME": "/fixture/poison",
             "PATH": "/fixture/miniconda/envs/x/bin:/opt/ms_tools:/usr/local/bin:/usr/bin",
             "LD_LIBRARY_PATH": "/fixture/miniconda/envs/x/lib:/usr/local/cuda/lib64:/usr/lib",
             "CUDA_VISIBLE_DEVICES": "parent-visible"}):
            before = dict(os.environ)
            child_env = pa.cpu_helper_env()
            self.assertEqual(dict(os.environ), before)
            self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "")
            self.assertFalse(any(k.startswith("CONDA_") for k in child_env))
            self.assertNotIn("PYTHONPATH", child_env)
            self.assertNotIn("PYTHONHOME", child_env)
            self.assertEqual(child_env["PATH"], "/opt/ms_tools:/usr/local/bin:/usr/bin")
            self.assertEqual(child_env["LD_LIBRARY_PATH"], "/usr/local/cuda/lib64:/usr/lib")
        self.assertEqual(dict(os.environ), parent_env)
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            pa.main(["--cpu-helper", "0" * 64])
        with tempfile.TemporaryDirectory(prefix="prepare-ahead-case3-", dir="/tmp") as directory:
            for signum in (signal.SIGTERM, signal.SIGINT):
                with self.subTest(signal=signum):
                    root = Path(directory) / str(signum)
                    root.mkdir()
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = "parent-env-unchanged"
                    parent = subprocess.Popen([sys.executable, "-u", __file__, "--fixture-parent", str(root)],
                                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, close_fds=True)
                    try:
                        wait_file(root / "active.pid", parent)
                        wait_file(root / "parent.ready", parent)
                        helper_pid, active_pid = int((root / "helper.pid").read_text()), int((root / "active.pid").read_text())
                        parent.send_signal(signum)
                        wait_file(root / "helper.cleanup", parent)
                        parent.send_signal(signum)  # Cannot interrupt the exact-child wait.
                        stdout, stderr = parent.communicate(timeout=10)
                        self.assertEqual(parent.returncode, 128 + signum, (stdout, stderr))
                        restored = pa.pipeline.load(root / "restored.json")
                        self.assertTrue(all(restored["restored"].values()), restored)
                        self.assertEqual(restored["parent_CVD"], "parent-env-unchanged")
                        self.assertTrue(pa.pipeline.load(root / "helper.env.json")["CVD_empty"])
                        self.assertFalse(pa.pipeline.load(root / "active.env.json")["parent_lock_FD_inherited"])
                        for pid in (helper_pid, active_pid):
                            with self.assertRaises(ProcessLookupError):
                                os.kill(pid, 0)
                        self.assertFalse(list(root.glob("sources/*/*/consumer.terminal.json")))
                    finally:
                        if parent.poll() is None:
                            parent.terminate()
                            parent.communicate(timeout=10)


def fixture_helper(root):
    root = Path(root)
    (root / "helper.pid").write_text(str(os.getpid()))
    pa.pipeline.save(root / "helper.env.json", {"CVD_empty": os.environ.get("CUDA_VISIBLE_DEVICES") == ""})
    def fake_prepare(job, bar, visual, ref_cache):
        # Exercise the SAME subprocess.run cancellation behavior as native probe/decode.
        child = "import os,time,json; from pathlib import Path; p=Path(" + repr(str(root)) + "); "                 "(p/'active.pid').write_text(str(os.getpid())); "                 "fds=[os.readlink('/proc/self/fd/'+f) for f in os.listdir('/proc/self/fd') if f.isdigit() and os.path.exists('/proc/self/fd/'+f)]; "                 "(p/'active.env.json').write_text(json.dumps({'parent_lock_FD_inherited':any(x.endswith('/.lock') for x in fds)})); time.sleep(60)"
        subprocess.run([sys.executable, "-u", "-c", child], check=True, capture_output=True)
        return {"job_id": job["job_id"], "status": "shared_failure", "error": "unexpected fixture completion"}
    try:
        with patch.object(pa, "prepare_job", side_effect=fake_prepare):
            return pa.helper_loop()
    finally:
        (root / "helper.cleanup").write_text("event")
        time.sleep(.2)


def fixture_parent(root):
    fixture = Fixture(root)
    originals = {"lock": fixture.visual.bundle_lock, "save": pa.pipeline.save,
                 "process": fixture.bar.process_source, "prepare": fixture.bar.prepare_source,
                 "verify": fixture.bar.verify_prepared}
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    def consume(args):
        protocol = Path(root) / "runs" / "v0-fixture" / "protocol.json"
        pa.pipeline.save(protocol, {"mode": "consumer", "configuration": fixture.configuration})
        with fixture.visual.bundle_lock(Path(root) / "sources" / "train" / "case-A"):
            wait_file(Path(root) / "active.pid")
            wait_file(Path(root) / "active.env.json")
            (Path(root) / "parent.ready").write_text("event")
            while True:
                time.sleep(.1)
    factory = lambda: pa.HelperProcess([sys.executable, "-u", __file__, "--fixture-helper", str(root)])
    try:
        with fixture.patches(), patch.object(pa.cached, "run_consumer", side_effect=consume):
            pa.run_consumer(fixture.args, factory)
    finally:
        pa.pipeline.save(Path(root) / "restored.json", {
            "parent_CVD": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "restored": {"bundle_lock": fixture.visual.bundle_lock is originals["lock"],
                         "save": pa.pipeline.save is originals["save"],
                         "process": fixture.bar.process_source is originals["process"],
                         "prepare": fixture.bar.prepare_source is originals["prepare"],
                         "verify": fixture.bar.verify_prepared is originals["verify"],
                         "signals": all(signal.getsignal(s) == h for s, h in handlers.items())}})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--fixture-helper":
        sys.exit(fixture_helper(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--fixture-parent":
        fixture_parent(sys.argv[2])
    else:
        unittest.main(verbosity=2)
