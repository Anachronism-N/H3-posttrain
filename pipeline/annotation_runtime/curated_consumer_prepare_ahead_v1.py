"""One CPU prepare-ahead job around the unchanged cached native consumer."""
from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import importlib.util

RT = Path(__file__).resolve().parent
CACHE_WRAPPER_PATH = RT / "curated_consumer_image_cache_v1.py"
CACHE_WRAPPER_SHA = "69eca5c2feadc4e00cbf4ef3ad51a4dbb0027f17c70f1b46990d3b0b8f7490d0"
CPU_PYTHON = "/opt/venv_eval/bin/python"


def _load_cache_wrapper():
    raw = CACHE_WRAPPER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CACHE_WRAPPER_SHA:
        raise ValueError("Frozen cache wrapper SHA differs")
    spec = importlib.util.spec_from_file_location("_curated_prepare_ahead_cache_v1", CACHE_WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(CACHE_WRAPPER_PATH), "exec"), module.__dict__)
    return module


cached = _load_cache_wrapper()
pipeline = cached.pipeline
EXECUTION_REF = pipeline.ref(__file__)


def stop_signal(signum, frame):
    raise SystemExit(128 + signum)


@contextmanager
def managed_signals():
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for s in previous:
            signal.signal(s, stop_signal)
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def _binding(bar, visual, plan, raw, plan_ref, configuration):
    return {"source_binding": plan["binding"], "plan_sha256": plan_ref["sha256"],
            "full_source_row_sha256": visual.digest(plan),
            "raw_line_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest_record_sha256": visual.digest(plan["manifest_record"]),
            "params_sha256": visual.digest(configuration)}


def make_job(terminal, directory, root, args, configuration, bar, visual):
    raw = Path(terminal["plan_ref"]["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != terminal["plan_ref"]["sha256"]:
        raise ValueError("Prepare-ahead plan SHA changed")
    plan = bar.validate_row(pipeline.load(terminal["plan_ref"]["path"]))
    row = plan["manifest_record"]
    if (terminal["id"] != row["id"] or terminal["split"] != row["split"]
            or terminal["source_ordinal"] != row["source_ordinal"]
            or terminal["source_binding"] != plan["binding"]):
        raise ValueError("Prepare-ahead terminal/source binding differs")
    job = {"execution_ref": EXECUTION_REF, "source_id": row["id"], "directory": str(Path(directory).resolve()),
           "root": str(Path(root).resolve()), "producer_terminal_ref": pipeline.ref(directory / "producer.terminal.json"),
           "ready_terminal": terminal, "plan_ref": terminal["plan_ref"],
           "raw_ref": terminal["plan_ref"], "roi_ref": terminal["roi_ref"],
           "roi_receipt_refs": [pipeline.ref(Path(terminal["roi_ref"]["path"]).with_name(n))
                                for n in ("source_manifest.json", "extract_receipt.json")],
           "native_args": vars(args).copy(), "configuration": configuration,
           "binding": _binding(bar, visual, plan, raw, terminal["plan_ref"], configuration)}
    job["job_id"] = pipeline.digest(job)
    return job


def validate_job(job, bar, visual):
    if pipeline.digest({k: v for k, v in job.items() if k != "job_id"}) != job["job_id"]:
        raise ValueError("Prepare-ahead job identity differs")
    if job["execution_ref"] != EXECUTION_REF or pipeline.ref(__file__) != EXECUTION_REF:
        raise ValueError("Prepare-ahead execution source SHA differs")
    args = Namespace(**job["native_args"])
    out = Path(args.out).resolve()
    terminal = job["ready_terminal"]
    directory = out / "sources" / terminal["split"] / terminal["id"]
    root = out / "visual" / f"shard-{args.shard:04d}" / "sources" / terminal["id"]
    if (str(directory) != job["directory"] or str(root) != job["root"]
            or terminal["source_ordinal"] % args.nshard != args.shard
            or (args.ids and terminal["id"] not in args.ids)
            or terminal["status"] == "failed"
            or job["producer_terminal_ref"]["path"] != str(directory / "producer.terminal.json")
            or job["plan_ref"] != terminal["plan_ref"] or job["raw_ref"] != terminal["plan_ref"]
            or job["roi_ref"] != terminal["roi_ref"]
            or args.hand_roi_root != str(out / "roi")
            or Path(job["roi_ref"]["path"]).resolve() != out / "roi" / terminal["id"] / "roi_manifest.json"):
        raise ValueError("Prepare-ahead source/path/shard assignment differs")
    for ref in [job["producer_terminal_ref"], job["plan_ref"], job["raw_ref"],
                job["roi_ref"], *job["roi_receipt_refs"]]:
        if pipeline.sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Prepare-ahead evidence SHA changed: " + ref["path"])
    if pipeline.load(directory / "producer.terminal.json") != terminal:
        raise ValueError("Prepare-ahead ready terminal differs")
    expected_roi_refs = [str(Path(job["roi_ref"]["path"]).with_name(n))
                         for n in ("source_manifest.json", "extract_receipt.json")]
    if [r["path"] for r in job["roi_receipt_refs"]] != expected_roi_refs:
        raise ValueError("Prepare-ahead ROI receipt assignment differs")
    raw = Path(job["raw_ref"]["path"]).read_bytes()
    plan = bar.validate_row(pipeline.load(job["plan_ref"]["path"]))
    row = plan["manifest_record"]
    if (row["id"] != terminal["id"] or row["split"] != terminal["split"]
            or row["source_ordinal"] != terminal["source_ordinal"]
            or plan["binding"] != terminal["source_binding"]):
        raise ValueError("Prepare-ahead plan/source differs")
    configuration = bar.params(args, bar.code_identity())
    style = Path(bar.__file__).with_name("ms_dance_style.py")
    configuration["style_module_sha256"] = pipeline.sha(style) if style.exists() else None
    if configuration != job["configuration"] or _binding(bar, visual, plan, raw, job["plan_ref"], configuration) != job["binding"]:
        raise ValueError("Prepare-ahead native parameters/full source body binding differs")
    return args, plan, raw


def finish_partial_receipt(bar, original_prepare, root, row, raw, args, binding, ref_cache):
    if (root / "prepared.json").exists() and not (root / "prepared.receipt.json").exists():
        # Re-enter the original write-once tail; never fabricate a receipt for a
        # possibly different prepared body or overwrite interrupted artifacts.
        original_prepare(row, raw, root, args, binding, ref_cache)


def prepare_job(job, bar, visual, ref_cache):
    directory, root = Path(job["directory"]), Path(job["root"])
    started = time.perf_counter()
    prepare_P = 0.0
    try:
        with visual.bundle_lock(directory):
            with visual.bundle_lock(root):
                args, plan, raw = validate_job(job, bar, visual)
                if (directory / "consumer.terminal.json").exists():
                    return {"job_id": job["job_id"], "status": "skipped_terminal", "prepare_P": 0.0}
                p = time.perf_counter()
                if not (root / "prepared.json").exists() or not (root / "prepared.receipt.json").exists():
                    bar.prepare_source(plan, raw, root, args, job["binding"], ref_cache)
                    prepare_P = time.perf_counter() - p
                else:
                    return {"job_id": job["job_id"], "status": "skipped_prepared", "prepare_P": 0.0}
                receipt = pipeline.load(root / "prepared.receipt.json")
                if receipt["binding"] != job["binding"] or receipt["prepared_sha256"] != pipeline.sha(root / "prepared.json"):
                    raise ValueError("Prepare-ahead completed receipt differs")
                receipt_ref = pipeline.ref(root / "prepared.receipt.json")
    except ValueError as error:
        # The native LOCK_NB wrapper preserves BlockingIOError as its cause.
        if isinstance(error.__cause__, BlockingIOError):
            return {"job_id": job["job_id"], "status": "skipped_lock", "prepare_P": 0.0}
        raise
    # A success response is created only after BOTH native locks were released.
    return {"job_id": job["job_id"], "status": "prepared", "receipt": receipt,
            "receipt_ref": receipt_ref, "prepare_P": prepare_P,
            "helper_wall": time.perf_counter() - started}


def helper_loop(instream=None, outstream=None):
    instream, outstream = instream or sys.stdin, outstream or sys.stdout
    ref_cache = {}
    with managed_signals():
        for line in instream:
            job = None
            try:
                job = json.loads(line)
                if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                    raise ValueError("CPU helper must have no CUDA devices")
                args = Namespace(**job["native_args"])
                with redirect_stdout(sys.stderr):
                    bar, _, visual = pipeline.modules(args.repo, args.ms_source)
                    if pipeline.sha(bar.__file__) != cached.BAR_SHA:
                        raise ValueError("Original bar source SHA mismatch")
                    response = prepare_job(job, bar, visual, ref_cache)
            except Exception as error:
                response = {"job_id": job.get("job_id") if isinstance(job, dict) else None,
                            "status": "shared_failure", "error": f"{type(error).__name__}: {error}"}
                outstream.write(json.dumps(response, sort_keys=True) + "\n")
                outstream.flush()
                return 2
            outstream.write(json.dumps(response, sort_keys=True) + "\n")
            outstream.flush()
    return 0


def cpu_helper_env():
    env = os.environ.copy()
    prefixes = [v for k, v in env.items() if k.startswith("CONDA_PREFIX") and v]
    for key in list(env):
        if key.startswith("CONDA_") or key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key)
    for key in ("PATH", "LD_LIBRARY_PATH"):
        if key in env:
            env[key] = ":".join(part for part in env[key].split(":")
                if part and not any(p.lower() in ("conda", "miniconda", "miniconda3", "anaconda", "anaconda3")
                                    for p in Path(part).parts)
                and not any(part == prefix or part.startswith(prefix.rstrip("/") + "/") for prefix in prefixes))
    env.update(CUDA_VISIBLE_DEVICES="",
               MS_VISUAL_CPU_SITEPATH="/opt/venv_eval/lib/python3.11/site-packages")
    return env


class HelperProcess:
    def __init__(self, argv=None):
        env = cpu_helper_env()
        self.process = subprocess.Popen(argv or [CPU_PYTHON, "-B", "-u", str(Path(__file__).resolve()),
                                                "--cpu-helper", EXECUTION_REF["sha256"]],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=None, text=True, bufsize=1, env=env, close_fds=True)
        self.responses = queue.Queue()
        self.pending = None
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(json.loads(line))
        except Exception as error:
            self.responses.put({"status": "shared_failure", "error": str(error)})
        finally:
            self.responses.put({"status": "shared_failure", "error": "CPU helper response stream closed"})

    def submit(self, job):
        if self.pending is not None:
            raise pipeline.SharedFailure("Only one prepare-ahead future is allowed")
        self.pending = job
        try:
            self.process.stdin.write(json.dumps(job, sort_keys=True) + "\n")
            self.process.stdin.flush()
        except Exception as error:
            raise pipeline.SharedFailure("Prepare-ahead helper submission failed: " + str(error)) from error

    def result(self, block):
        if self.pending is None:
            return None
        try:
            response = self.responses.get(block=block)
        except queue.Empty:
            return None
        if response.get("job_id") != self.pending["job_id"] or response.get("status") not in (
                "prepared", "skipped_lock", "skipped_terminal", "skipped_prepared"): 
            raise pipeline.SharedFailure("Prepare-ahead helper failed/identity differs: " + str(response))
        return response

    def close(self):
        # Only the exact owned child. SystemExit in its subprocess.run unwinds
        # the active ffprobe/ffmpeg, including Popen.__exit__'s wait.
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            # A second stop signal must not interrupt exact-child reclamation.
            for s in previous:
                signal.signal(s, signal.SIG_IGN)
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait()
        finally:
            try:
                self.process.stdin.close()
                self.reader.join()
                self.process.stdout.close()
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)


class PrepareAhead:
    def __init__(self, args, bar, visual, helper_factory=HelperProcess):
        self.args, self.bar, self.visual = args, bar, visual
        self.helper_factory = helper_factory
        self.original_lock = visual.bundle_lock
        self.helper = self.pending = self.response = None
        self.configuration = self.run_root = None
        self.visited = set()
        self.ready_seen = set()
        self.queue_position = 0
        self.metrics = {}

    def emit(self, value):
        if self.run_root is not None:
            pipeline.append(self.run_root / "prepare_ahead.timing.jsonl", value)

    def check(self):
        if self.helper is not None and self.pending is not None and self.response is None:
            self.response = self.helper.result(False)

    def wait_path(self, path):
        if self.pending is None or str(path) not in (self.pending["directory"], self.pending["root"]):
            return
        started = time.perf_counter()
        self.check()
        if self.response is None:
            self.response = self.helper.result(True)
        response, job = self.response, self.pending
        if response["status"] == "prepared":
            receipt_path = Path(job["root"]) / "prepared.receipt.json"
            if (response.get("receipt_ref", {}).get("path") != str(receipt_path)
                    or response.get("receipt", {}).get("binding") != job["binding"]
                    or pipeline.ref(receipt_path) != response["receipt_ref"]
                    or pipeline.load(receipt_path) != response["receipt"]):
                raise pipeline.SharedFailure("Prepare-ahead handoff receipt/source binding differs")
        self.emit({"event": "prepare_ahead_handoff", "source_id": job["source_id"],
                   "job_id": job["job_id"], "status": response["status"],
                   "prepare_P": response["prepare_P"], "prefetch_wait_W": time.perf_counter() - started})
        self.helper.pending = self.pending = self.response = None

    def select_next(self, directory):
        if self.args.stage != "run" or self.pending is not None or self.configuration is None:
            return
        out = Path(self.args.out).resolve()
        queue_path = out / "ready.jsonl"
        if not queue_path.exists():
            return
        current_found = False
        anticipated = set(self.ready_seen)
        with queue_path.open("rb") as stream:
            stream.seek(self.queue_position)
            while True:
                begin = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    break
                hint = json.loads(line)
                candidate = out / "sources" / hint["split"] / hint["id"]
                eligible = (hint["source_ordinal"] % self.args.nshard == self.args.shard
                            and (not self.args.ids or hint["id"] in self.args.ids))
                if not current_found:
                    if eligible:
                        anticipated.add(hint["id"])
                    if candidate == directory and eligible:
                        current_found = True
                        self.queue_position = stream.tell()
                        self.ready_seen = set(anticipated)
                    continue
                if not eligible:
                    continue
                anticipated.add(hint["id"])
                if self.args.limit and len(anticipated) > self.args.limit:
                    return
                if (candidate in self.visited or hint["status"] == "failed"
                        or (candidate / "consumer.terminal.json").exists()):
                    continue
                root = out / "visual" / f"shard-{self.args.shard:04d}" / "sources" / hint["id"]
                if (root / "prepared.json").exists() and (root / "prepared.receipt.json").exists():
                    continue
                terminal = pipeline.load(candidate / "producer.terminal.json")
                if terminal != hint:
                    raise ValueError("Prepare-ahead ready hint differs from producer terminal")
                job = make_job(terminal, candidate, root, self.args, self.configuration, self.bar, self.visual)
                if self.helper is None:
                    self.helper = self.helper_factory()
                self.pending = job
                self.helper.submit(job)
                return

    @contextmanager
    def bundle_lock(self, path):
        path = Path(path).resolve()
        sources = Path(self.args.out).resolve() / "sources"
        is_directory = path.parent.parent == sources and (path / "producer.terminal.json").exists()
        matches_future = self.pending is not None and str(path) in (self.pending["directory"], self.pending["root"])
        if not is_directory and not matches_future:
            with self.original_lock(path):
                yield
            return
        started = time.perf_counter()
        try:
            self.check()
            self.wait_path(path)  # BEFORE either of the native LOCK_NB attempts.
        except Exception as error:
            raise pipeline.SharedFailure("Prepare-ahead pre-lock handoff failed: " + str(error)) from error
        with self.original_lock(path):
            if is_directory:
                self.visited.add(path)
                try:
                    self.select_next(path)
                except Exception as error:
                    raise pipeline.SharedFailure("Prepare-ahead scheduling failed: " + str(error)) from error
            try:
                yield
            finally:
                if is_directory:
                    self.emit({"event": "current_source_wall", "source_id": path.name,
                               "current_source_wall": time.perf_counter() - started,
                               **self.metrics.pop(path.name, {"prepare_P": 0.0, "main_verify_V": 0.0})})

    def close(self):
        if self.helper is not None:
            self.helper.close()


def run_consumer(args, helper_factory=HelperProcess):
    if pipeline.ref(__file__) != EXECUTION_REF:
        raise ValueError("Prepare-ahead execution source SHA differs")
    if pipeline.sha(CACHE_WRAPPER_PATH) != CACHE_WRAPPER_SHA:
        raise ValueError("Frozen cache wrapper SHA differs")
    bar, _, visual = pipeline.modules(args.repo, args.ms_source)
    if pipeline.sha(bar.__file__) != cached.BAR_SHA:
        raise ValueError("Original bar source SHA mismatch")
    original_lock, original_save = visual.bundle_lock, pipeline.save
    original_process, original_prepare, original_verify = bar.process_source, bar.prepare_source, bar.verify_prepared
    if getattr(original_process, "_curated_prepare_ahead_v1", False):
        raise ValueError("Prepare-ahead consumer reentry is forbidden")
    state = PrepareAhead(args, bar, visual, helper_factory)

    def save_protocol(path, value):
        path = Path(path)
        if (path.name == "protocol.json" and path.parent.parent.resolve() == Path(args.out).resolve() / "runs"
                and path.parent.name.startswith(f"v{args.shard}-") and value.get("mode") == "consumer"):
            if path.exists():
                raise ValueError("Prepare-ahead requires a new run protocol")
            state.run_root, state.configuration = path.parent, value["configuration"]
            value = {**value, "prepare_ahead_execution_ref": pipeline.ref(__file__),
                     "frozen_cache_execution_ref": pipeline.ref(CACHE_WRAPPER_PATH),
                     "prepare_ahead_protocol": "one_exec_CPU_helper_one_future_native_double_LOCK_NB_v1",
                     "prepare_ahead_CPU_python": CPU_PYTHON,
                     "prepare_ahead_enabled": args.stage == "run",
                     "native_configuration_and_request_bytes_unchanged": True}
        state.check()
        return original_save(path, value)

    def measured_prepare(row, raw, root, native_args, binding, ref_cache):
        started = time.perf_counter()
        try:
            return original_prepare(row, raw, root, native_args, binding, ref_cache)
        finally:
            metrics = state.metrics.setdefault(row["id"], {"prepare_P": 0.0, "main_verify_V": 0.0})
            metrics["prepare_P"] += time.perf_counter() - started

    def measured_verify(root, row, raw, binding):
        try:
            finish_partial_receipt(bar, measured_prepare, root, row, raw, args, binding, {})
        except Exception as error:
            raise pipeline.SharedFailure("Interrupted prepared receipt recovery failed: " + str(error)) from error
        started = time.perf_counter()
        try:
            return original_verify(root, row, raw, binding)  # Entire original verification gate.
        finally:
            metrics = state.metrics.setdefault(row["id"], {"prepare_P": 0.0, "main_verify_V": 0.0})
            metrics["main_verify_V"] += time.perf_counter() - started

    def checked_process(row, raw, root, native_args, binding, runtime, ref_cache):
        state.check()
        if state.pending is not None and str(Path(root).resolve()) == state.pending["root"]:
            raise pipeline.SharedFailure("Native processing reached an unjoined prepare-ahead future")
        try:
            return original_process(row, raw, root, native_args, binding, runtime, ref_cache)
        finally:
            state.check()

    checked_process._curated_prepare_ahead_v1 = True
    visual.bundle_lock, pipeline.save = state.bundle_lock, save_protocol
    bar.process_source, bar.prepare_source, bar.verify_prepared = checked_process, measured_prepare, measured_verify
    try:
        with managed_signals():
            try:
                return cached.run_consumer(args)
            finally:
                state.close()
    finally:
        visual.bundle_lock, pipeline.save = original_lock, original_save
        bar.process_source, bar.prepare_source, bar.verify_prepared = original_process, original_prepare, original_verify


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--cpu-helper":
        if argv[1] != EXECUTION_REF["sha256"] or pipeline.ref(__file__) != EXECUTION_REF:
            raise ValueError("CPU helper execution SHA mismatch")
        return helper_loop()
    args = pipeline.parser().parse_args(argv)
    if args.mode != "consumer":
        raise ValueError("This wrapper only runs native consumer mode")
    return run_consumer(args)


if __name__ == "__main__":
    try:
        result = main()
        if len(sys.argv) == 3 and sys.argv[1] == "--cpu-helper":
            sys.exit(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        print(json.dumps({"status": "worker_stopped", "error": f"{type(error).__name__}: {error}",
                          "remaining_sources_not_marked_failed": isinstance(error, pipeline.SharedFailure),
                          **pipeline.FLAGS}), file=sys.stderr, flush=True)
        sys.exit(2)
