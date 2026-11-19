"""CPU-only real gate tests: GPU queries, physical locks/leases and signals are mocked."""
import ast
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def functions(path, names, globals_dict):
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, str(path), "exec"), globals_dict)
    return globals_dict


m = module("cached_gate_under_test", RT / "ms_cached_native_gate_v2.py")
base = module("cached_gate_real_base", RT / "resource_manager.py")
strict_globals = functions(RT / "visual_batch_fixed3_benchmark_v1.py", {"need", "strict"}, {"json": json})
lock_globals = functions(RT / "bench_manager_driver_v2.py", {"kernel_lock_matches"}, {"os": os})


class CachedGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "queue"
        self.queue.mkdir()
        self.source = self.root / "new_caller.py"
        self.source.write_bytes(b"# fixed new caller fixture\n" * 300)
        source_sha = base.sha(self.source)
        self.cfg = {"node": "N6", "gpus": [2], "queue": str(self.queue), "out": str(self.root / "out"),
            "phase": "ms_unified_native_v1_cached_test", "targetcount": 100,
            "_config_path": str(self.root / "config.json"), "_config_sha": "f" * 64,
            "env": {"CUDA_VISIBLE_DEVICES": "2"},
            "resource_manager_ref": {"path": str(self.source), "sha256": source_sha},
            "producer": {"path": str(self.source), "sha256": source_sha},
            "plan": {"path": str(self.root / "plan.jsonl"), "sha256": "d" * 64},
            "source_sha256": {str(self.source): source_sha, base.__file__: m.BASE_SHA},
            "modelref": {"model_identity_sha256": "e" * 64}, "holders": [{"gpu": 2, "uuid": "GPU-fixture"}]}
        self.cfg["argv"] = ["/opt/venv_eval/bin/python", "-B", str(self.source), "worker", "--config", self.cfg["_config_path"]]
        # Seed with real /proc identity fields, then replace only the fixture launch/env.
        self.parent = base.identity(os.getppid())
        self.current = base.identity(os.getpid())
        self.parent["argv"] = ["/opt/venv_eval/bin/python", "-B", str(self.source), "--config",
            self.cfg["_config_path"], "--config-sha", self.cfg["_config_sha"], "--manager-sha", m.BASE_SHA]
        self.parent["CVD"] = []
        self.current["argv"] = copy.deepcopy(self.cfg["argv"])
        self.current["CVD"] = ["CUDA_VISIBLE_DEVICES=2"]
        self.current["cgroup"] = self.parent["cgroup"]
        self.admission = {key: copy.deepcopy(self.cfg[key]) for key in
            ("phase", "targetcount", "node", "gpus", "out", "producer", "plan", "holders", "source_sha256")}
        self.admission.update(manager_identity=copy.deepcopy(self.parent), manager_sha256=m.BASE_SHA,
            config_sha256=self.cfg["_config_sha"], model_identity_sha256=self.cfg["modelref"]["model_identity_sha256"])
        self.live = dict(child_identity=copy.deepcopy(self.current), manager_identity=copy.deepcopy(self.parent),
            phase=self.cfg["phase"], targetcount=100, argv=copy.deepcopy(self.cfg["argv"]), early_exit_code=None)
        self.write_receipts()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.h = types.SimpleNamespace(bench=types.SimpleNamespace(strict=Mock(wraps=strict_globals["strict"])),
            kernel_lock_matches=Mock(wraps=lock_globals["kernel_lock_matches"]))
        self.identities = {self.parent["pid"]: self.parent, self.current["pid"]: self.current}
        self.stack.enter_context(patch.object(base, "identity", side_effect=lambda pid: copy.deepcopy(self.identities[pid])))
        self.apps = self.stack.enter_context(patch.object(base, "apps", return_value=[]))
        self.uuid = self.stack.enter_context(patch.object(base, "uuid_for", return_value="GPU-fixture"))
        self.stack.enter_context(patch.object(base, "sha", side_effect=AssertionError("Gate must not re-open via base.sha")))
        self.stack.enter_context(patch.object(os, "kill", side_effect=AssertionError("Signals forbidden")))
        self.stack.enter_context(patch.object(signal, "signal", side_effect=AssertionError("Signal handlers forbidden")))
        self.stack.enter_context(patch.object(subprocess, "check_output", side_effect=AssertionError("Real GPU command forbidden")))
        self.lock = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=os.getuid(),
                                         st_size=0, st_dev=os.makedev(8, 1), st_ino=555)
        self.lease = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=os.getuid(),
            st_size=0, st_dev=os.makedev(8, 1), st_ino=556, st_mtime=time.time())
        self.fd_lock, self.fd_lease = copy.copy(self.lock), copy.copy(self.lease)
        self.lock_lines = "%d: FLOCK ADVISORY WRITE %d 08:01:555 0 EOF" % (1, self.parent["pid"])
        self.fd_dir = "/proc/%d/fd" % self.parent["pid"]
        real_listdir, real_readlink, real_stat, real_lstat = os.listdir, os.readlink, os.stat, os.lstat
        real_read_text = Path.read_text
        self.stack.enter_context(patch.object(os, "listdir", side_effect=lambda path:
            ["10", "11"] if os.fspath(path) == self.fd_dir else real_listdir(path)))
        self.stack.enter_context(patch.object(os, "readlink", side_effect=lambda path, **kwargs:
            "/tmp/ms12k_gpu_2.lock" if os.fspath(path) == self.fd_dir + "/10" else
            "/tmp/od_reserve_2" if os.fspath(path) == self.fd_dir + "/11" else real_readlink(path, **kwargs)))
        self.stack.enter_context(patch.object(os, "stat", side_effect=lambda path, *args, **kwargs:
            self.fd_lock if os.fspath(path) == self.fd_dir + "/10" else
            self.fd_lease if os.fspath(path) == self.fd_dir + "/11" else real_stat(path, *args, **kwargs)))
        self.stack.enter_context(patch.object(os, "lstat", side_effect=lambda path, *args, **kwargs:
            self.lock if os.fspath(path) == "/tmp/ms12k_gpu_2.lock" else
            self.lease if os.fspath(path) == "/tmp/od_reserve_2" else real_lstat(path, *args, **kwargs)))
        self.stack.enter_context(patch.object(Path, "read_text", autospec=True, side_effect=lambda path, *args, **kwargs:
            self.lock_lines if str(path) == "/proc/locks" else real_read_text(path, *args, **kwargs)))

    def write_receipts(self):
        (self.queue / "admission.json").write_text(json.dumps(self.admission))
        (self.queue / "live.json").write_text(json.dumps(self.live))

    def gate(self):
        gate = m.Gate(base, self.h, self.cfg, self.source, "/opt/venv_eval/bin/python")
        self.addCleanup(gate.close)
        return gate

    def denied(self, gate, regex=None, before_load=False):
        with self.assertRaisesRegex((PermissionError, ValueError, KeyError, OSError), regex or ".*"):
            gate(before_load=before_load)
        self.assertTrue(all(evidence.fd is None for evidence in gate._files))
        with self.assertRaisesRegex(PermissionError, "closed or failed"):
            gate()

    def test_success_cache_once_dynamic_every_call_and_native_proof(self):
        expected_admission_sha = hashlib.sha256((self.queue / "admission.json").read_bytes()).hexdigest()
        with self.gate() as gate, patch.object(os, "read", wraps=os.read) as read, \
             patch.object(hashlib, "sha256", wraps=hashlib.sha256) as sha:
            proof = gate(self.cfg, before_load=True)
            self.assertTrue(proof["admitted"] and proof["caller_owned"] and proof["fixed_evidence_cache"])
            self.assertEqual(proof["parent_identity"], self.parent)
            self.assertEqual(proof["child_identity"], self.current)
            self.assertEqual(proof["gpu_uuids"], ["GPU-fixture"])
            self.assertEqual(proof["admission_ref"]["sha256"], expected_admission_sha)
            self.assertEqual(self.h.bench.strict.call_count, 2)
            self.assertEqual(sha.call_count, 3)
            for evidence in gate._files:
                self.assertEqual(fcntl.fcntl(evidence.fd, fcntl.F_GETFL) & os.O_ACCMODE, os.O_RDONLY)
                self.assertEqual(evidence.sha_calls, 1)
            read.reset_mock()
            sha.reset_mock()
            self.h.bench.strict.reset_mock()
            self.apps.reset_mock()
            self.uuid.reset_mock()
            self.h.kernel_lock_matches.reset_mock()
            for _ in range(3):
                self.assertTrue(gate()["admitted"])
            self.assertEqual(read.call_count, 0)
            self.assertEqual(sha.call_count, 0)
            self.assertEqual(self.h.bench.strict.call_count, 0)
            self.assertEqual(self.apps.call_count, 3)
            self.assertEqual(self.uuid.call_count, 3)
            self.assertEqual(self.h.kernel_lock_matches.call_count, 3)
            self.assertEqual(gate.profile["calls"], 4)
            self.assertGreater(gate.profile["fixed_load"], 0)
        self.assertTrue(all(evidence.fd is None for evidence in gate._files))

    def test_wrong_parent_argv_cvd_uid_cgroup_start_exe_denied(self):
        mutations = [("parent", "argv", ["/wrong/python"]), ("parent", "CVD", ["CUDA_VISIBLE_DEVICES=3"]),
            ("parent", "uid", os.getuid() + 1), ("parent", "cgroup", "wrong"),
            ("parent", "start_ticks", self.parent["start_ticks"] + 1), ("parent", "exe", "/wrong/exe"),
            ("current", "argv", ["wrong"]), ("current", "CVD", ["CUDA_VISIBLE_DEVICES=3"]),
            ("current", "uid", os.getuid() + 1), ("current", "cgroup", "wrong"),
            ("current", "start_ticks", self.current["start_ticks"] + 1), ("current", "ppid", 1),
            ("parent", "state", "Z"), ("current", "state", "Z")]
        for who, key, value in mutations:
            with self.subTest(who=who, key=key):
                gate = self.gate()
                gate()
                actual = getattr(self, who)
                old = actual[key]
                actual[key] = value
                try:
                    self.denied(gate)
                finally:
                    actual[key] = old
        gate = self.gate()
        gate()
        with patch.object(os, "getppid", return_value=999999):
            self.denied(gate)

    def test_parent_identity_read_error_denied(self):
        gate = self.gate()
        gate()
        with patch.object(base, "identity", side_effect=FileNotFoundError("dead parent")):
            self.denied(gate)

    def test_lease_flock_fd_safety_every_call(self):
        mutations = [(self.lease, "st_mtime", time.time() - 31), (self.lease, "st_mtime", time.time() + 31),
            (self.lease, "st_ino", 999), (self.fd_lease, "st_ino", 999),
            (self.fd_lock, "st_ino", 999), (self.fd_lock, "st_dev", os.makedev(9, 1)),
            (self.lock, "st_uid", os.getuid() + 1), (self.lock, "st_size", 1),
            (self.lock, "st_mode", stat.S_IFLNK | 0o600), (self.lease, "st_uid", os.getuid() + 1),
            (self.lease, "st_size", 1), (self.lease, "st_mode", stat.S_IFLNK | 0o600)]
        for obj, key, value in mutations:
            with self.subTest(key=key, value=value):
                gate = self.gate()
                gate()
                old = getattr(obj, key)
                setattr(obj, key, value)
                try:
                    self.denied(gate)
                finally:
                    setattr(obj, key, old)
        original_lines = self.lock_lines
        for lines in ("", original_lines.replace("FLOCK", "POSIX"), original_lines.replace("08:01:555", "08:01:999"),
                      original_lines.replace("WRITE %d" % self.parent["pid"], "WRITE 999999")):
            gate = self.gate()
            gate()
            self.lock_lines = lines
            try:
                self.denied(gate, "FLOCK")
            finally:
                self.lock_lines = original_lines

    def test_gpu_apps_before_model_load_and_external_subtree_denied(self):
        gate = self.gate()
        gate()
        self.apps.return_value = [{"uuid": "GPU-fixture", "pid": self.current["pid"]}]
        self.denied(gate, "occupied before", before_load=True)
        self.apps.return_value = []
        gate = self.gate()
        gate()
        self.identities[700001] = dict(copy.deepcopy(self.current), pid=700001, ppid=700001,
                                     start_ticks=self.current["start_ticks"] + 1)
        self.apps.return_value = [{"uuid": "GPU-fixture", "pid": 700001}]
        self.denied(gate, "outside sole")

    def test_gpu_apps_admitted_descendants_without_rank_cvd_requirement(self):
        self.identities[700001] = dict(copy.deepcopy(self.current), pid=700001, ppid=self.current["pid"],
            start_ticks=self.current["start_ticks"] + 1, argv=["native compute child"], CVD=[])
        self.apps.return_value = [{"uuid": "GPU-fixture", "pid": 700001}, {"uuid": "GPU-other", "pid": 999999}]
        self.assertTrue(self.gate()()["admitted"])
        for key, value in (("uid", os.getuid() + 1), ("cgroup", "wrong"),
                           ("start_ticks", self.current["start_ticks"] - 1), ("state", "Z")):
            ancestor = self.identities[700001]
            old = ancestor[key]
            ancestor[key] = value
            try:
                self.denied(self.gate(), "process identity")
            finally:
                ancestor[key] = old

    def test_gpu_uuid_change_denied(self):
        gate = self.gate()
        gate()
        self.uuid.return_value = "GPU-changed"
        self.denied(gate, "UUID")

    def test_fixed_evidence_changes_replacements_symlinks_same_bytes_denied(self):
        for path in (self.queue / "admission.json", self.queue / "live.json", self.source):
            original = path.read_bytes()
            for mutation in ("truncate", "replace", "symlink", "samebytes", "mtime", "ctime"):
                with self.subTest(path=path.name, mutation=mutation):
                    gate = self.gate()
                    gate()
                    before = path.stat()
                    if mutation == "truncate":
                        path.write_bytes(original[:-1])
                    elif mutation == "replace":
                        replacement = self.root / "replacement"
                        replacement.write_bytes(original)
                        os.replace(replacement, path)
                    elif mutation == "symlink":
                        replacement = self.root / "replacement"
                        replacement.write_bytes(original)
                        path.unlink()
                        path.symlink_to(replacement)
                    elif mutation == "samebytes":
                        path.write_bytes(original)
                        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
                    elif mutation == "mtime":
                        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
                    else:
                        time.sleep(.002)
                        os.chmod(path, (before.st_mode & 0o777) ^ 0o100)
                        os.chmod(path, before.st_mode & 0o777)
                        self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
                    try:
                        self.denied(gate, "Cached fixed evidence changed")
                    finally:
                        if path.is_symlink():
                            path.unlink()
                        path.write_bytes(original)

    def test_fixed_evidence_mutation_during_dynamic_check_denied(self):
        gate = self.gate()
        gate()
        path = self.queue / "admission.json"
        def mutate():
            before = path.stat()
            path.write_bytes(path.read_bytes())
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
            return []
        self.apps.side_effect = mutate
        self.denied(gate, "Cached fixed evidence changed")

    def test_fixed_evidence_init_sha_missing_fields_bad_json_denied(self):
        for key in ("phase", "model_identity_sha256", "holders", "source_sha256", "manager_identity"):
            old = self.admission.pop(key)
            self.write_receipts()
            try:
                self.denied(self.gate())
            finally:
                self.admission[key] = old
                self.write_receipts()
        old = self.cfg["resource_manager_ref"]["sha256"]
        self.cfg["resource_manager_ref"]["sha256"] = "0" * 64
        try:
            self.denied(self.gate(), "SHA")
        finally:
            self.cfg["resource_manager_ref"]["sha256"] = old
        for raw in (b'{"duplicate":1,"duplicate":2}', b'{"broken":', b'{"number":NaN}'):
            (self.queue / "admission.json").write_bytes(raw)
            try:
                self.denied(self.gate())
            finally:
                self.write_receipts()

    def test_receipt_binding_mismatches_and_forged_parent_argv_denied(self):
        changes = {"config_sha256": "wrong", "manager_sha256": "wrong", "model_identity_sha256": "wrong",
            "phase": "wrong", "targetcount": 99, "node": "N3", "gpus": [3], "out": "/wrong/output",
            "holders": [], "source_sha256": {str(self.source): "wrong"}}
        for key, value in changes.items():
            with self.subTest(receipt="admission", key=key):
                original = self.admission[key]
                self.admission[key] = value
                self.write_receipts()
                try:
                    self.denied(self.gate())
                finally:
                    self.admission[key] = original
                    self.write_receipts()
        changes = {"phase": "wrong", "targetcount": 99, "argv": ["wrong"], "early_exit_code": 1,
            "manager_identity": dict(self.parent, start_ticks=self.parent["start_ticks"] + 1),
            "child_identity": dict(self.current, start_ticks=self.current["start_ticks"] + 1)}
        for key, value in changes.items():
            with self.subTest(receipt="live", key=key):
                original = self.live[key]
                self.live[key] = value
                self.write_receipts()
                try:
                    self.denied(self.gate())
                finally:
                    self.live[key] = original
                    self.write_receipts()
        self.parent["argv"][0] = "/opt/venv_vllm/bin/python"
        self.admission["manager_identity"] = copy.deepcopy(self.parent)
        self.live["manager_identity"] = copy.deepcopy(self.parent)
        self.write_receipts()
        self.denied(self.gate(), "parent native manager argv")

    def test_gpu_ancestor_pid_start_change_during_check_denied(self):
        ancestor = dict(copy.deepcopy(self.current), pid=700001, ppid=self.current["pid"],
            start_ticks=self.current["start_ticks"] + 1)
        self.apps.return_value = [{"uuid": "GPU-fixture", "pid": 700001}]
        count = 0
        def identity(pid):
            nonlocal count
            if pid == 700001:
                count += 1
                return dict(ancestor, start_ticks=ancestor["start_ticks"] + (count - 1))
            return copy.deepcopy(self.identities[pid])
        with patch.object(base, "identity", side_effect=identity):
            self.denied(self.gate(), "ancestor identity changed")

    def test_partial_raw_read_and_init_failure_close_all_fds(self):
        gate = self.gate()
        real_read = os.read
        calls = 0
        def partial(fd, size):
            nonlocal calls
            calls += 1
            return real_read(fd, 5) if calls == 1 else b""
        with patch.object(os, "read", side_effect=partial), patch.object(os, "close", wraps=os.close) as close:
            self.denied(gate, "Partial fixed evidence")
            self.assertEqual(close.call_count, 1)
        (self.queue / "live.json").write_bytes(b"{")
        with patch.object(os, "close", wraps=os.close) as close:
            self.denied(self.gate())
            self.assertEqual(close.call_count, 3)

    def test_read_change_during_init_detected(self):
        real_read = os.read
        path = self.queue / "admission.json"
        changed = False
        def changing(fd, size):
            nonlocal changed
            raw = real_read(fd, size)
            if not changed:
                changed = True
                before = path.stat()
                path.write_bytes(path.read_bytes())
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
            return raw
        with patch.object(os, "read", side_effect=changing):
            self.denied(self.gate(), "Cached fixed evidence changed")

    def test_initial_symlink_fifo_owner_size_rejected_before_open(self):
        path = self.root / "evidence"
        path.symlink_to(self.source)
        with self.assertRaises(PermissionError):
            m.ReadonlyFileCache(path, 2**20)
        path.unlink()
        os.mkfifo(path)
        with patch.object(os, "open", wraps=os.open) as opened, self.assertRaises(PermissionError):
            m.ReadonlyFileCache(path, 2**20)
        opened.assert_not_called()
        path.unlink()
        path.write_bytes(b"x" * (2**20 + 1))
        with self.assertRaises(PermissionError):
            m.ReadonlyFileCache(path, 2**20)
        bad = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=os.getuid() + 1, st_size=2)
        with patch.object(os, "lstat", return_value=bad), self.assertRaises(PermissionError):
            m.ReadonlyFileCache(self.source, 2**20)

    def test_live_first_use_bounded_wait_only_once(self):
        real_lstat = os.lstat
        missing = 3
        def delayed(path, *args, **kwargs):
            nonlocal missing
            if os.fspath(path) == str(self.queue / "live.json") and missing:
                missing -= 1
                raise FileNotFoundError("manager delay")
            return real_lstat(path, *args, **kwargs)
        with patch.object(os, "lstat", side_effect=delayed), patch.object(time, "sleep") as sleep:
            gate = self.gate()
            gate()
            self.assertEqual(sleep.call_count, 3)
            sleep.reset_mock()
            gate()
            sleep.assert_not_called()
        real_lstat = os.lstat
        def absent(path, *args, **kwargs):
            if os.fspath(path) == str(self.queue / "live.json"):
                raise FileNotFoundError("missing live")
            return real_lstat(path, *args, **kwargs)
        with patch.object(os, "lstat", side_effect=absent), patch.object(time, "monotonic", side_effect=[0, 16]):
            self.denied(self.gate(), "15 seconds")

    def test_config_identity_freeze_wrong_cfg_and_mutation_denied(self):
        gate = self.gate()
        gate()
        with self.assertRaisesRegex(PermissionError, "different config"):
            gate(copy.deepcopy(self.cfg))
        gate = self.gate()
        gate()
        self.cfg["gpus"] = [2.0]
        self.denied(gate, "config changed")
        self.cfg["gpus"] = [2]
        gate = self.gate()
        gate()
        self.cfg["_config_sha"] = "changed"
        self.denied(gate, "config changed")

    def test_return_proof_mutation_does_not_change_cache(self):
        gate = self.gate()
        proof = gate()
        proof["admission_ref"]["sha256"] = "changed"
        proof["parent_identity"]["argv"].append("changed")
        self.assertEqual(gate()["admission_ref"]["sha256"], gate._files[0].sha256)

    def test_original_sources_unchanged_and_static_module_no_model_imports(self):
        expected = {"resource_manager.py": m.BASE_SHA,
            "bench_manager_driver_v2.py": "5b93d15fff8fa0473da0ecdeb3aa7dfc89dd9dab139d41bf355158bd053d930f",
            "ms_unified_native_archive_manager_v1.py": "48068981067a498b8fdacced7dd277a93dbe806942c0a3b7dfaa631eed9d5a14"}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((RT / name).read_bytes()).hexdigest(), digest)
        tree = ast.parse((RT / "ms_cached_native_gate_v1.py").read_text())
        imports = {node.names[0].name for node in tree.body if isinstance(node, ast.Import)}
        self.assertEqual(imports, {"hashlib", "json", "os", "stat", "time"})
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module != "pathlib" for node in tree.body))


if __name__ == "__main__":
    unittest.main()
