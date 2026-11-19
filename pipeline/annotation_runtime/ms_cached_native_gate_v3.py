"""Per-worker fixed evidence cache; all native physical ownership checks stay live."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

BASE_SHA = "6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6"
ALLOWED_GPUS = {"N6": (2, 3), "N3": (0, 1), "N2": (2, 3, 4, 5)}


def need(condition, message):
    if not condition:
        raise PermissionError(message)


class ReadonlyFileCache:
    """One bounded raw read, with a retained no-follow fd and immutable stat evidence."""

    @staticmethod
    def identity(st):
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_uid, st.st_mode)

    def __init__(self, path, maximum, strict=None):
        self.path = os.fspath(path)
        self.fd = None
        self.read_calls = self.read_bytes = self.sha_calls = 0
        need(os.path.isabs(self.path) and os.path.realpath(self.path) == self.path,
             "Stable absolute fixed evidence path required")
        initial = os.lstat(self.path)
        need(stat.S_ISREG(initial.st_mode) and initial.st_uid == os.getuid() and
             0 < initial.st_size <= maximum, "Unsafe fixed evidence type/UID/size: " + self.path)
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
            before = os.fstat(self.fd)
            self.snapshot = self.identity(before)
            need(self.snapshot == self.identity(initial), "Fixed evidence changed while opening: " + self.path)
            blocks, digest, size = [], hashlib.sha256(), 0
            self.sha_calls = 1
            while True:
                block = os.read(self.fd, 4096)
                self.read_calls += 1
                if not block:
                    break
                size += len(block)
                self.read_bytes += len(block)
                need(size <= maximum and size <= before.st_size, "Fixed evidence grew during raw read")
                blocks.append(block)
                digest.update(block)
            need(size == before.st_size, "Partial fixed evidence raw read")
            self.check()
            self.sha256 = digest.hexdigest()
            raw = b"".join(blocks)
            self.value = strict(raw) if strict is not None else None
            if strict is not None:
                need(isinstance(self.value, dict), "Fixed evidence JSON object required")
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        need(self.fd is not None, "Fixed evidence fd is closed")
        need(self.identity(os.fstat(self.fd)) == self.snapshot and
             self.identity(os.lstat(self.path)) == self.snapshot, "Cached fixed evidence changed: " + self.path)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class Gate:
    """Bind once to (base, h, cfg, self_path, python); call before every guarded stage."""

    def __init__(self, base, h, cfg, self_path, python):
        self.base, self.h = base, h
        self.cfg = cfg
        self._frozen_cfg = json.dumps(cfg, sort_keys=True, allow_nan=False, separators=(",", ":"))
        self.self_path, self.python = os.fspath(self_path), os.fspath(python)
        self._files = []
        self._ready = self._closed = self._failed = False
        self.profile = dict(calls=0, seconds=0.0, fixed_load=0.0, fixed_stat=0.0,
                            identities=0.0, parent_fds_locks=0.0, gpu_uuid=0.0, gpu_apps=0.0)
        need(self.python == "/opt/venv_eval/bin/python", "Exact current single-GPU native interpreter required")

    def _check_cfg(self, cfg):
        need(cfg is self.cfg and
             json.dumps(cfg, sort_keys=True, allow_nan=False, separators=(",", ":")) == self._frozen_cfg,
             "Bound native config changed or different config supplied")

    def _load_fixed(self):
        started = time.perf_counter()
        try:
            cfg = self.cfg
            need(cfg["node"] in ALLOWED_GPUS and len(cfg["gpus"]) == 1 and
                 type(cfg["gpus"][0]) is int and cfg["gpus"][0] in ALLOWED_GPUS[cfg["node"]],
                 "Explicit restored N6/N3 single GPU required")
            need(cfg["env"]["CUDA_VISIBLE_DEVICES"] == str(cfg["gpus"][0]), "Native sole GPU CVD differs")
            queue = Path(cfg["queue"])
            admission = ReadonlyFileCache(queue / "admission.json", 4 * 2**20, self.h.bench.strict)
            self._files.append(admission)
            source = ReadonlyFileCache(self.self_path, 2**20)
            self._files.append(source)
            deadline = time.monotonic() + 15
            while True:
                try:
                    os.lstat(queue / "live.json")
                    break
                except FileNotFoundError:
                    need(time.monotonic() < deadline, "Native live receipt not generated within 15 seconds")
                    time.sleep(.05)
            live = ReadonlyFileCache(queue / "live.json", 4 * 2**20, self.h.bench.strict)
            self._files.append(live)
            self._admission, self._live = admission.value, live.value
            self._admission_ref = dict(path=admission.path, sha256=admission.sha256)
            need(cfg["resource_manager_ref"] == dict(path=self.self_path, sha256=source.sha256) and
                 cfg["producer"] == cfg["resource_manager_ref"] and
                 cfg["source_sha256"][self.self_path] == source.sha256 ==
                 self._admission["source_sha256"][self.self_path], "Native manager source SHA binding differs")
            need(cfg["source_sha256"][os.fspath(self.base.__file__)] == BASE_SHA and
                 self._admission["manager_sha256"] == BASE_SHA, "Original native lifecycle admission SHA differs")
            need(self._admission["source_sha256"] == cfg["source_sha256"], "Native admitted source SHA map differs")
            for key in ("phase", "targetcount", "node", "gpus", "out", "producer", "plan"):
                need(self._admission[key] == cfg[key], "Native admission binding differs: " + key)
            need(self._admission["config_sha256"] == cfg["_config_sha"] and
                 self._admission["model_identity_sha256"] == cfg["modelref"]["model_identity_sha256"],
                 "Native admission config/model hash differs")
            need(len(self._admission["holders"]) == len(cfg["gpus"]) and
                 [holder["gpu"] for holder in self._admission["holders"]] == cfg["gpus"] and
                 [holder["uuid"] for holder in self._admission["holders"]] == [holder["uuid"] for holder in cfg["holders"]],
                 "Native ordered physical GPU inventory differs")
            for key in ("phase", "targetcount", "argv"):
                need(self._live[key] == cfg[key], "Native live binding differs: " + key)
            need(self._live["early_exit_code"] is None, "Native child already exited before live receipt")
            for key in self.base.IDENTITY_KEYS:
                need(self._live["manager_identity"][key] == self._admission["manager_identity"][key],
                     "Native admission/live manager identity differs")
            self._check_fixed()
            self._ready = True
        finally:
            self.profile["fixed_load"] += time.perf_counter() - started

    def _check_fixed(self):
        started = time.perf_counter()
        try:
            for evidence in self._files:
                evidence.check()
        finally:
            self.profile["fixed_stat"] += time.perf_counter() - started

    def _identity(self, pid):
        actual = self.base.identity(pid)
        need(actual["pid"] == pid and actual["start_ticks"] > 0 and actual["state"] not in ("Z", "X", "x"),
             "Native process identity is dead or invalid")
        return actual

    def _same_process(self, actual, expected):
        return all(actual[key] == expected[key] for key in self.base.IDENTITY_KEYS) and actual["ppid"] == expected["ppid"]

    def _dynamic(self, before_load):
        # This is the native parent_gate body, with only the fixed-file reads removed.
        cfg, base, admission, live = self.cfg, self.base, self._admission, self._live
        started = time.perf_counter()
        parent_pid, current_pid = os.getppid(), os.getpid()
        parent, current = self._identity(parent_pid), self._identity(current_pid)
        expected = admission["manager_identity"]
        need(all(parent[k] == expected[k] for k in base.IDENTITY_KEYS) and parent_pid == expected["pid"] and
             parent["uid"] == current["uid"] == os.getuid() and parent["cgroup"] == current["cgroup"] and
             current["ppid"] == parent_pid, "Direct native parent identity/UID/cgroup differs")
        need(parent["argv"] == [self.python, "-B", self.self_path, "--config", cfg["_config_path"],
             "--config-sha", cfg["_config_sha"], "--manager-sha", BASE_SHA], "Direct parent native manager argv differs")
        need(current["argv"] == cfg["argv"] and
             current["CVD"] == ["CUDA_VISIBLE_DEVICES=" + cfg["env"]["CUDA_VISIBLE_DEVICES"]],
             "Native admitted driver argv/CVD/config differs")
        need(live["child_identity"]["pid"] == current["pid"] and
             all(live["child_identity"][k] == current[k] for k in base.IDENTITY_KEYS),
             "Native direct child live receipt differs")
        self.profile["identities"] += time.perf_counter() - started
        started = time.perf_counter()
        parent_fds = []
        for name in os.listdir("/proc/%d/fd" % parent_pid):
            fd = "/proc/%d/fd/%s" % (parent_pid, name)
            try:
                parent_fds.append((os.readlink(fd), os.stat(fd)))
            except FileNotFoundError:
                pass
        locks = Path("/proc/locks").read_text().splitlines()
        for gpu in cfg["gpus"]:
            lock_path = "/tmp/ms12k_gpu_%d.lock" % gpu
            lock = os.lstat(lock_path)
            need(stat.S_ISREG(lock.st_mode) and lock.st_uid == os.getuid() and lock.st_size == 0,
                 "Unsafe native FLOCK inode/type/owner")
            need(any(name == lock_path and st.st_ino == lock.st_ino and st.st_dev == lock.st_dev for name, st in parent_fds) and
                 any(self.h.kernel_lock_matches(line, lock) and line.split()[4] == str(parent_pid) for line in locks),
                 "Native parent FLOCK fd/owner missing")
            lease_path = "/tmp/od_reserve_%d" % gpu
            lease = os.lstat(lease_path)
            need(stat.S_ISREG(lease.st_mode) and lease.st_uid == os.getuid() and lease.st_size == 0 and
                 0 <= time.time() - lease.st_mtime < 30 and
                 any(name == lease_path and st.st_ino == lease.st_ino and st.st_dev == lease.st_dev for name, st in parent_fds),
                 "Native parent lease inode/owner/live heartbeat missing")
        self.profile["parent_fds_locks"] += time.perf_counter() - started
        started = time.perf_counter()
        uuids = []
        # v1 deliberately re-queries hardware identity every call, as in the original gate.
        for gpu, holder in zip(cfg["gpus"], admission["holders"]):
            uuid = base.uuid_for(gpu)
            need(uuid == holder["uuid"], "Native GPU UUID differs")
            uuids.append(uuid)
        self.profile["gpu_uuid"] += time.perf_counter() - started
        started = time.perf_counter()
        for app in base.apps():
            if app["uuid"] not in uuids:
                continue
            need(not before_load, "GPU occupied before admitted model load")
            pid, ancestry = app["pid"], []
            for _ in range(20):
                if pid == current["pid"]:
                    break
                ancestor = self._identity(pid)
                need(ancestor["uid"] == os.getuid() and ancestor["cgroup"] == current["cgroup"] and
                     ancestor["start_ticks"] >= current["start_ticks"], "Unknown GPU process identity")
                ancestry.append(ancestor)
                pid = ancestor["ppid"]
            need(pid == current["pid"], "GPU app outside sole admitted driver tree; never signal unknown job")
            for ancestor in ancestry:
                need(self._same_process(self._identity(ancestor["pid"]), ancestor), "GPU ancestor identity changed during check")
        self.profile["gpu_apps"] += time.perf_counter() - started
        started = time.perf_counter()
        need(os.getppid() == parent_pid and os.getpid() == current_pid and
             self._same_process(self._identity(parent_pid), parent) and
             self._same_process(self._identity(current_pid), current), "Native parent/child changed during dynamic check")
        self.profile["identities"] += time.perf_counter() - started
        return dict(admitted=True, caller_owned=True, parent_identity=parent, child_identity=current,
                    gpu_uuids=uuids, admission_ref=dict(self._admission_ref), fixed_evidence_cache=True)

    def __call__(self, cfg=None, before_load=False):
        need(not self._closed and not self._failed, "Native cached gate is closed or failed; never revalidate/fallback")
        started = time.perf_counter()
        self.profile["calls"] += 1
        try:
            self._check_cfg(self.cfg if cfg is None else cfg)
            if not self._ready:
                self._load_fixed()
            self._check_fixed()
            proof = self._dynamic(before_load)
            self._check_cfg(self.cfg)
            self._check_fixed()
            return proof
        except BaseException:
            self._failed = True
            self.close()
            raise
        finally:
            self.profile["seconds"] += time.perf_counter() - started

    def close(self):
        self._closed = True
        for evidence in self._files:
            evidence.close()

    def __enter__(self):
        need(not self._closed and not self._failed, "Native cached gate is closed or failed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
