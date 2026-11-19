"""Read-only finite noVLM job monitor; metadata completion is not semantic approval."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
HOSTS = {"N3": "28.216.19.213", "N6": "28.216.19.225", "N2": "28.216.17.70"}
GPU_SCOPE = {"N3": (0, 1), "N6": (2, 3), "N2": (2, 3, 4, 5)}
SAFE_FLAGS = ("all_holders_safe", "locks_released", "leases_restored",
              "managed_parent_wait", "unknown_jobs_not_signaled")
UNREVIEWED = {"semantic_approval": False, "quality_approved": False,
              "training_eligible": False, "needs_visual_revision": True,
              "diagnostic_not_production": True, "vlm_called": False}

# Standalone remote payload: no imports from managers, no GPU queries, no writes.
REMOTE = r'''
import glob, hashlib, json, os, re, sys, time
from pathlib import Path

def read_json(path, warnings, limit=8*1024*1024):
    path = Path(path)
    try:
        if not path.is_absolute():
            raise ValueError("nonabsolute metadata path")
        with path.open("rb") as stream:
            raw = stream.read(limit+1)
        if len(raw) > limit:
            raise ValueError("metadata size limit")
        data = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))
        if not isinstance(data, dict):
            raise ValueError("expected metadata object")
        return data, hashlib.sha256(raw).hexdigest()
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as error:
        warnings.append(str(path) + ": " + str(error))
        return None, None

def proc(identity):
    if not isinstance(identity, dict):
        return {"alive": False, "identity_verified": False}
    pid, ticks = identity.get("pid"), identity.get("start_ticks")
    if type(pid) is not int or pid <= 0 or type(ticks) is not int:
        return {"alive": False, "identity_verified": False, "warning": "missing pid/start_ticks"}
    result = {"pid": pid, "expected_start_ticks": ticks, "alive": False, "identity_verified": False}
    try:
        text = Path("/proc/%d/stat" % pid).read_text()
        fields = text[text.rfind(")")+2:].split()
        result.update(start_ticks=int(fields[19]), state=fields[0])
        result["identity_verified"] = result["start_ticks"] == ticks
        result["alive"] = result["identity_verified"] and fields[0] not in ("Z", "X")
        result["wchan"] = Path("/proc/%d/wchan" % pid).read_text().strip()
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        result["started_unix"] = time.time()-uptime+ticks/os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError) as error:
        result["warning"] = str(error)
    return result

def resources(path, warnings):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size-8192))
            raw = stream.read(8192)
        lines = raw.splitlines()
        if size > 8192:
            lines = lines[1:]
        samples = []
        for line in lines:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    samples.append(value)
            except ValueError:
                continue
        return {"source": str(path), "tail_bytes_read": len(raw), "last_samples": samples[-3:]}
    except FileNotFoundError:
        return {"source": str(path), "tail_bytes_read": 0, "last_samples": []}
    except OSError as error:
        warnings.append(str(path) + ": " + str(error))
        return {"source": str(path), "last_samples": []}

def collect(job, roots):
    warnings = []
    result = {"job": job, "reachable": True, "warnings": warnings, "local_receipts": []}
    cfg, _ = read_json(job["config_path"], warnings)
    if cfg is None:
        result["collection_error"] = "config missing or invalid"
        return result
    result["config"] = cfg
    for key in ("node", "phase"):
        if cfg.get(key) != job[key]:
            result["collection_error"] = "registry/config " + key + " mismatch"
            return result
    if cfg.get("config_path") != job["config_path"]:
        result["collection_error"] = "registry/config path mismatch"
        return result
    for key in ("out", "queue"):
        if not isinstance(cfg.get(key), str) or not Path(cfg[key]).is_absolute():
            result["collection_error"] = "invalid " + key + " path"
            return result
    ids, n = cfg.get("selected_source_ids"), cfg.get("targetcount")
    if type(n) is not int or not 1 <= n <= 100 or not isinstance(ids, list) or len(ids) != n or not all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
        result["collection_error"] = "invalid finite source selection"
        return result
    queue, out = Path(cfg["queue"]), Path(cfg["out"])
    for name in ("live", "admission", "failure", "worker_exit", "resources_safe",
                 "pending_budget", "low_memory_warning", "child_cleanup_blocked"):
        result[name], _ = read_json(queue / (name + ".json"), warnings)
    result["resource_samples"] = resources(queue / "resource_samples.jsonl", warnings)
    result["gpu_sample_source"] = "native queue resource_samples.jsonl (not a fresh NVML query)"
    for gpu in cfg.get("gpus", []):
        if type(gpu) is int and 0 <= gpu <= 7:
            for name in ("restore_failure", "restore_blocked", "restore_unknown",
                         "restore_wait", "holder_exit_wait", "lease_cleanup_failure"):
                key = name + "_gpu%d" % gpu
                result[key], _ = read_json(queue / (key + ".json"), warnings)
    live = result.get("live") or {}
    admission = result.get("admission") or {}
    for name, value in (("live", live), ("admission", admission)):
        if value and value.get("phase") != cfg["phase"]:
            result["collection_error"] = name + " phase mismatch"
            return result
    child = live.get("child_identity") or {}
    if child and (child.get("argv") != cfg.get("argv") or not isinstance(child.get("argv"), list) or "worker" not in child["argv"] or job["config_path"] not in child["argv"]):
        result["collection_error"] = "worker argv/config identity mismatch; holders are not workers"
        return result
    if "gpu" in job and cfg.get("gpus") != [job["gpu"]]:
        result["collection_error"] = "registry/config GPU mismatch"
        return result
    result["worker_process"] = proc(child)
    result["manager_process"] = proc(live.get("manager_identity") or admission.get("manager_identity"))
    ticket_ref = cfg.get("ticket") or {}
    ticket, ticket_sha = read_json(ticket_ref.get("path", "/nonexistent-monitor-ticket"), warnings)
    result["ticket"] = ticket
    if not ticket or ticket_sha != ticket_ref.get("sha256") or ticket.get("edition") != cfg["phase"] or ticket.get("output_root") != cfg["out"] or not isinstance(ticket.get("ticket_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ticket["ticket_id"]):
        result["collection_error"] = "ticket identity missing or mismatched"
        return result
    batch = out / cfg["phase"] / "batches" / ticket["ticket_id"]
    result["commit_path"] = str(batch / "commit.json")
    result["summary"], _ = read_json(out / "summary.json", warnings)
    result["commit"], result["commit_sha256"] = read_json(batch / "commit.json", warnings)
    if result["worker_process"].get("alive") and not result.get("worker_exit"):
        sources = ticket.get("sources") or []
        if len(sources) != n or [row.get("id") for row in sources if isinstance(row, dict)] != ids:
            warnings.append("ticket source rows differ from selected ids")
            return result
        started = result["worker_process"].get("started_unix")
        if started is None:
            warnings.append("worker start time unavailable; local receipts not attributed")
            return result
        if "paths" not in roots:
            paths = sorted(glob.glob("/tmp/ms-unified-*"))
            roots["paths"], roots["truncated"] = paths[:64], len(paths) > 64
        if roots["truncated"]:
            warnings.append("local roots limited to 64; local progress may be incomplete")
        latest_by_source = {}
        for root in roots["paths"]:
            for row in sources[:100]:
                split = row.get("split")
                if not isinstance(split, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", split):
                    continue
                case = Path(root) / "cases" / split / row["id"]
                receipt, _ = read_json(case / "receipt.json", warnings, 65536)
                if not receipt or receipt.get("source_row") != row or (receipt.get("locator") or {}).get("archive_path") != str(batch / "batch.tar"):
                    continue
                times = {}
                for name in ("receipt.json", "fullframe_manifest.json", "beats.json", "draft.json", "pose.npz"):
                    try:
                        times[name] = (case / name).stat().st_mtime
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        warnings.append(str(case / name) + ": " + str(error))
                receipt_time = times.get("receipt.json", 0)
                if receipt_time < started:
                    continue
                key = (split, row["id"])
                if receipt_time >= latest_by_source.get(key, {}).get("receipt_mtime", 0):
                    latest_by_source[key] = {"root": root, "phase": cfg["phase"], "source_row": row,
                        "status": receipt.get("status"), "receipt_path": str(case / "receipt.json"),
                        "receipt_mtime": receipt_time, "age_seconds": max(0, time.time()-receipt_time),
                        "file_mtimes": times, "profile_seconds": receipt.get("profile_seconds", {}),
                        "published": False}
        result["local_receipts"] = list(latest_by_source.values())
    return result

jobs = json.loads(sys.argv[1])
roots = {}
results = []
for job in jobs:
    try:
        result = collect(job, roots)
        results.append(result)
    except Exception as error:
        results.append({"job": job, "reachable": True, "collection_error": str(error), "warnings": [str(error)]})
print(json.dumps(results, allow_nan=False))
'''


def validate_registry(jobs):
    if isinstance(jobs, dict):
        if jobs.get("schema") != "ms_novlm_live_jobs_registry_v1":
            raise ValueError("unknown registry schema")
        jobs = jobs.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("registry must be a JSON array")
    seen = set()
    for job in jobs:
        if not isinstance(job, dict) or not all(key in job for key in ("node", "host", "phase", "config_path", "task_handle")):
            raise ValueError("registry jobs require node/host/phase/config_path/task_handle")
        if not isinstance(job["host"], str) or not isinstance(job["node"], str) or job["host"] not in HOSTS.values() or HOSTS.get(job["node"]) != job["host"]:
            raise ValueError("host/node outside the explicit three-node allowlist")
        if "gpu" in job and (type(job["gpu"]) is not int or job["gpu"] not in GPU_SCOPE[job["node"]]):
            raise ValueError("registry GPU outside explicit node scope")
        if not isinstance(job["config_path"], str) or not Path(job["config_path"]).is_absolute():
            raise ValueError("config_path must be absolute")
        if not isinstance(job["phase"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job["phase"]):
            raise ValueError("invalid phase")
        key = (job["host"], job["phase"])
        if key in seen:
            raise ValueError("duplicate registry phase")
        seen.add(key)
    return jobs


def collect_host(host, jobs, runner=subprocess.run):
    command = "python3 -B -c " + shlex.quote(REMOTE) + " " + shlex.quote(json.dumps(jobs))
    try:
        completed = runner(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
                           capture_output=True, text=True, timeout=20, check=False)
        if completed.returncode != 0:
            raise ValueError("SSH exit %d: %s" % (completed.returncode, completed.stderr[-1000:]))
        observations = json.loads(completed.stdout)
        if not isinstance(observations, list) or len(observations) != len(jobs):
            raise ValueError("SSH collector returned invalid job count")
        if any(not isinstance(o, dict) or o.get("job") != j for o, j in zip(observations, jobs)):
            raise ValueError("SSH collector returned mismatched job identity")
        return observations
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return [{"job": job, "reachable": False, "warnings": ["SSH collection unavailable: " + str(error)]} for job in jobs]


def collect_jobs(jobs):
    grouped = {}
    for job in jobs:
        grouped.setdefault(job["host"], []).append(job)
    with ThreadPoolExecutor(max_workers=max(1, len(grouped))) as pool:
        futures = [pool.submit(collect_host, host, rows) for host, rows in grouped.items()]
        return [observation for future in futures for observation in future.result()]


def source_key(row):
    if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ("split", "id")):
        raise ValueError("invalid source_row split/id")
    return row["split"], row["id"]


def validate_commit(observation):
    cfg = observation.get("config") or {}
    ticket = observation.get("ticket") or {}
    summary = observation.get("summary") or {}
    commit = observation.get("commit") or {}
    if not summary or not commit:
        return False, [], "summary/commit not both present"
    try:
        n = cfg.get("targetcount")
        rows = ticket.get("sources")
        if type(n) is not int or not 1 <= n <= 100 or not isinstance(rows, list) or len(rows) != n:
            raise ValueError("invalid planned N")
        planned = [source_key(row) for row in rows]
        if len(set(planned)) != n or [row["id"] for row in rows] != cfg.get("selected_source_ids"):
            raise ValueError("planned sources must be exact and unique")
        receipts = commit.get("receipts")
        if not isinstance(receipts, list) or len(receipts) != n or [r.get("source_row") for r in receipts] != rows:
            raise ValueError("actual source rows/order not exactly planned")
        actual = [source_key(r["source_row"]) for r in receipts]
        if len(set(actual)) != n or set(actual) != set(planned):
            raise ValueError("actual source set not exact/unique")
        if commit.get("schema") != "ms_unified_novlm_batch_commit_v2" or commit.get("edition") != cfg.get("phase") or commit.get("ticket_id") != ticket.get("ticket_id") or commit.get("unique_sources_attempted") != n:
            raise ValueError("commit phase/ticket/N mismatch")
        if type(commit.get("unique_sources_attempted")) is not int:
            raise ValueError("commit N must be an integer")
        digest = commit.get("batch_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or commit.get("archive_sha256") != digest or type(commit.get("archive_bytes")) is not int or commit["archive_bytes"] <= 0 or commit.get("archive_path") != str(Path(observation["commit_path"]).parent / "batch.tar"):
            raise ValueError("archive locator/hash/size metadata inconsistent (archive not read)")
        if commit.get("prefetch_drained") is not True or commit.get("vlm_called") is not False or any(commit.get(k) is not False for k in ("quality_approved", "beat_alignment_claimed", "production_complete_claimed")):
            raise ValueError("commit not drained/noVLM/unreviewed")
        if any(any(r.get(k) is not v for k, v in UNREVIEWED.items()) for r in receipts):
            raise ValueError("source noVLM/default unreviewed flags missing or changed")
        if any(r.get("status") not in ("extracted_rule_draft_unapproved", "failed_sealed_no_retry") for r in receipts):
            raise ValueError("unexpected source outcome")
        if summary.get("schema") != "ms_unified_native_summary_v1" or summary.get("selected_source_ids") != cfg.get("selected_source_ids") or summary.get("selected") != n or summary.get("processed") != n or summary.get("source_outcomes") != receipts:
            raise ValueError("summary N/source outcomes mismatch")
        if any(type(summary.get(key)) is not int for key in ("selected", "processed")):
            raise ValueError("summary N must be an integer")
        if summary.get("prefetch_drained") is not True or summary.get("human_review_completed") is not False or summary.get("production_complete_claimed") is not False:
            raise ValueError("summary not drained/unreviewed")
        reference = summary.get("commit_ref") or {}
        if reference.get("path") != observation.get("commit_path") or reference.get("sha256") != observation.get("commit_sha256"):
            raise ValueError("summary commit path/hash mismatch")
        return True, actual, None
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return False, [], str(error)


def summarize_job(observation):
    job = observation["job"]
    result = {**job, "reachable": observation.get("reachable") is True,
              "warnings": list(observation.get("warnings", [])), "worker_finished": False,
              "reliable_published": False, "expected_running": False, "committed_sources": []}
    if not result["reachable"] or observation.get("collection_error"):
        result["status"] = "unreachable" if not result["reachable"] else "collection_error"
        result["warnings"].extend([observation["collection_error"]] if observation.get("collection_error") else [])
        return result
    cfg = observation.get("config") or {}
    result.update(gpus=cfg.get("gpus", []), targetcount=cfg.get("targetcount"),
                  output_root=cfg.get("out"), queue=cfg.get("queue"),
                  worker_process=observation.get("worker_process", {}),
                  manager_process=observation.get("manager_process", {}),
                  resource_samples=observation.get("resource_samples", {}),
                  gpu_sample_source=observation.get("gpu_sample_source"))
    good, keys, reason = validate_commit(observation)
    result["reliable_published"] = good
    result["committed_sources"] = [{"split": split, "id": sid} for split, sid in keys]
    if reason and (observation.get("summary") or observation.get("commit")):
        result["warnings"].append("commit metadata not accepted: " + reason)
    safe = observation.get("resources_safe") or {}
    worker_exit = observation.get("worker_exit") or {}
    phase = cfg.get("phase")
    safe_ok = safe.get("phase") == phase and all(safe.get(k) is True for k in SAFE_FLAGS)
    exit_ok = worker_exit.get("phase") == phase and worker_exit.get("exit_code") == 0 and type(worker_exit.get("exit_code")) is int
    failed = bool(observation.get("failure")) or bool(worker_exit and not exit_ok)
    result["worker_finished"] = good and safe_ok and exit_ok and not failed
    result["resources_safe"] = safe_ok
    result["worker_exit_code"] = worker_exit.get("exit_code")
    paused = bool(observation.get("pending_budget")) or observation.get("known_paused") is True or job.get("expected_running") is False
    alive = (observation.get("worker_process") or {}).get("alive") is True
    result["expected_running"] = alive and not paused and not failed and not worker_exit and not good
    result["status"] = ("worker_finished" if result["worker_finished"] else "failed" if failed else
                        "known_paused" if paused else "published_awaiting_worker_exit_or_resources" if good else
                        "worker_exited_awaiting_commit_or_resources" if worker_exit else
                        "worker_running" if alive else "parent_wait" if (observation.get("manager_process") or {}).get("alive") else "unknown")
    result["local_receipts"] = observation.get("local_receipts", [])
    local_keys = {source_key(r["source_row"]) for r in result["local_receipts"]}
    result["local_completed_unpublished_count"] = len(local_keys)
    result["local_progress_is_not_publication"] = True
    ages = [r["age_seconds"] for r in result["local_receipts"] if "age_seconds" in r]
    result["latest_local_receipt_age_seconds"] = min(ages) if ages else None
    result["outcome_counts"] = dict(Counter(r.get("status") for r in (observation.get("commit") or {}).get("receipts", [])))
    source_profile = Counter()
    for receipt in (observation.get("commit") or {}).get("receipts", []):
        for name, seconds in receipt.get("profile_seconds", {}).items():
            if type(seconds) in (int, float):
                source_profile[name] += seconds
    result["final_profile"] = {"batch_profile_seconds": (observation.get("commit") or {}).get("profile_seconds", {}),
        "source_profile_seconds_totals": dict(source_profile),
        "runtime_gate_profile": (observation.get("summary") or {}).get("runtime_gate_profile", {}),
        "runtime_gate_components": (observation.get("summary") or {}).get("runtime_gate_components", {})}
    result["safety_and_failure_metadata"] = {key: value for key, value in observation.items() if value and
        (key in ("failure", "low_memory_warning", "pending_budget", "child_cleanup_blocked") or key.startswith(("restore_", "holder_exit_wait_", "lease_cleanup_failure_")))}
    return result


class Monitor:
    def __init__(self, interval=60):
        self.interval = interval
        self.seen = set()
        self.baseline = set()
        self.observed_jobs = set()
        self.started_time = None
        self.previous_time = None
        self.previous_complete = False
        self.rate_intervals = deque()
        self.idle = {}
        self.cycle = 0

    def observe(self, observations, now=None, wall=None):
        now = time.monotonic() if now is None else now
        wall = time.time() if wall is None else wall
        jobs = []
        for observation in observations:
            try:
                jobs.append(summarize_job(observation))
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                jobs.append(summarize_job({"job": observation.get("job", {}), "reachable": True,
                    "collection_error": "invalid job metadata: " + str(error), "warnings": []}))
        new = set()
        complete = bool(jobs) and all(j["reachable"] and j["status"] != "collection_error" for j in jobs)
        dt = 0 if self.previous_time is None else max(0, now-self.previous_time)
        if self.started_time is None:
            self.started_time = now
        reliable_interval = complete and self.previous_complete and dt <= max(180, 3*self.interval)
        for job in jobs:
            identity = (job["host"], job["phase"], job["config_path"])
            available = job["reachable"] and job["status"] != "collection_error"
            keys = {source_key(row) for row in job["committed_sources"]}
            if available:
                if identity not in self.observed_jobs:
                    self.baseline.update(keys-self.seen)
                    self.observed_jobs.add(identity)
                else:
                    new.update(keys-self.seen)
                self.seen.update(keys)
            old = self.idle.get(identity, {"seconds": 0, "running": False, "time": now})
            if available and job["expected_running"]:
                elapsed = max(0, now-old["time"])
                seconds = old["seconds"] + (elapsed if old["running"] and elapsed <= max(180, 3*self.interval) else 0)
                if keys:
                    seconds = 0
                self.idle[identity] = {"seconds": seconds, "running": True, "time": now}
                job["no_new_commit_observed_seconds"] = seconds
                age = job.get("latest_local_receipt_age_seconds")
                job["idle_alarm"] = age >= 600 if age is not None else seconds >= 600
                job["idle_alarm_basis"] = "latest_local_receipt_age" if age is not None else "no_receipt_running_observations"
            else:
                self.idle[identity] = {"seconds": 0, "running": False, "time": now}
                job["idle_alarm"] = False
                job["no_new_commit_observed_seconds"] = 0
        self.rate_intervals.append((now-dt, now, len(new), reliable_interval))
        cutoff = now-900
        while self.rate_intervals and self.rate_intervals[0][1] <= cutoff:
            self.rate_intervals.popleft()
        window_seconds = min(900, max(0, now-self.started_time))
        rolling_new = sum(count for _, end, count, _ in self.rate_intervals if end > cutoff)
        incomplete = not complete or any(not reliable and end > max(start, cutoff) for start, end, _, reliable in self.rate_intervals)
        lower_bound = rolling_new*60/window_seconds if window_seconds > 0 else None
        self.cycle += 1
        self.previous_time, self.previous_complete = now, complete
        return {"schema": "ms_novlm_jobs_monitor_v1", "cycle": self.cycle, "observed_unix": wall,
            "observation_complete": complete, "jobs": jobs, "warnings": [w for j in jobs for w in j["warnings"]],
            "committed_id_count": len(self.seen), "committed_source_count": len(self.seen),
            "dedup_key": "global(split,id); local receipts never added to commit counts",
            "baseline_committed_source_count": len(self.baseline),
            "new_committed_source_count": len(self.seen-self.baseline),
            "new_this_cycle": [{"split": split, "id": sid} for split, sid in sorted(new)],
            "rolling_15min": {"window_seconds": window_seconds, "new_sources": rolling_new,
                              "measurement_incomplete": incomplete,
                              "observed_lower_bound_sources_per_minute": lower_bound,
                              "sources_per_minute": lower_bound if not incomplete else None,
                              "new_in_unknown_gap_this_cycle": len(new) if not reliable_interval else 0},
            "worker_finished_jobs": sum(j["worker_finished"] for j in jobs),
            "semantic_approval": False, "training_eligible": False, "production_complete_claimed": False,
            "archive_contents_verified": False, "automatic_restart": False}


def run(registry, out, interval=60, once=False, collector=collect_jobs, clock=time.monotonic, sleeper=time.sleep):
    registry, out = Path(registry), Path(out)
    if not registry.is_absolute() or not out.is_absolute():
        raise ValueError("registry/out paths must be absolute")
    if interval <= 0:
        raise ValueError("interval must be positive")
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out / "monitor.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("monitor already running for this output", file=sys.stderr, flush=True)
            return 2
        monitor = Monitor(interval)
        while not (out / "STOP").exists():
            started = clock()
            # This flushed heartbeat must precede even the slowest SSH collection.
            print(json.dumps({"event": "heartbeat_before_ssh", "cycle": monitor.cycle+1, "unix": time.time()}), flush=True)
            try:
                jobs = validate_registry(json.loads(registry.read_text()))
                observations = collector(jobs)
                summary = monitor.observe(observations, clock())
            except (OSError, ValueError) as error:
                summary = monitor.observe([], clock())
                summary["warnings"].append("registry/collection error: " + str(error))
            encoded = json.dumps(summary, ensure_ascii=False, allow_nan=False)
            temporary = out / "latest_summary.json.tmp"
            temporary.write_text(encoded + "\n")
            os.replace(temporary, out / "latest_summary.json")
            with (out / "monitor.jsonl").open("a") as log:
                log.write(encoded + "\n")
            print(json.dumps({"event": "cycle_complete", "cycle": summary["cycle"],
                              "committed_sources": summary.get("committed_source_count"),
                              "finished_jobs": summary.get("worker_finished_jobs"),
                              "warning_count": len(summary["warnings"])}), flush=True)
            if once:
                return 0
            remaining = max(0, interval-(clock()-started))
            deadline = clock()+remaining
            while not (out / "STOP").exists() and clock() < deadline:
                sleeper(min(1, max(0, deadline-clock())))
    print("monitor stopped normally by output STOP", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, help="absolute JSON registry path (array or schema/jobs envelope)")
    parser.add_argument("--out", default=str(ROOT / "ms_novlm_jobs_monitor_v1"))
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args.registry, args.out, args.interval, args.once)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
