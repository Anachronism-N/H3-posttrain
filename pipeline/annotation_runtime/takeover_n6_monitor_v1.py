#!/usr/bin/env python3
"""Read-only observations of new n6g23 terminals; no pool-wide verdicts."""

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

BASE = Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1")
DEFAULT_CONFIG = BASE / "runtime/caption_union_native_v4_n6pilot_20261026_n6g23.json"
DEFAULT_OUT = BASE / "takeover_n6_monitor_v1"
WORKER_ROOT = BASE / "caption_union_native_v4_n6pilot_20261026/workers/n6g23"
MAX_JSON_BYTES = 256 * 1024
MAX_PROGRESS_BYTES = 1024 * 1024
FLAGS = dict(training_eligible=False, semantic_approval=False, quality_approved=False,
             needs_visual_revision=True, pixel_quality_review_completed=False)
STATUSES = ("draft_generated", "draft_partial", "draft_needs_regeneration")
SUCCESS = STATUSES[:2]
SCOPE = "single_lane_observed_since_start"


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def reject_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def read_json(path, limit=MAX_JSON_BYTES):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb", buffering=0) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"JSON exceeds {limit // 1024} KiB or is not a regular file")
        chunks, remaining = [], info.st_size
        while remaining:
            chunk = stream.read(min(4096, remaining))
            if not chunk:
                raise ValueError("JSON changed or was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.fstat(stream.fileno()).st_size != info.st_size:
            raise ValueError("JSON size changed during read")
        raw = b"".join(chunks)
    if b"\x00" in raw:
        raise ValueError("NUL in JSON")
    value = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def progress_fields(progress, pool_sha):
    rows = progress["completed_source_keys"]
    if not isinstance(rows, list):
        raise ValueError("completed_source_keys must be a list")
    keys = set()
    for row in rows:
        if not (isinstance(row, list) and len(row) == 2
                and isinstance(row[0], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", row[0])
                and isinstance(row[1], str) and row[1] in ("train", "val", "test")):
            raise ValueError("invalid source key")
        keys.add(tuple(row))
    counts = progress["counts"]
    if not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("invalid counts")
    for name in ("processed", "reserved"):
        if type(progress[name]) is not int or progress[name] < 0:
            raise ValueError("invalid " + name)
    if not isinstance(progress["UTC"], str):
        raise ValueError("invalid UTC")
    if progress.get("worker_id", "n6g23") != "n6g23" or progress.get("pool_sha256", pool_sha) != pool_sha:
        raise ValueError("foreign progress")
    return keys, {name: progress[name] for name in ("counts", "processed", "reserved", "UTC")}


def terminal_status(terminal, key, pool_sha):
    if terminal.get("id") != key[0] or terminal.get("split") != key[1]:
        raise ValueError("terminal id/split mismatch")
    owner = terminal.get("owner")
    if not isinstance(owner, dict) or owner.get("worker_id") != "n6g23" or owner.get("node") != "N6":
        raise ValueError("terminal owner mismatch")
    if terminal.get("pool_sha256") != pool_sha:
        raise ValueError("terminal pool mismatch")
    if any(terminal.get(name) is not value for name, value in FLAGS.items()):
        raise ValueError("terminal draft flags mismatch")
    status = terminal.get("status")
    if status not in STATUSES:
        raise ValueError("unknown terminal status")
    return status


def status_counts(counter):
    return {status: counter[status] for status in STATUSES}


class Monitor:
    def __init__(self, config, worker_root=WORKER_ROOT, clock=time.monotonic):
        if config.get("node") != "N6" or config.get("pool_worker_id") != "n6g23":
            raise ValueError("only N6/n6g23 is supported")
        self.progress_path = Path(config["draft_out"]) / "progress.json"
        self.pool_sha = config["plan"]["sha256"]
        if not isinstance(self.pool_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", self.pool_sha):
            raise ValueError("invalid plan sha256")
        self.worker_root = Path(worker_root)
        self.clock = clock
        self.started = clock()
        self.started_utc = utc_now()
        self.baseline = None
        self.baseline_utc = None
        self.baseline_started = None
        self.seen = set()
        self.pending = {}
        self.counts = Counter()
        self.events = deque()
        self.last_success = None
        self.last_progress = None

    def sample(self):
        now = self.clock()
        state, progress_error = "observing", None
        try:
            keys, fields = progress_fields(read_json(self.progress_path, MAX_PROGRESS_BYTES), self.pool_sha)
            self.last_progress = fields
            if self.baseline is None:
                self.baseline = keys
                self.baseline_utc = utc_now()
                self.baseline_started = now = self.clock()
                self.last_success = now
                state = "baseline_initialized"
            else:
                for key in keys - self.baseline - self.seen:
                    self.pending.setdefault(key, "awaiting terminal")
        except (OSError, ValueError, KeyError, TypeError) as error:
            state = "pending_start" if self.baseline is None else "progress_pending"
            progress_error = type(error).__name__ + ": " + str(error)
        new_counts = Counter()
        new_sources = []
        for key in sorted(self.pending):
            path = self.worker_root / "terminals" / (digest([self.pool_sha, *key]) + ".json")
            try:
                status = terminal_status(read_json(path), key, self.pool_sha)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.pending[key] = type(error).__name__ + ": " + str(error)
                continue
            self.seen.add(key)
            del self.pending[key]
            self.counts[status] += 1
            new_counts[status] += 1
            self.events.append((now, status))
            new_sources.append(dict(id=key[0], split=key[1], status=status))
            if status in SUCCESS:
                self.last_success = now
        while self.events and self.events[0][0] <= now - 900:
            self.events.popleft()
        rolling = Counter(status for _, status in self.events)
        success_total = sum(self.counts[status] for status in SUCCESS)
        rolling_success = sum(rolling[status] for status in SUCCESS)
        duration = 0 if self.baseline_started is None else min(900, max(0, now - self.baseline_started))
        no_success = None if self.last_success is None else max(0, now - self.last_success)
        return dict(
            schema="takeover_n6_monitor_v1", scope=SCOPE, worker_id="n6g23", node="N6",
            UTC=utc_now(), session_started_UTC=self.started_utc, monitor_pid=os.getpid(),
            state=state, baseline_UTC=self.baseline_utc,
            baseline_source_count=None if self.baseline is None else len(self.baseline),
            source_progress=self.last_progress, progress_error=progress_error,
            observed_since_start=dict(status_counts(self.counts),
                                      success_generated_or_partial=success_total,
                                      regeneration=self.counts[STATUSES[2]]),
            new_since_previous_observation=dict(counts=status_counts(new_counts), sources=new_sources),
            pending_count=len(self.pending),
            pending_sources=[dict(id=k[0], split=k[1], reason=v) for k, v in sorted(self.pending.items())],
            rolling_15min=dict(counts=status_counts(rolling),
                               success_generated_or_partial=rolling_success,
                               regeneration=rolling[STATUSES[2]], observed_seconds=duration,
                               full_window=duration >= 900,
                               success_per_minute=rolling_success * 60 / duration if duration else None),
            seconds_without_new_success=no_success,
            alarms=["no_new_generated_or_partial_for_10min"] if no_success is not None and no_success >= 600 else [])


def publish(out, summary):
    raw = encoded(summary)
    temporary = out / "summary_latest.json.tmp"
    with temporary.open("wb") as stream:
        stream.write(raw)
    os.replace(temporary, out / "summary_latest.json")
    with (out / "observations.jsonl").open("ab") as stream:
        stream.write(raw)


def local_lock_path(out):
    return Path(tempfile.gettempdir()) / ("takeover_n6_monitor_v1-" + digest(str(out.resolve())) + ".lock")


def run(args):
    monitor = Monitor(read_json(args.config))
    out = Path(args.out).resolve()
    with local_lock_path(out).open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("monitor_already_running", file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        while not (out / "STOP").exists():
            summary = monitor.sample()
            publish(out, summary)
            print(encoded(summary).decode(), end="", flush=True)
            if args.once:
                return 0
            deadline = time.monotonic() + args.interval
            while time.monotonic() < deadline:
                if (out / "STOP").exists():
                    break
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        print("stopped: monitor output STOP observed", flush=True)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("monitor_error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
