#!/usr/bin/env python3
"""Metadata-only observation. Config explicitly selects a queue-bound current ticket.

Config: {"nodes": [{"queue_path": "/.../node.queue.json",
"current_ticket_path": "/.../node.ticket.json", "layout": "flat"}]}
Layouts: flat batches/gpuN-[run-]bNNNN; nested batches/gpuN/run-batchNNN.
No queue-tail guessing, assignment/media/archive reads, models, claims or signals.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import time

ROOT = pathlib.Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1")
PARTITIONS = ("assignment row index modulo 4 equals physical gpu index", "zero-based assignment row index modulo4 equals GPU index; do not reuse source_ordinal modulo")
MAX_METADATA = 1024 * 1024
MAX_ENTRIES = 12000
MAX_PENDING = 128
MAX_PER_LANE = 12
FLAGS = ("semantic_approval", "quality_approved", "training_eligible")
SUCCESS = "extracted_rule_draft_unapproved"
FAILURE = "failed_sealed_no_retry"
SHA = re.compile(r"[0-9a-f]{64}")
FLAT = re.compile(r"gpu([0-3])-(?:[A-Za-z0-9_.-]+-)?b[0-9]+")
NESTED = re.compile(r"[A-Za-z0-9_.-]+-batch[0-9]+")


def fingerprint(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def metadata(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_METADATA:
            raise ValueError("metadata_not_regular_or_over_1MiB")
        raw = f.read(MAX_METADATA + 1)
        after = os.fstat(f.fileno())
    if len(raw) > MAX_METADATA or b"\0" in raw or fingerprint(before) != fingerprint(after):
        raise ValueError("invalid_or_mutating_metadata")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("metadata_object_required")
    return obj, hashlib.sha256(raw).hexdigest(), fingerprint(after)


def absolute(path):
    if not isinstance(path, str) or not pathlib.Path(path).is_absolute() or ".." in pathlib.Path(path).parts:
        raise ValueError("absolute_explicit_path_required")
    return pathlib.Path(path)


def bound_ticket(spec):
    queue_path = absolute(spec["queue_path"])
    ticket_path = absolute(spec["current_ticket_path"])
    queue, _, _ = metadata(queue_path)
    if queue.get("schema") != "ms_external_authorized_rollover_queue_v1":
        raise ValueError("queue_schema_mismatch")
    refs = queue.get("tickets")
    if not isinstance(refs, list) or not 1 <= len(refs) <= 32:
        raise ValueError("bounded_ticket_refs_required")
    matches = [r for r in refs if isinstance(r, dict) and r.get("path") == str(ticket_path)]
    if len(matches) != 1 or not SHA.fullmatch(str(matches[0].get("sha256", ""))):
        raise ValueError("current_ticket_not_uniquely_queue_bound")
    ticket, ticket_sha, _ = metadata(ticket_path)
    if ticket_sha != matches[0]["sha256"]:
        raise ValueError("ticket_sha_mismatch")
    if ticket.get("schema") != "ms_external_unified_assignment_v1" or ticket.get("worker_id") != queue.get("worker_id"):
        raise ValueError("ticket_worker_schema_mismatch")
    gpus = ticket.get("allowed_gpu_indices")
    if not isinstance(gpus, list) or not gpus or len(gpus) > 4 or len(set(gpus)) != len(gpus) or any(type(g) is not int or not 0 <= g < 4 for g in gpus):
        raise ValueError("ticket_gpu_scope_mismatch")
    if type(ticket.get("source_count")) is not int or not 0 < ticket["source_count"] <= 1000000:
        raise ValueError("ticket_source_count_not_bounded")
    if ticket.get("publication_sources_per_batch") != 20 or ticket.get("gpu_partition") not in PARTITIONS:
        raise ValueError("ticket_batch_partition_mismatch")
    if not SHA.fullmatch(str(ticket.get("assignment_sha256", ""))) or not isinstance(ticket.get("edition"), str):
        raise ValueError("ticket_assignment_binding_missing")
    if ticket.get("no_VLM") is not True or any(ticket.get(k) is not False for k in FLAGS):
        raise ValueError("ticket_unreviewed_policy_mismatch")
    if spec.get("layout") not in ("flat", "nested"):
        raise ValueError("explicit_known_layout_required")
    absolute(ticket["output_root"])
    return ticket, ticket_sha


def paths_for_node(ticket, layout, counters):
    root = absolute(ticket["output_root"]) / "batches"
    result = {g: set() for g in ticket["allowed_gpu_indices"]}
    # One flat node enumeration, not four copies of the same directory scan.
    roots = [(None, root)] if layout == "flat" else [(g, root / ("gpu%d" % g)) for g in result]
    total = 0
    for gpu, directory in roots:
        if not stat.S_ISDIR(directory.stat(follow_symlinks=False).st_mode):
            raise ValueError("current_batch_root_not_regular_directory")
        counters["directory_enumerations"] += 1
        with os.scandir(directory) as entries:
            for entry in entries:
                total += 1
                counters["directory_entries"] += 1
                if total > MAX_ENTRIES:
                    raise ValueError("node_directory_entry_bound_exceeded")
                match = FLAT.fullmatch(entry.name) if layout == "flat" else NESTED.fullmatch(entry.name)
                if match:
                    lane = int(match.group(1)) if layout == "flat" else gpu
                    if lane in result:
                        if not entry.is_dir(follow_symlinks=False):
                            raise ValueError("batch_directory_not_regular_directory")
                        result[lane].add(str(directory / entry.name / "commit.json"))
    return result


def commit_evidence(path, ticket, ticket_sha, gpu, layout):
    c, sha, fp = metadata(path)
    if c.get("schema") != "ms_unified_novlm_batch_commit_v1" or c.get("worker") != ticket["worker_id"] or type(c.get("gpu")) is not int or c["gpu"] != gpu or c.get("edition") != ticket["edition"] or c.get("assignment_sha256") != ticket["assignment_sha256"]:
        raise ValueError("commit_ticket_binding_mismatch")
    receipts = c.get("receipts")
    if not isinstance(receipts, list) or not 1 <= len(receipts) <= 20 or any(not isinstance(r, dict) for r in receipts):
        raise ValueError("bounded_receipts_required")
    if any(c.get(k) is True for k in FLAGS) or c.get("vlm_called") is True or any(any(r.get(k) is True for k in FLAGS) or r.get("vlm_called") is True for r in receipts):
        raise ValueError("explicit_approval_or_vlm_true_rejects_entire_batch")
    if any(c.get(k) is not False for k in FLAGS):
        raise ValueError("commit_unreviewed_flags_missing")
    indices = [r["source_row"]["assignment_index"] if layout == "flat" else r["assignment_index"] for r in receipts]
    if layout == "nested" and c.get("assignment_indices") != indices:
        raise ValueError("commit_assignment_indices_mismatch")
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < ticket["source_count"] or i % 4 != gpu for i in indices):
        raise ValueError("assignment_partition_or_scope_mismatch")
    records = []
    gaps = []
    for r, index in zip(receipts, indices):
        s = r["source_row"]
        split, identity = s.get("split"), s.get("id")
        if split not in ("train", "val", "test") or not isinstance(identity, str) or not identity or pathlib.Path(identity).name != identity or identity in (".", ".."):
            raise ValueError("source_identity_or_split_mismatch")
        video, audio = absolute(s["video_path"]), absolute(s["audio_path"])
        if video.stem != identity or audio.stem != identity:
            raise ValueError("source_path_identity_mismatch")
        if layout == "nested" and "assignment_index" in s and s["assignment_index"] != index:
            raise ValueError("nested_source_assignment_index_mismatch")
        record = {"split": split, "id": identity, "video_path": str(video), "audio_path": str(audio), "assignment_index": index, "status": r.get("status")}
        for k in ("source_ordinal", "meta_path"):
            if k in s:
                record[k] = s[k]
        for k in ("vlm_called",) + FLAGS:
            if k in r:
                record[k] = r[k]
        if r.get("status") == SUCCESS:
            if r.get("vlm_called") is not False or any(r.get(k) is not False for k in FLAGS):
                raise ValueError("successful_receipt_strict_unreviewed_flags_missing")
            record["no_vlm_proven"] = True
        elif r.get("status") == FAILURE:
            record["no_vlm_proven"] = r.get("vlm_called") is False
            missing = []
            if r.get("vlm_called") is not False:
                missing.append("no_vlm_not_proven")
            missing.extend(k + "_false_not_proven" for k in FLAGS if r.get(k) is not False)
            reason = {k: r[k] for k in ("reason", "error", "failure_reason") if k in r}
            record["failure_evidence"] = reason
            if not reason:
                missing.append("failure_reason_not_declared")
            record["evidence_gaps"] = missing
            gaps.extend(missing)
        else:
            raise ValueError("unknown_receipt_status")
        records.append(record)
    expected_archive = pathlib.Path(path).parent / "batch.tar"
    if layout == "flat":
        declaration = c["package"]
    else:
        pubs = c.get("publications")
        if not isinstance(pubs, list):
            raise ValueError("publications_required")
        archives = [p for p in pubs if isinstance(p, dict) and p.get("path") == str(expected_archive)]
        if len(archives) != 1:
            raise ValueError("single_current_batch_archive_required")
        declaration = archives[0]["receipt"]
        if archives[0].get("sha256") != c.get("batch_sha256"):
            raise ValueError("archive_publication_sha_mismatch")
    if declaration.get("path") != str(expected_archive) or not SHA.fullmatch(str(c.get("batch_sha256", ""))) or declaration.get("sha256") != c["batch_sha256"] or type(declaration.get("bytes")) is not int or declaration["bytes"] <= 0:
        raise ValueError("archive_declaration_binding_mismatch")
    archive_stat = expected_archive.stat(follow_symlinks=False)
    if not stat.S_ISREG(archive_stat.st_mode) or archive_stat.st_size != declaration["bytes"]:
        raise ValueError("archive_existence_size_mismatch")
    event = {"commit_path": str(path), "commit_sha256": sha, "ticket_path_binding_sha256": ticket_sha, "assignment_sha256": ticket["assignment_sha256"], "worker": ticket["worker_id"], "gpu": gpu, "edition": ticket["edition"], "layout": layout, "archive_path": str(expected_archive), "archive_bytes_observed": archive_stat.st_size, "archive_sha_declaration": c["batch_sha256"], "archive_payload_read": False, "records": records, "evidence_gaps": sorted(set(gaps))}
    for k in ("owner_pid", "commit_unix", "run_id", "ticket_id"):
        if k in c:
            event[k] = c[k]
    return event, fp


class Patrol:
    def __init__(self, nodes, output_dir, interval=900):
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 6:
            raise ValueError("one_to_six_explicit_node_specs_required")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("positive_finite_interval_required")
        self.nodes = nodes
        self.output = pathlib.Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        self.interval = interval
        self.state = {}
        self.success = set()
        self.failed = set()
        self.observed = set()
        self.baseline_keys = set()
        self.incomplete = False
        self.started_mono = None
        self.previous_mono = None
        self.previous_wall = None
        self.sample_index = 0

    def append(self, name, obj):
        with (self.output / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, sort_keys=True) + "\n")

    def publish(self, name, obj):
        p = self.output / name
        temp = self.output / (name + ".tmp")
        temp.write_text(json.dumps(obj, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, p)

    def sample(self):
        mono, wall = time.monotonic(), time.time()
        elapsed = None if self.previous_mono is None else mono - self.previous_mono
        wall_elapsed = None if self.previous_wall is None else wall - self.previous_wall
        if self.started_mono is None:
            self.started_mono = mono
        errors = []
        counters = {"directory_enumerations": 0, "directory_entries": 0, "commit_metadata_reads": 0}
        if elapsed is not None and (elapsed > self.interval + max(5, self.interval * .1) or elapsed <= 0 or abs(wall_elapsed - elapsed) > max(5, self.interval * .1)):
            errors.append({"error": "wall_clock_or_sampling_event_gap", "monotonic_seconds": elapsed, "wall_seconds": wall_elapsed})
        before_success, before_failed, before_observed = len(self.success), len(self.failed), len(self.observed)
        lane_reports = []
        for node_index, spec in enumerate(self.nodes):
            try:
                ticket, ticket_sha = bound_ticket(spec)
                binding = (ticket_sha, ticket["output_root"], spec["layout"])
                node = self.state.get(node_index)
                if node is not None and node["binding"] != binding:
                    raise ValueError("immutable_current_ticket_or_root_changed")
                visible = paths_for_node(ticket, spec["layout"], counters)
                if node is None:
                    node = {"binding": binding, "lanes": {}, "index_identity": {}}
                    self.state[node_index] = node
                for gpu, names in visible.items():
                    lane = node["lanes"].get(gpu)
                    report = {"worker": ticket["worker_id"], "gpu": gpu, "current_output_root": ticket["output_root"], "ticket_sha256": ticket_sha, "baseline": lane is None, "new_commits": 0, "commit_metadata_reads": 0}
                    lane_reports.append(report)
                    initial = lane is None
                    if initial:
                        lane = {"seen": set(names), "pending": collections.OrderedDict(), "read": {}}
                        node["lanes"][gpu] = lane
                        committed = []
                        for path in sorted(names):
                            try:
                                st = pathlib.Path(path).stat(follow_symlinks=False)
                                committed.append((st.st_mtime_ns, path))
                            except FileNotFoundError:
                                if len(lane["pending"]) < MAX_PENDING:
                                    lane["pending"][path] = False
                                else:
                                    errors.append({"worker": ticket["worker_id"], "gpu": gpu, "error": "baseline_missing_commit_pending_overflow_event_gap"})
                        candidates = [(p, True) for _, p in sorted(committed)[-2:]]
                    else:
                        vanished = lane["seen"] - names
                        if vanished:
                            errors.append({"worker": ticket["worker_id"], "gpu": gpu, "error": "batch_directory_disappeared_event_gap", "count": len(vanished)})
                        new_names = names - lane["seen"]
                        if len(lane["seen"] | names) > MAX_ENTRIES:
                            raise ValueError("lane_cursor_name_bound_exceeded")
                        lane["seen"].update(names)
                        for path in sorted(new_names):
                            if len(lane["pending"]) >= MAX_PENDING:
                                errors.append({"worker": ticket["worker_id"], "gpu": gpu, "error": "pending_overflow_event_gap"})
                                break
                            lane["pending"][path] = False
                        # Stat immutable commits; unchanged metadata is never re-read.
                        for path, old in lane["read"].items():
                            try:
                                if fingerprint(pathlib.Path(path).stat(follow_symlinks=False)) != old["fingerprint"]:
                                    raise ValueError("immutable_commit_changed_sha_not_recredited:" + old["sha"])
                            except Exception as e:
                                errors.append({"worker": ticket["worker_id"], "gpu": gpu, "path": path, "error": str(e)})
                        candidates = [lane["pending"].popitem(last=False) for _ in range(min(MAX_PER_LANE, len(lane["pending"])))]
                    for path, is_baseline in candidates:
                        try:
                            # Missing commit names are retried without opening payloads.
                            pathlib.Path(path).stat(follow_symlinks=False)
                            counters["commit_metadata_reads"] += 1
                            report["commit_metadata_reads"] += 1
                            event, fp = commit_evidence(path, ticket, ticket_sha, gpu, spec["layout"])
                            for record in event["records"]:
                                key = (record["split"], record["id"])
                                old_key = node["index_identity"].get(record["assignment_index"])
                                if old_key is not None and old_key != key:
                                    raise ValueError("assignment_index_source_identity_changed")
                            for record in event["records"]:
                                node["index_identity"][record["assignment_index"]] = (record["split"], record["id"])
                            lane["read"][path] = {"sha": event["commit_sha256"], "fingerprint": fp}
                            event["baseline_not_new_production"] = is_baseline
                            event["sample_index"] = self.sample_index
                            self.append("commit_events.jsonl", event)
                            if event["evidence_gaps"]:
                                errors.append({"worker": ticket["worker_id"], "gpu": gpu, "path": path, "error": "receipt_evidence_gap", "evidence_gaps": event["evidence_gaps"]})
                            for r in event["records"]:
                                key = (r["split"], r["id"])
                                if is_baseline:
                                    self.baseline_keys.add(key)
                                elif key not in self.baseline_keys:
                                    self.observed.add(key)
                                    (self.success if r["status"] == SUCCESS else self.failed).add(key)
                            if not is_baseline:
                                report["new_commits"] += 1
                        except FileNotFoundError:
                            # Only a genuinely missing commit is ordinary pending.
                            if not pathlib.Path(path).exists():
                                if len(lane["pending"]) < MAX_PENDING:
                                    lane["pending"][path] = is_baseline
                                else:
                                    errors.append({"worker": ticket["worker_id"], "gpu": gpu, "error": "missing_commit_pending_overflow_event_gap"})
                            else:
                                errors.append({"worker": ticket["worker_id"], "gpu": gpu, "path": path, "error": "declared_archive_missing"})
                        except Exception as e:
                            errors.append({"worker": ticket["worker_id"], "gpu": gpu, "path": path, "error": type(e).__name__ + ": " + str(e)})
                    report["pending_commits"] = len(lane["pending"])
                    report["cursor_read_commits"] = len(lane["read"])
                    if not initial and lane["pending"] and any(pathlib.Path(p).is_file() for p in lane["pending"]):
                        errors.append({"worker": ticket["worker_id"], "gpu": gpu, "error": "committed_candidate_budget_deferred_event_gap"})
            except Exception as e:
                errors.append({"node_index": node_index, "error": type(e).__name__ + ": " + str(e)})
        self.incomplete = self.incomplete or bool(errors)
        new_success = len(self.success) - before_success
        summary = {"schema": "ms_external_incremental_patrol_v2", "sample_index": self.sample_index, "sample_started_unix": wall, "sample_finished_unix": time.time(), "sample_duration_seconds": time.monotonic() - mono, "window_monotonic_seconds": elapsed, "window_wall_seconds": wall_elapsed, "total_observation_monotonic_seconds": mono - self.started_mono, "baseline_sample": self.sample_index == 0, "new_successful_unique_sources_lower_bound": new_success, "new_failed_unique_receipts_lower_bound": len(self.failed) - before_failed, "new_observed_unique_sources_lower_bound": len(self.observed) - before_observed, "cumulative_successful_unique_sources_lower_bound": len(self.success), "cumulative_failed_unique_receipts_lower_bound": len(self.failed), "measurement_incomplete": self.incomplete, "sample_measurement_incomplete": bool(errors), "observed_success_rate_per_hour": None if self.incomplete or elapsed is None or elapsed <= 0 else new_success * 3600 / elapsed, "rate_scope": "newly observed unique strict-success receipts of explicitly selected queue-authorized editions; not GPU liveness, corpus completion or semantic quality", "current_gpu_liveness_proven": False, "whole_production_complete": False, "monitor_GPU_calls": 0, "monitor_process_signals": 0, "monitor_claim_operations": 0, "archive_payload_reads": 0, "errors": errors, "io_counters": counters, "lanes": lane_reports}
        try:
            self.append("samples.jsonl", summary)
        except OSError as e:
            self.incomplete = summary["measurement_incomplete"] = summary["sample_measurement_incomplete"] = True
            summary["observed_success_rate_per_hour"] = None
            summary["errors"].append({"error": "sample_log_write_failed: " + str(e)})
        try:
            self.publish("latest_summary.json", summary)
        finally:
            self.publish("heartbeat.json", {"schema": "ms_external_incremental_patrol_heartbeat_v2", "sample_index": self.sample_index, "updated_unix": time.time(), "measurement_incomplete": summary["measurement_incomplete"], "sample_finished": True})
        self.previous_mono, self.previous_wall = mono, wall
        self.sample_index += 1
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, help="new directory, never existing monitor output")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=900)
    args = parser.parse_args()
    config, _, _ = metadata(absolute(args.config))
    patrol = Patrol(config["nodes"], args.output_dir, args.interval_seconds)
    deadline = time.monotonic()
    while True:
        summary = patrol.sample()
        print(json.dumps(summary, sort_keys=True), flush=True)
        if args.once:
            return
        deadline += args.interval_seconds
        now = time.monotonic()
        if deadline <= now:
            deadline = now + args.interval_seconds
        time.sleep(max(0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()
