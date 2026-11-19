"""Finite continuous noVLM worker; native lease covers every committed batch.

Model/preprocessing/rules/publication are frozen v2 helpers. CLI remains deny.
Only a durable batch commit permits reclaiming that batch's own local cache.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import time

import ms_unified_novlm_v2 as core
from ms_unified_novlm_v2 import (
    GIB, FILES, CACHE_SCHEMA, json_bytes, write_once, claim_once,
    publish_archive_once, artifact_metadata, resource_snapshot, deny_gate,
    load_models, copy_source, extract_and_rule, probe_video, decode_bgr,
    audio_metadata,
)

SCHEMA = "ms_unified_novlm_ticket_v3"
SUMMARY_SCHEMA = "ms_unified_novlm_continuous_summary_v1"
PROGRESS_SCHEMA = "ms_unified_novlm_continuous_progress_v1"


def stop_requested(ticket):
    if not ticket.get("stop_file"):
        return False
    path = Path(ticket["stop_file"])
    if path.resolve() != path:
        raise ValueError("STOP path may not traverse symlinks")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("STOP must be a regular non-symlink file")
    finally:
        os.close(fd)
    return True


def check_ticket(ticket, limit=3):
    if not isinstance(ticket, dict) or ticket.get("schema") != SCHEMA or type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError("Explicit v3 ticket and finite limit 1..10000 required")
    sources = ticket.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 10000:
        raise ValueError("Finite explicit assignment of 1..10000 sources required")
    batch_sources = ticket.get("batch_sources", 20)
    if type(batch_sources) is not int or not 1 <= batch_sources <= 100:
        raise ValueError("batch_sources must be 1..100 (default 20)")
    if "stop_file" in ticket and (not isinstance(ticket["stop_file"], str) or not Path(ticket["stop_file"]).is_absolute()):
        raise ValueError("Optional stop_file must be an explicit absolute path")
    stop_requested(ticket)
    seen = set()
    for offset in range(0, len(sources), 100):
        chunk = {**ticket, "schema": core.SCHEMA, "sources": sources[offset:offset + 100]}
        _, budget = core.check_ticket(chunk, 100)
        for row in chunk["sources"]:
            if row["id"] in seen:
                raise ValueError("Duplicate source ID, including split mixing")
            seen.add(row["id"])
    if len(json_bytes(ticket)) > 10 * 1024 ** 2:
        raise ValueError("Ticket metadata exceeds 10 MiB")
    return sources[:limit], budget


def outcome_counts(receipts):
    statuses = dict(Counter(item["status"] for item in receipts))
    drafts = Counter(item.get("draft_status") for item in receipts)
    counts = {"generated": statuses.get("extracted_rule_draft_unapproved", 0),
              "partial": drafts["partial"], "needs_regeneration": drafts["needs_regeneration"],
              "failed_sealed_no_retry": statuses.get("failed_sealed_no_retry", 0),
              "existing_claim_not_reassigned": statuses.get("existing_claim_not_reassigned", 0)}
    return counts, statuses


def publish_progress(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Progress target may not be a symlink")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".progress-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)


def run_batch(ticket, sources, batch_index, offset, edition, local_root, budget,
              models, probe, decoder, audio_info, stage_gate):
    batch_name = ticket["ticket_id"] + f"-b{batch_index:06d}"
    batch = edition / "batches" / batch_name
    # Preserve uncommitted case evidence on any publication/gate exception.
    work = Path(tempfile.mkdtemp(prefix="ms-unified-v3-", dir=local_root))
    started_batch = time.perf_counter()
    receipts = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending, todo = deque(), iter(sources)

        def enqueue():
            row = next(todo, None)
            if row is None:
                return False
            relative = Path("cases") / row["split"] / row["id"]
            try:
                claim_once(edition / "claims" / "sources" / (row["id"] + ".json"),
                           {"source_row": row, "ticket_id": ticket["ticket_id"], "batch_name": batch_name,
                            "owner_pid": os.getpid(), "budget_consumed": True, "requeue_allowed": False})
            except FileExistsError:
                result = {"schema": "ms_unified_novlm_case_receipt_v2", "source_row": row,
                          "status": "existing_claim_not_reassigned", "input_copy_completed": False,
                          "vlm_called": False, "rgb_or_crops_written": False,
                          **artifact_metadata(row, batch / relative, ("receipt.json",)), **core.v2.FLAGS}
                write_once(work / relative / "receipt.json", json_bytes(result))
                receipts[row["id"]] = result
                if stage_gate:
                    stage_gate(False)
                return True
            profile = {}
            source_started = time.perf_counter()
            pending.append((row, executor.submit(copy_source, row, work, budget // 4, profile), profile, source_started))
            return True

        while len(pending) < 2 and enqueue():
            pass
        while pending:
            row, future, case_profile, source_started = pending.popleft()
            relative = Path("cases") / row["split"] / row["id"]
            case = work / relative
            copied_ok = False
            try:
                copied, hashes = future.result()
                copied_ok = True
                used = sum(path.stat().st_size for path in (work / "cases").rglob("*") if path.is_file())
                result = extract_and_rule(row, copied, hashes, models, case, batch / relative,
                                          probe, decoder, budget // 2 - used, audio_info, case_profile, source_started)
            except Exception as error:
                case_profile["whole_case_e2e"] = time.perf_counter() - source_started
                result = {"schema": "ms_unified_novlm_case_receipt_v2", "source_row": row,
                          "status": "failed_sealed_no_retry", "input_copy_completed": copied_ok,
                          "profile_seconds": case_profile, "error": str(error), "budget_consumed": True,
                          "vlm_called": False, "rgb_or_crops_written": False,
                          **artifact_metadata(row, batch / relative, ("receipt.json",)), **core.v2.FLAGS}
                write_once(case / "receipt.json", json_bytes(result))
            receipts[row["id"]] = result
            inputs = work / row["split"] / row["id"]
            if inputs.exists():
                shutil.rmtree(inputs)
            if stage_gate:
                stage_gate(False)
            while len(pending) < 2 and enqueue():
                pass
    ordered = [receipts[row["id"]] for row in sources]
    started = time.perf_counter()
    archive_index = {}
    with tempfile.TemporaryFile(dir=local_root) as archive:
        with tarfile.open(fileobj=archive, mode="w") as pack:
            for result in ordered:
                row, members = result["source_row"], result["artifact_members"]
                archive_index[row["id"]] = {"source_row": row, "source_case_prefix": result["locator"]["source_case_prefix"],
                                            "members": members, "receipt_member": members["receipt.json"]}
                for member in members.values():
                    pack.add(work / member, arcname=member, recursive=False)
        archive_bytes, archive_sha256 = publish_archive_once(batch / "batch.tar", archive)
    profile = {"pack_publish": time.perf_counter() - started, "whole_e2e": time.perf_counter() - started_batch}
    commit = {"schema": "ms_unified_novlm_batch_commit_v3", "ticket_id": batch_name,
              "parent_ticket_id": ticket["ticket_id"], "batch_index": batch_index,
              "selected_range": {"start": offset, "end": offset + len(sources)},
              "edition": ticket["edition"], "batch_sha256": archive_sha256, "archive_sha256": archive_sha256,
              "archive_bytes": archive_bytes, "archive_path": str(batch / "batch.tar"),
              "archive_index": archive_index, "receipts": ordered, "prefetch_drained": not pending,
              "profile_seconds": profile, "publication_scope": "one archive + commit, no shared per-source files",
              "unique_sources_attempted": sum(item["status"] != "existing_claim_not_reassigned" for item in ordered),
              "quality_approved": False, "beat_alignment_claimed": False,
              "production_complete_claimed": False, "vlm_called": False}
    raw = json_bytes(commit)
    write_once(batch / "commit.json", raw)
    shutil.rmtree(work)  # Only our own cache, only after durable archive+commit.
    return commit, str(batch / "commit.json"), hashlib.sha256(raw).hexdigest()


def run_ticket(ticket, limit=3, resource_gate=deny_gate, model_loader=load_models, probe=probe_video,
               decoder=decode_bgr, resource_reader=None, audio_info=audio_metadata, stage_gate=None):
    started_e2e = time.perf_counter()
    ticket = json.loads(json_bytes(ticket))
    sources, budget = check_ticket(ticket, limit)
    local_root = Path(ticket["local_cache_root"])
    with resource_gate(ticket) as evidence:
        if not isinstance(evidence, dict) or evidence.get("granted") is not True or evidence.get("owner_pid") != os.getpid() or evidence.get("gpu_index") != ticket["gpu_index"] or not evidence.get("gpu_uuid") or evidence.get("gpu_observed_free") is not True or evidence.get("native_memory_gate_preserved") is not True or not 0 <= time.time() - evidence.get("observed_unix", 0) <= 30:
            raise PermissionError("Fresh actual GPU ownership/free/native memory evidence required")
        snapshot = (resource_reader or resource_snapshot)(local_root)
        if snapshot["cgroup_headroom"] < 32 * GIB or snapshot["local_free"] < budget + 8 * GIB:
            raise PermissionError("Need actual >=32 GiB cgroup headroom plus cache and 8 GiB local reserve")
        edition = Path(ticket["output_root"]) / ticket["edition"]
        if not edition.is_absolute() or edition.resolve() != edition:
            raise PermissionError("Output edition may not traverse symlinks")
        raw_ticket = json_bytes(ticket)
        claim_ticket = ticket if len(raw_ticket) <= 1024 ** 2 else {key: value for key, value in ticket.items() if key != "sources"}
        claim_once(edition / "claims" / "tickets" / (ticket["ticket_id"] + ".json"),
                   {"ticket": claim_ticket, "ticket_sha256": hashlib.sha256(raw_ticket).hexdigest(),
                    "source_count": len(sources), "sources_omitted": len(raw_ticket) > 1024 ** 2,
                    "owner_pid": os.getpid(), "resource_evidence": evidence})
        started = time.perf_counter()
        models = model_loader(ticket)  # Caller wrapper must run its actual gate(True) here.
        profile = {"models_load_once": time.perf_counter() - started,
                   **dict.fromkeys(("copy", "probe", "decode_pose", "rules", "beat", "write", "pack_publish"), 0.0)}
        if stage_gate:
            stage_gate(False)
        actual_probe = (lambda path: probe_video(path, ticket["ffprobe_path"])) if probe is probe_video and "ffprobe_path" in ticket else probe
        actual_decoder = (lambda path, width, height: decode_bgr(path, width, height, ticket["ffmpeg_path"])) if decoder is decode_bgr and "ffmpeg_path" in ticket else decoder
        summary = {"schema": SUMMARY_SCHEMA, "ticket_id": ticket["ticket_id"], "edition": ticket["edition"],
                   "status": "running", "source_count": len(sources), "processed": 0,
                   "selected_source_ids": [row["id"] for row in sources], "all_outcomes": [],
                   "counts": outcome_counts([])[0], "statuscounts": {}, "model_init_count": 1, "total_frames": 0,
                   "completed_batches": [], "allprefetch_drained": True, "uncommitted_prefetch": 0, "vlm_called": False,
                   "quality_approved": False, "beat_alignment_claimed": False, "production_complete_claimed": False,
                   "progress_path": str(edition / "progress.json"), "profile_seconds": profile}

        def update_progress():
            profile["whole_e2e"] = time.perf_counter() - started_e2e
            value = {key: summary[key] for key in ("ticket_id", "edition", "status", "source_count", "processed", "counts",
                     "statuscounts", "model_init_count", "total_frames", "allprefetch_drained", "uncommitted_prefetch", "vlm_called", "quality_approved",
                     "beat_alignment_claimed", "production_complete_claimed", "profile_seconds")}
            value.update(schema=PROGRESS_SCHEMA, completed_batch_count=len(summary["completed_batches"]),
                         latest_completed_batch=summary["completed_batches"][-1] if summary["completed_batches"] else None)
            publish_progress(summary["progress_path"], value)

        batch_sources = ticket.get("batch_sources", 20)
        for offset in range(0, len(sources), batch_sources):
            if stop_requested(ticket):
                summary["status"] = "drained_not_complete"
                break
            batch_index = len(summary["completed_batches"]) + 1
            commit, path, digest = run_batch(ticket, sources[offset:offset + batch_sources], batch_index, offset,
                                              edition, local_root, budget, models, actual_probe, actual_decoder, audio_info, stage_gate)
            counts, statuses = outcome_counts(commit["receipts"])
            summary["completed_batches"].append({"batch_index": batch_index, "commit_path": path, "commit_sha256": digest,
                                                  "source_count": len(commit["receipts"]), "statuscounts": statuses, "counts": counts})
            for result in commit["receipts"]:
                row = result["source_row"]
                summary["all_outcomes"].append({"id": row["id"], "split": row["split"], "status": result["status"],
                                                "draft_status": result.get("draft_status"), "input_copy_completed": result["input_copy_completed"],
                                                "frame_count": result.get("frame_count", 0), "batch_index": batch_index,
                                                "receipt_member": result["artifact_members"]["receipt.json"]})
                for name in ("copy", "probe", "decode_pose", "rules", "beat", "write"):
                    profile[name] += result.get("profile_seconds", {}).get(name, 0.0)
            profile["pack_publish"] += commit["profile_seconds"]["pack_publish"]
            summary["processed"] = len(summary["all_outcomes"])
            summary["counts"], summary["statuscounts"] = outcome_counts(summary["all_outcomes"])
            summary["total_frames"] += sum(result.get("frame_count", 0) for result in commit["receipts"])
            summary["status"] = "completed" if summary["processed"] == len(sources) else "running"
            update_progress()
            print(f"batch {batch_index} generated={counts['generated']} sources={len(commit['receipts'])} processed={summary['processed']}/{len(sources)}", flush=True)
            if stage_gate:
                stage_gate(False)
        summary["unprocessed_source_ids"] = summary["selected_source_ids"][summary["processed"]:]
        update_progress()
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--cpu-test", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_test:
        if args.ticket:
            parser.error("--cpu-test never accepts production tickets")
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_ms_unified_novlm_v3.py")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.ticket:
        parser.error("--ticket required")
    check_ticket(json.loads(Path(args.ticket).read_text()), args.limit)
    parser.error("Production disabled: native caller lease required; CLI ticket grants no GPU/claims")


if __name__ == "__main__":
    raise SystemExit(main())
