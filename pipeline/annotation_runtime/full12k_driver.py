"""Isolated full-pool ROI producer and terminal-gated resident visual consumer."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

TERMINAL = {"candidate", "abstain", "failed"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def load(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raw(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing different immutable output: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def verify_contract(path):
    contract = load(path)
    for name, expected in contract["files"].items():
        if sha(name) != expected:
            raise ValueError(f"Frozen input/code changed: {name}")
    return contract


def bound_terminal(row, plan_sha, status, **fields):
    return {"schema": "full12k_roi_terminal_v1", "id": row["id"],
            "plan_sha256": plan_sha, "video_sha256": row["binding"]["video_sha256"],
            "status": status, "semantic_approval": False, **fields}


def check_terminal(value, row, plan_sha):
    if (value.get("schema") != "full12k_roi_terminal_v1" or value.get("id") != row["id"]
            or value.get("plan_sha256") != plan_sha
            or value.get("video_sha256") != row["binding"]["video_sha256"]
            or value.get("status") not in {"ready", "failed"}):
        raise ValueError("ROI terminal source/full-plan binding mismatch")
    if value["status"] == "failed":
        if not value.get("reason") or value.get("roi_extraction_attempted") is not True:
            raise ValueError("ROI failure requires an explicit attempted-extraction reason")
    else:
        manifest = Path(value["roi_manifest"])
        proof = Path(value["cuda_execution_receipt"])
        required = {str(manifest), str(manifest.with_name("source_manifest.json")),
                    str(manifest.with_name("extract_receipt.json")), str(proof)}
        if not manifest.is_absolute() or not required <= set(value["files"]):
            raise ValueError("Ready ROI terminal lacks manifest/source/extraction/CUDA SHA receipts")
        if load(proof).get("real_CUDA_nodes_executed") is not True:
            raise ValueError("Ready ROI terminal has no executed CUDA proof")
        for name, expected in value["files"].items():
            if sha(name) != expected:
                raise ValueError(f"ROI terminal immutable artifact changed: {name}")
    return value


def completed_roi(roi, directory, row, model_ref, params):
    source = load(directory / "source_manifest.json")
    roi.verify_identity(source["source"])
    if (source["id"] != row["id"] or source["video_sha256"] != row["binding"]["video_sha256"]
            or source["source"]["path"] != str(Path(row["manifest_record"]["video_path"]).resolve())
            or source["params"] != params):
        raise ValueError("ROI reuse source/video/params mismatch")
    for key, identity in model_ref["files"].items():
        if source["model_ref"]["files"][key]["sha256"] != identity["sha256"]:
            raise ValueError("ROI reuse model/code bytes mismatch")
    roi.verify_model(source["model_ref"])
    manifest = roi.validate_completed(directory, source)
    files = {str(directory / name): sha(directory / name) for name in
             ("source_manifest.json", "prepare_receipt.json", "roi_manifest.json", "extract_receipt.json")}
    for asset in [manifest["raw"]] + manifest["crops"]:
        files[asset["path"]] = asset["sha256"]
    return manifest, files


def finish_profile(model, out):
    models = {}
    for name, session in (("detector", model.session_det), ("pose", model.session_pose)):
        path = Path(session.end_profiling())
        counts = Counter(e.get("args", {}).get("provider") for e in load(path))
        counts.pop(None, None)
        if not counts.get("CUDAExecutionProvider"):
            raise RuntimeError(f"{name} executed no CUDA nodes; CPU-only success forbidden")
        models[name] = {"executed_node_counts": dict(counts), "active_providers": session.get_providers(),
                        "profile_path": str(path), "profile_sha256": sha(path)}
    value = {"real_CUDA_nodes_executed": True, "CPU_shape_control_nodes_allowed": True,
             "profiling_ended_before_remainder_of_full_pool": True, "models": models,
             "semantic_approval": False}
    immutable(out / "cuda_execution_receipt.json", value)
    return out / "cuda_execution_receipt.json"


def roi_main(options, extra, contract):
    sys.path.insert(0, contract["roi_source_dir"])
    roi = importlib.import_module("ms_hand_roi")
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=("run",), default="run")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    p.add_argument("--ffmpeg", required=True)
    p.add_argument("--ffprobe", required=True)
    a = p.parse_args(extra)
    if a.limit != 0 or a.shard != 0 or a.nshard != 1 or str(Path(a.plan).resolve()) != contract["plan"]:
        raise ValueError("Full ROI producer must select exact authorized 12000 pool")
    rows, manifest_ref = roi.read_manifest(a.plan, limit=0)
    plan_rows = {r["id"]: r for r in (json.loads(line) for line in Path(a.plan).read_text().splitlines() if line.strip())}
    if len(rows) != 12000:
        raise ValueError("Expected exactly 12000 sources")
    out = Path(a.out).resolve()
    if str(out) != contract["roi_out"]:
        raise ValueError("ROI output differs from isolated contract root")
    out.mkdir(parents=True, exist_ok=True)
    params = contract["roi_params"]
    roi.validate_params(params)
    model_ref = load(contract["roi_modelref"])
    roi.verify_model(model_ref)
    model, profile_receipt, profile_attempts = None, None, 0
    results = []
    run_dir = out / "runtime_profile" / str(os.getpid())
    for item in rows:
        row = plan_rows[item["id"]]
        terminal_path = out / "terminal" / (row["id"] + ".json")
        with roi_lock(out, row["id"]):
            if terminal_path.exists():
                value = check_terminal(load(terminal_path), row, manifest_ref["sha256"])
            else:
                loading_model = False
                checking_profile = False
                try:
                    reuse = contract["reuse"].get(row["id"])
                    if reuse:
                        for name, expected in reuse["files"].items():
                            if sha(name) != expected:
                                raise ValueError("Frozen three-source reuse bytes changed")
                        directory = Path(reuse["roi_manifest"]).parent
                        manifest, files = completed_roi(roi, directory, row, model_ref, params)
                        proof = Path(reuse["cuda_receipt"])
                        if load(proof).get("real_CUDA_nodes_executed") is not True:
                            raise ValueError("Reused ROI lacks actual CUDA execution evidence")
                        files[str(proof)] = sha(proof)
                        value = bound_terminal(row, manifest_ref["sha256"], "ready", files=files,
                            roi_manifest=str(directory / "roi_manifest.json"), cuda_execution_receipt=str(proof),
                            original_binding_preserved=True,
                            reuse_existing_exact_SHA=True, crop_count=len(manifest["crops"]))
                    else:
                        source = roi.prepare_source(item, manifest_ref, model_ref, params, a.ffprobe, a.ffmpeg)
                        if source["video_sha256"] != row["binding"]["video_sha256"]:
                            raise ValueError("Actual video SHA differs from full plan")
                        directory = roi.ensure_prepared(out, source)
                        if not (directory / "extract_receipt.json").exists():
                            if model is None:
                                loading_model = True
                                model = roi.load_cuda_model(model_ref, params, run_dir)
                                loading_model = False
                            profile_attempts += 1
                            manifest = roi.extract_source(directory, source, model)
                        else:
                            manifest = roi.validate_completed(directory, source)
                        if profile_receipt is None:
                            if model is not None:
                                checking_profile = True
                                profile_receipt = finish_profile(model, run_dir)
                                checking_profile = False
                            else:
                                proofs = sorted((out / "runtime_profile").glob("*/cuda_execution_receipt.json"))
                                if not proofs:
                                    raise ValueError("Completed orphan ROI lacks bounded CUDA receipt")
                                profile_receipt = proofs[-1]
                        manifest, files = completed_roi(roi, directory, row, model_ref, params)
                        files[str(profile_receipt)] = sha(profile_receipt)
                        value = bound_terminal(row, manifest_ref["sha256"], "ready", files=files,
                            roi_manifest=str(directory / "roi_manifest.json"), cuda_execution_receipt=str(profile_receipt),
                            reuse_existing_exact_SHA=False,
                            crop_count=len(manifest["crops"]))
                except Exception as error:
                    if loading_model or checking_profile:
                        raise
                    # Per-source failures are terminal and auditable; no fake points/crops.
                    value = bound_terminal(row, manifest_ref["sha256"], "failed",
                        roi_extraction_attempted=True, reason=f"{type(error).__name__}: {error}")
                if model is not None and profile_receipt is None and profile_attempts >= 3:
                    profile_receipt = finish_profile(model, run_dir)
                immutable(terminal_path, value)
        results.append(value)
        summary = {"schema": "full12k_roi_progress_v1", "stage": "run", "plan_sha256": manifest_ref["sha256"],
            "total_source_count": len(rows), "selected_source_count": len(rows),
            "processed_source_count": len(results), "status_counts": dict(Counter(r["status"] for r in results)),
            "last_source_id": row["id"], "one_resident_ORT_model": True, "semantic_approval": False}
        atomic(out / "progress.json", raw(summary))
        print(json.dumps({"id": row["id"], "roi_status": value["status"], "processed": len(results)}), flush=True)
    if model is not None and profile_receipt is None:
        finish_profile(model, run_dir)
    atomic(out / "summary.json", raw({"summary": summary, "selection": {"selected_ids": [r["id"] for r in results]}}))
    return 0


class roi_lock:
    def __init__(self, root, sid):
        self.path = Path(root) / "locks" / (sid + ".lock")
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self
    def __exit__(self, *exc):
        os.close(self.fd)


def eligible(result):
    if result.get("status") != "candidate" or result.get("final", {}).get("status") != "candidate":
        return False
    windows = result.get("original_windows", [])
    captions = result.get("final", {}).get("json", {}).get("windows", [])
    observations = result.get("window_observations", [])
    return bool(windows) and len(captions) == len(windows) and len(observations) == len(windows) and (
        [w["window_id"] for w in captions] == [o["window"]["window_id"] for o in observations]) and all(
        isinstance(w.get("caption"), str) and w["caption"].strip() for w in captions) and all(
        o.get("status") in {"candidate", "abstain"} for o in observations) and bool(result.get("final_timed_caption"))


def append_export(path, result):
    with Path(path).open("ab") as stream:
        stream.write(raw(result))
        stream.flush()
        os.fsync(stream.fileno())


def visual_main(options, extra, contract):
    sys.path.insert(0, contract["visual_source_dir"])
    bar = importlib.import_module("ms_bar_annotation")
    a = bar.parser().parse_args(extra)
    if (str(Path(a.plan).resolve()) != contract["plan"] or a.stage != "run" or a.limit != 0
            or a.nshard != 4 or not 0 <= a.shard < 4 or a.tp != 2):
        raise ValueError("Visual consumer must select an explicit full-plan 3000-source shard")
    if str(Path(a.hand_roi_root).resolve()) != contract["roi_out"]:
        raise ValueError("Hand ROI gate root differs from frozen contract")
    rows, plan_sha, total = bar.read_plan(a.plan, a.shard, a.nshard, a.limit)
    if total != 12000 or len(rows) != 3000:
        raise ValueError("Expected disjoint 4x3000 sharding")
    out = Path(a.out).resolve()
    expected_out = Path(contract["visual_out"]) / ("shard" + str(a.shard))
    if out != expected_out:
        raise ValueError("Visual output differs from isolated contract root")
    out.mkdir(parents=True, exist_ok=True)
    code = bar.code_identity()
    configuration = {**bar.params(a, code), "full12k_driver_sha256": sha(__file__),
                     "roi_gate_contract_sha256": sha(options.contract)}
    bar.core.save_json(out / "protocol.json", {"schema": bar.OUTPUT_SCHEMA,
        "plan_path": str(Path(a.plan).resolve()), "plan_sha256": plan_sha,
        "params": configuration, "source_count": total, **bar.FLAGS})
    runtime = bar.runtime_loader(a, out, code)
    original_roi_loader = bar.load_hand_roi
    active_gate = {}
    def terminal_roi_loader(row, probe, ignored):
        gate = active_gate[row["id"]]
        check_terminal(gate, row, plan_sha)
        if gate["status"] == "ready":
            manifest, metadata = original_roi_loader(row, probe, gate["roi_manifest"])
            if not metadata.get("available"):
                raise ValueError("Ready terminal did not yield complete native ROI")
            return manifest, {**metadata, "full_pool_terminal_gated": True}
        return None, {"available": False, "reason": "ROI_extraction_failed: " + gate["reason"],
                      "roi_extraction_attempted": True, "base_RGB_not_blocked": True,
                      "full_pool_terminal_gated": True, "semantic_approval": False}
    bar.load_hand_roi = terminal_roi_loader
    with roi_lock(out, "consumer"):
        # Rebuild progressive views from immutable completed records on resume.
        terminal_results = {}
        for path in (out / "completed").glob("*.json"):
            result = load(path)
            if result["binding"]["plan_sha256"] != plan_sha or result["status"] not in TERMINAL:
                raise ValueError("Foreign/incomplete completed visual record")
            terminal_results[result["id"]] = result
        selected_ids = [row["id"] for row, _ in rows]
        if set(terminal_results) - set(selected_ids):
            raise ValueError("Completed export contains sources outside this shard")
        for name, values in (("completed_labels.jsonl", terminal_results.values()),
                             ("trainable_weak_labels.jsonl", [r for r in terminal_results.values() if eligible(r)])):
            atomic(out / name, b"".join(raw(r) for r in values))
        pending = [(row, line) for row, line in rows if row["id"] not in terminal_results]
        deadline = time.monotonic() + options.roi_wait_seconds
        refs = {}
        while pending:
            ready = next(((i, load(Path(a.hand_roi_root) / "terminal" / (row["id"] + ".json")))
                          for i, (row, _) in enumerate(pending)
                          if (Path(a.hand_roi_root) / "terminal" / (row["id"] + ".json")).exists()), None)
            if ready is None and time.monotonic() < deadline:
                atomic(out / "roi_wait.json", raw({"pending_count": len(pending),
                    "reason": "No terminal ROI receipt in selected shard", "no_RGB_prepare_without_ROI_terminal": True}))
                time.sleep(options.roi_poll_seconds)
                continue
            if ready is None:
                atomic(out / "blocked.json", raw({"reason": "Bounded ROI terminal wait expired",
                    "pending_ids": [row["id"] for row, _ in pending],
                    "pending_sources_not_marked_failed": True, "no_RGB_prepare_without_ROI_terminal": True}))
                raise RuntimeError("Shared ROI producer wait expired; pending source records preserved")
            index, gate = ready
            row, line = pending.pop(index)
            binding = {"source_binding": row["binding"], "plan_sha256": plan_sha,
                       "full_source_row_sha256": bar.batch.digest(row), "raw_line_sha256": bar.sha(line),
                       "manifest_record_sha256": bar.batch.digest(row["manifest_record"]),
                       "params_sha256": bar.batch.digest(configuration)}
            root = out / "sources" / row["id"]
            try:
                if gate is None:
                    raise TimeoutError("Bounded ROI terminal wait expired; RGB was not prepared without extraction evidence")
                check_terminal(gate, row, plan_sha)
                active_gate[row["id"]] = gate
                with bar.batch.bundle_lock(root):
                    bar.core.save_json(root / "roi_gate.json", gate)
                    result = bar.process_source(row, line, root, a, binding, runtime, refs)
                if result["status"] not in TERMINAL:
                    raise ValueError("run returned a nonterminal visual record")
            except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
                result = {"schema": bar.OUTPUT_SCHEMA, "source": bar.OUTPUT_SCHEMA, "id": row["id"],
                    "status": "failed", "binding": binding, "manifest_record": row["manifest_record"],
                    "original_windows": row["windows"], "final_timed_caption": "",
                    "error": f"{type(error).__name__}: {error}", "roi_gate": gate, **bar.FLAGS}
                failure = root / "failures" / (bar.batch.digest(result) + ".json")
                bar.core.save_json(failure, result)
                result["result_ref"] = {"path": str(failure), "sha256": sha(failure)}
            result = {**result, "human_approved": False, "semantic_approval": False,
                      "full12k_terminal_verified": True, "roi_terminal_received": gate is not None,
                      "trainable_weak_candidate": eligible(result)}
            immutable(out / "completed" / (row["id"] + ".json"), result)
            append_export(out / "completed_labels.jsonl", result)
            if result["trainable_weak_candidate"]:
                append_export(out / "trainable_weak_labels.jsonl", result)
            terminal_results[row["id"]] = result
            results = list(terminal_results.values())
            summary = bar.summarize(results, "run", total, len(rows), plan_sha)
            summary.update(status_counts=dict(Counter(r["status"] for r in results)),
                trainable_weak_count=sum(eligible(r) for r in results),
                empty_window_count=sum(len(r.get("original_windows", [])) -
                    sum(bool(w.get("caption", "").strip()) for w in r.get("final", {}).get("json", {}).get("windows", []))
                    for r in results),
                partial_caption_count=sum(r["status"] == "candidate" and not eligible(r) for r in results),
                human_approved=False, semantic_approval=False)
            atomic(out / "progress.json", raw(summary))
            atomic(out / "run.summary.json", raw({"summary": summary, "selection": {"shard": a.shard,
                "nshard": a.nshard, "selected_ids": selected_ids}, "progressive_completed_only_export": True}))
            print(json.dumps({"id": row["id"], "status": result["status"],
                "processed": len(results), "trainable_weak_count": summary["trainable_weak_count"]}), flush=True)
            deadline = time.monotonic() + options.roi_wait_seconds
        if not pending:
            atomic(out / "completed.receipt.json", raw({"all_selected_sources_terminal": True,
                "selected_source_count": len(rows), "plan_sha256": plan_sha,
                "completed_labels_sha256": sha(out / "completed_labels.jsonl"),
                "trainable_weak_labels_sha256": sha(out / "trainable_weak_labels.jsonl"),
                "one_resident_visual_model_per_shard": True, "human_approved": False}))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("mode", choices=("roi", "visual"))
    p.add_argument("--contract", required=True)
    p.add_argument("--roi-wait-seconds", type=float, default=1800)
    p.add_argument("--roi-poll-seconds", type=float, default=5)
    options, extra = p.parse_known_args(argv)
    if options.roi_wait_seconds <= 0 or options.roi_poll_seconds <= 0:
        raise ValueError("Positive bounded ROI waiting budgets required")
    contract = verify_contract(options.contract)
    return roi_main(options, extra, contract) if options.mode == "roi" else visual_main(options, extra, contract)


if __name__ == "__main__":
    raise SystemExit(main())
