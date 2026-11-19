"""Fixed-three CPU-only whole-event selection; original transactions remain failed."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
OUT = RT.parent / "curated_coarse_fixed3_whole_event_salvage_v1"
REVISION = "coarse-fixed3-whole-event-salvage-20261026-v1"
SCHEMA = "curated_coarse_fixed3_whole_event_salvage_v1"
WORKER_SHA = "45f9b6917cc4c53c6cfb40e47456352fa8869b797fbdba6a7790cd59103ea3d5"
WORKER_TEST_SHA = "c104349c42e75ef250ec8deb5bfab2345b085ccdf337a2c9e75c2198ea074f00"
TRANSACTION_FILES = {"request.json", "binding.json", "response.raw.txt", "generation.json", "response.json", "final.json"}


def need(value, message):
    if not value:
        raise ValueError(message)


def worker_module():
    for name, digest in (("curated_coarse_reobserve_fixed3_v1.py", WORKER_SHA),
                         ("test_curated_coarse_reobserve_fixed3_v1.py", WORKER_TEST_SHA)):
        need(hashlib.sha256((RT / name).read_bytes()).hexdigest() == digest, "Frozen original worker/test changed")
    spec = importlib.util.spec_from_file_location("fixed3_whole_event_private_worker", RT / "curated_coarse_reobserve_fixed3_v1.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    need(worker.q.load(worker.OUT / "source.freeze.json") == worker.source_freeze(), "Original source freeze differs")
    return worker


def verify_unchanged(q, refs):
    for ref in refs:
        q.verify_bytes(ref)


def revision_snapshot(worker, job):
    root = worker.OUT / "revisions" / worker.REVISION / job["source_key"]
    refs = [worker.q.identity(p) for p in sorted(root.rglob("*")) if p.is_file()]
    return {"root": str(root), "files": refs}


def verify_snapshot(worker, snapshot):
    root = Path(snapshot["root"])
    need([str(p.resolve()) for p in sorted(root.rglob("*")) if p.is_file()]
         == [ref["path"] for ref in snapshot["files"]], "Original isolated revision inventory changed")
    verify_unchanged(worker.q, snapshot["files"])


def select_whole_events(worker, raw, manifest, forbidden):
    need(isinstance(raw, dict) and isinstance(raw.get("events"), list) and len(raw["events"]) <= 32, "invalid events")
    need(type(raw.get("abstain")) is bool and raw["abstain"] == (not raw["events"]),
         "no supported events require abstention; supported events cannot abstain")
    ids = [e.get("id") if isinstance(e, dict) else None for e in raw["events"]]
    need(all(isinstance(eid, str) and eid in {f"e{i}" for i in range(1, 33)} for eid in ids)
         and len(ids) == len(set(ids)), "invalid/duplicate event ID")
    empty = copy.deepcopy(raw)
    empty.update(events=[], abstain=True)
    worker.validate_coarse(empty, manifest, forbidden)
    accepted, records = [], []
    supports = {p["part"]: set(p["frame_indices"]) for p in raw["part_support"]}
    for index, event in enumerate(raw["events"]):
        one = copy.deepcopy(raw)
        one.update(events=[copy.deepcopy(event)], abstain=False)
        record = {"raw_event_index": index, "raw_event": copy.deepcopy(event)}
        try:
            filtered = worker.validate_coarse(one, manifest, forbidden)
            need(len(filtered["events"]) == 1, "Whole original event was excluded by native filter")
            admitted = filtered["events"][0]
            clause = admitted["source_clause"]
            need(admitted["description"] == event["description"] == clause["raw_source_clause"]
                 == clause["original_description"] and clause["raw_event_id"] == event["id"]
                 and clause["raw_clause_index"] == 0 and admitted["frame_indices"] == event["frame_indices"]
                 and set(admitted["parts"]) <= set(event["parts"]), "Whole verbatim event/ref/subset guard differs")
            accepted.append(copy.deepcopy(event))
            record.update(status="admitted", normalized_event=copy.deepcopy(admitted), original_error=None)
        except ValueError as error:
            if isinstance(error, worker.q.SourceChanged):
                raise
            record.update(status="excluded", original_error=str(error), error_type=type(error).__name__)
        if isinstance(event.get("parts"), list) and isinstance(event.get("frame_indices"), list):
            record["own_part_support_intersections"] = {part: sorted(set(event["frame_indices"]) & supports.get(part, set()))
                                                       for part in event["parts"]}
            record["own_part_forbidden_intersections"] = {part: sorted(set(event["frame_indices"]) & set(forbidden.get(part, [])))
                                                         for part in event["parts"]}
        records.append(record)
    projection = copy.deepcopy(raw)
    projection.update(events=accepted, abstain=not accepted)
    filtered = worker.validate_coarse(projection, manifest, forbidden)
    return projection, filtered, records


def original_transaction(worker, job_ref, job, original, prepared, parent):
    q, bar = worker.q, worker.bar
    root = worker.OUT / "revisions" / worker.REVISION / job["source_key"]
    attempt = root / "windows" / job["target_window_id"] / "attempt-000001"
    manifest = worker.verify_window(job, prepared)
    preview_ref = q.identity(worker.OUT / "request-previews" / job["source_key"] / job["target_window_id"] / "request.json")
    request = worker.make_request(job, job_ref, prepared, worker.worker_args())
    need(request == q.checked(preview_ref), "Original reconstructed request/preview changed")
    binding = {**original["binding"], "repair_revision": worker.REVISION, "current_job_ref": job_ref,
               "original_canonical_job_ref": job["refs"]["original_canonical_job"], "ledger_bindings": job["ledger_bindings"],
               "window": manifest["window"], "frames_sha256": bar.batch.digest(manifest["frames"]),
               "request_preview_ref": preview_ref}
    final = q.load(attempt / "final.json")
    item = {"root": attempt, "request": request, "binding": binding}
    model_sha = worker.protocol()["model_identity_sha256"]
    _, receipt_ref = worker.transaction_proof(item, final, model_sha)
    need(receipt_ref == parent["transaction_receipt_ref"], "Original parent receipt binding changed")
    receipt = q.checked(receipt_ref)
    actual = {name for name in TRANSACTION_FILES if (attempt / name).is_file()}
    need(actual == set(receipt["files_sha256"]) and actual >= TRANSACTION_FILES - {"response.json"},
         "Actual successful/failed transaction inventory differs")
    generation = bar.core.validate_generation(attempt, request)
    need(generation["model_identity_sha256"] == model_sha, "Actual native generation model digest differs")
    raw = bar.core.strict_json((attempt / "response.raw.txt").read_bytes())
    try:
        original_filtered = worker.validate_coarse(raw, manifest, prepared["forbidden"])
        full_error = None
    except ValueError as error:
        if isinstance(error, q.SourceChanged):
            raise
        original_filtered, full_error = None, str(error)
    if final["status"] == "failed":
        need(full_error is not None and final.get("error") == receipt.get("error") == "ValueError: " + full_error,
             "Original full-raw failure does not exactly replay")
    else:
        need(final["status"] == "candidate" and full_error is None and final["json"] == original_filtered,
             "Original successful full-raw validation differs")
    refs = [receipt_ref, preview_ref] + [q.identity(attempt / name) for name in sorted(actual)]
    return raw, manifest, attempt, refs, {"actual_inventory": sorted(actual), "original_receipt_ref": receipt_ref,
        "original_final_status": final["status"], "original_full_raw_validator_passed": full_error is None,
        "original_full_raw_validation_error": full_error, "original_generation": generation,
        "original_request_sha256": q.sha(attempt / "request.json"),
        "original_raw_sha256": q.sha(attempt / "response.raw.txt")}


def derive(worker, job_ref, probe=True):
    q = worker.q
    job = q.checked(job_ref)
    need(job["id"] in worker.TARGETS and Path(job_ref["path"]).resolve()
         == worker.OUT / "jobs" / (job["source_key"] + ".json"), "Fixed-three source job path differs")
    original, old_joint, prepared, plan = worker.verify_job(job, probe=probe)
    snapshot = revision_snapshot(worker, job)
    root = Path(snapshot["root"])
    parent_ref = q.identity(root / "result.json")
    parent = q.checked(parent_ref)
    terminal_ref = q.identity(worker.OUT / "worker-terminals" / (job["source_key"] + ".json"))
    terminal = q.checked(terminal_ref)
    budget_ref = worker.check_budget(job)
    need(budget_ref is not None, "Original canonical budget was not reserved")
    claim = q.checked(budget_ref)
    need(claim == q.load(root / "canonical_budget_claim.json") and claim["global_revision"] == worker.REVISION
         and parent["primary_budget_ref"] == budget_ref, "Original canonical spent budget changed")
    for value in (parent, terminal):
        need(value["current_job_ref"] == job_ref and value["original_job_ref"] == job["refs"]["original_canonical_job"]
             and value["global_revision"] == worker.REVISION and value["id"] == job["id"],
             "Original terminal/parent source provenance differs")
    need(terminal["joint_result_ref"] == parent_ref and not terminal["shared_dispatch_failure"]
         and parent["new_visual_requests"] == 1 and parent["new_text_model_requests"] == 0
         and parent["original_source_binding"] == job["source_binding"]
         and parent["style_result"] == old_joint["style_result"], "Original actual finite execution differs")
    q.checked(parent["visual_result_ref"])
    raw, manifest, attempt, refs, transaction = original_transaction(worker, job_ref, job, original, prepared, parent)
    projection, filtered, records = select_whole_events(worker, raw, manifest, prepared["forbidden"])
    observations = copy.deepcopy(original["window_observations"])
    target = next(i for i, o in enumerate(observations) if o["window"]["window_id"] == job["target_window_id"])
    observations[target] = {"status": "abstain" if filtered["abstain"] else "candidate", "json": filtered,
        "window": observations[target]["window"], "attempt_path": str(attempt),
        "derivation": "independent_whole_event_raw_projection_not_original_raw_success",
        "original_failed_transaction_preserved": transaction["original_final_status"] == "failed", **q.FLAGS}
    # Explicit raw_target is the selected projection, not a falsely approved full raw.
    visual, joint = worker.package(job, observations, original, old_joint, prepared, plan, raw_target=projection)
    bindings = [job_ref, terminal_ref, parent_ref, parent["visual_result_ref"], budget_ref,
                q.identity(worker.OUT / "protocol.json"), q.identity(worker.OUT / "source.freeze.json"),
                q.identity(worker.OUT / "jobs.manifest.json")] + refs
    proof = {"schema": SCHEMA, "global_revision": REVISION, "source_key": job["source_key"], "id": job["id"],
        "target_window_id": job["target_window_id"], "original_worker_revision": worker.REVISION,
        "original_job_ref": job["refs"]["original_canonical_job"], "original_fixed3_job_ref": job_ref,
        "original_fixed3_result_ref": parent_ref, "original_isolated_revision_snapshot": snapshot,
        "original_binding_refs": bindings, "original_transaction": transaction,
        "event_selection": records, "admitted_raw_event_ids": [e["id"] for e in projection["events"]],
        "excluded_raw_event_ids": [r["raw_event"]["id"] for r in records if r["status"] == "excluded"],
        "projection_is_derived_not_original_model_response": True,
        "selected_projection_native_digest": worker.bar.batch.digest(projection),
        "derivation": "independent_whole_event_raw_projection_not_original_raw_success",
        "original_visual_model_calls": 1, "new_visual_requests": 0, "new_text_model_requests": 0,
        "new_model_calls": 0, "extra_source_attempts": 0, "new_budget_reservations": 0,
        "primary_budget_before_ref": budget_ref, "primary_budget_after_ref": q.identity(worker.budget_path(job)),
        "canonical_budget_unchanged": True, "original_raw_or_terminal_modified": False, "style_label_changed": False,
        "independent_review_status": "pending_independent_vlm_review", "new_label_review_required": True,
        "training_approved": False, "repair_completed": False, "recovery_code_ref": q.identity(__file__), **q.FLAGS}
    need(proof["primary_budget_before_ref"] == proof["primary_budget_after_ref"], "Canonical budget bytes changed")
    verify_unchanged(q, bindings)
    verify_snapshot(worker, snapshot)
    worker.verify_job(job, probe=probe)
    visual.update(proof)
    joint.update(proof)
    joint["status"] = "whole_event_salvage_joint_candidate" if joint["joint_complete"] else "whole_event_salvage_partial_candidate"
    return job, projection, visual, joint, proof


def run():
    worker = worker_module()
    _, refs = worker.scope_manifest()
    rows = []
    for ref in refs:
        job, projection, visual, joint, proof = derive(worker, ref)
        root = OUT / "revisions" / REVISION / job["source_key"]
        worker.q.save_once(root / "selected_raw_projection.json", projection)
        proof["selected_raw_projection_ref"] = worker.q.identity(root / "selected_raw_projection.json")
        visual["selected_raw_projection_ref"] = proof["selected_raw_projection_ref"]
        joint["selected_raw_projection_ref"] = proof["selected_raw_projection_ref"]
        worker.q.save_once(root / "derivation.receipt.json", proof)
        worker.q.save_once(root / "visual_result.json", visual)
        joint["visual_result_ref"] = worker.q.identity(root / "visual_result.json")
        worker.q.save_once(root / "result.json", joint)
        rows.append({"id": job["id"], "source_key": job["source_key"], "target_window_id": job["target_window_id"],
                     "status": joint["status"], "joint_complete": joint["joint_complete"],
                     "admitted_raw_event_ids": proof["admitted_raw_event_ids"],
                     "excluded_raw_event_ids": proof["excluded_raw_event_ids"],
                     "result_ref": worker.q.identity(root / "result.json"),
                     "derivation_receipt_ref": worker.q.identity(root / "derivation.receipt.json")})
    summary = {"schema": SCHEMA, "global_revision": REVISION, "source_count": 3, "sources": rows,
               "original_model_calls": 3, "new_model_calls": 0, "new_budget_reservations": 0,
               "original_population_completed": False, "independent_review_status": "pending_independent_vlm_review",
               "training_approved": False, **worker.q.FLAGS}
    worker.q.save_once(OUT / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False), flush=True)
