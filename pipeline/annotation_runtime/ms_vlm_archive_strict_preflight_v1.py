"""Separate frozen small-JSON preflight; never open or rehash any archive/media."""
import argparse
import hashlib
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT))
import ms_vlm_archive_inputs_v1 as a
v = a.v
MEMBERS = {"draft.json", "fullframe_manifest.json", "pose.npz", "receipt.json", "beats.json"}


def strict_row(row):
    draft, manifest = v.input_pair(row)
    proof = v.checked(row["origin_proof_ref"])
    commit = v.frozen_commit(proof["commit_ref"])
    item = commit["archive_index"][row["id"]]
    members = item["members"]
    receipt = proof["receipt"]
    parent = Path(proof["commit_ref"]["path"]).parent
    prefix = "cases/" + row["split"] + "/" + row["id"]
    v.need(set(members) == MEMBERS and all(name == prefix + "/" + key for key, name in members.items()) and
           item["receipt_member"] == members["receipt.json"], "Required exact five-member index/receipt binding differs")
    v.need(receipt.get("schema") == "ms_unified_novlm_case_receipt_v2" and
           receipt.get("status") == a.SUCCESS and receipt.get("vlm_called") is False,
           "Required successful original no-VLM receipt schema differs")
    v.need(commit.get("archive_path") == str(parent / "batch.tar") and
           commit.get("archive_sha256") == commit.get("batch_sha256") and
           re.fullmatch(r"[a-f0-9]{64}", commit.get("archive_sha256", "")) and
           type(commit.get("archive_bytes")) is int and commit["archive_bytes"] > 0 and
           commit.get("vlm_called") is False, "Required original archive path/SHA/no-VLM differs")
    locator = dict(archive_path=commit["archive_path"], source_case_prefix=prefix,
                   fullframe_manifest_member=members["fullframe_manifest.json"], npz_member=members["pose.npz"])
    for value in (draft["source"], manifest, receipt):
        v.need(value.get("locator") == locator and value.get("artifact_storage") == "archive_member" and
               value.get("artifact_members") == members, "Required original three locators/storage/member maps differ")
    v.need(draft.get("locator") == locator and draft.get("artifact_storage") == "archive_member" and
           ("artifact_members" not in draft or draft["artifact_members"] == members), "Original top-level draft locator/storage differs")
    logical_npz = str(parent / members["pose.npz"])
    v.need(draft["source"].get("npz_path") == manifest.get("npz_path") == proof.get("original_npz_path") == logical_npz,
           "Required original logical NPZ binding differs")
    v.need(proof.get("original_binding") == draft["source"].get("fullframe_binding"), "Original complete binding differs")
    verification = v.checked(proof["archive_verification_ref"])
    v.need(all(verification.get(k) is True for k in ("no_npz_decoded", "no_media_input_read", "no_gpu_calls",
                                                  "hash_includes_npz_packaged_bytes")), "Original once-only CPU batch verification flags differ")
    refs = [row[k] for k in ("rule_draft_ref", "fullframe_manifest_ref", "origin_proof_ref")]
    refs.extend([proof["commit_ref"], proof["archive_verification_ref"]])
    return refs


def old500_snapshot(selection):
    plans = {ref["path"]: ref for ref in selection["historical_refs"] if ref["path"].endswith("_plan.jsonl")}
    v.need(len(plans) == 11, "Exact frozen eight-old plus three-user200 plan snapshot required")
    keys = []
    for reference in plans.values():
        path = Path(reference["path"])
        raw = path.read_bytes()
        v.need(path.is_absolute() and str(path.resolve(strict=True)) == str(path) and
               hashlib.sha256(raw).hexdigest() == reference["sha256"], "Frozen historical plan changed")
        keys.extend((r["split"], r["id"]) for r in (v.h.bench.strict(line) for line in raw.splitlines() if line.strip()))
    v.need(len(keys) == len(set(keys)) == len({id_.casefold() for _, id_ in keys}) == 500,
           "Exact old500 snapshot/casefold uniqueness differs")
    return {id_.casefold() for _, id_ in keys}, list(plans.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--plan-sha", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--selection-sha", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plan_ref = dict(path=args.plan, sha256=args.plan_sha)
    selection_ref = dict(path=args.selection, sha256=args.selection_sha)
    path = Path(args.plan)
    raw = path.read_bytes()
    v.need(path.is_absolute() and str(path.resolve(strict=True)) == str(path) and
           hashlib.sha256(raw).hexdigest() == args.plan_sha, "Exact stable frozen plan SHA required")
    rows = [v.h.bench.strict(line) for line in raw.splitlines() if line.strip()]
    selection = v.checked(selection_ref)
    v.need(len(rows) in (250, 3000) and selection.get("source_count") == 3000, "Finite first250/master3000 preflight only")
    keys = [[row["split"], row["id"]] for row in rows]
    ids = [row["id"] for row in rows]
    v.need(keys == selection["source_keys"][:len(rows)] and all(row["split"] == "train" for row in rows) and
           len(ids) == len(set(ids)) == len({id_.casefold() for id_ in ids}), "Exact selected original unique train sources differ")
    old, history_refs = old500_snapshot(selection)
    v.need(not old & {id_.casefold() for id_ in ids}, "Selected source overlaps frozen old500")
    refs = {r["path"]: r for row in rows for r in strict_row(row)}
    report = dict(schema="ms_archive_vlm_strict_preflight_report_v1", status="strict_small_json_preflight_passed",
        producer=v.ref(__file__), plan_ref=plan_ref, selection_ref=selection_ref, selected_source_ids=ids,
        source_keys_sha256=hashlib.sha256(a.encoded(keys)).hexdigest(), actual_checked_source_count=len(rows),
        required_fields_missing=0, source_mismatches=0, binding_failures=0, reference_mismatches=0,
        historical_snapshot_source_count=500, historical_overlap_exact=0, historical_overlap_casefold=0,
        frozen_historical_plan_refs=history_refs, all_checked_json_refs=list(refs.values()),
        whole_archive_hash_reads_by_preflight=0, archive_payload_read_by_preflight=False,
        no_gpu_calls=True, no_model_load=True, no_media_input_read=True, no_npz_decoded=True,
        original_raw_bytes_preserved=True, original_binding_digests_unchanged=True,
        no_run_configs_created=True, no_run_queues_created=True, **v.FLAGS)
    report_ref = a.json_once(args.out, report)
    print(v.json.dumps(dict(report_ref=report_ref, source_count=len(rows), plan_ref=plan_ref,
                            producer=report["producer"], whole_archive_hash_reads=0, no_gpu_calls=True), sort_keys=True))


if __name__ == "__main__":
    main()
