"""Bounded independent recovery: plan, run once per failed window, then export.

The original eight workers and verified export are immutable. This module does
not extend the free-decoding producer's resume protocol. Durable exclusive call
claims prevent reissuing an uncertain call after a crash. All outputs remain
unreviewed; true abstentions and source errors leave caption-delivery gaps.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import hashlib
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

import ms_visual_batch as batch
import ms_visual_export as original

SCHEMA = "motionscript_visual_bounded_recovery_v1"
PRODUCER_SHA = "b362dc5275b19c70a6f4f4d149b974e75571ad07e7e4f1f0a2251222fd839204"
EXPORTER_SHA = "4a562a5ff7d5bfcc4b898b60f2e4883e628b3f2f8482e43431286de7f558a558"
MODEL_SHA = "26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c"
CLARIFICATION = """Re-observe only the supplied RGB sequence and describe supported movement
phases or transitions chronologically. Aim for at most 90 English words in
motion_caption; the original hard limit remains 140 words. A movement or
transition needs two distinct actually supplied source frames that genuinely
support it. Adding another index does not support an isolated single-frame
pose. Omit unsupported instantaneous details from caption and events; put
unresolved ambiguity in uncertainties. A held phase is allowed only within a
supported real movement sequence. If no movement is supported, return exactly
abstain=true, motion_caption="", events=[]; do not replace an abstention with a
static pose caption or invented movement. Event IDs must be unique nonempty
strings. Do not guess anatomical sides or add scene, identity, clothing,
props, music, or unobserved contacts. Return the system's four JSON keys only.
"""


def need(condition,message):
    if not condition:
        raise ValueError(message)


def load(path):
    return batch.strict_json(Path(path).read_text(encoding="utf-8"))


def target_digest(target):
    # Targets are stored as canonical JSONL; their identity must not depend on
    # dictionary insertion order before or after a serialization round trip.
    value={k:v for k,v in target.items() if k!="target_sha256"}
    return hashlib.sha256(original.canonical_json(value)).hexdigest()


def payload(rows):
    return b"".join(original.canonical_json(row)+b"\n" for row in rows)


def reference(path):
    path=Path(path).resolve(strict=True)
    return {"path":str(path),"sha256":batch.sha256(path)}


def read_reference(ref):
    path=Path(ref["path"])
    need(path.is_absolute(),"reference path must be absolute")
    original.sha_file(path,ref["sha256"])
    return path


def exclusive(path,data):
    path=Path(path)
    with path.open("xb") as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    descriptor=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def safe_out(out,roots):
    out=Path(out).resolve()
    for root in roots:
        need(not batch.overlaps(out,Path(root)),"independent recovery output overlaps immutable source/model/another output")
    return out


def verify_original(export_root,launch_path):
    launch,_=original.read_plan(launch_path)
    original.terminal_snapshot(launch)
    with ExitStack() as locks:
        for job in sorted(launch["jobs"],key=lambda j:j["shard"]):
            path=original.control_path(job,8)/".lock"
            need(path.is_file(),"original producer lock missing")
            stream=locks.enter_context(path.open("rb"))
            try:fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError as error:raise original.Pending("original worker still active: "+str(path)) from error
        return _verify_original(export_root,launch_path)


def _verify_original(export_root,launch_path):
    launch,inputs=original.read_plan(launch_path)
    original.terminal_snapshot(launch)  # No active frame/media traversal before this gate.
    root=Path(export_root).resolve()
    manifest_path=root/"manifest.json"
    if not manifest_path.is_file():
        raise original.Pending("original 4a verified export manifest is absent")
    manifest=load(manifest_path)
    need(manifest["schema"]==original.SCHEMA and manifest["status"]=="terminal_verified_unreviewed", "original export is not a verified terminal ledger")
    original.false_approval(manifest)
    need(manifest["exporter_sha256"]==EXPORTER_SHA==batch.sha256(Path(original.__file__)),"original exporter is not frozen 4a")
    need(manifest["producer_sha256"]==PRODUCER_SHA==batch.sha256(Path(batch.__file__)),"original producer identity changed")
    need(manifest["launch_plan_sha256"]==batch.sha256(Path(launch_path)) and manifest["input_manifest_sha256"]==launch["input_manifest_sha256"],"original launch/input binding differs")
    need(manifest["all_planned_workers_terminal"] is True and manifest["all_expected_ids_accounted_for"] is True
        and type(manifest["verified_ids"]) is int and manifest["verified_ids"]==12000,"original complete accounting was not actually verified")
    captions_path=root/"captions.jsonl"
    original.sha_file(captions_path,manifest["captions_sha256"])
    rows=[batch.strict_json(line) for line in captions_path.read_text().splitlines() if line.strip()]
    need([r["id"] for r in rows]==[r["id"] for r in inputs] and len(rows)==12000,"original verified export has wrong/duplicate/missing source IDs")
    availability=original.caption_availability(rows)
    need(all(original.same_json(manifest[k],v) for k,v in availability.items()),"original export caption counts/types differ from its records")
    for row,input_row in zip(rows,inputs):
        original.false_approval(row)
        need(row["status"] in original.TERMINAL and row["source_group"]==input_row["source_group"] and row["split"]==input_row["split"]
            and original.same_json(row["original_input"],input_row),"original source context mismatch")
        if row["windows"]:
            need(len(row["windows"])==2 and all(w["status"] in original.TERMINAL for w in row["windows"]),"original export includes active windows")
        else:need(row["status"]=="failed" and row.get("expected_windows_have_no_terminal_receipt") is True,"missing original windows not explicitly source-error blocked")
    need(len(manifest["lineage"])==8 and sorted(line["shard"] for line in manifest["lineage"])==list(range(8)),"original export lacks eight unique worker proofs")
    reference_path=Path(manifest["model_reference_receipt_path"])
    original.sha_file(reference_path,manifest["model_reference_receipt_sha256"])
    ref=load(reference_path)["model_identity"]
    need(ref["sha256"]==MODEL_SHA,"original reference model differs")
    current_code=batch.code_identity()
    for line in manifest["lineage"]:
        job=launch["jobs"][line["shard"]]
        need(job["out"]==line["out"] and job["resource_queue"]==line["resource_queue"],"original worker lineage is not the planned job")
        control=original.control_path(job,8)
        for name,key in (("summary.json","summary_sha256"),("index.jsonl","index_sha256"),("run.receipt.json","run_receipt_sha256")):
            original.sha_file(control/name,line[key])
        original_run=load(control/"run.receipt.json")
        need(original_run["code_identity"]["sha256"]==current_code["sha256"] and original.content_inventory(original_run["code_identity"])==original.content_inventory(current_code),"original code dependency identity changed")
        original.verify_model(original_run["model_identity"],ref,original.argv_values(job)["--model"])
        original.sha_file(Path(job["resource_queue"])/"worker_exit.json",line["worker_exit_sha256"])
        need(original.same_json(original.verify_terminal(job,launch),line["resource_worker_exit"]),"original worker proof changed after verified export")
    return manifest,rows,launch


def targets_from_rows(rows,run_output_path):
    need(Path(run_output_path).is_absolute(),"explicit absolute unique recovery run output required")
    run_output_path=str(Path(run_output_path).resolve())
    targets=[];blocked=[];seen=set()
    for row in rows:
        if not row["windows"]:
            blocked.append({"id":row["id"],"status":"blocked_source_error_without_original_window_evidence",
                "source_identity":row["source_identity"],"original_failure_receipt_path":row.get("failure_receipt_path"),
                "original_failure_receipt_sha256":row.get("failure_receipt_sha256"),"expected_windows":row.get("expected_windows",[]),
                "no_window_or_model_response_fabricated":True})
        for index,window in enumerate(row["windows"]):
            if window["status"]!="failed":continue
            key=f"{row['id']}:w{index}"
            need(key not in seen,"duplicate failed target key");seen.add(key)
            required=("manifest.json","request.json","response.raw.txt","generation.json")
            bundle=Path(window["bundle"]).resolve()
            if not all((bundle/name).is_file() for name in required):
                blocked.append({"id":row["id"],"window_index":index,"window":window["window"],"status":"blocked_missing_original_frame_or_generation_evidence","original_bundle":str(bundle),"original_receipt_sha256":window["receipt_sha256"],"no_window_or_model_response_fabricated":True})
                continue
            target={"schema":SCHEMA,"key":key,"id":row["id"],"window_index":index,"window":window["window"],
                "source_group":row["source_group"],"split":row["split"],"original_input":row["original_input"],
                "original_source_identity":row["source_identity"],"original_source_sha256":row["source_identity"]["sha256"],
                "original_model_identity_sha256":row["model_instance_identity_sha256"],"original_params_sha256":row["params_sha256"],
                "original_failure_reason":window["error"],"original_bundle":str(bundle),"original_frames":window["source_frames"],
                "original_refs":{name:reference(bundle/name) for name in (*required,"manifest.receipt.json","receipt.json")},
                "original_final":reference(row["final_path"]),"max_new_observations":1,"run_output_path":run_output_path}
            target["target_sha256"]=target_digest(target)
            targets.append(target)
    return targets,blocked


def plan_stage(args):
    manifest,rows,launch=verify_original(args.original_export,args.original_plan)
    out=safe_out(args.out,[args.original_export,Path(args.original_plan).parent,*[j["out"] for j in launch["jobs"]]])
    need(not out.exists(),"new exclusive recovery plan directory required")
    run_output=safe_out(args.run_out,[out,args.original_export,Path(args.original_plan).parent,*[j["out"] for j in launch["jobs"]]])
    need(Path(args.run_out).is_absolute(),"explicit absolute unique run output path required")
    targets,blocked=targets_from_rows(rows,str(run_output))
    out.mkdir(parents=True,exist_ok=False)
    raw=payload(targets)
    exclusive(out/"targets.jsonl",raw)
    plan={"schema":SCHEMA,"stage":"plan","planning_complete":True,"original_export":str(Path(args.original_export).resolve()),
        "original_manifest_ref":reference(Path(args.original_export)/"manifest.json"),
        "original_captions_ref":reference(Path(args.original_export)/"captions.jsonl"),"original_launch_ref":reference(args.original_plan),
        "original_input_manifest_sha256":launch["input_manifest_sha256"],"original_producer_sha256":PRODUCER_SHA,"original_exporter_sha256":EXPORTER_SHA,
        "model_reference_ref":reference(manifest["model_reference_receipt_path"]),"expected_model_identity_sha256":MODEL_SHA,
        "targets_ref":{"path":str(out/"targets.jsonl"),"sha256":hashlib.sha256(raw).hexdigest()},
        "selected_failed_windows":len(targets),"blocked_scope":blocked,"blocked_scope_count":len(blocked),
        "preserved_good_candidate_windows":sum(w["status"]=="candidate" for r in rows for w in r["windows"]),
        "preserved_valid_abstain_windows":sum(w["status"]=="abstain" for r in rows for w in r["windows"]),
        "max_new_observations_per_target":1,"one_pass_only":True,"no_good_or_valid_abstain_regeneration":True,
        "new_protocol_not_original_resume":True,"same_original_RGB_only":True,"run_output_path":str(run_output),
        "planner_source_sha256":batch.sha256(Path(__file__)),"semantic_approval":False,"human_review_completed":False,"formal_training":False,"full_pool_updated":False}
    exclusive(out/"plan.json",batch.json_bytes(plan))
    return {"status":"recovery_plan_created","out":str(out),"targets":len(targets),"blocked_scope_count":len(blocked),"GPU_run_performed":False,"caption_delivery_complete":False}


def load_plan(path):
    path=Path(path).resolve(strict=True);plan=load(path)
    need(plan["schema"]==SCHEMA and plan["stage"]=="plan" and plan["planning_complete"] is True,"not a completed recovery selection")
    original.false_approval(plan)
    need(plan["original_producer_sha256"]==PRODUCER_SHA and plan["original_exporter_sha256"]==EXPORTER_SHA and plan["expected_model_identity_sha256"]==MODEL_SHA,"recovery plan original producer/exporter/model identity changed")
    need(Path(plan["run_output_path"]).is_absolute() and str(Path(plan["run_output_path"]).resolve())==plan["run_output_path"],"unique planned run output path invalid")
    need(plan["planner_source_sha256"]==batch.sha256(Path(__file__)),"recovery planner/runner source changed after immutable plan")
    need(type(plan["max_new_observations_per_target"]) is int and plan["max_new_observations_per_target"]==1
        and plan["one_pass_only"] is True and plan["no_good_or_valid_abstain_regeneration"] is True,"unbounded recovery budget")
    need(Path(plan["original_manifest_ref"]["path"]).resolve()==(Path(plan["original_export"])/"manifest.json").resolve() and Path(plan["original_captions_ref"]["path"]).resolve()==(Path(plan["original_export"])/"captions.jsonl").resolve(),"original reference paths differ from selected verified export")
    for key in ("original_manifest_ref","original_captions_ref","original_launch_ref","model_reference_ref"):read_reference(plan[key])
    target_path=read_reference(plan["targets_ref"])
    need(target_path.is_relative_to(path.parent),"target list outside recovery plan")
    targets=[batch.strict_json(line) for line in target_path.read_text().splitlines() if line.strip()]
    need(type(plan["selected_failed_windows"]) is int and len(targets)==plan["selected_failed_windows"] and len({t["key"] for t in targets})==len(targets),"recovery targets duplicated/missing")
    for target in targets:
        need(target_digest(target)==target["target_sha256"],"target content SHA mismatch")
        need(target["run_output_path"]==plan["run_output_path"],"target points to another recovery run tree")
        need(type(target["max_new_observations"]) is int and target["max_new_observations"]==1 and type(target["window_index"]) is int and target["window_index"] in (0,1),"target budget/index invalid")
    return plan,targets


def target_dir(out,target):
    need(str(Path(out).resolve())==target["run_output_path"],"target can only use the immutable plan run output")
    need(isinstance(target["id"],str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",target["id"]),"unsafe target SID")
    return original.beneath(Path(out)/"attempt1"/target["id"]/f"w{target['window_index']:04d}",Path(out))


def inspect_target(target,allowed_roots):
    old=Path(target["original_bundle"]).resolve()
    need(any(old.is_relative_to(Path(root).resolve()) for root in allowed_roots),"original target outside explicitly allowed producer output")
    for ref in target["original_refs"].values():
        path=read_reference(ref);need(path.resolve().is_relative_to(old),"target reference escapes original bundle")
    final_path=read_reference(target["original_final"]).resolve()
    need(any(final_path.is_relative_to(Path(root).resolve()) for root in allowed_roots),"original final outside allowed producer roots")
    manifest=load(old/"manifest.json")
    need(original.same_json(manifest["input"],target["original_input"]) and original.same_json(manifest["window"],target["window"]),"original target source/window differs")
    source=batch.source_identity(manifest["input"])
    need(source["sha256"]==target["original_source_sha256"],"original source content changed")
    binding=manifest["binding"]
    need(binding["source_sha256"]==source["sha256"] and binding["model_sha256"]==target["original_model_identity_sha256"]
        and binding["params_sha256"]==target["original_params_sha256"],"original target binding changed")
    batch.verify_evidence(old,binding,source)
    receipt=batch.resume_receipt(old,binding,source)
    need(receipt and receipt["status"]=="failed" and receipt["error"]==target["original_failure_reason"],"do not recover a successful or changed original window")
    generation=load(old/"generation.json")
    need(generation.get("model_call_performed") is True and generation["raw_response_sha256"]==batch.sha256(old/"response.raw.txt"),"original generation/raw SHA missing")
    need(original.same_json(manifest["frames"],target["original_frames"]),"original supplied RGB frame inventory differs")
    request=load(old/"request.json")
    original.verify_request(old,manifest,{"values":{"model_path":request["model"]}})
    need(request["messages"][0]["content"]==batch.SYSTEM,"system movement/abstention contract changed")
    return manifest,source,request


def shape_schema(indices):
    need(isinstance(indices,list) and 2<=len(indices)<=16 and all(type(i) is int for i in indices) and indices==sorted(set(indices)),"invalid actual source indices")
    event={"type":"object","properties":{"id":{"type":"string","minLength":1},"description":{"type":"string","minLength":1},
        "frame_indices":{"type":"array","minItems":2,"items":{"type":"integer","enum":indices}}},
        "required":["id","description","frame_indices"],"additionalProperties":False}
    def branch(abstain):return {"type":"object","properties":{"motion_caption":{"const":""} if abstain else {"type":"string","minLength":1},
        "events":{"type":"array","maxItems":0} if abstain else {"type":"array","minItems":1,"items":event},
        "uncertainties":{"type":"array","items":{"type":"string"}},"abstain":{"const":abstain}},
        "required":["motion_caption","events","uncertainties","abstain"],"additionalProperties":False}
    return {"oneOf":[branch(False),branch(True)]}


def expected_attempt_artifacts(target,manifest,params_sha):
    need(target_digest(target)==target["target_sha256"],"immutable target digest changed")
    old=Path(target["original_bundle"]).resolve()
    for name in ("manifest.json","manifest.receipt.json","request.json","response.raw.txt","generation.json","receipt.json"):
        path=read_reference(target["original_refs"][name]).resolve()
        need(path==old/name,"original reference is not its pinned bundle artifact")
    anchored=load(old/"manifest.json")
    need(original.same_json(anchored,manifest) and original.same_json(anchored["input"],target["original_input"])
        and original.same_json(anchored["window"],target["window"]) and original.same_json(anchored["frames"],target["original_frames"]),"original manifest/input/frames not anchored to target")
    # The producer hashes insertion-ordered input JSON. Reuse its pinned
    # manifest order rather than the canonical recovery JSONL representation.
    source=batch.source_identity(anchored["input"])
    need(source["sha256"]==target["original_source_sha256"],"target original source changed")
    binding=anchored["binding"]
    need(binding["source_sha256"]==source["sha256"] and binding["model_sha256"]==target["original_model_identity_sha256"]
        and binding["params_sha256"]==target["original_params_sha256"],"target original model/source/params binding differs")
    batch.verify_evidence(old,binding,source)
    old_receipt=load(old/"receipt.json")
    need(old_receipt["status"]=="failed" and old_receipt["error"]==target["original_failure_reason"],"target is not the pinned original failure")
    old_generation=load(old/"generation.json")
    need(old_generation.get("model_call_performed") is True and old_generation["raw_response_sha256"]==batch.sha256(old/"response.raw.txt"),"pinned original generation/raw proof missing")
    request=load(old/"request.json")
    original.verify_request(old,anchored,{"values":{"model_path":request["model"]}})
    schema=shape_schema([f["index"] for f in anchored["frames"]]);new=request_for_recovery(request)
    lineage={"schema":SCHEMA,"target_sha256":target["target_sha256"],"original_refs":target["original_refs"],
        "original_frames":target["original_frames"],"original_input":target["original_input"],
        "original_source_sha256":target["original_source_sha256"],"original_model_identity_sha256":target["original_model_identity_sha256"],
        "original_params_sha256":target["original_params_sha256"],"window":target["window"],"run_output_path":target["run_output_path"],
        "new_request_sha256":hashlib.sha256(batch.json_bytes(new)).hexdigest(),"shape_schema_sha256":batch.digest(schema),"new_params_sha256":params_sha,
        "clarification_sha256":hashlib.sha256(CLARIFICATION.encode()).hexdigest(),
        "old_caption_supplied_to_model":False,"same_original_RGB_no_resampling":True,"original_failures_preserved":True,"full_pool_updated":False}
    return new,schema,lineage


def verify_generation_stats(generation):
    need(type(generation.get("completion_tokens")) is int and generation["completion_tokens"]>=0
        and type(generation.get("prompt_tokens")) is int and generation["prompt_tokens"]>=0
        and type(generation.get("max_tokens")) is int and generation["max_tokens"]==1200,"incomplete or untyped token statistics")
    need(isinstance(generation.get("finish_reason"),str) and "stop_reason" in generation
        and type(generation.get("eos_token_id")) is int and (generation.get("last_output_token_id") is None or type(generation["last_output_token_id"]) is int)
        and "last_output_token_id" in generation and type(generation.get("last_output_token_is_eos")) is bool,"incomplete or untyped EOS/termination statistics")
    need(generation["last_output_token_is_eos"] is (generation["last_output_token_id"] is not None and generation["last_output_token_id"]==generation["eos_token_id"]),"inconsistent returned last token/EOS statistics")
    need(original.same_json(generation.get("versions"),{"vllm":"0.31.0","xgrammar":"0.2.7","llguidance":"1.7.6"}),"generation uses unverified backend versions")


def request_for_recovery(request):
    result=copy.deepcopy(request)
    result["messages"][1]["content"].append({"type":"text","text":CLARIFICATION})
    return result


def claim_call(directory,target,params_sha):
    need(Path(directory).resolve()==target_dir(target["run_output_path"],target),"call claim outside unique planned target directory")
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/"call.claim.json"
    value={"schema":SCHEMA,"target_sha256":target["target_sha256"],"new_params_sha256":params_sha,"attempt_index":1,
        "at_most_one_new_model_call":True,"claim_is_not_generation_evidence":True}
    if path.exists():
        need(original.same_json(load(path),value),"claim belongs to a different target/protocol")
        return False
    exclusive(path,batch.json_bytes(value))
    return True


def interrupted_claim(directory,target,failure_detail=None):
    path=Path(directory)/"receipt.json"
    value={"schema":SCHEMA,"id":target["id"],"window":target["window"],"target_sha256":target["target_sha256"],
        "status":"failed","error":"claimed_call_interrupted_or_response_missing; no reissue under one-call budget",
        "failure_detail":failure_detail,
        "attempt_index":1,"model_call_performed":None,"model_call_state":"unknown_after_durable_claim",
        "raw_response_available":(Path(directory)/"response.raw.txt").is_file(),
        "files_sha256":{p.name:batch.sha256(p) for p in Path(directory).iterdir() if p.is_file() and p.name!="receipt.json" and not p.name.startswith(".")},
        "claim_sha256":batch.sha256(Path(directory)/"call.claim.json"),
        "semantic_approval":False,"human_review_completed":False,"formal_training":False,"full_pool_updated":False}
    if not path.exists():exclusive(path,batch.json_bytes(value))
    else:need(original.same_json(load(path),value),"interrupted receipt changed")
    return value


def write_completion(directory,target,manifest,raw,generation,params_sha,schema_sha):
    directory=Path(directory)
    need((directory/"call.claim.json").is_file(),"no durable claim before model call")
    need(not (directory/"receipt.json").exists(),"terminal target cannot be regenerated")
    expected_request,expected_schema,expected_lineage=expected_attempt_artifacts(target,manifest,params_sha)
    need(original.same_json(load(directory/"request.json"),expected_request) and original.same_json(load(directory/"schema.json"),expected_schema)
        and original.same_json(load(directory/"lineage.json"),expected_lineage),"completion prepared request/schema/lineage not source-anchored")
    need(schema_sha==batch.digest(expected_schema),"completion schema SHA differs from original frame-driven shape")
    generation={**generation,"model_call_performed":True,"new_params_sha256":params_sha,"shape_schema_sha256":schema_sha,
        "raw_response_sha256":hashlib.sha256(raw).hexdigest(),"structured_backend":"xgrammar"}
    for name,data in (("generation.json",batch.json_bytes(generation)),("response.raw.txt",raw)):
        path=directory/name
        if path.exists():need(path.read_bytes()==data,"interrupted raw/generation differs; never overwrite")
        else:exclusive(path,data)
    verify_generation_stats(generation)
    status,error="failed",None
    try:
        need(generation["finish_reason"]=="stop" and type(generation["completion_tokens"]) is int and 0<=generation["completion_tokens"]<1200 and type(generation.get("max_tokens")) is int and generation["max_tokens"]==1200,"truncated or non-stop recovery response")
        value=batch.parse_response(batch.json_bytes(raw.decode("utf-8")))
        candidate=batch.validate_candidate(value,manifest)
        candidate.update(recovery_protocol=SCHEMA,target_sha256=target["target_sha256"],new_params_sha256=params_sha,
            raw_response_sha256=hashlib.sha256(raw).hexdigest(),attempt_index=1,full_pool_updated=False,formal_training=False)
        candidate_path=directory/"candidate.json"
        candidate_data=batch.json_bytes(candidate)
        if candidate_path.exists():need(candidate_path.read_bytes()==candidate_data,"interrupted candidate differs from original raw")
        else:exclusive(candidate_path,candidate_data)
        status="abstain" if candidate["abstain"] else "candidate"
    except (ValueError,OSError,KeyError,TypeError,UnicodeDecodeError) as failure:error=str(failure)
    receipt={"schema":SCHEMA,"id":target["id"],"window":target["window"],"target_sha256":target["target_sha256"],"new_params_sha256":params_sha,
        "status":status,"error":error,"attempt_index":1,"model_call_performed":True,"claim_sha256":batch.sha256(directory/"call.claim.json"),
        "files_sha256":{p.name:batch.sha256(p) for p in directory.iterdir() if p.is_file() and p.name!="receipt.json" and not p.name.startswith(".")},
        "semantic_approval":False,"human_review_completed":False,"formal_training":False,"full_pool_updated":False}
    exclusive(directory/"receipt.json",batch.json_bytes(receipt))
    return receipt


def verify_attempt(directory,target,manifest,params_sha):
    directory=Path(directory).resolve()
    need(directory==target_dir(target["run_output_path"],target),"attempt outside immutable planned output")
    expected_request,expected_schema,expected_lineage=expected_attempt_artifacts(target,manifest,params_sha)
    need(original.same_json(load(directory/"request.json"),expected_request),"new request is not pinned original RGB messages plus unique clarification")
    need(original.same_json(load(directory/"schema.json"),expected_schema),"new schema differs from exact shape_schema(actual indices)")
    need(original.same_json(load(directory/"lineage.json"),expected_lineage),"new lineage is not anchored to original target/source/frames/params")
    receipt=load(directory/"receipt.json")
    need(receipt["schema"]==SCHEMA and receipt["id"]==target["id"] and receipt["target_sha256"]==target["target_sha256"] and original.same_json(receipt["window"],target["window"])
        and type(receipt["attempt_index"]) is int and receipt["attempt_index"]==1,"new attempt is not this exact target/protocol")
    original.false_approval(receipt)
    claim=load(directory/"call.claim.json")
    need(original.same_json(claim,{"schema":SCHEMA,"target_sha256":target["target_sha256"],"new_params_sha256":params_sha,"attempt_index":1,
        "at_most_one_new_model_call":True,"claim_is_not_generation_evidence":True}),"attempt claim target/params changed")
    need(batch.sha256(directory/"call.claim.json")==receipt["claim_sha256"],"claim proof changed")
    if receipt.get("model_call_state")=="unknown_after_durable_claim":
        need(receipt["status"]=="failed" and receipt["model_call_performed"] is None
            and receipt["raw_response_available"] is (directory/"response.raw.txt").is_file(),"unresolved call was promoted to candidate")
        need(not (directory/"candidate.json").exists(),"unresolved call still has candidate artifact")
        for name,sha in receipt["files_sha256"].items():original.sha_file(original.beneath(directory/name,directory),sha)
        return {"status":"failed","caption":"","events":[],"error":receipt["error"],"model_call_performed":None,
            "attempt_directory":str(directory),"raw_response_path":str(directory/"response.raw.txt") if (directory/"response.raw.txt").exists() else None,"raw_lexical_risk_flags":{}}
    need(receipt["new_params_sha256"]==params_sha and receipt["model_call_performed"] is True,"new protocol/generation identity differs")
    for name,sha in receipt["files_sha256"].items():original.sha_file(original.beneath(directory/name,directory),sha)
    for required in ("request.json","schema.json","lineage.json","generation.json","response.raw.txt"):
        need(required in receipt["files_sha256"],"new response lineage lacks "+required)
    generation=load(directory/"generation.json");raw=(directory/"response.raw.txt").read_bytes()
    need(generation["new_params_sha256"]==params_sha and generation["shape_schema_sha256"]==batch.digest(load(directory/"schema.json")) and generation["raw_response_sha256"]==hashlib.sha256(raw).hexdigest()
        and generation["model_call_performed"] is True and generation["structured_backend"]=="xgrammar","new generation/raw/backend binding differs")
    verify_generation_stats(generation)
    candidate=None;error=None;value=None
    try:
        need(generation["finish_reason"]=="stop" and type(generation["completion_tokens"]) is int and 0<=generation["completion_tokens"]<1200 and type(generation.get("max_tokens")) is int and generation["max_tokens"]==1200,"truncated or non-stop recovery response")
        value=batch.parse_response(batch.json_bytes(raw.decode("utf-8")))
        candidate=batch.validate_candidate(value,manifest)
    except (ValueError,TypeError,KeyError,UnicodeDecodeError) as failure:error=str(failure)
    if error is not None:
        need(receipt["status"]=="failed" and receipt["error"]==error and not (directory/"candidate.json").exists(),"failed raw was cleaned or promoted")
        return {"status":"failed","caption":"","events":[],"error":error,"generation":generation,"model_call_performed":True,
            "attempt_directory":str(directory),"raw_response_path":str(directory/"response.raw.txt"),"raw_lexical_risk_flags":original.risk_flags(value) if value else {}}
    candidate.update(recovery_protocol=SCHEMA,target_sha256=target["target_sha256"],new_params_sha256=params_sha,
        raw_response_sha256=hashlib.sha256(raw).hexdigest(),attempt_index=1,full_pool_updated=False,formal_training=False)
    need(original.same_json(load(directory/"candidate.json"),candidate),"new candidate is not the typed original new raw response")
    need(receipt["status"]==("abstain" if candidate["abstain"] else "candidate") and receipt["error"] is None,"new status differs from original validator")
    return {"status":receipt["status"],"caption":candidate["motion_caption"],"events":candidate["events"],"error":None,"generation":generation,"model_call_performed":True,
        "attempt_directory":str(directory),"raw_response_path":str(directory/"response.raw.txt"),"raw_lexical_risk_flags":original.risk_flags(value)}


def recovery_params(plan_path,plan,model_path):
    return {"schema":SCHEMA,"source_sha256":batch.sha256(Path(__file__)),"original_producer_sha256":PRODUCER_SHA,"original_exporter_sha256":EXPORTER_SHA,
            "plan_ref":reference(plan_path),"run_output_path":plan["run_output_path"],"model_identity_sha256":MODEL_SHA,"model_path":model_path,"temperature":0.0,"seed":0,
            "max_tokens":1200,"max_model_len":16384,"image_limit":16,"max_pixels":200704,"tp":2,"batch_size":1,"max_num_seqs":1,
            "gpu_memory_utilization":.88,"target_words":90,"hard_words":140,"structured_backend":"xgrammar","shape_only_no_pair_forcing":True,
            "clarification_sha256":hashlib.sha256(CLARIFICATION.encode()).hexdigest(),"max_new_observations_per_target":1,
            "versions":{"vllm":"0.31.0","xgrammar":"0.2.7","llguidance":"1.7.6"}}


def run_stage(args):
    need(Path(args.model).is_absolute(),"explicit absolute local model path required")
    plan,targets=load_plan(args.plan)
    need(str(Path(args.out).resolve())==plan["run_output_path"],"run --out differs from immutable plan run output")
    manifest,original_rows,launch=verify_original(plan["original_export"],read_reference(plan["original_launch_ref"]))
    expected,blocked=targets_from_rows(original_rows,plan["run_output_path"])
    need(original.same_json(expected,targets) and original.same_json(blocked,plan["blocked_scope"]),"recovery targets no longer match only original failures")
    out=safe_out(args.out,[plan["original_export"],Path(args.plan).parent,args.model,*[j["out"] for j in launch["jobs"]]])
    with batch.bundle_lock(out):
        inspected={t["key"]:inspect_target(t,[j["out"] for j in launch["jobs"]]) for t in targets}
        ref=load(read_reference(plan["model_reference_ref"]))["model_identity"]
        need(ref["sha256"]==MODEL_SHA and str(Path(args.model).resolve())==ref["root"],"different explicitly bound local model path")
        own=batch.file_identity(Path(__file__))
        params=recovery_params(args.plan,plan,ref["root"])
        params_sha=batch.digest(params)
        run_path=out/"run.receipt.json"
        if run_path.exists():need(original.same_json(load(run_path)["params"],params),"existing run belongs to another protocol")
        else:exclusive(run_path,batch.json_bytes({"schema":SCHEMA,"params":params,"params_sha256":params_sha,"model_reference_ref":plan["model_reference_ref"],
            "selected_target_keys":[t["key"] for t in targets],"model_loaded":False,"semantic_approval":False,"human_review_completed":False,"formal_training":False}))
        pending=[]
        for t in targets:
            directory=target_dir(out,t);directory.mkdir(parents=True,exist_ok=True)
            m,source,request=inspected[t["key"]];new=request_for_recovery(request);schema=shape_schema([f["index"] for f in m["frames"]])
            expected_new,expected_schema,lineage=expected_attempt_artifacts(t,m,params_sha)
            need(original.same_json(new,expected_new) and original.same_json(schema,expected_schema),"prepared request/schema not original target")
            for name,value in (("request.json",new),("schema.json",schema),("lineage.json",lineage)):
                path=directory/name
                if path.exists():need(original.same_json(load(path),value),"prepared new attempt protocol changed")
                else:exclusive(path,batch.json_bytes(value))
            if (directory/"receipt.json").exists():verify_attempt(directory,t,m,params_sha);continue
            if (directory/"call.claim.json").exists():
                # A crash after receiving both raw and token statistics can be
                # finalized, but an uncertain request is never sent again.
                claim_call(directory,t,params_sha)
                if (directory/"response.raw.txt").exists() and (directory/"generation.json").exists():
                    g=load(directory/"generation.json");write_completion(directory,t,m,(directory/"response.raw.txt").read_bytes(),g,params_sha,batch.digest(schema))
                else:interrupted_claim(directory,t)
                continue
            pending.append((t,directory,m,source,new,schema))
        if pending:
            for env in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_HUB_DISABLE_TELEMETRY","VLLM_NO_USAGE_STATS","DO_NOT_TRACK"):os.environ[env]="1"
            from vllm import LLM,SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            from vllm.config import StructuredOutputsConfig
            from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar
            versions={name:importlib.metadata.version(name) for name in ("vllm","xgrammar","llguidance")}
            need(versions=={"vllm":"0.31.0","xgrammar":"0.2.7","llguidance":"1.7.6"},"unverified structured backend versions")
            sampling=[SamplingParams(temperature=0.0,seed=0,max_tokens=1200,structured_outputs=StructuredOutputsParams(json=p[-1])) for p in pending]
            for sp in sampling:validate_xgrammar_grammar(sp)
            model=batch.model_identity(args.model);need(model["sha256"]==MODEL_SHA,"local model content changed")
            original.verify_model(model,ref,ref["root"]);batch.verify_current(model,model=True)
            need(batch.sha256(Path(__file__))==own["sha256"],"recovery runner changed before loading")
            llm=LLM(model=ref["root"],tensor_parallel_size=2,max_model_len=16384,max_num_seqs=1,gpu_memory_utilization=.88,
                limit_mm_per_prompt={"image":16,"video":0},mm_processor_kwargs={"max_pixels":200704},trust_remote_code=False,seed=0,
                enable_prefix_caching=True,structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
            need(llm.llm_engine.vllm_config.structured_outputs_config.backend=="xgrammar","structured backend drifted/fallback")
            loaded={"schema":SCHEMA,"model_identity":model,"params_sha256":params_sha,"versions":versions,
                "actual_backend":"xgrammar","fallback_allowed":False,"model_loaded":True,"semantic_approval":False,"formal_training":False}
            load_path=out/"load.receipt.json"
            if load_path.exists():
                previous=load(load_path)
                need(previous["params_sha256"]==params_sha and previous["versions"]==versions and previous["actual_backend"]=="xgrammar" and previous["fallback_allowed"] is False,"existing model load belongs to a different protocol")
                original.verify_model(previous["model_identity"],ref,ref["root"])
            else:exclusive(load_path,batch.json_bytes(loaded))
            for (t,directory,m,source,request,schema),sp in zip(pending,sampling):
                inspect_target(t,[j["out"] for j in launch["jobs"]])
                if not claim_call(directory,t,params_sha):raise ValueError("concurrent duplicate call claim; no new request sent")
                try:
                    outputs=llm.chat([request["messages"]],sp,use_tqdm=False)
                    need(len(outputs)==1 and len(outputs[0].outputs)==1,"unexpected response count")
                    output,completion=outputs[0],outputs[0].outputs[0];tokens=list(completion.token_ids);eos=llm.get_tokenizer().eos_token_id
                    g={"finish_reason":completion.finish_reason,"stop_reason":completion.stop_reason,"prompt_tokens":len(output.prompt_token_ids or []),
                        "completion_tokens":len(tokens),"max_tokens":1200,"eos_token_id":eos,"last_output_token_id":tokens[-1] if tokens else None,
                        "last_output_token_is_eos":bool(tokens) and tokens[-1]==eos,"versions":versions}
                    write_completion(directory,t,m,completion.text.encode("utf-8"),g,params_sha,batch.digest(schema))
                except Exception as failure:
                    if not (directory/"receipt.json").exists():
                        if (directory/"candidate.json").exists():raise
                        interrupted_claim(directory,t,f"{type(failure).__name__}: {failure}")
        results=[]
        for t in targets:
            directory=target_dir(out,t)
            checked=verify_attempt(directory,t,inspected[t["key"]][0],params_sha)
            results.append({"key":t["key"],"status":checked["status"],"receipt_ref":reference(directory/"receipt.json")})
        summary={"schema":SCHEMA,"selected_targets":len(targets),"processed_targets":len(results),"new_window_counts":dict(Counter(r["status"] for r in results)),
            "plan_ref":reference(args.plan),"params_sha256":params_sha,"results":results,"one_pass_terminal":True,"max_new_observations_per_target":1,
            "semantic_approval":False,"human_review_completed":False,"formal_training":False,"full_pool_updated":False}
        path=out/"summary.json"
        if path.exists():need(original.same_json(load(path),summary),"completed recovery summary differs")
        else:exclusive(path,batch.json_bytes(summary))
        return {"status":"one_pass_recovery_terminal_unreviewed","targets":len(results),"counts":summary["new_window_counts"],"formal_training":False}


def merge_rows(rows,targets,replacements):
    lookup={t["key"]:t for t in targets}
    need(set(replacements)==set(lookup),"missing/foreign recovery result; do not fill pending slots")
    output=[]
    for source_row in rows:
        row=copy.deepcopy(source_row)
        for i,w in enumerate(row["windows"]):
            key=f"{row['id']}:w{i}"
            if key not in lookup:continue
            need(w["status"]=="failed","recovery cannot replace an original good candidate or valid abstention")
            chosen=replacements[key]
            need(chosen["status"] in original.TERMINAL,"new result is not terminal")
            row["windows"][i]={**w,"status":chosen["status"],"caption":chosen["caption"],"events":chosen["events"],"error":chosen["error"],
                "chosen_protocol":SCHEMA,"original_window":w,"recovery":chosen,"original_raw_preserved":True,
                "original_bundle":w.get("bundle"),"bundle":chosen.get("attempt_directory"),
                "raw_response_path":chosen.get("raw_response_path"),"generation":chosen.get("generation"),
                "raw_lexical_risk_flags":chosen.get("raw_lexical_risk_flags",{}),"flags_are_diagnostic_not_semantic_approval":True}
        counts=Counter(w["status"] for w in row["windows"])
        row["status"]="failed" if not row["windows"] or counts["failed"] else "candidate" if counts["candidate"] else "abstain"
        row["original_status"]=source_row["status"]
        row["original_source"]=source_row.get("source")
        row["source"]="independent_visual_candidates_with_explicit_single_recovery_lineage"
        row["not_original_free_protocol_export"]=True
        row["timed_caption"]="\n".join(f"[{w['window']['start']:.3f}-{w['window']['end']:.3f}s] {w['caption']}" for w in row["windows"] if w["status"]=="candidate" and w["caption"])
        row.update(semantic_approval=False,human_review_completed=False,formal_training=False,ground_truth=False)
        output.append(row)
    return output


def verify_preserved_rows(rows,launch):
    runs={j["shard"]:load(original.control_path(j,8)/"run.receipt.json") for j in launch["jobs"]}
    for row in rows:
        job=launch["jobs"][row["shard"]]
        if row["windows"]:
            preserved_input=load(row["final_path"])["input"]
        else:
            preserved_input=load(row["failure_receipt_path"])["input"]
        need(original.same_json(preserved_input,row["original_input"]),"preserved original input differs from producer artifact")
        source=batch.source_identity(preserved_input)
        probe=batch.probe_video(preserved_input["video_path"],original.argv_values(job)["--ffprobe"])
        need(source["sha256"]==row["source_identity"]["sha256"],"original source bytes changed before recovery export")
        if row["windows"]:
            path=Path(row["final_path"]);original.sha_file(path,row["final_sha256"])
            final=load(path)
            indexed={"id":row["id"],"status":row["status"],"final_path":str(path),"final_sha256":row["final_sha256"],
                "counts":final["counts"],"observation_complete_without_failures":final["observation_complete_without_failures"]}
        else:indexed={"id":row["id"],"status":"failed","error":row["error"]}
        rebuilt=original.verify_record(indexed,preserved_input,job,runs[row["shard"]],probe,source,
            (original.control_path(job,8)/"load.receipt.json").is_file())
        for key,value in rebuilt.items():
            need(key in row and original.same_json(row[key],value),"original verified-export row differs from rebuilt producer proof: "+key)


def export_stage(args):
    plan,targets=load_plan(args.plan)
    manifest,rows,launch=verify_original(plan["original_export"],read_reference(plan["original_launch_ref"]))
    expect,blocked=targets_from_rows(rows,plan["run_output_path"])
    need(original.same_json(expect,targets) and original.same_json(blocked,plan["blocked_scope"]),"original failure target selection changed")
    need(str(Path(args.run_out).resolve())==plan["run_output_path"],"export --run-out differs from immutable plan run output")
    runroot=Path(args.run_out).resolve();summary_path=runroot/"summary.json"
    if not summary_path.is_file() or not Path(args.resource_exit).is_file():raise original.Pending("recovery worker/summary not actually terminal")
    exit_receipt=load(args.resource_exit);need(type(exit_receipt["exit_code"]) is int and exit_receipt["exit_code"] in (0,1),"recovery resource child did not exit normally")
    need(exit_receipt["source_unchanged"] is True and exit_receipt["inputs_unchanged"] is True,"recovery resource reports changed source/input")
    summary=load(summary_path);run=load(runroot/"run.receipt.json");params=run["params"]
    need(exit_receipt["summary_path"]==str(summary_path) and original.same_json(exit_receipt["summary"],summary)
        and exit_receipt["run_source_sha256"]==params["source_sha256"] and exit_receipt["plan_sha256"]==batch.sha256(Path(args.plan)),
        "resource exit is not the exact recovery source/plan/run summary")
    ref=load(read_reference(plan["model_reference_ref"]))["model_identity"]
    need(original.same_json(params,recovery_params(args.plan,plan,ref["root"])) and batch.digest(params)==run["params_sha256"]==summary["params_sha256"],"recovery source/fixed parameters mismatch")
    need(original.same_json(params["plan_ref"],reference(args.plan)),"recovery run refers to another immutable target plan")
    need(run["schema"]==summary["schema"]==SCHEMA and run["selected_target_keys"]==[t["key"] for t in targets],"recovery run/source target order differs")
    original.false_approval(run);original.false_approval(summary);original.false_approval(exit_receipt)
    need(type(summary["max_new_observations_per_target"]) is int and summary["max_new_observations_per_target"]==1,"recovery summary budget changed")
    need(summary["one_pass_terminal"] is True and type(summary["selected_targets"]) is int and type(summary["processed_targets"]) is int and summary["selected_targets"]==summary["processed_targets"]==len(targets)
        and [r["key"] for r in summary["results"]]==[t["key"] for t in targets],"partial/duplicate/reordered recovery summary")
    if (runroot/"load.receipt.json").exists():
        loaded=load(runroot/"load.receipt.json")
        need(loaded["params_sha256"]==run["params_sha256"] and loaded["actual_backend"]=="xgrammar" and loaded["fallback_allowed"] is False and original.same_json(loaded["versions"],params["versions"]),"recovery load backend/protocol mismatch")
        ref=load(read_reference(plan["model_reference_ref"]))["model_identity"]
        original.verify_model(loaded["model_identity"],ref,params["model_path"])
    verify_preserved_rows(rows,launch)  # Final audit, only after both original and recovery workers are terminal.
    replacements={}
    for t,result in zip(targets,summary["results"]):
        directory=target_dir(runroot,t);read_reference(result["receipt_ref"])
        need(Path(result["receipt_ref"]["path"]).resolve()==(directory/"receipt.json").resolve(),"new result path differs from target")
        m,source,request=inspect_target(t,[j["out"] for j in launch["jobs"]])
        schema=shape_schema([f["index"] for f in m["frames"]]);new=request_for_recovery(request)
        need(original.same_json(load(directory/"request.json"),new) and original.same_json(load(directory/"schema.json"),schema),"recovery request contains old-caption prior, modified RGB or another decoding schema")
        lineage=load(directory/"lineage.json")
        need(lineage["target_sha256"]==t["target_sha256"] and lineage["new_params_sha256"]==run["params_sha256"]
            and lineage["old_caption_supplied_to_model"] is False and lineage["same_original_RGB_no_resampling"] is True
            and original.same_json(lineage["original_refs"],t["original_refs"]) and original.same_json(lineage["original_frames"],t["original_frames"])
            and lineage["new_request_sha256"]==batch.sha256(directory/"request.json") and lineage["shape_schema_sha256"]==batch.digest(schema),"new attempt lineage differs from its source target")
        checked=verify_attempt(directory,t,m,run["params_sha256"])
        need(checked["status"]==result["status"],"new summary status changed")
        if checked["model_call_performed"] is True:need((runroot/"load.receipt.json").is_file(),"new generation lacks actual load receipt")
        checked.update(receipt_ref=reference(directory/"receipt.json"),original_target_sha256=t["target_sha256"],attempt_index=1,new_protocol=SCHEMA)
        replacements[t["key"]]=checked
    need(original.same_json(summary["new_window_counts"],dict(Counter(v["status"] for v in replacements.values()))),"recovery summary counts differ from actual attempts")
    merged=merge_rows(rows,targets,replacements);availability=original.caption_availability(merged)
    availability["max_additional_failed_attempts"]=1
    counts=Counter(r["status"] for r in merged);windows=Counter(w["status"] for r in merged for w in r["windows"])
    raw=payload(merged)
    result={**availability,"schema":SCHEMA,"status":"terminal_verified_caption_delivery_unreviewed" if availability["caption_delivery_complete"] else "terminal_verified_caption_incomplete",
        "technical_terminal_accounting_complete":True,"expected_ids":12000,"verified_ids":len(merged),"expected_window_count":24000,
        "original_export_manifest_ref":plan["original_manifest_ref"],"original_export_captions_ref":plan["original_captions_ref"],
        "recovery_plan_ref":reference(args.plan),"recovery_summary_ref":reference(summary_path),"resource_exit_ref":reference(args.resource_exit),
        "recovery_run_ref":reference(runroot/"run.receipt.json"),
        "recovery_load_ref":reference(runroot/"load.receipt.json") if (runroot/"load.receipt.json").is_file() else None,
        "uncertain_model_call_target_count":sum(v["model_call_performed"] is None for v in replacements.values()),
        "new_source_sha256":batch.sha256(Path(__file__)),"new_window_counts":summary["new_window_counts"],"selected_failed_windows":len(targets),
        "blocked_scope":blocked,"original_sid_status_counts":manifest["sid_status_counts"],"original_window_status_counts":manifest["window_status_counts"],
        "chosen_sid_status_counts":dict(counts),"chosen_window_status_counts":dict(windows),"captions_sha256":hashlib.sha256(raw).hexdigest(),
        "chosen_raw_lexical_flag_window_counts":dict(Counter(k for r in merged for w in r["windows"] for k in w.get("raw_lexical_risk_flags",{}))),
        "flags_are_diagnostic_not_semantic_verdicts":True,
        "every_original_raw_preserved":True,"old_good_and_valid_abstain_never_regenerated":True,"remaining_caption_gaps_need_honest_report_or_explicit_decision":not availability["caption_delivery_complete"],
        "semantic_approval":False,"human_review_completed":False,"formal_training":False,"ground_truth":False,"full_pool_updated":False,
        "audio_video_alignment_verified":False,"motions_on_beats_verified":False}
    out=safe_out(args.out,[plan["original_export"],Path(args.plan).parent,runroot,*[j["out"] for j in launch["jobs"]]])
    need(not out.exists(),"new independent recovery export destination required")
    out.mkdir(parents=True,exist_ok=False);exclusive(out/"captions.jsonl",raw);exclusive(out/"manifest.json",batch.json_bytes(result))
    return {"status":result["status"],"out":str(out),"usable_full_caption_count":availability["usable_full_caption_count"],"caption_delivery_complete":availability["caption_delivery_complete"],"formal_training":False}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest="stage",required=True)
    plan=sub.add_parser("plan");plan.add_argument("--original-plan",required=True);plan.add_argument("--original-export",required=True);plan.add_argument("--out",required=True);plan.add_argument("--run-out",required=True)
    run=sub.add_parser("run");run.add_argument("--plan",required=True);run.add_argument("--out",required=True);run.add_argument("--model",required=True)
    exp=sub.add_parser("export");exp.add_argument("--plan",required=True);exp.add_argument("--run-out",required=True);exp.add_argument("--resource-exit",required=True);exp.add_argument("--out",required=True)
    args=ap.parse_args(argv)
    try:result={"plan":plan_stage,"run":run_stage,"export":export_stage}[args.stage](args)
    except original.Pending as error:
        print(json.dumps({"status":"pending","reason":str(error),"stage":args.stage,"recovery_performed":False,"formal_training":False}),flush=True);return 2
    print(json.dumps(result),flush=True);return 0


if __name__=="__main__":
    try:sys.exit(main())
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
        print(json.dumps({"status":"rejected","reason":f"{type(error).__name__}: {error}","formal_training":False}),file=sys.stderr,flush=True);sys.exit(1)
