"""One independent constrained-decoding trial for two existing failed windows.

Never resumes or rewrites the free-decoding producer. Read original evidence
and messages, then let the local model generate a new response under a new
schema. A valid schema is not a semantic approval, and no evidence is patched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import sys

import ms_visual_batch as batch

SCHEMA = "motionscript_visual_constrained_failed2_v1"
PRODUCER_SHA = "b362dc5275b19c70a6f4f4d149b974e75571ad07e7e4f1f0a2251222fd839204"
MODEL_SHA = "26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c"
CASE_SPECS = [
    {"id":"TKBOLWtIPQzcxGLa","shard":1,"window":{"start":0.0,"end":2.0},
        "request_sha256":"81247c9d30a9e659fd18ca84e810ccfca4a2a0d7d4d73ed4c6aced446f6611df",
        "raw_sha256":"3ef8a2d314761cd6c7d0f60c66eea79f541c97a4858e2c89644256fbd60fc8d9"},
    {"id":"HtezJbaEumeOaxQT","shard":2,"window":{"start":2.0,"end":5.0625},
        "request_sha256":"6793d79eba5db5330b248492b60bff6330d4d28451bf6990f074338043cc782a",
        "raw_sha256":"adc6ccd63e44457b497948164bee363666a39f310f2725f3baee113f7b9a1547"},
]


def need(value,message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return batch.strict_json(Path(path).read_text(encoding="utf-8"))


def typed_equal(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False,separators=(",",":")) == json.dumps(b,sort_keys=True,allow_nan=False,separators=(",",":"))


def window_schema(indices):
    need(isinstance(indices,list) and 2 <= len(indices) <= 16 and all(type(i) is int for i in indices)
         and indices == sorted(set(indices)), "need actual ordered distinct source frame indices")
    pairs = [list(pair) for pair in itertools.combinations(indices,2)]
    frames = {"type":"array","minItems":2,"maxItems":2,
        "items":{"type":"integer","enum":indices},"enum":pairs}
    event = {"type":"object","properties":{"id":{"type":"string","minLength":1},
        "description":{"type":"string","minLength":1},"frame_indices":frames},
        "required":["id","description","frame_indices"],"additionalProperties":False}
    def branch(abstain):
        return {"type":"object","properties":{
            "motion_caption":{"const":""} if abstain else {"type":"string","minLength":1},
            "events":{"type":"array","maxItems":0} if abstain else {"type":"array","minItems":1,"items":event},
            "uncertainties":{"type":"array","items":{"type":"string"}},"abstain":{"const":abstain}},
            "required":["motion_caption","events","uncertainties","abstain"],"additionalProperties":False}
    return {"oneOf":[branch(False),branch(True)]}


def inspect_original(bundle,run_path,spec,ffprobe):
    bundle = Path(bundle).resolve(strict=True)
    manifest = read_json(bundle/"manifest.json")
    row = manifest["input"]
    need(row["id"] == spec["id"] and typed_equal(manifest["window"],spec["window"]), "only the two explicitly selected original failed windows are allowed")
    source = batch.source_identity(row)
    binding = manifest["binding"]
    run = read_json(run_path)
    need(run["code_identity"]["files"]["ms_visual_batch.py"]["sha256"] == PRODUCER_SHA, "original producer is not frozen b362dc")
    need(run["model_identity"]["sha256"] == binding["model_sha256"] == MODEL_SHA, "original failure is not the specified local 32B identity")
    need(binding["source_sha256"] == source["sha256"] and binding["params_sha256"] == run["params_identity"]["sha256"], "original failure source/parameters changed")
    batch.verify_evidence(bundle,binding,source)
    receipt = batch.resume_receipt(bundle,binding,source)
    need(receipt is not None and receipt["status"] == "failed", "do not regenerate a successful or active original window")
    need(receipt["error"] == "event evidence must name at least two distinct supplied frames", "selected original failure reason differs from authorized trial")
    need(batch.sha256(bundle/"request.json") == spec["request_sha256"] and batch.sha256(bundle/"response.raw.txt") == spec["raw_sha256"], "authorized original request/raw content changed")
    batch.validate_generation(bundle,required=True)
    generation = read_json(bundle/"generation.json")
    need(generation["model_call_performed"] is True, "original failure lacks a real generation receipt")
    probe = batch.probe_video(row["video_path"],ffprobe)
    need(manifest["media_end"] == probe["media_end"], "source video end changed")
    selected = batch.select_frames(probe["times"],spec["window"]["start"],spec["window"]["end"])
    need([frame["index"] for frame in manifest["frames"]] == selected, "original frame selection differs from actual video")
    for frame in manifest["frames"]:
        i = frame["index"]
        need(frame["time"] == probe["times"][i] and frame["source_timestamp"] == probe["source_timestamps"][i], "source PTS changed")
    request_bytes = (bundle/"request.json").read_bytes()
    request = batch.strict_json(request_bytes)
    need(request["messages"][0] == {"role":"system","content":batch.SYSTEM}, "original system prompt changed")
    need(request["temperature"] == 0.0 and request["max_tokens"] == 1200, "original sampling protocol changed")
    need(len(request["messages"]) == 2 and request["messages"][1]["role"] == "user", "unexpected original message structure")
    need(sum(c["type"] == "image_url" for c in request["messages"][1]["content"]) == len(selected), "original multi-image count changed")
    need(request["messages"][1]["content"][0]["text"].startswith(batch.OBSERVE_TEXT), "original user schema reminder changed")
    need("Allowed Source frame indices for this window: "+json.dumps(selected) in request["messages"][1]["content"][0]["text"], "original per-window index reminder changed")
    names = ("manifest.json","manifest.receipt.json","request.json","response.raw.txt","generation.json","receipt.json")
    hashes = {name:batch.sha256(bundle/name) for name in names}
    raw_original = (bundle/"response.raw.txt").read_text(encoding="utf-8")
    facts = batch.parse_response(batch.json_bytes(raw_original))
    return {"bundle":str(bundle),"run_receipt_path":str(Path(run_path).resolve()),"run_receipt_sha256":batch.sha256(Path(run_path)),
        "manifest":manifest,"source":source,"request_bytes":request_bytes,"request":request,
        "original_files_sha256":hashes,"original_raw":raw_original,"original_motion_caption":facts.get("motion_caption"),
        "original_failure_reason":receipt["error"],"schema":window_schema(selected)}


def verify_original_unchanged(case):
    bundle = Path(case["bundle"])
    for name,digest in case["original_files_sha256"].items():
        need(batch.sha256(bundle/name) == digest, "original evidence/response changed during trial: "+name)
    need(batch.sha256(Path(case["run_receipt_path"])) == case["run_receipt_sha256"], "original run receipt changed")
    for frame in case["manifest"]["frames"]:
        path = (bundle/frame["file"]).resolve()
        need(path.is_relative_to(bundle) and batch.sha256(path) == frame["sha256"], "original frame evidence changed")
    batch.verify_source(case["source"])


def prepare_case(case,out,model_identity,new_params_sha):
    index = 0 if case["manifest"]["window"]["start"] == 0.0 else 1
    destination = Path(out)/"cases"/(case["manifest"]["input"]["id"]+f"_w{index:04d}")
    # The window bounds, not this display name, determine the case identity.
    destination.mkdir(parents=True,exist_ok=False)
    batch.atomic_write(destination/"request.json",case["request_bytes"])
    batch.atomic_write(destination/"schema.json",batch.json_bytes(case["schema"]))
    frames = []
    old = Path(case["bundle"])
    for frame in case["manifest"]["frames"]:
        path = (old/frame["file"]).resolve()
        need(path.is_relative_to(old), "original frame outside immutable bundle")
        frames.append({**frame,"original_absolute_path":str(path)})
    lineage = {"schema":SCHEMA,"id":case["manifest"]["input"]["id"],"window":case["manifest"]["window"],
        "original_bundle":case["bundle"],"original_files_sha256":case["original_files_sha256"],
        "original_run_receipt_path":case["run_receipt_path"],"original_run_receipt_sha256":case["run_receipt_sha256"],
        "original_motion_caption":case["original_motion_caption"],"original_failure_reason":case["original_failure_reason"],
        "original_input":case["manifest"]["input"],"original_source_identity":case["source"],
        "original_frames":frames,"new_params_sha256":new_params_sha,"new_model_identity_sha256":model_identity["sha256"],
        "new_schema_sha256":batch.digest(case["schema"]),"new_request_sha256":batch.sha256(destination/"request.json"),
        "same_original_request_bytes":True,"same_original_frames_no_resampling":True,"new_independent_decoding_protocol":True,
        "not_free_decoding_resume":True,"full_pool_updated":False,"human_review_completed":False,"semantic_approval":False,"formal_training":False}
    need(lineage["new_request_sha256"] == case["original_files_sha256"]["request.json"], "new messages differ from original evidence")
    batch.atomic_write(destination/"lineage.json",batch.json_bytes(lineage))
    return destination,lineage


def finish_response(destination,case,lineage,raw,generation):
    destination = Path(destination)
    need(not (destination/"response.raw.txt").exists() and not (destination/"receipt.json").exists(), "one trial only; existing response cannot be overwritten or silently resumed")
    generation = {**generation,"raw_response_sha256":hashlib.sha256(raw).hexdigest(),
        "decoding_schema_sha256":lineage["new_schema_sha256"],"backend":"xgrammar","model_call_performed":True}
    batch.atomic_write(destination/"generation.json",batch.json_bytes(generation))
    batch.atomic_write(destination/"response.raw.txt",raw)
    status,error = "failed",None
    try:
        need(generation["finish_reason"] == "stop" and type(generation["completion_tokens"]) is int and 0 <= generation["completion_tokens"] < 1200 and generation["max_tokens"] == 1200, "generation truncated or not stopped normally")
        verify_original_unchanged(case)
        original = batch.parse_response(batch.json_bytes(raw.decode("utf-8")))
        candidate = batch.validate_candidate(original,case["manifest"])
        pairs = case["schema"]["oneOf"][0]["properties"]["events"]["items"]["properties"]["frame_indices"]["enum"]
        for event in original["events"]:
            need(event["frame_indices"] in pairs, "model did not generate an exact allowed distinct source-frame pair")
        candidate.update(decoding_protocol=SCHEMA,new_schema_sha256=lineage["new_schema_sha256"],
            new_params_sha256=lineage["new_params_sha256"],lineage_sha256=batch.sha256(destination/"lineage.json"),
            raw_response_sha256=hashlib.sha256(raw).hexdigest(),original_request_sha256=case["original_files_sha256"]["request.json"],
            original_bundle=case["bundle"],new_independent_candidate=True,full_pool_updated=False,formal_training=False)
        batch.atomic_write(destination/"candidate.json",batch.json_bytes(candidate))
        status = "abstain" if candidate["abstain"] else "candidate"
    except (ValueError,KeyError,TypeError,UnicodeDecodeError,OSError) as failure:
        error = str(failure)
    receipt = {"schema":SCHEMA,"id":lineage["id"],"window":lineage["window"],"status":status,"error":error,
        "original_failure_preserved":True,"lineage_sha256":batch.sha256(destination/"lineage.json"),
        "files_sha256":{name:batch.sha256(destination/name) for name in ("request.json","schema.json","lineage.json","response.raw.txt","generation.json","candidate.json") if (destination/name).exists()},
        "schema_validity_is_not_motion_fact_approval":True,"full_pool_updated":False,
        "semantic_approval":False,"human_review_completed":False,"formal_training":False}
    batch.atomic_write(destination/"receipt.json",batch.json_bytes(receipt))
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",action="append",required=True,help="Exactly TKBOL first window and Htez second window, in this order")
    ap.add_argument("--out",required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--model-reference",required=True)
    ap.add_argument("--original-full-root",required=True)
    ap.add_argument("--ffprobe",default="/opt/ms_tools/ffprobe")
    ap.add_argument("--tp",type=int,default=2)
    ap.add_argument("--max-num-seqs",type=int,default=1)
    ap.add_argument("--max-pixels",type=int,default=200704)
    ap.add_argument("--gpu-memory-utilization",type=float,default=.88)
    args = ap.parse_args(argv)
    need(len(args.bundle)==2 and args.tp==2 and args.max_num_seqs==1 and args.max_pixels==200704
        and args.gpu_memory_utilization==.88, "only the authorized two-window TP2/batch1/seq1 fixed-budget trial is supported")
    need(Path(args.model).is_absolute(), "local model path must be absolute")
    need(batch.sha256(Path(batch.__file__)) == PRODUCER_SHA, "do not modify or replace the frozen original producer")
    out = Path(args.out).resolve()
    full = Path(args.original_full_root).resolve(strict=True)
    need(not batch.overlaps(out,full), "independent output must not contain or overwrite the full producer tree")
    need(not batch.overlaps(out,Path(args.model)), "never write trial artifacts inside the model directory")
    need(not Path(args.model_reference).resolve().is_relative_to(out), "trial output must not contain model reference receipt")
    for path in args.bundle:
        need(Path(path).is_absolute() and Path(path).resolve().is_relative_to(full), "original bundle is not inside the explicitly selected full output")
    need(not (out/"run.receipt.json").exists() and not (out/"cases").exists(), "trial already prepared; use an explicitly new trial, never overwrite this attempt")
    with batch.bundle_lock(out):
        code = batch.code_identity()
        self_identity = batch.file_identity(Path(__file__))
        cases = []
        for bundle,spec in zip(args.bundle,CASE_SPECS):
            run_path = full/f"shard{spec['shard']}"/"shards"/f"shard-{spec['shard']:05d}-of-00008"/"run.receipt.json"
            need(Path(bundle).resolve().is_relative_to(full/f"shard{spec['shard']}"/"ids"/spec["id"]), "bundle is not from its explicitly selected original SID/shard")
            cases.append(inspect_original(bundle,run_path,spec,args.ffprobe))
        model = batch.model_identity(args.model)
        reference = read_json(args.model_reference)["model_identity"]
        need(model["sha256"] == reference["sha256"] == MODEL_SHA and len(model["files"]) == 25
            and set(model["files"]) == set(reference["files"]), "new trial does not use the exact existing 25-file local 32B model identity")
        for name,row in model["files"].items():
            need(row["sha256"] == reference["files"][name]["sha256"] and row["stat"]["size"] == reference["files"][name]["stat"]["size"], "local model differs from reference")
        need(all(c["request"]["model"] == model["root"] for c in cases), "model path differs from original messages; no request rewrite permitted")
        params = {"schema":SCHEMA,"temperature":0.0,"seed":0,"max_tokens":1200,"tp":2,"batch_size":1,
            "max_num_seqs":1,"max_model_len":16384,"max_pixels":200704,"image_limit":16,"video_limit":0,
            "gpu_memory_utilization":.88,"structured_backend":"xgrammar","disable_any_whitespace":False,
            "constraint":"exactly two distinct ascending model-selected indices from actual supplied frame pairs",
            "event_id_uniqueness_and_140word_limit_still_checked_by_original_validator":True,
            "new_source_identity":self_identity,"original_code_identity":code,"model_identity_sha256":model["sha256"],
            "window_schema_sha256":[batch.digest(c["schema"]) for c in cases],
            "original_request_sha256":[c["original_files_sha256"]["request.json"] for c in cases]}
        params_sha = batch.digest(params)
        prepared = [prepare_case(case,out,model,params_sha) for case in cases]
        batch.verify_current(model,model=True)
        batch.verify_current(code,rehash=True)
        need(batch.sha256(Path(__file__)) == self_identity["sha256"], "new runner changed before loading")
        for case in cases:
            verify_original_unchanged(case)
        os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",VLLM_NO_USAGE_STATS="1",DO_NOT_TRACK="1")
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        from vllm.config import StructuredOutputsConfig
        from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar
        sampling = [SamplingParams(temperature=0.0,seed=0,max_tokens=1200,
            structured_outputs=StructuredOutputsParams(json=case["schema"])) for case in cases]
        for sp in sampling:
            validate_xgrammar_grammar(sp)
        versions = {name:importlib.metadata.version(name) for name in ("vllm","xgrammar","llguidance")}
        need(versions == {"vllm":"0.31.0","xgrammar":"0.2.7","llguidance":"1.7.6"}, "structured backend versions differ from the CPU-verified trial")
        batch.atomic_write(out/"run.receipt.json",batch.json_bytes({"schema":SCHEMA,"params":params,"params_sha256":params_sha,
            "model_identity":model,"model_reference_path":str(Path(args.model_reference).resolve()),
            "model_reference_sha256":batch.sha256(Path(args.model_reference)),"versions":versions,
            "original_cases":[{"bundle":c["bundle"],"files_sha256":c["original_files_sha256"]} for c in cases],
            "local_only":True,"new_protocol_not_free_decoding_resume":True,"one_trial_only":True,
            "semantic_approval":False,"human_review_completed":False,"formal_training":False,"full_pool_updated":False}))
        llm = LLM(model=model["root"],tensor_parallel_size=2,max_model_len=16384,max_num_seqs=1,
            gpu_memory_utilization=.88,limit_mm_per_prompt={"image":16,"video":0},
            mm_processor_kwargs={"max_pixels":200704},enable_prefix_caching=True,trust_remote_code=False,seed=0,
            structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"))
        backend = llm.llm_engine.vllm_config.structured_outputs_config.backend
        need(backend == "xgrammar", "requested explicit xgrammar was not the actual engine backend")
        batch.atomic_write(out/"load.receipt.json",batch.json_bytes({"model_identity_sha256":model["sha256"],
            "actual_structured_backend":backend,"fallback_allowed":False,"versions":versions,"params_sha256":params_sha,
            "current_model_inventory_and_stat_verified_before_load":True,"current_code_sha_verified_before_load":True}))
        results = []
        for case,(destination,lineage),sp in zip(cases,prepared,sampling):
            verify_original_unchanged(case)
            try:
                outputs = llm.chat([case["request"]["messages"]],sp,use_tqdm=False)
                need(len(outputs)==1 and len(outputs[0].outputs)==1, "expected exactly one new completion for this one trial")
                output,completion = outputs[0],outputs[0].outputs[0]
                tokens = list(completion.token_ids)
                eos = llm.get_tokenizer().eos_token_id
                generation = {"finish_reason":completion.finish_reason,"stop_reason":completion.stop_reason,
                    "prompt_tokens":len(output.prompt_token_ids or []),"completion_tokens":len(tokens),"max_tokens":1200,
                    "eos_token_id":eos,"last_output_token_id":tokens[-1] if tokens else None,
                    "last_output_token_is_eos":bool(tokens) and tokens[-1]==eos,
                    "eos_is_not_required_in_returned_token_ids":True,"actual_engine_backend":backend,"versions":versions}
                result = finish_response(destination,case,lineage,completion.text.encode("utf-8"),generation)
            except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
                result = {"schema":SCHEMA,"id":lineage["id"],"window":lineage["window"],"status":"failed",
                    "error":f"local constrained trial failed: {type(error).__name__}: {error}","full_pool_updated":False,
                    "original_failure_preserved":True,"semantic_approval":False,"human_review_completed":False,"formal_training":False}
                batch.atomic_write(destination/"failure.json",batch.json_bytes(result))
            results.append({**result,"case_out":str(destination)})
            print(json.dumps({"id":result["id"],"window":result["window"],"status":result["status"],"error":result["error"]}),flush=True)
        for case in cases:
            verify_original_unchanged(case)
        counts = {name:sum(r["status"]==name for r in results) for name in ("candidate","abstain","failed")}
        summary = {"schema":SCHEMA,"selected_cases":2,"processed_cases":len(results),"case_status_counts":counts,
            "nonempty_candidate_window_count":counts["candidate"],"params_sha256":params_sha,"results":results,
            "original_all_files_unchanged":True,"original_failures_not_overwritten":True,"full_pool_updated":False,
            "all_new_windows_terminal":len(results)==2,"all_new_windows_format_valid":counts["failed"]==0,
            "schema_is_not_semantic_approval":True,"semantic_approval":False,"human_review_completed":False,"formal_training":False}
        batch.atomic_write(out/"summary.json",batch.json_bytes(summary))
        print(json.dumps(summary),flush=True)
        return 1 if counts["failed"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
        print(json.dumps({"status":"failed","error":f"{type(error).__name__}: {error}",
            "model_or_trial_completed":False,"formal_training":False,"full_pool_updated":False}),file=sys.stderr,flush=True)
        sys.exit(1)
