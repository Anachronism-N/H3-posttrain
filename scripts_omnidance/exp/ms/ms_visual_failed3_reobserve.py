"""One new free-decoding observation of three explicitly pinned failed windows.

No previous caption is supplied to the model. Reuse the original RGB evidence,
retain its failures, and append only a generic clarification of the existing
motion/evidence/abstention contract. This trial never changes the full pool.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import ms_visual_batch as batch

SCHEMA = "motionscript_visual_failed3_reobserve_v1"
PRODUCER_SHA = "b362dc5275b19c70a6f4f4d149b974e75571ad07e7e4f1f0a2251222fd839204"
MODEL_SHA = "26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c"
SPECS = [
    {"id":"ydBjlNRKChxUjCzl","shard":2,"window":{"start":0.0,"end":2.0},"failure_class":"frame_support",
     "error":"event evidence must name at least two distinct supplied frames",
     "request_sha256":"b9790c11cd6704e466c8621c3b0f3e87b642310cbf16d9a79b4ce8d7916da54e",
     "raw_sha256":"05ef0a35196158ef344b1c25f9b2efdce0c3b585167441a3fd83ecec4cf77343",
     "generation_sha256":"25eb14007cd514c685aa32025084bd0ef629b2291ae852a8e23827b9ba419dfb"},
    {"id":"nVtCIoarVOUShrzG","shard":3,"window":{"start":0.0,"end":2.0},"failure_class":"word_limit",
     "error":"caption exceeds 140 words; do not pad detailed rule clauses",
     "request_sha256":"70f3007e6ba9dd0e76b7586f827616b9967737e7d52fbb1b32d03e117805fcad",
     "raw_sha256":"da6784ca37490100f3b5e25faafd537abaf53bdd82a91870ae208a859d3a4edf",
     "generation_sha256":"dcc991c22d0e2d00320779aa602810349ed3cf40af0d304dc8568cc2eab985bb"},
    {"id":"MtxhlICkhdLzjAlQ","shard":7,"window":{"start":0.0,"end":2.0},"failure_class":"abstention_contract",
     "error":"abstention must have empty caption and events",
     "request_sha256":"e4157059354da3848303ab4647cdab746f5c4a112e7931f7a05e36ff8e176a93",
     "raw_sha256":"e9b8ee8fcd55b1c11ea78b3728366e9c894ef423d97236d220b9af47059f313f",
     "generation_sha256":"9c834fb89f4e49469953d40fc9b7f35390b15a49225be79fef81885deeb00428"},
]
CLARIFICATION = """Re-observe the supplied RGB frames, without inventing facts to fill a caption.
Keep motion_caption concise: aim for at most 90 English words; the hard limit
remains 140 words. Describe supported movement phases chronologically, not a
list of isolated single-frame poses. Each event must cite at least two distinct
actually supplied Source frame indices that genuinely support the stated
movement or transition. Merely adding a second index does not support a pose
visible in only one frame. Do not describe unsupported instantaneous details
in motion_caption or events; state unresolved ambiguity in uncertainties.
A held-pose phase may be described when it is visibly supported within a real
movement sequence. A purely static sequence with no supported movement must
return abstain=true, motion_caption="", events=[]; do not pad it with invented
movement or a static-pose caption. Never force a movement to avoid abstention.
Event IDs must be unique nonempty JSON strings. Do not add scene, identity,
clothing, props, music, anatomical side guesses, or unobserved contacts.
Return only the same four-key JSON object requested by the system.
"""


def need(condition,message):
    if not condition:
        raise ValueError(message)


def load(path):
    return batch.strict_json(Path(path).read_text(encoding="utf-8"))


def equal(a,b):
    return json.dumps(a,sort_keys=True,separators=(",",":"),allow_nan=False) == json.dumps(b,sort_keys=True,separators=(",",":"),allow_nan=False)


def new_request(original):
    need(original["messages"][0] == {"role":"system","content":batch.SYSTEM}, "do not replace the original system contract")
    need(len(original["messages"])==2 and original["messages"][1]["role"]=="user"
         and isinstance(original["messages"][1]["content"],list), "unexpected original message format")
    result = copy.deepcopy(original)
    result["messages"][1]["content"].append({"type":"text","text":CLARIFICATION})
    return result


def unchanged(case):
    old = Path(case["bundle"])
    for name,digest in case["original_files_sha256"].items():
        need(batch.sha256(old/name)==digest,"original file changed: "+name)
    need(batch.sha256(Path(case["original_run_receipt_path"]))==case["original_run_receipt_sha256"], "original run receipt changed")
    for frame in case["manifest"]["frames"]:
        path = (old/frame["file"]).resolve()
        need(path.is_relative_to(old) and batch.sha256(path)==frame["sha256"], "original frame changed")
    batch.verify_source(case["source"])


def inspect_case(full,spec,ffprobe):
    control = Path(full)/f"shard{spec['shard']}"/"shards"/f"shard-{spec['shard']:05d}-of-00008"
    indexed = load(control/"rows"/(spec["id"]+".json"))
    need(indexed["id"]==spec["id"] and indexed["status"]=="failed", "only the selected actual failed SID is allowed")
    final_path = Path(indexed["final_path"]).resolve()
    allowed = (Path(full)/f"shard{spec['shard']}"/"ids"/spec["id"]).resolve()
    need(final_path.is_relative_to(allowed) and batch.sha256(final_path)==indexed["final_sha256"], "original final artifact changed or outside its SID")
    final = load(final_path)
    matching = [w for w in final["segments"] if equal({"start":w["start"],"end":w["end"]},spec["window"])]
    need(len(matching)==1 and matching[0]["status"]=="failed", "selected original window is not failed")
    bundle = Path(matching[0]["bundle"]).resolve()
    need(bundle.is_relative_to(allowed), "original bundle outside selected source")
    manifest = load(bundle/"manifest.json")
    row = manifest["input"]
    need(row["id"]==spec["id"] and equal(manifest["window"],spec["window"]), "wrong source/window")
    run_path = control/"run.receipt.json"
    run = load(run_path)
    need(run["code_identity"]["files"]["ms_visual_batch.py"]["sha256"]==PRODUCER_SHA, "original producer changed")
    source = batch.source_identity(row)
    binding = manifest["binding"]
    need(binding["source_sha256"]==source["sha256"] and binding["model_sha256"]==run["model_identity"]["sha256"]==MODEL_SHA
         and binding["params_sha256"]==run["params_identity"]["sha256"], "original source/model/params identity changed")
    batch.verify_evidence(bundle,binding,source)
    receipt = batch.resume_receipt(bundle,binding,source)
    need(receipt and receipt["status"]=="failed" and receipt["error"]==spec["error"], "original failure class changed")
    need(batch.sha256(bundle/"request.json")==spec["request_sha256"] and batch.sha256(bundle/"response.raw.txt")==spec["raw_sha256"]
         and batch.sha256(bundle/"generation.json")==spec["generation_sha256"], "pinned original request/raw/generation changed")
    batch.validate_generation(bundle,required=True)
    probe = batch.probe_video(row["video_path"],ffprobe)
    selected = batch.select_frames(probe["times"],spec["window"]["start"],spec["window"]["end"])
    need(manifest["media_end"]==probe["media_end"] and [f["index"] for f in manifest["frames"]]==selected, "original sampling changed")
    for frame in manifest["frames"]:
        i=frame["index"]
        need(frame["time"]==probe["times"][i] and frame["source_timestamp"]==probe["source_timestamps"][i], "original PTS changed")
    request = load(bundle/"request.json")
    need(request["temperature"]==0.0 and request["max_tokens"]==1200
         and request["messages"][1]["content"][0]["text"].startswith(batch.OBSERVE_TEXT), "original request protocol changed")
    hashes = {name:batch.sha256(bundle/name) for name in ("manifest.json","manifest.receipt.json","request.json","response.raw.txt","generation.json","receipt.json")}
    # Old caption bytes are hashed and retained, but never decoded into a new prompt.
    return {"spec":spec,"bundle":str(bundle),"manifest":manifest,"source":source,"original_request":request,
        "new_request":new_request(request),"original_files_sha256":hashes,"original_run_receipt_path":str(run_path),
        "original_run_receipt_sha256":batch.sha256(run_path)}


def prepare(case,out,params_sha):
    dest = Path(out)/"cases"/(case["spec"]["id"]+"_w0000")
    dest.mkdir(parents=True,exist_ok=False)
    batch.atomic_write(dest/"request.original.json",(Path(case["bundle"])/"request.json").read_bytes())
    batch.atomic_write(dest/"request.json",batch.json_bytes(case["new_request"]))
    frames = [{**f,"original_absolute_path":str((Path(case["bundle"])/f["file"]).resolve())} for f in case["manifest"]["frames"]]
    lineage = {"schema":SCHEMA,"id":case["spec"]["id"],"window":case["spec"]["window"],"failure_class":case["spec"]["failure_class"],
        "original_failure_reason":case["spec"]["error"],"original_bundle":case["bundle"],
        "original_files_sha256":case["original_files_sha256"],"original_run_receipt_path":case["original_run_receipt_path"],
        "original_run_receipt_sha256":case["original_run_receipt_sha256"],"original_input":case["manifest"]["input"],
        "source_identity":case["source"],"original_frames":frames,"new_request_sha256":batch.sha256(dest/"request.json"),
        "clarification_sha256":hashlib.sha256(CLARIFICATION.encode()).hexdigest(),"new_params_sha256":params_sha,
        "old_caption_supplied_to_model":False,"same_source_rgb_frames_no_resampling":True,
        "same_system_and_validator":True,"structured_pair_enum_used":False,"one_trial_only":True,
        "new_protocol_not_free_producer_resume":True,"full_pool_updated":False,"semantic_approval":False,"human_review_completed":False,"formal_training":False}
    batch.atomic_write(dest/"lineage.json",batch.json_bytes(lineage))
    return dest,lineage


def finish(dest,case,lineage,raw,generation):
    dest = Path(dest)
    need(not (dest/"response.raw.txt").exists() and not (dest/"receipt.json").exists(), "one trial only; never overwrite an existing response")
    generation = {**generation,"model_call_performed":True,"raw_response_sha256":hashlib.sha256(raw).hexdigest(),
        "new_request_sha256":lineage["new_request_sha256"],"decoding_protocol":SCHEMA,"structured_output_constraint":None}
    batch.atomic_write(dest/"generation.json",batch.json_bytes(generation))
    batch.atomic_write(dest/"response.raw.txt",raw)
    status,error = "failed",None
    try:
        need(generation["finish_reason"]=="stop" and type(generation["completion_tokens"]) is int
             and 0 <= generation["completion_tokens"] < 1200 and generation["max_tokens"]==1200, "truncated or abnormally terminated completion")
        unchanged(case)
        value = batch.parse_response(batch.json_bytes(raw.decode("utf-8")))
        candidate = batch.validate_candidate(value,case["manifest"])
        candidate.update(decoding_protocol=SCHEMA,new_params_sha256=lineage["new_params_sha256"],
            new_request_sha256=lineage["new_request_sha256"],raw_response_sha256=hashlib.sha256(raw).hexdigest(),
            lineage_sha256=batch.sha256(dest/"lineage.json"),original_bundle=case["bundle"],
            full_pool_updated=False,formal_training=False,goal_caption_words=90)
        batch.atomic_write(dest/"candidate.json",batch.json_bytes(candidate))
        status = "abstain" if candidate["abstain"] else "candidate"
    except (ValueError,OSError,KeyError,TypeError,UnicodeDecodeError) as failure:
        error = str(failure)
    result = {"schema":SCHEMA,"id":lineage["id"],"window":lineage["window"],"failure_class":lineage["failure_class"],"status":status,"error":error,
        "files_sha256":{name:batch.sha256(dest/name) for name in ("request.original.json","request.json","lineage.json","response.raw.txt","generation.json","candidate.json") if (dest/name).is_file()},
        "original_failures_preserved":True,"no_response_cleaning_or_frame_completion":True,
        "full_pool_updated":False,"semantic_approval":False,"human_review_completed":False,"formal_training":False}
    batch.atomic_write(dest/"receipt.json",batch.json_bytes(result))
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-root",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--model-reference",required=True)
    ap.add_argument("--ffprobe",default="/opt/ms_tools/ffprobe")
    ap.add_argument("--tp",type=int,default=2)
    ap.add_argument("--max-num-seqs",type=int,default=1)
    ap.add_argument("--max-pixels",type=int,default=200704)
    ap.add_argument("--gpu-memory-utilization",type=float,default=.88)
    args=ap.parse_args(argv)
    need(args.tp==2 and args.max_num_seqs==1 and args.max_pixels==200704 and args.gpu_memory_utilization==.88,
        "authorized trial is exactly three windows, TP2/batch1/seq1/fixed pixels and memory fraction")
    full,out=Path(args.full_root).resolve(strict=True),Path(args.out).resolve()
    need(Path(args.model).is_absolute() and not batch.overlaps(out,full) and not batch.overlaps(out,Path(args.model))
         and not Path(args.model_reference).resolve().is_relative_to(out), "trial output must be independent of original pool/model/reference")
    need(not (out/"run.receipt.json").exists() and not (out/"cases").exists(), "trial output already used; no automatic retry or overwrite")
    need(batch.sha256(Path(batch.__file__))==PRODUCER_SHA, "frozen original producer changed")
    with batch.bundle_lock(out):
        cases=[inspect_case(full,spec,args.ffprobe) for spec in SPECS]
        model=batch.model_identity(args.model)
        reference=load(args.model_reference)["model_identity"]
        need(model["sha256"]==reference["sha256"]==MODEL_SHA and len(model["files"])==25
             and set(model["files"])==set(reference["files"]), "different local model identity")
        for name,row in model["files"].items():
            need(row["sha256"]==reference["files"][name]["sha256"] and row["stat"]["size"]==reference["files"][name]["stat"]["size"], "local model file changed")
        need(all(c["new_request"]["model"]==model["root"] for c in cases), "original request model path differs; do not rewrite")
        code=batch.code_identity();self_identity=batch.file_identity(Path(__file__))
        params={"schema":SCHEMA,"new_source_identity":self_identity,"original_code_identity":code,
            "model_identity_sha256":model["sha256"],"temperature":0.0,"seed":0,"max_tokens":1200,
            "tp":2,"batch_size":1,"max_num_seqs":1,"max_model_len":16384,"image_limit":16,"video_limit":0,
            "max_pixels":200704,"gpu_memory_utilization":.88,"structured_outputs":None,
            "clarification_sha256":hashlib.sha256(CLARIFICATION.encode()).hexdigest(),"target_caption_words":90,"original_validator_hard_word_limit":140,
            "old_requests_sha256":[c["spec"]["request_sha256"] for c in cases],
            "new_requests_sha256":[hashlib.sha256(batch.json_bytes(c["new_request"])).hexdigest() for c in cases]}
        params_sha=batch.digest(params)
        prepared=[prepare(case,out,params_sha) for case in cases]
        batch.verify_current(model,model=True);batch.verify_current(code,rehash=True)
        need(batch.sha256(Path(__file__))==self_identity["sha256"], "new runner changed before model loading")
        for case in cases:unchanged(case)
        os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",VLLM_NO_USAGE_STATS="1",DO_NOT_TRACK="1")
        from vllm import LLM,SamplingParams
        versions={name:importlib.metadata.version(name) for name in ("vllm","torch","transformers")}
        need(versions["vllm"]=="0.31.0", "runtime vLLM differs from verified local environment")
        batch.atomic_write(out/"run.receipt.json",batch.json_bytes({"schema":SCHEMA,"params":params,"params_sha256":params_sha,
            "model_identity":model,"model_reference_path":str(Path(args.model_reference).resolve()),"model_reference_sha256":batch.sha256(Path(args.model_reference)),
            "versions":versions,"case_specs":SPECS,"full_pool_updated":False,"semantic_approval":False,"human_review_completed":False,"formal_training":False}))
        llm=LLM(model=model["root"],tensor_parallel_size=2,max_model_len=16384,max_num_seqs=1,gpu_memory_utilization=.88,
            limit_mm_per_prompt={"image":16,"video":0},mm_processor_kwargs={"max_pixels":200704},enable_prefix_caching=True,trust_remote_code=False,seed=0)
        sampling=SamplingParams(temperature=0.0,seed=0,max_tokens=1200)
        need(sampling.structured_outputs is None, "no structured pair forcing in this free re-observation trial")
        batch.atomic_write(out/"load.receipt.json",batch.json_bytes({"model_identity_sha256":model["sha256"],"params_sha256":params_sha,
            "actual_decoding_mode":"free_json_no_structured_constraint","structured_outputs":None,"versions":versions,
            "current_model_inventory_and_stat_verified_before_load":True,"current_code_sha_verified_before_load":True}))
        results=[]
        for case,(dest,lineage) in zip(cases,prepared):
            unchanged(case)
            try:
                outputs=llm.chat([case["new_request"]["messages"]],sampling,use_tqdm=False)
                need(len(outputs)==1 and len(outputs[0].outputs)==1, "expected one actual completion per authorized window")
                output,completion=outputs[0],outputs[0].outputs[0];tokens=list(completion.token_ids);eos=llm.get_tokenizer().eos_token_id
                generation={"finish_reason":completion.finish_reason,"stop_reason":completion.stop_reason,
                    "prompt_tokens":len(output.prompt_token_ids or []),"completion_tokens":len(tokens),"max_tokens":1200,
                    "eos_token_id":eos,"last_output_token_id":tokens[-1] if tokens else None,
                    "last_output_token_is_eos":bool(tokens) and tokens[-1]==eos,"eos_is_not_required_in_returned_token_ids":True,"versions":versions}
                result=finish(dest,case,lineage,completion.text.encode("utf-8"),generation)
            except (ValueError,OSError,KeyError,TypeError,RuntimeError) as failure:
                result={"schema":SCHEMA,"id":case["spec"]["id"],"window":case["spec"]["window"],"failure_class":case["spec"]["failure_class"],
                    "status":"failed","error":f"local re-observation failed: {type(failure).__name__}: {failure}",
                    "original_failures_preserved":True,"full_pool_updated":False,"semantic_approval":False,"human_review_completed":False,"formal_training":False}
                batch.atomic_write(dest/"failure.json",batch.json_bytes(result))
            results.append({**result,"case_out":str(dest)})
            print(json.dumps({"id":result["id"],"status":result["status"],"error":result["error"]}),flush=True)
        for case in cases:unchanged(case)
        counts={status:sum(r["status"]==status for r in results) for status in ("candidate","abstain","failed")}
        summary={"schema":SCHEMA,"selected_cases":3,"processed_cases":len(results),"case_status_counts":counts,
            "nonempty_candidate_window_count":counts["candidate"],"results":results,"params_sha256":params_sha,
            "one_trial_completed":len(results)==3,"original_all_files_and_frames_unchanged":True,"old_failures_not_replaced":True,
            "abstention_is_valid_but_does_not_fill_caption_delivery":True,"full_pool_updated":False,"semantic_approval":False,"human_review_completed":False,"formal_training":False}
        batch.atomic_write(out/"summary.json",batch.json_bytes(summary));print(json.dumps(summary),flush=True)
        return 1 if counts["failed"] else 0


if __name__=="__main__":
    try:sys.exit(main())
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
        print(json.dumps({"status":"failed","error":f"{type(error).__name__}: {error}","model_or_trial_completed":False,"full_pool_updated":False,"formal_training":False}),file=sys.stderr,flush=True)
        sys.exit(1)
