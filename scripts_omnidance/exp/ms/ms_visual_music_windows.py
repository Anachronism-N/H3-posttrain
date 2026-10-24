"""Independent video-only VLM consumer of already executed shared music windows.

Only the first three authority sources are observed. The shared window JSON is
consumed unchanged; music/beat/BPM labels and MS captions never enter the VLM.
Short windows with fewer than two actual source frames are evidence abstentions,
not invented VLM completions. No fixed-window or full-pool producer is modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import numpy as np

import ms_visual_batch as batch
import ms_music_windows as music
import verify_music_window_pts_and_types as exact_pts

SCHEMA = "motionscript_visual_shared_music_windows_v1"
IDS = ["hwrlCvOCYUKpljlV","LglrebUOmlpWTFrr","qKgmlBaDdOeqvUZZ"]
PRODUCER_SHA = "b362dc5275b19c70a6f4f4d149b974e75571ad07e7e4f1f0a2251222fd839204"
MODEL_SHA = "26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c"
PTS_HELPER_SHA = "26091e34109c0ee0f10ca1d55885c2c98bddcd67e3f63a2ee41e006af23cbd24"


def need(condition,message):
    if not condition:
        raise ValueError(message)


def load(path):
    return batch.strict_json(Path(path).read_text(encoding="utf-8"))


def equal(a,b):
    return json.dumps(a,sort_keys=True,separators=(",",":"),allow_nan=False) == json.dumps(b,sort_keys=True,separators=(",",":"),allow_nan=False)


def read_shared(path,expected_sha):
    path=Path(path).resolve(strict=True);raw=path.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==expected_sha,"shared JSONL changed")
    summary_path=path.with_name("summary.json");summary_bytes=summary_path.read_bytes();summary=load(summary_path)
    receipt_path=path.with_name("summary.receipt.json");receipt=load(receipt_path)
    need(receipt["summary_sha256"]==hashlib.sha256(summary_bytes).hexdigest() and receipt["window_manifest_sha256"]==expected_sha and summary["window_manifest_sha256"]==expected_sha,"shared producer summary receipt does not bind these windows")
    need(type(summary["processed_count"]) is int and summary["processed_count"]>=3 and summary["complete"] is True and type(summary["failed_count"]) is int and summary["failed_count"]==0 and not summary["errors"],"shared MS window producer is not terminal without errors")
    rows=[batch.strict_json(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    need(len({r["id"] for r in rows})==len(rows),"duplicate shared plan ID")
    need([r["id"] for r in rows[:3]]==IDS,"shared plans must select the same first three authority sources, not heldout or reordered IDs")
    records={r["id"]:r for r in summary["records"]}
    for plan in rows[:3]:
        rec=records[plan["id"]];single=path.parent/rec["window_file"]
        need(single.resolve().is_relative_to(path.parent) and batch.sha256(single)==rec["window_sha256"],"per-source shared window receipt changed")
        need(equal(load(single),plan),"JSONL plan differs from source's exact window artifact")
        need(plan["schema"]==music.SCHEMA and plan["status"] in ("ready","insufficient_downbeats"),"unknown shared window status")
        music.validate_shared_plan(plan,plan["binding"])
        exact_pts.validate_plan(plan,plan["binding"])
        need(plan["semantic_approval"] is False and plan["human_review_completed"] is False,"shared plan cannot grant semantic approval")
    return rows[:3],{"path":str(path),"sha256":expected_sha,"summary_path":str(summary_path),"summary_sha256":hashlib.sha256(summary_bytes).hexdigest(),"summary_receipt_sha256":batch.sha256(receipt_path)}


def verify_pts_receipt(path,expected_sha,strategies,manifest_sha):
    path=Path(path).resolve(strict=True);receipt=load(path)
    need(batch.sha256(path)==expected_sha and receipt["schema"]=="music_window_exact_pts_typed_verification_v1" and receipt["complete"] is True, "independent exact-PTS/type verification is missing or changed")
    need(receipt["input_manifest_sha256"]==manifest_sha and receipt["actual_video_pts_probed_not_inferred_from_nlf"] is True, "independent receipt did not verify this actual video PTS/input")
    need(receipt["old_reports_and_captions_modified"] is False and receipt["MS_or_model_inference_repeated"] is False and receipt["GPU_used"] is False, "unexpected independent PTS verifier execution claims")
    need(receipt["source_sha256"]==batch.sha256(Path(music.__file__)) and receipt["verifier_sha256"]==PTS_HELPER_SHA==batch.sha256(Path(exact_pts.__file__)) and receipt["verifier_sha256"]==batch.sha256(Path(receipt["verifier_path"])),  "PTS verification helper/verifier source changed")
    for plans,ref in strategies:
        root=str(Path(ref["path"]).parent)
        need(receipt["bundle_counts"][root]["summary_sha256"]==ref["summary_sha256"], "independent PTS verification bound a different strategy summary")
        for plan in plans:
            policy=plan["params"].get("policy","bpm")
            wanted=str(Path(root)/(plan["id"]+".windows.json"))
            records=[r for r in receipt["records"] if r["id"]==plan["id"] and r["policy"]==policy and r["plan"]["path"]==wanted]
            need(len(records)==1 and records[0]["exact_pts_digest_equal_original"] is True and records[0]["canonical_typed_plan_verified"] is True, "selected source/strategy has no independent exact PTS/type receipt")
            need(records[0]["plan"]["sha256"]==batch.sha256(Path(wanted)) and records[0]["actual_video_pts_sha256"]==plan["binding"]["actual_video_pts_sha256"], "independent PTS/plan content binding differs")
    return {"path":str(path),"sha256":expected_sha,"schema":receipt["schema"],"actual_video_pts_probed_not_inferred_from_nlf":True,"formal_training":False}


def actual_binding(row,source,probe):
    with np.load(row["nlf_path"],allow_pickle=False) as archive:
        fps=float(archive["fps"])
        count=len(archive["pose"])
        timestamps=np.asarray(archive["timestamps"],dtype=float).tolist() if "timestamps" in archive else (np.arange(count)/fps).tolist()
    need(len(probe["times"])==count and fps>0 and np.isfinite(timestamps).all(),"actual video and NLF temporal counts differ")
    need(equal(timestamps,probe["times"]),"NLF time grid does not equal actual source-video PTS; no timeline guessed")
    return {"id":row["id"],"input_sha256":source["files"]["nlf_path"]["sha256"],"video_sha256":source["files"]["video_path"]["sha256"],
        "audio_sha256":source["files"]["audio_path"]["sha256"],"actual_video_pts_sha256":hashlib.sha256(batch.json_bytes(probe["times"])).hexdigest(),
        "media_start":0.0,"media_end":probe["media_end"],"video_pts_offset":probe["time_origin_source_timestamp"],"video_fps":fps,"decoded_video_frames":count}


def select_policy(times,window):
    actual=[i for i,t in enumerate(times) if window["start"]<=t<window["end"]]
    if len(actual)<2:
        return actual,"abstain_insufficient_source_frames"
    return batch.select_frames(times,window["start"],window["end"],4.,16),"prepared"


def prepare_source(plan,row,source,probe,ref,root,params,model,args):
    expected=actual_binding(row,source,probe)
    need(equal(expected,plan["binding"]),"shared plan binding differs from actual video/audio/NLF/PTS")
    music.validate_shared_plan(plan,expected)
    exact_pts.validate_plan(plan,expected)
    for key in ("id","video_path","audio_path","nlf_path","source_group","split"):
        need(plan["manifest_record"][key]==row[key],"shared plan source context differs from selected authority input")
    tag = "bpm" if plan["strategy"]=="bpm_duration_grid_not_detected_beats" else "bar_"+plan["params"]["policy"]
    clip=Path(root)/tag/row["id"]
    clip.mkdir(parents=True,exist_ok=True)
    plan_sha=batch.digest(plan)
    shared_path=clip/"shared_plan.json"
    if shared_path.exists():need(shared_path.read_bytes()==batch.json_bytes(plan),"shared plan changed after preparation")
    else:batch.atomic_write(shared_path,batch.json_bytes(plan))
    record={"id":row["id"],"tag":tag,"shared_plan":plan,"shared_jsonl_reference":ref,"shared_plan_sha256":plan_sha,"source_identity":source,
        "raw_windows":plan["raw_windows"],"windows":[],"plan_status":plan["status"],"source_bound_actual_pts_verified":True,
        "BPM_or_downbeat_values_supplied_to_model":False,"MS_action_captions_supplied_to_model":False,
        "audio_video_alignment_verified":False,"motions_on_beats_verified":False,"semantic_approval":False,"human_review_completed":False,"formal_training":False}
    if plan["status"]!="ready":
        record["status"]="pending_insufficient_downbeats";return record,[]
    work=[]
    for i,window in enumerate(plan["windows"]):
        selected,status=select_policy(probe["times"],window)
        bundle=clip/f"w{i:04d}"
        binding=batch.binding_for(row,source,model,params,window)
        binding.update(shared_plan_sha256=plan_sha,shared_jsonl_sha256=ref["sha256"],decoding_protocol=SCHEMA)
        bundle.mkdir(exist_ok=True)
        if status=="abstain_insufficient_source_frames":
            value={"schema":SCHEMA,"id":row["id"],"window":window,"binding":binding,"status":"abstain", "abstain":True,
                "motion_caption":"","events":[],"reason":"fewer_than_two_actual_decoded_video_frames_inside_unchanged_half_open_window",
                "actual_frame_indices":selected,"actual_frame_times":[probe["times"][j] for j in selected],
                "model_call_performed":False,"window_extended":False,"frame_evidence_filled":False,"semantic_approval":False,"human_review_completed":False}
            path=bundle/"insufficient_support.receipt.json"
            if path.exists():need(equal(load(path),value),"short-window abstention was altered")
            else:batch.atomic_write(path,batch.json_bytes(value))
            record["windows"].append({"window_index":i,"window":window,"status":"abstain","caption":"","events":[],"bundle":str(bundle),
                "model_call_performed":False,"support_reason":value["reason"],"support_receipt_sha256":batch.sha256(path)})
            continue
        if (bundle/"manifest.json").exists():
            manifest=batch.verify_evidence(bundle,binding,source)
        else:manifest=batch.prepare_window(bundle,row,source,probe,window,binding,args)
        need([f["index"] for f in manifest["frames"]]==selected,"shared-window actual sampling changed")
        receipt=batch.resume_receipt(bundle,binding,source)
        if receipt is None and (bundle/"response.raw.txt").exists():receipt=batch.finish_response(bundle,binding,manifest,(bundle/"response.raw.txt").read_bytes())
        if receipt is None:batch.write_receipt(bundle,binding,"prepared")
        item={"bundle":str(bundle),"window_root":str(bundle),"binding":binding,"receipt":receipt}
        view={"window_index":i,"window":window,"status":receipt["status"] if receipt else "prepared","caption":"","events":[],"bundle":str(bundle),"model_call_performed":receipt is not None and "generation.json" in receipt["files_sha256"]}
        record["windows"].append(view);work.append((item,view,source))
    record["status"]="prepared"
    return record,work


def summarize(records):
    windows=[w for r in records for w in r["windows"]]
    counts=Counter(w["status"] for w in windows)
    pending_plans=sum(r["plan_status"]!="ready" for r in records)
    return {"schema":SCHEMA,"source_count":len(IDS),"strategy_source_count":len(records),"window_count":len(windows),
        "window_status_counts":{k:counts[k] for k in ("candidate","abstain","failed","prepared")},
        "model_executed_window_count":sum(w["model_call_performed"] for w in windows),
        "insufficient_frame_abstention_count":sum(w.get("support_reason") is not None for w in windows),
        "pending_insufficient_downbeat_plan_count":pending_plans,
        "all_ready_windows_terminal":counts["prepared"]==0,"all_plans_executed":pending_plans==0 and counts["prepared"]==0,
        "all_ready_windows_without_failures":counts["prepared"]==0 and counts["failed"]==0,
        "usable_complete_strategy_source_count":sum(r["plan_status"]=="ready" and bool(r["windows"]) and all(w["status"]=="candidate" and w["caption"].strip() for w in r["windows"]) for r in records),
        "VLM_shared_manifest_actual_execution":sum(w["model_call_performed"] for w in windows)>0,
        "not_fixed_window_baseline_reused":True,"audio_video_alignment_verified":False,"motions_on_beats_verified":False,
        "semantic_approval":False,"human_review_completed":False,"formal_training":False,"full_pool_updated":False,"records":records}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows",action="append",required=True)
    ap.add_argument("--windows-sha",action="append",required=True)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--manifest-sha",required=True)
    ap.add_argument("--pts-verification",required=True)
    ap.add_argument("--pts-verification-sha",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--model-reference",required=True)
    ap.add_argument("--stage",choices=("prepare","run"),default="run")
    ap.add_argument("--ffmpeg",default="/usr/local/bin/ffmpeg")
    ap.add_argument("--ffprobe",default="/opt/ms_tools/ffprobe")
    ap.add_argument("--tp",type=int,default=2)
    args=ap.parse_args(argv)
    need(len(args.windows)==len(args.windows_sha) and 1<=len(args.windows)<=3 and args.tp==2,"one to three explicit ready strategies, fixed TP2 only")
    rows,sha=batch.read_manifest(args.manifest,limit=3)
    need(sha==args.manifest_sha and [r["id"] for r in rows]==IDS,"manifest changed or wrong fixed three authority sources")
    need(batch.sha256(Path(batch.__file__))==PRODUCER_SHA,"do not modify original fixed-window producer")
    out=Path(args.out).resolve()
    for path in [args.manifest,args.model_reference,args.pts_verification,*args.windows]:
        need(not Path(path).resolve().is_relative_to(out),"out must not contain original input artifacts")
    need(not batch.overlaps(out,Path(args.model)),"out must remain outside the model directory")
    with batch.bundle_lock(out):
        sources={r["id"]:batch.source_identity(r) for r in rows}
        probes={r["id"]:batch.probe_video(r["video_path"],args.ffprobe) for r in rows}
        strategies=[read_shared(path,sha) for path,sha in zip(args.windows,args.windows_sha)]
        pts_reference=verify_pts_receipt(args.pts_verification,args.pts_verification_sha,strategies,sha)
        tags=["bpm" if plans[0]["strategy"]=="bpm_duration_grid_not_detected_beats" else "bar_"+plans[0]["params"]["policy"] for plans,ref in strategies]
        need(len(tags)==len(set(tags)),"duplicate strategy tag; no silently repeated MS/window source")
        model=batch.model_identity(args.model);reference=load(args.model_reference)["model_identity"]
        need(model["sha256"]==reference["sha256"]==MODEL_SHA and len(model["files"])==25,"different local 32B identity")
        need(all(model["files"].get(k,{}).get("sha256")==v["sha256"] for k,v in reference["files"].items()),"local model file inventory differs")
        own=batch.file_identity(Path(__file__));helper=batch.file_identity(Path(music.__file__));pts_helper=batch.file_identity(Path(exact_pts.__file__));code=batch.code_identity()
        need(pts_helper["sha256"]==PTS_HELPER_SHA,"frozen exact-PTS/type helper changed")
        values={"schema":SCHEMA,"new_source_sha256":own["sha256"],"music_window_helper_sha256":helper["sha256"],"original_code_sha256":code["sha256"],
            "model_identity_sha256":model["sha256"],"manifest_sha256":sha,"shared_jsonl_sha256":args.windows_sha,
            "independent_pts_verification_sha256":args.pts_verification_sha,"exact_pts_helper_sha256":pts_helper["sha256"],
            "sample_fps":4.0,"max_frames":16,"context":16384,"max_pixels":200704,"tp":2,"batch_size":1,"max_num_seqs":1,
            "temperature":0.0,"seed":0,"max_tokens":1200,"gpu_memory_utilization":.88,"system_sha256":hashlib.sha256(batch.SYSTEM.encode()).hexdigest()}
        params={"values":values,"sha256":batch.digest(values)}
        receipt={"schema":SCHEMA,"params":params,"own_source":own,"helper_source":helper,"exact_pts_helper_source":pts_helper,"original_code":code,"model_identity":model,
            "model_reference_sha256":batch.sha256(Path(args.model_reference)),"strategy_references":[ref for plans,ref in strategies],
            "selected_ids":IDS,"manifest_path":str(Path(args.manifest).resolve()),"manifest_sha256":sha,"independent_pts_verification":pts_reference,
            "semantic_approval":False,"human_review_completed":False,"formal_training":False}
        rp=out/"run.receipt.json"
        if rp.exists():need(equal(load(rp),receipt),"existing shared consumer output belongs to another source/model/strategy protocol")
        else:batch.atomic_write(rp,batch.json_bytes(receipt))
        records=[];work=[]
        byid={r["id"]:r for r in rows}
        for plans,ref in strategies:
            for plan in plans:
                record,items=prepare_source(plan,byid[plan["id"]],sources[plan["id"]],probes[plan["id"]],ref,out,params,model,args)
                records.append(record);work.extend(items)
        pending=[item for item in work if item[0]["receipt"] is None]
        if args.stage=="run" and pending:
            need(batch.sha256(Path(args.manifest))==sha,"original input manifest changed before model loading")
            for plans,ref in strategies:need(batch.sha256(Path(ref["path"]))==ref["sha256"],"shared window JSONL changed before model loading")
            batch.verify_current(model,model=True);batch.verify_current(code,rehash=True)
            need(batch.sha256(Path(__file__))==own["sha256"] and batch.sha256(Path(music.__file__))==helper["sha256"] and batch.sha256(Path(exact_pts.__file__))==pts_helper["sha256"],"consumer/helper changed before loading")
            os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",VLLM_NO_USAGE_STATS="1",DO_NOT_TRACK="1")
            from vllm import LLM,SamplingParams
            llm=LLM(model=model["root"],tensor_parallel_size=2,max_model_len=16384,max_num_seqs=1,gpu_memory_utilization=.88,
                limit_mm_per_prompt={"image":16,"video":0},mm_processor_kwargs={"max_pixels":200704},enable_prefix_caching=True,trust_remote_code=False,seed=0)
            sampling=SamplingParams(temperature=0.0,seed=0,max_tokens=1200)
            batch.atomic_write(out/"load.receipt.json",batch.json_bytes({"schema":SCHEMA,"model_identity_sha256":model["sha256"],"params_sha256":params["sha256"],
                "actual_decoding":"local_vllm_LLM.chat_free_json","vllm_version":importlib.metadata.version("vllm"),"one_model_load_for_all_ready_strategies":True,
                "current_model_and_code_sha_verified_before_load":True,"semantic_approval":False,"formal_training":False}))
            for item,view,source in pending:
                batch.observe_batch([item],llm,sampling,source)
        for item,view,source in work:
            receipt=item["receipt"];bundle=Path(item["bundle"])
            if receipt is not None:
                view.update(status=receipt["status"],error=receipt["error"],receipt_sha256=batch.sha256(bundle/"receipt.json"),
                    model_call_performed="generation.json" in receipt["files_sha256"])
                if receipt["status"]=="candidate":
                    candidate=load(bundle/"candidate.json");view.update(caption=candidate["motion_caption"],events=candidate["events"])
        need(batch.sha256(Path(exact_pts.__file__))==pts_helper["sha256"],"exact-PTS helper changed during model run")
        need(batch.sha256(Path(args.pts_verification))==args.pts_verification_sha,"independent PTS receipt changed during shared-window run")
        need(batch.sha256(Path(args.manifest))==sha,"original input manifest changed during shared-window run")
        for plans,ref in strategies:need(batch.sha256(Path(ref["path"]))==ref["sha256"],"shared JSONL changed during model run")
        for source in sources.values():batch.verify_source(source)
        for record in records:
            if record["plan_status"]=="ready":
                states={w["status"] for w in record["windows"]}
                record["status"]="failed" if "failed" in states else "prepared" if "prepared" in states else "candidate" if "candidate" in states else "abstain"
            record["timed_caption"]="\n".join(f"[{w['window']['start']:.6f},{w['window']['end']:.6f}) {w['caption']}" for w in record["windows"] if w["caption"])
        summary=summarize(records);summary["stage"]=args.stage
        batch.atomic_write(out/"summary.json",batch.json_bytes(summary))
        print(json.dumps({k:v for k,v in summary.items() if k!="records"}),flush=True)
        return 1 if summary["window_status_counts"]["failed"] else 2 if summary["pending_insufficient_downbeat_plan_count"] else 0


if __name__=="__main__":
    try:sys.exit(main())
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
        print(json.dumps({"status":"failed","error":f"{type(error).__name__}: {error}","VLM_shared_manifest_actual_execution":False,"formal_training":False}),file=sys.stderr,flush=True)
        sys.exit(1)
