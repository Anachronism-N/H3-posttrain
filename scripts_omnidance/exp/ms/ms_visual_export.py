"""Read-only verification and terminal merge of local visual candidate shards.

Absent resource exit markers or producer terminal indexes mean pending, never
complete. Output is an unreviewed weak-label pool, not a training admission.
Original per-window raw responses, frames and receipts are never rewritten.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

# The existing producer uses the same explicit, append-only CPU dependency path.
_cpu_site = os.environ.get("MS_VISUAL_CPU_SITEPATH")
if _cpu_site:
    if not Path(_cpu_site).is_absolute() or not Path(_cpu_site).is_dir():
        raise ValueError("MS_VISUAL_CPU_SITEPATH must be an existing absolute directory")
    sys.path.append(_cpu_site)
import ms_visual_batch as batch

SCHEMA = "motionscript_visual_terminal_export_v1"
TERMINAL = ("candidate", "abstain", "failed")
PLAN_SCHEMA = "choreography_video_first_12000_32b_launch_v1"
FLAGS = {
    "scene_or_camera_lexical": r"\b(?:room|studio|stage|store|aisle|shelf|shelves|background|camera|wall|scenery|environment|audience|kitchen|street|outdoor|indoor)\b",
    "identity_clothing_hair_prop_lexical": r"\b(?:woman|man|girl|boy|female|male|young|bald|shirt|dress|jacket|skirt|clothing|wearing|outfit|hair|hat|shoes|pants|handbag|bag|bottle|phone|object)\b",
    "contact_claim_lexical": r"\b(?:touch(?:es|ing)?|contact|grasp(?:s|ing)?|grip(?:s|ping)?|grab(?:s|bing)?|holding)\b",
    "anatomical_side_claim_lexical": r"\b(?:left|right)\s+(?:hand|arm|leg|foot|knee|fist|elbow|shoulder|side)\b",
    "vague_time_claim_lexical": r"\b(?:around the middle|at the midpoint|halfway|toward the end|throughout|always|entire sequence)\b",
    "music_or_genre_lexical": r"\b(?:music|song|beat|bpm|tempo|hip[- ]hop|jazz|ballet|k[- ]?pop|genre)\b",
}


class Pending(Exception):
    pass


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def same_json(first,second):
    return canonical_json(first) == canonical_json(second)


def load(path):
    return batch.strict_json(Path(path).read_text(encoding="utf-8"))


def beneath(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    need(path.is_relative_to(root), f"artifact escapes allowed output tree: {path}")
    return path


def sha_file(path, expected):
    need(batch.sha256(Path(path)) == expected, f"artifact content SHA mismatch: {path}")


def false_approval(value):
    for key in ("semantic_approval", "human_review_completed", "training_entry_updated", "formal_training"):
        need(value.get(key, False) is False, f"unexpected approval claim: {key}")


def identity_digest(identity):
    content = {name: {"path": row["path"], "sha256": row["sha256"]}
               for name, row in identity["files"].items()}
    return batch.digest({"root": identity["root"], "files": content})


def content_inventory(identity):
    root = Path(identity["root"])
    need(root.is_absolute(), "model/source identity root must be explicit absolute path")
    result = {}
    for name, value in identity["files"].items():
        relative = Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts, "unsafe model inventory filename")
        path = Path(value["path"])
        need(path.is_absolute() and path == root / relative, "model file path differs from explicit identity root")
        digest = value["sha256"]
        need(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest), "invalid model file SHA")
        size = value["stat"]["size"]
        need(type(size) is int and size > 0, "invalid model inventory size")
        result[name] = {"sha256": digest, "bytes": size}
    need(identity_digest(identity) == identity["sha256"], "identity digest does not match its resolved path and file SHAs")
    return result


def verify_model(identity, reference, explicit_root):
    need(identity["root"] == explicit_root, "loaded model is not the job's explicit model path")
    actual, expected = content_inventory(identity), content_inventory(reference)
    need(actual == expected, "loaded model content inventory differs from reference; basename is not identity")
    return batch.digest(expected)


def argv_values(job):
    argv = job["argv"]
    need(isinstance(argv, list) and all(isinstance(x, str) for x in argv), "job argv must be a string list")
    values = {}
    for index, value in enumerate(argv):
        if value.startswith("--"):
            need(value not in values and index + 1 < len(argv), "duplicate or incomplete CLI flag")
            values[value] = argv[index + 1]
    return values


def read_plan(path):
    plan = load(path)
    need(plan.get("schema") == PLAN_SCHEMA, "unknown visual launch schema")
    need(plan["target_ids"] == 12000 and plan["shards"] == 8, "full launch must explicitly select 12000 IDs across 8 fixed shards")
    false_approval(plan)
    need(plan.get("weak_supervision_only") is True and plan.get("formal_approval") is False, "launch is not explicitly unreviewed weak supervision")
    jobs = plan["jobs"]
    need(len(jobs) == 8 and all(type(j["shard"]) is int for j in jobs) and sorted(j["shard"] for j in jobs) == list(range(8)), "missing or duplicate planned shards")
    need(len({j["out"] for j in jobs}) == len({j["resource_queue"] for j in jobs}) == 8, "shards must have distinct output/control trees")
    rows, manifest_sha = batch.read_manifest(plan["input_manifest"])
    need(manifest_sha == plan["input_manifest_sha256"], "input manifest changed after launch")
    need(len(rows) == 12000 and all(r["split"] == "train" for r in rows), "input union is not the expected 12000 explicitly declared train IDs")
    for job in jobs:
        options = argv_values(job)
        expected = {"--stage": "run", "--manifest": plan["input_manifest"], "--out": job["out"],
            "--shard": str(job["shard"]), "--nshard": "8", "--limit": "0", "--tp": "2",
            "--batch-size": "2", "--max-num-seqs": "2", "--prepare-workers": "2",
            "--max-pixels": "200704", "--gpu-memory-utilization": "0.88"}
        need(all(options.get(k) == v for k,v in expected.items()), "job argv differs from fixed launch protocol")
        need(type(job["expected_ids"]) is int and job["expected_ids"] == len(rows[job["shard"]::8]) == 1500, "job ID partition count mismatch")
        need(plan["producer"] in job["argv"], "job runs a different producer")
        need(job["env"]["CUDA_VISIBLE_DEVICES"] == ",".join(map(str,job["gpus"])), "CUDA assignment differs from plan")
        need(Path(job["out"]).is_absolute() and Path(job["resource_queue"]).is_absolute(), "outputs must be explicit absolute paths")
    plan["_verified_plan_sha256"] = batch.sha256(Path(path))
    return plan, rows


def control_path(job, nshard):
    return Path(job["out"]) / "shards" / f"shard-{job['shard']:05d}-of-{nshard:05d}"


def terminal_snapshot(plan):
    missing = []
    for job in plan["jobs"]:
        control = control_path(job, plan["shards"])
        for path in (Path(job["resource_queue"]) / "worker_exit.json", control / "summary.json", control / "index.jsonl", control / "run.receipt.json"):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise Pending("missing real terminal artifacts: " + ", ".join(missing))


def verify_terminal(job, plan):
    path = Path(job["resource_queue"]) / "worker_exit.json"
    value = load(path)
    need(type(value.get("exit_code")) is int and value["exit_code"] in (0,1), "resource worker has no ordinary 0/1 terminal exit")
    need(value.get("cancelled", False) is False, "cancelled worker is not a completed candidate shard")
    need(value.get("source_unchanged") is True and value.get("input_unchanged") is True, "terminal receipt reports changed source/input")
    false_approval(value)
    return value


def verify_admission(job, plan, selected, reference):
    path = Path(job["resource_queue"])/"admission.json"
    admitted = load(path)
    need(admitted["plan_sha256"] == plan["_verified_plan_sha256"] and same_json(admitted["job"],job), "resource admission is not this frozen job/plan")
    need(admitted["input_sha256"] == plan["input_manifest_sha256"] and admitted["producer_sha256"] == plan["producer_sha256"], "resource admission source/input mismatch")
    need(admitted["model_identity_sha256"] == reference["sha256"], "resource admitted a different model identity")
    need(admitted["selected_ids"] == [row["id"] for row in selected], "resource admitted different SID order or shard")
    lineage = {"admission_path":str(path), "admission_sha256":batch.sha256(path)}
    if job["node"] != 1:
        stage_path = Path(plan["model_stage_receipts_directory"])/f"N{job['node']}_v1.json"
        stage = load(stage_path)
        need(stage["complete"] is True and stage["node"] == job["node"], "model replica stage is not complete for the planned node")
        need(stage["root"] == argv_values(job)["--model"] and stage["model_identity_sha256"] == reference["sha256"], "stage root/reference identity mismatch")
        expected = content_inventory(reference)
        need(set(stage["files"]) == set(expected), "stage file inventory differs from full reference")
        for name, value in stage["files"].items():
            need(value["path"] == str(Path(stage["root"])/name) and value["sha256"] == expected[name]["sha256"] and value["size"] == expected[name]["bytes"], "staged file content/path/size mismatch")
        lineage.update(stage_receipt_path=str(stage_path),stage_receipt_sha256=batch.sha256(stage_path))
    return lineage


def verify_run(job, plan, reference, code):
    control = control_path(job, plan["shards"])
    run = load(control / "run.receipt.json")
    need(run["schema"] == batch.SCHEMA and run["stage"] == "run", "not a real producer run receipt")
    need(run["manifest_path"] == plan["input_manifest"] and run["manifest_sha256"] == plan["input_manifest_sha256"], "producer input binding mismatch")
    need((run["shard"],run["nshard"],run["limit"],run["selected_ids"]) == (job["shard"],8,0,1500), "producer run selected another partition")
    need(run["code_identity"]["root"] == code["root"] and run["code_identity"]["sha256"] == code["sha256"] and content_inventory(run["code_identity"]) == content_inventory(code), "producer code dependency content identity differs from frozen current code")
    model_root = argv_values(job)["--model"]
    common_content = verify_model(run["model_identity"], reference, model_root)
    params = run["params_identity"]
    need(batch.digest(params["values"]) == params["sha256"], "params SHA mismatch")
    expected = {"chunk_seconds":2.0,"min_tail_seconds":1.5,"sample_fps":4.0,"max_frames":16,
        "tensor_parallel_size":2,"batch_size":2,"max_num_seqs":2,"max_model_len":16384,
        "max_pixels":200704,"gpu_memory_utilization":0.88,"temperature":0.0,"max_tokens":1200,
        "seed":0,"trust_remote_code":False,"image_limit":16,"model_path":model_root,
        "code_sha256":code["sha256"],"system_sha256":hashlib.sha256(batch.SYSTEM.encode()).hexdigest(),
        "observe_text_sha256":hashlib.sha256(batch.OBSERVE_TEXT.encode()).hexdigest()}
    need(all(same_json(params["values"].get(k),v) for k,v in expected.items()), "producer parameters differ from approved fixed protocol")
    options = argv_values(job)
    for name in ("ffmpeg", "ffprobe"):
        need(params["values"][name] == str(Path(options["--" + name]).resolve()), "media executable resolved path binding changed")
    load_path = control / "load.receipt.json"
    if load_path.is_file():
        loaded = load(load_path)
        need(same_json(loaded["model_identity"],run["model_identity"]) and same_json(loaded["code_identity"],run["code_identity"]), "load receipt identity differs from run")
        for key in ("model_content_sha_verified_at_startup", "current_model_inventory_and_stat_verified_before_load", "current_code_sha_verified_before_load"):
            need(loaded.get(key) is True, "load receipt lacks verified pre-load identity")
        need(loaded["transport"] == "local_vllm_LLM.chat" and loaded["model_download_allowed"] is False and loaded["third_party_endpoint_used"] is False, "not a local offline VLM load")
    false_approval(run)
    return run, common_content, load_path.is_file()


def risk_flags(value):
    if not isinstance(value,dict) or not isinstance(value.get("motion_caption"),str) or not isinstance(value.get("events"),list):
        return {}
    valid_events = [e for e in value["events"] if isinstance(e,dict) and isinstance(e.get("description"),str)]
    text = value["motion_caption"] + " " + " ".join(e["description"] for e in valid_events)
    result = {name: sorted({m.group(0) for m in re.finditer(pattern,text,re.I)})
              for name,pattern in FLAGS.items() if re.search(pattern,text,re.I)}
    index_lists = [e.get("frame_indices") for e in valid_events]
    minima = [min(indices) for indices in index_lists if isinstance(indices,list) and indices and all(type(i) is int for i in indices)]
    if minima != sorted(minima):
        result["event_first_evidence_not_chronological"] = minima
    return result


def verify_request(bundle, manifest, params):
    request = load(bundle / "request.json")
    frames = manifest["frames"]
    indices = [f["index"] for f in frames]
    content = [{"type":"text", "text":batch.OBSERVE_TEXT + "Allowed Source frame indices for this window: " + json.dumps(indices) + ". Use only these actual indices; never fill in missing evidence."}]
    for frame in frames:
        image = beneath(bundle / frame["file"], bundle)
        content.extend([{"type":"text","text":f"Source frame {frame['index']}, time {frame['time']:.6f} seconds."},
            {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(image.read_bytes()).decode("ascii")}}])
    expected = {"model":params["values"]["model_path"],"messages":[{"role":"system","content":batch.SYSTEM},{"role":"user","content":content}],"temperature":0.0,"max_tokens":1200}
    need(same_json(request,expected), "request contains changed schema, foreign captions/music, wrong image order or different model parameters")


def verify_window(bundle, binding, source, row, probe, params, model_loaded):
    receipt = batch.resume_receipt(bundle,binding,source)
    need(receipt is not None and receipt["status"] in TERMINAL, "window has no real terminal receipt")
    false_approval(receipt)
    need(same_json(receipt["binding"],binding), "terminal window receipt typed binding differs")
    manifest_path = bundle / "manifest.json"
    manifest = batch.verify_evidence(bundle,binding,source) if manifest_path.is_file() else None
    window = binding["window"]
    if manifest:
        need(same_json(manifest["binding"],binding), "window manifest typed binding differs")
        need(same_json(manifest["input"],row) and manifest["batch_schema"] == batch.SCHEMA, "window source input changed")
        need(manifest["media_end"] == probe["media_end"], "window media_end differs from actual video end")
        selected = batch.select_frames(probe["times"],window["start"],window["end"])
        need(same_json([f["index"] for f in manifest["frames"]],selected), "supplied global frame IDs differ from real ffprobe selection")
        for frame in manifest["frames"]:
            i = frame["index"]
            need(same_json(frame["time"],probe["times"][i]) and same_json(frame["source_timestamp"],probe["source_timestamps"][i]), "supplied frame times differ from actual PTS")
        verify_request(bundle,manifest,params)
    original = None
    generation = None
    raw_path = bundle / "response.raw.txt"
    if raw_path.is_file():
        need("response.raw.txt" in receipt["files_sha256"], "raw response not bound to window receipt")
        try:
            original = batch.parse_response(batch.json_bytes(raw_path.read_bytes().decode("utf-8")))
        except (ValueError,UnicodeDecodeError):
            need(receipt["status"] == "failed", "invalid raw response was counted as candidate")
    generation_path = bundle / "generation.json"
    if generation_path.is_file():
        need("generation.json" in receipt["files_sha256"] and model_loaded, "generation lacks receipt/load lineage")
        generation = load(generation_path)
        need(generation.get("model_call_performed") is True, "generation is not an actual recorded model call")
        need(generation.get("max_tokens") == 1200 and type(generation.get("completion_tokens")) is int and generation["completion_tokens"] >= 0 and type(generation.get("prompt_tokens")) is int and generation["prompt_tokens"] >= 0, "generation token budget/usage differs from fixed protocol")
        if raw_path.is_file():
            need(generation["raw_response_sha256"] == batch.sha256(raw_path), "generation/raw binding differs")
    candidate = None
    if receipt["status"] in ("candidate","abstain"):
        need(manifest is not None and generation is not None and original is not None, "candidate lacks real frames/raw/model generation")
        batch.validate_generation(bundle,required=True)
        candidate = load(bundle / "candidate.json")
        checked = batch.validate_candidate(original,manifest)
        checked.update(video_sha256=manifest["video_sha256"],request_sha256=manifest["request_sha256"],
            raw_response_sha256=batch.sha256(raw_path),binding=binding)
        need(same_json(candidate,checked), "candidate differs from typed original validated response")
        false_approval(candidate)
        need("candidate.json" in receipt["files_sha256"], "candidate not bound to terminal receipt")
    else:
        need(isinstance(receipt.get("error"),str) and receipt["error"].strip(), "failed window has no failure reason")
        need(not (bundle / "candidate.json").exists(), "failed window still has a candidate artifact; inspect rather than silently approve")
    return {"status":receipt["status"],"error":receipt["error"],"window":window,"bundle":str(bundle),
        "receipt_sha256":batch.sha256(bundle / "receipt.json"),"files_sha256":receipt["files_sha256"],
        "caption":candidate["motion_caption"] if candidate and not candidate["abstain"] else "",
        "events":candidate["events"] if candidate else [],"generation":generation,
        "source_frames":manifest["frames"] if manifest else [],
        "raw_response_path":str(raw_path) if raw_path.exists() else None,
        "raw_lexical_risk_flags":risk_flags(original) if original else {},
        "flags_are_diagnostic_not_semantic_approval":True}


def verify_record(indexed,row,job,run,probe,source,model_loaded):
    root = Path(job["out"]).resolve()
    if "final_path" not in indexed:
        need(indexed["status"] == "failed", "missing final artifact not marked failed")
        failure_path = control_path(job,8) / "failures" / (row["id"]+".json")
        failure = load(failure_path)
        need(failure["id"] == row["id"] and same_json(failure["input"],row) and failure["status"] == "failed", "failed source receipt disagrees with original input")
        need(failure["error"] == indexed["error"] and isinstance(failure["error"],str), "row failure reason changed")
        need(failure["model_sha256"] == run["model_identity"]["sha256"] and failure["params_sha256"] == run["params_identity"]["sha256"], "row failure protocol identity differs")
        false_approval(failure)
        return {"id":row["id"],"status":"failed","error":failure["error"],"windows":[],
            "expected_windows":batch.fixed_windows(0,probe["media_end"],2,1.5),
            "expected_windows_have_no_terminal_receipt":True,"failure_receipt_path":str(failure_path),
            "failure_receipt_sha256":batch.sha256(failure_path),"source_identity":source}
    path = beneath(indexed["final_path"],root)
    sha_file(path,indexed["final_sha256"])
    final = load(path)
    false_approval(final)
    need(final["id"] == row["id"] and same_json(final["input"],row), "final ID/sourcegroup/split/original input differs")
    need(final["source_sha256"] == source["sha256"] and final["model_sha256"] == run["model_identity"]["sha256"] and final["params_sha256"] == run["params_identity"]["sha256"], "final source/model/params binding mismatch")
    expected = batch.fixed_windows(0,probe["media_end"],2,1.5)
    need(final["media_end"] == probe["media_end"] and len(final["segments"]) == len(expected) == 2, "source does not have the two real planned windows")
    windows = []
    for segment,window in zip(final["segments"],expected):
        need(same_json({"start":segment["start"],"end":segment["end"]},window),"window sequence or coverage changed")
        bundle = beneath(segment["bundle"],path.parent)
        binding = batch.binding_for(row,source,run["model_identity"],run["params_identity"],window)
        window_root = path.parent / f"w{len(windows):04d}"
        need(bundle == window_root or bundle.parent == window_root and re.fullmatch(r"attempt-[0-9]{6}",bundle.name),"unexpected window/attempt path")
        attempts = sorted(p for p in window_root.glob("attempt-*") if p.is_dir() and re.fullmatch(r"attempt-[0-9]{6}",p.name))
        need(bundle == (attempts[-1] if attempts else window_root),"final points to an obsolete retry attempt")
        need(len(attempts) <= 2 and [p.name for p in attempts] == [f"attempt-{n:06d}" for n in range(1,len(attempts)+1)], "retry exceeds bounded two additional attempts or attempt lineage has gaps")
        checked = verify_window(bundle,binding,source,row,probe,run["params_identity"],model_loaded)
        need(segment["status"] == checked["status"] and segment["receipt_sha256"] == checked["receipt_sha256"],"final window status/receipt SHA differs")
        need(segment["caption"] == checked["caption"] and same_json(segment["events"],checked["events"]),"final caption differs from original validated response")
        checked["prior_attempts"] = []
        for older in [window_root]+attempts:
            if older == bundle:
                break
            previous = batch.resume_receipt(older,binding,source)
            need(previous and previous["status"] == "failed", "retry overwrote or skipped a valid prior observation")
            checked["prior_attempts"].append({"path":str(older),"receipt_sha256":batch.sha256(older/"receipt.json"),"status":"failed"})
        windows.append(checked)
    counts = Counter(w["status"] for w in windows)
    expected_counts = {name:counts[name] for name in (*TERMINAL,"prepared")}
    need(same_json(final["counts"],expected_counts) and same_json(indexed["counts"],expected_counts),"final/index counts do not match actual windows")
    status = "failed" if counts["failed"] else "candidate" if counts["candidate"] else "abstain"
    need(final["status"] == indexed["status"] == status,"final/index status does not match actual observations")
    complete = counts["failed"] == 0
    need(final["observation_complete_without_failures"] is complete and indexed["observation_complete_without_failures"] is complete,"false observation completeness")
    timed = "\n".join(f"[{w['window']['start']:.3f}-{w['window']['end']:.3f}s] {w['caption']}" for w in windows if w["caption"])
    need(final["timed_caption"] == timed and final["whole_clip_summary_generated"] is False,"timed caption was rewritten or gaps filled")
    return {"id":row["id"],"status":status,"timed_caption":timed,"windows":windows,
        "final_path":str(path),"final_sha256":indexed["final_sha256"],"source_identity":source,
        "model_instance_identity_sha256":run["model_identity"]["sha256"],"params_sha256":run["params_identity"]["sha256"]}


def verify_all(plan,rows,reference,code,ffprobe,progress=None):
    terminal_snapshot(plan)
    records = {}
    lineage = []
    for job in sorted(plan["jobs"],key=lambda j:j["shard"]):
        terminal = verify_terminal(job,plan)
        run,content_sha,loaded = verify_run(job,plan,reference,code)
        control = control_path(job,8)
        summary = load(control/"summary.json")
        false_approval(summary)
        expected_rows = rows[job["shard"]::8]
        admission_lineage = verify_admission(job,plan,expected_rows,reference)
        need(terminal["expected_ids"] == 1500 and terminal["summary_path"] == str(control/"summary.json") and same_json(terminal["summary"],summary), "resource terminal summary does not match producer output")
        index = [batch.strict_json(line) for line in (control/"index.jsonl").read_text().splitlines() if line.strip()]
        need([r["id"] for r in index] == [r["id"] for r in expected_rows],"missing, duplicate, wrong-shard or reordered SID index")
        need(summary["selected_ids"] == summary["processed_ids"] == len(expected_rows),"partial shard summary cannot be exported")
        sid_counts = Counter()
        for row,indexed in zip(expected_rows,index):
            need(row["id"] not in records,"duplicate SID across shard outputs")
            source = batch.source_identity(row)
            probe = batch.probe_video(row["video_path"],ffprobe)
            record = verify_record(indexed,row,job,run,probe,source,loaded)
            record.update(source_group=row["source_group"],split=row["split"],original_input=row,original_input_row_sha256=row["input_row_sha256"],shard=job["shard"],host=job["host"],model_path=run["model_identity"]["root"],model_content_inventory_sha256=content_sha,
                source="independent_video_open_vocabulary_unreviewed",semantic_approval=False,human_review_completed=False,formal_training=False,ground_truth=False)
            records[row["id"]] = record
            sid_counts[record["status"]] += 1
            if progress:
                progress(len(records),row["id"])
        need(same_json(summary["counts"],{k:sid_counts[k] for k in (*TERMINAL,"prepared")}),"shard summary counts differ from verified index")
        success = sid_counts["failed"] == 0
        need(summary["all_selected_ids_observed_without_failures"] is success,"shard wrongly claims no failures")
        need(terminal["exit_code"] == (0 if success else 1),"resource exit disagrees with producer failure accounting")
        lineage.append({**admission_lineage,"shard":job["shard"],"host":job["host"],"out":job["out"],"resource_queue":job["resource_queue"],
            "summary_sha256":batch.sha256(control/"summary.json"),"index_sha256":batch.sha256(control/"index.jsonl"),
            "run_receipt_sha256":batch.sha256(control/"run.receipt.json"),"worker_exit_sha256":batch.sha256(Path(job["resource_queue"])/"worker_exit.json"),
            "model_identity":run["model_identity"],"model_content_inventory_sha256":content_sha,
            "params_identity":run["params_identity"],"resource_worker_exit":terminal})
    need(set(records) == {r["id"] for r in rows} and len(records)==12000,"export union is not exactly the expected 12000 SIDs")
    return [records[r["id"]] for r in rows],lineage


def caption_availability(records):
    full = sum(len(r["windows"]) == 2 and all(w["status"] == "candidate" and isinstance(w["caption"],str) and bool(w["caption"].strip()) for w in r["windows"]) for r in records)
    nonempty = sum(any(w["status"] == "candidate" and isinstance(w["caption"],str) and bool(w["caption"].strip()) for w in r["windows"]) for r in records)
    window_counts = Counter(w["status"] for r in records for w in r["windows"])
    return {"usable_full_caption_count":full,"incomplete_caption_count":len(records)-full,
        "empty_caption_count":len(records)-nonempty,"caption_delivery_complete":full == len(records),
        "clip_denominator":len(records),"window_denominator":2*len(records),
        "partial_nonempty_caption_count":nonempty-full,
        "any_nonempty_caption_count":nonempty,"sources_without_nonempty_caption":len(records)-nonempty,
        "actual_candidate_window_count":window_counts["candidate"],"actual_abstain_window_count":window_counts["abstain"],
        "actual_failed_window_count":window_counts["failed"],
        "source_error_missing_window_count":sum(len(r.get("expected_windows",[])) for r in records if not r["windows"]),
        "actual_terminal_window_receipt_count":sum(len(r["windows"]) for r in records),
        "goal_full_caption_coverage_satisfied":full == len(records),
        "complete_caption_definition":"exactly two candidate windows with nonempty original captions; candidate+abstain is partial, not full",
        "usable_caption_is_not_semantic_approval":True,"max_additional_failed_attempts":2}


def output_bundle(out,records,manifest):
    out = Path(out).resolve()
    if out.exists():
        need((out/"manifest.json").is_file(),"output exists without complete manifest; choose a new destination")
        previous = load(out/"manifest.json")
        need(same_json(previous,manifest),"existing export differs from rebuilt verified content")
        sha_file(out/"captions.jsonl",manifest["captions_sha256"])
        return "verified_existing"
    out.mkdir(parents=True,exist_ok=False)
    payload = b"".join((json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n").encode() for row in records)
    need(hashlib.sha256(payload).hexdigest() == manifest["captions_sha256"],"export payload digest changed")
    batch.atomic_write(out/"captions.jsonl",payload)
    batch.atomic_write(out/"manifest.json",batch.json_bytes(manifest))
    return "created"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--ffprobe",default="/opt/ms_tools/ffprobe")
    args = ap.parse_args(argv)
    plan_path = Path(args.plan).resolve()
    plan,rows = read_plan(plan_path)
    try:
        terminal_snapshot(plan)
    except Pending as error:
        print(json.dumps({"status":"pending","reason":str(error),"export_written":False,"formal_training":False}),flush=True)
        return 2
    import shutil
    need(shutil.which(args.ffprobe) is not None,"installed ffprobe unavailable for final source-frame verification")
    sha_file(plan["producer"],plan["producer_sha256"])
    code = batch.code_identity()
    reference_path = Path(plan["model_stage_receipts_directory"])/"source_reference_v1.json"
    reference = load(reference_path)["model_identity"]
    need(reference["sha256"] == plan["expected_model_identity_sha256"] and len(reference["files"]) == 25,"reference model is not the frozen 25-file model")
    content_inventory(reference)
    out = Path(args.out).resolve()
    for job in plan["jobs"]:
        need(not batch.overlaps(out,Path(job["out"])) and not batch.overlaps(out,Path(job["resource_queue"])),"export must be outside producer/resource output trees")
    need(out != plan_path and not plan_path.is_relative_to(out),"export must not overwrite launch plan")
    with ExitStack() as locks:
        for job in sorted(plan["jobs"],key=lambda j:j["shard"]):
            for name in (".lock",):
                path = control_path(job,8)/name
                need(path.is_file(),"producer control lock file missing")
                lock = locks.enter_context(path.open("rb"))
                try:
                    fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise Pending("producer is still active: "+str(path)) from error
        records,lineage = verify_all(plan,rows,reference,code,args.ffprobe,
            progress=lambda count,sid:print(json.dumps({"verified_sources":count,"id":sid}),flush=True) if count%100==0 else None)
        sid_counts = Counter(r["status"] for r in records)
        window_counts = Counter(w["status"] for r in records for w in r["windows"])
        flag_counts = Counter(key for r in records for w in r["windows"] for key in w["raw_lexical_risk_flags"])
        payload = b"".join((json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n").encode() for row in records)
        manifest = {**caption_availability(records),"schema":SCHEMA,"status":"terminal_verified_unreviewed","expected_ids":12000,"verified_ids":len(records),
            "all_planned_workers_terminal":True,"all_expected_ids_accounted_for":True,"expected_window_count":24000,
            "all_ids_have_two_terminal_windows":all(len(r["windows"])==2 for r in records),
            "sources_without_terminal_window_receipts":sum(not r["windows"] for r in records),
            "sid_status_counts":{k:sid_counts[k] for k in TERMINAL},"window_status_counts":{k:window_counts[k] for k in TERMINAL},
            "all_observed_without_failures":sid_counts["failed"]==0,
            "semantic_approval":False,"human_review_completed":False,"formal_training":False,"ground_truth":False,
            "captions_sha256":hashlib.sha256(payload).hexdigest(),"launch_plan_path":str(plan_path),"launch_plan_sha256":batch.sha256(plan_path),
            "input_manifest_sha256":plan["input_manifest_sha256"],"producer_sha256":plan["producer_sha256"],
            "exporter_sha256":batch.sha256(Path(__file__)),"model_reference_receipt_path":str(reference_path),
            "model_reference_receipt_sha256":batch.sha256(reference_path),"model_content_inventory_sha256":batch.digest(content_inventory(reference)),
            "lineage":lineage,"raw_lexical_flag_window_counts":dict(flag_counts),"flag_patterns":FLAGS,
            "flags_are_diagnostic_not_semantic_verdicts":True,"original_raw_or_caption_facts_modified":False,
            "formal_training_not_updated":True,"first50_not_silently_reused":True}
        disposition = output_bundle(out,records,manifest)
        print(json.dumps({"status":manifest["status"],"out":str(out),"disposition":disposition,"ids":len(records),"formal_training":False}),flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Pending as error:
        print(json.dumps({"status":"pending","reason":str(error),"export_written":False,"formal_training":False}),flush=True)
        sys.exit(2)
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps({"status":"rejected","reason":str(error),"export_written":False,"formal_training":False}),file=sys.stderr)
        sys.exit(1)
