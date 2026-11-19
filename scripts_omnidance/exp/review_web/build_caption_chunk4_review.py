"""Append receipt-checked chunk4 candidates to the existing read-only annotation view."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

EXPERIMENT = "caption_chunk4_experiment_20261024_v1"
SCHEMA = "annotation_compare_review_bundle_v1"
VALIDATOR_SOURCE = None
STRATEGIES = ("whole", "fixed2s", "bpm", "mergedbar")
LABELS = {"whole": "整段", "fixed2s": "固定2秒", "bpm": "BPM长度窗", "mergedbar": "小节merged"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), {"path": str(Path(path).resolve()), "sha256": digest(raw)}


def checked_json(path, receipt_path=None, digest_key="report_sha256"):
    record, ref = load_json(path)
    receipt_path = Path(receipt_path) if receipt_path else Path(path).with_suffix(".receipt.json")
    receipt, receipt_ref = load_json(receipt_path)
    if receipt.get(digest_key) != ref["sha256"]:
        raise ValueError(f"Receipt mismatch: {path}")
    ref["receipt"] = receipt_ref
    return record, ref


def append_cases(bundle, cases, tag=EXPERIMENT, run_version="v1"):
    if bundle.get("schema") != SCHEMA or not bundle.get("skeleton_cases"):
        raise ValueError("Existing annotation bundle schema/skeleton cases missing")
    if tag != EXPERIMENT:
        raise ValueError("Only this experiment tag may be appended")
    result = copy.deepcopy(bundle)
    by_id = {case["id"]: case for case in result["caption_cases"]}
    if len(by_id) != len(result["caption_cases"]):
        raise ValueError("Duplicate existing caption case IDs")
    seen = set()
    for incoming in cases:
        sid = incoming["id"]
        if sid in seen:
            raise ValueError("Duplicate incoming caption case IDs")
        seen.add(sid)
        views = incoming["views"]
        keys = [(v.get("strategy"), v.get("stage")) for v in views]
        if len(keys) != len(set(keys)) or any(v.get("experiment_tag") != tag or v.get("run_version", "v1") != run_version for v in views):
            raise ValueError("Duplicate view or foreign experiment tag")
        if not views:
            raise ValueError("Experiment views missing")
        if sid in by_id:
            current = by_id[sid]
            for name in ("video", "audio"):
                if Path(current[name]).resolve() != Path(incoming[name]).resolve():
                    raise ValueError(f"Same ID has different {name}: {sid}")
            current["views"] = [v for v in current["views"] if not (v.get("experiment_tag") == tag
                and v.get("run_version", "v1") == run_version)] + copy.deepcopy(views)
            refs = current.setdefault("experiment_refs", {})
            refs[tag if run_version == "v1" else tag + ":" + run_version] = copy.deepcopy(incoming["source_refs"])
        else:
            current = copy.deepcopy(incoming)
            current["audio_video_alignment_verified"] = False
            result["caption_cases"].append(current)
            by_id[sid] = current
    return result


def publish(path, expected_sha256, bundle):
    """Replace only a still-current snapshot; callers opt in after dry-run."""
    path = Path(path)
    lock_path = path.with_name(path.name + ".chunk4.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if digest(path.read_bytes()) != expected_sha256:
            raise ValueError("Annotation changed during build; rerun instead of overwriting")
        raw = (json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        if digest(raw) == expected_sha256:
            return False
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".chunk4.", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(raw)
                tmp.flush()
                os.fsync(tmp.fileno())
            if digest(path.read_bytes()) != expected_sha256:
                raise ValueError("Annotation changed before replacement; refusing stale write")
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
    return True


def inside_exp(path, exp_root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(exp_root).resolve()):
        raise ValueError("Caption body paths must remain inside EXP")
    if path.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("Private caption bodies cannot be written to the repository")
    return path


def object_digest(value):
    return digest((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def verify_file_refs(value):
    refs = []
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            _, ref = load_json(value["path"]) if str(value["path"]).endswith(".json") else (None, {
                "path": value["path"], "sha256": digest(Path(value["path"]).read_bytes())})
            if ref["sha256"] != value["sha256"]:
                raise ValueError("Original source/receipt SHA changed")
            refs.append(ref)
        for child in value.values():
            refs.extend(verify_file_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(verify_file_refs(child))
    return refs


def checked_original_row(row):
    report_ref = row["source_refs"]["report"]
    receipt_ref = row["source_refs"]["receipt"]
    report, ref = checked_json(report_ref["path"], receipt_ref["path"])
    if ref["sha256"] != report_ref["sha256"] or ref["receipt"]["sha256"] != receipt_ref["sha256"]:
        raise ValueError("Plan source report/receipt references differ")
    if report["id"] != row["id"]:
        raise ValueError("Source report ID differs")
    if row["strategy"] == "whole":
        originals = [report["A_whole_clip"]]
    elif row["strategy"] == "fixed2s":
        originals = report["B_fixed_windows"]
    else:
        originals = report["MS_windows"]
    if len(originals) != len(row["windows"]):
        raise ValueError("Selected plan window count differs from original report")
    for source, selected in zip(originals, row["windows"]):
        if any(selected.get(key) != value for key, value in source.items()):
            raise ValueError("Selected window changed original bounds/status/partial/raw caption")
    for key in ("video_sha256", "audio_sha256", "input_sha256"):
        expected = report.get("binding", report).get(key)
        if expected is not None and row["binding"].get(key) != expected:
            raise ValueError("Plan binding differs from original report")
    return ref


def import_validator():
    import sys
    ms = Path(VALIDATOR_SOURCE).resolve() if VALIDATOR_SOURCE else Path(__file__).resolve().parents[1] / "ms"
    if str(ms) not in sys.path:
        sys.path.insert(0, str(ms))
    import ms_caption_rewrite
    if Path(ms_caption_rewrite.__file__).resolve() != (ms / "ms_caption_rewrite.py").resolve():
        raise ValueError("Validator already imported from a different source snapshot")
    return ms_caption_rewrite


def checked_transaction(root, expected_binding, validator):
    """No receipt means pending, never borrowed from a previous run/prepare."""
    root = Path(root)
    if not (root / "receipt.json").is_file():
        return {"status": "pending", "reason": "latest run transaction has no terminal receipt"}, [], None
    receipt, receipt_ref = load_json(root / "receipt.json")
    binding = receipt["binding"]
    if any(binding.get(k) != v for k, v in expected_binding.items()):
        raise ValueError("Run receipt/source/plan/row binding mismatch")
    refs = [receipt_ref]
    for name, expected in receipt["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path.read_bytes()) != expected:
            raise ValueError("Transaction artifact SHA mismatch or escaped bundle")
        refs.append({"path": str(path), "sha256": expected})
    if receipt["files_sha256"].get("final.json") is None or receipt["files_sha256"].get("request.json") is None:
        raise ValueError("Terminal receipt lacks final/request hashes")
    final, _ = load_json(root / "final.json")
    request, _ = load_json(root / "request.json")
    if final["status"] != receipt["status"]:
        raise ValueError("Final status differs from receipt")
    if final["status"] == "candidate":
        required = ("binding.json", "response.raw.txt", "generation.json", "response.json")
        if any(name not in receipt["files_sha256"] for name in required):
            raise ValueError("Candidate lacks complete raw/generation receipt")
        saved_binding, _ = load_json(root / "binding.json")
        if saved_binding != binding:
            raise ValueError("Saved binding differs from terminal receipt")
        core = import_validator()
        generation = core.validate_generation(root, request)
        raw = (root / "response.raw.txt").read_text(encoding="utf-8")
        checked = validator(core.parse_response(json.dumps(raw).encode()))
        if checked != final["json"]:
            raise ValueError("Candidate differs from real raw response")
        final["review_evidence"] = {"generation": generation, "receipt": receipt_ref,
                                    "raw_response_path": str(root / "response.raw.txt")}
    elif final["status"] != "failed":
        raise ValueError("Unknown terminal transaction status")
    else:
        final["review_evidence"] = {"receipt": receipt_ref, "error": final.get("error"), "raw_response_path": None}
        if "response.raw.txt" in receipt["files_sha256"]:
            final["review_evidence"]["raw_response_path"] = str(root / "response.raw.txt")
        if "generation.json" in receipt["files_sha256"]:
            generation, _ = load_json(root / "generation.json")
            if (generation.get("raw_response_sha256") != receipt["files_sha256"].get("response.raw.txt")
                    or generation.get("request_sha256") != receipt["files_sha256"]["request.json"]):
                raise ValueError("Failed generation does not bind real raw/request hashes")
            final["review_evidence"]["generation"] = generation
    generation = final.get("review_evidence", {}).get("generation", {})
    final["actual_model_terminal"] = (generation.get("model_call_performed") is True
        and generation.get("finish_reason") == "stop"
        and type(generation.get("completion_tokens")) is int
        and type(generation.get("max_tokens")) is int
        and 0 < generation["completion_tokens"] < generation["max_tokens"]
        and "response.raw.txt" in receipt["files_sha256"])
    return final, refs, request


def original_review_windows(row):
    core = import_validator()
    windows = []
    for w in row["windows"]:
        start, end = core.window_bounds(w)
        original = w.get("original_ms_window", w)
        windows.append({"start": start, "end": end, "status": w["status"],
                        "caption": w["cleaned_motion_text"] if w["status"] == "candidate" else "",
                        "reason": w.get("reason", original.get("abstain_reason")),
                        "partial_caption": original.get("partial_supported_motion_text", w.get("partial_caption", "")),
                        "raw_original_MS": copy.deepcopy(original.get("original_captions", w.get("raw_original_MS", [])))})
    return windows


def stage_windows(row, result, supplied, boundaries):
    core = import_validator()
    output = result.get("json", {}).get("windows", [])
    if result["status"] == "candidate":
        core.check_rewrite(result["json"], row, supplied["facts"], boundaries)
    windows = []
    for i, w in enumerate(row["windows"]):
        start, end = core.window_bounds(w)
        stage = output[i] if result["status"] == "candidate" else {}
        evidence = {"original_source_status": w["status"], "original_abstain_reason": w.get("abstain_reason"),
                    "support_status": w.get("support_status"), "window_id": w["window_id"],
                    "used_fact_ids": stage.get("used_fact_ids", []),
                    "used_boundary_ids": stage.get("used_boundary_ids", []),
                    "issues": result.get("json", {}).get("issues", []),
                    "reused_from": result.get("reused_from"), "semantic_approval": False}
        evidence.update(result.get("review_evidence", {}))
        windows.append({"start": start, "end": end, "status": result["status"],
                        "caption": stage.get("caption", ""),
                        "reason": result.get("reason", result.get("error", "Unreviewed AI candidate; not semantic approval")),
                        "events": evidence})
    if result["status"] == "candidate":
        return [{"start": windows[0]["start"], "end": windows[-1]["end"], "status": "candidate",
                 "caption": result["review_full_text"],
                 "reason": ("真实保存全文；原非candidate窗=" + str(sum(w["status"] != "candidate" for w in row["windows"]))
                            + "，不是完整MS支持覆盖；visual_supplement仍为AI弱候选。时间头是原观察窗，不是精确动作边界"),
                 "events": {"per_window_outputs": windows, "semantic_approval": False}}]
    return windows


def checked_runtime(root, identity):
    path = Path(root) / "load.receipt.json"
    if not path.is_file():
        raise ValueError("Real candidate lacks local model runtime load receipt")
    receipt, ref = load_json(path)
    if (receipt.get("transport") != "local_vllm_LLM.chat" or receipt.get("tensor_parallel_size") != 2
            or receipt.get("third_party_endpoint_used") is not False
            or receipt.get("model_download_allowed") is not False):
        raise ValueError("Runtime receipt is not authorized local TP2 inference")
    for name in ("model_identity", "code_identity"):
        if receipt[name] != identity[name] or receipt[name]["sha256"] != identity[name.replace("_identity", "_sha256")]:
            raise ValueError("Runtime model/code identity differs from run binding")
    code = receipt["code_identity"]
    producer = code["files"].get("ms_caption_rewrite.py")
    if not producer or digest(Path(import_validator().__file__).read_bytes()) != producer["sha256"]:
        raise ValueError("Review validator differs from receipt-bound frozen producer")
    content = {name: {"path": file["path"], "sha256": file["sha256"]} for name, file in code["files"].items()}
    if object_digest({"root": code["root"], "files": content}) != code["sha256"]:
        raise ValueError("Frozen code tree identity digest differs")
    verify_file_refs(code["files"])
    return ref


def build_cases(plan_path, outputs, limit=3, run_version="v1"):
    core = import_validator()
    rows, plan_sha = core.read_plan(plan_path, limit=limit)
    roots = [Path(p).resolve() for p in (outputs if isinstance(outputs, (list, tuple)) else [outputs])]
    identities = []
    for root in roots:
        identity, identity_ref = load_json(root / "run.identity.json")
        if identity.get("plan_sha256") != plan_sha:
            raise ValueError("Output run used a different input plan SHA")
        if run_version == "v3":
            if (identity.get("generation_protocol") != "exact_windows_schema_v3"
                    or identity.get("fact_surface_policy") != "neutral_subject_no_synthetic_seconds_v1"
                    or identity.get("json_schemas_sha256") != core.digest(core.JSON_SCHEMAS)
                    or identity.get("rewrite_prompt_sha256") != digest(core.REWRITE_SYSTEM.encode())
                    or identity.get("boundary_prompt_sha256") != digest(core.BOUNDARY_SYSTEM.encode())):
                raise ValueError("v3 identity differs from frozen exact-window and surface protocol")
        identities.append((root, identity, identity_ref))
    grouped = {}
    for row in rows:
        sid, strategy = row["id"], row["strategy"]
        case = grouped.setdefault(sid, {"id": sid, "video": row["manifest_record"]["video_path"],
                                      "audio": row["manifest_record"]["audio_path"], "views": [], "source_refs": []})
        if any(row["manifest_record"][key] != case[target] for key, target in (("video_path", "video"), ("audio_path", "audio"))):
            raise ValueError("Four strategies do not use the same source media")
        case["source_refs"].append(checked_original_row(row))
        case["source_refs"].extend(verify_file_refs(row["source_refs"]))
        matches = [(root, identity, ref) for root, identity, ref in identities
                   if identity.get("strategy_filter") in (None, strategy)]
        if len(matches) != 1:
            raise ValueError("Supply exactly one explicit latest output root per strategy")
        out, identity, identity_ref = matches[0]
        root = out / "rows" / sid / strategy
        case["source_refs"].append(identity_ref)
        original, original_ref = load_json(root / "input.original.json") if (root / "input.original.json").exists() else (row, None)
        if original != row:
            raise ValueError("Output original windows/source differ from selected plan")
        if original_ref:
            case["source_refs"].append(original_ref)
        base = {k: identity[k] for k in ("model_sha256", "code_sha256", "plan_sha256", "rewrite_prompt_sha256",
                                        "boundary_prompt_sha256", "rewrite_tokens", "supplement_empty", "max_pixels",
                                        "max_num_seqs", "gpu_memory_utilization")}
        for key in ("generation_protocol", "json_schemas_sha256", "fact_surface_policy"):
            if key in identity:
                base[key] = identity[key]
        base["row_sha256"] = object_digest(row)
        source = core.batch.source_identity(row["manifest_record"])
        base["source_sha256"] = source["sha256"]
        if row["binding"].get("video_sha256") != source["files"]["video_path"]["sha256"] or row["binding"].get("audio_sha256") != source["files"]["audio_path"]["sha256"]:
            raise ValueError("Original binding does not match real source media SHA")
        supplements, visual_windows = {}, []
        for w in row["windows"]:
            s_root = root / "visual_supplement" / w["window_id"]
            if not (s_root / "receipt.json").is_file():
                continue
            if not core.supplement_eligible(w):
                raise ValueError("Visual supplement attached to nonempty or partial MS")
            binding = {**base, "kind": "visual_supplement", "window": copy.deepcopy(w)}
            receipt = core.batch.resume_receipt(s_root, binding, source)
            if receipt is None:
                continue
            manifest = core.batch.verify_evidence(s_root, binding, source)
            candidate = None
            if receipt["status"] in ("candidate", "abstain"):
                candidate, _ = load_json(s_root / "candidate.json")
                supplements[w["window_id"]] = candidate
            start, end = core.window_bounds(w)
            visual_windows.append({"start": start, "end": end, "status": receipt["status"],
                                   "caption": candidate.get("motion_caption", "") if candidate else "",
                                   "reason": "visual_supplement; original MS remains no_output; AI weak candidate",
                                   "events": {"candidate": candidate, "frames": manifest["frames"],
                                              "receipt_path": str(s_root / "receipt.json")}})
            case["source_refs"].append(load_json(s_root / "receipt.json")[1])
        facts = core.facts_for(row, supplements)
        supplied = core.rewrite_input(row, facts)
        if run_version == "v3" and (root / "run.facts.original.json").exists():
            audit, audit_ref = load_json(root / "run.facts.original.json")
            if audit != facts or any("raw_text" in f for f in supplied["facts"]):
                raise ValueError("v3 raw-fact audit or conditioned neutral facts changed")
            case["source_refs"].append(audit_ref)
        if (root / "run.facts.S.json").exists():
            actual, ref = load_json(root / "run.facts.S.json")
            if actual != supplied:
                raise ValueError("Run S facts differ from original MS/receipt-checked visual evidence")
            case["source_refs"].append(ref)
        boundaries, boundary_audit = [], []
        for left, right in zip(row["windows"], row["windows"][1:]):
            b_root = root / "boundaries" / f"{left['window_id']}--{right['window_id']}"
            if not (b_root / "chat" / "receipt.json").is_file():
                if (b_root / "unknown.json").is_file():
                    unknown, ref = load_json(b_root / "unknown.json")
                    if any(r.get("relation") != "unknown" for r in unknown.get("json", {}).get("relations", [])):
                        raise ValueError("No-call boundary record contains nonunknown relations")
                    boundary_audit.append(unknown)
                    case["source_refs"].append(ref)
                else:
                    boundary_audit.append({"status": "pending", "left_window_id": left["window_id"],
                                           "right_window_id": right["window_id"], "reason": "no terminal RGB boundary receipt"})
                continue
            evidence, _ = load_json(b_root / "evidence.json")
            manifest = core.batch.verify_evidence(b_root / "rgb_evidence", {**base, "kind": "boundary", "left": left["window_id"], "right": right["window_id"]}, source)
            if evidence["frames"] != manifest["frames"] or not evidence.get("rgb_only") or evidence.get("ms_and_music_prior_supplied"):
                raise ValueError("Boundary evidence is not receipt-checked independent RGB")
            metadata = {"id": f"b-{left['window_id']}-{right['window_id']}", "left_window_id": left["window_id"], "right_window_id": right["window_id"], "time": core.window_bounds(left)[1]}
            result, refs, boundary_request = checked_transaction(b_root / "chat", base,
                lambda v, e=evidence, m=metadata: core.check_boundary(v, e["left"], e["right"], m))
            if run_version == "v3" and boundary_request:
                if (boundary_request.get("json_schema") != core.boundary_schema(evidence["left"], evidence["right"])
                        or boundary_request.get("prompt_sha256") != identity["boundary_prompt_sha256"]
                        or boundary_request["messages"][0]["content"] != core.BOUNDARY_SYSTEM):
                    raise ValueError("v3 boundary request did not bind actual frames and ordered body parts")
            case["source_refs"].extend(refs)
            boundary_audit.append({**result, "frames": manifest["frames"], "evidence_path": str(b_root / "evidence.json")})
            if result["status"] == "candidate":
                boundaries.extend(r for r in result["json"]["relations"] if r["relation"] != "unknown")
        stages = {}
        for name, relations in (("R1", []), ("R2", boundaries)):
            if name == "R2" and len(row["windows"]) == 1:
                reference = root / "R2" / "run.reference.json"
                if reference.exists():
                    value, ref = load_json(reference)
                    if Path(value["reused_from"]).resolve() != (root / "R1" / "final.json").resolve() or value.get("duplicate_model_call_performed") is not False:
                        raise ValueError("Whole R2 reference is not the same real R1 result")
                    result = {**stages["R1"], "reused_from": "R1 (single original window: no boundary/no duplicate call)"}
                    case["source_refs"].append(ref)
                else:
                    result = {"status": "pending", "reason": "latest whole R2 reuse reference not written"}
            else:
                binding = {**base, "kind": name, "S_sha256": object_digest(supplied), "boundary_sha256": object_digest(relations)}
                result, refs, request = checked_transaction(root / name, binding,
                    lambda v, bs=relations: core.check_rewrite(v, row, supplied["facts"], bs))
                case["source_refs"].extend(refs)
                if request:
                    if run_version == "v3" and request.get("json_schema") != core.rewrite_schema(supplied):
                        raise ValueError("v3 rewrite did not bind exact count/order/window IDs and local refs")
                    user = json.loads(request["messages"][1]["content"])
                    observed = user.pop("boundary_relations", [])
                    if user != supplied or observed != relations:
                        raise ValueError("R1/R2 do not share S facts or supplied real boundaries")
                    if request["sampling"] != {"temperature": 0.0, "seed": 0, "max_tokens": identity["rewrite_tokens"]}:
                        raise ValueError("R1/R2 sampling differs from shared seed/temperature/token budget")
                    if request["prompt_sha256"] != identity["rewrite_prompt_sha256"] or digest(request["messages"][0]["content"].encode()) != identity["rewrite_prompt_sha256"]:
                        raise ValueError("R1/R2 do not share the receipt-bound rewrite prompt")
            if result["status"] == "candidate":
                text_path = root / ("R1" if len(row["windows"]) == 1 and name == "R2" else name) / "timed.txt"
                if not text_path.is_file():
                    result = {"status": "pending", "reason": "terminal receipt exists but latest full text not saved yet"}
                else:
                    actual_text = text_path.read_text(encoding="utf-8")
                    if actual_text != core.timed_text(row, result["json"]["windows"]):
                        raise ValueError("Saved full caption differs from real generation window outputs")
                    result["review_full_text"] = actual_text
                    case["source_refs"].append({"path": str(text_path), "sha256": digest(text_path.read_bytes())})
            if name == "R2":
                result.setdefault("review_evidence", {})["boundary_audit"] = copy.deepcopy(boundary_audit)
            if result["status"] == "candidate":
                case["source_refs"].append(checked_runtime(out, identity))
            stages[name] = result
        views = [("R0", "MotionScript", original_review_windows(row))]
        if visual_windows:
            views.append(("visual_supplement", "独立VLM RGB", visual_windows))
        views.extend((name, "MotionScript", stage_windows(row, stages[name], supplied, [] if name == "R1" else boundaries)) for name in ("R1", "R2"))
        for name, source_name, windows in views:
            case["views"].append({"experiment_tag": EXPERIMENT, "run_version": run_version,
                                  "strategy": strategy, "stage": name,
                                  "actual_model_terminal": stages.get(name, {}).get("actual_model_terminal", False),
                                  "label": f"chunk4 {run_version} · {LABELS[strategy]} · {name}", "source": source_name,
                                  "windows": windows, "semantic_approval": False})
    for case in grouped.values():
        if {v["strategy"] for v in case["views"]} != set(STRATEGIES):
            raise ValueError("Review requires all four strategies for each same-source case")
        case["source_refs"].append({"path": str(Path(plan_path).resolve()), "sha256": plan_sha})
    return list(grouped.values())


def mark_interruptions(cases, controls):
    for strategy, path in controls.items():
        if strategy not in STRATEGIES:
            raise ValueError("Unknown interrupted strategy")
        control, ref = load_json(path)
        if (control.get("worker_started") is not True or control.get("partial_outputs_preserved") is not True
                or "manager received signal 15" not in control.get("error", "")):
            raise ValueError("Control receipt does not show preserved actual worker interruption")
        for case in cases:
            affected = False
            for view in case["views"]:
                if view["strategy"] != strategy or view["stage"] not in ("R1", "R2"):
                    continue
                for window in view["windows"]:
                    if window["status"] != "pending":
                        continue
                    window["status"] = "interrupted"
                    window["reason"] = "资源worker被signal15中断；无该窗正常终态generation，不是完整caption"
                    window.setdefault("events", {})["interruption_control"] = ref
                    affected = True
            if affected:
                case["source_refs"].append(ref)
    return cases


def publishable(cases):
    return any(v["stage"] == "R1" and v.get("actual_model_terminal") is True
               and any(w["status"] in ("candidate", "failed") for w in v["windows"])
               for c in cases for v in c["views"])


def main():
    global VALIDATOR_SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--validator-source", help="Explicit immutable producer snapshot for this run")
    parser.add_argument("--outputs", required=True, nargs="+", help="Explicit latest run roots, one per strategy or a shared root")
    parser.add_argument("--interrupted", action="append", default=[], metavar="STRATEGY=CONTROL_PATH",
                        help="Explicit version-specific interruption receipt; never marks outputs complete")
    parser.add_argument("--run-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--limit", type=int, choices=(3, 50), default=3)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--publish", action="store_true", help="Explicitly authorize local annotation update")
    args = parser.parse_args()
    if args.run_version == "v3" and not args.validator_source:
        parser.error("v3 requires an explicit frozen validator source")
    VALIDATOR_SOURCE = str(inside_exp(args.validator_source, args.exp_root)) if args.validator_source else None
    plan = inside_exp(args.plan, args.exp_root)
    outputs = [inside_exp(p, args.exp_root) for p in args.outputs]
    annotation = inside_exp(args.annotation, args.exp_root)
    bundle, ref = load_json(annotation)
    cases = build_cases(plan, outputs, args.limit, args.run_version)
    interruptions = {}
    for specification in args.interrupted:
        strategy, path = specification.split("=", 1)
        if strategy in interruptions:
            raise ValueError("Duplicate interrupted strategy")
        interruptions[strategy] = inside_exp(path, args.exp_root)
    mark_interruptions(cases, interruptions)
    if args.publish and not publishable(cases):
        raise ValueError("No normal-stop actual-model R1 terminal; refusing prepared-only publication")
    merged = append_cases(bundle, cases, run_version=args.run_version)
    changed = publish(annotation, ref["sha256"], merged) if args.publish else False
    print(json.dumps({"experiment_tag": EXPERIMENT, "case_count": len(cases),
                      "view_count": sum(len(c["views"]) for c in cases),
                      "published": args.publish, "changed": changed,
                      "semantic_approval": False, "reviews_or_scores_modified": False}))


if __name__ == "__main__":
    main()
