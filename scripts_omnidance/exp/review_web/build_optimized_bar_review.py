"""Receipt-checked optimized-bar diagnostic views; dry-run unless explicitly published."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import FunctionType

import build_caption_chunk4_review as review

IDS = ("hwrlCvOCYUKpljlV", "LglrebUOmlpWTFrr", "qKgmlBaDdOeqvUZZ")
PIPELINE = "bar_annotation_optimized_20261024_v1"
VERSION = "stable_roi3_v2"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def checked_ref(ref):
    value, actual = review.load_json(ref["path"])
    need(actual["sha256"] == ref["sha256"], "Referenced artifact SHA changed")
    return value, actual


def import_bar(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    import ms_bar_annotation as bar
    need(Path(bar.__file__).resolve() == source / "ms_bar_annotation.py", "Different bar validator already imported")
    need(Path(bar.core.__file__).resolve() == source / "ms_caption_rewrite.py", "Different core validator already imported")
    review.VALIDATOR_SOURCE = source
    return bar


def checked_attempt(path, binding, validator, model_sha, expected_request=None):
    value, refs, request = review.checked_transaction(path, binding, validator)
    need(value["status"] != "pending", "Missing terminal receipt; no old run fallback")
    if expected_request is not None:
        need(request == expected_request, "Request differs from actual frozen runtime inputs")
        receipt, _ = review.load_json(Path(path) / "receipt.json")
        need(receipt["binding"] == binding, "Terminal receipt has a different exact binding")
    generation = value.get("review_evidence", {}).get("generation")
    if generation is not None:
        need(generation.get("model_identity_sha256") == model_sha, "Generation model identity differs")
    if value["status"] == "failed":
        core = review.import_validator()
        reproduced = None
        try:
            core.validate_generation(Path(path), request)
            raw = (Path(path) / "response.raw.txt").read_text(encoding="utf-8")
            validator(core.parse_response(json.dumps(raw).encode()))
        except Exception as error:
            reproduced = f"{type(error).__name__}: {error}"
        need(reproduced == value.get("error"), "Saved failure not reproduced from actual raw/current validator")
        value["review_evidence"]["failure_reproduced_from_raw"] = True
        dispatch_path = Path(path) / "dispatch_error.json"
        if dispatch_path.is_file():
            dispatch, dispatch_ref = review.load_json(dispatch_path)
            receipt, _ = review.load_json(Path(path) / "receipt.json")
            expected_dispatch_sha = receipt["files_sha256"].get("dispatch_error.json")
            need(expected_dispatch_sha is None or dispatch_ref["sha256"] == expected_dispatch_sha, "Dispatch error SHA differs from terminal receipt")
            need(isinstance(dispatch.get("error"), str) and dispatch["error"].strip(), "Dispatch error text missing")
            value["review_evidence"].update(actualdispatch_error=dispatch["error"], dispatch_error_ref=dispatch_ref,
                dispatch_error_terminal_receipt_sha_bound=expected_dispatch_sha is not None)
            refs.append(dispatch_ref)
    return value, refs


def checked_prepared(bar, root, row, raw, binding, params):
    need(bar.load(root / "binding.json") == binding and bar.load(root / "plan.json") == row
         and (root / "plan.line.raw.json").read_bytes() == raw, "Source plan/binding changed")
    prepared = bar.load(root / "prepared.json")
    receipt = bar.load(root / "prepared.receipt.json")
    need(receipt["prepared_sha256"] == bar.batch.sha256(root / "prepared.json")
         and receipt["binding"] == binding == prepared["binding"], "Prepared receipt/binding changed")
    need(set(prepared["source_identity"]["files"]) == set(bar.batch.SOURCE_FIELDS), "Prepared source identity omits required files")
    # st_dev is mount-local across worker/reviewer nodes; validate stable actual bytes instead.
    for name, value in prepared["source_identity"]["files"].items():
        path = Path(value["path"]).resolve(strict=True)
        need(path == Path(row["manifest_record"][name]).resolve(), "Source path differs")
        before = bar.batch.snapshot(path)
        need(bar.batch.sha256(path) == value["sha256"] and bar.batch.snapshot(path) == before, "Source content changed during stable SHA verification")
        expected = {"video_path": "video_sha256", "audio_path": "audio_sha256", "nlf_path": "input_sha256"}[name]
        need(value["sha256"] == row["binding"][expected], "Source SHA differs from plan")
    review.verify_file_refs(prepared["source_refs"])
    forbidden, mask_meta = bar.load_hard_mask(row, prepared["probe"])
    need(mask_meta == prepared["raw2d_mask"] and {p: sorted(v) for p, v in forbidden.items()} == prepared["forbidden"],
         "Raw2D mask or actual forbidden part/frame support differs")
    need(bar.batch.sha256(root / "rgb/manifest.json") == prepared["union_manifest_sha256"], "Union manifest changed")
    union = bar.load(root / "rgb/manifest.json")
    need(union["binding"] == binding, "Union frame binding changed")
    for frame in union["frames"]:
        path = (root / "rgb" / frame["file"]).resolve(strict=True)
        need(path.is_relative_to((root / "rgb").resolve()) and bar.batch.sha256(path) == frame["sha256"], "Original RGB artifact changed")
    roi_manifest, roi_meta = bar.load_hand_roi(row, prepared["probe"], params["hand_roi_root"])
    need(roi_meta == prepared["hand_roi"] and roi_meta["available"], "Requested complete ROI source receipt differs or is unavailable")
    need([m["window"] for m in prepared["windows"]] == bar.normalized_windows(row), "Prepared normalized windows differ")
    need(prepared["boundaries"] == bar.boundary_specs(bar.normalized_windows(row), prepared["probe"], params["boundary_review"] == "on"),
         "Prepared boundary frame selection differs")
    for manifest in prepared["windows"]:
        window = manifest["window"]
        start, end = bar.core.window_bounds(window)
        selected, policy = bar.select_window_frames(prepared["probe"]["times"], start, end, params["sample_fps"])
        selected, evidence = bar.select_window_roi(roi_manifest, window, selected, prepared["probe"]["times"])
        if evidence and evidence.get("image_budget"):
            policy = {**policy, "body_budget_reduced_for_same_frame_ROI": True, "maximum_body_images": 8,
                      "actual_body_frame_count": len(selected), "maximum_total_body_plus_ROI_images": 16}
        need([f["index"] for f in manifest["frames"]] == selected and manifest["sampling_policy"] == policy
             and manifest["hand_roi_evidence"] == evidence, "ROI same-frame selection/evidence or body image budget differs")
    return prepared


def selected_plan(bar, protocol, latest):
    selection, summary = latest["selection"], latest["summary"]
    need(selection["selected_ids"] == list(IDS) and summary["selected_source_count"] == len(IDS),
         "Only the fixed three pilot sources may be reviewed")
    rows, plan_sha, total = bar.read_plan(protocol["plan_path"], selection["shard"], selection["nshard"], limit=len(IDS))
    need([r["id"] for r, _ in rows] == list(IDS), "Selected pilot plan differs from fixed three sources")
    need(total == protocol["source_count"] == summary["total_source_count"], "Full plan source count differs")
    need(plan_sha == protocol["plan_sha256"] == summary["plan_sha256"], "Run plan SHA differs")
    return rows, plan_sha


def checked_rewrite(bar, root, args, binding, windows, observations, facts, boundaries, saved, model_sha):
    supplied = {"windows": [{"window_id": w["window_id"], "source_status": o["status"],
                             "source_status_is_audit_only": True,
                             "admitted_fact_count": sum(f["window_id"] == w["window_id"] for f in facts)}
                            for w, o in zip(windows, observations)],
                "facts": [{k: copy.deepcopy(v) for k, v in f.items() if k != "raw_text"} for f in facts]}
    path = Path(saved["attempt_path"]).resolve()
    original = Path(saved.get("original_attempt", {}).get("attempt_path", path)).resolve()
    need(original.parent == (root / "rewrite").resolve() and original.name in ("attempt-0", "attempt-1"),
         "Rewrite attempt escaped source or is not an executed attempt")
    item = {"root": original, "request": bar.bar_rewrite_request(args, supplied, boundaries),
            "validator": lambda v: bar.check_supported_rewrite(v, windows, facts, boundaries),
            "binding": {**binding, "facts_sha256": bar.batch.digest(facts), "boundaries_sha256": bar.batch.digest(boundaries)}}
    refs, attempts = [], []
    def replay(items, runtime, replay_args):
        need(len(items) == 1 and len(attempts) < 2, "More than one rewrite reference repair")
        current = items[0]
        expected_root = original if not attempts else original / "reference-repair"
        need(current["root"] == expected_root, "Rewrite repair escaped original attempt")
        value, found = checked_attempt(current["root"], current["binding"], current["validator"],
                                       model_sha, current["request"])
        attempts.append(current["root"])
        refs.extend(found)
        return {str(current["root"]): value}
    # Replay the executed repair algorithm with receipt-only transactions, never writes or inference.
    execute = bar.rewrite_with_reference_repair
    replay_rewrite = FunctionType(execute.__code__, {**execute.__globals__, "transactions": replay},
                                  execute.__name__, execute.__defaults__, execute.__closure__)
    checked = replay_rewrite(item, None, args, facts)
    need(Path(checked["attempt_path"]).resolve() == path, "Final rewrite is not the executed one-repair selection")
    for field in ("status", "json", "error", "reference_repair"):
        need(checked.get(field) == saved.get(field), "Final rewrite differs from real raw or one-repair binding")
    original_saved, original_checked = saved.get("original_attempt"), checked.get("original_attempt")
    need(bool(original_saved) == bool(original_checked), "Original rewrite repair audit differs")
    if original_saved:
        for field in ("status", "json", "error", "attempt_path"):
            need(original_saved.get(field) == original_checked.get(field), "Original failed rewrite differs from real raw")
    return checked, refs


def validate_run(run, validator_source):
    run = Path(run).resolve()
    need(run.name == VERSION, "Run directory differs from explicit review version")
    bar = import_bar(validator_source)
    protocol, protocol_ref = review.load_json(run / "protocol.json")
    for path, expected in protocol["params"]["code_sha256"].items():
        need(review.digest(Path(path).read_bytes()) == expected, "Frozen runtime source changed")
    need(protocol["params"]["code_sha256"] == bar.code_identity(), "Validator is not the executed frozen runtime")
    latest, latest_ref = review.load_json(run / "latest.summary.json")
    snapshot, snapshot_ref = checked_ref(latest["snapshot_ref"])
    need(latest["summary"] == snapshot["summary"] and latest["selection"] == snapshot["selection"], "Latest summary differs from snapshot")
    need(latest["summary"]["stage"] == "run" and latest["summary"]["processed_source_count"] == 3
         and latest["summary"]["prepared_source_count"] == 0, "Pilot is not terminal for three sources")
    rows, plan_sha = selected_plan(bar, protocol, latest)
    args = argparse.Namespace(**protocol["params"], model=protocol["params"]["model_path"], retry_failed=False)
    need(protocol["params"]["system_sha256"] == bar.sha(bar.SYSTEM.encode())
         and protocol["params"]["rewrite_system_sha256"] == bar.sha((bar.core.REWRITE_SYSTEM + bar.BAR_REWRITE_RULES).encode()),
         "Executed visual/rewrite system prompt differs")
    model, model_ref = review.load_json(run / "model.identity.json")
    content = {name: {"path": v["path"], "sha256": v["sha256"]} for name, v in model["files"].items()}
    need(model["sha256"] == bar.batch.digest({"root": model["root"], "files": content}), "Model content identity digest differs")
    refs = [protocol_ref, latest_ref, snapshot_ref, model_ref]
    records = {r["id"]: r for r in snapshot["records"]}
    need(len(snapshot["records"]) == len(records) == 3 and set(records) == set(IDS), "Duplicate or missing snapshot sources")
    materials = []
    for row, raw in rows:
        sid = row["id"]
        record, result_ref = checked_ref(records[sid]["result_ref"])
        need(record == {k: v for k, v in records[sid].items() if k != "result_ref"}, "Snapshot record differs from saved result")
        need(record["schema"] == bar.OUTPUT_SCHEMA and record["status"] in ("candidate", "failed", "abstain"), "Nonterminal result")
        need(record["model_identity_ref"] == model_ref, "Result model reference differs")
        root = run / "sources" / sid
        binding = bar.load(root / "binding.json")
        expected_binding = {"source_binding": row["binding"], "plan_sha256": plan_sha, "full_source_row_sha256": bar.batch.digest(row),
                            "raw_line_sha256": bar.sha(raw), "manifest_record_sha256": bar.batch.digest(row["manifest_record"]), "params_sha256": bar.batch.digest(protocol["params"])}
        need(binding == record["binding"] == expected_binding, "Result/source binding differs from executed plan and params")
        prepared = checked_prepared(bar, root, row, raw, binding, protocol["params"])
        need(record["hand_roi"] == prepared["hand_roi"], "Result ROI receipt differs")
        need(prepared["original_windows"] == row["windows"] == record["original_windows"]
             and prepared["window_manifest"] == row["window_manifest"] == record["original_window_manifest"], "Original windows changed")
        need(abs(bar.core.window_bounds(row["windows"][-1])[1] - row["binding"]["media_end"]) <= 1e-8, "Tail window does not cover source media")
        pts = bar.load(root / "actual_pts.json")
        need(pts == prepared["probe"] and bar.sha(bar.batch.json_bytes(pts["times"])) == row["binding"]["actual_video_pts_sha256"], "Actual source PTS differs")
        union = bar.load(root / "rgb/manifest.json")
        by_index = {f["index"]: f for f in union["frames"]}
        for frame in union["frames"]:
            need(frame["time"] == pts["times"][frame["index"]]
                 and frame["source_timestamp"] == pts["source_timestamps"][frame["index"]], "RGB frame PTS differs")
        forbidden = {p: set(v) for p, v in prepared["forbidden"].items()}
        observations, evidence_refs = [], [result_ref, {"path": str(root / "prepared.json"), "sha256": review.digest((root / "prepared.json").read_bytes())}, record["raw_rgb_manifest"]]
        roi_path = Path(prepared["hand_roi"]["path"])
        for path, key in ((roi_path, "sha256"), (roi_path.with_name("source_manifest.json"), "source_manifest_sha256"),
                          (roi_path.with_name("extract_receipt.json"), "extract_receipt_sha256")):
            evidence_refs.append({"path": str(path), "sha256": prepared["hand_roi"][key]})
        for manifest in prepared["windows"]:
            for part in ("left_hand", "right_hand"):
                for crop in (manifest.get("hand_roi_evidence") or {}).get(part, {}).get("crops", []):
                    evidence_refs.append({"path": crop["path"], "sha256": crop["sha256"]})
        need(len(prepared["windows"]) == len(record["window_observations"]), "Observation window count differs")
        for manifest, saved in zip(prepared["windows"], record["window_observations"]):
            window = manifest["window"]
            need(saved["window"] == window and all(by_index[f["index"]] == f for f in manifest["frames"]), "Observation window/frame binding differs")
            if len(manifest["frames"]) < 2:
                checked = bar.load(root / "windows" / window["window_id"] / "evidence_abstention.json")
                need(checked["status"] == "abstain" and checked["json"]["abstain"], "Short window must abstain")
            else:
                path = Path(saved["attempt_path"]).resolve()
                need(path.is_relative_to(root / "windows" / window["window_id"]), "Observation attempt escaped source window")
                expected = {**binding, "window": window, "frames_sha256": bar.batch.digest(manifest["frames"])}
                need(path.parent == (root / "windows" / window["window_id"]).resolve() and path.name in ("attempt-0", "attempt-1"),
                     "Observation is not an executed original window attempt")
                checked, found = checked_attempt(path, expected, lambda v, m=manifest: bar.filter_supported_observation(v, m, forbidden),
                                                 model["sha256"], bar.visual_request(args, root / "rgb", manifest, forbidden))
                if checked["status"] == "candidate":
                    checked["raw_observation"] = bar.core.parse_response(bar.batch.json_bytes((path / "response.raw.txt").read_text(encoding="utf-8")))
                evidence_refs.extend(found)
                if checked["status"] == "candidate" and checked["json"]["abstain"]:
                    checked["status"] = "abstain"
            need(checked["status"] == saved["status"] and checked.get("json") == saved.get("json")
                 and checked.get("error") == saved.get("error"), "Saved observation differs from checked real raw")
            observations.append(checked)
        windows = [m["window"] for m in prepared["windows"]]
        facts = bar.facts_from_observations(windows, observations)
        need(facts == record["facts"], "Facts differ from raw observations")
        boundaries, boundary_results, usable_results = [], [], []
        if facts:
            for spec in prepared["boundaries"]:
                item = bar.boundary_item(spec, prepared, root, args, binding, observations)
                if "root" not in item:
                    boundary_results.append(item)
                    continue
                checked, found = checked_attempt(item["root"], item["binding"], item["validator"], model["sha256"], item["request"])
                evidence_refs.extend(found)
                usable_results.append(checked)
            boundary_results.extend(usable_results)
            for checked, saved in zip(boundary_results, record["boundary_observations"]):
                need(all(checked.get(k) == saved.get(k) for k in ("status", "json", "error")),
                     "Boundary support-gated relations differ from checked raw")
            need(len(boundary_results) == len(record["boundary_observations"]), "Boundary observation count differs")
            boundaries = [rel for checked in boundary_results if checked["status"] in ("candidate", "unknown") for rel in checked["json"]["relations"]]
        else:
            need(not record["boundary_observations"], "No facts may not dispatch boundaries")
        final = copy.deepcopy(record["final"])
        if facts:
            need("attempt_path" in final, "Supported facts require a terminal rewrite transaction")
            final, found = checked_rewrite(bar, root, args, binding, windows, observations, facts, boundaries, final, model["sha256"])
            evidence_refs.extend(found)
        else:
            need(final["status"] == "abstain" and final["json"] == record["visual_baseline"]
                 and final.get("model_call_performed") is False and "attempt_path" not in final, "No-facts source must abstain without rewrite")
        rewrite_calls = sum(attempt.get("review_evidence", {}).get("generation", {}).get("model_call_performed") is True
                            for attempt in (final, final.get("original_attempt", {})))
        need(record["rewrite_fulltext_calls_per_attempt"] == rewrite_calls, "Rewrite call count differs from actual generation receipts")
        need(record["visual_baseline"] == {"windows": [{"window_id": w["window_id"], "caption": " ".join(f["raw_text"] for f in facts if f["window_id"] == w["window_id"]), "used_fact_ids": [f["fact_id"] for f in facts if f["window_id"] == w["window_id"]], "used_boundary_ids": []} for w in windows], "issues": []}, "Baseline is not verbatim raw facts")
        need(record["visual_baseline_timed_caption"] == bar.core.timed_text({"windows": windows}, record["visual_baseline"]["windows"]),
             "Saved raw baseline timing differs")
        need(record["final_timed_caption"] == (bar.core.timed_text({"windows": windows}, final["json"]["windows"]) if "json" in final else ""), "Saved final timing differs")
        material = {"record": record, "prepared": prepared, "observations": observations, "final": final, "root": root, "refs": refs + evidence_refs}
        materials.append(material)
    return materials, latest["summary"]


def build_case(material, existing):
    record, prepared = material["record"], material["prepared"]
    windows = [m["window"] for m in prepared["windows"]]
    original = [v for v in existing["views"] if v.get("source") == "MotionScript" and v.get("label") == "小节 merged短边缘"]
    need(len(original) == 1 and [(w["start"], w["end"]) for w in original[0]["windows"]] == [(w["start"], w["end"]) for w in windows], "Original merged MS comparison differs")
    meta = {"experiment_tag": review.EXPERIMENT, "run_version": VERSION, "strategy": "mergedbar", "annotation_pipeline": PIPELINE, "semantic_approval": False}
    views = [{**meta, "stage": "R0", "label": f"optimized-bar {VERSION} · 原merged MS · R0（无新调用）", "source": "MotionScript（原有对照）", "actual_model_terminal": False, "windows": copy.deepcopy(original[0]["windows"])}]
    for stage, outputs, label in (("R1", record["visual_baseline"]["windows"], "原视觉事件baseline（不是改写）"),
                                  ("R2", material["final"].get("json", {}).get("windows", []), "保真全文＋边界（至多一次显式引用repair；非新观察）")):
        by_id = {w["window_id"]: w for w in outputs}
        rendered = []
        for manifest, observation in zip(prepared["windows"], material["observations"]):
            w = manifest["window"]
            out = by_id.get(w["window_id"], {})
            frames = [{**copy.deepcopy(f), "path": str(material["root"] / "rgb" / f["file"])} for f in manifest["frames"]]
            detail = {"window_id": w["window_id"], "raw_window_indices": w["raw_window_indices"], "part_support": copy.deepcopy(observation.get("json", {}).get("part_support", [])),
                      "raw_events": copy.deepcopy(observation.get("raw_observation", {}).get("events", [])),
                      "admitted_events": copy.deepcopy(observation.get("json", {}).get("events", [])),
                      "support_filter_audit": copy.deepcopy(observation.get("json", {}).get("support_filter_audit", {})),
                      "raw_observation": copy.deepcopy(observation.get("raw_observation")),
                      "hand_roi_evidence": copy.deepcopy(manifest.get("hand_roi_evidence")), "hand_roi": copy.deepcopy(prepared.get("hand_roi")),
                      "uncertainties": copy.deepcopy(observation.get("json", {}).get("uncertainties", [])),
                      "frames": frames, "used_fact_ids": out.get("used_fact_ids", []), "used_boundary_ids": out.get("used_boundary_ids", []),
                      "observation_status": observation["status"], "observation_error": observation.get("error"), "observation_transaction": observation.get("review_evidence"),
                      "rewrite_transaction": material["final"].get("review_evidence") if stage == "R2" else None,
                      "rewrite_error": material["final"].get("error") if stage == "R2" else None,
                      "actualdispatch_error": material["final"].get("review_evidence", {}).get("actualdispatch_error") if stage == "R2" else None,
                      "dispatch_error_ref": material["final"].get("review_evidence", {}).get("dispatch_error_ref") if stage == "R2" else None,
                      "dispatch_error_terminal_receipt_sha_bound": material["final"].get("review_evidence", {}).get("dispatch_error_terminal_receipt_sha_bound") if stage == "R2" else None,
                      "original_rewrite_transaction": material["final"].get("original_attempt", {}).get("review_evidence") if stage == "R2" else None,
                      "reference_repair": material["final"].get("reference_repair") if stage == "R2" else None, "source_refs": material["refs"],
                      "part_support_is_AI_candidate_not_pixel_GT": True, "boundary_observations": record["boundary_observations"] if stage == "R2" else []}
            status = observation["status"]
            if stage == "R2" and material["final"]["status"] == "failed":
                status = "failed"
            rendered.append({"start": w["start"], "end": w["end"], "status": status, "caption": out.get("caption", ""),
                             "reason": "全小节独立视觉弱标签；unknown/failed不是静止；原窗不是精确动作边界；未人工批准", "events": detail})
        actual = (all(o.get("actual_model_terminal", False) for o in material["observations"]) if stage == "R1" else material["final"].get("actual_model_terminal", False))
        views.append({**meta, "stage": stage, "label": f"optimized-bar {VERSION} · " + label + " · " + stage, "source": "全小节独立视觉弱标签（非MS补空）",
                      "actual_model_terminal": actual, "source_status": record["status"], "full_text": record["visual_baseline_timed_caption" if stage == "R1" else "final_timed_caption"], "windows": rendered})
    return {"id": record["id"], "video": record["manifest_record"]["video_path"], "audio": record["manifest_record"]["audio_path"], "views": views, "source_refs": material["refs"]}


def build(bundle, materials):
    need([m["record"]["id"] for m in materials] == list(IDS), "Only fixed three cases may be appended")
    existing = {c["id"]: c for c in bundle["caption_cases"]}
    need(all(sid in existing for sid in IDS), "Fixed original cases missing")
    cases = [build_case(m, existing[m["record"]["id"]]) for m in materials]
    return review.append_cases(bundle, cases, run_version=VERSION), cases


def main(argv=None):
    global VERSION
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--run-version", required=True, choices=("stable_roi3_v2", "stable_roi3_v3"))
    p.add_argument("--validator-source", required=True)
    p.add_argument("--annotation", required=True)
    p.add_argument("--publish", action="store_true", help="Only after explicit pilot-review authorization")
    args = p.parse_args(argv)
    VERSION = args.run_version
    bundle, source_ref = review.load_json(args.annotation)
    materials, summary = validate_run(args.run, args.validator_source)
    updated, cases = build(bundle, materials)
    changed = review.publish(args.annotation, source_ref["sha256"], updated) if args.publish else False
    print(json.dumps({"dry_run": not args.publish, "published": changed, "input_annotation_sha256": source_ref["sha256"], "candidate_annotation_sha256": review.object_digest(updated),
                      "run_version": VERSION, "caption_cases": len(cases), "new_views": sum(len(c["views"]) for c in cases), "run_summary": summary,
                      "scores_modified": False, "model_inference_repeated": False, "semantic_approval": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
