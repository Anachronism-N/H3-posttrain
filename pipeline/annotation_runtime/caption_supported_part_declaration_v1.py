"""Isolated CPU declaration audit; raw observations and native validators stay frozen."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
sys.path.insert(0, str(RT))
import caption_direct_facts_v1 as direct
import caption_direct_cpu_completion_fastlane_v1 as fast

RECIPE = "caption_supported_part_declaration_v1"
FLAGS = dict(direct.FLAGS, semantic_validation_completed=False, pixel_quality_review_completed=False)
PILOT = BAR / "caption_batch8_direct_full_pilot_v1"
SOURCE_IDS = ("VGfurWgIdrVQNyhI", "HVsstNGHcxHNsWkM", "VrydLuJOvAnzhiQt")
REPORT = RT / "caption_supported_part_declaration_pilot3_v1.json"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def checked_filter(raw, manifest, forbidden):
    bar, _ = direct.native()
    try:
        value = bar.filter_supported_observation(raw, manifest, forbidden)
    except (ValueError, TypeError, KeyError) as error:
        return dict(status="native_rejected", error=type(error).__name__ + ": " + str(error))
    return dict(status="abstain" if value["abstain"] else "candidate", json=value)


def required_declarations(event, raw, manifest, forbidden):
    """Only original lexical singleton requirements; never choose an ambiguous side."""
    bar, _ = direct.native()
    groups = bar.claimed_part_groups(event["description"])
    missing = [g for g in groups if not (g & set(event["parts"]))]
    if not missing:
        return [], "no_missing_native_required_part"
    if any(len(g) != 1 for g in groups):
        return [], "anatomical_side_unknown"
    text = event["description"]
    if re.search(r"\b(?:fingers?|thumbs?|palms?|wrists?|fists?|knuckles?|nails?|curled|cupped|splayed|clenched|shape|shapes|gesture|gestures|pinch|pinches|heart|okay|ok)\b|\bV[- ]shape\b|\b(?:open|closed)\s+hands?\b", text, re.I):
        return [], "hand_shape_or_fine_anatomy_not_repaired"
    if re.search(r"\b(?:screen|viewer|camera|image)[ -]?(?:left|right)\b|\b(?:left|right)[ -]?(?:screen|viewer|camera|image)\b|\b(?:unclear|uncertain|ambiguous|unknown|possibly|maybe|either)\b", text, re.I):
        return [], "anatomical_side_unknown"
    if any(re.search(r"\b(?:side|left|right|anatomical)\b", x, re.I) for x in raw["uncertainties"]):
        return [], "anatomical_side_uncertainty_preserved"
    supports = {s["part"]: s for s in raw["part_support"]}
    required = set().union(*groups)
    refs = bar.check_indices(event["frame_indices"], {f["index"] for f in manifest["frames"]}, 2)
    # Check all required parts, including both sides of plural claims, atomically.
    for part in required:
        row = supports[part]
        available = bar.check_indices(row["frame_indices"], {f["index"] for f in manifest["frames"]})
        if len(refs & available) < 2:
            return [], "each_required_part_needs_two_original_event_frames"
        if refs & forbidden.get(part, set()):
            return [], "rawmask_forbids_original_event_part_frame"
        if not re.fullmatch(r"(?:fully[ -])?visible[.!]?", row["reason"].strip(), re.I):
            return [], "original_support_reason_not_unambiguously_visible"
    return [p for p in bar.PARTS if any(p in g for g in missing)], "original_two_frame_supported_singleton_declarations"


def derive(raw, manifest, forbidden=None):
    """Full original native validation is mandatory; failure is an auditable no-op."""
    bar, _ = direct.native()
    forbidden = forbidden or {}
    original = copy.deepcopy(raw)
    raw_result = checked_filter(original, manifest, forbidden)
    candidate = copy.deepcopy(original)
    delta, decisions = [], []
    original_error = None
    try:
        bar.validate_observation(original, manifest, forbidden)
    except (ValueError, TypeError, KeyError) as error:
        original_error = type(error).__name__ + ": " + str(error)
    if original_error is None:
        for event in candidate["events"]:
            additions, reason = required_declarations(event, original, manifest, forbidden)
            decisions.append(dict(raw_event_id=event["id"], reason=reason))
            if additions:
                before = list(event["parts"])
                event["parts"].extend(additions)
                delta.append(dict(raw_event_id=event["id"], before_parts=before,
                                  added_parts=additions, after_parts=list(event["parts"])))
        bar.validate_observation(candidate, manifest, forbidden)
    new_result = checked_filter(candidate, manifest, forbidden)
    need(raw == original, "Caller raw was mutated")
    for old, new in zip(original.get("events", []), candidate.get("events", [])):
        need({k: v for k, v in old.items() if k != "parts"} ==
             {k: v for k, v in new.items() if k != "parts"}, "Non-declaration event field changed")
    need({k: v for k, v in original.items() if k != "events"} ==
         {k: v for k, v in candidate.items() if k != "events"}, "Original support or uncertainty changed")
    return dict(recipe=RECIPE, raw_original=original, declared_observation=candidate,
        declared_parts_delta=delta, declaration_decisions=decisions,
        original_native_validate_passed=original_error is None, original_native_validate_error=original_error,
        original_raw_filter=raw_result, new_checked_filter=new_result,
        raw_original_unchanged=True, original_text_ids_frames_support_uncertainties_unchanged=True,
        full_original_validation_required=True,
        gate_limitation="Frozen validate_observation includes semantic part declarations, not only JSON schema; missing declarations therefore remain unrepaired.",
        native_filter_functions_unchanged=True, new_model_calls=0, new_gpu_use=False, **FLAGS)


def pilot3():
    """Read only three already completed full-pilot sources, annotation bytes only."""
    bar, _ = direct.native()
    evidence = fast.Evidence()
    report = evidence.load(PILOT / "report.json")
    need(tuple(report["source_ids"]) == SOURCE_IDS and report["source_count"] == 3, "Fixed three-source pilot identity differs")
    rows = report["results"]
    need(tuple(r["id"] for r in rows) == SOURCE_IDS, "Fixed report result order differs")
    sources = []
    for row in rows:
        sid = row["id"]
        original = evidence.checked(row["joint_candidate"]["visual_result_ref"])
        need(row["visual"]["result_ref"] == row["joint_candidate"]["visual_result_ref"] and
             dict(original, result_ref=row["visual"]["result_ref"]) == row["visual"],
             "Report visual differs from actual original result")
        root = PILOT / "sources" / sid
        receipt = evidence.load(root / "prepared.receipt.json")
        prepared = evidence.load(root / "prepared.json", receipt["prepared_sha256"])
        need(receipt["binding"] == prepared["binding"] == original["binding"] and
             prepared["original_windows"] == original["original_windows"], "Original window/source binding differs")
        union = evidence.checked(original["raw_rgb_manifest"])
        need(union["binding"] == original["binding"] and prepared["union_manifest_sha256"] ==
             original["raw_rgb_manifest"]["sha256"], "Original union binding differs")
        model = evidence.checked(original["model_identity_ref"])
        forbidden = {p: set(v) for p, v in prepared["forbidden"].items()}
        need(len(prepared["windows"]) == len(original["window_observations"]), "Window count differs")
        windows = []
        for manifest, observation in zip(prepared["windows"], original["window_observations"]):
            wid = manifest["window"]["window_id"]
            need(re.fullmatch(r"w[0-9]{5}", wid) is not None and observation["window"] == manifest["window"], "Original window identity differs")
            local = root / "windows" / wid / "manifest.json"
            need(evidence.load(local) == dict(manifest, binding=original["binding"],
                 union_manifest_sha256=prepared["union_manifest_sha256"], raw2d_mask=prepared["raw2d_mask"], **bar.FLAGS),
                 "Original local/prepared manifest differs")
            need(all(frame in union["frames"] for frame in manifest["frames"]), "Event evidence outside original union")
            attempt = Path(observation["attempt_path"]).resolve()
            need(attempt.parent == root / "windows" / wid and re.fullmatch(r"attempt-[0-9]+", attempt.name) is not None, "Original attempt path differs")
            raw_receipt = evidence.load(attempt / "receipt.json")
            need(raw_receipt["status"] == "candidate" and fast.REQUIRED <= set(raw_receipt["files_sha256"]), "Original actual generation receipt missing")
            for name, sha in raw_receipt["files_sha256"].items():
                need(Path(name).name == name, "Unsafe annotation receipt name")
                evidence.raw(attempt / name, sha)
            binding = evidence.load(attempt / "binding.json")
            need(binding == raw_receipt["binding"] == dict(original["binding"],
                 window=manifest["window"], frames_sha256=bar.batch.digest(manifest["frames"])), "Original raw generation/frame binding differs")
            request = evidence.load(attempt / "request.json")
            fast.generation_check(evidence, attempt, request, model)
            raw = evidence.load(attempt / "response.raw.txt")
            audit = derive(raw, manifest, forbidden)
            need(audit["original_raw_filter"].get("json") == observation["json"] ==
                 evidence.load(attempt / "response.json") == evidence.load(attempt / "final.json")["json"],
                 "Original complete raw filter replay differs")
            need(audit["original_raw_filter"]["status"] == observation["status"], "Original candidate status differs")
            windows.append(dict(window=manifest["window"], original_raw_ref=evidence.ref(attempt / "response.raw.txt"),
                                original_manifest_ref=evidence.ref(local), audit=audit))
        baseline = sum(w["audit"]["original_raw_filter"]["status"] == "candidate" for w in windows)
        checked = sum(w["audit"]["new_checked_filter"]["status"] == "candidate" for w in windows)
        sources.append(dict(id=sid, original_visual_result_ref=row["joint_candidate"]["visual_result_ref"],
            window_count=len(windows), original_candidate_windows=baseline, new_checked_candidate_windows=checked,
            recovered_window_program_upper_bound=max(0, checked - baseline),
            applied_part_declarations=sum(len(d["added_parts"]) for w in windows for d in w["audit"]["declared_parts_delta"]), windows=windows))
    boundary = evidence.boundary()
    return dict(schema="caption_supported_part_declaration_pilot3_v1", recipe=RECIPE, sources=sources,
        source_count=3, window_count=sum(s["window_count"] for s in sources),
        recovered_window_program_upper_bound=sum(s["recovered_window_program_upper_bound"] for s in sources),
        applied_part_declarations=sum(s["applied_part_declarations"] for s in sources),
        annotation_files_read_once=len(evidence.cache), annotation_bytes_read_once=sum(len(r[0]) for r in evidence.cache.values()),
        immutable_annotation_refs=boundary, media_or_weights_content_read=False,
        new_model_calls=0, new_gpu_use=False, production_counts_changed=False,
        cpu_program_checks_are_not_pixel_or_semantic_quality_approval=True, **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot3", action="store_true", required=True)
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()
    path = Path(args.report).resolve()
    need(path == REPORT and not path.exists(), "Only fresh isolated fixed CPU audit report allowed")
    result = pilot3()
    with path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("source_count", "window_count", "applied_part_declarations",
        "recovered_window_program_upper_bound", "annotation_bytes_read_once")}, sort_keys=True))


if __name__ == "__main__":
    main()
