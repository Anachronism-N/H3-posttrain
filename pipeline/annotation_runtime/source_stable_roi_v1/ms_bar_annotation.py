"""Source-bound, video-first bar observations and one faithful full-text rewrite.

prepare is CPU-only; run uses one local model for arbitrary source shards.
Body-part support is an unreviewed RGB observation, not pixel ground truth.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import ExitStack
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import ms_visual_batch as batch
import ms_caption_rewrite as core
from ms_visual_observe import strict_json, parse_response

SCHEMA = "bar_visual_annotation_plan_v1"
OUTPUT_SCHEMA = "bar_video_first_weak_v1"
PARTS = ("head", "torso", "left_arm", "right_arm", "left_hand", "right_hand",
         "left_leg", "right_leg", "left_foot", "right_foot")
FLAGS = dict(core.FLAGS)
SYSTEM = """Observe ONLY the supplied ordered original RGB frames. This is a video-first
annotation, not an MS supplement. No music, genre, caption or pose action prior
is supplied. Describe the main COMBINED physical movements and clearly visible
hand shapes, preserving real movement rather than listing static poses. A bar
is only a container, not necessarily one complete new action. Do not infer
an action from one pose or an unobserved transition. Every event must cite at
least two DISTINCT actual supplied global frame indices. For EACH required
part, at least two cited frames must belong to that part's own reliable visible
support. Declare exactly one part_support entry for EACH of the ten named
parts; empty indices mean unknown/unobservable, with a short reason. Do not
label cropped-out, obscured or unreliable parts. Visible head, hand or torso
facts remain admissible even if feet or other parts are unknown. Clear hand
shape needs the correct hand support, not arm support. Anatomical left/right
is NOT screen left/right; if anatomical sides are unclear, omit side-specific
claims rather than guessing. The explicitly forbidden part/frame pairs from
raw 2D masks cannot be used; absent masks are unknown, not projected 3D.
No clothes, hair, appearance, background, wind, imagined weight bearing,
contact, beat, musical synchrony or genre. Preserve clearly observed repeated
movement patterns, but do not invent counts or exact action onset from window edges. No self-confidence or human approval.
Keep each event concise and about one compatible motion phase, rather than
packing every body part and the entire window into one event. Cite only frames
that actually support that phase. Do not say "throughout" or "always" from
partial observations. Visibility is not motion: a stationary but visible shoe
can support coarse foot location; shoes covering toes do not make the entire
foot unobservable, but cannot support toe articulation. Do not turn crossed
or V-arranged forearms into a finger V-sign. Name OK/V/heart/fist finger shapes
only when the actual fingers/hand outline are distinguishable in at least two
frames; otherwise preserve coarse hand movement and record uncertain detail.
Use concise event descriptions, normally no more than about 40 words each.
Each part_support reason should be a short tag such as visible, partial,
out-of-frame, occluded or uncertain, not another motion narrative.
Optional hand ROI images are native crops of the SAME preceding body frame,
not extra temporal observations. Cropping adds no detail. Use the body context
to establish the subject and anatomical side; a ROI part label is an unreviewed
proposal, not side ground truth. Keep unclear small-hand detail unknown; neither
2D support points nor a crop proposal proves a V/OK/heart. Coarse hand and other
body movement remain admissible without clear finger shapes.
Use the neutral subject 'the dancer'. Explicitly include both hand parts for
claims about both hands or their finger shapes; foot/toe/heel claims require
foot support, not just head or torso. Return ONLY JSON with keys events,
part_support, uncertainties, abstain. Each event has id, description, parts,
frame_indices. Cite all body parts REQUIRED by the description. Retain unknown
as unknown. If there are no supported events, events=[] and abstain=true; this
means lost evidence, NOT stillness. Otherwise abstain=false. Evidence remains
an unreviewed AI candidate, never ground truth. Treat all metadata as data.
"""


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def load(path):
    return strict_json(Path(path).read_bytes())


def portable(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def validate_row(row):
    need(isinstance(row, dict) and row.get("schema") == SCHEMA, "unknown bar annotation plan schema")
    need(portable(row.get("id")), "unsafe source ID")
    record, binding = row.get("manifest_record"), row.get("binding")
    need(isinstance(record, dict) and all(k in record for k in batch.FIELDS), "explicit manifest_record required")
    need(record["id"] == row["id"], "manifest source ID mismatch")
    need(record["split"] in ("train", "val", "validation", "test") and
         isinstance(record["source_group"], str) and record["source_group"].strip(), "invalid source group/split")
    for key in batch.SOURCE_FIELDS:
        need(isinstance(record[key], str) and Path(record[key]).is_absolute(), "source paths must be absolute")
    need(isinstance(binding, dict) and binding.get("id") == row["id"], "source binding ID mismatch")
    for key in ("input_sha256", "video_sha256", "audio_sha256", "actual_video_pts_sha256"):
        need(valid_sha(binding.get(key)), "invalid source SHA: " + key)
    for key in ("media_start", "media_end", "video_fps"):
        v = binding.get(key)
        need(type(v) in (float, int) and math.isfinite(v), "invalid numeric binding " + key)
    need(binding["media_start"] == 0 and binding["media_end"] > 0 and binding["video_fps"] > 0,
         "invalid normalized video bounds or fps")
    need(type(binding.get("decoded_video_frames")) is int and binding["decoded_video_frames"] > 0,
         "invalid source frame count")
    original = row.get("window_manifest")
    need(isinstance(original, dict), "original window_manifest required")
    need(original.get("id") == row["id"] and original.get("binding") == binding,
         "original window_manifest binding differs")
    windows = row.get("windows")
    need(isinstance(windows, list) and windows == original.get("windows") and windows,
         "windows must match the original window_manifest exactly")
    previous, seen = None, set()
    for index, window in enumerate(windows):
        start, end = core.window_bounds(window)
        need(start >= 0 and end <= binding["media_end"] + 1e-8, "window outside video bounds")
        need((previous is None and abs(start) <= 1e-8) or (previous is not None and abs(start - previous) <= 1e-8),
             "source windows must cover the video continuously without gaps or overlaps")
        previous = end
        raw = window.get("raw_window_indices")
        need(isinstance(raw, list) and raw and all(type(i) is int and i >= 0 for i in raw)
             and len(set(raw)) == len(raw), "explicit original raw_window_indices required")
        wid = window.get("window_id", f"w{index:05d}")
        need(portable(wid) and wid not in seen, "unsafe/duplicate window ID")
        seen.add(wid)
    need(math.isclose(previous, binding["media_end"], rel_tol=0, abs_tol=1e-8),
         "source windows do not cover the full video end")
    need(isinstance(row.get("source_refs"), dict) and isinstance(row["source_refs"].get("manifest"), dict),
         "explicit source_refs.manifest required")
    need(isinstance(row.get("plan_status"), str), "explicit original plan_status required")
    return row


def read_plan(path, shard=0, nshard=1, limit=0):
    need(type(nshard) is int and nshard > 0 and 0 <= shard < nshard and limit >= 0, "invalid shard or limit")
    raw = Path(path).read_bytes()
    rows, seen = [], set()
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = validate_row(strict_json(line))
        key = row["id"].casefold()
        need(key not in seen, "duplicate source ID in full plan")
        seen.add(key)
        rows.append((row, line.encode("utf-8")))
    need(rows, "empty input plan")
    selected = rows[shard::nshard]
    return (selected[:limit] if limit else selected), sha(raw), len(rows)


def normalized_windows(row):
    return [{**copy.deepcopy(w), "window_id": w.get("window_id", f"w{i:05d}")}
            for i, w in enumerate(row["windows"])]


def observation_schema(indices):
    need(2 <= len(indices) <= 16 and len(indices) == len(set(indices)), "schema requires 2..16 distinct actual frames")
    refs = {"type": "array", "maxItems": len(indices), "items": {"type": "integer", "enum": indices}}
    event = {"type": "object", "properties": {
        "id": {"type": "string", "enum": [f"e{i}" for i in range(1, 33)]},
        "description": {"type": "string", "minLength": 1, "maxLength": 1600},
        "parts": {"type": "array", "minItems": 1, "maxItems": len(PARTS),
                  "items": {"type": "string", "enum": list(PARTS)}},
        "frame_indices": {**copy.deepcopy(refs), "minItems": 2}},
        "required": ["id", "description", "parts", "frame_indices"], "additionalProperties": False}
    supports = []
    for part in PARTS:
        supports.append({"type": "object", "properties": {
            "part": {"type": "string", "const": part}, "frame_indices": copy.deepcopy(refs),
            "reason": {"type": "string", "minLength": 1, "maxLength": 600}},
            "required": ["part", "frame_indices", "reason"], "additionalProperties": False})
    return {"type": "object", "properties": {
        "events": {"type": "array", "maxItems": 32, "items": event},
        "part_support": {"type": "array", "prefixItems": supports, "items": False,
                         "minItems": len(PARTS), "maxItems": len(PARTS)},
        "uncertainties": {"type": "array", "maxItems": 20, "items": {"type": "string", "maxLength": 600}},
        "abstain": {"type": "boolean"}},
        "required": ["events", "part_support", "uncertainties", "abstain"], "additionalProperties": False}


def check_indices(indices, allowed, minimum=0):
    need(isinstance(indices, list) and all(type(i) is int and i in allowed for i in indices),
         "frame ref outside actual supplied source frames")
    need(len(indices) == len(set(indices)) and len(indices) >= minimum and len(indices) <= 16,
         "evidence needs distinct actual frames, not duplicate indices")
    need(indices == sorted(indices), "frame evidence must be chronological")
    return set(indices)


def claimed_part_groups(text):
    """Conservative lexical support requirements, not semantic truth scoring."""
    noun_to_kind = {"hand": "hand", "hands": "hand", "finger": "hand", "fingers": "hand",
        "palm": "hand", "palms": "hand", "wrist": "hand", "wrists": "hand", "fist": "hand", "fists": "hand",
        "thumb": "hand", "thumbs": "hand", "foot": "foot", "feet": "foot", "toe": "foot", "toes": "foot",
        "heel": "foot", "heels": "foot", "arm": "arm", "arms": "arm", "elbow": "arm", "elbows": "arm",
        "shoulder": "arm", "shoulders": "arm", "leg": "leg", "legs": "leg", "knee": "leg", "knees": "leg",
        "thigh": "leg", "thighs": "leg"}
    pattern = r"\b(?:(left|right)\s+)?(" + "|".join(sorted(noun_to_kind, key=len, reverse=True)) + r")\b"
    mentions = [(m.group(1), m.group(2), noun_to_kind[m.group(2)]) for m in re.finditer(pattern, text.lower())]
    groups = []
    for side, noun, kind in mentions:
        if side:
            groups.append({side + "_" + kind})
        elif noun in ("hands", "feet", "arms", "legs"):
            explicit = {s + "_" + kind for s, _, k in mentions if s and k == kind}
            if explicit and not re.search(r"\b(?:both|two)\s+" + noun + r"\b", text, re.I):
                groups.append(explicit)
            else:
                groups.extend(({"left_" + kind}, {"right_" + kind}))
        else:
            groups.append({"left_" + kind, "right_" + kind})
    for words, part in ((r"head|neck", "head"), (r"torso|hips?|waist|pelvis", "torso")):
        if re.search(r"\b(?:" + words + r")\b", text, re.I):
            groups.append({part})
    return groups


def motion_counts(text):
    pattern = r"\b(?:once|twice|thrice)\b|\b(?:exactly\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:times|repetitions|cycles)\b"
    return {re.sub(r"^exactly\s+", "", m.group().lower()) for m in re.finditer(pattern, text, re.I)}


def named_fact_id(text, fid):
    return bool(re.search(r"(?<![A-Za-z0-9_:-])" + re.escape(fid) + r"(?![A-Za-z0-9_-])", text))


def missing_fact_refs(value, facts):
    """Only explicit IDs are coverage evidence, not textual similarity."""
    return {item["window_id"]: [f["fact_id"] for f in facts
        if f["window_id"] == item["window_id"] and f["fact_id"] not in item["used_fact_ids"]
        and not any(named_fact_id(issue, f["fact_id"]) for issue in value["issues"])]
        for item in value["windows"]}


def check_supported_rewrite(value, windows, facts, boundaries):
    core.check_rewrite(value, {"windows": windows}, facts, boundaries)
    need(not any(missing_fact_refs(value, facts).values()), "omitted facts require exact issues naming their IDs")
    for item in value["windows"]:
        need(not excluded_semantic_claim(item["caption"]), "rewritten caption introduced excluded non-motion claims")
        local = [f for f in facts if f["window_id"] == item["window_id"]
                 and f["fact_id"] in item["used_fact_ids"]]
        parts = {part for f in local for part in f.get("parts", [])}
        for required in claimed_part_groups(item["caption"]):
            need(bool(required & parts), "rewritten caption names a body part absent from local observed facts")
        allowed_counts = set().union(*(motion_counts(f["raw_text"]) for f in local)) if local else set()
        need(motion_counts(item["caption"]) <= allowed_counts, "rewritten caption invented a motion count")
    return value


def validate_observation(value, manifest, forbidden=None):
    forbidden = forbidden or {}
    need(isinstance(value, dict) and set(value) == {"events", "part_support", "uncertainties", "abstain"},
         "invalid observation JSON keys")
    frames = manifest["frames"]
    allowed = {f["index"] for f in frames}
    start, end = core.window_bounds(manifest["window"])
    need(len(allowed) == len(frames) and all(start <= f["time"] < end for f in frames),
         "manifest evidence outside original half-open window")
    need(isinstance(value["part_support"], list) and len(value["part_support"]) == len(PARTS),
         "exactly ten part_support rows required")
    supports = {}
    for item in value["part_support"]:
        need(isinstance(item, dict) and set(item) == {"part", "frame_indices", "reason"}, "invalid part support keys")
        part = item["part"]
        need(part in PARTS and part not in supports, "unknown or duplicate part support")
        need(isinstance(item["reason"], str) and item["reason"].strip(), "explicit support/unknown reason required")
        supports[part] = check_indices(item["frame_indices"], allowed)
        need(not supports[part].intersection(forbidden.get(part, set())), "raw 2D mask forbids this part/frame support")
    need(isinstance(value["events"], list) and len(value["events"]) <= 32, "invalid events")
    ids = set()
    for event in value["events"]:
        need(isinstance(event, dict) and set(event) == {"id", "description", "parts", "frame_indices"}, "invalid event keys")
        eid = event["id"]
        need(isinstance(eid, str) and eid in {f"e{i}" for i in range(1, 33)} and eid not in ids, "invalid/duplicate event ID")
        ids.add(eid)
        text = event["description"]
        need(isinstance(text, str) and text.strip(), "empty event description")
        parts = event["parts"]
        need(isinstance(parts, list) and parts and all(p in PARTS for p in parts) and len(parts) == len(set(parts)),
             "invalid event required parts")
        refs = check_indices(event["frame_indices"], allowed, 2)
        for part in parts:
            need(len(refs.intersection(supports[part])) >= 2, "each event required part needs its own two visible evidence frames")
            need(not refs.intersection(forbidden.get(part, set())), "raw 2D mask forbids event part/frame")
        for required in claimed_part_groups(text):
            need(bool(required & set(parts)), "event text lacks correct part support, including plural or finger claims")
        for side in ("left", "right"):
            for noun in ("arm", "hand", "leg", "foot"):
                if re.search(r"\b" + side + r"[ -]" + noun + r"\b", text, re.I):
                    need(side + "_" + noun in parts, "description names a body part without correct part support")
        for noun, part in (("head", "head"), ("torso", "torso")):
            if re.search(r"\b" + noun + r"\b", text, re.I):
                need(part in parts, "description names a body part without correct part support")
    need(isinstance(value["uncertainties"], list) and all(isinstance(x, str) for x in value["uncertainties"]), "invalid uncertainties")
    need(type(value["abstain"]) is bool and value["abstain"] == (not value["events"]),
         "no supported events require abstention; supported events cannot abstain")
    return value


def excluded_semantic_claim(text):
    return bool(re.search(r"\b(?:hair|hairstyle|clothes|clothing|outfit|wind|breeze)\b|\bweight[- ]bearing\b|\bbearing\s+weight\b", text, re.I))


def evidence_only_note(text):
    pattern = r"\s*(?:this|the)\s+(?:(?:motion|movement|gesture|action)\s+)?(?:is\s+)?visible\s+(?:from|across|in|at)\s+frames?\s+\d+(?:\s*(?:to|through|-|and)\s*(?:frames?\s*)?\d+)?[.;]?\s*"
    return isinstance(text, str) and bool(re.fullmatch(pattern, text, re.I))


def filter_supported_observation(value, manifest, forbidden=None):
    """Keep independently supported source clauses; never repair the raw response."""
    need(isinstance(value, dict) and set(value) == {"events", "part_support", "uncertainties", "abstain"},
         "invalid raw observation structure")
    need(type(value["abstain"]) is bool and isinstance(value["events"], list), "invalid raw abstention/events")
    need(not value["abstain"] or not value["events"], "raw abstention must not contain motion events")
    shell = copy.deepcopy(value)
    shell.update(events=[], abstain=True)
    validate_observation(shell, manifest, forbidden)
    supports = {p["part"]: set(p["frame_indices"]) for p in value["part_support"]}
    accepted, rejected, evidence_notes, seen = [], [], [], set()
    for event in value["events"]:
        if not isinstance(event, dict) or not isinstance(event.get("id"), str) or event.get("id") in seen:
            rejected.append({"raw_event_id": event.get("id") if isinstance(event, dict) else None,
                             "reason": "invalid or duplicate raw event ID"})
            continue
        seen.add(event.get("id"))
        try:
            need(set(event) == {"id", "description", "parts", "frame_indices"}, "invalid raw event keys")
            need(event["id"] in {f"e{i}" for i in range(1, 33)}, "invalid raw event ID")
            need(isinstance(event["description"], str), "invalid raw description")
            need(isinstance(event["parts"], list) and event["parts"] and
                 all(p in PARTS for p in event["parts"]) and len(set(event["parts"])) == len(event["parts"]),
                 "invalid raw required parts")
            check_indices(event["frame_indices"], {f["index"] for f in manifest["frames"]}, 2)
        except (ValueError, TypeError, KeyError) as error:
            rejected.append({"raw_event_id": event.get("id"), "reason": str(error)})
            continue
        description = event["description"]
        clauses = re.split(r"(?<=[.!?;])\s+", description) if isinstance(description, str) else [description]
        for ci, clause in enumerate(clauses):
            if isinstance(clause, str) and evidence_only_note(clause):
                evidence_notes.append({"raw_event_id": event["id"], "raw_clause_index": ci,
                    "text": clause, "original_description": description,
                    "frame_indices": copy.deepcopy(event["frame_indices"]),
                    "textual_frame_refs_are_unreviewed": True,
                    "stored_as_evidence_not_motion_fact": True})
                continue
            candidate = copy.deepcopy(event)
            candidate["description"] = clause
            try:
                need(isinstance(clause, str) and clause.strip(), "empty source clause")
                need(not excluded_semantic_claim(clause), "appearance/environment/weight-bearing clause is not a motion fact")
                groups = claimed_part_groups(clause)
                if groups:
                    parts = set()
                    refs = set(candidate["frame_indices"])
                    for required in groups:
                        available = required & set(event["parts"])
                        # Head/torso may be an explicit visible reference omitted in model metadata.
                        if not available and len(required) == 1 and required <= {"head", "torso"}:
                            available = {part for part in required if len(refs & supports.get(part, set())) >= 2}
                        need(available, "source clause lacks declared required body-part support")
                        parts.update(available)
                    candidate["parts"] = [part for part in PARTS if part in parts]
                trial = copy.deepcopy(shell)
                trial.update(events=[candidate], abstain=False)
                validate_observation(trial, manifest, forbidden)
            except (ValueError, TypeError, KeyError) as error:
                rejected.append({"raw_event_id": event.get("id"), "raw_clause_index": ci, "reason": str(error)})
            else:
                if len(clauses) > 1 or candidate["parts"] != event["parts"]:
                    candidate["id"] = event["id"] + "_c" + str(ci)
                candidate["source_clause"] = {"raw_event_id": event["id"], "raw_clause_index": ci,
                    "raw_source_clause": clause, "original_description": description,
                    "source_text_is_verbatim_substring": True}
                accepted.append(candidate)
    result = copy.deepcopy(value)
    result.update(events=accepted, abstain=not accepted)
    result["support_filter_audit"] = {"raw_event_count": len(value["events"]),
        "admitted_event_count": len(accepted), "rejected_events": rejected, "evidence_only_notes": evidence_notes,
        "raw_response_changed": False, "remaining_reliable_parts_preserved": True,
        "filter_is_not_semantic_truth_review": True}
    return result


def empty_observation(reason):
    return {"events": [], "part_support": [{"part": p, "frame_indices": [], "reason": reason} for p in PARTS],
            "uncertainties": [reason], "abstain": True}


def probe_actual(video, ffprobe):
    # Same actual-PTS helper as the original music-window producer, once per video.
    from ms_landmarks import probe_video
    info = probe_video(video, ffprobe)
    need(info["media_end"] is not None, "actual video media_end unavailable")
    times = info["time"].tolist()
    return {"times": times, "source_timestamps": [t + info["pts_offset"] for t in times],
            "media_start": 0., "media_end": info["media_end"], "video_fps": info["fps"],
            "video_pts_offset": info["pts_offset"], "media_end_basis": info["media_end_source"],
            "actual_video_pts_sha256": sha(batch.json_bytes(times))}


def check_source_binding(row, source, probe, refs):
    binding = row["binding"]
    need(binding["video_sha256"] == source["files"]["video_path"]["sha256"] and
         binding["audio_sha256"] == source["files"]["audio_path"]["sha256"], "source media SHA differs from plan")
    need(binding["actual_video_pts_sha256"] == probe["actual_video_pts_sha256"], "actual video PTS SHA differs from plan")
    need(binding["decoded_video_frames"] == len(probe["times"]), "actual video frame count differs from plan")
    for key in ("media_start", "media_end", "video_fps"):
        need(math.isclose(binding[key], probe[key], rel_tol=1e-9, abs_tol=1e-8), "actual video binding differs: " + key)
    if "video_pts_offset" in binding:
        need(math.isclose(binding["video_pts_offset"], probe["video_pts_offset"], abs_tol=1e-9), "video PTS origin differs")
    need(binding["input_sha256"] == source["files"]["nlf_path"]["sha256"], "NLF cache SHA binding differs")
    # Manifest/source references have their own independent content bindings.
    core.verify_references(refs)


def load_hard_mask(row, probe):
    ref = row.get("raw2d_mask")
    if ref is None:
        return {}, {"available": False, "part_support_source": "unreviewed_VLM_RGB_candidate_not_pixel_GT"}
    need(isinstance(ref, dict) and valid_sha(ref.get("sha256")), "explicit raw2d mask file/SHA required")
    path = Path(ref["path"]).resolve(strict=True)
    need(batch.sha256(path) == ref["sha256"], "raw2d mask SHA changed")
    mask = load(path)
    need(mask.get("schema") == "bar_raw2d_part_mask_v1", "unsupported explicit raw2d part mask schema")
    for key in ("id", "video_sha256", "actual_video_pts_sha256"):
        expected = row["id"] if key == "id" else row["binding"][key]
        need(mask.get(key) == expected, "raw2d mask source binding differs: " + key)
    forbidden = {}
    for part, frames in mask.get("parts", {}).items():
        need(part in PARTS and isinstance(frames, dict), "invalid raw2d part map")
        denied = set()
        for index, visible in frames.items():
            need(isinstance(index, str) and index.isdigit() and int(index) < len(probe["times"])
                 and type(visible) is bool, "invalid raw2d global frame/bool")
            if not visible:
                denied.add(int(index))
        forbidden[part] = denied
    return forbidden, {"available": True, "path": str(path), "sha256": ref["sha256"],
                       "only_explicit_false_hard_excluded": True, "pixel_visibility_is_not_semantic_GT": True}


def boundary_specs(windows, probe, enabled):
    specs = []
    if not enabled:
        return specs
    for left_w, right_w in zip(windows, windows[1:]):
        time = core.window_bounds(left_w)[1]
        left, right = core.boundary_indices(probe, time)
        ls, le = core.window_bounds(left_w)
        rs, re_ = core.window_bounds(right_w)
        left = [i for i in left if ls <= probe["times"][i] < le]
        right = [i for i in right if rs <= probe["times"][i] < re_]
        specs.append({"id": f"b-{left_w['window_id']}-{right_w['window_id']}",
                      "left_window_id": left_w["window_id"], "right_window_id": right_w["window_id"],
                      "time": time, "left": left, "right": right,
                      "usable": abs(time - rs) <= 1e-8 and len(left) >= 2 and len(right) >= 2})
    return specs


def union_evidence(root, row, source, probe, selected, args, binding):
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load(manifest_path)
        receipt = load(root / "manifest.receipt.json")
        need(receipt["manifest_sha256"] == batch.sha256(manifest_path) and manifest["binding"] == binding,
             "union frame manifest binding/SHA differs")
        need([f["index"] for f in manifest["frames"]] == selected, "union actual frame selection changed")
        for frame in manifest["frames"]:
            path = (root / frame["file"]).resolve(strict=True)
            need(path.is_relative_to(root.resolve()) and batch.sha256(path) == frame["sha256"], "union RGB evidence changed")
        batch.verify_source(source)
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    if selected:
        frame_dir = Path(tempfile.mkdtemp(prefix="rgb-", dir=root))
        # Filter script avoids command-line length limits on longer videos. One decode per source.
        expression = "+".join(f"eq(n\\,{i})" for i in selected)
        filter_path = frame_dir / "select.ffilter"
        core.save_once(filter_path, (f"select={expression},scale=640:640:force_original_aspect_ratio=decrease").encode())
        subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-i", row["manifest_record"]["video_path"],
                        "-map", "0:v:0", "-filter_script:v", str(filter_path), "-fps_mode", "vfr", "-q:v", "3",
                        str(frame_dir / "%06d.jpg")], check=True, capture_output=True)
        images = sorted(frame_dir.glob("*.jpg"))
        need(len(images) == len(selected), "one-decode union RGB frame count differs from actual selection")
        frames = [{"index": index, "time": probe["times"][index], "source_timestamp": probe["source_timestamps"][index],
                   "file": image.relative_to(root).as_posix(), "sha256": batch.sha256(image)}
                  for index, image in zip(selected, images)]
    batch.verify_source(source)
    manifest = {"schema": "bar_union_rgb_evidence_v1", "binding": binding, "source_identity": source,
                "video_sha256": row["binding"]["video_sha256"], "actual_video_pts_sha256": probe["actual_video_pts_sha256"],
                "frames": frames, "one_video_decode_for_all_window_and_boundary_union_frames": True,
                "frame_indices_are_original_global": True, **FLAGS}
    core.save_json(manifest_path, manifest)
    core.save_json(root / "manifest.receipt.json", {"manifest_sha256": batch.sha256(manifest_path)})
    return manifest


def rgb_content(root, frames, header, hand_evidence=None):
    content = [{"type": "text", "text": json.dumps(header, ensure_ascii=False, allow_nan=False)}]
    for frame in frames:
        content.extend([{"type": "text", "text": f"Source frame {frame['index']}, time {frame['time']:.9f}s, original PTS {frame['source_timestamp']:.9f}s."},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                         base64.b64encode((Path(root) / frame["file"]).read_bytes()).decode("ascii")}}])
        if hand_evidence:
            for part in ("left_hand", "right_hand"):
                for crop in hand_evidence[part]["crops"]:
                    if crop["frame_index"] != frame["index"]:
                        continue
                    need(batch.sha256(Path(crop["path"])) == crop["sha256"], "hand ROI JPEG changed before request")
                    metadata = {k: crop[k] for k in ("part", "frame_index", "time", "source_timestamp",
                        "crop_xyxy", "source_frame_size", "crop_pixel_size", "ROI_to_source_affine",
                        "source_to_ROI_affine", "supportpointindices", "hand_support_xyxy", "rawframe_sha256")}
                    metadata.update({"same_source_frame_as_body_context": True, "not_another_time_observation": True,
                        "crop_adds_native_detail": False, "anatomical_side_is_unreviewed": True})
                    content.extend([{"type": "text", "text": json.dumps(metadata, allow_nan=False)},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                            base64.b64encode(Path(crop["path"]).read_bytes()).decode("ascii")}}])
    need(sum(c["type"] == "image_url" for c in content) <= 16, "body plus ROI exceeds 16 image budget")
    return content


def visual_request(args, rgb_root, manifest, forbidden):
    indices = [f["index"] for f in manifest["frames"]]
    denied = {part: sorted(set(indices).intersection(values)) for part, values in forbidden.items()
              if set(indices).intersection(values)}
    return {"model": str(Path(args.model).resolve()), "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": rgb_content(rgb_root, manifest["frames"],
         {"allowed_source_frame_indices": indices, "parts": list(PARTS), "explicit_raw2d_forbidden_part_frames": denied,
          "window_is_only_container_not_new_action": True,
          "ROI_same_frame_is_not_extra_temporal_support": True,
          "hand_detail_unknown_unless_actual_fingers_distinguishable_in_two_frames": True,
          "hand_ROI_support": {p: manifest["hand_roi_evidence"][p]["frame_indices"] for p in ("left_hand", "right_hand")}
              if manifest.get("hand_roi_evidence") else {},
          "ROI_proposal_is_not_hand_shape_or_anatomical_side_truth": True}, manifest.get("hand_roi_evidence"))}],
        "sampling": {"temperature": 0., "seed": 0, "max_tokens": args.visual_max_tokens},
        "json_schema": observation_schema(indices), "prompt_sha256": sha(SYSTEM.encode())}


def attempt_root(root, retry_failed):
    first, second = Path(root) / "attempt-0", Path(root) / "attempt-1"
    if second.exists():
        return second
    if retry_failed and (first / "receipt.json").exists() and load(first / "receipt.json")["status"] == "failed":
        return second
    return first


def persist_completion(root, request, output, completion, llm):
    raw = completion.text.encode("utf-8")
    core.save_once(root / "response.raw.txt", raw)
    tokens = list(completion.token_ids)
    eos = llm.get_tokenizer().eos_token_id
    core.save_json(root / "generation.json", {
        "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
        "prompt_tokens": len(output.prompt_token_ids or []), "completion_tokens": len(tokens),
        "max_tokens": request["sampling"]["max_tokens"], "eos_token_id": eos,
        "last_output_token_id": tokens[-1] if tokens else None,
        "last_output_token_is_eos": bool(tokens) and tokens[-1] == eos,
        "eos_is_not_required_in_returned_token_ids": True, "model_call_performed": True,
        "raw_response_sha256": sha(raw), "request_sha256": batch.sha256(root / "request.json"),
        "structured_output_schema_sha256": batch.digest(request["json_schema"]),
        "model_identity_sha256": getattr(llm, "_ms_bar_model_sha256", None)})


def transactions(items, runtime, args):
    results, pending = {}, []
    for item in items:
        root = item["root"]
        result = core.transaction(root, item["request"], None, None, item["validator"], item["binding"])
        results[str(root)] = result
        if result["status"] == "prepared" and args.stage == "run":
            pending.append(item)
    for offset in range(0, len(pending), args.batch_size):
        group = pending[offset:offset + args.batch_size]
        try:
            llm, sampling_cls = runtime()
        except Exception as error:
            raise RuntimeError(f"shared model load failed; not retrying per window: {error}") from error
        with ExitStack() as locks:
            for item in group:
                locks.enter_context(batch.bundle_lock(item["root"]))
                need(core.needs_model(item["root"]), "observation became active/terminal before dispatch")
            sampling = [core.structured_sampling(sampling_cls, item["request"]["json_schema"],
                        item["request"]["sampling"]["max_tokens"]) for item in group]
            try:
                outputs = llm.chat([item["request"]["messages"] for item in group], sampling, use_tqdm=False)
                need(len(outputs) == len(group) and all(len(o.outputs) == 1 for o in outputs), "expected one completion per batched request")
                for item, output in zip(group, outputs):
                    persist_completion(item["root"], item["request"], output, output.outputs[0], llm)
            except Exception as error:
                # Persist dispatch failures too; a resume never silently resubmits them.
                for item in group:
                    if not (item["root"] / "response.raw.txt").exists():
                        core.save_once(item["root"] / "response.raw.txt", b"")
                        core.save_json(item["root"] / "dispatch_error.json", {"error": f"{type(error).__name__}: {error}"})
        for item in group:
            results[str(item["root"])] = core.transaction(item["root"], item["request"], None, None,
                                                         item["validator"], item["binding"])
    return results


def facts_from_observations(windows, observations):
    facts = []
    for window, observation in zip(windows, observations):
        if observation["status"] not in ("candidate", "abstain"):
            continue
        for event in observation["json"]["events"]:
            fid = window["window_id"] + ":" + event["id"]
            facts.append({"fact_id": fid, "id": fid, "window_id": window["window_id"],
                          "text": core.neutral_fact_text(event["description"]), "raw_text": event["description"],
                          "source_clause": copy.deepcopy(event["source_clause"]),
                          "parts": copy.deepcopy(event["parts"]), "frame_indices": copy.deepcopy(event["frame_indices"]),
                          "frame_bounds_are_not_precise_event_times": True, "source": OUTPUT_SCHEMA,
                          "ai_weak_candidate": True, **FLAGS})
    return facts


BAR_REWRITE_RULES = """
Every window's used_fact_ids is the explicit coverage declaration for that
window, including ALL IDs of compatible facts merged into one phrase. It is
not enough that a motion happens to appear in the caption without its ID.
Use the source-ordered fact groups as a checklist, retaining sustained motion,
minor adjustments and real states as well as large movements. Do not repeat
pure frame-reference notes as caption motion. Never cite a fact only to pass
coverage: it must be expressed in the caption, or omitted with an exact fact
ID and an honest reason in issues. Do not fill every ID mechanically.
This is source fidelity auditing, not human semantic verification.
"""


def bar_rewrite_request(args, supplied, boundaries):
    supplied = copy.deepcopy(supplied)
    supplied["ordered_fact_groups"] = [{"window_id": w["window_id"],
        "facts_in_source_order": [{"fact_id": f["fact_id"], "text": f["text"],
            "source_clause": f["source_clause"]} for f in supplied["facts"]
            if f["window_id"] == w["window_id"]],
        "merged_compatible_facts_require_all_explicit_used_fact_ids": True}
        for w in supplied["windows"]]
    request = core.rewrite_request(args.model, supplied, boundaries, args.text_max_tokens)
    system = core.REWRITE_SYSTEM + BAR_REWRITE_RULES
    request["messages"][0]["content"] = system
    request["prompt_sha256"] = sha(system.encode())
    for item in request["json_schema"]["properties"]["windows"]["prefixItems"]:
        wid = item["properties"]["window_id"]["const"]
        for key, ids in (("used_fact_ids", [f["fact_id"] for f in supplied["facts"] if f["window_id"] == wid]),
                        ("used_boundary_ids", [b["boundary_id"] for b in boundaries if b["relation"] != "unknown"
                         and wid in (b["left_window_id"], b["right_window_id"])])):
            item["properties"][key] = {"type": "array", "maxItems": len(ids), "uniqueItems": True,
                "items": {"type": "string", "enum": ids} if ids else {"type": "string"}}
    return request


def rewrite_with_reference_repair(item, runtime, args, facts):
    first = transactions([item], runtime, args)[str(item["root"])]
    first = {**first, "attempt_path": str(item["root"])}
    if first["status"] != "failed" or args.rewrite_reference_repair != "on":
        return first
    raw_path = item["root"] / "response.raw.txt"
    generation_path = item["root"] / "generation.json"
    try:
        core.validate_generation(item["root"], item["request"])
        value = parse_response(raw_path.read_bytes())
        missing = {wid: refs for wid, refs in missing_fact_refs(value, facts).items() if refs}
        need(missing, "not an explicit missing-reference failure")
        supplied = json.loads(item["request"]["messages"][1]["content"])
        need(len(value["windows"]) == len(supplied["windows"]), "invalid reference repair window count")
        # Check EVERY window's structure even if the ordinary validator stopped
        # at an earlier missing ID. No IDs or issues are synthesized here.
        for original, output in zip(supplied["windows"], value["windows"]):
            referenced = [f for f in facts if f["fact_id"] in output["used_fact_ids"]]
            core.check_rewrite({"windows": [output], "issues": value["issues"]},
                {"windows": [original]}, referenced, supplied.get("boundary_relations", []))
            need(not excluded_semantic_claim(output["caption"]), "excluded caption claim is not a reference failure")
        # Keep other validation failures terminal, including malformed JSON and truncation.
        try:
            item["validator"](value)
        except ValueError as error:
            need(str(error) in ("omitted facts require issues naming their IDs",
                                "omitted facts require exact issues naming their IDs"),
                 "not an explicit missing-reference failure")
        else:
            return first
    except (ValueError, TypeError, KeyError, OSError):
        return first
    repair_root = item["root"] / "reference-repair"
    request = copy.deepcopy(item["request"])
    request["messages"].extend([
        {"role": "assistant", "content": raw_path.read_text()},
        {"role": "user", "content": json.dumps({"missing_fact_ids_by_window": missing,
            "instruction": "Reconsider the COMPLETE prior response against ALL supplied facts and boundaries. "
                "For each missing ID, either express its actual source motion and cite it in its own window, "
                "include its ID when merging a compatible expressed fact, or state the exact ID and honest "
                "omission reason in issues. Do not copy all IDs mechanically. Keep all original windows. "
                "Return one complete JSON, not a patch. No new observations are supplied."})}])
    request["reference_repair"] = {"original_attempt_path": str(item["root"]),
        "original_request_sha256": batch.sha256(item["root"] / "request.json"),
        "original_response_sha256": batch.sha256(raw_path),
        "original_generation_sha256": batch.sha256(generation_path),
        "missing_fact_ids_by_window": missing, "maximum_additional_model_requests": 1,
        "original_context_preserved": True, "automatic_ID_fill_performed": False}
    repair_item = {**item, "root": repair_root, "request": request,
        "binding": {**item["binding"], "reference_repair_sha256": batch.digest(request["reference_repair"])}}
    final = transactions([repair_item], runtime, args)[str(repair_root)]
    return {**final, "attempt_path": str(repair_root), "original_attempt": first,
            "reference_repair": request["reference_repair"]}


def model_reference(args):
    if not args.model_reference:
        return batch.model_identity(args.model)
    reference = load(args.model_reference)
    identity = reference.get("model", reference.get("model_identity", reference))
    need(isinstance(identity, dict) and {"root", "files", "sha256"} <= set(identity), "model reference lacks verified full-content identity")
    need(Path(identity["root"]).resolve() == Path(args.model).resolve(), "model reference has a different root")
    need(identity.get("full_content_hash_computed_once_per_process") is True, "model reference does not attest full-content hash")
    content = {name: {"path": v["path"], "sha256": v["sha256"]} for name, v in identity["files"].items()}
    need(batch.digest({"root": identity["root"], "files": content}) == identity["sha256"], "model reference content digest differs")
    model_root = Path(args.model).resolve()
    files = {str(path.relative_to(model_root)): path for path in batch.model_files(model_root)}
    need(set(files) == set(identity["files"]), "local model inventory differs from verified source")
    for name, path in files.items():
        need(str(path.resolve()) == identity["files"][name]["path"]
             and path.stat().st_size == identity["files"][name]["stat"]["size"], "local staged model path/size differs")
    for name in ("config.json", "model.safetensors.index.json"):
        need(batch.sha256(files[name]) == identity["files"][name]["sha256"], "local model config/index changed")
    marker_path = Path(str(model_root) + ".stage_receipt.json")
    if marker_path.exists():
        marker = load(marker_path)
        need(marker.get("complete") is True and marker.get("model_identity_sha256") == identity["sha256"],
             "local staged full-content receipt differs")
    local = copy.deepcopy(identity)
    for name, path in files.items():
        local["files"][name]["stat"] = batch.snapshot(path)
    local["source_reference_sha256"] = batch.sha256(Path(args.model_reference))
    local["local_stats_refreshed_from_verified_staged_identity"] = True
    batch.verify_current(local, model=True)
    return local


def code_identity():
    paths = [Path(__file__), Path(core.__file__), Path(batch.__file__), Path(batch.__file__).with_name("ms_landmarks.py"),
             Path(batch.__file__).with_name("ms_visual_observe.py"), Path(batch.__file__).with_name("ms_optimize.py"),
             Path(batch.__file__).with_name("ms_hand_roi.py"), Path(batch.__file__).with_name("dwpose_full_extract.py")]
    return {str(p): batch.sha256(p) for p in paths}


def runtime_loader(args, out, code):
    loaded = []
    identity_path = out / "model.identity.json"
    prior_model = load(identity_path) if identity_path.exists() else None
    if prior_model is not None:
        need(Path(prior_model["root"]).resolve() == Path(args.model).resolve(), "resume model root differs")
        batch.verify_current(prior_model, model=True)
    def runtime():
        if loaded:
            return loaded[0]
        model = model_reference(args)
        need(code == code_identity(), "code changed before loading")
        batch.verify_current(model, model=True)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
        from vllm import LLM, SamplingParams
        llm = LLM(model=model["root"], tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
                  max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization,
                  limit_mm_per_prompt={"image": 16, "video": 0}, mm_processor_kwargs={"max_pixels": args.max_pixels},
                  enable_prefix_caching=True, trust_remote_code=False, seed=0)
        core.save_json(out / "model.identity.json", model)
        llm._ms_bar_model_sha256 = model["sha256"]
        # Runtime receipts are one per process, not overwritten on a later resume.
        receipt_dir = out / "loads"
        receipt_dir.mkdir(exist_ok=True)
        receipt = {"schema": OUTPUT_SCHEMA, "model_identity_sha256": model["sha256"], "code_sha256": code,
                   "one_model_load_for_all_sources_windows_boundaries_and_rewrite": True,
                   "verified_reference_current_stat_used": bool(args.model_reference),
                   "tensor_parallel_size": args.tp, "batch_size": args.batch_size,
                   "max_num_seqs": args.max_num_seqs, **FLAGS}
        load_path = receipt_dir / (str(os.getpid()) + ".json")
        core.save_json(load_path, receipt)
        batch.atomic_write(out / "load.receipt.json", batch.json_bytes({**receipt,
            "actual_load_ref": {"path": str(load_path), "sha256": batch.sha256(load_path)},
            "model_load_completed": True}))
        loaded.append((llm, SamplingParams))
        return loaded[0]
    return runtime


def source_references(row, cache):
    refs = []
    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                key = (value["path"], value["sha256"])
                if key not in cache:
                    identity = batch.file_identity(Path(value["path"]))
                    need(identity["sha256"] == value["sha256"], "source ref SHA mismatch")
                    cache[key] = identity
                identity = cache[key]
                need(batch.snapshot(Path(identity["path"])) == identity["stat"], "cached source ref changed")
                refs.append(identity)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(row["source_refs"])
    need(refs, "source_refs need file/SHA binding")
    ref = row["source_refs"]["manifest"]
    manifest_key = ("manifest_records", ref["path"], ref["sha256"])
    if manifest_key not in cache:
        records = {}
        for line in Path(ref["path"]).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = strict_json(line)
            need(isinstance(record, dict) and isinstance(record.get("id"), str), "invalid source manifest record")
            key = record["id"].casefold()
            need(key not in records, "duplicate input manifest ID")
            records[key] = record
        cache[manifest_key] = records
    original = cache[manifest_key].get(row["id"].casefold())
    need(original is not None and all(original.get(k) == row["manifest_record"][k] for k in batch.FIELDS),
         "manifest_record does not match bound input manifest source")
    return refs


def load_hand_roi(row, probe, roi_root):
    if not roi_root:
        return None, {"available": False, "reason": "not_requested"}
    root = Path(roi_root).resolve(strict=True)
    path = root if root.is_file() else root / row["id"] / "roi_manifest.json"
    if not path.exists():
        return None, {"available": False, "reason": "source_roi_not_extracted", "base_RGB_not_blocked": True}
    import ms_hand_roi as roi
    source = load(path.with_name("source_manifest.json"))
    manifest = roi.validate_completed(path.parent, source)
    need(manifest["id"] == row["id"] and manifest["video_sha256"] == row["binding"]["video_sha256"],
         "hand ROI source ID/video SHA differs")
    rp = manifest["probe"]
    need(rp["decoded_video_frames"] == len(probe["times"]) and
         len(rp["frame_times"]) == len(probe["times"]) and
         len(rp["actual_video_pts"]) == len(probe["source_timestamps"]), "hand ROI actual PTS length differs")
    need(all(math.isclose(a, b, rel_tol=0, abs_tol=1e-8) for a, b in zip(rp["frame_times"], probe["times"])) and
         all(math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
             for a, b in zip(rp["actual_video_pts"], probe["source_timestamps"])), "hand ROI actual PTS differs")
    for frame in manifest["selected_frames"]:
        i = frame["frame_index"]
        need(type(i) is int and 0 <= i < len(probe["times"]) and
             math.isclose(frame["time"], probe["times"][i], abs_tol=1e-8) and
             math.isclose(frame["source_timestamp"], probe["source_timestamps"][i], abs_tol=1e-8),
             "hand ROI selected global frame/PTS differs")
    width, height = rp["width"], rp["height"]
    for crop in manifest["crops"]:
        crop_path = Path(crop["path"]).resolve(strict=True)
        need(crop_path.is_relative_to(path.parent.resolve()), "hand ROI crop escapes source ROI directory")
        x0, y0, x1, y1 = crop["crop_xyxy"]
        need(all(type(v) is int for v in (x0, y0, x1, y1)) and
             0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and
             crop["source_frame_size"] == [width, height] and
             crop["crop_pixel_size"] == [x1 - x0, y1 - y0], "hand ROI native crop bounds/size differs")
        need(crop["ROI_to_source_affine"] == [[1., 0., x0], [0., 1., y0]] and
             crop["source_to_ROI_affine"] == [[1., 0., -x0], [0., 1., -y0]] and
             all(crop.get(k) is False for k in ("resized", "rotated", "mirrored", "crop_adds_native_detail")),
             "hand ROI nonnative crop transform")
    return manifest, {"available": True, "path": str(path), "sha256": batch.sha256(path),
        "actual_PTS_values_compared_not_cross_schema_hashes": True,
        "source_manifest_sha256": batch.sha256(path.with_name("source_manifest.json")),
        "extract_receipt_sha256": batch.sha256(path.with_name("extract_receipt.json")),
        "semantic_approval": False}


def select_window_roi(manifest, window, selected, times):
    if manifest is None:
        return selected, None
    import ms_hand_roi as roi
    start, end = core.window_bounds(window)
    evidence = roi.select_roi_evidence(manifest, start, end, max_per_hand=3)
    required = set(evidence["distinct_source_frame_indices"])
    if not required:
        return selected, evidence
    # Every crop gets SAME-frame body context; the two hands never count twice in time.
    body = set(required)
    available = [i for i, t in enumerate(times) if start <= t < end]
    candidates = sorted(set(selected) | {available[0], available[-1]})
    for k in range(len(candidates)):
        if len(body) >= 8:
            break
        # Include endpoints, then fill evenly without increasing image count.
        pos = 0 if k == 0 else len(candidates) - 1 if k == 1 else round((k - 1) * (len(candidates) - 1) / max(1, len(candidates) - 2))
        body.add(candidates[pos])
    for i in candidates:
        if len(body) >= min(8, len(set(candidates) | required)):
            break
        body.add(i)
    need(2 <= len(body) <= 8 and required <= body, "ROI body-context image budget exceeded")
    evidence["image_budget"] = {"body": len(body), "ROI": sum(len(evidence[p]["crops"]) for p in ("left_hand", "right_hand")),
        "maximum_total": 16, "same_frame_body_context_required": True, "crop_adds_native_detail": False}
    return sorted(body), evidence


def select_window_frames(times, start, end, sample_fps):
    available = [i for i, time in enumerate(times) if start <= time < end]
    native_short = end - start < 2 / sample_fps
    if len(available) < 2 or native_short:
        selected = available
        if len(selected) > 16:
            selected = [available[round(k * (len(available) - 1) / 15)] for k in range(16)]
    else:
        selected = batch.select_frames(times, start, end, sample_fps, 16)
    return selected, {"requested_fps": sample_fps, "max_frames": 16,
                      "native_short_window_fallback": native_short, "native_available_frame_count": len(available),
                      "half_open_window": True, "indices_are_original_global": True}


def prepare_source(row, raw_line, root, args, binding, ref_cache):
    core.save_once(root / "plan.line.raw.json", raw_line)
    core.save_json(root / "plan.json", row)
    core.save_json(root / "binding.json", binding)
    source = batch.source_identity(row["manifest_record"])
    refs = source_references(row, ref_cache)
    probe = probe_actual(row["manifest_record"]["video_path"], args.ffprobe)
    check_source_binding(row, source, probe, refs)
    core.save_json(root / "source.identity.json", source)
    core.save_json(root / "source.refs.json", refs)
    core.save_json(root / "actual_pts.json", probe)
    forbidden, mask_meta = load_hard_mask(row, probe)
    windows = normalized_windows(row)
    roi_manifest, roi_meta = load_hand_roi(row, probe, args.hand_roi_root)
    selections, sampling_policies, roi_evidence = [], [], []
    for window in windows:
        start, end = core.window_bounds(window)
        selected, policy = select_window_frames(probe["times"], start, end, args.sample_fps)
        selected, hand_evidence = select_window_roi(roi_manifest, window, selected, probe["times"])
        selections.append(selected)
        if hand_evidence and hand_evidence.get("image_budget"):
            policy = {**policy, "body_budget_reduced_for_same_frame_ROI": True, "maximum_body_images": 8,
                      "actual_body_frame_count": len(selected), "maximum_total_body_plus_ROI_images": 16}
        sampling_policies.append(policy)
        roi_evidence.append(hand_evidence)
    specs = boundary_specs(windows, probe, args.boundary_review == "on")
    union = sorted({i for selected in selections for i in selected} |
                   {i for spec in specs if spec["usable"] for i in spec["left"] + spec["right"]})
    evidence = union_evidence(root / "rgb", row, source, probe, union, args, binding)
    by_index = {f["index"]: f for f in evidence["frames"]}
    prepared = {"schema": OUTPUT_SCHEMA, "binding": binding, "source_identity": source, "source_refs": refs,
                "probe": probe, "original_windows": row["windows"], "window_manifest": row["window_manifest"],
                "windows": [{"window": w, "frames": [by_index[i] for i in selected], "sampling_policy": policy,
                             "hand_roi_evidence": hand_evidence}
                            for w, selected, policy, hand_evidence in zip(windows, selections, sampling_policies, roi_evidence)],
                "hand_roi": roi_meta,
                "boundaries": specs, "raw2d_mask": mask_meta, "forbidden": {p: sorted(v) for p, v in forbidden.items()},
                "union_manifest_sha256": batch.sha256(root / "rgb" / "manifest.json"),
                "VLM_observed_full_continuous_video": False, "window_edges_are_event_edges": False, **FLAGS}
    core.save_json(root / "prepared.json", prepared)
    core.save_json(root / "prepared.receipt.json", {"prepared_sha256": batch.sha256(root / "prepared.json"),
                                                   "binding": binding})
    return prepared


def verify_prepared(root, row, raw_line, binding):
    need(load(root / "binding.json") == binding and load(root / "plan.json") == row
         and (root / "plan.line.raw.json").read_bytes() == raw_line, "resume source plan/binding mismatch")
    prepared = load(root / "prepared.json")
    receipt = load(root / "prepared.receipt.json")
    need(receipt["prepared_sha256"] == batch.sha256(root / "prepared.json") and receipt["binding"] == binding
         and prepared["binding"] == binding, "prepared receipt/source binding differs")
    batch.verify_source(prepared["source_identity"])
    core.verify_references(prepared["source_refs"])
    if prepared["raw2d_mask"].get("available"):
        need(batch.sha256(Path(prepared["raw2d_mask"]["path"])) == prepared["raw2d_mask"]["sha256"], "raw2d mask changed on resume")
    roi_meta = prepared["hand_roi"]
    if roi_meta["available"]:
        roi_path = Path(roi_meta["path"])
        for path, key in ((roi_path, "sha256"), (roi_path.with_name("source_manifest.json"), "source_manifest_sha256"),
                          (roi_path.with_name("extract_receipt.json"), "extract_receipt_sha256")):
            need(batch.sha256(path) == roi_meta[key], "hand ROI source receipt changed on resume")
        load_hand_roi(row, prepared["probe"], str(roi_path))
    need(batch.sha256(root / "rgb" / "manifest.json") == prepared["union_manifest_sha256"], "union manifest changed")
    union = load(root / "rgb" / "manifest.json")
    need(union["binding"] == binding, "union binding changed")
    for frame in union["frames"]:
        path = (root / "rgb" / frame["file"]).resolve(strict=True)
        need(path.is_relative_to((root / "rgb").resolve()) and batch.sha256(path) == frame["sha256"], "RGB artifact changed")
    return prepared


def gate_boundary_support(checked, spec, windows, observations):
    lookup = {w["window_id"]: o for w, o in zip(windows, observations)}
    left_obs = lookup.get(spec["left_window_id"], {})
    right_obs = lookup.get(spec["right_window_id"], {})
    def supports(observation, part):
        return next((set(p["frame_indices"]) for p in observation.get("json", {}).get("part_support", [])
                     if p["part"] == part), set()) if observation.get("json") else set()
    result = copy.deepcopy(checked)
    for relation in result["relations"]:
        if relation["relation"] == "unknown":
            continue
        part = relation["part"]
        left = supports(left_obs, part) & set(relation["left_frame_indices"])
        right = supports(right_obs, part) & set(relation["right_frame_indices"])
        if len(left) < 2 or len(right) < 2:
            relation["raw_relation"] = relation["relation"]
            relation["relation"] = "unknown"
            relation["left_frame_indices"] = []
            relation["right_frame_indices"] = []
            relation["support_gate_reason"] = "core window body-part evidence does not support two frames on each side"
        else:
            relation["core_supported_left_frames"] = sorted(left)
            relation["core_supported_right_frames"] = sorted(right)
    return result


def boundary_item(spec, prepared, root, args, binding, observations=None):
    metadata = {k: spec[k] for k in ("id", "left_window_id", "right_window_id", "time")}
    if not spec["usable"]:
        return {"status": "unknown", "json": {"relations": [{"part": p, "relation": "unknown",
            "left_frame_indices": [], "right_frame_indices": [], **metadata,
            "boundary_id": spec["id"] + ":" + p, "ai_candidate": True, **FLAGS} for p in core.PARTS],
            "issues": ["nonadjacent or fewer than two actual frames per side"]}, **FLAGS}
    union = load(root / "rgb" / "manifest.json")
    by_index = {f["index"]: f for f in union["frames"]}
    left, right = spec["left"], spec["right"]
    frames = [by_index[i] for i in left + right]
    request = {"model": str(Path(args.model).resolve()), "messages": [
        {"role": "system", "content": core.BOUNDARY_SYSTEM}, {"role": "user", "content": rgb_content(root / "rgb", frames,
         {"boundary_time": spec["time"], "left_frame_indices": left, "right_frame_indices": right})}],
        "sampling": {"temperature": 0., "seed": 0, "max_tokens": 1200}, "json_schema": core.boundary_schema(left, right),
        "prompt_sha256": sha(core.BOUNDARY_SYSTEM.encode())}
    forbidden = {p: set(indices) for p, indices in prepared["forbidden"].items()}
    def validator(value):
        checked = core.check_boundary(value, left, right, metadata)
        for relation in checked["relations"]:
            if relation["relation"] != "unknown":
                need(not set(relation["left_frame_indices"] + relation["right_frame_indices"]).intersection(
                    forbidden.get(relation["part"], set())), "boundary raw2d part/frame forbidden")
        windows = [m["window"] for m in prepared["windows"]]
        return gate_boundary_support(checked, spec, windows, observations or [{} for _ in windows])
    return {"root": attempt_root(root / "boundaries" / spec["id"], args.retry_failed), "request": request,
            "validator": validator, "binding": {**binding, "boundary": spec}}


def process_source(row, raw_line, root, args, binding, runtime, ref_cache):
    if (root / "prepared.json").exists():
        prepared = verify_prepared(root, row, raw_line, binding)
    elif args.stage == "export":
        return {"id": row["id"], "status": "unprocessed", "binding": binding, **FLAGS}
    else:
        prepared = prepare_source(row, raw_line, root, args, binding, ref_cache)
    forbidden = {p: set(v) for p, v in prepared["forbidden"].items()}
    windows = [m["window"] for m in prepared["windows"]]
    items, positions, observations = [], [], [None] * len(windows)
    for index, manifest in enumerate(prepared["windows"]):
        wid = manifest["window"]["window_id"]
        wroot = root / "windows" / wid
        core.save_json(wroot / "manifest.json", {**manifest, "binding": binding, "union_manifest_sha256": prepared["union_manifest_sha256"],
                                                "raw2d_mask": prepared["raw2d_mask"], **FLAGS})
        if len(manifest["frames"]) < 2:
            observations[index] = {"status": "abstain", "json": empty_observation("fewer than two actual frames in original half-open window"),
                                   "model_call_performed": False, **FLAGS}
            core.save_json(wroot / "evidence_abstention.json", observations[index])
            continue
        request = visual_request(args, root / "rgb", manifest, forbidden)
        attempt = attempt_root(wroot, args.retry_failed)
        positions.append((index, attempt))
        items.append({"root": attempt, "request": request,
                      "validator": lambda value, m=manifest: filter_supported_observation(value, m, forbidden),
                      "binding": {**binding, "window": manifest["window"], "frames_sha256": batch.digest(manifest["frames"])}})
    if args.stage == "export":
        for item in items:
            need((item["root"] / "request.json").exists(), "export missing observation request")
    results = transactions(items, runtime, args)
    for index, attempt in positions:
        observation = results[str(attempt)]
        if observation["status"] == "candidate" and observation["json"]["abstain"]:
            observation = {**observation, "status": "abstain"}
        observations[index] = {**observation, "attempt_path": str(attempt)}
    facts = facts_from_observations(windows, observations)
    baseline = {"windows": [{"window_id": w["window_id"], "caption": " ".join(f["raw_text"] for f in facts if f["window_id"] == w["window_id"]),
                              "used_fact_ids": [f["fact_id"] for f in facts if f["window_id"] == w["window_id"]], "used_boundary_ids": []}
                             for w in windows], "issues": []}
    pending = any(o["status"] == "prepared" for o in observations)
    boundaries, boundary_results, bitems = [], [], []
    if not pending and facts:
        for spec in prepared["boundaries"]:
            item = boundary_item(spec, prepared, root, args, binding, observations)
            if "root" in item:
                bitems.append(item)
            else:
                boundary_results.append(item)
        boundary_results.extend(transactions(bitems, runtime, args).values())
        boundaries = [r for result in boundary_results if result["status"] in ("candidate", "unknown")
                      for r in result["json"]["relations"]]
        pending = any(result["status"] == "prepared" for result in boundary_results)
    if pending:
        final = {"status": "prepared", **FLAGS}
    elif not facts:
        final = {"status": "abstain", "json": baseline, "model_call_performed": False, **FLAGS}
    else:
        supplied = {"windows": [{"window_id": w["window_id"], "source_status": o["status"],
                                  "source_status_is_audit_only": True,
                                  "admitted_fact_count": sum(f["window_id"] == w["window_id"] for f in facts)}
                                 for w, o in zip(windows, observations)],
                    "facts": [{k: copy.deepcopy(v) for k, v in f.items() if k != "raw_text"} for f in facts]}
        request = bar_rewrite_request(args, supplied, boundaries)
        attempt = attempt_root(root / "rewrite", args.retry_failed)
        item = {"root": attempt, "request": request, "validator": lambda v: check_supported_rewrite(v, windows, facts, boundaries),
                "binding": {**binding, "facts_sha256": batch.digest(facts), "boundaries_sha256": batch.digest(boundaries)}}
        final = rewrite_with_reference_repair(item, runtime, args, facts)
    pending = pending or final["status"] == "prepared"
    failed = any(o["status"] == "failed" for o in observations) or final["status"] == "failed"
    rewrite_generation = Path(final["attempt_path"]) / "generation.json" if "attempt_path" in final else None
    rewrite_calls = int(bool(rewrite_generation and rewrite_generation.exists() and load(rewrite_generation).get("model_call_performed")))
    original_rewrite = final.get("original_attempt", {}).get("attempt_path")
    if original_rewrite:
        original_generation = Path(original_rewrite) / "generation.json"
        rewrite_calls += int(original_generation.exists() and bool(load(original_generation).get("model_call_performed")))
    model_identity_path = root.parent.parent / "model.identity.json"
    status = "prepared" if pending else "failed" if failed else "candidate" if facts else "abstain"
    result = {"schema": OUTPUT_SCHEMA, "source": OUTPUT_SCHEMA, "id": row["id"], "status": status,
              "binding": binding, "manifest_record": row["manifest_record"], "source_refs": row["source_refs"],
              "original_window_manifest": row["window_manifest"], "original_windows": row["windows"], "plan_status": row["plan_status"],
              "window_observations": [{"window": w, **o} for w, o in zip(windows, observations)],
              "facts": facts, "visual_baseline": baseline,
              "visual_baseline_timed_caption": core.timed_text({"windows": windows}, baseline["windows"]),
              "boundary_observations": boundary_results, "final": final,
              "final_timed_caption": core.timed_text({"windows": windows}, final["json"]["windows"]) if "json" in final else "",
              "coverage": {"window_count": len(windows), "window_status_counts": dict(Counter(o["status"] for o in observations)),
                "part_unknown_counts": dict(Counter(p["part"] for o in observations if "json" in o
                                          for p in o["json"]["part_support"] if len(p["frame_indices"]) < 2)),
                "admitted_fact_count": len(facts), "full_window_caption": bool(facts) and status == "candidate"
                    and all(w["caption"].strip() for w in final.get("json", {}).get("windows", []))
                    and len(final.get("json", {}).get("windows", [])) == len(windows),
                "omitted_fact_count": len(facts) - sum(len(w["used_fact_ids"]) for w in final.get("json", {}).get("windows", [])),
                "boundary_failure_count": sum(b["status"] == "failed" for b in boundary_results)},
              "hand_roi": prepared["hand_roi"],
              "rewrite_coverage_is_explicit_ID_audit_not_semantic_truth": True,
              "raw2d_mask": prepared["raw2d_mask"], "raw_rgb_manifest": {"path": str(root / "rgb" / "manifest.json"),
              "sha256": prepared["union_manifest_sha256"]}, "part_support_is_AI_candidate_not_pixel_GT": True,
              "MS_rich_music_genre_supplied_to_visual_model": False,
              "raw_facts_modified": False, "rewrite_fulltext_calls_per_attempt": rewrite_calls,
              "model_identity_ref": {"path": str(model_identity_path), "sha256": batch.sha256(model_identity_path)}
                  if model_identity_path.exists() else None, **FLAGS}
    # Stage snapshots are append-only; an enlarged/resumed run has a new content key.
    result_path = root / "results" / (args.stage + "-" + batch.digest(result) + ".json")
    core.save_json(result_path, result)
    result["result_ref"] = {"path": str(result_path), "sha256": batch.sha256(result_path)}
    return result


def params(args, code):
    return {"model_path": str(Path(args.model).resolve()), "sample_fps": args.sample_fps, "max_frames": 16,
            "tp": args.tp, "batch_size": args.batch_size, "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len, "max_pixels": args.max_pixels,
            "gpu_memory_utilization": args.gpu_memory_utilization, "visual_max_tokens": args.visual_max_tokens,
            "text_max_tokens": args.text_max_tokens, "boundary_review": args.boundary_review,
            "rewrite_reference_repair": args.rewrite_reference_repair,
            "hand_roi_root": str(Path(args.hand_roi_root).resolve()) if args.hand_roi_root else None,
            "rewrite_system_sha256": sha((core.REWRITE_SYSTEM + BAR_REWRITE_RULES).encode()),
            "system_sha256": sha(SYSTEM.encode()), "code_sha256": code,
            "ffmpeg": args.ffmpeg, "ffprobe": args.ffprobe}


def summarize(results, stage, total, selected, plan_sha):
    return {"schema": OUTPUT_SCHEMA, "stage": stage, "plan_sha256": plan_sha, "total_source_count": total,
            "selected_source_count": selected, "processed_source_count": sum(r["status"] != "unprocessed" for r in results),
            "complete_caption_count": sum(r.get("coverage", {}).get("full_window_caption", False) for r in results),
            "nonempty_weak_caption_count": sum(bool(r.get("final_timed_caption")) and r["status"] == "candidate" for r in results),
            "abstain_source_count": sum(r["status"] == "abstain" for r in results),
            "failed_source_count": sum(r["status"] == "failed" for r in results),
            "prepared_source_count": sum(r["status"] == "prepared" for r in results),
            "part_unknown_counts": dict(sum((Counter(r.get("coverage", {}).get("part_unknown_counts", {})) for r in results), Counter())),
            "all_parts_valid_required": False, "weak_label_semantic_human_approval_required": False, **FLAGS}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=("prepare", "run", "export"), default="prepare")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-num-seqs", type=int, default=2)
    p.add_argument("--model-reference")
    p.add_argument("--sample-fps", type=float, choices=(4., 8.), default=8.)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-pixels", type=int, default=200704)
    p.add_argument("--visual-max-tokens", type=int, default=1800)
    p.add_argument("--text-max-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=.88)
    p.add_argument("--boundary-review", choices=("on", "off"), default="on")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--rewrite-reference-repair", choices=("on", "off"), default="on")
    p.add_argument("--hand-roi-root", help="Existing native ROI root (SID/roi_manifest.json) or one exact roi_manifest.json; no extraction")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    need(args.tp > 0 and args.batch_size > 0 and args.max_num_seqs >= args.batch_size,
         "positive TP/batch and max_num_seqs >= batch-size required")
    need(0 < args.gpu_memory_utilization < 1 and args.visual_max_tokens > 0 and args.text_max_tokens > 0,
         "invalid runtime budgets")
    rows, plan_sha, total = read_plan(args.plan, args.shard, args.nshard, args.limit)
    out = Path(args.out).absolute()
    need(not out.is_symlink(), "output cannot be a symlink")
    for row, _ in rows:
        for key in batch.SOURCE_FIELDS:
            source_path = Path(row["manifest_record"][key]).resolve()
            need(out != source_path and not source_path.is_relative_to(out), "output contains original source")
    out.mkdir(parents=True, exist_ok=True)
    code = code_identity()
    configuration = params(args, code)
    protocol = {"schema": OUTPUT_SCHEMA, "plan_path": str(Path(args.plan).resolve()), "plan_sha256": plan_sha,
                "params": configuration, "source_count": total, **FLAGS}
    core.save_json(out / "protocol.json", protocol)
    runtime = runtime_loader(args, out, code)
    results, ref_cache = [], {}
    for row, raw_line in rows:
        root = out / "sources" / row["id"]
        binding = {"source_binding": row["binding"], "plan_sha256": plan_sha,
                   "full_source_row_sha256": batch.digest(row), "raw_line_sha256": sha(raw_line),
                   "manifest_record_sha256": batch.digest(row["manifest_record"]), "params_sha256": batch.digest(configuration)}
        with batch.bundle_lock(root):
            try:
                result = process_source(row, raw_line, root, args, binding, runtime, ref_cache)
            except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
                result = {"schema": OUTPUT_SCHEMA, "source": OUTPUT_SCHEMA, "id": row["id"],
                          "status": "failed", "binding": binding, "manifest_record": row["manifest_record"],
                          "source_refs": row["source_refs"], "original_window_manifest": row["window_manifest"],
                          "original_windows": row["windows"], "plan_status": row["plan_status"],
                          "error": f"{type(error).__name__}: {error}", "source_or_execution_validation_failed": True,
                          "final_timed_caption": "", **FLAGS}
                failure_path = root / "failures" / (batch.digest(result) + ".json")
                core.save_json(failure_path, result)
                result["result_ref"] = {"path": str(failure_path), "sha256": batch.sha256(failure_path)}
        results.append(result)
        batch.atomic_write(out / "progress.json", batch.json_bytes({"schema": OUTPUT_SCHEMA,
            "stage": args.stage, "plan_sha256": plan_sha, "selected_source_count": len(rows),
            "processed_source_count": len(results), "status_counts": dict(Counter(r["status"] for r in results)),
            "last_source_id": row["id"], "last_result_ref": result.get("result_ref"), **FLAGS}))
        print(json.dumps({"id": row["id"], "status": result["status"], "facts": len(result.get("facts", []))}, ensure_ascii=False), flush=True)
    summary = summarize(results, args.stage, total, len(rows), plan_sha)
    export_rows = [r for r in results if r["status"] in ("candidate", "abstain", "failed")]
    selection = {"shard": args.shard, "nshard": args.nshard, "selected_ids": [r["id"] for r in results]}
    snapshot = {"summary": summary, "selection": selection, "records": results}
    snapshot_dir = out / "snapshots" / batch.digest(snapshot)
    core.save_json(snapshot_dir / "summary.json", snapshot)
    batch.atomic_write(out / "latest.summary.json", batch.json_bytes({"summary": summary,
        "selection": selection, "snapshot_ref": {"path": str(snapshot_dir / "summary.json"),
        "sha256": batch.sha256(snapshot_dir / "summary.json")}, **FLAGS}))
    if args.stage == "export":
        payload = b"".join(batch.json_bytes(r) + b"\n" for r in export_rows)
        core.save_once(snapshot_dir / "weak_labels.jsonl", payload)
        core.save_json(snapshot_dir / "export.receipt.json", {"weak_labels_sha256": sha(payload),
            "summary_sha256": batch.sha256(snapshot_dir / "summary.json"), "plan_sha256": plan_sha, **FLAGS})
    print(json.dumps({**summary, "snapshot_dir": str(snapshot_dir)}, ensure_ascii=False), flush=True)
    return 1 if summary["failed_source_count"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}", **FLAGS}), file=sys.stderr, flush=True)
        sys.exit(1)
