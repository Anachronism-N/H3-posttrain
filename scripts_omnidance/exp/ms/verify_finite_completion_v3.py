"""Read-only finite ledger checks. No producers, source media, weights or GPUs."""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

SCHEMA = "finite_completion_verification_v3"
CONTRACT_SCHEMA = "finite_completion_contract_v3"
CHECKS = ("ms12k", "vlm12k", "eval15", "gitpackage", "resources", "scope")
LATEST_SCOPE_TRIALS = ("freshdetector32", "bpm_windows", "truebar")
SHARED_VLM_REQUIRED = ("shared_music_windows_vlm",)
SHARED_MODEL_IDENTITY_SHA = "26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c"
SHARED_MODEL_CONTENT_SHA = "0a7c2e55550712af20019a6d683e5b1d0e9232589a764a3ab728d2bb0c55fa6a"
ORIGINAL_VISUAL_SCHEMA_SOURCE = "b83a048d885edb4f6ae4dc41bb5aeaabe24b8646f764bfea9308164423cf2e6d"
SHARED_MUSIC_SOURCE = "882bdb6ee77e9efb7bebd6339f285393259da939ceb084d62b49aed27bf63338"
SHARED_MUSIC_IDS = ("hwrlCvOCYUKpljlV", "LglrebUOmlpWTFrr", "qKgmlBaDdOeqvUZZ")
SHARED_MUSIC_WINDOW_SHAS = ("b5c3de6a1a1b533ae2f2a81ceb5b1fd38cd21b5877b31e6365ccce4307bd4ac0", "efa82987b171ec93e0abc93bd66f77a8d8b1bd8eb7f1dcd7ba553ed131db52f3", "1341205415afad839ec512d9566f7a5e8daa85cce0c22798027b8e8de55d007f")
EXACT_MUSIC_PTS_REQUIRED = True
EXACT_MUSIC_PTS_SHA = "89f7cee6d8df7155fecd2a5fb8b42edb79df7f0da6a33274c27bcccc0d54bda7"
EXACT_MUSIC_PTS_VERIFIER = "26091e34109c0ee0f10ca1d55885c2c98bddcd67e3f63a2ee41e006af23cbd24"
PACKAGE_MINIMUM = {
    "scripts_omnidance/exp/ms/verify_finite_completion_v3.py",
    "scripts_omnidance/exp/ms/ms_chunk_export_v2.py",
    "scripts_omnidance/exp/ms/ms_visual_export.py",
    "scripts_omnidance/exp/ms/ms_visual_batch.py",
    "scripts_omnidance/exp/ms/runtime_jobs/publish_ready_eval_review_v6.py",
    "tests/test_verify_finite_completion_v3.py",
    "tests/test_ms_chunk_export_v2.py",
    "tests/test_ms_visual_export.py",
    "tests/test_ms_visual_batch.py",
    "tests/test_publish_ready_eval_review_v6.py",
    "scripts_omnidance/exp/ms/ms_nlf_rgb32.py",
    "scripts_omnidance/exp/ms/ms_music_windows.py",
    "scripts_omnidance/exp/ms/ms_truebar_caption.py",
    "scripts_omnidance/exp/ms/verify_music_window_pts_and_types.py",
    "scripts_omnidance/exp/ms/ms_visual_music_windows.py",
    "scripts_omnidance/exp/ms/runtime_jobs/nlf_rgb32_fresh_compare_v1.py",
    "scripts_omnidance/exp/ms/runtime_jobs/nlf_rgb32_fresh_caption_control_v1.py",
    "scripts_omnidance/exp/ms/runtime_jobs/beatthis_audio_v1.py",
    "scripts_omnidance/exp/ms/runtime_jobs/beatthis_audio_v2.py",
    "tests/test_ms_nlf_rgb32.py",
    "tests/test_ms_music_windows.py",
    "tests/test_verify_music_window_pts_and_types.py",
    "tests/test_ms_visual_music_windows.py",
    "tests/test_beatthis_audio_v1.py",
    "tests/test_beatthis_audio_v2.py",
    "notes/30_motionscript_chunk_annotation_plan_20261024.md",
    "notes/31_bar_chunking_and_rgb32_execution_plan_20261024.md",
    "results/nlf_rgb32_fresh_detector_fixed3_execution_compare_20261024_v1.json",
    "results/nlf_rgb32_fresh_fixed3_finite_closeout_20261024_v1.json",
    "results/music_window_exact_pts_typed_verification_20261024_v1.json",
    "results/beatthis_independent_output_validation_20261024_v2.json",
}
COUNT = 12000
INPUT_SHA = "4b2efc8233fed8dd7cdb46acf8da318db76cb575faa1177e9e2bc113d453d283"
MS_SOURCE = "5fcf77e46a028d55a574d2dc22c7656c1bd0e8b81b462cb1a1bdc9cd8a1e5b34"
MS_VERIFIER = "b65a9db1f54419665394da921d4eb2456954bad9e4a877f06e9f8893f3346264"
MS_PRODUCER = "8057c48e7b02000c0d28a7dfbc584eaad984239670740037c1d10a5be8c68a00"
VLM_PLAN = "43a38cb9c5d730ea5c34a30bcc97c18394946dd1cee0a7a0f99996d53d8b2e18"
VLM_PRODUCER = "b362dc5275b19c70a6f4f4d149b974e75571ad07e7e4f1f0a2251222fd839204"
VLM_EXPORTER = "4a562a5ff7d5bfcc4b898b60f2e4883e628b3f2f8482e43431286de7f558a558"
NATIVE_SHA = "c56cc1dd36481b3796c2bcce92d599559fabfa255cf37e057924a2a8a2789468"
EVAL_SHA = "e49bea0233da82cd29a05f716d71c3d79cf42ccc9ae54369bca4fd715f650c1a"
TRUEBAR_ADAPTER = "c1b925e1949692a6b98da7b12ef72fb42be02581dac6c2778dce431539b816be"
TRUEBAR_SUMMARIES = {"strict": "9f110fb60c6b2653d9596448ff206ee11f2104d513efbb98497f887aef4dd0de", "merge_short_edges": "57b7810b0e4178cf55f2881bb00486507fd5b32cd13fa2f097dc998918272440"}
BEATTHIS_POST_SHA = "31d4ae9eb6348720fb95fed8d1e378ec5e6376d913b4455159eeec3b8db3aa41"
BEATTHIS_V1_HELPER = "2a06e5baebbf5115825359fe50835656057bb83bca8e5d42399d05198d3c1784"
BEATTHIS_V2_VERIFIER = "83131d4e4f0b69ddbc5bc96cca1ca1d9753962df39cbfd25428be1081bde1075"
BEATTHIS_MODEL = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
FIRST50_INPUT_SHA = "64ae58dd23f5bd3ee05677a2669fec626ef8bcf0ba54adc3b3269f4cbf401b3b"
MUSIC_WINDOWS_SOURCE = "1b1a1309c92f25bd9faf86cc663ab61686e78f668e20398c817ce15a305bed82"
BPM_SUMMARY_SHA = "702269bcfab5f5b536e03af2809ff5d51ff5822ae08fd565dc2c201f9635f16f"
FRESH32_CLOSEOUT_SHA = "026896f861081b4d02f9a1c7f775cf5774dea7c61611547abcbf0e0f6edcc41f"
FRESH32_CAPTION_SHA = "7b9ecc068b530886571835b096870853aae82a7e42b4a0d9ad3bc9fac53f5d9d"
FRESH32_CAPTION_HELPER = "c189030c12d71a19078c460f377ed602ec6ae86117b1d6c6c0b014685f5107f8"
FRESH32_COMPARISON_SHA = "8dc98a5e470b45fac3e5ed02a4fec45e911ff7e6b493e6aef6d6752826407461"
FRESH32_COMPARE_HELPER = "3f21369fbe60ab5f2f341260b1eab682c6fb35bb563f0aa577640d794e82146b"
FRESH32_SOURCE = "8875af505ead1e86313bcafe4aaf18f76c87eac0287d6dd1a2c0877e61c985d5"
FIXED3_INPUT_SHA = "14a38a179bc0a1f0afc4fd9d6eef6e74797564db696cad20503e495b57d34361"
RGB_CONTROL_SHA = "231a8119b7d5adb7eea6204c012910f3c799b1699c7cedc15c9be55c1bc22164"
RGB_REVIEW_SHA = "6cb221c4e4245e134d3715b3c9adc5367a56b3c9f3e4b8bb1e85e1f6be72291d"
RGB_SPEC = "bcb8f9afcb8d0dcc62640e193e3de31dc875020f7e0191fc8af3c5800dad555f"
RGB_MODEL = "52bee28edb6ea9148691331df87cfc238d7e3d9134dc60104a5aaed282a9ddad"
RGB_SOURCE = "f7b3ee3c1c59b9ac19b96f2a73f3bb18a79d4804e715bc1f7334c3e363c467c1"
ANCHOR_SHA = "9609fb8738a5f28e91e7b9916438b1f77fdfa40f4920ac58b30aedbce0594d58"
IDS = {"wJPptuhclZmKAEWs", "wJPrpFsThWAcBQIw", "wJQTclrlOLEYYfTB"}
ROLES = {
    "MS/review_h3_purefm100_n4_chunked_v2/videos",
    "MS/review_h3_anchor01_s100_n4_v3/videos",
    "MS/review_omni6k_u3000_n6_video_v2/videos",
    "MS/review_omni6k_final_u3750_n6_v1/videos",
    "MS/music_direct_full6k_final2250_video_n4_v3",
}


class Pending(Exception):
    pass


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def integer(value, expected=None):
    need(type(value) is int and value >= 0, "count must be a nonnegative typed integer")
    if expected is not None:
        need(value == expected, "wrong exact count: " + repr(value))
    return value


def number(value):
    need(type(value) in (int, float) and math.isfinite(value), "not a finite typed number")
    return value


def same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def sha(value):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid typed SHA256")
    return value


def parse(content):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(content, object_pairs_hook=pairs, parse_constant=nonfinite)


def safe_file(path):
    path = Path(path)
    need(not any(p.is_symlink() for p in (path, *path.parents)), "symlink artifact not allowed")
    if not path.is_file():
        raise Pending("missing artifact: " + str(path))
    return path


def digest_file(path):
    h = hashlib.sha256()
    with safe_file(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref_json(ref, pinned=None, *, object_only=True):
    if not ref or not ref.get("path"):
        raise Pending("explicit artifact reference absent")
    path = safe_file(ref["path"])
    if not ref.get("sha256"):
        raise Pending("expected SHA not yet supplied: " + str(path))
    expected = sha(ref["sha256"])
    if pinned is not None:
        need(expected == pinned, "contract substituted fixed artifact SHA")
    need(path.stat().st_size <= 4 * 1024 * 1024, "receipt is not small JSON")
    content = path.read_bytes()
    need(hashlib.sha256(content).hexdigest() == expected, "receipt SHA mismatch: " + str(path))
    value = parse(content)
    need(type(value) is dict if object_only else type(value) in (dict, list), "receipt has wrong object/array type")
    return value


def stream_jsonl(path, expected_sha):
    h = hashlib.sha256()
    with safe_file(path).open("rb") as stream:
        for raw in stream:
            h.update(raw)
            need(raw.strip(), "blank JSONL row")
            need(len(raw) <= 4 * 1024 * 1024, "oversized JSONL row")
            value = parse(raw)
            need(type(value) is dict, "JSONL row must be an object")
            yield value
    need(h.hexdigest() == sha(expected_sha), "streamed artifact SHA mismatch")


def no_approval(record, keys):
    for key in keys:
        need(record.get(key) is False, "approval must be typed false: " + key)


def input_rows(contract):
    ref = contract.get("input")
    if not ref or not ref.get("path"):
        raise Pending("fixed 12000 input manifest absent")
    need(ref.get("sha256") == INPUT_SHA, "not the frozen 12000 source manifest")
    rows = list(stream_jsonl(ref["path"], INPUT_SHA))
    integer(len(rows), COUNT)
    ids = [r["id"] for r in rows]
    need(all(type(i) is str and i for i in ids) and len(set(ids)) == COUNT, "input SID uniqueness/count invalid")
    for row in rows:
        need(type(row.get("source_group")) is str and row["source_group"], "input group missing")
        need(row.get("split") == "train", "input is not exact train selection")
    return rows


def ordered_rows(path, expected_sha, rows):
    count = 0
    for count, row in enumerate(stream_jsonl(path, expected_sha), 1):
        need(count <= len(rows), "extra exported SID")
        expected = rows[count - 1]
        for key in ("id", "source_group", "split"):
            need(same(row.get(key), expected[key]), "missing, reordered or substituted SID/group/split")
        yield row
    integer(count, len(rows))


def ms_window(value, mode):
    need(value["source"] == "original_motionscript_rules_on_v2_coordinates" and value["mode"] == mode, "A/B source substitution")
    no_approval(value, ("formal_training_eligible", "formal_approval"))
    need(type(value["raw_original_description_nonempty"]) is bool, "raw nonempty flag not typed")
    start, end = number(value["window_start"]), number(value["window_end"])
    need(0 <= start < end, "invalid MS window bounds")
    status = value["status"]
    need(status in ("candidate", "abstain"), "nonterminal MS window")
    candidate = status == "candidate"
    need(value.get("candidate_text_eligible") is candidate, "candidate eligibility differs")
    if candidate:
        need(type(value["motion_caption"]) is str and bool(value["motion_caption"].strip()), "empty candidate caption")
        need(value["support_status"] == "complete_available_samples" and value["abstain_reason"] is None, "partial support cannot be filled")
    else:
        need(value["motion_caption"] is None and type(value["abstain_reason"]) is str and value["abstain_reason"], "abstention must remain null with reason")
    return candidate


def check_ms12k(section, contract, offline):
    complete = ref_json(section.get("complete"))
    need(complete.get("schema") == "motionscript_verified_A_B_export_v1", "wrong MS export schema")
    receipt = ref_json(section.get("complete_receipt"))
    need(receipt["complete_sha256"] == section["complete"]["sha256"], "first complete receipt binding differs")
    verification = ref_json(section.get("verification"))
    need(verification.get("schema") == "motionscript_existing_export_verification_v2", "wrong MS postverification schema")
    need(verification.get("complete") is True and verification.get("status") == "verified", "MS postverification is not complete")
    need(verification.get("exact_expected_bytes_verified") is True and verification.get("existing_export_modified") is False, "not readonly canonical MS postverification")
    for key, expected in (("first_export_source_sha256", MS_SOURCE), ("verifier_source_sha256", MS_VERIFIER), ("producer_source_sha256", MS_PRODUCER), ("input_manifest_sha256", INPUT_SHA), ("existing_complete_sha256", section["complete"]["sha256"])):
        need(verification.get(key) == expected, "MS verification binding differs: " + key)
    need(Path(verification["existing_complete_path"]).resolve() == Path(section["complete"]["path"]).resolve(), "MS postreceipt binds different export")
    need(complete.get("complete") is True and complete.get("status") == "completed", "MS first export not terminal complete")
    for key in ("all_shards_terminal", "all_expected_ids_accounted", "exported"):
        need(complete.get(key) is True, "MS terminal ledger flag absent")
    for key in ("unique_id_count", "successfully_processed_count"):
        integer(complete[key], COUNT)
    integer(complete["failed_source_count"], 0)
    integer(verification["unique_id_count"], COUNT)
    integer(verification["failed_source_count"], 0)
    no_approval(complete, ("formal_approval", "formal_training_eligible", "accepted", "semantic_approval", "human_review_completed"))
    no_approval(verification, ("formal_approval", "formal_training_eligible", "accepted", "semantic_approval", "human_review_completed"))
    binding = complete["binding"]
    for key, expected in (("input_manifest_sha256", INPUT_SHA), ("exporter_source_sha256", MS_SOURCE), ("producer_source_sha256", MS_PRODUCER)):
        need(binding[key] == expected, "first MS binding differs")
    files = complete["file_sha256"]
    need(set(files) == {"captions.jsonl", "lineage.jsonl", "errors.jsonl"} and same(files, verification["canonical_file_sha256"]), "canonical three MS SHA bindings differ")
    root = Path(section["complete"]["path"]).parent
    rows = input_rows(contract)
    counts = Counter()
    full = 0
    for row in ordered_rows(root / "captions.jsonl", files["captions.jsonl"], rows):
        no_approval(row, ("formal_approval", "accepted", "semantic_approval", "human_review_completed"))
        need(row["schema"] == complete["schema"] and row["status"] == "processed", "nonprocessed MS SID")
        ms_window(row["A_whole_clip"], "A_original_rules_whole_clip_same_coordinates")
        windows = row["B_windows"]
        integer(len(windows), 2)
        eligible = [ms_window(w, "B_fixed_window") for w in windows]
        need(number(windows[0]["window_start"]) == 0 and windows[0]["window_end"] == windows[1]["window_start"], "MS windows not contiguous global time")
        for w in windows:
            counts[w["status"]] += 1
        text = "\n".join(w["motion_caption"] for w in windows) if all(eligible) else None
        need(row["B_complete_clip_motion_caption"] == text, "A fallback, partial fill or changed B full text")
        full += all(eligible)
    for name in ("lineage.jsonl", "errors.jsonl"):
        need(digest_file(root / name) == sha(files[name]), "canonical MS file altered")
    coverage = complete["caption_coverage"]
    integer(coverage["target_clip_denominator"], COUNT)
    integer(coverage["observed_B_window_denominator"], 2 * COUNT)
    integer(coverage["B_all_planned_windows_candidate_clip_count"], full)
    need(same(complete["B_window_status_counts"], dict(counts)), "MS candidate/abstain count differs")
    need(same(verification["caption_coverage"], coverage), "MS postreceipt coverage differs")
    return {"terminal_ledger_complete": True, "usable_full_caption_count": full, "candidate_windows": counts["candidate"], "abstain_windows": counts["abstain"], "caption_delivery_complete": False, "source": "original_MS_diagnostic_not_video_first"}


def visual_window(window):
    status = window["status"]
    need(status in ("candidate", "abstain", "failed"), "nonterminal visual window")
    bounds = window["window"]
    start, end = number(bounds["start"]), number(bounds["end"])
    need(0 <= start < end, "invalid visual global window")
    sha(window["receipt_sha256"])
    need(window.get("flags_are_diagnostic_not_semantic_approval") is True, "risk flags treated as approval")
    caption = window["caption"]
    need(type(caption) is str, "caption must be a string")
    if status == "failed":
        need(caption == "" and not window["events"] and type(window["error"]) is str and window["error"], "failed window silently filled")
        return False
    generation = window["generation"]
    need(type(generation) is dict and generation.get("model_call_performed") is True, "missing real generation/EOS")
    integer(generation["max_tokens"], 1200)
    tokens = integer(generation["completion_tokens"])
    integer(generation["prompt_tokens"])
    need(generation["finish_reason"] == "stop" and 0 < tokens < 1200, "not normal EOS/stop or token truncated")
    raw_sha = sha(generation["raw_response_sha256"])
    need(window["files_sha256"]["response.raw.txt"] == raw_sha, "generation rawSHA not bound to postreceipt")
    for key in ("generation.json", "candidate.json", "manifest.json"):
        sha(window["files_sha256"][key])
    frames = {}
    for frame in window["source_frames"]:
        index = integer(frame["index"])
        need(index not in frames, "duplicate source frame")
        time = number(frame["time"])
        need(start <= time < end, "frame outside actual half-open window")
        number(frame["source_timestamp"])
        sha(frame["sha256"])
        frames[index] = time
    need(len(frames) >= 2, "fewer than two supplied real frames")
    events = window["events"]
    need(type(events) is list, "events not typed list")
    if status == "abstain":
        need(caption == "" and not events, "abstention filled with actions")
        return False
    need(caption.strip() and events, "candidate caption or event facts empty")
    event_ids = set()
    for event in events:
        key = event["id"]
        need(type(key) is str and key and key not in event_ids, "invalid event id")
        event_ids.add(key)
        need(type(event["description"]) is str and event["description"].strip(), "event description empty")
        indices = event["frame_indices"]
        need(type(indices) is list and len(indices) >= 2 and all(type(i) is int for i in indices) and len(set(indices)) == len(indices) and set(indices) <= frames.keys(), "event lacks two distinct real supplied frames")
        times = [frames[i] for i in indices]
        need(same(event["evidence_time_range"], [min(times), max(times)]) and event.get("time_range_is_exact_event_boundary") is False, "frame support falsely relabeled action boundary")
    return True


def model_identity_digest(identity):
    content = {name: {"path": item["path"], "sha256": sha(item["sha256"])} for name, item in identity["files"].items()}
    return hashlib.sha256(canonical_bytes({"root": identity["root"], "files": content})).hexdigest()


def model_inventory(identity):
    files = identity["files"]
    need(model_identity_digest(identity) == sha(identity["sha256"]), "model canonical identity digest differs from declared root/file paths and content SHAs")
    integer(len(files), 25)
    result = {}
    for name, value in files.items():
        relative = Path(name)
        need(Path(identity["root"]).is_absolute() and Path(value["path"]) == Path(identity["root"]) / relative, "model file not bound to explicit model path")
        need(not relative.is_absolute() and ".." not in relative.parts, "unsafe model inventory path")
        result[name] = {"sha256": sha(value["sha256"]), "bytes": integer(value["stat"]["size"])}
    return result


def check_vlm12k(section, contract, offline):
    plan = ref_json(section.get("plan"), VLM_PLAN)
    need(plan["schema"] == "choreography_video_first_12000_32b_launch_v1", "wrong VLM full plan")
    integer(plan["target_ids"], COUNT)
    integer(plan["shards"], 8)
    need(plan["producer_sha256"] == VLM_PRODUCER and plan["input_manifest_sha256"] == INPUT_SHA, "wrong VLM full source/input")
    no_approval(plan, ("formal_approval", "human_review_completed", "semantic_approval"))
    manifest = ref_json(section.get("manifest"))
    need(manifest["schema"] == "motionscript_visual_terminal_export_v1" and manifest["status"] == "terminal_verified_unreviewed", "not final VLM postexport manifest")
    need("complete" not in manifest, "invented VLM complete schema field")
    no_approval(manifest, ("semantic_approval", "human_review_completed", "formal_training", "ground_truth", "original_raw_or_caption_facts_modified"))
    for key in ("all_planned_workers_terminal", "all_expected_ids_accounted_for", "flags_are_diagnostic_not_semantic_verdicts", "formal_training_not_updated", "first50_not_silently_reused"):
        need(manifest.get(key) is True, "missing actual full VLM verification: " + key)
    for key in ("expected_ids", "verified_ids"):
        integer(manifest[key], COUNT)
    integer(manifest["expected_window_count"], COUNT * 2)
    for key, expected in (("launch_plan_sha256", VLM_PLAN), ("input_manifest_sha256", INPUT_SHA), ("producer_sha256", VLM_PRODUCER)):
        need(manifest[key] == expected, "VLM manifest binding differs")
    exporter = section.get("exporter")
    if not exporter:
        raise Pending("actual VLM writer source ref not supplied")
    need(manifest["exporter_sha256"] == exporter.get("sha256") == VLM_EXPORTER and digest_file(exporter["path"]) == sha(exporter["sha256"]), "VLM writer source differs")
    jobs = sorted(plan["jobs"], key=lambda j: j["shard"])
    need([j["shard"] for j in jobs] == list(range(8)), "missing/duplicate full VLM shards")
    lineage = sorted(manifest["lineage"], key=lambda j: j["shard"])
    need(len(lineage) == 8, "partial VLM worker lineage")
    reference = ref_json({"path": manifest["model_reference_receipt_path"], "sha256": manifest["model_reference_receipt_sha256"]})["model_identity"]
    need(reference["sha256"] == plan["expected_model_identity_sha256"], "not frozen 32B reference model identity")
    content_inventory = model_inventory(reference)
    inventory_sha = hashlib.sha256((json.dumps(content_inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    need(inventory_sha == manifest["model_content_inventory_sha256"], "25file inventory digest differs")
    for job, link in zip(jobs, lineage):
        integer(job["expected_ids"], COUNT // 8)
        args = job["argv"]
        for flag, value in (("--shard", str(job["shard"])), ("--nshard", "8"), ("--limit", "0"), ("--tp", "2"), ("--batch-size", "2"), ("--max-num-seqs", "2"), ("--prepare-workers", "2")):
            need(args.count(flag) == 1 and args[args.index(flag) + 1] == value, "not frozen full TP2 batch2 plan")
        for key in ("shard", "host", "out", "resource_queue"):
            need(same(job[key], link[key]), "worker lineage does not match frozen full job")
        identity = link["model_identity"]
        need(identity["root"] == args[args.index("--model") + 1], "explicit worker model path unbound")
        inventory = model_inventory(identity)
        need(content_inventory is None or same(content_inventory, inventory), "cross-host model content differs")
        content_inventory = inventory
        need(link["model_content_inventory_sha256"] == manifest["model_content_inventory_sha256"], "model content identity lineage differs")
        terminal = link["resource_worker_exit"]
        need(terminal.get("source_unchanged") is True and terminal.get("input_unchanged") is True, "worker source/input changed")
        integer(terminal["expected_ids"], COUNT // 8)
        integer(terminal["exit_code"])
        need(terminal["exit_code"] in (0, 1), "worker not a valid finite exit")
        queue = Path(job["resource_queue"])
        need(digest_file(queue / "worker_exit.json") == sha(link["worker_exit_sha256"]), "worker exit receipt replaced")
        need(same(parse((queue / "worker_exit.json").read_bytes()), terminal), "worker exit typed body differs")
        need(digest_file(link["admission_path"]) == sha(link["admission_sha256"]), "admission receipt replaced")
    rows = input_rows(contract)
    full = nonempty = no_windows = 0
    sid_counts, window_counts = Counter(), Counter()
    for index, row in enumerate(ordered_rows(Path(section["manifest"]["path"]).parent / "captions.jsonl", manifest["captions_sha256"], rows)):
        need(same(row["original_input"], rows[index]) and row["original_input_row_sha256"] == rows[index]["input_row_sha256"], "VLM original source manifest row changed")
        integer(row["shard"], index % 8)
        job = jobs[index % 8]
        need(row["host"] == job["host"] and row["model_path"] == job["argv"][job["argv"].index("--model") + 1], "VLM source row instance lineage changed")
        need(row["source"] == "independent_video_open_vocabulary_unreviewed", "not actual video-first source")
        no_approval(row, ("semantic_approval", "human_review_completed", "formal_training", "ground_truth"))
        status = row["status"]
        need(status in ("candidate", "abstain", "failed"), "nonterminal VLM SID")
        sid_counts[status] += 1
        windows = row["windows"]
        if not windows:
            need(status == "failed" and row.get("expected_windows_have_no_terminal_receipt") is True, "missing windows not an explicit source failure")
            sha(row["failure_receipt_sha256"])
            integer(len(row["expected_windows"]), 2)
            no_windows += 1
            continue
        integer(len(windows), 2)
        usable = [visual_window(w) for w in windows]
        derived_status = "failed" if any(w["status"] == "failed" for w in windows) else "candidate" if any(usable) else "abstain"
        need(status == derived_status, "visual SID terminal status differs from its actual windows")
        need(windows[0]["window"]["start"] == 0 and windows[0]["window"]["end"] == windows[1]["window"]["start"], "visual windows rebased, overlapping or gapped")
        for w in windows:
            window_counts[w["status"]] += 1
        full += all(usable)
        nonempty += any(usable)
    need(same(manifest["sid_status_counts"], {k: sid_counts[k] for k in ("candidate", "abstain", "failed")}), "SID candidate/abstain/failed counts differ")
    need(same(manifest["window_status_counts"], {k: window_counts[k] for k in ("candidate", "abstain", "failed")}), "window counts differ")
    for key, expected in (("usable_full_caption_count", full), ("incomplete_caption_count", COUNT - full), ("empty_caption_count", COUNT - nonempty), ("clip_denominator", COUNT), ("window_denominator", 2 * COUNT), ("actual_candidate_window_count", window_counts["candidate"]), ("actual_abstain_window_count", window_counts["abstain"]), ("actual_failed_window_count", window_counts["failed"]), ("partial_nonempty_caption_count", nonempty - full), ("any_nonempty_caption_count", nonempty), ("sources_without_nonempty_caption", COUNT - nonempty), ("sources_without_terminal_window_receipts", no_windows), ("source_error_missing_window_count", no_windows * 2), ("actual_terminal_window_receipt_count", sum(window_counts.values()))):
        integer(manifest[key], expected)
    need(manifest["goal_full_caption_coverage_satisfied"] is (full == COUNT) and manifest["caption_delivery_complete"] is (full == COUNT), "ledger mislabeled complete caption delivery")
    integer(manifest["max_additional_failed_attempts"], 2)
    return {"terminal_ledger_complete": True, "usable_full_caption_count": full, "caption_delivery_complete": full == COUNT, "sid_status_counts": dict(sid_counts), "window_status_counts": dict(window_counts)}


def http_bytes(url, expected_status, limit, headers=None):
    need(type(url) is str and url.startswith(("http://", "https://")), "explicit HTTP URL absent")
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            need(response.status == expected_status, "wrong current HTTP status")
            data = response.read(limit + 1)
            need(len(data) <= limit, "HTTP response larger than expected")
            return data, response.headers
    except (OSError, TimeoutError) as error:
        raise Pending("current HTTP unavailable: " + str(error)) from error


def check_eval15(section, contract, offline):
    report = ref_json(section.get("report"), EVAL_SHA)
    need(report["protocol"] == "focused_eval15_publication_independent_v6" and report.get("complete") is True, "eval15 report not complete")
    integer(report["complete_case_count"], 15)
    integer(report["complete_group_count"], 5)
    integer(report["manifest_http_status"], 200)
    for key in ("all15_fresh_source_gate", "all15_http_range_bytes_verified", "no_v1_alias"):
        need(report.get(key) is True, "eval15 actual provenance gate absent")
    need(report["overall_goal_completed"] is False and report["numeric_scores_written"] is False, "technical report falsely approves full goal or human scores")
    checks = report["checks"]
    need(len(checks) == 15 and {(r["model_id"], r["sample_id"]) for r in checks} == {(role, sid) for role in ROLES for sid in IDS}, "wrong final15 roles or fixed3 SIDs")
    for row in checks:
        integer(row["status"], 206)
        integer(row["bytes_verified"], 1024)
        need(row["range_bytes_match_disk"] is True and row["producer_full_decode_gate"] is True, "eval15 incomplete Range/full decode proof")
        need(row["provenance"]["video_sha256"] == sha(row["video_sha256"]), "video provenance SHA differs")
        sha(row["provenance"]["audit_sha256"])
        video = Path(row["actual_video"])
        native_role = row["model_id"] == "MS/music_direct_full6k_final2250_video_n4_v3"
        audit_path = video.with_suffix(".audit.json") if native_role else video.parent / (row["sample_id"] + ".audit.json")
        audit = ref_json({"path": str(audit_path), "sha256": row["provenance"]["audit_sha256"]})
        need(audit.get("complete") is True, "producer full audit incomplete")
        if native_role:
            validated_path = video.parent.parent / (row["sample_id"] + ".validated.json")
            validated = ref_json({"path": str(validated_path), "sha256": row["provenance"]["validation_sha256"]})
            need(validated["full_media_decode_checked"] is True and validated["full_video_audio_decode"] is True, "native complete AV decode receipt missing")
            integer(validated["ffmpeg_decoded_video_frames"], 124)
            videos = [s for s in validated["streams"] if s["codec_type"] == "video"]
            need(len(videos) == 1 and any(s["codec_type"] == "audio" for s in validated["streams"]), "native actual AV stream count differs")
            for key, value in (("width", 704), ("height", 1280), ("decoded_frames", 124)):
                integer(videos[0][key], value)
            need(number(videos[0]["fps"]) == 24 and validated["sha256"] == audit["video_sha256"] == row["video_sha256"], "native decoded/audit/video bytes SHA differs")
            inputs, calls = audit["input"], audit["audit"]
            need(inputs["case"]["sample_id"] == row["sample_id"] and inputs["case"]["condition"] == "joint", "native role/case swapped")
            need(same(calls["scheduler_updates"], {"video": 50, "audio": 50}), "native original AV scheduler must be 50+50")
            need(calls["native_audio_condition"] == "real_reference_audio" and calls["music_mode"] == "ref" and calls["original_joint_pipeline"] is True, "native original real AV protocol differs")
            for key in ("full_cached_prefix_preserved", "original_cached_prefix_tags_preserved", "packed_text_tags_exact"):
                need(calls[key] is True, "native cachedP/tags not preserved")
            integer(calls["text_encoder_calls"], 0)
            no_approval(calls, ("text_encoder_loaded", "tokenizer_loaded", "processor_loaded"))
            no_approval(inputs, ("student_reads_target_or_caption", "teacher_identity", "student_zero_input"))
            need(inputs["semantic_text"] is None and inputs["output_audio"] == "original_mux", "native input uses caption/GT or changed audio protocol")
            need(audit["mux_audio_sha256"] == inputs["native_waveform_sha256"] == calls["waveform_sha256"], "native original audio mux SHA differs")
        else:
            need(audit["id"] == row["sample_id"], "pure audit SID differs")
            media = audit["media"]
            need(media.get("complete_decode") is True, "pure full decode receipt missing")
            for key, count in (("frames", 124), ("height", 1280), ("width", 704), ("video_streams", 1), ("audio_streams", 0)):
                integer(media[key], count)
            need(number(media["fps"]) == 24, "pure fps differs")
            integer(audit["sampling"]["video_scheduler_updates"], 50)
            need(audit["sampling"]["target_audio_generated"] is False, "pure sampler generated audio")
            guard = audit["audio_guard"]
            for value in guard["calls"].values():
                integer(value, 0)
            need(guard["audio_vae_is_none"] is True and guard["scheduler_audio_is_none"] is True, "pure audio modules loaded")
            identity = report["source_identities"][row["model_id"]]
            sha(identity["core_source"]["sha256"])
            sha(identity["wrapper_source"]["sha256"])
            if "purefm100" in row["model_id"] or "anchor01" in row["model_id"]:
                need(number(identity["lora_alpha"]) == 1, "LoRA training scaling applied twice")
                runtime = identity["runtime"]
                need(runtime["full_sequence_retained"] is True and runtime["weights_unchanged"] is True, "runtime chunk changes sequence or weights")
                need(runtime["original_inference_sha256"] == identity["core_source"]["sha256"], "runtime not actual original H3 inference")
    anchor = ref_json(section.get("anchor_state"), ANCHOR_SHA)
    need(anchor.get("complete") is True, "anchor fullstate incomplete")
    integer(anchor["successful_updates"], 100)
    integer(anchor["rng_rank_count"], 8)
    for key, expected in (("actual_training_updates", 100), ("world_size", 8), ("lora_rank", 32), ("lora_alpha", 64), ("complete_state_members", 28)):
        integer(anchor[key], expected)
    need(number(anchor["anchor_lambda"]) == 0.1 and number(anchor["training_cfg_scale"]) == 1.0, "anchor actual FM recipe differs")
    for key in ("full_checkpoint_reverified", "independent_audit", "fm_recipe_verified", "typed_numeric_verified", "all_file_sha256_verified", "model_and_adam_and_rng_and_scheduler_finite", "lora_all_ab_source_verified"):
        need(anchor[key] is True, "anchor complete state/source proof missing")
    need(anchor["human_quality_verified"] is False, "anchor technical state relabeled human quality")
    sha(anchor["lora_sha256"])
    for key in ("optimizer_shards", "model_shards"):
        paths = anchor[key]
        need(type(paths) is list and len(paths) == len(set(paths)) == 8, "anchor 8-rank state missing")
        need(all(type(p) is str and not Path(p).is_absolute() and ".." not in Path(p).parts for p in paths), "unsafe anchor shard path")
    native = ref_json(section.get("native_state"), NATIVE_SHA)
    need(native.get("complete") is True, "native fullstate pending")
    for key, expected in (("steps", 2250), ("exposures", 18000), ("state_tensors", 467), ("optimizer_states", 402), ("trainable_tensors", 402)):
        integer(native[key], expected)
    for key in ("actual_three_unique_epochs", "full_saved_Adam_RNG_sampler_pointer", "strict_loaded_complete_MusicFM"):
        need(native.get(key) is True, "native full Adam/RNG/MusicFM state missing")
    for key in ("local_stage_actual_sha256", "fresh_final_sha256", "fresh_local_stage_sha256"):
        need(sha(native[key]) == sha(native["checkpoint_sha256"]), "native actual final/local stage checkpoint SHA differs")
    original = ref_json({"path": native["original_receipt"], "sha256": native["original_receipt_sha256"]})
    need(original.get("complete") is True and original["checkpoint_sha256"] == native["checkpoint_sha256"], "native original state receipt chain differs")
    training = native["training_finished"]
    need(training.get("complete") is True, "native training terminal incomplete")
    for key, expected in (("actual_successful_updates", 2250), ("actual_exposures", 18000), ("validation_epochs", 3), ("final_format4_state_tensors", 467)):
        integer(training[key], expected)
    for item in report["independent_receipts"]:
        actual = ref_json(item)
        need(actual.get("complete") is True, "independent eval actual receipt incomplete")
    failure = report["preserved_original_manager_publication_failure"]
    need(failure["record"]["complete"] is False and failure["actual_new_v3_HTTP_independently_verified"] is True and "BrokenPipe" in failure["record"]["error"], "original failure silently rewritten")
    need(digest_file(failure["path"]) == sha(failure["sha256"]), "original manager failure altered")
    if offline:
        raise Pending("offline: fresh eval15 HTTP not executed; archived technical evidence valid")
    manifest_url = section.get("manifest_url")
    if not manifest_url:
        raise Pending("explicit current manifest URL absent")
    limit = 4 * 1024 * 1024
    data, headers = http_bytes(manifest_url, 200, limit)
    encoding = headers.get("Content-Encoding", "identity").lower()
    need(encoding in ("identity", "gzip"), "unexpected manifest content encoding")
    if encoding == "gzip":
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            data = stream.read(limit + 1)
        need(len(data) <= limit, "decompressed manifest is not small JSON")
    need(hashlib.sha256(data).hexdigest() == report["manifest_sha256"], "current HTTP manifest not archived final15 manifest")
    for row in checks:
        data, headers = http_bytes(row["media_url"], 206, 1024, {"Range": "bytes=0-1023", "Accept-Encoding": "identity"})
        need(headers.get("Content-Encoding", "identity").lower() == "identity", "Range media bytes are unexpectedly encoded")
        with safe_file(row["actual_video"]).open("rb") as stream:
            need(data == stream.read(1024) and len(data) == 1024, "current Range bytes differ from existing final video")
        need(headers.get("Content-Range") == row["content_range"], "current Range header differs")
    return {"cases": 15, "groups": 5, "full_decode_repeated": False, "human_quality": "unknown"}


def git(repo, *args):
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Pending("git unavailable: " + str(error)) from error
    if result.returncode != 0 and args[0] == "ls-remote":
        raise Pending("current remote Git unavailable: " + result.stderr.decode(errors="replace")[:200])
    need(result.returncode == 0, "git verification command failed: " + result.stderr.decode(errors="replace")[:200])
    return result.stdout


def check_gitpackage(section, contract, offline):
    repo = safe_file(Path(section["repo"]) / ".git" / "HEAD").parents[1]
    branch = section.get("branch")
    need(branch == "codex/motionscript-optimization" and section.get("remote_branch") == branch, "not chosen same remote branch")
    head = section.get("head")
    if not head:
        raise Pending("final HEAD not selected yet")
    need(type(head) is str and re.fullmatch(r"[0-9a-f]{40}", head), "invalid final HEAD")
    files = section.get("files")
    if not files:
        raise Pending("explicit final code/test/notes package absent")
    names = [item["path"] for item in files]
    required_files = section.get("required_files")
    if not required_files:
        raise Pending("explicit final required file inventory absent")
    need(type(required_files) is list and len(required_files) == len(set(required_files)), "invalid required file inventory")
    need(set(required_files) == set(names) and PACKAGE_MINIMUM <= set(names), "package omits required final checker/exporter/producer/publisher code or tests")
    need(len(names) == len(set(names)), "duplicate package path")
    need(any(p.startswith("scripts_") and p.endswith(".py") for p in names) and any(p.startswith("tests/") for p in names) and any(p.startswith("notes/") for p in names), "package lacks code/tests/notes")
    for item in files:
        path = Path(item["path"])
        need(not path.is_absolute() and ".." not in path.parts and path.suffix in (".py", ".md", ".json"), "private/raw asset included in package")
        need(not any(re.search(r"(^\.|credential|secret|password|^captions$|^weights$|^media$|^raw$)", part, re.I) for part in path.parts if part != "scripts_omnidance"), "private asset path included")
        actual = safe_file(repo / path)
        need(actual.stat().st_size <= 2 * 1024 * 1024, "not a small code package file")
        need(digest_file(actual) == sha(item["sha256"]), "working package bytes differ")
        if path.suffix == ".json":
            stack = [parse(actual.read_bytes())]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    need(not {"motion_caption", "raw_captions", "raw_response", "access_token", "private_key", "password", "image_base64", "video_base64"} & value.keys(), "raw captions/media/credentials embedded in package JSON")
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
        committed = git(repo, "show", head + ":" + path.as_posix())
        need(hashlib.sha256(committed).hexdigest() == item["sha256"], "only staged/uncommitted package or different HEAD bytes")
        need(git(repo, "ls-files", "--error-unmatch", "--", path.as_posix()).strip(), "package file untracked")
    need(git(repo, "branch", "--show-current").decode().strip() == branch and git(repo, "rev-parse", "HEAD").decode().strip() == head, "local chosen branch or final HEAD differs")
    if offline:
        raise Pending("offline: actual remote HEAD not checked")
    remote = section.get("remote")
    need(type(remote) is str and remote and not remote.startswith("-"), "explicit safe git remote absent")
    result = git(repo, "ls-remote", "--heads", remote, "refs/heads/" + branch).decode().split()
    need(result == [head, "refs/heads/" + branch], "remote chosen branch final HEAD differs")
    return {"head": head, "packaged_files": len(files), "unrelated_dirty_files_preserved": True}


def utc(value):
    need(type(value) is str, "timestamp must be explicit UTC string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(result.tzinfo is not None, "timestamp lacks timezone")
    return result.timestamp()


def utc_nanoseconds(value):
    need(type(value) is str, "timestamp must be an explicit string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(parsed.tzinfo is not None, "timestamp lacks timezone")
    delta = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


def check_resources(section, contract, offline):
    snapshot = ref_json(section.get("snapshot"), object_only=False)
    need(type(snapshot) is list and len(snapshot) == 8, "final resources must be owner raw eight-node snapshot")
    need({n["node"] for n in snapshot} == set(range(1, 9)), "not eight unique resource nodes")
    plan = ref_json(contract.get("vlm12k", {}).get("plan"), VLM_PLAN)
    jobs = plan["jobs"]
    need(len(jobs) == 8 and {j["shard"] for j in jobs} == set(range(8)), "not all final eight workers")
    exit_refs = section.get("worker_exits")
    restore_refs = section.get("holder_restores")
    if not exit_refs or not restore_refs or not section.get("baseline"):
        raise Pending("owner final eight exits, sixteen restores and known-holder baseline refs absent")
    need(len(exit_refs) == 8 and len(restore_refs) == 16, "partial final worker/holder receipt coverage")
    exits = {str(Path(r["path"]).resolve()): r for r in exit_refs}
    restores = {str(Path(r["path"]).resolve()): r for r in restore_refs}
    need(len(exits) == 8 and len(restores) == 16, "duplicate final receipt reference")
    baseline = ref_json(section["baseline"], object_only=False)
    need(type(baseline) is list and {n["node"] for n in baseline} == set(range(1, 9)), "known holder baseline not eight nodes")
    known = {}
    for node in baseline:
        integer(node["node"])
        processes = node["data"]["apps"] if "data" in node else node["receipt"]["processes"]
        for process in processes:
            if process.get("holder") is True:
                known[node["node"], process["uuid"]] = process
    latest_ns = max((utc_nanoseconds(n["data"]["utc"]) for n in baseline if "data" in n), default=0)
    parents = set()
    for job in jobs:
        key = str((Path(job["resource_queue"]) / "worker_exit.json").resolve())
        need(key in exits, "worker exit is not this frozen full-plan queue")
        record = ref_json(exits[key])
        integer(record["exit_code"])
        need(record["exit_code"] in (0, 1) and record.get("cancelled", False) is False, "worker not ordinary finite terminal")
        need(record["source_unchanged"] is True and record["input_unchanged"] is True, "worker source/input changed")
        integer(record["expected_ids"], COUNT // 8)
        latest_ns = max(latest_ns, safe_file(key).stat().st_mtime_ns)
        for gpu in job["gpus"]:
            key = str((Path(job["resource_queue"]) / ("holder_restored_gpu" + str(gpu) + ".json")).resolve())
            need(key in restores, "missing actual planned GPU holder restoration")
            restored = ref_json(restores[key])
            integer(restored["gpu"], gpu)
            need(restored["managed_parent_wait"] is True, "restored holder lacks managed parent wait")
            identity = restored["holder_identity"]
            manager = restored["manager_identity"]
            need(identity["ppid"] == manager["pid"], "holder parent-chain differs from managed restore receipt")
            integer(manager["pid"])
            parents.add((job["node"], manager["pid"]))
            known[job["node"], restored["uuid"]] = identity
            latest_ns = max(latest_ns, safe_file(key).stat().st_mtime_ns)
    for supplemental in section.get("supplemental_holder_restores", []):
        record = ref_json(supplemental.get("receipt"))
        node = integer(supplemental["node"])
        need(1 <= node <= 8, "supplemental holder node out of range")
        identity = record["holder_identity"]
        manager = record["manager_identity"] if "manager_identity" in record else record["parent_identity"]
        need(identity["ppid"] == manager["pid"], "supplemental holder parent-chain not actual restored manager")
        need(record.get("managed_parent_wait") is True, "supplemental holder restore is not managed parent wait")
        gpu = integer(supplemental["gpu"])
        gpu_uuid = record["uuid"] if "uuid" in record else record["gpu_uuid"]
        need(gpu < 8 and type(gpu_uuid) is str, "supplemental physical GPU binding differs")
        if "gpu" in record:
            integer(record["gpu"], gpu)
        environment = identity["CVD"] if "CVD" in identity else identity["environment"]
        need([value for value in environment if value.startswith("CUDA_VISIBLE_DEVICES=")] == ["CUDA_VISIBLE_DEVICES=" + str(gpu)], "supplemental holder physical GPU environment differs")
        for terminal_ref in supplemental.get("terminal_receipts", []):
            terminal = ref_json(terminal_ref)
            need(terminal.get("complete") is True or type(terminal.get("exit_code")) is int and terminal["exit_code"] in (0, 1), "supplemental predecessor experiment not ordinary finite terminal")
            need(terminal.get("cancelled", False) is False, "cancelled supplemental experiment cannot imply finite completion")
            latest_ns = max(latest_ns, safe_file(terminal_ref["path"]).stat().st_mtime_ns)
        need(supplemental.get("terminal_receipts"), "supplemental holder missing actual predecessor terminal refs")
        parents.add((node, integer(manager["pid"])))
        known[node, gpu_uuid] = identity
        latest_ns = max(latest_ns, safe_file(supplemental["receipt"]["path"]).stat().st_mtime_ns)
    uuids = set()
    timestamps = []
    inactive_reserve_count = 0
    known_reserves = section.get("known_reserves", [])
    need(type(known_reserves) is list, "known reserve ownership must be explicit list")
    reserve_keys = {(r["node"], r["path"]): r for r in known_reserves}
    for node in snapshot:
        integer(node["node"])
        need(type(node["host"]) is str and node["host"], "resource node host missing")
        data = node["data"]
        timestamp_ns = utc_nanoseconds(data["utc"])
        timestamp = timestamp_ns / 1e9
        need(timestamp_ns + 1_000_000_000 >= latest_ns, "snapshot predates final worker exits or holder restoration")
        timestamps.append(timestamp)
        lines = [line.split(",") for line in data["gpus"].splitlines() if line.strip()]
        need(len(lines) == 8 and {int(line[0]) for line in lines} == set(range(8)), "not eight GPU UUIDs per node")
        need(all(len(line) == 4 for line in lines), "GPU CSV fields differ")
        uuid_to_index = {line[1].strip(): int(line[0]) for line in lines}
        node_uuids = set(uuid_to_index)
        need(len(node_uuids) == 8 and not node_uuids & uuids, "duplicate GPU UUID")
        uuids.update(node_uuids)
        apps = data["apps"]
        need(len(apps) == 8 and {a["uuid"] for a in apps} == node_uuids, "GPU must have exactly one known holder, no active sampler")
        for app in apps:
            need(type(app.get("holder")) is bool, "holder flag not boolean")
            if app["holder"] is False:
                raise Pending("active VLM TP worker/sampler remains; blank CVD is not an exemption")
            integer(app["uid"], 0)
            need(integer(app["pid"]) > 0 and integer(app["start_ticks"]) > 0, "holder PID/start identity missing")
            need(type(app["CVD"]) is list and app["CVD"] == ["CUDA_VISIBLE_DEVICES=" + str(uuid_to_index[app["uuid"]])], "holder physical CUDA GPU environment differs")
            expected = known.get((node["node"], app["uuid"]))
            need(expected is not None, "unknown/orphan GPU holder identity")
            for key in ("pid", "ppid", "uid", "start_ticks", "argv", "cgroup"):
                need(same(app[key], expected[key]), "holder UUID/identity/parentchain differs: " + key)
            need(any(Path(arg).name == "hold_gpu.py" for arg in app["argv"]), "not project holder executable")
        for reserve in data["reserves"]:
            integer(reserve["uid"])
            integer(reserve["size"])
            mtime_ns = integer(reserve["mtime_ns"])
            age_ns = timestamp_ns - mtime_ns
            need(age_ns >= -1_000_000_000, "reservation file mtime is after snapshot")
            match = re.fullmatch(r"/tmp/od_reserve_([0-7])", reserve["path"])
            if age_ns >= 1_200_000_000_000 and reserve["uid"] == 0 and reserve["size"] == 0 and match:
                inactive_reserve_count += 1
                continue
            owner = reserve_keys.get((node["node"], reserve["path"]))
            if owner is None:
                raise Pending("fresh unknown/orphan reservation lease remains")
            need(owner.get("role") == "project_reservation_association", "reservation association must not impersonate a kernel FD writer")
            for key in ("uid", "size", "mtime_ns"):
                need(same(reserve[key], owner[key]), "reserve identity differs")
            associated_pid = integer(owner["associated_pid"])
            need((node["node"], associated_pid) in parents or any(a["pid"] == associated_pid for a in apps), "active reservation not associated with known holder/managed-wait parent")
    integer(len(uuids), 64)
    if offline:
        raise Pending("offline: resource freshness not verified")
    now = datetime.now(timezone.utc).timestamp()
    if not all(0 <= now - timestamp <= 300 for timestamp in timestamps):
        raise Pending("stale final64 snapshot; owner must capture fresh postrestore receipt")
    return {"nodes": 8, "gpus": 64, "single_known_holders": 64, "managed_parent_wait_not_exit0": True, "inactive_project_reserves": inactive_reserve_count, "pure_checker_no_signals": True}

def check_freshdetector32(task):
    terminal = ref_json(task.get("evidence"))
    need(terminal["schema"] == "nlf-fresh-detector32-terminal-v1" and terminal.get("complete") is True, "not actual fresh-detector32 terminal")
    integer(terminal["expected"], 3)
    integer(terminal["actual"], 3)
    need(terminal["implementation_sha256"] == FRESH32_SOURCE and terminal["input_manifest_sha256"] == FIXED3_INPUT_SHA, "fresh-detector32 actual source/input binding differs")
    extractor = task.get("extractor")
    if not extractor:
        raise Pending("actual fresh detector source ref absent")
    need(extractor["sha256"] == FRESH32_SOURCE and digest_file(extractor["path"]) == FRESH32_SOURCE, "fresh-detector32 extractor source changed")
    rows = list(stream_jsonl(terminal["input_manifest_path"], FIXED3_INPUT_SHA))
    need(len(rows) == 3 and {r["id"] for r in rows} == IDS, "fresh-detector32 not fixed3 input")
    cases = terminal["cases"]
    need([r["id"] for r in cases] == [r["id"] for r in rows], "fresh-detector32 missing/reordered cases")
    need(terminal["no_external_boxes_supplied"] is True, "fresh detector was conditioned on cached boxes")
    no_approval(terminal, ("person_tracking_added", "smpl_fitting_used", "template_fk_used", "visibility_verified", "semantic_approval"))
    for case, row in zip(cases, rows):
        manifest = ref_json({"path": case["manifest_path"], "sha256": case["manifest_sha256"]})
        need(manifest["schema"] == "motionscript_direct_landmarks_v1" and manifest["source"] == "nlf_fresh_detector_rgb_landmarks", "old cached-box32 is not fresh detector execution")
        need(same(manifest["input_record"], row) and manifest["id"] == row["id"], "fresh-detector32 original input row changed")
        for key in ("fresh_detector_used", "no_external_boxes_supplied", "all_detection_boxes_and_direct_geometry_preserved", "body_anchors_queried_together_with_extras", "zero_detections_keep_nan_mask", "detector_is_separate_network", "crop_features_shared_across_32_queries"):
            need(manifest[key] is True, "fresh-detector32 actual protocol flag absent: " + key)
        no_approval(manifest, ("external_pose_or_box_cache_conditioning", "person_tracking_added", "smpl_fitting_used", "template_fk_used", "visibility_verified", "semantic_approval", "raw_is_ground_truth", "single_total_network_forward"))
        need(manifest["implementation_sha256"] == FRESH32_SOURCE and manifest["source_sha256"]["model"] == RGB_MODEL and manifest["source_sha256"]["spec"] == RGB_SPEC and manifest["source_sha256"]["input_manifest"] == FIXED3_INPUT_SHA, "fresh detector source/model/spec/input binding differs")
        need(sha(manifest["raw_sha256"]) == sha(case["raw_sha256"]), "fresh detector raw receipt binding differs")
        integer(manifest["point_count"], 32)
        frames = integer(manifest["frame_count"])
        integer(case["frame_count"], frames)
        counts, selected = manifest["detection_count"], manifest["selected_index"]
        need(len(counts) == len(selected) == frames and frames > 0, "fresh detector counts/selection frame denominator differs")
        for count, index in zip(counts, selected):
            integer(count)
            need(type(index) is int and (index == -1 if count == 0 else 0 <= index < count), "fresh detector missing-person or selected index invalid")
        integer(case["detected_frames"], sum(c > 0 for c in counts))
        integer(case["multiple_person_frames"], sum(c > 1 for c in counts))
        need(manifest["detector_config"]["extra_boxes"] is None and manifest["selection_policy"] == "per-frame largest XYWH area; argmax first tie; not tracking or guaranteed same identity", "fresh detector actual selection policy differs")
    return {"cases": 3, "source": "nlf_fresh_detector_rgb_landmarks", "visibility_approved": False, "raw_geometry_reloaded": False}


def check_freshdetector32_comparison(task):
    comparison_ref = task.get("comparison")
    if not comparison_ref:
        raise Pending("fresh-detector32 extraction verified, but required actual comparison receipt absent")
    report = ref_json(comparison_ref, FRESH32_COMPARISON_SHA)
    need(report["schema"] == "nlf-fresh-detector32-fixed3-execution-and-compare-terminal-v1" and report["complete"] is True and report["actual_fresh_detector_execution"] is True, "not actual fresh-detector32 execution-and-compare terminal")
    integer(report["actual_fixed3"], 3)
    need(report["source_sha256"] == FRESH32_SOURCE and report["producer_terminal_sha256"] == task["evidence"]["sha256"], "fresh comparison source/producer binding differs")
    need(Path(report["producer_terminal_path"]).resolve() == Path(task["evidence"]["path"]).resolve(), "fresh comparison binds different extraction")
    need(report["compare_helper_sha256"] == FRESH32_COMPARE_HELPER and digest_file(report["compare_helper_path"]) == FRESH32_COMPARE_HELPER, "fresh comparison helper source differs")
    for key in ("same_video_spec_model_names_and_PTS_comparison_verified", "selected32_and_all_ragged_preserved", "no_legacy_nlfbox_or_pose_conditioning", "no_smplfit_or_templateFK", "detector_is_separate_neural_network"):
        need(report[key] is True, "fresh comparison actual protocol proof absent")
    no_approval(report, ("single_total_neural_forward_claim", "selection_is_person_tracking", "inframe_mask_is_visibility", "accuracy_approved", "training_approved", "raw_is_GT", "frozen_library_or_fullVLM_modified", "private_media_uploaded"))
    verified = ref_json({"path": report["independent_comparison_receipt_path"], "sha256": report["independent_comparison_receipt_sha256"]})
    need(verified["schema"] == "nlf-fresh-detector32-fixed3-independent-comparison-verification-v1" and verified["comparison_complete"] is True and verified["accuracy_approved"] is False, "fresh comparison independent evidence not terminal")
    need(len(report["cases"]) == len(verified["cases"]) == 3 and {r["id"] for r in report["cases"]} == IDS and {r["id"] for r in verified["cases"]} == IDS, "fresh comparison not exact fixed3")
    producer = ref_json(task["evidence"])
    producer_cases = {r["id"]: r for r in producer["cases"]}
    independent_cases = {r["id"]: r for r in verified["cases"]}
    for case in report["cases"]:
        original = producer_cases[case["id"]]
        checked = independent_cases[case["id"]]
        for key in ("manifest_sha256", "raw_sha256", "frame_count", "detected_frames", "multiple_person_frames"):
            need(same(case[key], original[key]), "fresh compare report/extraction typed source differs")
        need(case["independent_allpeople_offsets_largestarea_and_emptyshape_verified"] is True, "fresh allpeople selection/offsets not verified")
        need(case["comparison_receipt_sha256"] == checked["comparison_receipt_sha256"], "fresh independent per-case comparison SHA differs")
        receipt = ref_json({"path": case["comparison_receipt_path"], "sha256": case["comparison_receipt_sha256"]})
        need(receipt["schema"] == "nlf-fresh-detector32-cachedbox32-compare-v1" and receipt["actual_fresh_detection"] is True, "comparison does not use actual fresh detections")
        need(receipt["fresh_manifest_sha256"] == case["manifest_sha256"], "comparison uses different fresh manifest")
        no_approval(receipt, ("old_cache_conditioning_used", "person_tracking", "difference_is_accuracy_metric", "in_frame_mask_is_visibility", "accuracy_approved", "training_approved", "source_or_raw_modified"))
        integer(receipt["frame_count"], case["frame_count"])
        need(same(case["necessary_pointframes_cached_fresh"], checked["necessary_pointframes"]) and same(case["necessary_pointframes_cached_fresh"], [receipt["cached_necessary_pointframes"], receipt["fresh_necessary_pointframes"]]), "fresh comparison necessary-support denominators differ")
    refs = report["resource_refs"]
    worker = ref_json(refs["worker_exit.json"])
    integer(worker["exit_code"], 0)
    restored = ref_json(refs["holder_restored.json"])
    need(restored["managed_parent_wait"] is True and restored["holder_identity"]["ppid"] == restored["manager_identity"]["pid"], "fresh resource holder restoration parent-chain differs")
    return {"cases": 3, "comparison_complete": True, "accuracy_approved": False}


def check_freshdetector32_caption_control(task):
    if not task.get("caption_control"):
        raise Pending("fresh extraction and point comparison verified, but new required fresh mask-caption control receipt absent")
    closeout = ref_json(task.get("closeout"), FRESH32_CLOSEOUT_SHA)
    need(closeout["schema"] == "nlf-fresh-detector32-finite-execution-caption-control-closeout-v1" and closeout["complete"] is True, "not actual fresh caption control closeout")
    need(closeout["execution_ref"]["sha256"] == task["comparison"]["sha256"], "fresh caption closeout binds different execution/comparison")
    for key in ("fresh_actual_fixed3_gpu_executed", "comparison_actual_fixed3_cpu_completed", "source_and_old_public_libraries_frozen", "missing_rejected_points_not_filled"):
        need(closeout[key] is True, "fresh caption finite execution proof absent")
    no_approval(closeout, ("mask_support_is_visibility", "accuracy_advantage_established", "GT_approved", "training_approved", "tracking_guaranteed", "raw_updated"))
    integer(closeout["geometry_caption_support_controls_verified"], 12)
    control = ref_json(task["caption_control"], FRESH32_CAPTION_SHA)
    need(closeout["caption_control_ref"]["sha256"] == task["caption_control"]["sha256"] and Path(closeout["caption_control_ref"]["path"]).resolve() == Path(task["caption_control"]["path"]).resolve(), "fresh closeout/caption terminal binding differs")
    need(control["schema"] == "nlf-fresh-detector32-caption-support-control-terminal-v1" and control["complete"] is True, "wrong fresh support caption terminal")
    integer(control["fixed3_count"], 3)
    integer(control["support_control_runs"], 12)
    for key in ("necessary_support_not_visibility", "missing_points_not_interpolated_across_rejection", "redshirt_outside_foot_events_blocked"):
        need(control[key] is True, "fresh necessary support or no-fill proof absent")
    no_approval(control, ("old_cached_raw_or_reports_modified", "fresh_raw_or_producer_modified", "caption_accuracy_advantage_established", "semantic_approval", "training_approved", "gpu_used"))
    helper = closeout["helper_ref"]
    need(helper["sha256"] == control["helper_sha256"] == FRESH32_CAPTION_HELPER and digest_file(helper["path"]) == FRESH32_CAPTION_HELPER, "fresh caption actual helper source changed")
    independent = ref_json(closeout["independent_control_ref"])
    need(independent["independently_verified"] is True and independent["accuracy_approved"] is False and independent["frozen_source_unchanged"] is True, "fresh caption independent verification absent")
    integer(independent["reports_verified"], 12)
    need(independent["control_terminal_sha256"] == FRESH32_CAPTION_SHA, "fresh caption independent terminal binding differs")
    need(len(control["cases"]) == 3 and {c["id"] for c in control["cases"]} == IDS, "fresh caption control cases differ")
    producer = ref_json(task["evidence"])
    fresh = {c["id"]: c for c in producer["cases"]}
    checked = {(c["id"], c["source"], c["mask"]): c for c in independent["controls"]}
    need(len(checked) == 12, "fresh caption independent control union differs")
    for case in control["cases"]:
        runs = case["runs"]
        need(len(runs) == 4 and {(r["source"], r["mode"]) for r in runs} == {(s, m) for s in ("fresh", "cached") for m in ("numeric", "necessary")}, "missing fresh/cached numeric/necessary control")
        for run in runs:
            check = checked[case["id"], run["source"], run["mode"]]
            need(run["report_sha256"] == check["report_sha256"] and run["coords_sha256"] == check["coords_sha256"], "fresh caption independent report/coords SHA differs")
            report = ref_json({"path": run["report_path"], "sha256": run["report_sha256"]})
            need(report["schema"] == "nlf-fresh-cached32-support-caption-control-v1" and report["id"] == case["id"] and report["source_mode"] == run["source"] and report["support_mode"] == run["mode"], "fresh caption actual source/control mode differs")
            for key in ("source_manifest_sha256", "source_raw_sha256", "numeric_valid_pointframes", "selected_support_pointframes", "excluded_pointframes", "whole_event_count", "foot_event_count"):
                need(same(report[key], run[key]), "fresh caption typed source/count binding differs")
            if run["source"] == "fresh":
                need(run["source_manifest_sha256"] == fresh[case["id"]]["manifest_sha256"] and run["source_raw_sha256"] == fresh[case["id"]]["raw_sha256"], "fresh caption uses old cached source")
            need(report["target_support_both_raw_brackets_verified"] is True and report["no_missing_point_fill"] is True, "fresh caption missing-point fill not excluded")
            no_approval(report, ("actual_inframe_is_visibility", "calibrated_geometry_accuracy", "caption_is_training_target", "semantic_approval", "raw_modified"))
            integer(run["whole_event_count"], len(report["events"]))
            need(integer(run["selected_support_pointframes"]) + integer(run["excluded_pointframes"]) == integer(run["numeric_valid_pointframes"]), "fresh caption support denominator differs")
            if run["mode"] == "necessary" and case["id"] == "wJPrpFsThWAcBQIw":
                integer(run["foot_event_count"], 0)
    return {"cases": 3, "controls": 12, "accuracy_approved": False}


def verify_bpm_plan(plan):
    need(plan["schema"] == "motionscript_music_windows_v1" and plan["strategy"] == "bpm_duration_grid_not_detected_beats" and plan["status"] == "ready", "not actual BPM-duration grid")
    binding = plan["binding"]
    for key in ("input_sha256", "video_sha256", "audio_sha256", "actual_video_pts_sha256"):
        sha(binding[key])
    need(number(binding["media_start"]) == 0, "BPM normalized start differs")
    end = number(binding["media_end"])
    need(end > 0 and number(binding["video_fps"]) > 0, "BPM actual media bounds invalid")
    number(binding["video_pts_offset"])
    need(integer(binding["decoded_video_frames"]) > 0, "BPM missing decoded frame denominator")
    need(plan["window_interval"] == "[start,end)" and plan["time_basis"] == "global_media_seconds" and plan["continuous_exact_media_coverage"] is True, "BPM windows changed time basis")
    no_approval(plan, ("audio_video_alignment_verified", "beat_phase_verified", "downbeat_verified", "motions_on_beats_verified", "window_boundaries_are_action_boundaries", "semantic_approval", "human_review_completed"))
    params = plan["params"]
    bpm = number(params["bpm_estimate"])
    beats = integer(params["beats_per_chunk"])
    tail = number(params["min_tail_seconds"])
    need(bpm > 0 and beats > 0 and tail >= 0 and params["first_zero_is_beat_phase"] is False, "BPM duration/phase configuration invalid")
    length = 60 * beats / bpm
    need(number(params["target_seconds"]) == length and math.ceil(end / length) <= 100000, "BPM duration not 60*requested_beats/BPM")
    raw = [{"start": i * length, "end": min((i + 1) * length, end), "kind": "bpm_duration_grid", "grid_index": i, "ends_at_actual_media_end": min((i + 1) * length, end) == end} for i in range(math.ceil(end / length))]
    windows = [{"start": r["start"], "end": r["end"], "raw_window_indices": [i]} for i, r in enumerate(raw)]
    if len(windows) > 1 and windows[-1]["end"] - windows[-1]["start"] < tail:
        last = windows.pop()
        windows[-1]["end"] = last["end"]
        windows[-1]["raw_window_indices"] += last["raw_window_indices"]
    for key in ("raw_windows", "windows"):
        for window in plan[key]:
            need(number(window["start"]) >= 0 and number(window["end"]) > number(window["start"]), "BPM window boundary not typed positive span")
    need(same(plan["raw_windows"], raw) and same(plan["windows"], windows), "BPM raw grid/tail mapping differs from canonical derivation")
    return windows


def check_music_caption(task, *, strategy="bpm", policy=None):
    summary = ref_json(task.get("evidence"), BPM_SUMMARY_SHA if strategy == "bpm" else TRUEBAR_SUMMARIES[policy])
    need(summary["schema"] == "motionscript_music_window_caption_v1" and summary["strategy"] == strategy, "old C metadata or a different strategy cannot substitute actual music-window caption execution")
    source = task.get("source")
    if not source:
        raise Pending("actual BPM source reference absent")
    need(source["sha256"] == MUSIC_WINDOWS_SOURCE and digest_file(source["path"]) == MUSIC_WINDOWS_SOURCE, "actual BPM source changed")
    root = Path(task["evidence"]["path"]).parent
    receipt_path = safe_file(root / "summary.receipt.json")
    receipt = parse(receipt_path.read_bytes())
    need(receipt["summary_sha256"] == task["evidence"]["sha256"] and receipt["window_manifest_sha256"] == summary["window_manifest_sha256"] and receipt["chunks_sha256"] == summary["chunks_sha256"], "BPM summary receipt bytes binding differs")
    protocol = parse(safe_file(root / "protocol.json").read_bytes())
    value = protocol["protocol"]
    fingerprint = hashlib.sha256((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    need(fingerprint == protocol["fingerprint"] == summary["fingerprint"], "BPM canonical source protocol fingerprint differs")
    need(value["code_sha256"]["ms_music_windows.py"] == MUSIC_WINDOWS_SOURCE and value["manifest_sha256"] == FIRST50_INPUT_SHA, "BPM protocol source/input binding differs")
    input_ref = task.get("input")
    if not input_ref:
        raise Pending("BPM actual same50 input ref absent")
    need(input_ref["sha256"] == FIRST50_INPUT_SHA, "BPM not frozen same50 source input")
    rows = list(stream_jsonl(input_ref["path"], FIRST50_INPUT_SHA))
    for line, row in enumerate(rows, 1):
        row["manifest_line"] = line
        for key in ("nlf_path", "video_path", "audio_path"):
            row[key] = os.path.abspath(os.path.join(Path(input_ref["path"]).parent, row[key]))
    integer(len(rows), 50)
    need([r["id"] for r in rows] == value["selected_ids"] and len({r["id"] for r in rows}) == 50, "BPM exact ordered same50 selection differs")
    integer(summary["selected_count"], 50)
    processed, failed = integer(summary["processed_count"]), integer(summary["failed_count"])
    need(processed + failed == 50 and summary["complete"] is (failed == 0), "BPM finite successful/failed ledger differs")
    no_approval(summary, ("accepted", "formal_approval", "semantic_approval", "human_review_completed", "VLM_shared_manifest_actual_execution"))
    need(summary["actual_MS_caption_execution"] is True and value["caption_substitution_applied"] is False and value["VLM_shared_manifest_actual_execution"] is False, "BPM MS execution/shared VLM status misrepresented")
    records = summary["records"]
    need(len(records) == processed and len(summary["errors"]) == failed, "BPM report/error counts differ")
    expected = {r["id"]: r for r in rows}
    need(len({r["id"] for r in records + summary["errors"]}) == 50 and {r["id"] for r in records + summary["errors"]} == set(expected), "BPM success/error exact ID union differs")
    plans = list(stream_jsonl(root / "windows.jsonl", summary["window_manifest_sha256"]))
    need([p["id"] for p in plans] == [r["id"] for r in records], "BPM aggregate plans missing/reordered sources")
    expected_chunks, counts, reasons, support_counts = [], Counter(), Counter(), Counter()
    full = 0
    for record, plan in zip(records, plans):
        sid = record["id"]
        need(record["report_file"] == sid + ".json" and record["window_file"] == sid + ".windows.json", "unsafe BPM report/window paths")
        report = ref_json({"path": str(root / record["report_file"]), "sha256": record["report_sha256"]})
        saved = ref_json({"path": str(root / record["window_file"]), "sha256": record["window_sha256"]})
        need(same(plan, saved) and same(plan, report["window_manifest"]) and report["window_manifest_sha256"] == record["window_sha256"], "BPM shared plan/report exact typed bytes differ")
        need(plan["id"] == sid and same(plan["manifest_record"], expected[sid]) and same(report["binding"], plan["binding"]), "BPM original input/binding differs")
        need(report["source_group"] == expected[sid]["source_group"] and report["split"] == expected[sid]["split"] and report["fingerprint"] == fingerprint, "BPM source split/group/protocol differs")
        if strategy == "bar":
            detection = task["detections_by_id"][sid]
            normalized_source = dict(detection["source_row"], manifest_line=expected[sid]["manifest_line"])
            need(same(plan["manifest_record"], normalized_source), "bar raw model source row changed")
        windows = verify_bpm_plan(plan) if strategy == "bpm" else verify_bar_plan(plan, task["detections_by_id"][sid], policy)
        baseline = ref_json(plan["source_C_report"])
        for key in ("input_sha256", "video_sha256", "audio_sha256"):
            need(plan["binding"][key] == baseline[key], "BPM reused source receipt SHA differs")
        for key in ("media_end_verified_from_video", "nlf_video_timestamps_matched", "nlf_video_frame_count_matched"):
            need(baseline["media"][key] is True, "BPM NLF timestamps not verified against actual video")
        need(plan["coordinates_reference"]["sha256"] == record["coords_sha256"] == baseline["coords_sha256"], "BPM reused whole coordinates receipt differs")
        no_approval(report, ("accepted", "formal_approval", "semantic_approval", "human_review_completed", "baseline_caption_substitution_applied", "MS_whole_window_semantics_verified"))
        need(report["caption_execution"] == "actual_original_MS_on_receipt_verified_whole_C_coordinates", "BPM not actual original MS execution")
        views = report["MS_windows"]
        need(len(views) == len(windows), "BPM planned windows not all actually captioned")
        eligible = []
        for index, (view, window) in enumerate(zip(views, windows)):
            integer(view["chunk_index"], index)
            need(number(view["window_start"]) == window["start"] and number(view["window_end"]) == window["end"] and view["window_interval"] == "[start,end)" and view["time_basis"] == "global_media_seconds", "BPM MS view not this source/window")
            no_approval(view, ("semantic_approval", "human_review_completed", "window_boundaries_are_action_boundaries", "sample_span_is_complete_window_evidence", "complete_available_samples_is_verified_window_semantics", "coordinate_reset_applied"))
            need(view["segment_ids_preserved"] is True and view["source_arrays_retained_full_length"] is True, "BPM source coordinates sliced or reset")
            status = view["status"]
            need(status in ("candidate", "abstain"), "BPM unaccounted window execution status")
            text = view["cleaned_motion_text"]
            need(type(text) is str, "BPM caption text not typed")
            candidate = status == "candidate"
            if candidate:
                need(text.strip() and view["support_status"] == "complete_available_samples" and view["abstain_reason"] is None, "BPM partial/no-output promoted to candidate")
            else:
                need(text == "" and type(view["abstain_reason"]) is str and view["abstain_reason"], "BPM abstention silently filled")
                reasons[view["abstain_reason"]] += 1
            eligible.append(candidate)
            counts[status] += 1
            support_counts[view["support_status"]] += 1
            expected_chunks.append({"schema": report["schema"], "id": sid, "binding": report["binding"], "window_manifest_sha256": report["window_manifest_sha256"], **view})
        complete_text = " \n".join(v["cleaned_motion_text"] for v in views) if all(eligible) else ""
        need(report["MS_full_available_support_coverage"] is all(eligible) and report["MS_all_windows_caption_text"] == complete_text, "BPM full-caption fallback/substitution or incomplete coverage relabeled complete")
        full += all(eligible)
    chunks = list(stream_jsonl(root / "chunks.jsonl", summary["chunks_sha256"]))
    need(same(chunks, expected_chunks), "BPM final aggregate chunks differ from actual source reports")
    stats = summary["A_B_C_newMS_same_processed_source_stats"]["MS"]
    integer(stats["source_clip_denominator"], processed)
    integer(stats["window_denominator"], len(chunks))
    need(same(stats["status_counts"], dict(counts)) and same(stats["abstain_reasons"], dict(reasons)) and same(stats["support_status_counts"], dict(support_counts)), "BPM candidate/abstain/support counts differ")
    integer(summary["full_available_support_clip_count"], full)
    return {"selected_sources": 50, "processed_sources": processed, "failed_sources": failed, "windows": len(chunks), "candidate_windows": counts["candidate"], "abstain_windows": counts["abstain"], "full_available_support_clips": full, "shared_VLM_executed": False, "audio_video_alignment_approved": False}


def check_bpm_windows(task):
    return check_music_caption(task)


def verify_bar_plan(plan, detection, policy):
    need(plan["schema"] == "motionscript_music_windows_v1" and plan["strategy"] == "model_downbeat_bars", "BPM grid or old C windows cannot substitute model downbeat bars")
    source = plan["provenance"]
    need(source["method"] == "beat_this.inference.File2Beats" and source["checkpoint_name"] == "final0" and source["device"] == "cpu" and source["downbeats_origin"] == "model_downbeat_head_postprocessed", "bar provenance not actual official downbeat head")
    need(source["dbn"] is False and source["synthetic"] is False and source["checkpoint_sha256"] == BEATTHIS_MODEL and source["adapter_sha256"] == TRUEBAR_ADAPTER, "bar source uses wrong model/adapter or synthetic data")
    need(source["audio_sha256"] == plan["binding"]["audio_sha256"] == detection["audio_sha256"], "bar audio source differs")
    need(source["helper_sha256"] == BEATTHIS_V1_HELPER and source["raw_detections_sha256"] == detection["_output_sha256"] and source["run_receipt_sha256"] == detection["_run_sha256"], "bar raw detection/original run/helper SHA chain differs")
    integer(source["raw_detection_line"], detection["_line_number"])
    need(source["raw_detection_line_sha256"] == detection["_raw_line_sha256"], "bar original model response raw line differs")
    need(same(source["raw_beats_seconds"], detection["beats"]) and same(source["raw_downbeats_seconds"], detection["downbeats"]), "bar model raw timestamps modified")
    db = plan["downbeats_seconds"]
    need(same(db, detection["downbeats"]), "bar downbeats synthesized from BPM or every-four beats")
    end = number(plan["binding"]["media_end"])
    for timestamps in (detection["beats"], db):
        values = [number(t) for t in timestamps]
        need(all(0 <= t <= end for t in values) and all(a < b for a, b in zip(values, values[1:])), "bar timestamps outside actual media or unordered")
    need(all(t in detection["beats"] for t in db), "downbeats with empty/foreign beats are invalid")
    no_approval(plan, ("beat_phase_verified", "downbeat_verified", "audio_video_alignment_verified", "motions_on_beats_verified", "semantic_approval", "human_review_completed", "window_boundaries_are_action_boundaries"))
    need(plan["model_downbeat_timestamps_consumed"] is True and plan["detected_downbeat_boundaries_verified_in_source"] is True, "actual model timestamps were not consumed")
    need(plan["status"] == ("ready" if len(db) >= 2 else "insufficient_downbeats"), "insufficient-downbeat status not preserved")
    params = plan["params"]
    need(params["policy"] == policy and params["short_interior_bars_merged"] is False and params["insufficient_downbeats_fallback"] is False, "bar residuals or insufficient intervals mislabeled as complete bars")
    minimum = number(params["min_edge_seconds"])
    bars = [{"start": a, "end": b, "bar_index": i, "downbeat_indices": [i, i + 1]} for i, (a, b) in enumerate(zip(db, db[1:]))]
    raw = []
    if not db:
        raw = [{"start": 0.0, "end": end, "kind": "unresolved_no_downbeats", "bar_index": None}]
    else:
        if db[0] > 0:
            raw.append({"start": 0.0, "end": db[0], "kind": "prefix_residual", "bar_index": None})
        raw.extend({**bar, "kind": "complete_detected_bar"} for bar in bars)
        if db[-1] < end:
            raw.append({"start": db[-1], "end": end, "kind": "suffix_residual", "bar_index": None})
    windows = [{"start": r["start"], "end": r["end"], "raw_window_indices": [i]} for i, r in enumerate(raw)]
    if policy == "merge_short_edges" and bars:
        if len(windows) > 1 and raw[0]["kind"] == "prefix_residual" and raw[0]["end"] < minimum:
            prefix = windows.pop(0)
            windows[0]["start"] = 0.0
            windows[0]["raw_window_indices"] = prefix["raw_window_indices"] + windows[0]["raw_window_indices"]
        if len(windows) > 1 and raw[-1]["kind"] == "suffix_residual" and raw[-1]["end"] - raw[-1]["start"] < minimum:
            tail = windows.pop()
            windows[-1]["end"] = end
            windows[-1]["raw_window_indices"] += tail["raw_window_indices"]
    for entries in (plan["raw_windows"], plan["windows"], plan["raw_bars"]):
        for entry in entries:
            need(number(entry["start"]) >= 0 and number(entry["end"]) > number(entry["start"]), "bar boundaries must be typed numeric spans")
    need(same(plan["raw_bars"], bars) and same(plan["raw_windows"], raw) and same(plan["windows"], windows), "bar/raw residual/window mapping changed")
    return windows


def check_truebar(task):
    strict = task.get("strict")
    merged = task.get("merge_short_edges")
    if not strict or not merged or not task.get("beatthis_validation"):
        raise Pending("actual same50 strict and merged bar MS executions plus BeatThis postvalidation refs absent")
    adapter = task.get("adapter")
    if not adapter:
        raise Pending("actual truebar adapter source reference absent")
    need(adapter["sha256"] == TRUEBAR_ADAPTER and digest_file(adapter["path"]) == TRUEBAR_ADAPTER, "actual truebar adapter changed")
    post = ref_json(task["beatthis_validation"], BEATTHIS_POST_SHA)
    need(post["schema"] == "beatthis_independent_output_validation_v2" and post["status"] == "completed", "strict BeatThis independent postvalidation missing")
    integer(post["exit_code"], 0)
    integer(post["validated_count"], 53)
    integer(post["invalid_count"], 0)
    need(post["actual_model_inference_rerun"] is False and post["original_v1_outputs_and_source_untouched"] is True and post["invalid_emptybeats_nonempty_downbeats_rejected"] is True, "new BeatThis verifier impersonates original inference or missed invalid subset")
    need(post["validation_helper_sha256"] == BEATTHIS_V2_VERIFIER and post["actual_v1_helper_sha256"] == BEATTHIS_V1_HELPER and post["checkpoint_sha256"] == BEATTHIS_MODEL, "BeatThis actual producer/postverifier/model binding differs")
    group = post["groups"]["first50"]
    for key in ("expected_count", "checked_count", "valid_count"):
        integer(group[key], 50)
    integer(group["invalid_count"], 0)
    need(group["all_outputs_equal_v2_strict_revalidation"] is True, "BeatThis raw output altered during postvalidation")
    run = ref_json({"path": group["actual_v1_run_receipt_path"], "sha256": group["actual_v1_run_receipt_sha256"]})
    need(run["schema"] == "beatthis_audio_detections_v1" and run["status"] == "completed" and run["runtime"]["helper_sha256"] == BEATTHIS_V1_HELPER, "BeatThis original runtime receipt differs")
    integer(run["exit_code"], 0)
    integer(run["successful_count"], 50)
    integer(run["failed_count"], 0)
    for key in ("expected_count", "attempted_count"):
        integer(run[key], 50)
    need(run["input_manifest_sha256"] == FIRST50_INPUT_SHA, "BeatThis actual same50 input differs")
    need(run["params"]["device"] == "cpu" and run["params"]["dbn"] is False and run["params"]["float16"] is False and run["params"]["postprocessor"] == "minimal", "BeatThis actual runtime is not CPU FP32 minimal postprocessor")
    need(same(run["model_runtime"]["parameter_devices"], ["cpu"]) and same(run["model_runtime"]["parameter_dtypes"], ["torch.float32"]), "BeatThis parameter runtime differs")
    need(run["output_sha256"] == group["detections_sha256"] and run["model"]["checkpoint_sha256"] == BEATTHIS_MODEL, "BeatThis actual output/model binding differs")
    raw_detection_bytes = safe_file(group["detections_path"]).read_bytes()
    need(hashlib.sha256(raw_detection_bytes).hexdigest() == group["detections_sha256"], "BeatThis actual output bytes differ")
    detections = [parse(line) for line in raw_detection_bytes.splitlines() if line.strip()]
    integer(len(detections), 50)
    by_id = {d["id"]: d for d in detections}
    need(len(by_id) == 50, "duplicate BeatThis source ID")
    for index, detection in enumerate(detections, 1):
        detection["_line_number"] = index
        detection["_raw_line_sha256"] = hashlib.sha256(raw_detection_bytes.splitlines()[index - 1]).hexdigest()
        detection["_run_sha256"] = group["actual_v1_run_receipt_sha256"]
        detection["_output_sha256"] = group["detections_sha256"]
        need(detection["schema"] == "beatthis_audio_detections_v1" and detection["status"] == "ok" and detection["inference_called"] is True and detection["model_prediction_returned"] is True, "BeatThis no actual model prediction")
        need(detection["helper_sha256"] == BEATTHIS_V1_HELPER and detection["model_sha256"] == BEATTHIS_MODEL, "BeatThis per-source producer/model differs")
        integer(detection["beat_count"], len(detection["beats"]))
        integer(detection["downbeat_count"], len(detection["downbeats"]))
        need(all(t in detection["beats"] for t in detection["downbeats"]), "empty beats with nonempty downbeats cannot pass")
        no_approval(detection, ("bar_phase_accuracy_verified", "beat_accuracy_verified", "downbeat_accuracy_verified", "audio_video_alignment_verified"))
    results = {}
    for policy, subtask in (("strict", strict), ("merge_short_edges", merged)):
        results[policy] = check_music_caption({**subtask, "source": task["source"], "input": task["input"], "detections_by_id": by_id}, strategy="bar", policy=policy)
    return {"selected_sources": 50, "actual_detected_downbeat_intervals": sum(max(0, len(d["downbeats"]) - 1) for d in detections), "insufficient_downbeat_sources": sum(len(d["downbeats"]) < 2 for d in detections), "policies": results, "human_bar_accuracy_approved": False}


def check_music_exact_pts(task, policies):
    if not task.get("pts_validation"):
        raise Pending("MS finite captions verified, but independent exact actual-video-PTS and canonical typed-plan postreceipt absent")
    post = ref_json(task["pts_validation"], EXACT_MUSIC_PTS_SHA)
    need(post["schema"] == "music_window_exact_pts_typed_verification_v1" and post["complete"] is True, "not exact video-PTS typed-plan terminal")
    need(post["source_sha256"] == MUSIC_WINDOWS_SOURCE and post["input_manifest_sha256"] == FIRST50_INPUT_SHA, "exact PTS postreceipt source/input binding differs")
    need(post["actual_video_pts_probed_not_inferred_from_nlf"] is True, "NLF timestamps tolerance match is not exact actual video PTS proof")
    no_approval(post, ("old_reports_and_captions_modified", "MS_or_model_inference_repeated", "source_video_bytes_rehashed", "audio_video_alignment_verified", "motion_accuracy_verified", "GPU_used"))
    integer(post["video_probes"], 50)
    integer(post["reports_verified"], 150)
    need(post["verifier_sha256"] == EXACT_MUSIC_PTS_VERIFIER and digest_file(post["verifier_path"]) == sha(post["verifier_sha256"]), "exact PTS postverifier source changed")
    records = post["records"]
    need(len(records) == 150 and len({(r["id"], r["policy"]) for r in records}) == 150, "exact PTS typed record union differs")
    for policy in policies:
        summary_ref = task["evidence"] if policy == "bpm" else task[policy]["evidence"]
        root = Path(summary_ref["path"]).parent
        summary = ref_json(summary_ref)
        counts = post["bundle_counts"].get(str(root.resolve()))
        need(counts is not None and counts["summary_sha256"] == summary_ref["sha256"], "exact PTS postreceipt binds different music bundle summary")
        integer(counts["clips"], 50)
        selected = {r["id"]: r for r in records if r["policy"] == policy}
        need(set(selected) == {r["id"] for r in summary["records"]} and len(selected) == 50, "exact PTS source/policy coverage incomplete")
        for source_record in summary["records"]:
            record = selected[source_record["id"]]
            need(record["exact_pts_digest_equal_original"] is True and record["canonical_typed_plan_verified"] is True, "exact actual PTS or canonical type check not performed")
            need(record["report"]["sha256"] == source_record["report_sha256"] and record["plan"]["sha256"] == source_record["window_sha256"], "exact PTS per-source receipt SHA differs")
            need(Path(record["report"]["path"]).resolve() == (root / source_record["report_file"]).resolve() and Path(record["plan"]["path"]).resolve() == (root / source_record["window_file"]).resolve(), "exact PTS postreceipt source path differs")
            plan = ref_json(record["plan"])
            need(sha(record["actual_video_pts_sha256"]) == plan["binding"]["actual_video_pts_sha256"], "true video PTS digest not equal original source-bound plan")
            integer(record["frame_count"], plan["binding"]["decoded_video_frames"])
    return {"video_probes": 50, "reports_verified": 150, "exact_PTS_source_bound": True}


def original_visual_functions():
    path = Path(__file__).with_name("ms_visual_observe.py")
    need(digest_file(path) == ORIGINAL_VISUAL_SCHEMA_SOURCE, "original visual parser/validator source changed")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"sentence_key", "strict_json", "parse_response", "validate_candidate"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    need({node.name for node in functions} == names, "original visual pure validators missing")
    namespace = {"json": json, "re": re, "SCHEMA": "motionscript_open_visual_v1"}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parse_response"], namespace["validate_candidate"]


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def shared_source_identity(source, row, binding):
    keys = (("video_path", "video_sha256"), ("audio_path", "audio_sha256"), ("nlf_path", "input_sha256"))
    need(set(source["files"]) == {key for key, _ in keys}, "shared source file inventory differs")
    content = {}
    for key, digest_key in keys:
        item = source["files"][key]
        need(item["path"] == row[key] and sha(item["sha256"]) == binding[digest_key], "shared source file path/SHA not pinned to input and shared plan")
        content[key] = {"path": item["path"], "sha256": item["sha256"]}
    expected = hashlib.sha256(canonical_bytes({"row": row, "files": content})).hexdigest()
    need(source["sha256"] == expected and source["audio_and_nlf_are_provenance_only_not_model_inputs"] is True, "shared source canonical identity differs from pinned input/plan")
    return expected


def shared_raw_response(bundle, files, generation=None):
    path = Path(bundle) / "response.raw.txt"
    need(path.is_file(), "actual original raw response file missing")
    path = safe_file(path)
    need(path.stat().st_size <= 4 * 1024 * 1024, "raw response is not a small model text artifact")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    need(digest == sha(files["response.raw.txt"]), "actual original raw response bytes SHA differs from receipt")
    if generation is not None:
        need(generation["raw_response_sha256"] == digest, "actual original raw bytes differ from generation receipt")
    return content, digest


def check_shared_music_windows_vlm(task):
    summary = ref_json(task.get("evidence"))
    need(summary["schema"] == "motionscript_visual_shared_music_windows_v1", "MS caption metadata is not actual shared-window VLM execution")
    if summary.get("stage") != "run" or summary.get("all_ready_windows_terminal") is not True:
        raise Pending("shared-window VLM prepared/not terminal; no actual complete run")
    run = ref_json(task.get("run_receipt"))
    loaded = ref_json(task.get("load_receipt"))
    source = task.get("source")
    if not source:
        raise Pending("shared-window VLM actual source ref absent")
    need(source["sha256"] == SHARED_MUSIC_SOURCE and digest_file(source["path"]) == SHARED_MUSIC_SOURCE, "shared VLM consumer source differs")
    need(run["schema"] == loaded["schema"] == summary["schema"] and run["own_source"]["sha256"] == SHARED_MUSIC_SOURCE and run["helper_source"]["sha256"] == MUSIC_WINDOWS_SOURCE, "shared VLM source/helper run identity differs")
    need(run["exact_pts_helper_source"]["sha256"] == EXACT_MUSIC_PTS_VERIFIER, "shared VLM did not bind the actual canonical PTS/type helper")
    need(run["original_code"]["files"]["ms_visual_batch.py"]["sha256"] == VLM_PRODUCER, "shared consumer mutated original full-pool producer")
    need(run["manifest_sha256"] == FIRST50_INPUT_SHA and same(run["selected_ids"], list(SHARED_MUSIC_IDS)), "shared VLM not exact authority first3")
    need(run["independent_pts_verification"]["sha256"] == EXACT_MUSIC_PTS_SHA and run["independent_pts_verification"]["actual_video_pts_probed_not_inferred_from_nlf"] is True, "shared VLM lacks independent actual video PTS receipt")
    need(run["original_code"]["files"]["ms_visual_observe.py"]["sha256"] == ORIGINAL_VISUAL_SCHEMA_SOURCE, "shared run used a different original visual parser/validator")
    parse_response, validate_candidate = original_visual_functions()
    input_rows = list(stream_jsonl(run["manifest_path"], FIRST50_INPUT_SHA))
    fields = ("id", "video_path", "audio_path", "nlf_path", "source_group", "split")
    normalized_rows = {}
    for line, original in enumerate(input_rows[:3], 1):
        row = {key: original[key] for key in fields}
        row["input_line"] = line
        row["input_row_sha256"] = hashlib.sha256(canonical_bytes(original)).hexdigest()
        normalized_rows[row["id"]] = row
    need(list(normalized_rows) == list(SHARED_MUSIC_IDS), "shared actual run input missing/reordered authority sources")
    pts = ref_json(run["independent_pts_verification"], EXACT_MUSIC_PTS_SHA)
    need(pts["actual_video_pts_probed_not_inferred_from_nlf"] is True and pts["complete"] is True, "shared VLM independent actual PTS proof absent")
    integer(summary["source_count"], 3)
    integer(summary["strategy_source_count"], 9)
    integer(summary["window_count"], 22)
    integer(summary["pending_insufficient_downbeat_plan_count"], 0)
    need(summary["all_plans_executed"] is True and summary["VLM_shared_manifest_actual_execution"] is True and summary["not_fixed_window_baseline_reused"] is True, "shared VLM did not actually execute all ready strategies")
    no_approval(summary, ("audio_video_alignment_verified", "motions_on_beats_verified", "semantic_approval", "human_review_completed", "formal_training", "full_pool_updated"))
    need(loaded["one_model_load_for_all_ready_strategies"] is True and loaded["current_model_and_code_sha_verified_before_load"] is True and loaded["actual_decoding"] == "local_vllm_LLM.chat_free_json", "shared VLM is not actual single-load free decoding")
    params = run["params"]
    values = params["values"]
    params_sha = hashlib.sha256((json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    need(params_sha == params["sha256"] == loaded["params_sha256"], "shared VLM typed parameter digest differs")
    for key, count in (("tp", 2), ("context", 16384), ("max_frames", 16), ("max_pixels", 200704), ("batch_size", 1), ("max_num_seqs", 1), ("max_tokens", 1200)):
        integer(values[key], count)
    need(values["new_source_sha256"] == SHARED_MUSIC_SOURCE and values["music_window_helper_sha256"] == MUSIC_WINDOWS_SOURCE and values["independent_pts_verification_sha256"] == EXACT_MUSIC_PTS_SHA, "shared VLM params not actual consumer/helper/PTS receipt")
    need(values["exact_pts_helper_sha256"] == EXACT_MUSIC_PTS_VERIFIER, "shared VLM parameter receipt exact-PTS helper identity differs")
    need(same(values["shared_jsonl_sha256"], list(SHARED_MUSIC_WINDOW_SHAS)), "shared VLM missing/reordered strategies")
    model = run["model_identity"]
    need(model["sha256"] == loaded["model_identity_sha256"] == values["model_identity_sha256"] == SHARED_MODEL_IDENTITY_SHA, "shared VLM model identity differs")
    inventory = model_inventory(model)
    need(hashlib.sha256(canonical_bytes(inventory)).hexdigest() == SHARED_MODEL_CONTENT_SHA, "shared VLM fixed 25-file content/size inventory differs")
    refs = run["strategy_references"]
    need(len(refs) == 3 and [r["sha256"] for r in refs] == list(SHARED_MUSIC_WINDOW_SHAS), "shared VLM three exact source JSONL refs differ")
    plans = {}
    for tag, ref in zip(("bpm", "bar_strict", "bar_merge_short_edges"), refs):
        rows = list(stream_jsonl(ref["path"], ref["sha256"]))
        need([r["id"] for r in rows[:3]] == list(SHARED_MUSIC_IDS), "shared VLM source first3 reordered or heldout substitution")
        for row in rows[:3]:
            plans[tag, row["id"]] = row
    records = summary["records"]
    need(len(records) == 9 and {(r["tag"], r["id"]) for r in records} == set(plans), "shared VLM strategy/source union incomplete")
    counts, model_calls, short, full = Counter(), 0, 0, 0
    for record in records:
        plan = plans[record["tag"], record["id"]]
        row = normalized_rows[record["id"]]
        need(plan["binding"]["id"] == row["id"] and all(same(plan["manifest_record"][key], row[key]) for key in fields), "shared plan source ID/context not bound to actual input")
        source_sha = shared_source_identity(record["source_identity"], row, plan["binding"])
        need(same(record["shared_plan"], plan) and same(record["raw_windows"], plan["raw_windows"]), "shared VLM changed windows or residual mapping")
        need(record["source_bound_actual_pts_verified"] is True and record["plan_status"] == "ready", "shared VLM unverified/insufficient plan not counted separately")
        no_approval(record, ("BPM_or_downbeat_values_supplied_to_model", "MS_action_captions_supplied_to_model", "audio_video_alignment_verified", "motions_on_beats_verified", "semantic_approval", "human_review_completed", "formal_training"))
        expected_plan_sha = hashlib.sha256((json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
        need(record["shared_plan_sha256"] == expected_plan_sha, "shared VLM bound different per-source plan")
        windows = record["windows"]
        need(len(windows) == len(plan["windows"]), "shared VLM missing actual planned window")
        for index, (window, planned) in enumerate(zip(windows, plan["windows"])):
            integer(window["window_index"], index)
            need(same(window["window"], planned), "shared VLM actual window extended/rebased/changed")
            status = window["status"]
            if status == "prepared":
                raise Pending("shared VLM window not generated/terminal")
            need(status in ("candidate", "abstain", "failed"), "shared VLM unknown window terminal status")
            bundle = Path(window["bundle"])
            expected_binding = {"id": row["id"], "source_sha256": source_sha, "model_sha256": model["sha256"], "params_sha256": params_sha, "window": planned, "shared_plan_sha256": record["shared_plan_sha256"], "shared_jsonl_sha256": refs[("bpm", "bar_strict", "bar_merge_short_edges").index(record["tag"])]["sha256"], "decoding_protocol": summary["schema"]}
            if window.get("support_reason"):
                support = ref_json({"path": str(bundle / "insufficient_support.receipt.json"), "sha256": window["support_receipt_sha256"]})
                need(support["schema"] == summary["schema"] and support["status"] == "abstain", "short-support receipt schema/status differs")
                need(support["id"] == row["id"] and same(support["binding"], expected_binding), "short-support receipt not bound to actual source/window/model")
                need(window["caption"] == "" and window["events"] == [], "short-support abstention summary contains fabricated caption/events")
                count = integer(plan["binding"]["decoded_video_frames"])
                fps = number(plan["binding"]["video_fps"])
                times = [i / fps for i in range(count)]
                if hashlib.sha256(canonical_bytes(times)).hexdigest() != plan["binding"]["actual_video_pts_sha256"]:
                    raise Pending("short-support exact original PTS list unavailable; cannot infer frame evidence from FPS")
                actual = [i for i, time in enumerate(times) if planned["start"] <= time < planned["end"]]
                need(len(actual) < 2 and same(support["actual_frame_indices"], actual) and same(support["actual_frame_times"], [times[i] for i in actual]), "short-support frame indices/times do not match exact source PTS")
                need(not (bundle / "generation.json").exists() and not (bundle / "candidate.json").exists() and not (bundle / "response.raw.txt").exists(), "insufficient-frame abstention has fabricated model generation/raw")
                need(status == "abstain" and window["model_call_performed"] is False and support["model_call_performed"] is False and support["abstain"] is True and support["motion_caption"] == "" and not support["events"], "insufficient source frames fabricated a generation")
                no_approval(support, ("window_extended", "frame_evidence_filled", "semantic_approval", "human_review_completed"))
                need(len(support["actual_frame_indices"]) < 2 and same(support["window"], planned), "short-window support evidence differs")
                short += 1
            else:
                receipt = ref_json({"path": str(bundle / "receipt.json"), "sha256": window["receipt_sha256"]})
                need(receipt["binding"]["source_sha256"] == record["source_identity"]["sha256"] and receipt["binding"]["model_sha256"] == model["sha256"], "shared VLM receipt uses different source/model")
                need(receipt["status"] == status and same(receipt["binding"]["window"], planned) and receipt["binding"]["shared_plan_sha256"] == record["shared_plan_sha256"] and receipt["binding"]["params_sha256"] == params_sha, "shared VLM actual terminal receipt binding differs")
                need(same(receipt["binding"], expected_binding), "shared actual receipt ID/source/window/params differs from pinned input")
                no_approval(receipt, ("semantic_approval", "human_review_completed", "training_entry_updated"))
                if status in ("candidate", "abstain"):
                    files = receipt["files_sha256"]
                    manifest = ref_json({"path": str(bundle / "manifest.json"), "sha256": files["manifest.json"]})
                    generation = ref_json({"path": str(bundle / "generation.json"), "sha256": files["generation.json"]})
                    candidate = ref_json({"path": str(bundle / "candidate.json"), "sha256": files["candidate.json"]})
                    need(same(manifest["binding"], expected_binding) and same(manifest["input"], row) and same(manifest["source_identity"], record["source_identity"]) and same(manifest["window"], planned), "shared evidence manifest source/input/window binding differs")
                    need(manifest["video_sha256"] == plan["binding"]["video_sha256"] and manifest["request_sha256"] == files["request.json"] and digest_file(bundle / "request.json") == manifest["request_sha256"], "shared original request/video SHA binding differs")
                    raw, raw_sha = shared_raw_response(bundle, files, generation)
                    original = parse_response(canonical_bytes(raw.decode("utf-8")))
                    parse(canonical_bytes(original))
                    rebuilt = validate_candidate(original, manifest)
                    need(all(same(candidate.get(key), value) for key, value in rebuilt.items()), "shared candidate facts differ from original raw parser/validator reconstruction")
                    need(candidate["raw_response_sha256"] == raw_sha and same(candidate["binding"], expected_binding) and candidate["video_sha256"] == manifest["video_sha256"] and candidate["request_sha256"] == manifest["request_sha256"], "shared candidate raw/source/model/request binding differs")
                    need(candidate["motion_caption"] == window["caption"] and same(candidate["events"], window["events"]), "shared VLM summary caption/events differ from actual candidate")
                    if status == "candidate":
                        need(candidate["abstain"] is False, "abstention mislabeled candidate")
                    else:
                        need(candidate["abstain"] is True, "candidate mislabeled abstention")
                    need(window["model_call_performed"] is True, "shared VLM candidate/abstain lacks actual recorded model call")
                    visual_window({"status": status, "error": receipt["error"], "window": planned, "receipt_sha256": window["receipt_sha256"], "flags_are_diagnostic_not_semantic_approval": True, "caption": window["caption"], "generation": generation, "files_sha256": files, "source_frames": manifest["frames"], "events": window["events"]})
                else:
                    files = receipt["files_sha256"]
                    need(not (bundle / "candidate.json").exists(), "failed shared window retains a falsely approved candidate")
                    if (bundle / "response.raw.txt").exists() or "response.raw.txt" in files:
                        generation = ref_json({"path": str(bundle / "generation.json"), "sha256": files["generation.json"]}) if "generation.json" in files else None
                        shared_raw_response(bundle, files, generation)
                    need(window["caption"] == "" and not window["events"] and type(receipt["error"]) is str and receipt["error"], "failed shared VLM window silently repaired")
                need(type(window["model_call_performed"]) is bool, "shared VLM model call flag not boolean")
            model_calls += window["model_call_performed"]
            counts[status] += 1
        full += all(w["status"] == "candidate" and type(w["caption"]) is str and bool(w["caption"].strip()) for w in windows)
    need(sum(counts.values()) == 22 and same(summary["window_status_counts"], {k: counts[k] for k in ("candidate", "abstain", "failed", "prepared")}), "shared VLM actual window status counts differ")
    need(summary["VLM_shared_manifest_actual_execution"] is (model_calls > 0), "shared model-execution summary flag contradicts actual calls")
    integer(summary["model_executed_window_count"], model_calls)
    integer(summary["insufficient_frame_abstention_count"], short)
    integer(summary["usable_complete_strategy_source_count"], full)
    need(summary["all_ready_windows_without_failures"] is (counts["failed"] == 0), "shared VLM failure flag differs")
    return {"sources": 3, "strategy_sources": 9, "windows": 22, "model_calls": model_calls, "short_frame_abstentions": short, "usable_strategy_sources": full, "window_status_counts": dict(counts), "full_pool_updated": False}


def check_scope(section, contract, offline):
    tasks = section.get("tasks")
    if not tasks:
        raise Pending("explicit final required scope absent")
    needed = {"ms12k", "video_first12k", "eval15", "gitpackage", "resources", "rgb32points", "beat50"}
    need(set(tasks) >= needed, "new user-required RGB32point/beat50 scope omitted")
    needed.update(LATEST_SCOPE_TRIALS + SHARED_VLM_REQUIRED)
    details = {}
    pending = [name + ":missing_actual_contract" for name in LATEST_SCOPE_TRIALS + SHARED_VLM_REQUIRED if name not in tasks]
    for name, task in tasks.items():
        need(type(task.get("required")) is bool, "scope required flag not boolean")
        status = task.get("status")
        need(status in ("verified", "pending", "blocked", "superseded", "optional"), "unknown/fictional scope completion status")
        if name in needed:
            need(task["required"] is True, "user-required task made optional")
        if status == "superseded":
            need(name == "old6kfusion" and type(task.get("superseded_by")) is str and task["superseded_by"] in tasks, "unjustified superseded required task")
            details[name] = {"status": "superseded", "executed": False}
        elif status == "optional":
            need(task["required"] is False, "required task labeled optional")
            details[name] = {"status": "optional", "executed": False}
        elif status in ("pending", "blocked"):
            details[name] = {"status": status, "executed": False}
            if task["required"]:
                pending.append(name + ":" + status)
        else:
            need(task.get("actual_executed") is True, "verified scope task not actually executed")
            evidence = task.get("evidence")
            if not evidence:
                raise Pending("actual scope task evidence absent: " + name)
            actual = ref_json(evidence)
            if name != "shared_music_windows_vlm":
                need(actual.get("complete") is True or actual.get("status") in ("verified", "terminal_verified_unreviewed"), "scope evidence is not a real terminal receipt")
            if name == "freshdetector32":
                check_freshdetector32(task)
                check_freshdetector32_comparison(task)
                check_freshdetector32_caption_control(task)
            if name == "bpm_windows":
                check_bpm_windows(task)
                if EXACT_MUSIC_PTS_REQUIRED:
                    check_music_exact_pts(task, ("bpm",))
            if name == "truebar":
                need(actual.get("schema") == "motionscript_music_window_caption_v1" and actual.get("strategy") == "bar", "old C/BPM/BeatThis detections cannot substitute truebar MS consumer")
                check_truebar(task)
                if EXACT_MUSIC_PTS_REQUIRED:
                    check_music_exact_pts(task, ("strict", "merge_short_edges"))
            if name == "shared_music_windows_vlm":
                check_shared_music_windows_vlm(task)
            if name == "beat50":
                need(actual.get("schema") == "ms_beat_C_matched50_independent_verification_v1", "wrong actual matched50 beat verification schema")
                integer(actual["processed_source_count"], 50)
                window_count = integer(actual["C_window_count"])
                need(set(actual["C_status_counts"]) <= {"candidate", "abstain"} and sum(integer(v) for v in actual["C_status_counts"].values()) == window_count, "beat50 terminal counts differ")
                distribution = actual["C_windows_per_clip_distribution"]
                need(sum(integer(v) for v in distribution.values()) == 50 and sum(int(k) * integer(v) for k, v in distribution.items()) == window_count, "beat50 window/source denominators differ")
                need(same(actual["strategy_counts"], {"detected_beats": 50}), "beat50 not actual detected-beat source trial")
                no_approval(actual, ("formal_approval", "semantic_approval", "motions_on_beats_verified"))
                for key in ("all_native_audio_length_and_source_bytes_matched", "all_C_coords_identical_to_A_B_coords_sha", "all_detected_internal_boundaries_match_actual_estimator_beats", "all_half_open_original_MS_support_verified"):
                    need(actual[key] is True, "beat50 independent source/time gate absent")
                sha(actual["summary_sha256"])
                sha(actual["chunks_sha256"])
                root = Path(evidence["path"]).parent
                need(digest_file(root / "summary.json") == actual["summary_sha256"] and digest_file(root / "chunks.jsonl") == actual["chunks_sha256"], "actual beat50 outputs replaced")
            if name == "rgb32points" and actual.get("schema") == "nlf-direct32-inframe-control-terminal-v1":
                need(evidence["sha256"] == RGB_CONTROL_SHA, "RGB32 control is not frozen actual terminal")
                integer(actual["fixed3_count"], 3)
                for key in ("original_events_preserved", "immutable_operators_used", "rejected_points_not_gap_filled", "redshirt_outside_foot_events_blocked", "inframe_occlusion_and_heel_location_not_resolved"):
                    need(actual[key] is True, "RGB32 necessary-support terminal proof missing")
                no_approval(actual, ("mask_is_visibility_estimate", "raw_geometry_reextracted", "geometry_or_caption_accuracy_advantage_established", "calibrated_accuracy_claim", "training_approved", "source_modified", "gpu_used"))
                review = ref_json(task.get("extraction_review"), RGB_REVIEW_SHA)
                need(review["schema"] == "nlf-direct32-fixed3-real-extraction-review-delivery-v1", "RGB32 real extraction/overlay companion schema differs")
                for key in ("three_real_raw_verified", "three_cpu_overlays_generated_and_decoded", "three_whole_and_fixed2s_descriptions_verified", "real_rgb_points_not_template"):
                    need(review[key] is True, "RGB32 extraction/overlay/description not actually delivered")
                no_approval(review, ("accuracy_completed", "new_points_better_than_complete_old_pose_proven", "raw_is_ground_truth", "description_training_approved", "full_video_human_review_completed", "source_or_full_vlm_modified"))
                need(review["actual_spec_sha256"] == RGB_SPEC and review["actual_model_sha256"] == RGB_MODEL and review["extractor_source_sha256"] == RGB_SOURCE, "RGB32 actual source/spec/model differ")
                need(len(actual["cases"]) == len(review["cases"]) == 3 and {r["id"] for r in actual["cases"]} == IDS and {r["id"] for r in review["cases"]} == IDS, "RGB32 control/review fixed cases differ")
                originals = {r["id"]: r for r in review["cases"]}
                for path, checksum in actual["implementation_sha256"].items():
                    need(digest_file(path) == sha(checksum), "RGB32 immutable operator source changed")
                for case in actual["cases"]:
                    original = originals[case["id"]]
                    need(case["source_manifest_sha256"] == original["raw_manifest_sha256"] and case["source_raw_sha256"] == original["raw_npz_sha256"], "RGB32 derived control changed original raw source")
                    integer(case["numeric_valid_point_frames"], 2592)
                    valid = integer(case["necessary_support_point_frames"])
                    excluded = integer(case["excluded_point_frames"])
                    need(valid + excluded == 2592 and sum(integer(v) for v in case["excluded_by_point"].values()) == excluded, "RGB32 necessary support denominator differs")
                    need(case["original_no_gap_baseline_exact_equal"] is True and case["no_rejected_point_gap_fill"] is True, "RGB32 original baseline changed or missing points filled")
                    report = ref_json({"path": case["report_path"], "sha256": case["report_sha256"]})
                    need(report["schema"] == "nlf-direct32-inframe-necessary-support-candidates-v1" and report["id"] == case["id"], "RGB32 actual derived report schema/case differs")
                    for key in ("source_manifest_sha256", "source_raw_sha256", "masks_sha256", "derived_coords_sha256", "numeric_valid_point_frames", "necessary_support_point_frames", "excluded_point_frames", "original_event_count", "derived_event_count", "derived_foot_event_count"):
                        need(same(report[key], case[key]), "RGB32 control/report typed binding differs: " + key)
                    need(report["target_valid_requires_both_raw_brackets"] is True and report["no_rejected_point_gap_fill"] is True and report["original_no_gap_coords_exact_equal"] is True, "RGB32 necessary-support gap policy not verified")
                    no_approval(report, ("precise_geometry_approved", "caption_is_training_target", "semantic_approval", "raw_modified"))
                    integer(report["derived_event_count"], len(report["events"]))
                    if case["id"] == "wJPrpFsThWAcBQIw":
                        integer(case["derived_foot_event_count"], 0)
            if name == "rgb32points" and actual.get("schema") != "nlf-direct32-inframe-control-terminal-v1":
                extractor = task.get("extractor")
                if not extractor or not extractor.get("sha256"):
                    raise Pending("actual RGB32 extractor source reference absent")
                need(extractor["sha256"] == RGB_SOURCE and digest_file(extractor["path"]) == RGB_SOURCE, "RGB32 not original actual extractor source")
                integer(actual["expected"], 3)
                need(actual["source_unchanged"] is True and actual["spec_unchanged"] is True and actual["full_VLM_untouched"] is True, "RGB32 actual execution source changed")
                need(actual["accuracy_approved"] is False, "raw finite values relabeled RGB32 accuracy approval")
                records = actual["records"]
                need(len(records) == 3 and {r["id"] for r in records} == IDS, "RGB32 not actual fixed three source cases")
                reviews = task.get("reviews")
                descriptions = task.get("descriptions")
                if not reviews or not descriptions:
                    raise Pending("RGB32 three CPU overlays and same-window description receipts absent")
                need(len(reviews) == len(descriptions) == 3 and {r["id"] for r in reviews} == IDS and {r["id"] for r in descriptions} == IDS, "RGB32 incomplete review/description cases")
                review_by_id = {r["id"]: r["receipt"] for r in reviews}
                description_by_id = {r["id"]: r for r in descriptions}
                for record in records:
                    sid = record["id"]
                    need(record["complete"] is True and record["finite_valid"] is True, "RGB32 source extraction not actual finite terminal")
                    no_approval(record, ("accuracy_approved", "smpl_fitting_used", "template_fk_used"))
                    need(same(record["shapes"]["points_m"], [81, 32, 3]) and same(record["shapes"]["points2d_px"], [81, 32, 2]) and same(record["shapes"]["valid"], [81, 32]), "RGB32 actual shape differs")
                    integer(record["valid_points"], 2592)
                    sha(record["raw_sha256"])
                    need(record["source_sha256"]["spec"] == RGB_SPEC and record["source_sha256"]["model"] == RGB_MODEL, "RGB32 canonical spec/model source differs")
                    review = ref_json(review_by_id[sid])
                    need(review["schema"] == "nlf-direct32-same-window-review-v1" and review["actual_direct32_loaded"] is True, "not actual direct32 CPU overlay review")
                    integer(review["source_frames"], 81)
                    need(number(review["source_fps"]) == 16, "RGB32 review source time differs")
                    need(review["bundle_manifest_sha256"] == record["manifest_sha256"] and review["inputs"]["raw"]["sha256"] == record["raw_sha256"], "RGB32 overlay source manifest/raw binding differs")
                    manifest = ref_json(review["inputs"]["bundle_manifest"])
                    need(manifest["source"] == "nlf_direct_rgb_landmarks" and type(manifest["point_count"]) is int and manifest["point_count"] == 32, "RGB32 overlay uses template/foreign points")
                    need(same(manifest["source_sha256"], record["source_sha256"]), "RGB32 extractor/overlay source SHA changed")
                    need(review["legacy_full_pose72_preserved"] is True and review["legacy_global_rotation24_saved"] is True, "RGB32 legacy comparison omitted actual full pose")
                    no_approval(review, ("smpl_fk_used_for_new_points", "projection_residual_is_accuracy_metric", "source_frames_filled_or_interpolated", "semantic_approval", "accuracy_review_completed", "gpu_used"))
                    overlay = review["artifacts"]["rgb_direct32_vs_legacy_pose.mp4"]
                    sha(overlay["sha256"])
                    need(safe_file(overlay["path"]).stat().st_size > 0, "RGB32 overlay artifact missing/empty")
                    reports = []
                    for mode in ("whole", "fixed2s"):
                        receipt_ref = description_by_id[sid].get(mode)
                        receipt = ref_json(receipt_ref)
                        report_path = Path(receipt_ref["path"]).parent / "report.json"
                        report = ref_json({"path": str(report_path), "sha256": receipt["report_sha256"]})
                        need(report["schema"] == "motionscript_direct_landmarks_v1" and report["source"] == "nlf_direct_rgb_landmarks", "RGB32 description source/schema differs")
                        integer(report["point_count"], 32)
                        need(report["source_manifest_sha256"] == record["manifest_sha256"] and report["raw_sha256"] == record["raw_sha256"], "RGB32 description not actual source raw")
                        no_approval(report, ("semantic_approval", "new_information_accuracy_validated"))
                        reports.append(report)
                    need(reports[0]["coords_sha256"] == reports[1]["coords_sha256"], "RGB32 whole/fixed descriptions use different complete coordinates")
                    fixed = reports[1]
                    need(len(fixed["chunks"]) == 2 and fixed["window_spec"]["strategy"] == "fixed", "RGB32 not matched two fixed-window description")
                    windows = fixed["window_spec"]["windows"]
                    need(same(windows, [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": number(review["media_bounds"]["end"])}]), "RGB32 description windows not actual shared global media time")
                    for chunk, window in zip(fixed["chunks"], windows):
                        need(same(chunk["window"], window), "RGB32 chunk window changed")
                        no_approval(chunk, ("semantic_approval", "caption_is_training_target"))
            if name == "rgb32points" and "support_control" in task and actual.get("schema") != "nlf-direct32-inframe-control-terminal-v1":
                control = ref_json(task["support_control"], RGB_CONTROL_SHA)
                need(control["schema"] == "nlf-direct32-inframe-control-terminal-v1" and control["complete"] is True, "RGB32 necessary-support control not actual terminal")
                integer(control["fixed3_count"], 3)
                for key in ("original_events_preserved", "immutable_operators_used", "rejected_points_not_gap_filled", "redshirt_outside_foot_events_blocked"):
                    need(control[key] is True, "RGB32 support control proof missing")
                no_approval(control, ("source_modified", "raw_geometry_reextracted", "mask_is_visibility_estimate", "geometry_or_caption_accuracy_advantage_established", "calibrated_accuracy_claim", "training_approved"))
                need(len(control["cases"]) == 3 and {r["id"] for r in control["cases"]} == IDS, "RGB32 support control case IDs differ")
                originals = {r["id"]: r for r in actual["records"]}
                for case in control["cases"]:
                    original = originals[case["id"]]
                    need(case["source_manifest_sha256"] == original["manifest_sha256"] and case["source_raw_sha256"] == original["raw_sha256"], "RGB32 support control uses different original source")
                    need(integer(case["numeric_valid_point_frames"]) == original["valid_points"], "RGB32 control numeric raw denominator differs")
                    need(integer(case["necessary_support_point_frames"]) + integer(case["excluded_point_frames"]) == original["valid_points"], "RGB32 support/excluded denominator differs")
                    if case["id"] == "wJPrpFsThWAcBQIw":
                        integer(case["derived_foot_event_count"], 0)
                        integer(case["necessary_support_point_frames"], 1575)
            details[name] = {"status": "verified", "executed": True}
    if pending:
        raise Pending("required scope pending: " + ", ".join(pending))
    return {"tasks": details, "optional_is_not_executed": True}


def verify(contract, offline=False):
    need(type(contract) is dict, "contract must be a JSON object")
    checks = {}
    contract_error = None
    try:
        need(contract.get("schema") == CONTRACT_SCHEMA and contract.get("feature_enabled") is True, "v3 contract feature not explicitly enabled")
        need(same(contract.get("required_checks"), list(CHECKS)), "all six checks must be explicitly required")
    except ValueError as error:
        contract_error = str(error)
    for name in CHECKS:
        section = contract.get(name)
        try:
            if not section:
                raise Pending("explicit check contract absent")
            result = globals()["check_" + name](section, contract, offline)
            checks[name] = {"status": "verified", "executed": True, "details": result}
        except Pending as error:
            checks[name] = {"status": "pending", "executed": True, "reason": str(error)}
        except (ValueError, TypeError, KeyError, OSError) as error:
            checks[name] = {"status": "rejected", "executed": True, "reason": str(error)}
    statuses = [r["status"] for r in checks.values()]
    ledger_complete = not contract_error and all(s == "verified" for s in statuses)
    rejected = bool(contract_error) or "rejected" in statuses
    vlm = checks["vlm12k"]
    caption_complete = vlm["status"] == "verified" and vlm["details"]["caption_delivery_complete"] is True
    usable_full = vlm["details"]["usable_full_caption_count"] if vlm["status"] == "verified" else None
    goal_complete = ledger_complete and caption_complete
    required_pending = [name for name, check in checks.items() if check["status"] == "pending"]
    if vlm["status"] == "verified" and not caption_complete:
        required_pending.append("caption_delivery")
    status = "rejected" if rejected else "verified" if goal_complete else "pending"
    quality = contract.get("human_quality", "unknown")
    need(quality in ("unknown", "negative", "positive_partial"), "human quality cannot be silently approved")
    return {"schema": SCHEMA, "status": status, "contract_error": contract_error, "checks": checks, "all_required_checks_executed": True, "terminal_ledger_complete": ledger_complete, "caption_delivery_complete": caption_complete, "usable_full_caption_count": usable_full, "partial_or_unusable_count": COUNT - usable_full if usable_full is not None else None, "technical_complete": ledger_complete, "human_quality_pending": True, "technical_goal_complete": goal_complete, "required_pending": required_pending, "completion_scope": "finite_technical_ledger_and_caption_delivery_not_human_quality", "human_quality": quality, "human_quality_approved": False, "offline": offline, "producers_started": False, "source_media_or_weights_read": False}


def main(argv=None):
    source_path = Path(__file__).resolve()
    source_sha = digest_file(source_path)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)
    contract_path = safe_file(args.contract)
    need(contract_path.stat().st_size <= 1024 * 1024, "contract must be small JSON")
    contract_content = contract_path.read_bytes()
    contract = parse(contract_content)
    out = Path(args.out).absolute()
    need(not out.exists() and not out.is_symlink(), "output must be new exclusive JSON")
    need(out.suffix == ".json" and out.parent.is_dir() and not any(p.is_symlink() for p in out.parents), "unsafe/nonexistent output parent")
    def protect(value):
        if isinstance(value, dict):
            if "path" in value and isinstance(value["path"], str):
                source = Path(value["path"]).absolute()
                need(out != source and not source.is_relative_to(out) and not out.is_relative_to(source.parent), "output overlaps input/export evidence tree")
            for child in value.values():
                protect(child)
        elif isinstance(value, list):
            for child in value:
                protect(child)
    need(out != contract_path.absolute(), "output overlaps contract")
    protect(contract)
    result = verify(contract, args.offline)
    source_sha_after = digest_file(source_path)
    result.update(checker_source_path=str(source_path), checker_source_sha256=source_sha, checker_source_sha256_after=source_sha_after, checker_source_unchanged=source_sha_after == source_sha, contract_sha256=hashlib.sha256(contract_content).hexdigest())
    if source_sha_after != source_sha:
        result.update(status="rejected", technical_complete=False, technical_goal_complete=False, rejection_reason="checker source changed during execution; not a final verification")
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "out": str(out), "terminal_ledger_complete": result["terminal_ledger_complete"], "caption_delivery_complete": result["caption_delivery_complete"]}))
    return 0 if result["technical_goal_complete"] is True and result["status"] == "verified" else 1 if result["status"] == "rejected" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, Pending, OSError, TypeError, KeyError) as error:
        print(json.dumps({"status": "rejected", "reason": str(error)}), file=sys.stderr)
        sys.exit(1)
