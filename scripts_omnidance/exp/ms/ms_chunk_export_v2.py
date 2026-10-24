"""Verify canonical A/B output bytes, including immutable v1 exports (v2).

Never starts captioning, modifies producers, repairs masks, hashes source media,
or turns an abstention into an action. Pending producers produce no export.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from ms_chunk_caption import CLEAN_RULES, SCHEMA as INPUT_SCHEMA, load_manifest, neutral_clean, parser as producer_parser
from motionscript_v2.chunks import fixed_windows
from ms_optimize import atomic_write, json_bytes, overlaps

SCHEMA = "motionscript_verified_A_B_export_v1"
VERIFICATION_SCHEMA = "motionscript_existing_export_verification_v2"
V1_SOURCE_SHA256 = "5fcf77e46a028d55a574d2dc22c7656c1bd0e8b81b462cb1a1bdc9cd8a1e5b34"
_FALSE_FLAGS = ("accepted", "semantic_approval", "human_review_completed")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_artifact(path, *, expected_sha=None):
    require(not path.is_symlink(), f"artifact is a symlink: {path}")
    content = path.read_bytes()
    if expected_sha is not None:
        require(digest(content) == expected_sha, f"altered artifact: {path}")
    return content


def read_json(path, *, expected_sha=None):
    return json.loads(read_artifact(path, expected_sha=expected_sha))


def false_flags(record, *, accepted=False):
    for key in (("accepted",) if accepted else ()) + ("semantic_approval", "human_review_completed"):
        require(record.get(key) is False, f"approval flag must be explicitly false: {key}")


def hash_field(record, key):
    value = record.get(key)
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"invalid {key}")
    return value


def launch_context(root):
    launch_bytes = read_artifact(root / "launch.json")
    launch = json.loads(launch_bytes)
    jobs = launch.get("jobs")
    nshard = launch.get("shards")
    require(isinstance(nshard, int) and nshard > 0 and isinstance(jobs, list) and len(jobs) == nshard, "invalid launch shards/jobs")
    require(sorted(job["shard"] for job in jobs) == list(range(nshard)), "launch shard IDs must be unique and complete")
    false_flags(launch)
    contexts = []
    for job in sorted(jobs, key=lambda row: row["shard"]):
        argv = job["argv"]
        script = next((i for i, value in enumerate(argv) if Path(value).name == "ms_chunk_caption.py"), None)
        require(script is not None, "launch argv has no original caption producer")
        args = producer_parser().parse_args(argv[script + 1:])
        shard = job["shard"]
        require(args.shard == shard and args.nshard == nshard and args.limit == 0, "launch is not a complete limit0 shard selection")
        directory = root / f"shard{shard}"
        require(Path(args.out).resolve() == directory.resolve(), "launch output differs from canonical shard directory")
        require(str(Path(args.manifest).resolve()) == str(Path(launch["input_manifest"]).resolve()), "launch manifest paths disagree")
        require(digest(read_artifact(Path(argv[script]))) == launch["source_sha256"], "frozen producer source differs from launch")
        config = {name: getattr(args, name) for name in (
            "chunk_seconds", "min_tail_seconds", "max_gap_seconds", "smooth_seconds",
            "max_joint_uncertainty", "shard", "nshard", "limit", "ffprobe", "workers")}
        contexts.append({"shard": shard, "directory": directory, "config": config})
    require(sum(ctx["config"]["workers"] for ctx in contexts) == launch["actual_workers"], "launch worker count differs")
    return launch, digest(launch_bytes), contexts


def readiness(contexts):
    pending = []
    for context in contexts:
        directory = context["directory"]
        reasons = []
        if (directory / ".run.lock").exists():
            reasons.append("producer_lock_present")
        for name in ("protocol.json", "manifest.json", "manifest.receipt.json", "chunks.jsonl"):
            if not (directory / name).is_file():
                reasons.append("missing_" + name)
        if (directory / "resume.errors.json").exists():
            reasons.append("rejected_resume_requires_operator_review")
        if reasons:
            pending.append({"shard": context["shard"], "reasons": reasons})
    return pending


def usable_spans(arrays, selected):
    valid = arrays["valid"][selected] & arrays["joint_valid"][selected].all(axis=1)
    root = arrays["segment_id"][selected]
    joint = arrays["joint_segment_id"][selected]
    valid &= (root >= 0) & (joint >= 0).all(axis=1)
    same = (root[1:] == root[:-1]) & (joint[1:] == joint[:-1]).all(axis=1)
    starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1] & same])
    stops = np.flatnonzero(valid & ~np.r_[valid[1:] & same, False]) + 1
    return [selected[start:stop] for start, stop in zip(starts, stops) if stop - start >= 4]


def verify_caption(view, arrays, start, end):
    false_flags(view)
    require(view["window_start"] == start and view["window_end"] == end and view["window_interval"] == "[start,end)", "wrong planned window")
    require(view.get("time_basis") == "global_media_seconds", "rebased window time")
    require(view.get("coordinate_reset_applied") is False and view.get("segment_ids_preserved") is True, "reset or repaired window coordinates")
    require(view.get("source_arrays_retained_full_length") is True, "source arrays were not retained")
    require(view.get("sample_span_is_complete_window_evidence") is False and view.get("window_boundaries_are_action_boundaries") is False, "window/sample support relabeled as action boundaries")
    time = arrays["time"]
    selected = np.flatnonzero((time >= start) & (time < end))
    source_selected = np.flatnonzero((arrays["source_time"] >= start) & (arrays["source_time"] < end))
    require(view["selected_target_indices"] == selected.tolist(), "half-open target indices differ")
    require(view["source_frame_indices_in_window"] == source_selected.tolist(), "half-open raw source indices differ")
    require(view["selected_sample_count"] == len(selected), "sample count differs")
    require(view["first_sample_time"] == (float(time[selected[0]]) if len(selected) else None), "first supported sample differs")
    require(view["last_sample_time"] == (float(time[selected[-1]]) if len(selected) else None), "last supported sample differs")
    spans = usable_spans(arrays, selected)
    raw = view["original_captions"]
    supports = view["supported_descriptions"]
    require(len(raw) == len(supports) == len(spans), "upstream captions stitch or omit usable support segments")
    texts = []
    for segment, support, indices in zip(raw, supports, spans):
        first, last = float(time[indices[0]]), float(time[indices[-1]])
        require(segment["start"] == support["start"] == first and segment["end"] == support["end"] == last, "upstream support time differs")
        require(segment["source"] == "original_motionscript_rules_on_v2_coordinates", "wrong raw caption source")
        require(support["target_indices"] == indices.tolist() and support["sample_count"] == len(indices), "support indices cross a gap")
        for key in ("ms", "ms_nonagg"):
            require(isinstance(segment[key], list) and all(isinstance(item, str) for item in segment[key]), "raw original MS text missing")
        text = " ".join(neutral_clean(item) for item in segment["ms"] if item.strip())
        require(support["text"] == text, "cleaned support added or changed source actions")
        if text:
            texts.append(text)
    usable = arrays["valid"][selected] & arrays["joint_valid"][selected].all(axis=1)
    source_usable = arrays["source_valid"][source_selected] & arrays["source_joint_valid"][source_selected].all(axis=1)
    continuous = bool(len(selected) and len(spans) == 1 and usable.all() and source_usable.all() and len(spans[0]) == len(selected))
    partial = bool(spans) and not continuous
    reason = "no_samples_in_window" if not len(selected) else "partial_support" if partial else "no_output" if not texts else None
    candidate = bool(texts) and continuous
    require(view["status"] == ("candidate" if candidate else "abstain") and view["abstain_reason"] == reason, "candidate/partial/no_output status differs")
    require(view["usable_target_sample_count"] == int(usable.sum()), "usable sample count differs")
    require(view["support_status"] == ("partial" if partial else "complete_available_samples" if continuous else "none"), "support category differs")
    body = " ".join(texts)
    expected_text = f"[{start:.6f}, {end:.6f}) seconds: {body}" if candidate else ""
    require(view["cleaned_motion_text"] == expected_text, "candidate text is not independently supported by this mode/window")
    require(view["partial_supported_motion_text"] == (body if partial else ""), "partial text not separated from final candidate")
    require(view["clean_rules"] == CLEAN_RULES, "cleaning rules differ")
    return {"source": "original_motionscript_rules_on_v2_coordinates", "mode": view.get("mode", "B_fixed_window"),
            "window_start": start, "window_end": end, "status": view["status"], "abstain_reason": reason,
            "support_status": view["support_status"], "candidate_text_eligible": candidate,
            "raw_original_description_nonempty": any(item.strip() for segment in raw for item in segment["ms"]),
            "motion_caption": view["cleaned_motion_text"] if candidate else None,
            "formal_training_eligible": False, "formal_approval": False}


def verify_report(report, expected, protocol, coords_content):
    require(report.get("schema") == INPUT_SCHEMA and report.get("id") == expected["id"], "wrong report schema/id")
    false_flags(report, accepted=True)
    require(report.get("source") == "original_motionscript_rules_on_v2_coordinates", "report is not original MS source")
    require(Path(report["input_path"]).resolve() == Path(expected["nlf_path"]).resolve(), "report input_path differs from expected NLF path")
    require(report.get("dataset_processed") is True, "report not actually processed")
    require(report["source_group"] == expected["source_group"] and report["split"] == expected["split"], "source_group/split differ from expected")
    require(report["manifest_record"] == expected, "source manifest context differs")
    require(report["fingerprint"] == protocol["fingerprint"] and report["config_hash"] == protocol["protocol"]["config_hash"], "report protocol binding differs")
    for key in ("input_sha256", "video_sha256", "audio_sha256", "coords_sha256"):
        hash_field(report, key)
    require(digest(coords_content) == report["coords_sha256"], "altered coordinates cache")
    media, quality = report["media"], report["quality"]
    for key in ("media_end_verified_from_video", "nlf_video_frame_count_matched", "nlf_video_timestamps_matched", "nlf_video_fps_matched"):
        require(media.get(key) is True, "actual media/time gate missing: " + key)
    require(media["media_start"] == 0 and media["timestamp_tolerance_seconds"] == 1e-5, "wrong actual time basis")
    require(media.get("cache_video_identity_verified") is False and "unknown" in media["audio_offset_verification"], "cache identity/audio alignment falsely approved")
    with np.load(io.BytesIO(coords_content), allow_pickle=False) as archive:
        names = ("time", "source_time", "valid", "joint_valid", "segment_id", "joint_segment_id",
                 "source_valid", "source_joint_valid", "source_detection_valid", "observed", "interpolated", "coords")
        arrays = {name: archive[name] for name in names}
    time, source_time = arrays["time"], arrays["source_time"]
    require(time.ndim == source_time.ndim == 1 and np.isfinite(time).all() and np.isfinite(source_time).all(), "invalid time arrays")
    require((np.diff(time) > 0).all() and (np.diff(source_time) > 0).all(), "nonmonotonic times")
    require(len(time) == quality["output_frames"] and len(source_time) == quality["source_frames"] == media["decoded_video_frames"], "full coordinate/source counts differ")
    require(arrays["coords"].shape == (len(time), 54, 3), "full original coordinate layout differs")
    require(arrays["valid"].shape == arrays["segment_id"].shape == (len(time),), "root support shape differs")
    require(arrays["joint_valid"].shape == arrays["joint_segment_id"].shape == (len(time), 22), "joint support shape differs")
    require(arrays["source_valid"].shape == (len(source_time),) and arrays["source_joint_valid"].shape == (len(source_time), 22), "full source support shape differs")
    for key in ("valid", "joint_valid", "source_valid", "source_joint_valid", "source_detection_valid", "observed", "interpolated"):
        require(arrays[key].dtype.kind == "b", "coordinate support mask must remain boolean: " + key)
    require(arrays["source_detection_valid"].shape == (len(source_time),), "original detection mask was truncated")
    for key, array in (("source_valid_frames", "source_valid"), ("valid_frames", "valid"),
                       ("observed_frames", "observed"), ("interpolated_frames", "interpolated"),
                       ("original_source_valid_frames", "source_detection_valid")):
        require(quality[key] == int(arrays[array].sum()), "quality/missing count differs: " + key)
    require(np.isclose(quality["source_valid_ratio"], arrays["source_valid"].mean())
            and np.isclose(quality["valid_ratio"], arrays["valid"].mean()), "quality support ratios differ")
    require(np.isclose(quality["output_fps"], 20) and np.allclose(np.diff(time), 0.05, rtol=0, atol=1e-8), "wrong full output fps")
    require(np.isclose(quality["source_fps"], media["video_fps"], rtol=0, atol=1e-5), "source/video fps evidence differs")
    require(len(source_time) > 0 and source_time[0] == media["video_first_sample"] and source_time[-1] == media["video_last_sample"], "raw source timestamps differ from actual media evidence")
    require(media["media_end"] > source_time[-1] and time[-1] <= source_time[-1] + 1e-8, "invented final media end or extrapolated coords")
    require(quality.get("hand_fingers_observed") is False and report.get("hand_fingers_observed") is False, "finger observations falsely claimed")
    complete_body = arrays["valid"] & arrays["joint_valid"].all(axis=1)
    require(np.isfinite(arrays["coords"][complete_body]).all(), "usable complete-body geometry is nonfinite")
    whole = report["whole_clip_original"]
    require(whole.get("mode") == "A_original_rules_whole_clip_same_coordinates", "A source/mode missing")
    A = verify_caption(whole, arrays, 0., media["media_end"])
    windows = fixed_windows(0, media["media_end"], protocol["protocol"]["config"]["chunk_seconds"], protocol["protocol"]["config"]["min_tail_seconds"])
    require(len(report["chunks"]) == len(windows), "planned B windows missing")
    B = []
    for index, (chunk, window) in enumerate(zip(report["chunks"], windows)):
        require(chunk["chunk_index"] == index, "wrong B window index")
        require(chunk.get("mode", "B_fixed_window") == "B_fixed_window", "B window mode is not canonical B_fixed_window")
        B.append(verify_caption(chunk, arrays, window["start"], window["end"]))
    require(report.get("mode") == "B_fixed_windows_same_whole_clip_coordinates", "B source/mode missing")
    require(report["status"] == ("candidate" if any(row["status"] == "candidate" for row in B) else "abstain"), "clip B status differs")
    return A, B


def coverage_summary(outputs):
    processed = [row for row in outputs.values() if row["status"] == "processed"]
    failed = [row for row in outputs.values() if row["status"] == "failed"]
    two_windows = [row for row in processed if len(row["B_windows"]) == 2]
    outcomes = Counter("|".join(window["status"] if window["status"] == "candidate" else window["abstain_reason"]
                               for window in row["B_windows"]) for row in processed)
    windows = [window for row in processed for window in row["B_windows"]]
    candidate_counts = Counter(str(sum(window["status"] == "candidate" for window in row["B_windows"])) for row in processed)
    return {
        "target_clip_denominator": len(outputs), "processed_clip_denominator": len(processed),
        "failed_clip_count": len(failed), "observed_B_window_denominator": len(windows),
        "failed_source_window_count": None if failed else 0,
        "A_raw_original_nonempty_clip_count": sum(row["A_whole_clip"]["raw_original_description_nonempty"] for row in processed),
        "A_raw_original_empty_clip_count": sum(not row["A_whole_clip"]["raw_original_description_nonempty"] for row in processed),
        "A_strict_candidate_clip_count": sum(row["A_whole_clip"]["status"] == "candidate" for row in processed),
        "B_two_window_clip_denominator": len(two_windows),
        "B_non_two_window_clip_count": len(processed) - len(two_windows),
        "B_two_windows_both_candidate_clip_count": sum(all(window["status"] == "candidate" for window in row["B_windows"]) for row in two_windows),
        "B_two_windows_only_one_candidate_clip_count": sum(sum(window["status"] == "candidate" for window in row["B_windows"]) == 1 for row in two_windows),
        "B_two_windows_all_abstain_clip_count": sum(all(window["status"] == "abstain" for window in row["B_windows"]) for row in two_windows),
        "B_all_planned_windows_candidate_clip_count": sum(row["B_complete_clip_motion_caption"] is not None for row in processed),
        "B_incomplete_candidate_coverage_clip_count": sum(row["B_complete_clip_motion_caption"] is None for row in processed),
        "B_at_least_one_no_output_clip_count": sum(any(window["abstain_reason"] == "no_output" for window in row["B_windows"]) for row in processed),
        "B_at_least_one_partial_support_clip_count": sum(any(window["abstain_reason"] == "partial_support" for window in row["B_windows"]) for row in processed),
        "B_candidate_window_count_per_clip": dict(candidate_counts), "B_ordered_window_outcomes": dict(outcomes),
        "B_raw_original_nonempty_window_count": sum(window["raw_original_description_nonempty"] for window in windows),
        "execution_error_type_counts": dict(Counter(row["error"]["error_type"] for row in failed)),
        "definitions": {
            "raw_nonempty": "original aggregated MS contains nonblank text; not verified dynamic actions or semantic quality",
            "B_complete_text": "only independently eligible B candidates covering every planned window, retaining each actual window header; never A fallback",
            "processed_is_not_complete_caption": True,
            "clip_reason_buckets_overlap": "at-least-one partial/no_output buckets may overlap; ordered outcomes are exclusive",
            "failed_window_count": "unknown for failed source; do not infer unprocessed window descriptions",
            "semantic_layer_scope": "original-MS engineering/source/time gate only; independent video-first semantic sources require separate review and source fields",
        },
        "formal_approval": False, "formal_training_eligible": False,
    }


def run(args):
    root = Path(args.run_root).resolve()
    launch, launch_sha, contexts = launch_context(root)
    pending = readiness(contexts)
    if pending:
        return {"schema": SCHEMA, "status": "pending", "complete": False, "all_shards_terminal": False,
                "exported": False, "pending_shards": pending, "formal_approval": False}
    expected_path = Path(args.manifest or launch["input_manifest"]).resolve()
    expected, expected_sha = load_manifest(expected_path)
    require(expected_sha == launch["input_manifest_sha256"] and len(expected) == launch["target_ids"], "expected manifest SHA/count differs from launch")
    all_expected = {row["id"]: row for row in expected}
    outputs, lineage, errors, snapshots = {}, {}, [], []
    common = None
    status_A, status_B, reason_A, reason_B = Counter(), Counter(), Counter(), Counter()
    for context in contexts:
        directory, shard = context["directory"], context["shard"]
        require(not (directory / ".run.lock").exists(), "producer restarted during verification")
        marker_bytes = read_artifact(directory / "protocol.json")
        marker = json.loads(marker_bytes)
        protocol = marker["protocol"]
        require(protocol["schema"] == INPUT_SCHEMA and marker["fingerprint"] == digest(json_bytes(protocol)), "altered shard protocol fingerprint")
        false_flags(protocol)
        selected_ids = [row["id"] for row in expected[shard::len(contexts)]]
        require(protocol["manifest_sha256"] == expected_sha and protocol["selected_ids"] == selected_ids, "shard expected selection differs")
        require(protocol["config"] == context["config"] and protocol["config_hash"] == digest(json_bytes(context["config"])), "shard config differs from launch")
        require(protocol["code_sha256"].get("ms_chunk_caption.py") == launch["source_sha256"], "shard producer code differs from launch")
        require(protocol["clean_rules"] == CLEAN_RULES and protocol["output_fps"] == 20., "shard clean/time protocol differs")
        comparable = {key: value for key, value in protocol.items() if key not in ("selected_ids", "config", "config_hash")}
        comparable["config"] = {key: value for key, value in protocol["config"].items() if key != "shard"}
        require(common is None or comparable == common, "shard upstream/template/code/version/config contexts differ")
        common = comparable
        receipt_bytes = read_artifact(directory / "manifest.receipt.json")
        receipt = json.loads(receipt_bytes)
        manifest_bytes = read_artifact(directory / "manifest.json", expected_sha=receipt["manifest.json_sha256"])
        chunk_bytes = read_artifact(directory / "chunks.jsonl", expected_sha=receipt["chunks.jsonl_sha256"])
        manifest = json.loads(manifest_bytes)
        false_flags(manifest, accepted=True)
        require(manifest["fingerprint"] == marker["fingerprint"] and manifest["config_hash"] == protocol["config_hash"], "terminal manifest fingerprint differs")
        require(manifest["input_manifest_count"] == len(expected) and manifest["selected_count"] == len(selected_ids), "terminal manifest selection/count differs")
        records, failures = manifest["records"], manifest["errors"]
        successful_ids = [row["id"] for row in records]
        failed_ids = [row["id"] for row in failures]
        require(len(successful_ids + failed_ids) == len(set(successful_ids + failed_ids)) and set(successful_ids + failed_ids) == set(selected_ids), "shard successful/failed union is not exact expected IDs")
        require(manifest["completed_count"] == len(records) and manifest["complete"] is (not failures) and manifest["dataset_processed"] is (not failures), "terminal success/error flags differ")
        chunk_rows = [json.loads(line) for line in chunk_bytes.decode("utf-8").splitlines() if line.strip()]
        indexed_chunks = {}
        for row in chunk_rows:
            key = (row["id"], row["chunk_index"])
            require(key not in indexed_chunks, "duplicate shard chunk row")
            indexed_chunks[key] = row
        expected_chunk_keys = set()
        shard_status = Counter()
        for record in records:
            sid = record["id"]
            require(record["report_file"] == f"{sid}.json", "unsafe/noncanonical report path")
            report_bytes = read_artifact(directory / record["report_file"], expected_sha=record["report_sha256"])
            report_receipt = read_json(directory / f"{sid}.receipt.json")
            require(report_receipt["report_sha256"] == digest(report_bytes), "per-ID report receipt differs")
            report = json.loads(report_bytes)
            require(report["coords_file"] == f"{sid}.coords.npz", "unsafe/noncanonical coords path")
            coords_bytes = read_artifact(directory / report["coords_file"], expected_sha=report["coords_sha256"])
            A, B = verify_report(report, all_expected[sid], marker, coords_bytes)
            require(sid not in outputs, "duplicate ID across terminal shards")
            outputs[sid] = {"schema": SCHEMA, "id": sid, "source_group": report["source_group"], "split": report["split"],
                            "status": "processed", "A_whole_clip": A, "B_windows": B,
                            "B_complete_clip_motion_caption": "\n".join(row["motion_caption"] for row in B) if B and all(row["candidate_text_eligible"] for row in B) else None,
                            "formal_approval": False,
                            "accepted": False, "semantic_approval": False, "human_review_completed": False}
            lineage[sid] = {"id": sid, "shard": shard, "status": "processed", "manifest_record": all_expected[sid],
                            "report_path": str(directory / record["report_file"]), "report_sha256": digest(report_bytes),
                            "report_receipt_sha256": digest(read_artifact(directory / f"{sid}.receipt.json")),
                            "coords_path": str(directory / report["coords_file"]), "coords_sha256": report["coords_sha256"],
                            "input_sha256": report["input_sha256"], "video_sha256": report["video_sha256"], "audio_sha256": report["audio_sha256"],
                            "producer_fingerprint": marker["fingerprint"], "config_hash": protocol["config_hash"],
                            "actual_media": report["media"], "quality": report["quality"],
                            "raw_MS_and_support_indices_retained_in_report": True,
                            "source_media_rehashed_by_exporter": False, "raw_NLF_reloaded_by_exporter": False,
                            "cache_video_identity_verified": False, "formal_approval": False}
            status_A[A["status"]] += 1
            if A["abstain_reason"]:
                reason_A[A["abstain_reason"]] += 1
            for row, original in zip(B, report["chunks"]):
                status_B[row["status"]] += 1
                shard_status[row["status"]] += 1
                if row["abstain_reason"]:
                    reason_B[row["abstain_reason"]] += 1
                key = (sid, original["chunk_index"])
                expected_chunk_keys.add(key)
                expected_row = {"schema": INPUT_SCHEMA, "id": sid, "source_group": report["source_group"], "split": report["split"],
                                "fingerprint": marker["fingerprint"], "config_hash": protocol["config_hash"],
                                "input_sha256": report["input_sha256"], "video_sha256": report["video_sha256"], "audio_sha256": report["audio_sha256"],
                                "coords_file": report["coords_file"], "coords_sha256": report["coords_sha256"], "accepted": False, **original}
                require(indexed_chunks.get(key) == expected_row, "aggregate raw caption/context differs from per-ID receipt")
        require(set(indexed_chunks) == expected_chunk_keys and manifest["chunk_count"] == len(expected_chunk_keys), "terminal chunk union/count differs")
        require(manifest["status_counts"] == dict(shard_status), "terminal candidate/abstain counts differ")
        for error in failures:
            sid = error["id"]
            require(not error.get("cached_report_rejected"), "rejected committed cache requires operator inspection")
            require(read_json(directory / f"{sid}.error.json") == error, "failed ID evidence differs from terminal manifest")
            require(sid not in outputs, "failed ID duplicates successful ID")
            errors.append({"id": sid, "shard": shard, **error})
            outputs[sid] = {"schema": SCHEMA, "id": sid, "source_group": all_expected[sid]["source_group"], "split": all_expected[sid]["split"],
                            "status": "failed", "A_whole_clip": None, "B_windows": None,
                            "B_complete_clip_motion_caption": None, "error": error,
                            "formal_approval": False, "accepted": False, "semantic_approval": False, "human_review_completed": False}
            lineage[sid] = {"id": sid, "shard": shard, "status": "failed", "manifest_record": all_expected[sid],
                            "error_path": str(directory / f"{sid}.error.json"),
                            "error_sha256": digest(read_artifact(directory / f"{sid}.error.json")), "formal_approval": False}
        snapshots.append({"shard": shard, "protocol_sha256": digest(marker_bytes), "manifest_sha256": digest(manifest_bytes),
                          "manifest_receipt_sha256": digest(receipt_bytes), "chunks_sha256": digest(chunk_bytes)})
    require(set(outputs) == set(all_expected) and len(outputs) == launch["target_ids"], "final unique ID union is not exact expected manifest")
    # A restarted producer or final-aggregate replacement invalidates this view.
    require(not readiness(contexts), "shard ceased to be terminal during verification")
    for context, snapshot in zip(contexts, snapshots):
        for name, key in (("protocol.json", "protocol_sha256"), ("manifest.json", "manifest_sha256"), ("manifest.receipt.json", "manifest_receipt_sha256"), ("chunks.jsonl", "chunks_sha256")):
            require(digest(read_artifact(context["directory"] / name)) == snapshot[key], "terminal snapshot changed during verification")
    serialize = lambda values: b"".join((json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for value in values)
    input_order = {row["id"]: index for index, row in enumerate(expected)}
    artifacts = {"captions.jsonl": serialize(outputs[row["id"]] for row in expected),
                 "lineage.jsonl": serialize(lineage[row["id"]] for row in expected),
                 "errors.jsonl": serialize(sorted(errors, key=lambda row: input_order[row["id"]]))}
    expected_file_sha256 = {name: digest(content) for name, content in artifacts.items()}
    binding = {"schema": SCHEMA, "launch_sha256": launch_sha, "input_manifest_sha256": expected_sha,
               "producer_source_sha256": launch["source_sha256"], "exporter_source_sha256": digest(Path(__file__).read_bytes()),
               "shards": snapshots, "target_count": len(expected), "formal_approval": False}
    fingerprint = digest(json_bytes(binding))
    summary = {"schema": SCHEMA, "fingerprint": fingerprint, "binding": binding,
               "status": "completed" if not errors else "completed_with_source_errors", "complete": not errors,
               "all_shards_terminal": True, "all_expected_ids_accounted": True, "unique_id_count": len(outputs),
               "successfully_processed_count": len(outputs) - len(errors), "failed_source_count": len(errors),
               "all_expected_successfully_processed": not errors, "A_clip_status_counts": dict(status_A),
               "A_clip_abstain_reasons": dict(reason_A), "B_window_status_counts": dict(status_B), "B_window_abstain_reasons": dict(reason_B),
               "caption_coverage": coverage_summary(outputs),
               "formal_approval": False, "formal_training_eligible": False, "accepted": False,
               "semantic_approval": False, "human_review_completed": False,
               "no_A_to_B_fallback": True, "no_abstain_to_static_action": True,
               "source_media_rehashed": False, "raw_NLF_reloaded": False, "exported": not args.check_only}
    if args.verify_existing:
        require(args.out is not None and not args.check_only, "--verify-existing requires --out, not --check-only")
        out = Path(args.out).resolve()
        require(not overlaps(out, root), "existing export must be separate from run root")
        require(not (out / ".export.lock").exists(), "existing exporter still holds its lock")
        saved_bytes = read_artifact(out / "complete.json")
        saved = json.loads(saved_bytes)
        first_source = saved["binding"]["exporter_source_sha256"]
        require(first_source in (V1_SOURCE_SHA256, binding["exporter_source_sha256"]), "unrecognized first-export source")
        historical_binding = {**binding, "exporter_source_sha256": first_source}
        expected_saved = {**summary, "binding": historical_binding, "fingerprint": digest(json_bytes(historical_binding)),
                          "file_sha256": expected_file_sha256, "exported": True}
        require(json_bytes(saved) == json_bytes(expected_saved), "existing export summary/sha differs from canonical producer reconstruction")
        require(read_json(out / "complete.receipt.json")["complete_sha256"] == digest(saved_bytes), "altered existing complete receipt")
        for name, content in artifacts.items():
            require(read_artifact(out / name, expected_sha=expected_file_sha256[name]) == content, "existing export bytes differ from canonical producer reconstruction: " + name)
        require(digest(read_artifact(out / "complete.json")) == digest(saved_bytes), "existing complete changed during verification")
        result = {"schema": VERIFICATION_SCHEMA, "complete": not errors, "status": "verified" if not errors else "verified_with_source_errors",
                  "existing_export_modified": False, "source_media_rehashed": False, "raw_NLF_reloaded": False,
                  "first_export_source_sha256": first_source, "verifier_source_sha256": binding["exporter_source_sha256"],
                  "producer_source_sha256": launch["source_sha256"], "input_manifest_sha256": expected_sha,
                  "existing_complete_path": str(out / "complete.json"), "existing_complete_sha256": digest(saved_bytes),
                  "canonical_file_sha256": expected_file_sha256, "exact_expected_bytes_verified": True,
                  "unique_id_count": len(outputs), "failed_source_count": len(errors), "caption_coverage": summary["caption_coverage"],
                  "source_identity_trust_boundary": "producer receipt agreement and declared paths/SHA only; no independent RGB-to-NLF causal identity proof",
                  "formal_approval": False, "formal_training_eligible": False, "accepted": False, "semantic_approval": False, "human_review_completed": False}
        if args.verification_receipt:
            receipt_path = Path(args.verification_receipt).resolve()
            require(not overlaps(receipt_path, out) and not overlaps(receipt_path, root), "verification receipt must be separate from historical export and producers")
            require(not receipt_path.is_symlink(), "verification receipt must not be a symlink")
            receipt_content = json_bytes(result)
            if receipt_path.exists():
                require(read_artifact(receipt_path) == receipt_content, "verification receipt already exists with different binding")
            else:
                atomic_write(receipt_path, receipt_content)
        return result
    if args.check_only:
        return summary
    summary["file_sha256"] = expected_file_sha256
    require(args.out is not None, "--out is required unless --check-only")
    out = Path(args.out).resolve()
    require(not overlaps(out, root) and out != expected_path and out not in expected_path.parents, "export output must be separate from source run/manifest")
    out.mkdir(parents=True, exist_ok=True)
    require(not any(path.is_symlink() for path in out.iterdir()), "output artifacts must not be symlinks")
    lock = out / ".export.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        final = out / "complete.json"
        if final.exists():
            require(args.resume, "export exists; use identical --resume or a new output")
            saved_bytes = read_artifact(final)
            saved = json.loads(saved_bytes)
            require(saved["fingerprint"] == fingerprint, "resume export binding differs")
            require(json_bytes(saved) == json_bytes(summary), "completed export summary/approval/sha differs from canonical producer reconstruction")
            require(set(saved["file_sha256"]) == {"captions.jsonl", "lineage.jsonl", "errors.jsonl"}, "noncanonical export artifact paths")
            require(read_json(out / "complete.receipt.json")["complete_sha256"] == digest(saved_bytes), "altered complete receipt")
            for name, content in artifacts.items():
                require(read_artifact(out / name, expected_sha=expected_file_sha256[name]) == content, "completed export bytes differ from canonical producer reconstruction: " + name)
            return saved
        require(not any(path.name != lock.name for path in out.iterdir()), "export output must be new/empty; inspect incomplete output before recovery")

        for name, content in artifacts.items():
            atomic_write(out / name, content)
        complete_bytes = json_bytes(summary)
        atomic_write(out / "complete.receipt.json", json_bytes({"complete_sha256": digest(complete_bytes)}))
        atomic_write(final, complete_bytes)  # Final verified-export commit marker.
        return summary
    finally:
        lock.unlink()


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", required=True, help="Producer run containing launch.json and shardN")
    ap.add_argument("--manifest", help="Expected immutable JSONL; defaults to launch input_manifest")
    ap.add_argument("--out", help="New independent final-export directory")
    ap.add_argument("--check-only", action="store_true", help="Pending/read-only validation, without exporting")
    ap.add_argument("--resume", action="store_true", help="Only completed exports matching newly reconstructed canonical bytes")
    ap.add_argument("--verify-existing", action="store_true", help="Read-only canonical validation of a v1/v2 export; never rewrite its complete.json")
    ap.add_argument("--verification-receipt", help="Optional independent verification JSON outside producer/export trees")
    return ap


if __name__ == "__main__":
    try:
        result = run(parser().parse_args())
        print(json.dumps(result, ensure_ascii=False), flush=True)
        sys.exit(0 if result["complete"] else 2 if result["status"] == "pending" else 1)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
