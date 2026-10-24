"""Source-bound BPM-duration or real downbeat windows, actually described by MS.

Independent finite CPU runner: reuse receipt-verified whole-clip coordinates;
never reset coordinates, infer bars from tempo, or substitute baseline text.
The exported window manifest is the common input for separate MS/VLM consumers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys

import numpy as np

SCHEMA = "motionscript_music_windows_v1"
REPORT_SCHEMA = "motionscript_music_window_caption_v1"
CONTEXT = None
ALIGNMENT = "normalized audio t=0 assumed to align normalized video t=0; original audio/video offset unknown and not independently verified"


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite typed number")
    if positive and value <= 0:
        raise ValueError(name + " must be positive")
    return float(value)


def digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


def binding_check(binding):
    if not isinstance(binding, dict):
        raise ValueError("binding must be an object")
    if not isinstance(binding.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", binding["id"]):
        raise ValueError("binding id must be a portable identifier")
    for key in ("input_sha256", "video_sha256", "audio_sha256", "actual_video_pts_sha256"):
        digest(binding.get(key), key)
    if number(binding.get("media_start"), "media_start") != 0:
        raise ValueError("normalized media must start at zero")
    number(binding.get("media_end"), "media_end", positive=True)
    number(binding.get("video_pts_offset"), "video_pts_offset")
    number(binding.get("video_fps"), "video_fps", positive=True)
    count = binding.get("decoded_video_frames")
    if type(count) is not int or count < 1:
        raise ValueError("decoded_video_frames must be a positive integer")


def base_plan(binding, strategy, raw, windows, params, provenance, *, status="ready"):
    binding_check(binding)
    return {"schema": SCHEMA, "id": binding["id"], "binding": dict(binding),
            "strategy": strategy, "status": status, "params": params, "provenance": provenance,
            "raw_windows": raw, "windows": windows, "window_interval": "[start,end)",
            "time_basis": "global_media_seconds", "continuous_exact_media_coverage": True,
            "audio_video_time_assumption": ALIGNMENT, "audio_video_alignment_verified": False,
            "beat_phase_verified": False, "downbeat_verified": False,
            "motions_on_beats_verified": False, "window_boundaries_are_action_boundaries": False,
            "semantic_approval": False, "human_review_completed": False}


def mapped_windows(raw):
    return [{"start": w["start"], "end": w["end"], "raw_window_indices": [i]} for i, w in enumerate(raw)]


def bpm_duration_windows(binding, bpm, *, beats_per_chunk=4, min_tail_seconds=1.5, provenance=None):
    binding_check(binding)
    bpm = number(bpm, "bpm", positive=True)
    if type(beats_per_chunk) is not int or beats_per_chunk < 1:
        raise ValueError("beats_per_chunk must be a positive integer")
    minimum = number(min_tail_seconds, "min_tail_seconds")
    if minimum < 0:
        raise ValueError("min_tail_seconds must be nonnegative")
    length = 60 * beats_per_chunk / bpm
    end = binding["media_end"]
    if not math.isfinite(length) or length <= 0 or math.ceil(end / length) > 100000:
        raise ValueError("BPM duration would produce an invalid/unbounded window grid")
    raw = []
    for i in range(math.ceil(end / length)):
        start, stop = i * length, min((i + 1) * length, end)
        if stop > start:
            raw.append({"start": start, "end": stop, "kind": "bpm_duration_grid",
                        "grid_index": i, "ends_at_actual_media_end": stop == end})
    windows = mapped_windows(raw)
    if len(windows) > 1 and windows[-1]["end"] - windows[-1]["start"] < minimum:
        tail = windows.pop()
        windows[-1]["end"] = tail["end"]
        windows[-1]["raw_window_indices"] += tail["raw_window_indices"]
    return base_plan(binding, "bpm_duration_grid_not_detected_beats", raw, windows,
                     {"bpm_estimate": bpm, "beats_per_chunk": beats_per_chunk,
                      "target_seconds": length, "min_tail_seconds": minimum,
                      "tail_policy": "merge_short_tail_with_previous; short_whole_clip_retained",
                      "first_zero_is_beat_phase": False}, provenance or {})


def actual_downbeat_source(source, binding):
    if not isinstance(source, dict) or source.get("method") != "beat_this.inference.File2Beats":
        raise ValueError("real bars require official beat_this.inference.File2Beats provenance")
    if source.get("checkpoint_name") != "final0" or source.get("dbn") is not False:
        raise ValueError("bar source requires official final0 with typed dbn=false")
    if source.get("device") != "cpu" or source.get("synthetic") is not False:
        raise ValueError("bar source must be actual CPU inference, not synthetic")
    if source.get("downbeats_origin") != "model_downbeat_head_postprocessed":
        raise ValueError("downbeats must be model outputs, not every-four-beats or tempo grid")
    for key in ("checkpoint_sha256", "audio_sha256", "library_source_sha256"):
        digest(source.get(key), key)
    if source["audio_sha256"] != binding["audio_sha256"]:
        raise ValueError("downbeat audio SHA differs from shared binding")
    if not isinstance(source.get("library_version"), str) or not source["library_version"].strip():
        raise ValueError("bar source requires library_version")
    return source


def bar_windows(binding, downbeats_seconds, *, source, policy="strict", min_edge_seconds=1.5):
    binding_check(binding)
    actual_downbeat_source(source, binding)
    if policy not in ("strict", "merge_short_edges"):
        raise ValueError("bar policy must be strict or merge_short_edges")
    minimum = number(min_edge_seconds, "min_edge_seconds")
    if minimum < 0:
        raise ValueError("min_edge_seconds must be nonnegative")
    if not isinstance(downbeats_seconds, list):
        raise ValueError("downbeats_seconds must be a typed list")
    db = [number(t, "downbeat") for t in downbeats_seconds]
    if any(t < 0 or t > binding["media_end"] for t in db) or any(b <= a for a, b in zip(db, db[1:])):
        raise ValueError("downbeats must be increasing and inside actual media")
    raw_bars = [{"start": a, "end": b, "bar_index": i, "downbeat_indices": [i, i+1]}
                for i, (a, b) in enumerate(zip(db, db[1:]))]
    raw = []
    if not db:
        raw.append({"start": 0., "end": binding["media_end"], "kind": "unresolved_no_downbeats", "bar_index": None})
    else:
        if db[0] > 0:
            raw.append({"start": 0., "end": db[0], "kind": "prefix_residual", "bar_index": None})
        raw.extend({**bar, "kind": "complete_detected_bar"} for bar in raw_bars)
        if db[-1] < binding["media_end"]:
            raw.append({"start": db[-1], "end": binding["media_end"], "kind": "suffix_residual", "bar_index": None})
    windows = mapped_windows(raw)
    if policy == "merge_short_edges" and raw_bars:
        if len(windows) > 1 and raw[0]["kind"] == "prefix_residual" and raw[0]["end"] < minimum:
            prefix = windows.pop(0)
            windows[0]["start"] = 0.
            windows[0]["raw_window_indices"] = prefix["raw_window_indices"] + windows[0]["raw_window_indices"]
        if len(windows) > 1 and raw[-1]["kind"] == "suffix_residual" and raw[-1]["end"] - raw[-1]["start"] < minimum:
            tail = windows.pop()
            windows[-1]["end"] = binding["media_end"]
            windows[-1]["raw_window_indices"] += tail["raw_window_indices"]
    plan = base_plan(binding, "model_downbeat_bars", raw, windows,
                     {"policy": policy, "min_edge_seconds": minimum,
                      "short_interior_bars_merged": False, "insufficient_downbeats_fallback": False},
                     source, status="ready" if len(db) >= 2 else "insufficient_downbeats")
    plan.update(downbeats_seconds=db, raw_bars=raw_bars,
                model_downbeat_timestamps_consumed=True, detected_downbeat_boundaries_verified_in_source=True)
    return plan


def validate_shared_plan(plan, expected_binding):
    """Reject altered/source-mismatched plans before either consumer uses them."""
    binding_check(expected_binding)
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("unknown shared window schema")
    binding_check(plan.get("binding"))
    if plan["binding"] != expected_binding or plan.get("id") != expected_binding["id"]:
        raise ValueError("shared window source binding mismatch")
    if plan.get("window_interval") != "[start,end)" or plan.get("time_basis") != "global_media_seconds":
        raise ValueError("shared window must use half-open global-media time")
    if any(plan.get(key) is not False for key in ("beat_phase_verified", "downbeat_verified", "audio_video_alignment_verified", "motions_on_beats_verified")):
        raise ValueError("unverified musical/alignment claims cannot be promoted")
    for key in ("raw_windows", "windows"):
        entries = plan.get(key)
        if not isinstance(entries, list) or not entries:
            raise ValueError(key + " must be a nonempty typed list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("window must be an object")
            start, end = number(entry.get("start"), "window_start"), number(entry.get("end"), "window_end")
            if start < 0 or end <= start or end > expected_binding["media_end"]:
                raise ValueError("window lies outside actual media")
            if key == "windows":
                mapping = entry.get("raw_window_indices")
                if not isinstance(mapping, list) or not mapping or any(type(i) is not int or i < 0 for i in mapping):
                    raise ValueError("raw window mapping must contain nonnegative typed integers")
    params = plan.get("params")
    if not isinstance(params, dict):
        raise ValueError("plan params must be an object")
    if plan.get("strategy") == "bpm_duration_grid_not_detected_beats":
        rebuilt = bpm_duration_windows(expected_binding, params.get("bpm_estimate"),
                    beats_per_chunk=params.get("beats_per_chunk"), min_tail_seconds=params.get("min_tail_seconds"),
                    provenance=plan.get("provenance"))
    elif plan.get("strategy") == "model_downbeat_bars":
        rebuilt = bar_windows(expected_binding, plan.get("downbeats_seconds"), source=plan.get("provenance"),
                    policy=params.get("policy"), min_edge_seconds=params.get("min_edge_seconds"))
    else:
        raise ValueError("unknown musical window strategy")
    for key in rebuilt:
        if plan.get(key) != rebuilt[key]:
            raise ValueError("shared window derivation differs: " + key)
    return plan


def describe_shared_windows(arrays, plan, binding, src, shim, *, captioner=None):
    from ms_chunk_caption import describe_windows
    validate_shared_plan(plan, binding)
    return describe_windows(arrays, plan["windows"], src, shim, captioner=captioner)


def read_verified_report(path):
    from ms_optimize import sha256
    payload = path.read_bytes()
    receipt = json.loads(path.with_suffix(".receipt.json").read_bytes())
    if receipt.get("report_sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("altered source report receipt: " + str(path))
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def initialize(context):
    global CONTEXT
    CONTEXT = context
    from ms_chunk_caption import init_worker
    init_worker(context)


def run_one(row):
    from ms_chunk_caption import media_bounds
    from ms_optimize import atomic_write, json_bytes, sha256
    out, sid, ctx = Path(CONTEXT["out"]), row["id"], CONTEXT
    try:
        source_path = Path(ctx["beat_baseline"]) / (sid + ".json")
        baseline, report_sha = read_verified_report(source_path)
        if baseline.get("schema") != "motionscript_detected_beat_C_compare_v1" or baseline.get("id") != sid:
            raise ValueError("wrong C50 report schema/id")
        for key in ("source_group", "split"):
            if baseline.get(key) != row[key]:
                raise ValueError("source split/group differs")
        for key in ("nlf_path", "video_path", "audio_path"):
            if baseline["manifest_record"][key] != row[key]:
                raise ValueError("manifest path differs from same-source C50")
        source_bytes = Path(row["nlf_path"]).read_bytes()
        actual = {"input_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "video_sha256": sha256(Path(row["video_path"])), "audio_sha256": sha256(Path(row["audio_path"]))}
        for key, value in actual.items():
            digest(baseline.get(key), key)
            if baseline[key] != value:
                raise ValueError("actual source SHA mismatch: " + key)
        with np.load(io.BytesIO(source_bytes), allow_pickle=False) as archive:
            nlf = {key: archive[key] for key in archive.files}
        end, media = media_bounds(row, nlf, ffprobe=ctx["ffprobe"])
        if media != baseline["media"]:
            raise ValueError("actual media/PTS differs from verified C50")
        coords_name = baseline["coords_file"]
        if coords_name != sid + ".coords.npz":
            raise ValueError("source coords must be the bound bare sid file")
        coords_path = source_path.parent / coords_name
        coords_bytes = coords_path.read_bytes()
        if hashlib.sha256(coords_bytes).hexdigest() != baseline["coords_sha256"]:
            raise ValueError("whole-clip coordinates SHA mismatch")
        with np.load(io.BytesIO(coords_bytes), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        pts = np.asarray(nlf["timestamps"], dtype=float) if "timestamps" in nlf else np.arange(len(nlf["pose"])) / float(nlf["fps"])
        binding = {"id": sid, **actual, "actual_video_pts_sha256": hashlib.sha256(json_bytes(pts.tolist())).hexdigest(),
                   **{key: media[key] for key in ("media_start", "media_end", "video_pts_offset", "video_fps", "decoded_video_frames")}}
        if ctx["strategy"] == "bpm":
            audio = baseline["audio"]
            if audio.get("audio_sha256") != actual["audio_sha256"]:
                raise ValueError("BPM estimator audio SHA mismatch")
            rate = number(audio.get("native_sample_rate"), "sample_rate", positive=True)
            if abs(number(audio.get("duration_seconds"), "decoded duration", positive=True)-end) > 1/rate:
                raise ValueError("decoded audio length differs from actual video duration")
            provenance = {"method": "reused_actual_C50_bpm_estimate", "estimator": audio["estimator"],
                          "source_report_path": str(source_path), "source_report_sha256": report_sha,
                          "audio_sha256": actual["audio_sha256"], "calibrated": False}
            plan = bpm_duration_windows(binding, audio["bpm_estimate"], beats_per_chunk=ctx["beats_per_chunk"],
                                        min_tail_seconds=ctx["min_seconds"], provenance=provenance)
        else:
            record = ctx["bar_records"][sid]
            plan = bar_windows(binding, record["downbeats_seconds"], source=record["source"],
                               policy=ctx["bar_policy"], min_edge_seconds=ctx["min_seconds"])
        plan["manifest_record"] = row
        plan["coordinates_reference"] = {"path": str(coords_path), "sha256": baseline["coords_sha256"],
                                          "preprocessing_reused_whole_clip": True, "coordinate_reset_applied": False}
        plan["source_C_report"] = {"path": str(source_path), "sha256": report_sha}
        if ctx["strategy"] == "bar":
            plan["bar_input_reference"] = {"path": ctx["bar_input"], "sha256": ctx["bar_input_sha256"]}
        plan_bytes = json_bytes(plan)
        atomic_write(out / (sid + ".windows.json"), plan_bytes)
        views = describe_shared_windows(arrays, plan, binding, Path(ctx["src"]), Path(ctx["shim"]) if ctx["shim"] else None)
        full = all(view["status"] == "candidate" for view in views)
        report = {"schema": REPORT_SCHEMA, "id": sid, "fingerprint": ctx["fingerprint"],
                  "binding": binding, "source_group": row["source_group"], "split": row["split"],
                  "window_manifest": plan, "window_manifest_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                  "MS_windows": views, "caption_execution": "actual_original_MS_on_receipt_verified_whole_C_coordinates",
                  "MS_full_available_support_coverage": full,
                  "MS_all_windows_caption_text": " \n".join(view["cleaned_motion_text"] for view in views) if full else "",
                  "MS_whole_window_semantics_verified": False,
                  "baseline_comparison_only": {"A_whole_clip": baseline["A_whole_clip"],
                    "B_fixed_windows": baseline["B_fixed_windows"], "C_detected_beat_windows": baseline["C_windows"]},
                  "baseline_caption_substitution_applied": False, "quality": baseline["quality"],
                  "coordinates_reference": plan["coordinates_reference"],
                  "accepted": False, "formal_approval": False, "semantic_approval": False, "human_review_completed": False}
        payload = json_bytes(report)
        atomic_write(out / (sid + ".receipt.json"), json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest(),
                                              "window_manifest_sha256": hashlib.sha256(plan_bytes).hexdigest()}))
        atomic_write(out / (sid + ".json"), payload)
        return report, None
    except Exception as error:
        result = {"id": sid, "error_type": type(error).__name__, "error": str(error)}
        atomic_write(out / (sid + ".error.json"), json_bytes(result))
        return None, result


def stats(reports, key):
    from ms_beat_chunk_compare import word_stats
    views = [view for report in reports for view in (report["MS_windows"] if key == "MS" else
             [report["baseline_comparison_only"][key]] if key == "A_whole_clip" else report["baseline_comparison_only"][key])]
    raw = [" ".join(text for segment in view["original_captions"] for text in segment["ms"] if text.strip()) for view in views]
    return {"source_clip_denominator": len(reports), "window_denominator": len(views),
            "status_counts": dict(Counter(view["status"] for view in views)),
            "support_status_counts": dict(Counter(view["support_status"] for view in views)),
            "abstain_reasons": dict(Counter(view["abstain_reason"] for view in views if view["status"] == "abstain")),
            "raw_original_text_repetition_only_not_accuracy": word_stats(raw)}


def run(args):
    from ms_chunk_caption import load_manifest
    from ms_optimize import atomic_write, json_bytes, overlaps, sha256, source_hashes, upstream_hashes
    if type(args.workers) is not int or args.workers < 1 or args.limit < 0:
        raise ValueError("invalid workers/limit")
    manifest = Path(args.manifest).resolve()
    rows, manifest_sha = load_manifest(manifest)
    rows = rows[:args.limit] if args.limit else rows
    if len(rows) > 50:
        raise ValueError("finite music comparison is limited to same first50; no 12k rerun")
    baseline = Path(args.beat_baseline).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError("output must be a new independent directory")
    for path in (manifest.parent, baseline, Path(args.upstream_src).resolve(), Path(args.upstream_shim).resolve() if args.upstream_shim else None):
        if path is not None and overlaps(out, path):
            raise ValueError("output overlaps source/baseline")
    for row in rows:
        for key in ("nlf_path", "audio_path", "video_path"):
            if overlaps(out, Path(row[key]).parent):
                raise ValueError("output overlaps media/pose")
    source_protocol_path = baseline / "protocol.json"
    source_protocol = json.loads(source_protocol_path.read_bytes())
    if [row["id"] for row in rows] != source_protocol["protocol"]["selected_ids"][:len(rows)]:
        raise ValueError("music selection must match same ordered C50 prefix")
    current_hashes = {**source_hashes(), "ms_chunk_caption.py": sha256(Path(__file__).with_name("ms_chunk_caption.py")),
                      "ms_landmarks.py": sha256(Path(__file__).with_name("ms_landmarks.py"))}
    if any(source_protocol["protocol"]["code_sha256"].get(key) != value for key, value in current_hashes.items()):
        raise ValueError("reused coordinate/original MS code identity differs from C50")
    src, shim = Path(args.upstream_src).resolve(), Path(args.upstream_shim).resolve() if args.upstream_shim else None
    upstream = upstream_hashes(src, shim)
    if upstream != source_protocol["protocol"]["upstream_sha256"]:
        raise ValueError("original MS source identity differs from C50")
    config = {key: getattr(args, key) for key in ("strategy", "bar_policy", "beats_per_chunk", "min_seconds", "workers", "limit", "ffprobe")}
    ctx = {"out": str(out), "beat_baseline": str(baseline), "src": str(src), "shim": str(shim) if shim else None, **config}
    protocol = {"schema": REPORT_SCHEMA, "manifest_sha256": manifest_sha, "selected_ids": [r["id"] for r in rows],
                "source_C_protocol_sha256": sha256(source_protocol_path), "source_C_protocol": source_protocol,
                "code_sha256": {**current_hashes, Path(__file__).name: sha256(Path(__file__)),
                                  "ms_beat_chunk_compare.py": sha256(Path(__file__).with_name("ms_beat_chunk_compare.py"))},
                "upstream_sha256": upstream, "sampler": {"target_fps": 20., "max_gap_seconds": .2, "smooth_seconds": .0625,
                    "half_open_selection": True, "full_coordinates_reused": True, "seed_each_contiguous_interval": 0,
                    "original_minimum_interval_samples": 4, "support_end": "inclusive last usable sample, not plan end"},
                "config": config, "caption_substitution_applied": False, "VLM_shared_manifest_actual_execution": False}
    if args.strategy == "bar":
        if not args.bar_input:
            raise ValueError("bar strategy requires actual BeatThis --bar-input")
        bar_input = Path(args.bar_input).resolve()
        payload = bar_input.read_bytes()
        bar_rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        by_id = {r["id"]: r for r in bar_rows}
        if len(by_id) != len(bar_rows) or any(r["id"] not in by_id for r in rows):
            raise ValueError("bar input duplicate/missing selected source")
        ctx.update(bar_records=by_id, bar_input=str(bar_input), bar_input_sha256=hashlib.sha256(payload).hexdigest())
        protocol["bar_input"] = {"path": str(bar_input), "sha256": ctx["bar_input_sha256"]}
    ctx["fingerprint"] = hashlib.sha256(json_bytes(protocol)).hexdigest()
    initialize(ctx)
    out.mkdir()
    atomic_write(out / "protocol.json", json_bytes({"fingerprint": ctx["fingerprint"], "protocol": protocol}))
    reports, errors, pool = [], [], None
    if args.workers == 1:
        iterator = map(run_one, rows)
    else:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=initialize, initargs=(ctx,))
        iterator = pool.imap_unordered(run_one, rows)
    try:
        for report, error in iterator:
            if error:
                errors.append(error)
                print(json.dumps(error), flush=True)
            else:
                reports.append(report)
                print(json.dumps({"id": report["id"], "MS_windows": len(report["MS_windows"]),
                                  "full_available_support": report["MS_full_available_support_coverage"]}), flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    order = {r["id"]: i for i, r in enumerate(rows)}
    reports.sort(key=lambda r: order[r["id"]]); errors.sort(key=lambda r: order[r["id"]])
    windows_bytes = b"".join((json.dumps(r["window_manifest"], ensure_ascii=False, allow_nan=False) + "\n").encode() for r in reports)
    chunks_bytes = b"".join((json.dumps({"schema": REPORT_SCHEMA, "id": r["id"], "binding": r["binding"],
                        "window_manifest_sha256": r["window_manifest_sha256"], **v}, ensure_ascii=False, allow_nan=False)+"\n").encode()
                        for r in reports for v in r["MS_windows"])
    summary = {"schema": REPORT_SCHEMA, "fingerprint": ctx["fingerprint"], "selected_count": len(rows),
               "processed_count": len(reports), "failed_count": len(errors), "errors": errors, "complete": not errors,
               "strategy": args.strategy, "plan_status_counts": dict(Counter(r["window_manifest"]["status"] for r in reports)),
               "full_available_support_clip_count": sum(r["MS_full_available_support_coverage"] for r in reports),
               "actual_MS_caption_execution": True, "new_MS_caption_source": "MS_windows only; no A/B/C substitution",
               "VLM_shared_manifest_actual_execution": False, "A_B_C_newMS_same_processed_source_stats":
                 {key: stats(reports, key) for key in ("A_whole_clip", "B_fixed_windows", "C_detected_beat_windows", "MS")},
               "window_manifest_sha256": hashlib.sha256(windows_bytes).hexdigest(),
               "chunks_sha256": hashlib.sha256(chunks_bytes).hexdigest(),
               "accepted": False, "formal_approval": False, "semantic_approval": False, "human_review_completed": False,
               "accuracy_evaluation": "not evaluated: repetition/coverage are not semantic accuracy",
               "records": [{"id": r["id"], "report_file": r["id"]+".json",
                            "report_sha256": sha256(out / (r["id"]+".json")),
                            "window_file": r["id"]+".windows.json", "window_sha256": r["window_manifest_sha256"],
                            "coords_sha256": r["coordinates_reference"]["sha256"]} for r in reports]}
    atomic_write(out / "windows.jsonl", windows_bytes)
    atomic_write(out / "chunks.jsonl", chunks_bytes)
    payload = json_bytes(summary)
    atomic_write(out / "summary.receipt.json", json_bytes({"summary_sha256": hashlib.sha256(payload).hexdigest(),
                       "window_manifest_sha256": hashlib.sha256(windows_bytes).hexdigest(), "chunks_sha256": hashlib.sha256(chunks_bytes).hexdigest()}))
    atomic_write(out / "summary.json", payload)
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True); ap.add_argument("--beat-baseline", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--upstream-src", required=True); ap.add_argument("--upstream-shim")
    ap.add_argument("--strategy", choices=("bpm", "bar"), default="bpm"); ap.add_argument("--bar-input")
    ap.add_argument("--bar-policy", choices=("strict", "merge_short_edges"), default="strict")
    ap.add_argument("--beats-per-chunk", type=int, default=4); ap.add_argument("--min-seconds", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ffprobe", default="/opt/ms_tools/ffprobe")
    return ap


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    sys.exit(0 if result["complete"] else 1)
