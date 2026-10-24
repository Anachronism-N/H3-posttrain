"""Read-only exact video-PTS and typed-plan audit of frozen music-window runs."""
import argparse
import hashlib
import json
from pathlib import Path

import ms_music_windows as music
from ms_chunk_caption import load_manifest
from ms_landmarks import probe_video
from ms_optimize import json_bytes, sha256
from ms_visual_observe import strict_json

SCHEMA = "music_window_exact_pts_typed_verification_v1"
PRODUCER_SHA = "1b1a1309c92f25bd9faf86cc663ab61686e78f668e20398c817ce15a305bed82"


def validate_plan(plan, binding):
    music.binding_check(binding)
    if json_bytes(plan.get("binding")) != json_bytes(binding):
        raise ValueError("typed binding or exact video PTS differs")
    music.binding_check(plan["binding"])
    params = plan.get("params", {})
    if plan.get("strategy") == "bpm_duration_grid_not_detected_beats":
        expected = music.bpm_duration_windows(binding, params.get("bpm_estimate"),
            beats_per_chunk=params.get("beats_per_chunk"),
            min_tail_seconds=params.get("min_tail_seconds"), provenance=plan.get("provenance"))
    elif plan.get("strategy") == "model_downbeat_bars":
        expected = music.bar_windows(binding, plan.get("downbeats_seconds"),
            source=plan.get("provenance"), policy=params.get("policy"),
            min_edge_seconds=params.get("min_edge_seconds"))
    else:
        raise ValueError("unsupported music-window strategy")
    for key, value in expected.items():
        if json_bytes(plan.get(key)) != json_bytes(value):
            raise ValueError("typed musical plan differs: " + key)
    return plan


def verify_actual_pts(plan, info):
    if info["media_end"] is None:
        raise ValueError("actual video duration missing")
    binding = dict(plan["binding"])
    binding.update(actual_video_pts_sha256=hashlib.sha256(json_bytes(info["time"].tolist())).hexdigest(),
        media_start=0., media_end=float(info["media_end"]), video_pts_offset=info["pts_offset"],
        video_fps=info["fps"], decoded_video_frames=len(info["time"]))
    validate_plan(plan, binding)
    return binding


def read_report(root, record):
    sid = record["id"]
    if record["report_file"] != sid + ".json" or record["window_file"] != sid + ".windows.json":
        raise ValueError("report/window path outside exact SID")
    report_path, plan_path = root / record["report_file"], root / record["window_file"]
    report_bytes, plan_bytes = report_path.read_bytes(), plan_path.read_bytes()
    report_sha, plan_sha = hashlib.sha256(report_bytes).hexdigest(), hashlib.sha256(plan_bytes).hexdigest()
    receipt = strict_json((root / (sid + ".receipt.json")).read_bytes())
    if report_sha != record["report_sha256"] or receipt["report_sha256"] != report_sha:
        raise ValueError("report source receipt changed")
    if plan_sha != record["window_sha256"] or receipt["window_manifest_sha256"] != plan_sha:
        raise ValueError("window source receipt changed")
    report, plan = strict_json(report_bytes), strict_json(plan_bytes)
    if json_bytes(report["window_manifest"]) != json_bytes(plan) or report["id"] != sid:
        raise ValueError("report differs from actual shared plan")
    if json_bytes(report["binding"]) != json_bytes(plan["binding"]):
        raise ValueError("report binding differs from plan")
    return report, plan, {"path": str(report_path), "sha256": report_sha}, {"path": str(plan_path), "sha256": plan_sha}


def run(args):
    out = Path(args.out).absolute()
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError("new exclusive output with existing parent required")
    roots = [Path(p).resolve() for p in args.bundles]
    manifest = Path(args.manifest).resolve()
    if len(roots) != 3 or len(set(roots)) != 3:
        raise ValueError("exactly BPM, strict-bar and merged-bar bundles required")
    if out == manifest or any(out.is_relative_to(p) or p.is_relative_to(out) for p in roots):
        raise ValueError("output overlaps source bundle")
    source = Path(music.__file__).resolve()
    if sha256(source) != PRODUCER_SHA:
        raise ValueError("frozen music-window producer changed")
    own_sha = sha256(Path(__file__))
    rows, manifest_sha = load_manifest(manifest)
    if len(rows) != 50:
        raise ValueError("exact same50 manifest required")
    probes, observed, counts = {}, [], {}
    strategies = set()
    for root in roots:
        summary_path = root / "summary.json"
        summary_bytes = summary_path.read_bytes()
        summary = strict_json(summary_bytes)
        if summary["complete"] is not True or type(summary["processed_count"]) is not int or summary["processed_count"] != 50 or summary["failed_count"] != 0:
            raise ValueError("bundle is not actual complete same50")
        records = summary["records"]
        if [r["id"] for r in records] != [r["id"] for r in rows]:
            raise ValueError("bundle order differs from exact manifest")
        window_count = 0
        for row, record in zip(rows, records):
            report, plan, report_ref, plan_ref = read_report(root, record)
            if json_bytes(plan["manifest_record"]) != json_bytes(row):
                raise ValueError("plan input record changed")
            sid = row["id"]
            if sid not in probes:
                video = Path(row["video_path"])
                before = video.stat()
                info = probe_video(video, args.ffprobe)
                after = video.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("video changed during actual PTS probe")
                probes[sid] = info
            binding = verify_actual_pts(plan, probes[sid])
            actual_bounds = [{"start": v["window_start"], "end": v["window_end"]} for v in report["MS_windows"]]
            planned_bounds = [{"start": v["start"], "end": v["end"]} for v in plan["windows"]]
            if json_bytes(actual_bounds) != json_bytes(planned_bounds):
                raise ValueError("actual MS window differs from shared plan")
            window_count += len(plan["windows"])
            policy = plan["params"].get("policy", "bpm")
            strategies.add((plan["strategy"], policy))
            observed.append({"id": sid, "policy": policy, "actual_video_pts_sha256": binding["actual_video_pts_sha256"],
                "frame_count": binding["decoded_video_frames"], "report": report_ref, "plan": plan_ref,
                "exact_pts_digest_equal_original": True, "canonical_typed_plan_verified": True})
        if sha256(summary_path) != hashlib.sha256(summary_bytes).hexdigest():
            raise ValueError("bundle summary changed during verification")
        counts[str(root)] = {"summary_sha256": hashlib.sha256(summary_bytes).hexdigest(), "clips": 50, "windows": window_count}
    if strategies != {("bpm_duration_grid_not_detected_beats", "bpm"), ("model_downbeat_bars", "strict"), ("model_downbeat_bars", "merge_short_edges")}:
        raise ValueError("required musical strategies differ")
    if sha256(source) != PRODUCER_SHA or sha256(Path(__file__)) != own_sha or sha256(manifest) != manifest_sha:
        raise ValueError("verification source or input manifest changed")
    result = {"schema": SCHEMA, "complete": True, "input_manifest_path": str(manifest),
        "input_manifest_sha256": manifest_sha, "video_probes": len(probes), "reports_verified": len(observed),
        "bundle_counts": counts, "records": observed, "source_path": str(source), "source_sha256": PRODUCER_SHA,
        "verifier_path": str(Path(__file__).resolve()), "verifier_sha256": own_sha,
        "actual_video_pts_probed_not_inferred_from_nlf": True, "old_reports_and_captions_modified": False,
        "MS_or_model_inference_repeated": False, "source_video_bytes_rehashed": False,
        "audio_video_alignment_verified": False, "motion_accuracy_verified": False, "GPU_used": False}
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bundles", nargs=3, required=True)
    parser.add_argument("--ffprobe", default="/opt/ms_tools/ffprobe")
    parser.add_argument("--out", required=True)
    result = run(parser.parse_args())
    print(json.dumps({"complete": result["complete"], "reports_verified": result["reports_verified"], "video_probes": result["video_probes"]}))
