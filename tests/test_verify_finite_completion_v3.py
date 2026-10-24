"""Synthetic CPU fixtures; never evidence that the real 12000-source goal finished."""

import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).parents[1] / "scripts_omnidance/exp/ms/verify_finite_completion_v3.py"
spec = importlib.util.spec_from_file_location("finite_v3", SOURCE)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(content)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def mswin(mode, start, end):
    return {"source": "original_motionscript_rules_on_v2_coordinates", "mode": mode,
            "window_start": start, "window_end": end, "status": "candidate",
            "abstain_reason": None, "support_status": "complete_available_samples",
            "candidate_text_eligible": True, "raw_original_description_nonempty": True,
            "motion_caption": f"[{start},{end}) original independent motion",
            "formal_training_eligible": False, "formal_approval": False}


def visualwin(start):
    raw = "a" * 64
    frames = [{"index": start * 10 + i, "time": float(start) + i / 10, "source_timestamp": float(start) + i / 10, "sha256": "b" * 64} for i in range(2)]
    return {"status": "candidate", "error": None, "window": {"start": start, "end": start + 2},
            "receipt_sha256": "c" * 64, "files_sha256": {"response.raw.txt": raw, "generation.json": "d" * 64, "candidate.json": "e" * 64, "manifest.json": "f" * 64},
            "caption": "original movement", "events": [{"id": "e1", "description": "supported movement", "frame_indices": [r["index"] for r in frames], "evidence_time_range": [r["time"] for r in frames], "time_range_is_exact_event_boundary": False}],
            "generation": {"model_call_performed": True, "max_tokens": 1200, "prompt_tokens": 100, "completion_tokens": 30, "finish_reason": "stop", "raw_response_sha256": raw},
            "source_frames": frames, "flags_are_diagnostic_not_semantic_approval": True}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "COUNT", 8)  # Explicit synthetic fixture only; CLI has no count override.
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ())  # Frozen historical core regressions, not proof of newest requested trials.
    monkeypatch.setattr(f, "SHARED_VLM_REQUIRED", ())  # Historical request snapshot; current-scope tests restore this requirement.
    monkeypatch.setattr(f, "EXACT_MUSIC_PTS_REQUIRED", False)  # Historical MS-only proof; not actual-video-PTS evidence.
    # Dedicated latest-scope tests below restore all three real mandatory names.
    rows = [{"id": f"synthetic-{i}", "source_group": "fixture", "split": "train", "input_row_sha256": "a" * 64} for i in range(8)]
    input_path = tmp_path / "input/manifest.jsonl"
    input_sha = jsonl(input_path, rows)
    monkeypatch.setattr(f, "INPUT_SHA", input_sha)
    contract = {"schema": f.CONTRACT_SCHEMA, "feature_enabled": True, "required_checks": list(f.CHECKS),
                "human_quality": "negative", "input": {"path": str(input_path), "sha256": input_sha}}
    root = tmp_path / "ms"
    msrows = []
    for row in rows:
        windows = [mswin("B_fixed_window", i, i + 2) for i in (0, 2)]
        msrows.append({**row, "schema": "motionscript_verified_A_B_export_v1", "status": "processed",
                       "A_whole_clip": mswin("A_original_rules_whole_clip_same_coordinates", 0, 4), "B_windows": windows,
                       "B_complete_clip_motion_caption": "\n".join(w["motion_caption"] for w in windows),
                       "formal_approval": False, "accepted": False, "semantic_approval": False, "human_review_completed": False})
    files = {"captions.jsonl": jsonl(root / "captions.jsonl", msrows), "lineage.jsonl": jsonl(root / "lineage.jsonl", rows), "errors.jsonl": jsonl(root / "errors.jsonl", [])}
    coverage = {"target_clip_denominator": 8, "observed_B_window_denominator": 16, "B_all_planned_windows_candidate_clip_count": 8}
    complete = {"schema": "motionscript_verified_A_B_export_v1", "complete": True, "status": "completed", "all_shards_terminal": True, "all_expected_ids_accounted": True, "exported": True,
                "formal_approval": False, "formal_training_eligible": False, "accepted": False, "semantic_approval": False, "human_review_completed": False, "unique_id_count": 8, "successfully_processed_count": 8, "failed_source_count": 0, "file_sha256": files, "caption_coverage": coverage, "B_window_status_counts": {"candidate": 16},
                "binding": {"input_manifest_sha256": input_sha, "exporter_source_sha256": f.MS_SOURCE, "producer_source_sha256": f.MS_PRODUCER}}
    cref = dump(root / "complete.json", complete)
    verify = {"schema": "motionscript_existing_export_verification_v2", "complete": True, "status": "verified", "exact_expected_bytes_verified": True, "existing_export_modified": False,
              "first_export_source_sha256": f.MS_SOURCE, "verifier_source_sha256": f.MS_VERIFIER, "producer_source_sha256": f.MS_PRODUCER, "input_manifest_sha256": input_sha,
              "formal_approval": False, "formal_training_eligible": False, "accepted": False, "semantic_approval": False, "human_review_completed": False, "existing_complete_sha256": cref["sha256"], "existing_complete_path": cref["path"], "canonical_file_sha256": files, "unique_id_count": 8, "failed_source_count": 0, "caption_coverage": coverage}
    contract["ms12k"] = {"complete": cref, "complete_receipt": dump(root / "complete.receipt.json", {"complete_sha256": cref["sha256"]}), "verification": dump(tmp_path / "ms_verify.json", verify)}
    jobs, lineage, exits = [], [], []
    now = datetime.now(timezone.utc).timestamp()
    identity = {"root": "/explicit/model", "sha256": "b" * 64, "files": {f"file-{i}": {"path": f"/explicit/model/file-{i}", "sha256": "a" * 64, "stat": {"size": i + 1, "inode": i}} for i in range(25)}}
    identity["files"] = dict(sorted(identity["files"].items()))
    identity["sha256"] = f.model_identity_digest(identity)
    modelref = dump(tmp_path / "model_reference.json", {"model_identity": identity})
    inventory = {name: {"sha256": v["sha256"], "bytes": v["stat"]["size"]} for name, v in sorted(identity["files"].items())}
    inventory_sha = hashlib.sha256((json.dumps(inventory, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
    for i in range(8):
        queue = tmp_path / f"queue{i}"
        terminal = {"expected_ids": 1, "exit_code": 0, "source_unchanged": True, "input_unchanged": True, "finished_unix": now - 5}
        exitref = dump(queue / "worker_exit.json", terminal)
        exits.append(exitref)
        admission = dump(queue / "admission.json", {"actual": "fixture-only"})
        args = ["python", "worker.py", "--shard", str(i), "--nshard", "8", "--limit", "0", "--tp", "2", "--batch-size", "2", "--max-num-seqs", "2", "--prepare-workers", "2", "--model", "/explicit/model"]
        job = {"node": i + 1, "gpus": [0, 1], "shard": i, "host": f"fixture-host{i}", "expected_ids": 1, "out": str(tmp_path / f"worker{i}"), "resource_queue": str(queue), "argv": args}
        jobs.append(job)
        model = copy.deepcopy(identity)
        model["files"]["file-0"]["stat"]["inode"] += i  # Different host stat is legitimate.
        lineage.append({**{k: job[k] for k in ("shard", "host", "out", "resource_queue")}, "model_identity": model, "model_content_inventory_sha256": inventory_sha,
                        "resource_worker_exit": terminal, "worker_exit_sha256": exitref["sha256"], "admission_path": admission["path"], "admission_sha256": admission["sha256"]})
    plan = {"schema": "choreography_video_first_12000_32b_launch_v1", "target_ids": 8, "shards": 8, "producer_sha256": f.VLM_PRODUCER, "input_manifest_sha256": input_sha,
            "formal_approval": False, "human_review_completed": False, "semantic_approval": False, "expected_model_identity_sha256": identity["sha256"], "jobs": jobs}
    planref = dump(tmp_path / "plan.json", plan)
    monkeypatch.setattr(f, "VLM_PLAN", planref["sha256"])
    vrows = [{**row, "original_input": row, "original_input_row_sha256": row["input_row_sha256"], "shard": i, "host": jobs[i]["host"], "model_path": "/explicit/model", "status": "candidate", "windows": [visualwin(0), visualwin(2)], "source": "independent_video_open_vocabulary_unreviewed",
              "semantic_approval": False, "human_review_completed": False, "formal_training": False, "ground_truth": False} for i, row in enumerate(rows)]
    vsha = jsonl(tmp_path / "vlm/captions.jsonl", vrows)
    exporter = dump(tmp_path / "writer.py", {"fixture_only": True})
    monkeypatch.setattr(f, "VLM_EXPORTER", exporter["sha256"])
    manifest = {"schema": "motionscript_visual_terminal_export_v1", "status": "terminal_verified_unreviewed", "expected_ids": 8, "verified_ids": 8, "expected_window_count": 16,
                "semantic_approval": False, "human_review_completed": False, "formal_training": False, "ground_truth": False, "original_raw_or_caption_facts_modified": False,
                "all_planned_workers_terminal": True, "all_expected_ids_accounted_for": True, "flags_are_diagnostic_not_semantic_verdicts": True, "formal_training_not_updated": True, "first50_not_silently_reused": True,
                "launch_plan_sha256": planref["sha256"], "input_manifest_sha256": input_sha, "producer_sha256": f.VLM_PRODUCER, "exporter_sha256": exporter["sha256"],
                "lineage": lineage, "model_reference_receipt_path": modelref["path"], "model_reference_receipt_sha256": modelref["sha256"], "model_content_inventory_sha256": inventory_sha, "captions_sha256": vsha,
                "sid_status_counts": {"candidate": 8, "abstain": 0, "failed": 0}, "window_status_counts": {"candidate": 16, "abstain": 0, "failed": 0},
                "usable_full_caption_count": 8, "incomplete_caption_count": 0, "empty_caption_count": 0, "clip_denominator": 8, "window_denominator": 16, "actual_candidate_window_count": 16, "actual_abstain_window_count": 0, "actual_failed_window_count": 0, "caption_delivery_complete": True, "max_additional_failed_attempts": 2, "partial_nonempty_caption_count": 0, "any_nonempty_caption_count": 8, "sources_without_nonempty_caption": 0,
                "sources_without_terminal_window_receipts": 0, "source_error_missing_window_count": 0, "actual_terminal_window_receipt_count": 16, "goal_full_caption_coverage_satisfied": True}
    contract["vlm12k"] = {"plan": planref, "manifest": dump(tmp_path / "vlm/manifest.json", manifest), "exporter": exporter}
    anchor = {"complete": True, "successful_updates": 100, "rng_rank_count": 8, "optimizer_shards": [f"optimizer-{i}.bin" for i in range(8)], "model_shards": [f"pytorch_model/shard-{i}.bin" for i in range(8)]}
    anchor.update(actual_training_updates=100, world_size=8, lora_rank=32, lora_alpha=64, complete_state_members=28, anchor_lambda=0.1, training_cfg_scale=1.0, full_checkpoint_reverified=True, independent_audit=True, fm_recipe_verified=True, typed_numeric_verified=True, all_file_sha256_verified=True, model_and_adam_and_rng_and_scheduler_finite=True, lora_all_ab_source_verified=True, human_quality_verified=False, lora_sha256="a" * 64)
    aref = dump(tmp_path / "eval/anchor.json", anchor)
    monkeypatch.setattr(f, "ANCHOR_SHA", aref["sha256"])
    native = {"complete": True, "steps": 2250, "exposures": 18000, "state_tensors": 467, "optimizer_states": 402, "actual_three_unique_epochs": True, "full_saved_Adam_RNG_sampler_pointer": True, "strict_loaded_complete_MusicFM": True, "training_finished": {"complete": True, "actual_successful_updates": 2250, "actual_exposures": 18000, "validation_epochs": 3, "final_format4_state_tensors": 467}}
    original_native = dump(tmp_path / "eval/original_native.json", {"complete": True, "checkpoint_sha256": "a" * 64})
    native.update(trainable_tensors=402, checkpoint_sha256="a" * 64, local_stage_actual_sha256="a" * 64, fresh_final_sha256="a" * 64, fresh_local_stage_sha256="a" * 64, original_receipt=original_native["path"], original_receipt_sha256=original_native["sha256"])
    manifest_bytes = b'{"fixture":true}'
    independent = dump(tmp_path / "eval/independent.json", {"complete": True})
    failure = dump(tmp_path / "eval/failure.json", {"complete": False, "error": "BrokenPipeError"})
    evalchecks = []
    for role in sorted(f.ROLES):
        for sid in sorted(f.IDS):
            media = tmp_path / "media" / (str(len(evalchecks)) + ".mp4")
            media.parent.mkdir(exist_ok=True)
            media.write_bytes(b"x" * 1024)
            if role == "MS/music_direct_full6k_final2250_video_n4_v3":
                audit = {"complete": True, "input": {"case": {"sample_id": sid, "condition": "joint"}, "student_reads_target_or_caption": False, "teacher_identity": False, "student_zero_input": False, "semantic_text": None, "output_audio": "original_mux", "native_waveform_sha256": "a" * 64}, "audit": {"scheduler_updates": {"video": 50, "audio": 50}, "native_audio_condition": "real_reference_audio", "music_mode": "ref", "original_joint_pipeline": True, "full_cached_prefix_preserved": True, "original_cached_prefix_tags_preserved": True, "packed_text_tags_exact": True, "text_encoder_calls": 0, "text_encoder_loaded": False, "tokenizer_loaded": False, "processor_loaded": False, "waveform_sha256": "a" * 64}, "mux_audio_sha256": "a" * 64}
                audit["video_sha256"] = "d" * 64
                auditref = dump(media.with_suffix(".audit.json"), audit)
                validation = {"full_media_decode_checked": True, "full_video_audio_decode": True, "ffmpeg_decoded_video_frames": 124, "sha256": "d" * 64, "streams": [{"codec_type": "video", "width": 704, "height": 1280, "decoded_frames": 124, "fps": 24}, {"codec_type": "audio"}]}
                validation_ref = dump(media.parent.parent / (sid + ".validated.json"), validation)
            else:
                audit = {"complete": True, "id": sid, "media": {"complete_decode": True, "frames": 124, "height": 1280, "width": 704, "video_streams": 1, "audio_streams": 0, "fps": 24}, "sampling": {"video_scheduler_updates": 50, "target_audio_generated": False}, "audio_guard": {"calls": {"audio_vae": 0}, "audio_vae_is_none": True, "scheduler_audio_is_none": True}}
                auditref = dump(media.parent / (sid + ".audit.json"), audit)
            evalchecks.append({"model_id": role, "sample_id": sid, "status": 206, "bytes_verified": 1024, "range_bytes_match_disk": True, "producer_full_decode_gate": True,
                               "video_sha256": "d" * 64, "provenance": {"video_sha256": "d" * 64, "audit_sha256": auditref["sha256"], **({"validation_sha256": validation_ref["sha256"]} if role == "MS/music_direct_full6k_final2250_video_n4_v3" else {})}, "media_url": "http://fixture/media", "actual_video": str(media), "content_range": "bytes 0-1023/1024"})
    report = {"protocol": "focused_eval15_publication_independent_v6", "complete": True, "complete_case_count": 15, "complete_group_count": 5, "manifest_http_status": 200,
              "all15_fresh_source_gate": True, "all15_http_range_bytes_verified": True, "no_v1_alias": True, "overall_goal_completed": False, "numeric_scores_written": False, "checks": evalchecks,
              "independent_receipts": [independent], "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "preserved_original_manager_publication_failure": {**failure, "record": {"complete": False, "error": "BrokenPipeError"}, "actual_new_v3_HTTP_independently_verified": True}}
    report["source_identities"] = {role: {"core_source": {"sha256": "a" * 64}, "wrapper_source": {"sha256": "b" * 64}, "lora_alpha": 1, "runtime": {"full_sequence_retained": True, "weights_unchanged": True, "original_inference_sha256": "a" * 64}} for role in f.ROLES}
    eref = dump(tmp_path / "eval/report.json", report)
    monkeypatch.setattr(f, "EVAL_SHA", eref["sha256"])
    nref = dump(tmp_path / "eval/native.json", native)
    monkeypatch.setattr(f, "NATIVE_SHA", nref["sha256"])
    contract["eval15"] = {"report": eref, "anchor_state": aref, "native_state": nref, "manifest_url": "http://fixture/manifest"}
    monkeypatch.setattr(f, "http_bytes", lambda url, status, limit, headers=None: (manifest_bytes, {}) if status == 200 else (b"x" * 1024, {"Content-Range": "bytes 0-1023/1024"}))  # Mock network, not any checker.
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git/HEAD").write_text("fixture")
    package = [dump(repo / p, {"fixture": True}) for p in sorted(f.PACKAGE_MINIMUM | {"notes/required.md"})]
    for item in package:
        item["path"] = str(Path(item["path"]).relative_to(repo))
    head = "a" * 40
    contract["gitpackage"] = {"repo": str(repo), "branch": "codex/motionscript-optimization", "remote_branch": "codex/motionscript-optimization", "head": head, "remote": "fixture", "files": package, "required_files": [item["path"] for item in package]}
    def fixture_git(root, *args):
        if args[0] == "show":
            return (repo / args[1].split(":", 1)[1]).read_bytes()
        if args[0] == "ls-files":
            return args[-1].encode()
        if args[0] == "branch":
            return b"codex/motionscript-optimization\n"
        if args[0] == "rev-parse":
            return head.encode()
        if args[0] == "ls-remote":
            return (head + "\trefs/heads/codex/motionscript-optimization\n").encode()
        raise AssertionError("unexpected git subprocess")
    monkeypatch.setattr(f, "git", fixture_git)  # Explicit synthetic external Git boundary.
    nodes, baseline, restore_refs = [], [], []
    for n in range(1, 9):
        apps = [{"uuid": f"GPU-{n}-{i}", "pid": n * 100 + i + 1, "ppid": n * 10000 if i < 2 else 1, "uid": 0, "start_ticks": n * 100000 + i, "argv": ["/usr/bin/python", "/project/hold_gpu.py"], "cgroup": "fixture", "CVD": ["CUDA_VISIBLE_DEVICES=" + str(i)], "holder": True} for i in range(8)]
        baseline.append({"node": n, "receipt": {"processes": copy.deepcopy(apps)}})
        for gpu in (0, 1):
            restore_refs.append(dump(tmp_path / f"queue{n-1}" / f"holder_restored_gpu{gpu}.json", {"holder_identity": apps[gpu], "manager_identity": {"pid": n * 10000}, "gpu": gpu, "uuid": apps[gpu]["uuid"], "managed_parent_wait": True}))
        nodes.append({"node": n, "host": "fixture-host" + str(n), "data": {"utc": datetime.now(timezone.utc).isoformat(), "gpus": "\n".join(f"{i}, GPU-{n}-{i}, 975, 100" for i in range(8)), "apps": apps, "reserves": []}})
    baseline_ref = dump(tmp_path / "baseline.json", baseline)
    for node in nodes:
        node["data"]["utc"] = datetime.now(timezone.utc).isoformat()
    contract["resources"] = {"snapshot": dump(tmp_path / "resources.json", nodes), "worker_exits": exits, "holder_restores": restore_refs, "baseline": baseline_ref, "known_reserves": []}
    tasks = {name: {"required": True, "status": "verified", "actual_executed": True, "evidence": independent} for name in ("ms12k", "video_first12k", "eval15", "gitpackage", "resources", "rgb32points", "beat50")}
    tasks["old6kfusion"] = {"required": True, "status": "superseded", "superseded_by": "ms12k"}
    tasks["not_requested"] = {"required": False, "status": "optional"}
    point_records, point_reviews, point_descriptions = [], [], []
    for sid in sorted(f.IDS):
        root = tmp_path / "points" / sid
        source_sha = {"video": "a" * 64, "spec": f.RGB_SPEC, "model": f.RGB_MODEL}
        manifest_ref = dump(root / "raw/manifest.json", {"source": "nlf_direct_rgb_landmarks", "point_count": 32, "source_sha256": source_sha})
        record = {"id": sid, "complete": True, "finite_valid": True, "accuracy_approved": False, "smpl_fitting_used": False, "template_fk_used": False, "shapes": {"points_m": [81, 32, 3], "points2d_px": [81, 32, 2], "valid": [81, 32]}, "valid_points": 2592, "raw_sha256": "a" * 64, "manifest_sha256": manifest_ref["sha256"], "source_sha256": source_sha}
        point_records.append(record)
        overlay = root / "review/rgb_direct32_vs_legacy_pose.mp4"
        overlay.parent.mkdir(parents=True)
        overlay.write_bytes(b"synthetic overlay fixture, not real media")
        review = {"schema": "nlf-direct32-same-window-review-v1", "actual_direct32_loaded": True, "source_frames": 81, "source_fps": 16.0, "bundle_manifest_sha256": manifest_ref["sha256"], "inputs": {"raw": {"sha256": "a" * 64}, "bundle_manifest": manifest_ref}, "legacy_full_pose72_preserved": True, "legacy_global_rotation24_saved": True, "smpl_fk_used_for_new_points": False, "projection_residual_is_accuracy_metric": False, "source_frames_filled_or_interpolated": False, "semantic_approval": False, "accuracy_review_completed": False, "gpu_used": False, "media_bounds": {"end": 5.0625}, "artifacts": {"rgb_direct32_vs_legacy_pose.mp4": {"path": str(overlay), "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest()}}}
        point_reviews.append({"id": sid, "receipt": dump(root / "review/review_receipt.json", review)})
        reports = {"id": sid}
        for mode in ("whole", "fixed2s"):
            report = {"schema": "motionscript_direct_landmarks_v1", "source": "nlf_direct_rgb_landmarks", "point_count": 32, "source_manifest_sha256": manifest_ref["sha256"], "raw_sha256": "a" * 64, "semantic_approval": False, "new_information_accuracy_validated": False, "coords_sha256": "d" * 64}
            if mode == "fixed2s":
                windows = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 5.0625}]
                report.update(window_spec={"strategy": "fixed", "windows": windows}, chunks=[{"window": window, "semantic_approval": False, "caption_is_training_target": False} for window in windows])
            report_ref = dump(root / "describe" / mode / "report.json", report)
            reports[mode] = dump(root / "describe" / mode / "report.receipt.json", {"report_sha256": report_ref["sha256"]})
        point_descriptions.append(reports)
    point_terminal = {"complete": True, "expected": 3, "records": point_records, "source_unchanged": True, "spec_unchanged": True, "full_VLM_untouched": True, "accuracy_approved": False}
    extractor = dump(tmp_path / "points/extractor.py", {"synthetic_extractor": True})
    monkeypatch.setattr(f, "RGB_SOURCE", extractor["sha256"])
    tasks["rgb32points"].update(extractor=extractor, evidence=dump(tmp_path / "points/finished.json", point_terminal), reviews=point_reviews, descriptions=point_descriptions)
    c_summary = dump(tmp_path / "beat/summary.json", {"complete": True})
    c_chunks = jsonl(tmp_path / "beat/chunks.jsonl", [{"fixture": True}])
    c_receipt = {"schema": "ms_beat_C_matched50_independent_verification_v1", "complete": True, "processed_source_count": 50, "C_window_count": 105, "C_status_counts": {"candidate": 95, "abstain": 10}, "C_windows_per_clip_distribution": {"2": 45, "3": 5}, "strategy_counts": {"detected_beats": 50}, "formal_approval": False, "semantic_approval": False, "motions_on_beats_verified": False, "all_native_audio_length_and_source_bytes_matched": True, "all_C_coords_identical_to_A_B_coords_sha": True, "all_detected_internal_boundaries_match_actual_estimator_beats": True, "all_half_open_original_MS_support_verified": True, "summary_sha256": c_summary["sha256"], "chunks_sha256": c_chunks}
    tasks["beat50"].update(evidence=dump(tmp_path / "beat/independent_verification.json", c_receipt))
    contract["scope"] = {"tasks": tasks}
    return contract


def reseal(ref, mutate):
    value = json.loads(Path(ref["path"]).read_bytes())
    mutate(value)
    ref.update(dump(Path(ref["path"]), value))


def edit_caption(contract, kind, mutate):
    section = contract[kind]
    ref = section["complete" if kind == "ms12k" else "manifest"]
    root = Path(ref["path"]).parent
    rows = [json.loads(line) for line in (root / "captions.jsonl").read_text().splitlines()]
    mutate(rows)
    checksum = jsonl(root / "captions.jsonl", rows)
    if kind == "vlm12k":
        reseal(ref, lambda v: v.update(captions_sha256=checksum))
    else:
        reseal(ref, lambda v: v["file_sha256"].update({"captions.jsonl": checksum}))
        reseal(section["complete_receipt"], lambda v: v.update(complete_sha256=ref["sha256"]))
        reseal(section["verification"], lambda v: (v.update(existing_complete_sha256=ref["sha256"]), v["canonical_file_sha256"].update({"captions.jsonl": checksum})))


def test_all_six_real_check_functions_execute_on_fixture(bundle):
    result = f.verify(bundle)
    assert result["status"] == "verified", result
    assert len(result["checks"]) == 6
    assert all(c["executed"] for c in result["checks"].values())
    assert result["terminal_ledger_complete"] and result["caption_delivery_complete"]
    assert result["human_quality"] == "negative" and not result["human_quality_approved"]
    optional = result["checks"]["scope"]["details"]["tasks"]["not_requested"]
    assert optional == {"status": "optional", "executed": False}


@pytest.mark.parametrize("value", [True, False, 1.0, "1", -1])
def test_typed_integer(value):
    with pytest.raises(ValueError):
        f.integer(value)


def test_typed_json_and_duplicates():
    assert not f.same({"x": False}, {"x": 0})
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        with pytest.raises(ValueError):
            f.parse(text)


@pytest.mark.parametrize("mutation", [
    lambda rs: rs.pop(),
    lambda rs: rs[1].update(id=rs[0]["id"]),
    lambda rs: rs[0]["B_windows"][0].update(mode="A_original_rules_whole_clip_same_coordinates"),
    lambda rs: rs[0]["B_windows"][0].update(support_status="partial"),
    lambda rs: rs[0].update(B_complete_clip_motion_caption=rs[0]["A_whole_clip"]["motion_caption"]),
    lambda rs: rs[0].update(semantic_approval=0),
])
def test_ms_resealed_negative(bundle, mutation):
    edit_caption(bundle, "ms12k", mutation)
    assert f.verify(bundle)["checks"]["ms12k"]["status"] == "rejected"


@pytest.mark.parametrize("mutation", [
    lambda rs: rs.pop(),
    lambda rs: rs[0]["windows"][0].update(generation=None),
    lambda rs: rs[0]["windows"][0]["generation"].update(finish_reason="length"),
    lambda rs: rs[0]["windows"][0]["events"][0].update(frame_indices=[0]),
    lambda rs: rs[0]["windows"][0]["events"][0].update(frame_indices=[0, 999]),
    lambda rs: rs[0].update(semantic_approval=0),
])
def test_vlm_resealed_negative(bundle, mutation):
    edit_caption(bundle, "vlm12k", mutation)
    assert f.verify(bundle)["checks"]["vlm12k"]["status"] == "rejected"


def test_candidate_plus_abstain_is_finite_not_full_caption(bundle):
    def partial(rows):
        rows[0]["windows"][1].update(status="abstain", caption="", events=[])
    edit_caption(bundle, "vlm12k", partial)
    def recount(m):
        m.update(usable_full_caption_count=7, incomplete_caption_count=1, partial_nonempty_caption_count=1, actual_candidate_window_count=15, actual_abstain_window_count=1, caption_delivery_complete=False, goal_full_caption_coverage_satisfied=False)
        m["window_status_counts"].update(candidate=15, abstain=1)
    reseal(bundle["vlm12k"]["manifest"], recount)
    result = f.verify(bundle)
    assert result["status"] == "pending", result
    assert result["required_pending"] == ["caption_delivery"]
    assert result["terminal_ledger_complete"] and not result["caption_delivery_complete"]
    assert not result["technical_goal_complete"]


def test_wrong_eval15_role(bundle, monkeypatch):
    reseal(bundle["eval15"]["report"], lambda r: r["checks"][0].update(model_id="MS/alias_v1"))
    monkeypatch.setattr(f, "EVAL_SHA", bundle["eval15"]["report"]["sha256"])
    assert f.verify(bundle)["checks"]["eval15"]["status"] == "rejected"


def test_stage_only_not_commit(bundle, monkeypatch):
    previous = f.git
    monkeypatch.setattr(f, "git", lambda repo, *args: b"old committed bytes" if args[0] == "show" else previous(repo, *args))
    assert f.verify(bundle)["checks"]["gitpackage"]["status"] == "rejected"


def test_remote_wrong_branch(bundle):
    bundle["gitpackage"]["remote_branch"] = "other"
    assert f.verify(bundle)["checks"]["gitpackage"]["status"] == "rejected"


def test_private_package(bundle):
    bundle["gitpackage"]["files"].append({"path": "raw/captions.jsonl", "sha256": "a" * 64})
    assert f.verify(bundle)["checks"]["gitpackage"]["status"] == "rejected"


@pytest.mark.parametrize("mutation,expected", [
    (lambda r: r[0]["data"]["apps"].pop(), "rejected"),
    (lambda r: r[0]["data"]["apps"][0].update(holder=False, CVD=[], argv=["VLLM::Worker_TP0"]), "pending"),
    (lambda r: r[0]["data"]["reserves"].append({"path": "/tmp/unknown", "uid": 0, "size": 0, "mtime_ns": int(datetime.now(timezone.utc).timestamp() * 1e9)}), "pending"),
    (lambda r: r[0]["data"].update(utc="2000-01-01T00:00:00+00:00"), "rejected"),
])
def test_final64_negative(bundle, mutation, expected):
    reseal(bundle["resources"]["snapshot"], mutation)
    assert f.verify(bundle)["checks"]["resources"]["status"] == expected


def test_offline_external_checks_pending(bundle):
    result = f.verify(bundle, offline=True)
    assert result["status"] == "pending"
    for name in ("eval15", "gitpackage", "resources"):
        assert result["checks"][name]["status"] == "pending"
    assert not result["terminal_ledger_complete"]


def test_no_optional_escape_or_disabled_feature(bundle):
    bundle["scope"]["tasks"]["rgb32points"]["required"] = False
    assert f.verify(bundle)["checks"]["scope"]["status"] == "rejected"
    bundle["feature_enabled"] = False
    result = f.verify(bundle)
    assert result["status"] == "rejected" and result["all_required_checks_executed"]


def test_required_blocked_is_pending(bundle):
    bundle["scope"]["tasks"]["rgb32points"].update(status="blocked", actual_executed=False)
    assert f.verify(bundle)["checks"]["scope"]["status"] == "pending"


def test_missing_artifact_pending(bundle):
    bundle["ms12k"]["complete"]["path"] += ".missing"
    assert f.verify(bundle)["checks"]["ms12k"]["status"] == "pending"


def test_bad_artifact_sha_rejected(bundle):
    bundle["ms12k"]["complete"]["sha256"] = "f" * 64
    assert f.verify(bundle)["checks"]["ms12k"]["status"] == "rejected"


def test_ms_abstain_preserves_terminal_and_usable_denominator(bundle):
    def abstain(rows):
        rows[0]["B_windows"][1].update(status="abstain", abstain_reason="partial_support", support_status="partial", candidate_text_eligible=False, motion_caption=None)
        rows[0]["B_complete_clip_motion_caption"] = None
    edit_caption(bundle, "ms12k", abstain)
    section = bundle["ms12k"]
    def counts(value):
        value["caption_coverage"]["B_all_planned_windows_candidate_clip_count"] = 7
        value["B_window_status_counts"] = {"candidate": 15, "abstain": 1}
    reseal(section["complete"], counts)
    reseal(section["complete_receipt"], lambda v: v.update(complete_sha256=section["complete"]["sha256"]))
    reseal(section["verification"], lambda v: (v.update(existing_complete_sha256=section["complete"]["sha256"]), v["caption_coverage"].update(B_all_planned_windows_candidate_clip_count=7)))
    result = f.verify(bundle)
    assert result["checks"]["ms12k"]["status"] == "verified", result
    detail = result["checks"]["ms12k"]["details"]
    assert detail["usable_full_caption_count"] == 7 and detail["abstain_windows"] == 1
    assert detail["caption_delivery_complete"] is False


def test_stale_after_restoration_is_pending(bundle, monkeypatch):
    original = f.datetime
    class Later(original):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(original.now(timezone.utc).timestamp() + 600, tz=tz)
    monkeypatch.setattr(f, "datetime", Later)
    assert f.verify(bundle)["checks"]["resources"]["status"] == "pending"


def test_remote_head_wrong(bundle, monkeypatch):
    previous = f.git
    monkeypatch.setattr(f, "git", lambda root, *args: b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\trefs/heads/codex/motionscript-optimization" if args[0] == "ls-remote" else previous(root, *args))
    assert f.verify(bundle)["checks"]["gitpackage"]["status"] == "rejected"


def test_new_scope_asset_stage_not_actual_execution(bundle):
    ref = bundle["scope"]["tasks"]["rgb32points"]["evidence"]
    reseal(ref, lambda value: value.update(records=[]))
    assert f.verify(bundle)["checks"]["scope"]["status"] == "rejected"


def test_false_bool_count_and_rawsha_binding(bundle):
    edit_caption(bundle, "vlm12k", lambda rows: rows[0]["windows"][0]["generation"].update(completion_tokens=True))
    assert f.verify(bundle)["checks"]["vlm12k"]["status"] == "rejected"


def test_raw_caption_embedded_small_json_is_private(bundle):
    section = bundle["gitpackage"]
    path = Path(section["repo"]) / "results/finite_metrics.json"
    item = dump(path, {"record": {"motion_caption": "private source caption"}})
    item["path"] = "results/finite_metrics.json"
    section["files"].append(item)
    section["required_files"].append(item["path"])
    assert f.verify(bundle)["checks"]["gitpackage"]["status"] == "rejected"


def test_parent_wait_restore_receipt_is_legal(bundle):
    result = f.verify(bundle)
    assert result["checks"]["resources"]["status"] == "verified", result
    assert result["checks"]["resources"]["details"]["managed_parent_wait_not_exit0"] is True


def test_private_actual_file_and_caption_delivery_receipt_absent(bundle):
    bundle["vlm12k"]["manifest"]["path"] += ".missing"
    result = f.verify(bundle)
    assert result["checks"]["vlm12k"]["status"] == "pending"
    assert result["caption_delivery_complete"] is False


def test_gzip_manifest_http_bytes_are_decoded(bundle, monkeypatch):
    original = f.http_bytes
    def compressed(url, status, limit, headers=None):
        payload, response_headers = original(url, status, limit, headers)
        return (f.gzip.compress(payload), {"Content-Encoding": "gzip"}) if status == 200 else (payload, response_headers)
    monkeypatch.setattr(f, "http_bytes", compressed)
    assert f.verify(bundle)["checks"]["eval15"]["status"] == "verified"


def test_unknown_http_encoding_rejected(bundle, monkeypatch):
    original = f.http_bytes
    monkeypatch.setattr(f, "http_bytes", lambda url, status, limit, headers=None: (original(url, status, limit, headers)[0], {"Content-Encoding": "br"}) if status == 200 else original(url, status, limit, headers))
    assert f.verify(bundle)["checks"]["eval15"]["status"] == "rejected"


def test_aged_project_reserve_needs_no_fictional_writer_pid(bundle):
    def aged(value):
        timestamp = f.utc(value[0]["data"]["utc"])
        value[0]["data"]["reserves"].append({"path": "/tmp/od_reserve_3", "uid": 0, "size": 0, "mtime_ns": int((timestamp - 1500) * 1e9), "age_s": 0.0})
    reseal(bundle["resources"]["snapshot"], aged)
    result = f.verify(bundle)
    assert result["checks"]["resources"]["status"] == "verified", result
    assert result["checks"]["resources"]["details"]["inactive_project_reserves"] == 1


def test_fresh_reserve_cannot_lie_about_age(bundle):
    def fresh(value):
        timestamp = f.utc(value[0]["data"]["utc"])
        value[0]["data"]["reserves"].append({"path": "/tmp/od_reserve_3", "uid": 0, "size": 0, "mtime_ns": int((timestamp - 1) * 1e9), "age_s": 100000.0})
    reseal(bundle["resources"]["snapshot"], fresh)
    assert f.verify(bundle)["checks"]["resources"]["status"] == "pending"


def test_raw_late_running_baseline_holder_binding_is_legal(bundle):
    snapshot = json.loads(Path(bundle["resources"]["snapshot"]["path"]).read_bytes())
    ref = bundle["resources"]["baseline"]
    ref.update(dump(Path(ref["path"]), snapshot))
    for row in snapshot:
        row["data"]["utc"] = datetime.now(timezone.utc).isoformat()
    snapshot_ref = bundle["resources"]["snapshot"]
    snapshot_ref.update(dump(Path(snapshot_ref["path"]), snapshot))
    assert f.verify(bundle)["checks"]["resources"]["status"] == "verified"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_supplemental_parent_identity_holder_successor_is_legal(bundle, exit_code):
    section = bundle["resources"]
    snapshot = json.loads(Path(section["snapshot"]["path"]).read_bytes())
    app = snapshot[0]["data"]["apps"][3]
    app.update(pid=99999, ppid=99998, start_ticks=99990)
    root = Path(section["snapshot"]["path"]).parent
    restored = {"holder_identity": copy.deepcopy(app), "parent_identity": {"pid": 99998}, "gpu_uuid": app["uuid"], "managed_parent_wait": True}
    terminal = dump(root / "diagnostic/generation_exit.json", {"exit_code": exit_code})
    restore = dump(root / "diagnostic/holder_restored.json", restored)
    section["supplemental_holder_restores"] = [{"node": 1, "gpu": 3, "receipt": restore, "terminal_receipts": [terminal]}]
    for row in snapshot:
        row["data"]["utc"] = datetime.now(timezone.utc).isoformat()
    section["snapshot"].update(dump(Path(section["snapshot"]["path"]), snapshot))
    assert f.verify(bundle)["checks"]["resources"]["status"] == "verified"


def test_holder_explicit_physical_cuda_assignment_rejected_if_wrong(bundle):
    reseal(bundle["resources"]["snapshot"], lambda value: value[0]["data"]["apps"][0].update(CVD=["CUDA_VISIBLE_DEVICES=7"]))
    assert f.verify(bundle)["checks"]["resources"]["status"] == "rejected"


def test_rgb32_missing_overlay_receipt_pending(bundle):
    bundle["scope"]["tasks"]["rgb32points"].pop("reviews")
    assert f.verify(bundle)["checks"]["scope"]["status"] == "pending"


def test_rgb32_finite_raw_not_accuracy_approval(bundle):
    ref = bundle["scope"]["tasks"]["rgb32points"]["evidence"]
    reseal(ref, lambda value: value.update(accuracy_approved=True))
    assert f.verify(bundle)["checks"]["scope"]["status"] == "rejected"


def test_rgb32_fixed_description_cannot_switch_source_raw(bundle):
    task = bundle["scope"]["tasks"]["rgb32points"]
    receipt_ref = task["descriptions"][0]["fixed2s"]
    receipt = json.loads(Path(receipt_ref["path"]).read_bytes())
    report_path = Path(receipt_ref["path"]).parent / "report.json"
    report = json.loads(report_path.read_bytes())
    report["raw_sha256"] = "f" * 64
    checksum = dump(report_path, report)["sha256"]
    reseal(receipt_ref, lambda value: value.update(report_sha256=checksum))
    assert f.verify(bundle)["checks"]["scope"]["status"] == "rejected"


@pytest.fixture
def derived_scope(bundle, tmp_path, monkeypatch):
    point = bundle["scope"]["tasks"]["rgb32points"]
    old = json.loads(Path(point["evidence"]["path"]).read_bytes())
    originals, cases = [], []
    for record in old["records"]:
        sid = record["id"]
        originals.append({"id": sid, "raw_manifest_sha256": record["manifest_sha256"], "raw_npz_sha256": record["raw_sha256"]})
        excluded = 1017 if sid == "wJPrpFsThWAcBQIw" else 0
        case = {"id": sid, "source_manifest_sha256": record["manifest_sha256"], "source_raw_sha256": record["raw_sha256"], "masks_sha256": "a" * 64, "derived_coords_sha256": "b" * 64, "numeric_valid_point_frames": 2592, "necessary_support_point_frames": 2592 - excluded, "excluded_point_frames": excluded, "excluded_by_point": {"synthetic": excluded}, "original_no_gap_baseline_exact_equal": True, "no_rejected_point_gap_fill": True, "original_event_count": 2, "derived_event_count": 1, "derived_foot_event_count": 0}
        report = {**case, "schema": "nlf-direct32-inframe-necessary-support-candidates-v1", "target_valid_requires_both_raw_brackets": True, "original_no_gap_coords_exact_equal": True, "precise_geometry_approved": False, "caption_is_training_target": False, "semantic_approval": False, "raw_modified": False, "events": [{"synthetic": True}]}
        ref = dump(tmp_path / "derived" / sid / "report.json", report)
        case.update(report_path=ref["path"], report_sha256=ref["sha256"])
        cases.append(case)
    review = {"schema": "nlf-direct32-fixed3-real-extraction-review-delivery-v1", "cases": originals, "three_real_raw_verified": True, "three_cpu_overlays_generated_and_decoded": True, "three_whole_and_fixed2s_descriptions_verified": True, "real_rgb_points_not_template": True, "accuracy_completed": False, "new_points_better_than_complete_old_pose_proven": False, "raw_is_ground_truth": False, "description_training_approved": False, "full_video_human_review_completed": False, "source_or_full_vlm_modified": False, "actual_spec_sha256": f.RGB_SPEC, "actual_model_sha256": f.RGB_MODEL, "extractor_source_sha256": f.RGB_SOURCE}
    review_ref = dump(tmp_path / "derived/originals.json", review)
    terminal = {"schema": "nlf-direct32-inframe-control-terminal-v1", "complete": True, "fixed3_count": 3, "cases": cases, "original_events_preserved": True, "immutable_operators_used": True, "rejected_points_not_gap_filled": True, "redshirt_outside_foot_events_blocked": True, "inframe_occlusion_and_heel_location_not_resolved": True, "mask_is_visibility_estimate": False, "raw_geometry_reextracted": False, "geometry_or_caption_accuracy_advantage_established": False, "calibrated_accuracy_claim": False, "training_approved": False, "source_modified": False, "gpu_used": False, "implementation_sha256": {str(SOURCE): hashlib.sha256(SOURCE.read_bytes()).hexdigest()}}
    terminal_ref = dump(tmp_path / "derived/terminal.json", terminal)
    point.update(evidence=terminal_ref, extraction_review=review_ref)
    monkeypatch.setattr(f, "RGB_REVIEW_SHA", review_ref["sha256"])
    monkeypatch.setattr(f, "RGB_CONTROL_SHA", terminal_ref["sha256"])
    return bundle


def test_actual_derived_scope_schema_accepts_finite_negative_accuracy(derived_scope):
    result = f.verify(derived_scope)
    assert result["checks"]["scope"]["status"] == "verified", result
    assert result["human_quality_approved"] is False


def test_derived_mask_cannot_be_called_visibility(derived_scope, monkeypatch):
    ref = derived_scope["scope"]["tasks"]["rgb32points"]["evidence"]
    reseal(ref, lambda value: value.update(mask_is_visibility_estimate=True))
    monkeypatch.setattr(f, "RGB_CONTROL_SHA", ref["sha256"])
    assert f.verify(derived_scope)["checks"]["scope"]["status"] == "rejected"


def test_derived_source_counts_not_filled_or_omitted(derived_scope, monkeypatch):
    ref = derived_scope["scope"]["tasks"]["rgb32points"]["evidence"]
    reseal(ref, lambda value: value["cases"][0].update(no_rejected_point_gap_fill=False))
    monkeypatch.setattr(f, "RGB_CONTROL_SHA", ref["sha256"])
    assert f.verify(derived_scope)["checks"]["scope"]["status"] == "rejected"


def test_legacy_rgb_contract_optional_support_control_actual_schema(derived_scope, tmp_path):
    task = derived_scope["scope"]["tasks"]["rgb32points"]
    task["support_control"] = task["evidence"]
    original_path = tmp_path / "points/finished.json"
    task["evidence"] = {"path": str(original_path), "sha256": hashlib.sha256(original_path.read_bytes()).hexdigest()}
    assert f.verify(derived_scope)["checks"]["scope"]["status"] == "verified"


def test_legacy_rgb_support_control_cannot_upgrade_visibility(derived_scope, tmp_path, monkeypatch):
    task = derived_scope["scope"]["tasks"]["rgb32points"]
    task["support_control"] = task["evidence"]
    original_path = tmp_path / "points/finished.json"
    task["evidence"] = {"path": str(original_path), "sha256": hashlib.sha256(original_path.read_bytes()).hexdigest()}
    reseal(task["support_control"], lambda value: value.update(mask_is_visibility_estimate=True))
    monkeypatch.setattr(f, "RGB_CONTROL_SHA", task["support_control"]["sha256"])
    assert f.verify(derived_scope)["checks"]["scope"]["status"] == "rejected"


@pytest.mark.parametrize("delivery,expected_exit", [(True, 0), (False, 2)])
def test_cli_goal_exit_only_with_full_delivery(bundle, tmp_path, monkeypatch, delivery, expected_exit):
    if not delivery:
        edit_caption(bundle, "vlm12k", lambda rows: rows[0]["windows"][1].update(status="abstain", caption="", events=[]))
        def recount(m):
            m.update(usable_full_caption_count=7, incomplete_caption_count=1, partial_nonempty_caption_count=1, actual_candidate_window_count=15, actual_abstain_window_count=1, caption_delivery_complete=False, goal_full_caption_coverage_satisfied=False)
            m["window_status_counts"].update(candidate=15, abstain=1)
        reseal(bundle["vlm12k"]["manifest"], recount)
    ledger_result = f.verify(bundle)  # All six real check functions execute on explicitly synthetic data.
    assert ledger_result["terminal_ledger_complete"] is True
    assert ledger_result["caption_delivery_complete"] is delivery
    # Only exercise CLI output/exit here. No reduced production denominator or fixture CLI flag.
    monkeypatch.setattr(f, "COUNT", 12000)
    monkeypatch.setattr(f, "verify", lambda contract, offline=False: copy.deepcopy(ledger_result))
    contract_path = tmp_path / "cli-contract.json"
    dump(contract_path, {"synthetic_unit_test": True})
    out = tmp_path / "cli-result.json"
    assert f.main(["--contract", str(contract_path), "--out", str(out)]) == expected_exit
    assert f.COUNT == 12000
    result = json.loads(out.read_bytes())
    assert result["checker_source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert result["checker_source_unchanged"] is True
    if not delivery:
        assert result["status"] == "pending" and result["required_pending"] == ["caption_delivery"]
        assert result["technical_goal_complete"] is False


def test_cli_source_version_change_rejects_final_without_modifying_source(tmp_path, monkeypatch):
    source_before = SOURCE.read_bytes()
    original_digest = f.digest_file
    calls = []
    def changing_digest(path):
        if Path(path).resolve() == SOURCE.resolve():
            calls.append(path)
            return "a" * 64 if len(calls) == 1 else "b" * 64
        return original_digest(path)
    monkeypatch.setattr(f, "digest_file", changing_digest)
    contract = tmp_path / "contract.json"
    dump(contract, {"schema": f.CONTRACT_SCHEMA, "feature_enabled": True, "required_checks": list(f.CHECKS)})
    out = tmp_path / "source-changed-result.json"
    assert f.main(["--contract", str(contract), "--out", str(out), "--offline"]) == 1
    result = json.loads(out.read_bytes())
    assert result["checker_source_sha256"] == "a" * 64
    assert result["checker_source_sha256_after"] == "b" * 64
    assert result["checker_source_unchanged"] is False and result["status"] == "rejected"
    assert result["technical_goal_complete"] is False
    assert SOURCE.read_bytes() == source_before


def test_latest_required_names_are_missing_not_completed_by_old_trials(bundle, monkeypatch):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    result = f.verify(bundle)
    assert result["status"] == "pending" and result["technical_goal_complete"] is False
    assert result["checks"]["scope"]["status"] == "pending"
    for name in f.LATEST_SCOPE_TRIALS:
        assert name + ":missing_actual_contract" in result["checks"]["scope"]["reason"]
    assert result["all_required_checks_executed"] is True


@pytest.mark.parametrize("name", ["freshdetector32", "bpm_windows", "truebar"])
def test_new_user_required_trial_cannot_be_optional(bundle, monkeypatch, name):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    bundle["scope"]["tasks"][name] = {"required": False, "status": "optional", "actual_executed": False}
    assert f.verify(bundle)["checks"]["scope"]["status"] == "rejected"


@pytest.mark.parametrize("name", ["bpm_windows", "truebar"])
def test_old_C50_cannot_substitute_actual_bpm_or_truebar(bundle, monkeypatch, name):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    task = copy.deepcopy(bundle["scope"]["tasks"]["beat50"])
    bundle["scope"]["tasks"][name] = task
    assert f.verify(bundle)["checks"]["scope"]["status"] == "rejected"


def test_old_cached_rgb_cannot_substitute_fresh_detection(bundle, monkeypatch):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    bundle["scope"]["tasks"]["freshdetector32"] = copy.deepcopy(bundle["scope"]["tasks"]["rgb32points"])
    assert f.verify(bundle)["checks"]["scope"]["status"] == "rejected"


def test_planned_latest_trials_have_no_executed_approval(bundle, monkeypatch):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    for name in f.LATEST_SCOPE_TRIALS:
        bundle["scope"]["tasks"][name] = {"required": True, "status": "pending", "actual_executed": False}
    result = f.verify(bundle)
    assert result["checks"]["scope"]["status"] == "pending" and result["status"] == "pending"
    assert result["terminal_ledger_complete"] is False


@pytest.fixture
def fresh_detector_fixture(tmp_path, monkeypatch):
    rows = [{"id": sid, "source_group": "synthetic", "split": "test", "video_path": "/synthetic/" + sid + ".mp4"} for sid in sorted(f.IDS)]
    path = tmp_path / "input/fixed3.jsonl"
    input_sha = jsonl(path, rows)
    monkeypatch.setattr(f, "FIXED3_INPUT_SHA", input_sha)
    extractor = dump(tmp_path / "fresh-extractor.py", {"synthetic": True})
    monkeypatch.setattr(f, "FRESH32_SOURCE", extractor["sha256"])
    cases = []
    for row in rows:
        report = {"schema": "motionscript_direct_landmarks_v1", "source": "nlf_fresh_detector_rgb_landmarks", "id": row["id"], "input_record": row, "point_count": 32, "frame_count": 3, "detection_count": [1, 0, 2], "selected_index": [0, -1, 1], "raw_sha256": "a" * 64, "fresh_detector_used": True, "no_external_boxes_supplied": True, "all_detection_boxes_and_direct_geometry_preserved": True, "body_anchors_queried_together_with_extras": True, "zero_detections_keep_nan_mask": True, "detector_is_separate_network": True, "crop_features_shared_across_32_queries": True, "external_pose_or_box_cache_conditioning": False, "person_tracking_added": False, "smpl_fitting_used": False, "template_fk_used": False, "visibility_verified": False, "semantic_approval": False, "raw_is_ground_truth": False, "single_total_network_forward": False, "implementation_sha256": extractor["sha256"], "source_sha256": {"model": f.RGB_MODEL, "spec": f.RGB_SPEC, "input_manifest": input_sha}, "detector_config": {"extra_boxes": None}, "selection_policy": "per-frame largest XYWH area; argmax first tie; not tracking or guaranteed same identity"}
        ref = dump(tmp_path / "fresh" / row["id"] / "manifest.json", report)
        cases.append({"id": row["id"], "manifest_path": ref["path"], "manifest_sha256": ref["sha256"], "raw_sha256": "a" * 64, "frame_count": 3, "detected_frames": 2, "multiple_person_frames": 1})
    terminal = {"schema": "nlf-fresh-detector32-terminal-v1", "complete": True, "expected": 3, "actual": 3, "cases": cases, "implementation_sha256": extractor["sha256"], "input_manifest_path": str(path), "input_manifest_sha256": input_sha, "no_external_boxes_supplied": True, "person_tracking_added": False, "smpl_fitting_used": False, "template_fk_used": False, "visibility_verified": False, "semantic_approval": False}
    return {"required": True, "status": "verified", "actual_executed": True, "extractor": extractor, "evidence": dump(tmp_path / "fresh/finished.json", terminal)}


def test_fresh_raw_complete_does_not_replace_required_comparison(bundle, fresh_detector_fixture, monkeypatch):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    bundle["scope"]["tasks"]["freshdetector32"] = fresh_detector_fixture
    result = f.verify(bundle)
    assert result["checks"]["scope"]["status"] == "pending"
    assert "required actual comparison receipt absent" in result["checks"]["scope"]["reason"]
    assert result["technical_goal_complete"] is False


def test_fresh_detector_actual_schema_with_zero_and_multiple_people(fresh_detector_fixture):
    result = f.check_freshdetector32(fresh_detector_fixture)
    assert result["cases"] == 3 and result["visibility_approved"] is False


@pytest.mark.parametrize("field,value", [("source", "nlf_direct_rgb_landmarks"), ("fresh_detector_used", 1), ("external_pose_or_box_cache_conditioning", True), ("selected_index", [0, 0, 1]), ("detection_count", [True, 0, 2]), ("single_total_network_forward", True)])
def test_fresh_detector_resealed_protocol_negative(fresh_detector_fixture, field, value):
    ref = fresh_detector_fixture["evidence"]
    terminal = json.loads(Path(ref["path"]).read_bytes())
    case = terminal["cases"][0]
    report_path = Path(case["manifest_path"])
    report = json.loads(report_path.read_bytes())
    report[field] = value
    case["manifest_sha256"] = dump(report_path, report)["sha256"]
    ref.update(dump(Path(ref["path"]), terminal))
    with pytest.raises(ValueError):
        f.check_freshdetector32(fresh_detector_fixture)


@pytest.fixture
def fresh_comparison_fixture(fresh_detector_fixture, tmp_path, monkeypatch):
    task = fresh_detector_fixture
    producer = json.loads(Path(task["evidence"]["path"]).read_bytes())
    cases, independent_cases = [], []
    for original in producer["cases"]:
        receipt = {"schema": "nlf-fresh-detector32-cachedbox32-compare-v1", "actual_fresh_detection": True, "fresh_manifest_sha256": original["manifest_sha256"], "frame_count": 3, "cached_necessary_pointframes": 60, "fresh_necessary_pointframes": 60, "old_cache_conditioning_used": False, "person_tracking": False, "difference_is_accuracy_metric": False, "in_frame_mask_is_visibility": False, "accuracy_approved": False, "training_approved": False, "source_or_raw_modified": False}
        ref = dump(tmp_path / "compare" / original["id"] / "comparison_receipt.json", receipt)
        case = {**original, "comparison_receipt_path": ref["path"], "comparison_receipt_sha256": ref["sha256"], "independent_allpeople_offsets_largestarea_and_emptyshape_verified": True, "necessary_pointframes_cached_fresh": [60, 60]}
        cases.append(case)
        independent_cases.append({"id": original["id"], "comparison_receipt_sha256": ref["sha256"], "necessary_pointframes": [60, 60]})
    verified_ref = dump(tmp_path / "compare/verified.json", {"schema": "nlf-fresh-detector32-fixed3-independent-comparison-verification-v1", "comparison_complete": True, "accuracy_approved": False, "cases": independent_cases})
    helper_ref = dump(tmp_path / "compare/helper.py", {"synthetic": True})
    monkeypatch.setattr(f, "FRESH32_COMPARE_HELPER", helper_ref["sha256"])
    worker = dump(tmp_path / "compare/worker_exit.json", {"exit_code": 0})
    restore = dump(tmp_path / "compare/restore.json", {"managed_parent_wait": True, "holder_identity": {"ppid": 99}, "manager_identity": {"pid": 99}})
    report = {"schema": "nlf-fresh-detector32-fixed3-execution-and-compare-terminal-v1", "complete": True, "actual_fresh_detector_execution": True, "actual_fixed3": 3, "source_sha256": f.FRESH32_SOURCE, "producer_terminal_sha256": task["evidence"]["sha256"], "producer_terminal_path": task["evidence"]["path"], "compare_helper_sha256": helper_ref["sha256"], "compare_helper_path": helper_ref["path"], "same_video_spec_model_names_and_PTS_comparison_verified": True, "selected32_and_all_ragged_preserved": True, "no_legacy_nlfbox_or_pose_conditioning": True, "no_smplfit_or_templateFK": True, "detector_is_separate_neural_network": True, "single_total_neural_forward_claim": False, "selection_is_person_tracking": False, "inframe_mask_is_visibility": False, "accuracy_approved": False, "training_approved": False, "raw_is_GT": False, "frozen_library_or_fullVLM_modified": False, "private_media_uploaded": False, "independent_comparison_receipt_path": verified_ref["path"], "independent_comparison_receipt_sha256": verified_ref["sha256"], "cases": cases, "resource_refs": {"worker_exit.json": worker, "holder_restored.json": restore}}
    task["comparison"] = dump(tmp_path / "compare/terminal.json", report)
    monkeypatch.setattr(f, "FRESH32_COMPARISON_SHA", task["comparison"]["sha256"])
    return task


def test_fresh_execution_and_comparison_actual_schema(fresh_comparison_fixture):
    assert f.check_freshdetector32_comparison(fresh_comparison_fixture)["comparison_complete"] is True


@pytest.mark.parametrize("key,value", [("accuracy_approved", True), ("actual_fresh_detector_execution", 1), ("source_sha256", "a" * 64), ("selection_is_person_tracking", True)])
def test_fresh_comparison_resealed_claims_rejected(fresh_comparison_fixture, monkeypatch, key, value):
    ref = fresh_comparison_fixture["comparison"]
    reseal(ref, lambda report: report.update({key: value}))
    monkeypatch.setattr(f, "FRESH32_COMPARISON_SHA", ref["sha256"])
    with pytest.raises(ValueError):
        f.check_freshdetector32_comparison(fresh_comparison_fixture)


def test_latest_point_comparison_complete_does_not_replace_fresh_caption_control(bundle, fresh_comparison_fixture, monkeypatch):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    bundle["scope"]["tasks"]["freshdetector32"] = fresh_comparison_fixture
    result = f.verify(bundle)
    assert result["checks"]["scope"]["status"] == "pending"
    assert "new required fresh mask-caption control receipt absent" in result["checks"]["scope"]["reason"]
    assert result["technical_goal_complete"] is False


@pytest.fixture
def fresh_caption_fixture(fresh_comparison_fixture, tmp_path, monkeypatch):
    task = fresh_comparison_fixture
    producer = json.loads(Path(task["evidence"]["path"]).read_bytes())
    cases, controls = [], []
    for original in producer["cases"]:
        runs = []
        for source in ("fresh", "cached"):
            for mode in ("numeric", "necessary"):
                report = {"schema": "nlf-fresh-cached32-support-caption-control-v1", "id": original["id"], "source_mode": source, "support_mode": mode, "source_manifest_sha256": original["manifest_sha256"], "source_raw_sha256": original["raw_sha256"], "numeric_valid_pointframes": 96, "selected_support_pointframes": 60, "excluded_pointframes": 36, "whole_event_count": 1, "foot_event_count": 0, "events": [{"synthetic": True}], "target_support_both_raw_brackets_verified": True, "no_missing_point_fill": True, "actual_inframe_is_visibility": False, "calibrated_geometry_accuracy": False, "caption_is_training_target": False, "semantic_approval": False, "raw_modified": False}
                ref = dump(tmp_path / "caption" / original["id"] / (source + mode) / "report.json", report)
                run = {k: report[k] for k in ("source_manifest_sha256", "source_raw_sha256", "numeric_valid_pointframes", "selected_support_pointframes", "excluded_pointframes", "whole_event_count", "foot_event_count")}
                run.update(source=source, mode=mode, report_path=ref["path"], report_sha256=ref["sha256"], coords_sha256="a" * 64)
                runs.append(run)
                controls.append({"id": original["id"], "source": source, "mask": mode, "report_sha256": ref["sha256"], "coords_sha256": "a" * 64})
        cases.append({"id": original["id"], "runs": runs})
    helper = dump(tmp_path / "caption/helper.py", {"synthetic": True})
    monkeypatch.setattr(f, "FRESH32_CAPTION_HELPER", helper["sha256"])
    terminal = {"schema": "nlf-fresh-detector32-caption-support-control-terminal-v1", "complete": True, "fixed3_count": 3, "support_control_runs": 12, "cases": cases, "necessary_support_not_visibility": True, "missing_points_not_interpolated_across_rejection": True, "redshirt_outside_foot_events_blocked": True, "old_cached_raw_or_reports_modified": False, "fresh_raw_or_producer_modified": False, "caption_accuracy_advantage_established": False, "semantic_approval": False, "training_approved": False, "gpu_used": False, "helper_sha256": helper["sha256"]}
    terminal_ref = dump(tmp_path / "caption/terminal.json", terminal)
    monkeypatch.setattr(f, "FRESH32_CAPTION_SHA", terminal_ref["sha256"])
    verified_ref = dump(tmp_path / "caption/verified.json", {"independently_verified": True, "accuracy_approved": False, "frozen_source_unchanged": True, "reports_verified": 12, "control_terminal_sha256": terminal_ref["sha256"], "controls": controls})
    closeout = {"schema": "nlf-fresh-detector32-finite-execution-caption-control-closeout-v1", "complete": True, "execution_ref": task["comparison"], "caption_control_ref": terminal_ref, "helper_ref": helper, "independent_control_ref": verified_ref, "fresh_actual_fixed3_gpu_executed": True, "comparison_actual_fixed3_cpu_completed": True, "source_and_old_public_libraries_frozen": True, "missing_rejected_points_not_filled": True, "mask_support_is_visibility": False, "accuracy_advantage_established": False, "GT_approved": False, "training_approved": False, "tracking_guaranteed": False, "raw_updated": False, "geometry_caption_support_controls_verified": 12}
    task["caption_control"] = terminal_ref
    task["closeout"] = dump(tmp_path / "caption/closeout.json", closeout)
    monkeypatch.setattr(f, "FRESH32_CLOSEOUT_SHA", task["closeout"]["sha256"])
    return task


def test_actual_fresh_caption_control_schema_positive(fresh_caption_fixture):
    assert f.check_freshdetector32_caption_control(fresh_caption_fixture)["controls"] == 12


@pytest.mark.parametrize("key,value", [("mask_support_is_visibility", True), ("accuracy_advantage_established", True), ("geometry_caption_support_controls_verified", True)])
def test_fresh_caption_resealed_quality_or_count_claim_rejected(fresh_caption_fixture, monkeypatch, key, value):
    ref = fresh_caption_fixture["closeout"]
    reseal(ref, lambda report: report.update({key: value}))
    monkeypatch.setattr(f, "FRESH32_CLOSEOUT_SHA", ref["sha256"])
    with pytest.raises(ValueError):
        f.check_freshdetector32_caption_control(fresh_caption_fixture)


def test_latest_fresh_all_layers_complete_leaves_bpm_truebar_pending(bundle, fresh_caption_fixture, monkeypatch):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    bundle["scope"]["tasks"]["freshdetector32"] = fresh_caption_fixture
    result = f.verify(bundle)
    assert result["checks"]["scope"]["status"] == "pending"
    assert "freshdetector32:" not in result["checks"]["scope"]["reason"]
    assert "bpm_windows:missing_actual_contract" in result["checks"]["scope"]["reason"]
    assert "truebar:missing_actual_contract" in result["checks"]["scope"]["reason"]


def bpm_plan(binding, row, baseline_ref):
    return {"schema": "motionscript_music_windows_v1", "id": row["id"], "binding": binding, "strategy": "bpm_duration_grid_not_detected_beats", "status": "ready", "window_interval": "[start,end)", "time_basis": "global_media_seconds", "continuous_exact_media_coverage": True, "audio_video_alignment_verified": False, "beat_phase_verified": False, "downbeat_verified": False, "motions_on_beats_verified": False, "window_boundaries_are_action_boundaries": False, "semantic_approval": False, "human_review_completed": False, "params": {"bpm_estimate": 120.0, "beats_per_chunk": 4, "target_seconds": 2.0, "min_tail_seconds": 1.5, "first_zero_is_beat_phase": False}, "raw_windows": [{"start": 0.0, "end": 2.0, "kind": "bpm_duration_grid", "grid_index": 0, "ends_at_actual_media_end": False}, {"start": 2.0, "end": 4.0, "kind": "bpm_duration_grid", "grid_index": 1, "ends_at_actual_media_end": True}], "windows": [{"start": 0.0, "end": 2.0, "raw_window_indices": [0]}, {"start": 2.0, "end": 4.0, "raw_window_indices": [1]}], "manifest_record": row, "coordinates_reference": {"sha256": "d" * 64}, "source_C_report": baseline_ref}


@pytest.fixture
def bpm_fixture(tmp_path, monkeypatch):
    rows = [{"id": "bpm-synthetic-" + str(i), "source_group": "synthetic", "split": "train", "nlf_path": "/synthetic/" + str(i) + ".npz", "video_path": "/synthetic/" + str(i) + ".mp4", "audio_path": "/synthetic/" + str(i) + ".wav"} for i in range(50)]
    input_path = tmp_path / "bpm-input/first50.jsonl"
    input_sha = jsonl(input_path, rows)
    monkeypatch.setattr(f, "FIRST50_INPUT_SHA", input_sha)
    source_ref = dump(tmp_path / "music-window-source.py", {"synthetic_source": True})
    monkeypatch.setattr(f, "MUSIC_WINDOWS_SOURCE", source_ref["sha256"])
    protocol = {"code_sha256": {"ms_music_windows.py": source_ref["sha256"]}, "manifest_sha256": input_sha, "selected_ids": [r["id"] for r in rows], "caption_substitution_applied": False, "VLM_shared_manifest_actual_execution": False}
    fingerprint = hashlib.sha256((json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    root = tmp_path / "bpm"
    dump(root / "protocol.json", {"protocol": protocol, "fingerprint": fingerprint})
    plans, chunks, records = [], [], []
    for i, original in enumerate(rows, 1):
        row = {**original, "manifest_line": i}
        sid = row["id"]
        binding = {"id": sid, "input_sha256": "a" * 64, "video_sha256": "b" * 64, "audio_sha256": "c" * 64, "actual_video_pts_sha256": "e" * 64, "media_start": 0.0, "media_end": 4.0, "video_pts_offset": 0.0, "video_fps": 16.0, "decoded_video_frames": 64}
        baseline = {**binding, "coords_sha256": "d" * 64, "media": {"media_end_verified_from_video": True, "nlf_video_timestamps_matched": True, "nlf_video_frame_count_matched": True}}
        baseline_ref = dump(tmp_path / "bpm-baseline" / (sid + ".json"), baseline)
        plan = bpm_plan(binding, row, baseline_ref)
        plan_ref = dump(root / (sid + ".windows.json"), plan)
        plans.append(plan)
        views = []
        for index, w in enumerate(plan["windows"]):
            views.append({"chunk_index": index, "window_start": w["start"], "window_end": w["end"], "window_interval": "[start,end)", "time_basis": "global_media_seconds", "semantic_approval": False, "human_review_completed": False, "window_boundaries_are_action_boundaries": False, "sample_span_is_complete_window_evidence": False, "complete_available_samples_is_verified_window_semantics": False, "coordinate_reset_applied": False, "segment_ids_preserved": True, "source_arrays_retained_full_length": True, "status": "candidate", "cleaned_motion_text": "independently supported synthetic movement " + str(index), "support_status": "complete_available_samples", "abstain_reason": None})
        report = {"schema": "motionscript_music_window_caption_v1", "id": sid, "window_manifest": plan, "window_manifest_sha256": plan_ref["sha256"], "binding": binding, "source_group": row["source_group"], "split": row["split"], "fingerprint": fingerprint, "accepted": False, "formal_approval": False, "semantic_approval": False, "human_review_completed": False, "baseline_caption_substitution_applied": False, "MS_whole_window_semantics_verified": False, "caption_execution": "actual_original_MS_on_receipt_verified_whole_C_coordinates", "MS_windows": views, "MS_full_available_support_coverage": True, "MS_all_windows_caption_text": " \n".join(v["cleaned_motion_text"] for v in views)}
        report_ref = dump(root / (sid + ".json"), report)
        records.append({"id": sid, "report_file": sid + ".json", "report_sha256": report_ref["sha256"], "window_file": sid + ".windows.json", "window_sha256": plan_ref["sha256"], "coords_sha256": "d" * 64})
        chunks.extend({"schema": report["schema"], "id": sid, "binding": binding, "window_manifest_sha256": plan_ref["sha256"], **view} for view in views)
    windows_sha = jsonl(root / "windows.jsonl", plans)
    chunks_sha = jsonl(root / "chunks.jsonl", chunks)
    summary = {"schema": "motionscript_music_window_caption_v1", "strategy": "bpm", "fingerprint": fingerprint, "selected_count": 50, "processed_count": 50, "failed_count": 0, "complete": True, "actual_MS_caption_execution": True, "VLM_shared_manifest_actual_execution": False, "accepted": False, "formal_approval": False, "semantic_approval": False, "human_review_completed": False, "records": records, "errors": [], "window_manifest_sha256": windows_sha, "chunks_sha256": chunks_sha, "full_available_support_clip_count": 50, "A_B_C_newMS_same_processed_source_stats": {"MS": {"source_clip_denominator": 50, "window_denominator": 100, "status_counts": {"candidate": 100}, "abstain_reasons": {}, "support_status_counts": {"complete_available_samples": 100}}}}
    summary_ref = dump(root / "summary.json", summary)
    dump(root / "summary.receipt.json", {"summary_sha256": summary_ref["sha256"], "window_manifest_sha256": windows_sha, "chunks_sha256": chunks_sha})
    monkeypatch.setattr(f, "BPM_SUMMARY_SHA", summary_ref["sha256"])
    return {"required": True, "status": "verified", "actual_executed": True, "evidence": summary_ref, "source": source_ref, "input": {"path": str(input_path), "sha256": input_sha}}


def test_actual_bpm_same50_schema_positive(bpm_fixture):
    detail = f.check_bpm_windows(bpm_fixture)
    assert detail["processed_sources"] == 50 and detail["windows"] == 100
    assert detail["shared_VLM_executed"] is False and detail["audio_video_alignment_approved"] is False


@pytest.mark.parametrize("mutation", [lambda p: p["raw_windows"][0].update(start=False), lambda p: p["windows"][0].update(start=False), lambda p: p["params"].update(first_zero_is_beat_phase=True), lambda p: p.update(downbeat_verified=True), lambda p: p["windows"][0].update(raw_window_indices=[True]), lambda p: p["params"].update(target_seconds=4.0)])
def test_bpm_plan_resealed_type_phase_mapping_negative(bpm_fixture, mutation):
    root = Path(bpm_fixture["evidence"]["path"]).parent
    plan = json.loads((root / "bpm-synthetic-0.windows.json").read_bytes())
    mutation(plan)
    with pytest.raises(ValueError):
        f.verify_bpm_plan(plan)


@pytest.mark.parametrize("key,value", [("selected_count", True), ("actual_MS_caption_execution", 1), ("VLM_shared_manifest_actual_execution", True), ("full_available_support_clip_count", 49)])
def test_bpm_actual_summary_resealed_counts_and_scope_rejected(bpm_fixture, monkeypatch, key, value):
    ref = bpm_fixture["evidence"]
    reseal(ref, lambda s: s.update({key: value}))
    monkeypatch.setattr(f, "BPM_SUMMARY_SHA", ref["sha256"])
    root = Path(ref["path"]).parent
    receipt = json.loads((root / "summary.receipt.json").read_bytes())
    receipt["summary_sha256"] = ref["sha256"]
    dump(root / "summary.receipt.json", receipt)
    with pytest.raises(ValueError):
        f.check_bpm_windows(bpm_fixture)


def test_latest_fresh_bpm_verified_but_truebar_missing_is_pending(bundle, fresh_caption_fixture, bpm_fixture, monkeypatch):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    bundle["scope"]["tasks"].update(freshdetector32=fresh_caption_fixture, bpm_windows=bpm_fixture)
    result = f.verify(bundle)
    assert result["checks"]["scope"]["status"] == "pending"
    assert "truebar:missing_actual_contract" in result["checks"]["scope"]["reason"]
    assert "freshdetector32:" not in result["checks"]["scope"]["reason"]
    assert "bpm_windows:" not in result["checks"]["scope"]["reason"]
    assert result["technical_goal_complete"] is False


@pytest.fixture
def truebar_fixture(bpm_fixture, tmp_path, monkeypatch):
    source = bpm_fixture["source"]
    adapter = dump(tmp_path / "bar-adapter.py", {"synthetic": True})
    monkeypatch.setattr(f, "TRUEBAR_ADAPTER", adapter["sha256"])
    input_rows = [json.loads(line) for line in Path(bpm_fixture["input"]["path"]).read_bytes().splitlines()]
    detections = [{"schema": "beatthis_audio_detections_v1", "id": row["id"], "status": "ok", "inference_called": True, "model_prediction_returned": True, "helper_sha256": f.BEATTHIS_V1_HELPER, "model_sha256": f.BEATTHIS_MODEL, "audio_sha256": "c" * 64, "beats": [0.0, 1.0, 2.0, 3.0, 4.0], "downbeats": [0.0, 2.0, 4.0], "beat_count": 5, "downbeat_count": 3, "source_row": row, "bar_phase_accuracy_verified": False, "beat_accuracy_verified": False, "downbeat_accuracy_verified": False, "audio_video_alignment_verified": False} for row in input_rows]
    detection_path = tmp_path / "beatthis/detections.jsonl"
    output_sha = jsonl(detection_path, detections)
    raw_lines = detection_path.read_bytes().splitlines()
    run = {"schema": "beatthis_audio_detections_v1", "status": "completed", "runtime": {"helper_sha256": f.BEATTHIS_V1_HELPER}, "exit_code": 0, "expected_count": 50, "attempted_count": 50, "input_manifest_sha256": f.FIRST50_INPUT_SHA, "params": {"device": "cpu", "dbn": False, "float16": False, "postprocessor": "minimal"}, "model_runtime": {"parameter_devices": ["cpu"], "parameter_dtypes": ["torch.float32"]}, "successful_count": 50, "failed_count": 0, "output_sha256": output_sha, "model": {"checkpoint_sha256": f.BEATTHIS_MODEL}}
    run_ref = dump(tmp_path / "beatthis/run_receipt.json", run)
    post = {"schema": "beatthis_independent_output_validation_v2", "status": "completed", "exit_code": 0, "validated_count": 53, "invalid_count": 0, "actual_model_inference_rerun": False, "original_v1_outputs_and_source_untouched": True, "invalid_emptybeats_nonempty_downbeats_rejected": True, "validation_helper_sha256": f.BEATTHIS_V2_VERIFIER, "actual_v1_helper_sha256": f.BEATTHIS_V1_HELPER, "checkpoint_sha256": f.BEATTHIS_MODEL, "groups": {"first50": {"expected_count": 50, "checked_count": 50, "valid_count": 50, "invalid_count": 0, "all_outputs_equal_v2_strict_revalidation": True, "actual_v1_run_receipt_path": run_ref["path"], "actual_v1_run_receipt_sha256": run_ref["sha256"], "detections_path": str(detection_path), "detections_sha256": output_sha}}}
    post_ref = dump(tmp_path / "beatthis/postvalidation.json", post)
    monkeypatch.setattr(f, "BEATTHIS_POST_SHA", post_ref["sha256"])
    roots = {}
    summary_pins = {}
    base = Path(bpm_fixture["evidence"]["path"]).parent
    original_summary = json.loads(Path(bpm_fixture["evidence"]["path"]).read_bytes())
    for policy in ("strict", "merge_short_edges"):
        root = tmp_path / ("bar_" + policy)
        protocol = json.loads((base / "protocol.json").read_bytes())
        dump(root / "protocol.json", protocol)
        summary = copy.deepcopy(original_summary)
        summary["strategy"] = "bar"
        plans, aggregate, records = [], [], []
        for i, original_record in enumerate(original_summary["records"]):
            sid = original_record["id"]
            report = json.loads((base / original_record["report_file"]).read_bytes())
            plan = report["window_manifest"]
            plan.update(strategy="model_downbeat_bars", downbeats_seconds=[0.0, 2.0, 4.0], model_downbeat_timestamps_consumed=True, detected_downbeat_boundaries_verified_in_source=True)
            plan["params"] = {"policy": policy, "min_edge_seconds": 1.5, "short_interior_bars_merged": False, "insufficient_downbeats_fallback": False}
            plan["raw_bars"] = [{"start": 0.0, "end": 2.0, "bar_index": 0, "downbeat_indices": [0, 1]}, {"start": 2.0, "end": 4.0, "bar_index": 1, "downbeat_indices": [1, 2]}]
            plan["raw_windows"] = [{**bar, "kind": "complete_detected_bar"} for bar in plan["raw_bars"]]
            plan["provenance"] = {"method": "beat_this.inference.File2Beats", "checkpoint_name": "final0", "device": "cpu", "downbeats_origin": "model_downbeat_head_postprocessed", "dbn": False, "synthetic": False, "checkpoint_sha256": f.BEATTHIS_MODEL, "adapter_sha256": adapter["sha256"], "audio_sha256": "c" * 64, "helper_sha256": f.BEATTHIS_V1_HELPER, "raw_detections_sha256": output_sha, "run_receipt_sha256": run_ref["sha256"], "raw_detection_line": i + 1, "raw_detection_line_sha256": hashlib.sha256(raw_lines[i]).hexdigest(), "raw_beats_seconds": detections[i]["beats"], "raw_downbeats_seconds": detections[i]["downbeats"]}
            plan_ref = dump(root / original_record["window_file"], plan)
            report["window_manifest_sha256"] = plan_ref["sha256"]
            report_ref = dump(root / original_record["report_file"], report)
            record = {**original_record, "report_sha256": report_ref["sha256"], "window_sha256": plan_ref["sha256"]}
            records.append(record)
            plans.append(plan)
            aggregate.extend({"schema": report["schema"], "id": sid, "binding": report["binding"], "window_manifest_sha256": plan_ref["sha256"], **view} for view in report["MS_windows"])
        summary.update(records=records, window_manifest_sha256=jsonl(root / "windows.jsonl", plans), chunks_sha256=jsonl(root / "chunks.jsonl", aggregate))
        summary_ref = dump(root / "summary.json", summary)
        dump(root / "summary.receipt.json", {"summary_sha256": summary_ref["sha256"], "window_manifest_sha256": summary["window_manifest_sha256"], "chunks_sha256": summary["chunks_sha256"]})
        roots[policy] = {"evidence": summary_ref}
        summary_pins[policy] = summary_ref["sha256"]
    monkeypatch.setattr(f, "TRUEBAR_SUMMARIES", summary_pins)
    return {"required": True, "status": "verified", "actual_executed": True, "evidence": roots["strict"]["evidence"], "source": source, "input": bpm_fixture["input"], "adapter": adapter, "beatthis_validation": post_ref, **roots}


def test_truebar_actual_strict_merged_schema_positive(truebar_fixture):
    result = f.check_truebar(truebar_fixture)
    assert result["selected_sources"] == 50
    assert result["actual_detected_downbeat_intervals"] == 100
    assert result["human_bar_accuracy_approved"] is False


def test_prior_three_trial_scope_all_six_gates_production12000_positive(bundle, fresh_caption_fixture, bpm_fixture, truebar_fixture, monkeypatch):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    bundle["scope"]["tasks"].update(freshdetector32=fresh_caption_fixture, bpm_windows=bpm_fixture, truebar=truebar_fixture)
    # Exercise the fixed production denominator on synthetic records, never real goal evidence.
    old_inputs = [json.loads(line) for line in Path(bundle["input"]["path"]).read_bytes().splitlines()]
    count = 12000
    inputs = [{**old_inputs[i % 8], "id": "production-shape-fixture-" + str(i)} for i in range(count)]
    input_sha = jsonl(Path(bundle["input"]["path"]), inputs)
    bundle["input"]["sha256"] = input_sha
    monkeypatch.setattr(f, "INPUT_SHA", input_sha)
    monkeypatch.setattr(f, "COUNT", count)
    ms = bundle["ms12k"]
    root = Path(ms["complete"]["path"]).parent
    old_rows = [json.loads(line) for line in (root / "captions.jsonl").read_bytes().splitlines()]
    ms_rows = [{**old_rows[i % 8], "id": row["id"]} for i, row in enumerate(inputs)]
    caption_sha = jsonl(root / "captions.jsonl", ms_rows)
    lineage_sha = jsonl(root / "lineage.jsonl", inputs)
    def update_ms(value):
        value.update(unique_id_count=count, successfully_processed_count=count, B_window_status_counts={"candidate": count * 2})
        value["file_sha256"].update({"captions.jsonl": caption_sha, "lineage.jsonl": lineage_sha})
        value["binding"]["input_manifest_sha256"] = input_sha
        value["caption_coverage"].update(target_clip_denominator=count, observed_B_window_denominator=count * 2, B_all_planned_windows_candidate_clip_count=count)
    reseal(ms["complete"], update_ms)
    complete = json.loads(Path(ms["complete"]["path"]).read_bytes())
    reseal(ms["complete_receipt"], lambda value: value.update(complete_sha256=ms["complete"]["sha256"]))
    reseal(ms["verification"], lambda value: value.update(existing_complete_sha256=ms["complete"]["sha256"], input_manifest_sha256=input_sha, unique_id_count=count, canonical_file_sha256=complete["file_sha256"], caption_coverage=complete["caption_coverage"]))
    vlm = bundle["vlm12k"]
    reseal(vlm["plan"], lambda value: (value.update(target_ids=count, input_manifest_sha256=input_sha), [job.update(expected_ids=1500) for job in value["jobs"]]))
    monkeypatch.setattr(f, "VLM_PLAN", vlm["plan"]["sha256"])
    manifest = json.loads(Path(vlm["manifest"]["path"]).read_bytes())
    for link in manifest["lineage"]:
        exit_ref = {"path": str(Path(link["resource_queue"]) / "worker_exit.json")}
        terminal = json.loads(Path(exit_ref["path"]).read_bytes())
        terminal["expected_ids"] = 1500
        link["worker_exit_sha256"] = dump(Path(exit_ref["path"]), terminal)["sha256"]
        link["resource_worker_exit"] = terminal
        for resource_exit in bundle["resources"]["worker_exits"]:
            if resource_exit["path"] == exit_ref["path"]:
                resource_exit["sha256"] = link["worker_exit_sha256"]
    vroot = Path(vlm["manifest"]["path"]).parent
    old_vrows = [json.loads(line) for line in (vroot / "captions.jsonl").read_bytes().splitlines()]
    vrows = [{**old_vrows[i % 8], **row, "original_input": row, "shard": i % 8} for i, row in enumerate(inputs)]
    manifest.update(captions_sha256=jsonl(vroot / "captions.jsonl", vrows), launch_plan_sha256=vlm["plan"]["sha256"], input_manifest_sha256=input_sha, expected_ids=count, verified_ids=count, expected_window_count=count * 2, usable_full_caption_count=count, clip_denominator=count, window_denominator=count * 2, actual_candidate_window_count=count * 2, any_nonempty_caption_count=count, actual_terminal_window_receipt_count=count * 2, sid_status_counts={"candidate": count, "abstain": 0, "failed": 0}, window_status_counts={"candidate": count * 2, "abstain": 0, "failed": 0})
    vlm["manifest"].update(dump(Path(vlm["manifest"]["path"]), manifest))
    snapshot_ref = bundle["resources"]["snapshot"]
    snapshot = json.loads(Path(snapshot_ref["path"]).read_bytes())
    for node in snapshot:
        node["data"]["utc"] = datetime.now(timezone.utc).isoformat()
    snapshot_ref.update(dump(Path(snapshot_ref["path"]), snapshot))
    result = f.verify(bundle)
    assert result["status"] == "verified", result
    assert result["terminal_ledger_complete"] is True and result["technical_goal_complete"] is True
    assert result["required_pending"] == []
    assert f.COUNT == 12000
    assert f.LATEST_SCOPE_TRIALS == ("freshdetector32", "bpm_windows", "truebar")
    for name in f.CHECKS:
        assert result["checks"][name]["executed"] is True
        assert result["checks"][name]["status"] == "verified"
    for name in f.LATEST_SCOPE_TRIALS:
        assert result["checks"]["scope"]["details"]["tasks"][name] == {"status": "verified", "executed": True}
    assert result["human_quality"] == "negative" and result["human_quality_approved"] is False


@pytest.mark.parametrize("mutation", [lambda p: p.update(downbeats_seconds=[0.0, 1.0, 4.0]), lambda p: p["raw_windows"][0].update(start=False), lambda p: p["provenance"].update(helper_sha256=f.BEATTHIS_V2_VERIFIER), lambda p: p["params"].update(insufficient_downbeats_fallback=True), lambda p: p.update(strategy="bpm_duration_grid_not_detected_beats"), lambda p: p.update(downbeat_verified=True)])
def test_truebar_plan_source_boolean_synthetic_and_phase_negative(truebar_fixture, mutation):
    root = Path(truebar_fixture["strict"]["evidence"]["path"]).parent
    plan = json.loads((root / "bpm-synthetic-0.windows.json").read_bytes())
    post = json.loads(Path(truebar_fixture["beatthis_validation"]["path"]).read_bytes())
    group = post["groups"]["first50"]
    raw = Path(group["detections_path"]).read_bytes().splitlines()[0]
    detection = json.loads(raw)
    detection.update(_line_number=1, _raw_line_sha256=hashlib.sha256(raw).hexdigest(), _run_sha256=group["actual_v1_run_receipt_sha256"], _output_sha256=group["detections_sha256"])
    mutation(plan)
    with pytest.raises(ValueError):
        f.verify_bar_plan(plan, detection, "strict")


def test_truebar_emptybeats_nonemptydownbeats_rejected_without_clipping(truebar_fixture):
    root = Path(truebar_fixture["strict"]["evidence"]["path"]).parent
    plan = json.loads((root / "bpm-synthetic-0.windows.json").read_bytes())
    detection = {"audio_sha256": "c" * 64, "beats": [], "downbeats": [0.4], "_line_number": 1, "_raw_line_sha256": plan["provenance"]["raw_detection_line_sha256"], "_run_sha256": plan["provenance"]["run_receipt_sha256"], "_output_sha256": plan["provenance"]["raw_detections_sha256"]}
    plan["downbeats_seconds"] = [0.4]
    plan["provenance"].update(raw_beats_seconds=[], raw_downbeats_seconds=[0.4])
    with pytest.raises(ValueError):
        f.verify_bar_plan(plan, detection, "strict")


@pytest.mark.parametrize("downbeats,beats,raw", [([], [], [{"start": 0.0, "end": 4.0, "kind": "unresolved_no_downbeats", "bar_index": None}]), ([0.5], [0.5, 1.0], [{"start": 0.0, "end": 0.5, "kind": "prefix_residual", "bar_index": None}, {"start": 0.5, "end": 4.0, "kind": "suffix_residual", "bar_index": None}])])
def test_truebar_insufficient_downbeats_are_finite_not_fabricated_bars(truebar_fixture, downbeats, beats, raw):
    root = Path(truebar_fixture["strict"]["evidence"]["path"]).parent
    plan = json.loads((root / "bpm-synthetic-0.windows.json").read_bytes())
    detection = {"audio_sha256": "c" * 64, "beats": beats, "downbeats": downbeats, "_line_number": 1, "_raw_line_sha256": plan["provenance"]["raw_detection_line_sha256"], "_run_sha256": plan["provenance"]["run_receipt_sha256"], "_output_sha256": plan["provenance"]["raw_detections_sha256"]}
    plan.update(downbeats_seconds=downbeats, raw_bars=[], raw_windows=raw, windows=[{"start": r["start"], "end": r["end"], "raw_window_indices": [i]} for i, r in enumerate(raw)], status="insufficient_downbeats")
    plan["provenance"].update(raw_beats_seconds=beats, raw_downbeats_seconds=downbeats)
    assert f.verify_bar_plan(plan, detection, "strict") == plan["windows"]
    assert plan["raw_bars"] == []


def test_shared_music_windows_vlm_missing_contract_blocks_latest_goal(bundle, fresh_caption_fixture, bpm_fixture, truebar_fixture, monkeypatch):
    monkeypatch.setattr(f, "LATEST_SCOPE_TRIALS", ("freshdetector32", "bpm_windows", "truebar"))
    monkeypatch.setattr(f, "SHARED_VLM_REQUIRED", ("shared_music_windows_vlm",))
    bundle["scope"]["tasks"].update(freshdetector32=fresh_caption_fixture, bpm_windows=bpm_fixture, truebar=truebar_fixture)
    result = f.verify(bundle)
    assert result["status"] == "pending" and result["technical_goal_complete"] is False
    assert "shared_music_windows_vlm:missing_actual_contract" in result["checks"]["scope"]["reason"]


def test_shared_vlm_latest_user_task_cannot_be_optional(bundle, monkeypatch):
    monkeypatch.setattr(f, "SHARED_VLM_REQUIRED", ("shared_music_windows_vlm",))
    bundle["scope"]["tasks"]["shared_music_windows_vlm"] = {"required": False, "status": "optional"}
    assert f.verify(bundle)["checks"]["scope"]["status"] == "rejected"


def test_bpm_captions_without_actual_video_pts_postreceipt_are_pending(bundle, bpm_fixture, monkeypatch):
    monkeypatch.setattr(f, "EXACT_MUSIC_PTS_REQUIRED", True)
    bundle["scope"]["tasks"]["bpm_windows"] = bpm_fixture
    result = f.verify(bundle)
    assert result["checks"]["scope"]["status"] == "pending"
    assert "independent exact actual-video-PTS" in result["checks"]["scope"]["reason"]
    assert result["technical_goal_complete"] is False


@pytest.fixture
def exact_pts_fixture(bpm_fixture, truebar_fixture, tmp_path, monkeypatch):
    tasks = {"bpm": bpm_fixture, "strict": truebar_fixture["strict"], "merge_short_edges": truebar_fixture["merge_short_edges"]}
    records, bundles = [], {}
    for policy, task in tasks.items():
        root = Path(task["evidence"]["path"]).parent
        summary = json.loads(Path(task["evidence"]["path"]).read_bytes())
        bundles[str(root.resolve())] = {"summary_sha256": task["evidence"]["sha256"], "clips": 50, "windows": 100}
        for record in summary["records"]:
            plan_path = root / record["window_file"]
            plan = json.loads(plan_path.read_bytes())
            records.append({"id": record["id"], "policy": policy, "actual_video_pts_sha256": plan["binding"]["actual_video_pts_sha256"], "frame_count": plan["binding"]["decoded_video_frames"], "report": {"path": str(root / record["report_file"]), "sha256": record["report_sha256"]}, "plan": {"path": str(plan_path), "sha256": record["window_sha256"]}, "exact_pts_digest_equal_original": True, "canonical_typed_plan_verified": True})
    verifier = dump(tmp_path / "exact-pts-verifier.py", {"synthetic_fixture": True})
    post = {"schema": "music_window_exact_pts_typed_verification_v1", "complete": True, "source_sha256": f.MUSIC_WINDOWS_SOURCE, "input_manifest_sha256": f.FIRST50_INPUT_SHA, "actual_video_pts_probed_not_inferred_from_nlf": True, "old_reports_and_captions_modified": False, "MS_or_model_inference_repeated": False, "source_video_bytes_rehashed": False, "audio_video_alignment_verified": False, "motion_accuracy_verified": False, "GPU_used": False, "video_probes": 50, "reports_verified": 150, "records": records, "bundle_counts": bundles, "verifier_path": verifier["path"], "verifier_sha256": verifier["sha256"]}
    post_ref = dump(tmp_path / "exact-pts-post.json", post)
    monkeypatch.setattr(f, "EXACT_MUSIC_PTS_SHA", post_ref["sha256"])
    monkeypatch.setattr(f, "EXACT_MUSIC_PTS_VERIFIER", verifier["sha256"])
    bpm_fixture["pts_validation"] = post_ref
    truebar_fixture["pts_validation"] = post_ref
    return bpm_fixture, truebar_fixture


def test_exact_video_pts_actual_schema_sourcebound_postreceipt_positive(exact_pts_fixture):
    bpm, bar = exact_pts_fixture
    assert f.check_music_exact_pts(bpm, ("bpm",))["exact_PTS_source_bound"] is True
    assert f.check_music_exact_pts(bar, ("strict", "merge_short_edges"))["reports_verified"] == 150


@pytest.mark.parametrize("mutation", [lambda p: p.update(actual_video_pts_probed_not_inferred_from_nlf=False), lambda p: p["records"][0].update(actual_video_pts_sha256="f" * 64), lambda p: p["records"][0].update(canonical_typed_plan_verified=1), lambda p: p.update(reports_verified=149)])
def test_exact_pts_cannot_be_replaced_by_tolerance_match_or_self_reseal(exact_pts_fixture, mutation):
    bpm, bar = exact_pts_fixture
    reseal(bpm["pts_validation"], mutation)
    with pytest.raises(ValueError):
        f.check_music_exact_pts(bpm, ("bpm",))


@pytest.fixture
def shared_vlm_fixture(bundle, bpm_fixture, exact_pts_fixture, tmp_path, monkeypatch):
    original_inputs = [json.loads(line) for line in Path(bpm_fixture["input"]["path"]).read_bytes().splitlines()][:3]
    fields = ("id", "video_path", "audio_path", "nlf_path", "source_group", "split")
    source_rows = {}
    for line, original in enumerate(original_inputs, 1):
        row = {key: original[key] for key in fields}
        row.update(input_line=line, input_row_sha256=hashlib.sha256(f.canonical_bytes(original)).hexdigest())
        source_rows[row["id"]] = row
    ids = tuple(source_rows)
    monkeypatch.setattr(f, "SHARED_MUSIC_IDS", ids)
    source = dump(tmp_path / "shared-source.py", {"synthetic_actual_schema": True})
    monkeypatch.setattr(f, "SHARED_MUSIC_SOURCE", source["sha256"])
    model = {"root": "/explicit/model", "sha256": "26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c", "files": {f"file-{i}": {"path": f"/explicit/model/file-{i}", "sha256": "a" * 64, "stat": {"size": i + 1}} for i in range(25)}}
    model["files"] = dict(sorted(model["files"].items()))
    model["sha256"] = f.model_identity_digest(model)
    monkeypatch.setattr(f, "SHARED_MODEL_IDENTITY_SHA", model["sha256"])
    model_content_sha = hashlib.sha256(f.canonical_bytes(f.model_inventory(model))).hexdigest()
    monkeypatch.setattr(f, "SHARED_MODEL_CONTENT_SHA", model_content_sha)
    references, records = [], []
    for tag in ("bpm", "bar_strict", "bar_merge_short_edges"):
        plans = []
        for i, sid in enumerate(ids):
            # 5 + 11 + 6 = 22 windows in the real protocol; synthetic interval lengths here.
            n = (1 if i == 0 else 2) if tag == "bpm" else ((3 if i == 0 else 4) if tag == "bar_strict" else 2)
            windows = [{"start": float(j), "end": float(j + 1), "raw_window_indices": [j]} for j in range(n)]
            binding = {"id": sid, "video_sha256": "b" * 64, "audio_sha256": "c" * 64, "input_sha256": "a" * 64, "actual_video_pts_sha256": hashlib.sha256(f.canonical_bytes([i / 16.0 for i in range(64)])).hexdigest(), "decoded_video_frames": 64, "video_fps": 16.0, "media_start": 0.0, "media_end": 4.0}
            plans.append({"id": sid, "schema": "motionscript_music_windows_v1", "binding": binding, "manifest_record": source_rows[sid], "raw_windows": [{"start": w["start"], "end": w["end"]} for w in windows], "windows": windows})
        path = tmp_path / (tag + "-shared/windows.jsonl")
        checksum = jsonl(path, plans)
        reference = {"path": str(path), "sha256": checksum}
        references.append(reference)
        for plan in plans:
            plan_sha = hashlib.sha256((json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
            row = source_rows[plan["id"]]
            source_files = {key: {"path": row[key], "sha256": plan["binding"][digest_key], "stat": {"size": 100}} for key, digest_key in (("video_path", "video_sha256"), ("audio_path", "audio_sha256"), ("nlf_path", "input_sha256"))}
            source_sha = hashlib.sha256(f.canonical_bytes({"row": row, "files": {key: {"path": value["path"], "sha256": value["sha256"]} for key, value in source_files.items()}})).hexdigest()
            identity = {"files": source_files, "sha256": source_sha, "audio_and_nlf_are_provenance_only_not_model_inputs": True}
            records.append({"id": plan["id"], "tag": tag, "shared_plan": plan, "raw_windows": plan["raw_windows"], "shared_plan_sha256": plan_sha, "source_identity": identity, "plan_status": "ready", "source_bound_actual_pts_verified": True, "BPM_or_downbeat_values_supplied_to_model": False, "MS_action_captions_supplied_to_model": False, "audio_video_alignment_verified": False, "motions_on_beats_verified": False, "semantic_approval": False, "human_review_completed": False, "formal_training": False, "windows": []})
    monkeypatch.setattr(f, "SHARED_MUSIC_WINDOW_SHAS", tuple(ref["sha256"] for ref in references))
    values = {"tp": 2, "context": 16384, "max_frames": 16, "max_pixels": 200704, "batch_size": 1, "max_num_seqs": 1, "max_tokens": 1200, "new_source_sha256": source["sha256"], "music_window_helper_sha256": f.MUSIC_WINDOWS_SOURCE, "independent_pts_verification_sha256": f.EXACT_MUSIC_PTS_SHA, "exact_pts_helper_sha256": f.EXACT_MUSIC_PTS_VERIFIER, "shared_jsonl_sha256": [r["sha256"] for r in references], "model_identity_sha256": model["sha256"]}
    params_sha = hashlib.sha256((json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    for record in records:
        for i, window in enumerate(record["shared_plan"]["windows"]):
            bundle_path = tmp_path / "shared-output" / record["tag"] / record["id"] / str(i)
            visual = visualwin(int(window["start"]))
            frame_refs = visual["source_frames"]
            binding = {"id": record["id"], "source_sha256": record["source_identity"]["sha256"], "model_sha256": model["sha256"], "params_sha256": params_sha, "window": window, "shared_plan_sha256": record["shared_plan_sha256"], "shared_jsonl_sha256": references[("bpm", "bar_strict", "bar_merge_short_edges").index(record["tag"])]["sha256"], "decoding_protocol": "motionscript_visual_shared_music_windows_v1"}
            request = dump(bundle_path / "request.json", {"fixture_only": True, "model": model["root"], "messages": []})
            manifest_value = {"frames": frame_refs, "binding": binding, "input": source_rows[record["id"]], "source_identity": record["source_identity"], "window": window, "video_sha256": "b" * 64, "request_sha256": request["sha256"]}
            manifest = dump(bundle_path / "manifest.json", manifest_value)
            raw_value = {"motion_caption": visual["caption"], "events": [{key: event[key] for key in ("id", "description", "frame_indices")} for event in visual["events"]], "uncertainties": [], "abstain": False}
            raw = json.dumps(raw_value, sort_keys=True).encode()
            (bundle_path / "response.raw.txt").write_bytes(raw)
            raw_sha = hashlib.sha256(raw).hexdigest()
            visual["generation"]["raw_response_sha256"] = raw_sha
            generation = dump(bundle_path / "generation.json", visual["generation"])
            _, validate = f.original_visual_functions()
            candidate_value = validate(raw_value, manifest_value)
            candidate_value.update(binding=binding, raw_response_sha256=raw_sha, video_sha256="b" * 64, request_sha256=request["sha256"])
            candidate = dump(bundle_path / "candidate.json", candidate_value)
            files = {"manifest.json": manifest["sha256"], "request.json": request["sha256"], "generation.json": generation["sha256"], "candidate.json": candidate["sha256"], "response.raw.txt": raw_sha}
            receipt = {"status": "candidate", "binding": binding, "files_sha256": files, "error": None, "semantic_approval": False, "human_review_completed": False, "training_entry_updated": False}
            receipt_ref = dump(bundle_path / "receipt.json", receipt)
            record["windows"].append({"window_index": i, "window": window, "status": "candidate", "bundle": str(bundle_path), "receipt_sha256": receipt_ref["sha256"], "caption": visual["caption"], "events": visual["events"], "model_call_performed": True})
    schema = "motionscript_visual_shared_music_windows_v1"
    run = {"schema": schema, "own_source": {"sha256": source["sha256"]}, "original_code": {"files": {"ms_visual_batch.py": {"sha256": f.VLM_PRODUCER}, "ms_visual_observe.py": {"sha256": f.ORIGINAL_VISUAL_SCHEMA_SOURCE}}}, "helper_source": {"sha256": f.MUSIC_WINDOWS_SOURCE}, "exact_pts_helper_source": {"sha256": f.EXACT_MUSIC_PTS_VERIFIER}, "manifest_path": bpm_fixture["input"]["path"], "manifest_sha256": f.FIRST50_INPUT_SHA, "selected_ids": list(ids), "independent_pts_verification": {"path": exact_pts_fixture[0]["pts_validation"]["path"], "sha256": f.EXACT_MUSIC_PTS_SHA, "actual_video_pts_probed_not_inferred_from_nlf": True}, "params": {"values": values, "sha256": params_sha}, "model_identity": model, "strategy_references": references}
    loaded = {"schema": schema, "one_model_load_for_all_ready_strategies": True, "current_model_and_code_sha_verified_before_load": True, "actual_decoding": "local_vllm_LLM.chat_free_json", "params_sha256": params_sha, "model_identity_sha256": model["sha256"]}
    summary = {"schema": schema, "stage": "run", "all_ready_windows_terminal": True, "source_count": 3, "strategy_source_count": 9, "window_count": 22, "pending_insufficient_downbeat_plan_count": 0, "all_plans_executed": True, "VLM_shared_manifest_actual_execution": True, "not_fixed_window_baseline_reused": True, "audio_video_alignment_verified": False, "motions_on_beats_verified": False, "semantic_approval": False, "human_review_completed": False, "formal_training": False, "full_pool_updated": False, "records": records, "window_status_counts": {"candidate": 22, "abstain": 0, "failed": 0, "prepared": 0}, "model_executed_window_count": 22, "insufficient_frame_abstention_count": 0, "usable_complete_strategy_source_count": 9, "all_ready_windows_without_failures": True}
    return {"required": True, "status": "verified", "actual_executed": True, "evidence": dump(tmp_path / "shared-output/summary.json", summary), "run_receipt": dump(tmp_path / "shared-output/run.receipt.json", run), "load_receipt": dump(tmp_path / "shared-output/load.receipt.json", loaded), "source": source}


def test_actual_shared_music_consumer_schema_positive(shared_vlm_fixture):
    result = f.check_shared_music_windows_vlm(shared_vlm_fixture)
    assert result["sources"] == 3 and result["windows"] == 22 and result["full_pool_updated"] is False


def test_current_all_six_gates_latest_four_and_exact_pts_positive(bundle, fresh_caption_fixture, bpm_fixture, truebar_fixture, exact_pts_fixture, shared_vlm_fixture, monkeypatch):
    # First enlarge the main ledger to the fixed production denominator using the prior scope fixture.
    test_prior_three_trial_scope_all_six_gates_production12000_positive(bundle, fresh_caption_fixture, bpm_fixture, truebar_fixture, monkeypatch)
    monkeypatch.setattr(f, "SHARED_VLM_REQUIRED", ("shared_music_windows_vlm",))
    monkeypatch.setattr(f, "EXACT_MUSIC_PTS_REQUIRED", True)
    bundle["scope"]["tasks"]["shared_music_windows_vlm"] = shared_vlm_fixture
    result = f.verify(bundle)
    assert f.COUNT == 12000 and f.LATEST_SCOPE_TRIALS == ("freshdetector32", "bpm_windows", "truebar")
    assert result["checks"]["scope"]["status"] == "verified", result["checks"]["scope"].get("reason")
    assert result["status"] == "verified" and result["technical_goal_complete"] is True
    assert result["required_pending"] == []
    assert all(result["checks"][name]["executed"] is True and result["checks"][name]["status"] == "verified" for name in f.CHECKS)
    assert result["checks"]["scope"]["details"]["tasks"]["shared_music_windows_vlm"] == {"status": "verified", "executed": True}
    assert result["human_quality_approved"] is False


@pytest.mark.parametrize("mutation", [lambda s: s.update(stage="prepare"), lambda s: s.update(VLM_shared_manifest_actual_execution=False), lambda s: s.update(strategy_source_count=8), lambda s: s["records"][0].update(MS_action_captions_supplied_to_model=True)])
def test_shared_music_prepare_partial_or_priors_cannot_pass(shared_vlm_fixture, mutation):
    ref = shared_vlm_fixture["evidence"]
    reseal(ref, mutation)
    with pytest.raises((ValueError, f.Pending)):
        f.check_shared_music_windows_vlm(shared_vlm_fixture)


def test_shared_music_eos_and_events_bound_not_just_model_call(shared_vlm_fixture):
    ref = shared_vlm_fixture["evidence"]
    summary = json.loads(Path(ref["path"]).read_bytes())
    window = summary["records"][0]["windows"][0]
    root = Path(window["bundle"])
    receipt = json.loads((root / "receipt.json").read_bytes())
    generation = json.loads((root / "generation.json").read_bytes())
    generation["finish_reason"] = "length"
    receipt["files_sha256"]["generation.json"] = dump(root / "generation.json", generation)["sha256"]
    window["receipt_sha256"] = dump(root / "receipt.json", receipt)["sha256"]
    ref.update(dump(Path(ref["path"]), summary))
    with pytest.raises(ValueError):
        f.check_shared_music_windows_vlm(shared_vlm_fixture)


def mutate_shared_window(task, mutate):
    summary_ref = task["evidence"]
    summary = json.loads(Path(summary_ref["path"]).read_bytes())
    record = summary["records"][0]
    window = record["windows"][0]
    root = Path(window["bundle"])
    receipt = json.loads((root / "receipt.json").read_bytes())
    mutate(summary, record, window, root, receipt)
    window["receipt_sha256"] = dump(root / "receipt.json", receipt)["sha256"]
    summary_ref.update(dump(Path(summary_ref["path"]), summary))


@pytest.mark.parametrize("case", ["raw_missing", "raw_corrupt", "forged_metadata", "candidate_text", "candidate_uncertainties", "candidate_binding", "record_sid", "source_reseal", "source_file_reseal", "manifest_source", "manifest_window", "request_corrupt"])
def test_shared_four_root_boundary_resealed_adversarial_rejected(shared_vlm_fixture, case):
    def mutate(summary, record, window, root, receipt):
        if case == "raw_missing":
            (root / "response.raw.txt").unlink()
        elif case == "raw_corrupt":
            (root / "response.raw.txt").write_bytes(b"corrupt original model response")
        elif case == "forged_metadata":
            raw = json.loads((root / "response.raw.txt").read_bytes())
            raw["motion_caption"] = "A different original model prediction"
            content = json.dumps(raw).encode()
            (root / "response.raw.txt").write_bytes(content)
            checksum = hashlib.sha256(content).hexdigest()
            generation = json.loads((root / "generation.json").read_bytes())
            generation["raw_response_sha256"] = checksum
            receipt["files_sha256"].update({"response.raw.txt": checksum, "generation.json": dump(root / "generation.json", generation)["sha256"]})
            candidate = json.loads((root / "candidate.json").read_bytes())
            candidate["raw_response_sha256"] = checksum
            receipt["files_sha256"]["candidate.json"] = dump(root / "candidate.json", candidate)["sha256"]
        elif case.startswith("candidate"):
            candidate = json.loads((root / "candidate.json").read_bytes())
            if case == "candidate_text":
                candidate["motion_caption"] = "Self-sealed text that was never the raw model prediction"
                window["caption"] = candidate["motion_caption"]
            elif case == "candidate_uncertainties":
                candidate["uncertainties"] = ["Added explanation absent from original raw response"]
            else:
                candidate["binding"]["id"] = "different-source-SID"
            receipt["files_sha256"]["candidate.json"] = dump(root / "candidate.json", candidate)["sha256"]
        elif case == "record_sid":
            receipt["binding"]["id"] = "different-source-SID"
        elif case.startswith("source"):
            source = record["source_identity"]
            if case == "source_file_reseal":
                source["files"]["video_path"]["sha256"] = "0" * 64
                row = json.loads((root / "manifest.json").read_bytes())["input"]
                source["sha256"] = hashlib.sha256(f.canonical_bytes({"row": row, "files": {key: {"path": value["path"], "sha256": value["sha256"]} for key, value in source["files"].items()}})).hexdigest()
            else:
                source["sha256"] = "0" * 64
            for other in record["windows"]:
                path = Path(other["bundle"]) / "receipt.json"
                value = json.loads(path.read_bytes())
                value["binding"]["source_sha256"] = source["sha256"]
                other["receipt_sha256"] = dump(path, value)["sha256"]
            receipt["binding"]["source_sha256"] = source["sha256"]
        elif case.startswith("manifest"):
            manifest = json.loads((root / "manifest.json").read_bytes())
            if case == "manifest_source":
                manifest["binding"]["source_sha256"] = "0" * 64
                manifest["source_identity"]["sha256"] = "0" * 64
            else:
                manifest["window"] = {"start": 999.0, "end": 1000.0}
            receipt["files_sha256"]["manifest.json"] = dump(root / "manifest.json", manifest)["sha256"]
        else:
            (root / "request.json").write_bytes(b"modified request bytes")
    mutate_shared_window(shared_vlm_fixture, mutate)
    with pytest.raises(ValueError):
        f.check_shared_music_windows_vlm(shared_vlm_fixture)


def test_original_fenced_raw_protocol_is_preserved_not_cleaned(shared_vlm_fixture):
    def mutate(summary, record, window, root, receipt):
        content = b"```json\n" + (root / "response.raw.txt").read_bytes() + b"\n```"
        (root / "response.raw.txt").write_bytes(content)
        checksum = hashlib.sha256(content).hexdigest()
        generation = json.loads((root / "generation.json").read_bytes())
        generation["raw_response_sha256"] = checksum
        candidate = json.loads((root / "candidate.json").read_bytes())
        candidate["raw_response_sha256"] = checksum
        receipt["files_sha256"].update({"response.raw.txt": checksum, "generation.json": dump(root / "generation.json", generation)["sha256"], "candidate.json": dump(root / "candidate.json", candidate)["sha256"]})
    mutate_shared_window(shared_vlm_fixture, mutate)
    assert f.check_shared_music_windows_vlm(shared_vlm_fixture)["windows"] == 22


@pytest.fixture
def short_support_fixture(shared_vlm_fixture, monkeypatch):
    task = shared_vlm_fixture
    summary = json.loads(Path(task["evidence"]["path"]).read_bytes())
    run = json.loads(Path(task["run_receipt"]["path"]).read_bytes())
    loaded = json.loads(Path(task["load_receipt"]["path"]).read_bytes())
    ref = run["strategy_references"][0]
    plans = [json.loads(line) for line in Path(ref["path"]).read_bytes().splitlines()]
    plans[0]["windows"][0]["end"] = 0.04
    plans[0]["raw_windows"][0]["end"] = 0.04
    ref["sha256"] = jsonl(Path(ref["path"]), plans)
    record = summary["records"][0]
    record["shared_plan"] = plans[0]
    record["raw_windows"] = plans[0]["raw_windows"]
    record["shared_plan_sha256"] = hashlib.sha256(f.canonical_bytes(plans[0])).hexdigest()
    refs = tuple(r["sha256"] for r in run["strategy_references"])
    monkeypatch.setattr(f, "SHARED_MUSIC_WINDOW_SHAS", refs)
    run["params"]["values"]["shared_jsonl_sha256"] = list(refs)
    params_sha = hashlib.sha256(f.canonical_bytes(run["params"]["values"])).hexdigest()
    run["params"]["sha256"] = loaded["params_sha256"] = params_sha
    for record in summary["records"]:
        for window in record["windows"]:
            root = Path(window["bundle"])
            receipt = json.loads((root / "receipt.json").read_bytes())
            binding = receipt["binding"]
            binding["params_sha256"] = params_sha
            binding["shared_jsonl_sha256"] = refs[("bpm", "bar_strict", "bar_merge_short_edges").index(record["tag"])]
            binding["shared_plan_sha256"] = record["shared_plan_sha256"]
            if record is summary["records"][0] and window is record["windows"][0]:
                window["window"] = plans[0]["windows"][0]
                binding["window"] = window["window"]
                support = {"schema": "motionscript_visual_shared_music_windows_v1", "id": record["id"], "binding": binding, "window": window["window"], "status": "abstain", "abstain": True, "motion_caption": "", "events": [], "actual_frame_indices": [0], "actual_frame_times": [0.0], "model_call_performed": False, "window_extended": False, "frame_evidence_filled": False, "semantic_approval": False, "human_review_completed": False}
                support_ref = dump(root / "insufficient_support.receipt.json", support)
                window.update(status="abstain", caption="", events=[], model_call_performed=False, support_reason="fewer_than_two_actual_decoded_video_frames_inside_unchanged_half_open_window", support_receipt_sha256=support_ref["sha256"])
                for name in ("receipt.json", "generation.json", "candidate.json", "response.raw.txt", "manifest.json", "request.json"):
                    (root / name).unlink()
            else:
                manifest = json.loads((root / "manifest.json").read_bytes())
                candidate = json.loads((root / "candidate.json").read_bytes())
                manifest["binding"] = candidate["binding"] = binding
                receipt["files_sha256"].update({"manifest.json": dump(root / "manifest.json", manifest)["sha256"], "candidate.json": dump(root / "candidate.json", candidate)["sha256"]})
                window["receipt_sha256"] = dump(root / "receipt.json", receipt)["sha256"]
    summary.update(window_status_counts={"candidate": 21, "abstain": 1, "failed": 0, "prepared": 0}, model_executed_window_count=21, insufficient_frame_abstention_count=1, usable_complete_strategy_source_count=8)
    task["run_receipt"].update(dump(Path(task["run_receipt"]["path"]), run))
    task["load_receipt"].update(dump(Path(task["load_receipt"]["path"]), loaded))
    task["evidence"].update(dump(Path(task["evidence"]["path"]), summary))
    return task


def test_short_support_exact_source_no_model_call_positive(short_support_fixture):
    result = f.check_shared_music_windows_vlm(short_support_fixture)
    assert result["windows"] == 22 and result["short_frame_abstentions"] == 1 and result["model_calls"] == 21


@pytest.mark.parametrize("case", ["caption_filled", "events_filled", "wrong_sid", "wrong_window", "wrong_frame", "wrong_time", "fake_gen"])
def test_short_support_source_bound_empty_and_actual_pts_negative(short_support_fixture, case):
    summary_ref = short_support_fixture["evidence"]
    summary = json.loads(Path(summary_ref["path"]).read_bytes())
    window = summary["records"][0]["windows"][0]
    root = Path(window["bundle"])
    support = json.loads((root / "insufficient_support.receipt.json").read_bytes())
    if case == "caption_filled":
        window["caption"] = "fabricated unsupported movement"
    elif case == "events_filled":
        window["events"] = [{"description": "fabricated unsupported event"}]
    elif case == "wrong_sid":
        support["id"] = "different-source"
    elif case == "wrong_window":
        support["binding"]["window"] = {"start": 99.0, "end": 100.0}
    elif case == "wrong_frame":
        support["actual_frame_indices"] = [9]
    elif case == "wrong_time":
        support["actual_frame_times"] = [2.0]
    else:
        dump(root / "generation.json", {"model_call_performed": True})
    window["support_receipt_sha256"] = dump(root / "insufficient_support.receipt.json", support)["sha256"]
    summary_ref.update(dump(Path(summary_ref["path"]), summary))
    with pytest.raises(ValueError):
        f.check_shared_music_windows_vlm(short_support_fixture)


def test_model_abstention_must_rebuild_from_original_raw_positive(shared_vlm_fixture):
    def mutate(summary, record, window, root, receipt):
        raw_value = {"motion_caption": "", "events": [], "uncertainties": ["Insufficient movement evidence"], "abstain": True}
        raw = json.dumps(raw_value).encode()
        (root / "response.raw.txt").write_bytes(raw)
        checksum = hashlib.sha256(raw).hexdigest()
        generation = json.loads((root / "generation.json").read_bytes())
        generation["raw_response_sha256"] = checksum
        manifest = json.loads((root / "manifest.json").read_bytes())
        _, validate = f.original_visual_functions()
        candidate = validate(raw_value, manifest)
        candidate.update(binding=receipt["binding"], raw_response_sha256=checksum, video_sha256=manifest["video_sha256"], request_sha256=manifest["request_sha256"])
        receipt.update(status="abstain")
        receipt["files_sha256"].update({"response.raw.txt": checksum, "generation.json": dump(root / "generation.json", generation)["sha256"], "candidate.json": dump(root / "candidate.json", candidate)["sha256"]})
        window.update(status="abstain", caption="", events=[])
        summary.update(window_status_counts={"candidate": 21, "abstain": 1, "failed": 0, "prepared": 0}, usable_complete_strategy_source_count=8)
    mutate_shared_window(shared_vlm_fixture, mutate)
    assert f.check_shared_music_windows_vlm(shared_vlm_fixture)["model_calls"] == 22


@pytest.mark.parametrize("has_raw", [True, False])
def test_failed_window_is_finite_not_usable_no_fabricated_raw(shared_vlm_fixture, has_raw):
    def mutate(summary, record, window, root, receipt):
        (root / "candidate.json").unlink()
        receipt["files_sha256"].pop("candidate.json")
        if not has_raw:
            (root / "response.raw.txt").unlink()
            receipt["files_sha256"].pop("response.raw.txt")
            (root / "generation.json").unlink()
            receipt["files_sha256"].pop("generation.json")
            window["model_call_performed"] = False
        else:
            content = b"non-JSON original model response retained as failed evidence"
            (root / "response.raw.txt").write_bytes(content)
            checksum = hashlib.sha256(content).hexdigest()
            generation = json.loads((root / "generation.json").read_bytes())
            generation["raw_response_sha256"] = checksum
            receipt["files_sha256"].update({"response.raw.txt": checksum, "generation.json": dump(root / "generation.json", generation)["sha256"]})
        receipt.update(status="failed", error="original output was not a valid JSON response")
        window.update(status="failed", caption="", events=[])
        summary.update(window_status_counts={"candidate": 21, "abstain": 0, "failed": 1, "prepared": 0}, model_executed_window_count=22 if has_raw else 21, usable_complete_strategy_source_count=8, all_ready_windows_without_failures=False)
    mutate_shared_window(shared_vlm_fixture, mutate)
    result = f.check_shared_music_windows_vlm(shared_vlm_fixture)
    assert result["window_status_counts"]["failed"] == 1 and result["usable_strategy_sources"] == 8


@pytest.mark.parametrize("mutation", [lambda m: m["files"]["file-0"].update(sha256="0" * 64), lambda m: m["files"]["file-0"].update(path="/different/model/file-0"), lambda m: m.update(root="/different/model"), lambda m: m["files"]["file-0"]["stat"].update(size=999999)])
def test_shared_fixed_model_inventory_resealed_metadata_rejected(shared_vlm_fixture, mutation):
    ref = shared_vlm_fixture["run_receipt"]
    reseal(ref, lambda run: mutation(run["model_identity"]))
    with pytest.raises(ValueError):
        f.check_shared_music_windows_vlm(shared_vlm_fixture)


def test_model_canonical_digest_ignores_host_stat_not_content(shared_vlm_fixture):
    ref = shared_vlm_fixture["run_receipt"]
    reseal(ref, lambda run: run["model_identity"]["files"]["file-0"]["stat"].update(inode=90000, mtime_ns=1, device=77))
    assert f.check_shared_music_windows_vlm(shared_vlm_fixture)["windows"] == 22


def test_model_identity_tree_preserves_producer_order_and_paths():
    model = {"root": "/fixture/model", "files": {"z-last": {"path": "/fixture/model/z-last", "sha256": "a" * 64}, "a-first": {"path": "/fixture/model/a-first", "sha256": "b" * 64}}}
    expected = hashlib.sha256(f.canonical_bytes({"root": model["root"], "files": {name: {"path": row["path"], "sha256": row["sha256"]} for name, row in model["files"].items()}})).hexdigest()
    assert f.model_identity_digest(model) == expected
    sorted_digest = hashlib.sha256((json.dumps({"root": model["root"], "files": {name: {"path": row["path"], "sha256": row["sha256"]} for name, row in model["files"].items()}}, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    assert expected != sorted_digest


@pytest.mark.parametrize("required_path", ["scripts_omnidance/exp/ms/ms_visual_music_windows.py", "scripts_omnidance/exp/ms/ms_music_windows.py", "scripts_omnidance/exp/ms/ms_truebar_caption.py", "scripts_omnidance/exp/ms/ms_nlf_rgb32.py", "scripts_omnidance/exp/ms/verify_music_window_pts_and_types.py", "tests/test_ms_visual_music_windows.py", "tests/test_verify_music_window_pts_and_types.py", "notes/30_motionscript_chunk_annotation_plan_20261024.md", "notes/31_bar_chunking_and_rgb32_execution_plan_20261024.md", "results/music_window_exact_pts_typed_verification_20261024_v1.json"])
def test_git_static_latest_required_files_cannot_be_dropped_by_caller(bundle, required_path):
    assert required_path in f.PACKAGE_MINIMUM
    section = bundle["gitpackage"]
    section["files"] = [item for item in section["files"] if item["path"] != required_path]
    section["required_files"].remove(required_path)
    with pytest.raises(ValueError):
        f.check_gitpackage(section, bundle, False)


def test_cli_exclusive_output(tmp_path):
    contract = tmp_path / "contract.json"
    dump(contract, {"schema": f.CONTRACT_SCHEMA, "feature_enabled": True, "required_checks": list(f.CHECKS)})
    out = tmp_path / "out.json"
    assert f.main(["--contract", str(contract), "--out", str(out), "--offline"]) == 2
    before = out.read_bytes()
    with pytest.raises(ValueError):
        f.main(["--contract", str(contract), "--out", str(out), "--offline"])
    assert out.read_bytes() == before
