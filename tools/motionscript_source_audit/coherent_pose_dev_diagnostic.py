"""Whole-pose medoid diagnostic from fixed developer augmentation caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementation/scripts_omnidance/exp/ms"))
from ms_optimize import atomic_write, json_bytes, save_npz, sha256


def coherent_medoid(poses, k, frame_size):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 4 or poses.shape[-1] != 3 or poses.shape[1] != 3:
        raise ValueError("Three aligned camera-coordinate augmentation poses required")
    width, height = frame_size
    finite = np.isfinite(poses).all(axis=-1) & (poses[..., 2] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pixels = poses[..., :2] / poses[..., 2:3] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    inside = finite & np.isfinite(pixels).all(axis=-1) & (pixels >= 0).all(axis=-1) & (pixels < [width, height]).all(axis=-1)
    common = inside.all(axis=1)
    output = np.full((poses.shape[0], poses.shape[2], 3), np.nan)
    indices = np.full(poses.shape[0], -1, dtype=int)
    costs = np.full((poses.shape[0], 3), np.nan)
    translations = np.full((poses.shape[0], 3), np.nan)
    for row in range(poses.shape[0]):
        eligible = common[row]
        if int(eligible.sum()) < 3:
            continue
        selected = poses[row][:, eligible]
        centers = selected.mean(axis=1, keepdims=True)
        shapes = selected - centers
        pairwise = np.sqrt(np.mean(np.sum((shapes[:, None] - shapes[None]) ** 2, axis=-1), axis=-1))
        costs[row] = pairwise.sum(axis=1)
        best = min(range(3), key=lambda index: (costs[row, index], abs(index - 1), index))
        offset = centers[:, 0].mean(axis=0) - centers[best, 0]
        output[row] = poses[row, best] + offset
        indices[row] = best
        translations[row] = offset
    return output, indices, costs, common, translations


def main():
    cache_root = ROOT / "augmentation-stage-development-v1"
    registry_path = cache_root / "finished.json"
    registry = json.loads(registry_path.read_text())
    if not registry["complete"] or registry["expected"] != 3:
        raise ValueError("Complete fixed-development augmentation-stage registry required")
    out = ROOT / "coherent-pose-development-v1"
    if out.exists():
        raise ValueError("Preserve prior medoid diagnostic")
    out.mkdir()
    development = {row["id"]: row for row in map(json.loads, (ROOT / "panel-200-v1/development.jsonl").read_text().splitlines())}
    summaries = []
    for case in registry["cases"]:
        sid = case["id"]
        if sid not in development:
            raise ValueError("A source outside frozen development was supplied")
        report_path = cache_root / (sid + ".json")
        if sha256(report_path) != case["report_sha256"]:
            raise ValueError("Augmentation report changed")
        source = json.loads(report_path.read_text())
        cached = cache_root / (sid + ".stages.npz")
        if sha256(cached) != source["cache_sha256"]:
            raise ValueError("Augmentation-stage coordinate cache changed")
        with np.load(cached, allow_pickle=False) as data:
            arrays = dict(data)
        poses = arrays["aligned_camera_points_m"]
        k = np.asarray(source["actual_v020_intrinsic_matrix"])
        probe = development[sid]["media_probe"]
        candidate, chosen, costs, common, translations = coherent_medoid(poses, k, [probe["width"], probe["height"]])
        names = arrays["names"].tolist()
        comparisons = []
        for row, frame in enumerate(arrays["frame_indices"].tolist()):
            for side in ("left", "right"):
                a, b = names.index(side + "_elbow"), names.index(side + "_wrist")
                existing = float(np.linalg.norm(arrays["official_fused_camera_points_m"][row, a] - arrays["official_fused_camera_points_m"][row, b]) * 1000)
                selected = None if chosen[row] < 0 else float(np.linalg.norm(poses[row, chosen[row], a] - poses[row, chosen[row], b]) * 1000)
                alternative = None if chosen[row] < 0 else float(np.linalg.norm(candidate[row, a] - candidate[row, b]) * 1000)
                if selected is not None and not np.isclose(selected, alternative, atol=1e-7):
                    raise ValueError("Whole-pose rigid translation changed a bone length")
                comparisons.append({"source_frame": frame, "side": side, "official_pointwise_fused_length_mm": existing,
                    "chosen_branch": int(chosen[row]), "selected_aligned_branch_length_mm": selected,
                    "whole_pose_candidate_length_mm": alternative, "selected_branch_length_preserved": selected is not None,
                    "source_geometry_or_accuracy_certified": False})
        save_npz(out / (sid + ".candidates.npz"), {"frame_indices": arrays["frame_indices"], "source_seconds": arrays["source_seconds"],
            "pts": arrays["pts"], "time_base": arrays["time_base"], "names": arrays["names"],
            "candidate_camera_points_m": candidate.astype(np.float32), "chosen_augmentation_index": chosen,
            "common_model_screen_support": common, "medoid_cost_m": costs, "shared_translation_m": translations})
        report = {"schema": "nlf-coherent-whole-pose-developer-diagnostic-v1", "id": sid, "source_role": "development",
            "source_augmentation_report_sha256": case["report_sha256"], "source_cache_sha256": source["cache_sha256"],
            "source_video_sha256": development[sid]["video_sha256"], "frame_indices": arrays["frame_indices"].tolist(),
            "selection_policy": "Unweighted sum of pairwise centered3D RMS over query points inside every branch projection; ties prefer central branch, then index.",
            "translation_policy": "Translate the selected entire pose to the common screened-point centroid; no per-joint fusion.",
            "insufficient_screen_support_is_unresolved": True, "screen_support_is_not_RGB_visibility": True,
            "chosen_augmentations": chosen.tolist(), "screen_supported_point_counts": common.sum(axis=1).tolist(),
            "bone_comparisons": comparisons, "bone_length_preservation_is_not_accuracy": True,
            "no_Smpl_template_or_new_joint_generation": True, "frozen_pipeline_default_changed": False,
            "raw_or_official_fusion_caches_modified": False, "event_thresholds_or_acceptance_sources_changed": False,
            "sparse_sample_frames_not_treated_as_continuous_motion": True, "independent_human_review_completed": False,
            "quality_gate_passed": False, "candidate_cache_sha256": sha256(out / (sid + ".candidates.npz")),
            "implementation_sha256": sha256(Path(__file__))}
        atomic_write(out / (sid + ".json"), json_bytes(report))
        summaries.append({"id": sid, "report_sha256": sha256(out / (sid + ".json"))})
        print(json.dumps({"id": sid, "selected_branches": chosen.tolist(), "screen_support_counts": common.sum(axis=1).tolist(),
                          "source_accuracy_certified": False}), flush=True)
    atomic_write(out / "finished.json", json_bytes({"schema": "nlf-whole-pose-medoid-dev-terminal-v1", "expected": 3,
        "cases": summaries, "complete": len(summaries) == 3, "source_registry_sha256": sha256(registry_path),
        "quality_gate_passed": False, "frozen_model_or_pipeline_defaults_changed": False,
        "implementation_sha256": sha256(Path(__file__))}))


if __name__ == "__main__":
    main()
