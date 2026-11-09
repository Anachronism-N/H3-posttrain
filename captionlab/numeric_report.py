"""Validate saved numeric evidence, without authenticating upstream operations.

Source tensors and comparison metrics are checked on CPU. No model/candidate
forward is rerun; manager and restore receipts remain operator declarations.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from captionlab.io import loads
from captionlab.numeric_common import digest_tensor, save_new


_EXACT_MODES = ("matched_BF16", "matched_FP32")
_PRECISION_MODES = ("bf16_autocast", "fp32_reference")
_DIFFERENCES = ("max_logit", "max_selected_logprob", "all_target_logprob_max",
                "all_target_logprob_mean", "NLL_delta")
_RESOURCE_FOLDERS = ("cache_manager", "zero_manager", "matched_portable_manager",
                     "frozen_portable_manager", "route_calibration_manager",
                     "norm_dtype_manager", "norm_embedding_dtype_manager")


def _number(value, name, nonnegative=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (nonnegative and value < 0):
        raise ValueError(name + " must be finite numeric data, not a boolean")
    return value


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(name + " must be lowercase SHA256")


def compact(rows):
    """Summarize nonempty position observations; never assume zero controls."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Numeric position observations must be nonempty")
    for row in rows:
        if not isinstance(row, dict) or type(row.get("top1_equal")) is not bool:
            raise ValueError("Position top1_equal must be an explicit boolean")
        for key in ("max_logit_abs_difference", "max_centered_logit_abs_difference", "total_variation"):
            _number(row.get(key), key, nonnegative=True)
        _number(row.get("target_logprob_difference"), "target_logprob_difference")
        if "reference_top1" in row and "candidate_top1" in row:
            if (any(type(row[k]) is not int or row[k] < 0 for k in ("reference_top1", "candidate_top1"))
                    or row["top1_equal"] != (row["reference_top1"] == row["candidate_top1"])):
                raise ValueError("Position top1 identity contradicts top1_equal")
        if row["max_logit_abs_difference"] == 0 and (
                not row["top1_equal"] or row["max_centered_logit_abs_difference"] != 0
                or row["total_variation"] != 0 or row["target_logprob_difference"] != 0):
            raise ValueError("Identical logits contradict probability/top1 differences")
    return dict(positions=len(rows), top1_changes=sum(not r["top1_equal"] for r in rows),
        max_absolute_logit_difference=max(r["max_logit_abs_difference"] for r in rows),
        max_centered_logit_difference=max(r["max_centered_logit_abs_difference"] for r in rows),
        max_target_logprob_difference=max(abs(r["target_logprob_difference"]) for r in rows),
        max_total_variation=max(r["total_variation"] for r in rows))


def _index(rows, fields, expected, name):
    if not isinstance(rows, list):
        raise ValueError(name + " must be a list")
    index = {}
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k] for k in fields):
            raise ValueError(name + " has an invalid case/mode identity")
        key = tuple(row[k] for k in fields)
        if key in index:
            raise ValueError(name + " contains duplicate case/mode evidence")
        index[key] = row
    if set(index) != expected:
        raise ValueError(name + " does not exactly cover the original source case/mode inventory")
    return index


def _source_signature(path, raw):
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Original numeric snapshot must be an object")
    tokens, labels = value.get("input_ids"), value.get("labels")
    if (not isinstance(tokens, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape != labels.shape
            or tokens.dtype not in (torch.int32, torch.int64) or labels.dtype not in (torch.int32, torch.int64)
            or not bool(tokens.ge(0).all())):
        raise ValueError("Original snapshot has invalid aligned input/label tensors")
    prefix = value.get("prefix_tokens")
    if (type(prefix) is not int or not 0 < prefix < tokens.shape[1]
            or not bool(labels[:, :prefix].eq(-100).all())
            or not torch.equal(labels[:, prefix:], tokens[:, prefix:])):
        raise ValueError("Original snapshot lacks contiguous nonempty supervised target support")
    probabilities = value.get("target_logprobs")
    if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 1
            or not probabilities.is_floating_point() or probabilities.numel() != tokens.shape[1] - prefix
            or not bool(torch.isfinite(probabilities).all()) or not bool(probabilities.le(0).all())):
        raise ValueError("Original target log-probabilities need finite nonempty aligned support")
    positions, logits = value.get("positions"), value.get("selected_logits")
    if (not isinstance(positions, list) or not positions
            or any(type(p) is not int or not prefix - 1 <= p < tokens.shape[1] - 1 for p in positions)
            or positions != sorted(set(positions)) or not isinstance(logits, torch.Tensor)
            or logits.ndim != 2 or logits.shape[0] != len(positions) or logits.shape[1] < 2
            or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())):
        raise ValueError("Original selected-position logits require valid unique nonempty support")
    for name, tensor in (("input_ids_sha256", tokens), ("labels_sha256", labels)):
        if value.get(name) != digest_tensor(tensor):
            raise ValueError("Original snapshot tensor digest differs: " + name)
    for name in ("input_feature_sha256", "feature_mask_sha256"):
        _sha(value.get(name), name)
    suffix = "-original_fp32_features.pt"
    return dict(case_id=path.name[:-len(suffix)], target_tokens=probabilities.numel(),
                input_ids_sha256=value["input_ids_sha256"], labels_sha256=value["labels_sha256"],
                feature_sha256=value["input_feature_sha256"], feature_mask_sha256=value["feature_mask_sha256"],
                selected_positions=positions)


def _comparison_pass(row, signature, require_exact):
    if (type(row.get("positions")) is not int or row["positions"] != len(signature["selected_positions"])
            or type(row.get("top1_changes")) is not int or not 0 <= row["top1_changes"] <= row["positions"]):
        raise ValueError("Comparison position support differs from its original snapshot")
    for name in ("exact_selected", "exact_all_target_logprob"):
        if type(row.get(name)) is not bool:
            raise ValueError("Exact comparison flags must be booleans")
    for name in _DIFFERENCES:
        _number(row.get(name), name, nonnegative=name != "NLL_delta")
    selected_zero = row["top1_changes"] == 0 and row["max_logit"] == row["max_selected_logprob"] == 0
    target_zero = row["all_target_logprob_max"] == row["all_target_logprob_mean"] == row["NLL_delta"] == 0
    if row["exact_selected"] != selected_zero or row["exact_all_target_logprob"] != target_zero:
        raise ValueError("Exact comparison flags contradict measured differences")
    if require_exact and not (row["exact_selected"] is True and row["exact_all_target_logprob"] is True):
        raise ValueError("Matched-precision ZeRO/portable equality not established")
    return selected_zero and target_zero


def _repeat_status(rows, contrast):
    observations = [row["contrasts"].get(contrast) for row in rows]
    if any(item is None for item in observations):
        return None, "unverified_missing_repeat_control"
    equal = all(item["top1_changes"] == 0 and all(item[k] == 0 for k in (
        "max_absolute_logit_difference", "max_centered_logit_difference",
        "max_target_logprob_difference", "max_total_variation")) for item in observations)
    return equal, "validated_position_metrics_equal" if equal else "validated_position_metrics_differ"


def assemble_report(root):
    """Read this experiment's complete evidence layout; run no model."""
    root = Path(root).resolve()
    evidence, references = root / "evidence", []

    def read_bytes(path):
        raw = path.read_bytes()
        references.append(dict(path=path.relative_to(root).as_posix(), sha256=hashlib.sha256(raw).hexdigest()))
        return raw

    def load(name):
        value = loads(read_bytes(evidence / name).decode("utf-8"))
        json.dumps(value, allow_nan=False)  # Includes overflow such as 1e999.
        return value

    snapshots = sorted((evidence / "zero_original").glob("*-original_fp32_features.pt"))
    if len(snapshots) != 3:
        raise ValueError("This report requires exactly three original source snapshots")
    signatures = [_source_signature(path, read_bytes(path)) for path in snapshots]
    by_case = {s["case_id"]: s for s in signatures}
    if len(by_case) != len(signatures) or any(not case for case in by_case):
        raise ValueError("Original source case identities must be nonempty and unique")
    cases = set(by_case)
    zp = load("zero_portable_same_inputs_comparison.json")
    if not isinstance(zp, list) or any(not isinstance(row, dict) for row in zp):
        raise ValueError("ZeRO/portable comparisons must be a list of objects")
    modes = set(_EXACT_MODES) | {"current_portable_BF16"}
    if any(not isinstance(row.get("comparison"), str) or row["comparison"] not in modes for row in zp):
        raise ValueError("Unknown comparison mode")
    exact = _index([row for row in zp if row["comparison"] in _EXACT_MODES], ("case", "comparison"),
                   {(case, mode) for case in cases for mode in _EXACT_MODES}, "Matched comparisons")
    checked = {key: _comparison_pass(row, by_case[key[0]], True) for key, row in exact.items()}
    precision_passes = {mode: all(checked[(case, mode)] for case in cases) for mode in _EXACT_MODES}
    current = [row for row in zp if row["comparison"] == "current_portable_BF16"]
    current_equal = None
    if current:
        current = _index(current, ("case",), {(case,) for case in cases}, "Current portable comparisons")
        current_equal = all([_comparison_pass(row, by_case[case], False) for (case,), row in current.items()])

    cache = load("cache_calibration/summary.json")
    matched = load("matched_portable/summary.json")
    route = load("route_calibration/summary.json")
    cache_rows = _index(cache["cases"], ("case_id",), {(case,) for case in cases}, "Cache controls")
    matched_rows = _index(matched["records"], ("case_id", "mode"),
                          {(case, mode) for case in cases for mode in _PRECISION_MODES}, "Precision controls")
    _index(route["records"], ("case_id",), {(case,) for case in cases}, "Route diagnostics")
    calibrations, higher = [], []
    for (case,), row in cache_rows.items():
        contrasts = {name: compact(values) for name, values in row["contrasts"].items()}
        if any(value["positions"] != len(by_case[case]["selected_positions"]) for value in contrasts.values()):
            raise ValueError("Cache control support differs from its source snapshot")
        calibrations.append(dict(case_id=case, contrasts=contrasts))
    for (case, mode), row in matched_rows.items():
        contrasts = {key: compact(row[key]) for key in ("repeat_control", "full_shortened", "full_cached", "shortened_cached") if key in row}
        if any(value["positions"] != len(by_case[case]["selected_positions"]) for value in contrasts.values()):
            raise ValueError("Precision control support differs from its source snapshot")
        higher.append(dict(case_id=case, mode=mode, contrasts=contrasts))
    full_equal, full_status = _repeat_status(calibrations, "repeat_full")
    cache_equal, cache_status = _repeat_status(calibrations, "repeat_cache")
    matched_repeat_equal, matched_repeat_status = _repeat_status(higher, "repeat_control")

    resources = {}
    for folder in _RESOURCE_FOLDERS:
        value = load(folder + "/resources_restored.json")
        if value.get("holder_compute_verified") is not True or value.get("unknown_jobs_not_signaled") is not True:
            raise ValueError("Missing explicit resource restoration declarations")
        resources[folder] = value
    zero_summary = load("zero_original/summary.json")
    restore = load("zero_original/restore_receipt.json")
    cursor = zero_summary.get("original_cursor")
    update = cursor.get("update") if isinstance(cursor, dict) else None
    if type(update) is not int or update < 0:
        raise ValueError("Original summary requires its declared checkpoint update")
    source_world = zero_summary.get("zero3_world_size")
    if type(source_world) is not int or source_world != 8:
        raise ValueError("Original summary must declare this experiment's world size 8")
    count = sum(s["target_tokens"] for s in signatures)
    positions = sum(len(s["selected_positions"]) for s in signatures)
    result = dict(schema="captionlab-calibrated-numeric-equivalence-v1", original_checkpoint_update=update,
        case_count=len(signatures), identity=dict(
            original_learned_tensor_FP32_exact_restore=None, ZeRO_world_size=source_world,
            original_optimizer_unchanged=None, original_checkpoint_not_written=None, independent_copy_used=None,
            same_input_ids_masks_and_frozen_features_verified=None, source_signatures=signatures,
            verification_status="source_snapshots_validated; upstream_restore_and_operation_claims_not_independently_verified",
            operator_declarations=dict(zero_summary=zero_summary, restore_receipt=restore)),
        ZeRO_portable=dict(matched_parameter_storage_FP32_passed=all(precision_passes.values()),
            matched_BF16_compute_selected_logits_bitwise_equal=precision_passes["matched_BF16"],
            matched_FP32_compute_selected_logits_bitwise_equal=precision_passes["matched_FP32"],
            matched_all_target_logprobs_bitwise_equal=all(checked.values()), target_tokens_per_precision=count,
            selected_positions_per_precision=positions, observed_comparisons=zp,
            comparison_evidence_status="validated_reported_zero_differences_and_original_support; candidate_tensors_not_recomputed",
            candidate_tensor_equality_independently_recomputed=False,
            current_single_GPU_BF16_storage_bitwise_equivalent=current_equal,
            normalization_only_diagnostic=load("FP32_norm_fix_comparison.json"),
            normalization_embedding_diagnostic=load("hybrid_norm_embedding_comparison.json"),
            normalizations_only_single_GPU_fix_passed=None, norm_and_embedding_single_GPU_fix_passed=None,
            single_gpu_interventions_status="original_diagnostics_retained; no_automatic_pass_claim",
            scope="original_standard_checkpoint_source_cases; selected_full_vocab_logits_and_all_supervised_target_probabilities; not_universal_output_equivalence"),
        cache_calibration=dict(BF16_controls=calibrations, matched_storage_BF16_and_FP32=higher,
            identical_forward_repeats_bitwise_equal=full_equal, identical_cache_repeats_bitwise_equal=cache_equal,
            matched_precision_forward_repeats_bitwise_equal=matched_repeat_equal,
            repeat_verification=dict(full=full_status, cache=cache_status, matched_precision=matched_repeat_status),
            blanket_BF16_cache_full_bitwise_equivalence=None, FP32_route_switches=route["records"],
            conclusion="Observed position controls are reported separately; no automatic attribution of residuals or global cache-equivalence guarantee.",
            tolerance_policy="no_global_relaxed_tolerance_or_claim_of_cache_bug_eliminated",
            frozen_reference_route_intervention_is_not_deployed=None,
            route_intervention_status="upstream_operator_declaration_only", route_source_declaration=route),
        CPU_feature_identity=load("portable_frozen_inputs/summary.json")["cases"],
        deployment_recommendation="Validate runtime storage, restored state, input features and resources independently before adopting the observed matched-precision policy.",
        no_retraining_required_for_state_restore=None, new_training_updates=None, new_generation_calls=None,
        report_assembly=dict(model_calls=0, training_updates=0), no_new_action_or_genre_quality_claim=True,
        resources_restored=resources, resource_restoration_status="operator_declared_receipts_not_independently_authenticated",
        resources_restored_independently_verified=False, implementation_tests=None)
    for ref in references:
        if hashlib.sha256((root / ref["path"]).read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError("Numeric evidence changed while assembling report: " + ref["path"])
    result["source_evidence"] = references
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.root / "numeric_equivalence_summary.json"
    if destination.exists():
        raise FileExistsError("Preserve previous numeric summary")
    result = assemble_report(args.root)
    save_new(destination, result)
    print(json.dumps(dict(matched_Zero_portable_reported_metrics_passed=result["ZeRO_portable"]["matched_parameter_storage_FP32_passed"],
                          target_tokens=result["ZeRO_portable"]["target_tokens_per_precision"],
                          selected_positions=result["ZeRO_portable"]["selected_positions_per_precision"],
                          resources_restoration="operator_declared_not_independently_verified")), flush=True)


if __name__ == "__main__":
    main()
