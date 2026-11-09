"""Tiny CPU evidence only; never rerun the published checkpoint experiment."""
from copy import deepcopy
import json

import pytest

torch = pytest.importorskip("torch")

from captionlab import numeric_report as report
from captionlab.numeric_common import digest_tensor


CASES = ("source-a", "source-b", "source-c")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(count=3):
    tokens = torch.tensor([[8, 9] + list(range(1, count + 1))], dtype=torch.long)
    labels = tokens.clone()
    labels[:, :2] = -100
    return dict(input_ids=tokens, labels=labels, prefix_tokens=2, positions=[1, count],
                selected_logits=torch.zeros(2, 12), target_logprobs=torch.full((count,), -1.),
                input_ids_sha256=digest_tensor(tokens), labels_sha256=digest_tensor(labels),
                input_feature_sha256="a" * 64, feature_mask_sha256="b" * 64)


def position_row():
    return dict(max_logit_abs_difference=0., max_centered_logit_abs_difference=0.,
                total_variation=0., target_logprob_difference=0., top1_equal=True,
                reference_top1=1, candidate_top1=1)


def comparison(case, mode):
    return dict(case=case, comparison=mode, positions=2, top1_changes=0, max_logit=0.,
                max_selected_logprob=0., all_target_logprob_max=0., all_target_logprob_mean=0.,
                NLL_delta=0., exact_selected=True, exact_all_target_logprob=True)


def fixture(root):
    evidence = root / "evidence"
    original = evidence / "zero_original"
    original.mkdir(parents=True)
    for index, case in enumerate(CASES):
        torch.save(snapshot(index + 3), original / (case + "-original_fp32_features.pt"))
    rows = [comparison(case, mode) for case in CASES for mode in report._EXACT_MODES]
    rows += [comparison(case, "current_portable_BF16") for case in CASES]
    write(evidence / "zero_portable_same_inputs_comparison.json", rows)
    write(evidence / "cache_calibration/summary.json", dict(cases=[
        dict(case_id=case, contrasts={name: [position_row(), position_row()]
             for name in ("repeat_full", "repeat_cache", "full_vs_cache")}) for case in CASES]))
    write(evidence / "matched_portable/summary.json", dict(records=[
        dict(case_id=case, mode=mode, **{name: [position_row(), position_row()]
             for name in ("repeat_control", "full_shortened", "full_cached", "shortened_cached")})
        for case in CASES for mode in report._PRECISION_MODES]))
    write(evidence / "route_calibration/summary.json", dict(records=[dict(case_id=case) for case in CASES]))
    for folder in report._RESOURCE_FOLDERS:
        write(evidence / folder / "resources_restored.json",
              dict(holder_compute_verified=True, unknown_jobs_not_signaled=True, fixture_only=True))
    write(original / "summary.json", dict(original_cursor=dict(update=123), zero3_world_size=8,
          actual_learned_FP32_tensor_restore_verified=True, optimizer_state_unchanged=True))
    write(original / "restore_receipt.json", dict(actual_trainable_tensors_match_saved_FP32=True,
          optimizer_restored_exact=True, checkpoint_copy="operator-supplied-path-not-independent-proof"))
    write(evidence / "FP32_norm_fix_comparison.json", [])
    write(evidence / "hybrid_norm_embedding_comparison.json", [])
    write(evidence / "portable_frozen_inputs/summary.json", dict(cases=[dict(case_id=case) for case in CASES]))
    return evidence


def test_complete_evidence_uses_actual_case_support_and_no_upstream_authentication(tmp_path):
    fixture(tmp_path)
    result = report.assemble_report(tmp_path)
    assert result["case_count"] == 3 and result["original_checkpoint_update"] == 123
    assert result["ZeRO_portable"]["target_tokens_per_precision"] == 12
    assert result["ZeRO_portable"]["selected_positions_per_precision"] == 6
    assert {row["case_id"] for row in result["identity"]["source_signatures"]} == set(CASES)
    assert result["ZeRO_portable"]["matched_all_target_logprobs_bitwise_equal"] is True
    assert result["ZeRO_portable"]["candidate_tensor_equality_independently_recomputed"] is False
    for field in ("original_learned_tensor_FP32_exact_restore", "original_optimizer_unchanged",
                  "original_checkpoint_not_written", "independent_copy_used",
                  "same_input_ids_masks_and_frozen_features_verified"):
        assert result["identity"][field] is None
    assert result["identity"]["operator_declarations"]["restore_receipt"]["optimizer_restored_exact"] is True
    assert result["resources_restored_independently_verified"] is False
    assert result["new_training_updates"] is None and result["implementation_tests"] is None
    assert result["report_assembly"] == dict(model_calls=0, training_updates=0)
    assert result["source_evidence"] and not (tmp_path / "numeric_equivalence_summary.json").exists()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", ["duplicate_six", "missing", "wrong_case", "wrong_mode", "extra"])
def test_exact_case_times_two_precision_inventory_is_required(tmp_path, change):
    evidence = fixture(tmp_path)
    path = evidence / "zero_portable_same_inputs_comparison.json"
    rows = read(path)
    if change == "duplicate_six": rows = [deepcopy(rows[0]) for _ in range(6)]
    elif change == "missing": rows.pop(0)
    elif change == "wrong_case": rows[0]["case"] = "not-an-original-source"
    elif change == "wrong_mode": rows[0]["comparison"] = "matched_FP16"
    else: rows.append(deepcopy(rows[0]))
    write(path, rows)
    with pytest.raises(ValueError):
        report.assemble_report(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("exact_selected", "true"), ("exact_all_target_logprob", 1),
    ("exact_selected", False), ("exact_all_target_logprob", False),
    ("positions", 1), ("positions", True), ("top1_changes", 1),
    ("max_logit", .001), ("max_selected_logprob", .001),
    ("all_target_logprob_max", .001), ("all_target_logprob_mean", .001), ("NLL_delta", -.001),
    ("max_logit", float("nan")), ("NLL_delta", float("inf")), ("max_selected_logprob", True),
])
def test_boolean_flags_support_and_real_zero_differences_all_mandatory(tmp_path, field, value):
    evidence = fixture(tmp_path)
    path = evidence / "zero_portable_same_inputs_comparison.json"
    rows = read(path)
    rows[0][field] = value
    write(path, rows)
    with pytest.raises(ValueError):
        report.assemble_report(tmp_path)


def test_missing_required_difference_rejected(tmp_path):
    evidence = fixture(tmp_path)
    path = evidence / "zero_portable_same_inputs_comparison.json"
    rows = read(path)
    del rows[0]["all_target_logprob_mean"]
    write(path, rows)
    with pytest.raises(ValueError):
        report.assemble_report(tmp_path)


def test_exponent_overflow_and_duplicate_json_keys_rejected(tmp_path):
    evidence = fixture(tmp_path)
    path = evidence / "zero_portable_same_inputs_comparison.json"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace('"max_logit": 0.0', '"max_logit": 1e999', 1), encoding="utf-8")
    with pytest.raises(ValueError):
        report.assemble_report(tmp_path)
    path.write_text(original.replace('"max_logit": 0.0', '"max_logit": 0.0, "max_logit": 0.0', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        report.assemble_report(tmp_path)


@pytest.mark.parametrize("change", ["missing", "extra", "empty_target", "nan_target", "empty_positions",
                                    "duplicate_positions", "position_oob", "bad_digest", "bad_labels", "nan_logits"])
def test_source_snapshots_are_real_nonempty_consistent_support(tmp_path, change):
    evidence = fixture(tmp_path)
    path = evidence / "zero_original" / (CASES[0] + "-original_fp32_features.pt")
    value = torch.load(path, weights_only=True)
    if change == "missing": path.unlink()
    elif change == "extra": torch.save(value, path.with_name("extra-original_fp32_features.pt"))
    else:
        if change == "empty_target": value["target_logprobs"] = torch.empty(0)
        elif change == "nan_target": value["target_logprobs"][0] = float("nan")
        elif change == "empty_positions": value["positions"] = []
        elif change == "duplicate_positions": value["positions"] = [1, 1]
        elif change == "position_oob": value["positions"] = [1, 5000]
        elif change == "bad_digest": value["input_ids_sha256"] = "0" * 64
        elif change == "bad_labels": value["labels"][0, 3] = -100
        else: value["selected_logits"][0, 0] = float("nan")
        torch.save(value, path)
    with pytest.raises(ValueError):
        report.assemble_report(tmp_path)


def test_repeat_equal_is_measured_nonzero_is_false_missing_is_unverified(tmp_path):
    evidence = fixture(tmp_path)
    result = report.assemble_report(tmp_path)
    assert result["cache_calibration"]["identical_forward_repeats_bitwise_equal"] is True
    assert result["cache_calibration"]["identical_cache_repeats_bitwise_equal"] is True
    path = evidence / "cache_calibration/summary.json"
    data = read(path)
    data["cases"][0]["contrasts"]["repeat_full"][0]["max_logit_abs_difference"] = .001
    write(path, data)
    result = report.assemble_report(tmp_path)
    assert result["cache_calibration"]["identical_forward_repeats_bitwise_equal"] is False
    del data["cases"][0]["contrasts"]["repeat_cache"]
    write(path, data)
    result = report.assemble_report(tmp_path)
    assert result["cache_calibration"]["identical_cache_repeats_bitwise_equal"] is None
    assert "unverified" in result["cache_calibration"]["repeat_verification"]["cache"]


def test_missing_matched_repeat_does_not_invent_a_pass(tmp_path):
    evidence = fixture(tmp_path)
    path = evidence / "matched_portable/summary.json"
    data = read(path)
    del data["records"][0]["repeat_control"]
    write(path, data)
    result = report.assemble_report(tmp_path)
    assert result["cache_calibration"]["matched_precision_forward_repeats_bitwise_equal"] is None


@pytest.mark.parametrize("file,change", [("cache_calibration/summary.json", "case"),
                                       ("matched_portable/summary.json", "duplicate"),
                                       ("route_calibration/summary.json", "case")])
def test_control_inventories_must_cover_same_original_sources(tmp_path, file, change):
    evidence = fixture(tmp_path)
    path = evidence / file
    value = read(path)
    rows = value.get("cases", value.get("records"))
    if change == "duplicate": rows[-1] = deepcopy(rows[0])
    else: rows[0]["case_id"] = "not-original"
    write(path, value)
    with pytest.raises(ValueError):
        report.assemble_report(tmp_path)


def test_current_portable_pass_is_derived_and_incomplete_support_rejected(tmp_path):
    evidence = fixture(tmp_path)
    path = evidence / "zero_portable_same_inputs_comparison.json"
    rows = read(path)
    current = rows[-1]
    current.update(max_logit=.1, max_selected_logprob=.01, all_target_logprob_max=.01,
                   all_target_logprob_mean=.005, NLL_delta=.001, exact_selected=False,
                   exact_all_target_logprob=False)
    write(path, rows)
    assert report.assemble_report(tmp_path)["ZeRO_portable"]["current_single_GPU_BF16_storage_bitwise_equivalent"] is False
    rows.pop()
    write(path, rows)
    with pytest.raises(ValueError):
        report.assemble_report(tmp_path)


@pytest.mark.parametrize("value", [False, "true", 1, None])
def test_resource_receipts_require_boolean_declarations_but_never_authenticate(tmp_path, value):
    evidence = fixture(tmp_path)
    path = evidence / report._RESOURCE_FOLDERS[0] / "resources_restored.json"
    write(path, dict(holder_compute_verified=value, unknown_jobs_not_signaled=True))
    with pytest.raises(ValueError):
        report.assemble_report(tmp_path)


def test_cli_preserves_existing_summary_and_rejects_failure_before_publish(tmp_path, capsys):
    evidence = fixture(tmp_path)
    report.main(["--root", str(tmp_path)])
    destination = tmp_path / "numeric_equivalence_summary.json"
    saved = destination.read_bytes()
    assert json.loads(capsys.readouterr().out)["target_tokens"] == 12
    with pytest.raises(FileExistsError):
        report.main(["--root", str(tmp_path)])
    assert destination.read_bytes() == saved
    destination.unlink()
    path = evidence / "zero_portable_same_inputs_comparison.json"
    write(path, [])
    with pytest.raises(ValueError):
        report.main(["--root", str(tmp_path)])
    assert not destination.exists()


def test_source_mutation_during_assembly_cannot_publish_a_mixed_snapshot(tmp_path, monkeypatch):
    fixture(tmp_path)
    original = report._source_signature
    changed = False
    def mutate(path, raw):
        nonlocal changed
        value = original(path, raw)
        if not changed:
            path.write_bytes(path.read_bytes() + b" ")
            changed = True
        return value
    monkeypatch.setattr(report, "_source_signature", mutate)
    with pytest.raises(ValueError, match="changed while"):
        report.main(["--root", str(tmp_path)])
    assert not (tmp_path / "numeric_equivalence_summary.json").exists()
