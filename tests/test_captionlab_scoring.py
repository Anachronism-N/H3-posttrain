"""Frozen music-mapping evidence: support, missingness, grouping and semantics."""

import copy
import hashlib
import json

import pytest

from captionlab.scoring import ScoringError, summarize_scores


def fixture_case(case_id="c1", group="g1", difference=1.0, action_mask=None):
    text = "The wrist moves left."
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    conditions = [{"condition_id": kind, "kind": kind, "audio_id": "audio_" + kind}
                  for kind in ("true", "random", "hard", "silence")]
    conditions[-1]["audio_id"] = conditions[0]["audio_id"]
    case = {"case_id": case_id, "source_group_id": group,
            "target_text": text, "target_sha256": digest, "conditions": conditions}
    records = []
    for condition in conditions:
        increment = 0 if condition["kind"] == "true" else difference
        record = {"case_id": case_id, "condition_id": condition["condition_id"],
                  "target_token_ids": [10, 11, 99], "token_nll": [10, 10, 10 + increment],
                  "eos_token_id": 99, "target_sha256": digest,
                  "model_identity": "frozen-model-sha", "tokenizer_identity": "tokenizer-sha"}
        if action_mask is not None:
            record["action_mask"] = list(action_mask)
        records.append(record)
    return case, records


def report(case=None, records=None, **kwargs):
    if case is None:
        case, records = fixture_case()
    return summarize_scores({"cases": [case]}, records, n_resamples=100, **kwargs)


def test_pairing_has_correct_sign_and_includes_eos_not_inferred_action():
    result = report()
    scope = result["cases"][0]["scopes"]
    assert scope["all_assistant"]["effective_token_counts"]["true"] == 3
    assert scope["all_assistant"]["comparisons"]["random"]["difference"] == pytest.approx(1 / 3)
    assert scope["explicit_action"]["comparisons"]["random"]["difference"] is None
    assert result["aggregation"]["explicit_action"]["random"]["interval_status"] == "no_complete_pairs"
    assert result["coverage"]["n_records_expected"] == 4
    assert result["status"] == "complete_records"
    assert result["reviewed_subset_ranking"]["all_assistant"]["n_cases_scored"] == 0
    assert "accuracy" not in result
    assert "p_value" not in result


def test_explicit_action_support_separate_from_eos_and_no_input_mutation():
    case, records = fixture_case(action_mask=[True, False, False])
    before = copy.deepcopy((case, records))
    result = report(case, records)
    assert (case, records) == before
    action = result["cases"][0]["scopes"]["explicit_action"]
    assert action["effective_token_counts"]["true"] == 1
    assert action["comparisons"]["hard"]["difference"] == 0
    assert action["descriptive_candidate_rank"]["true_rank_max"] == 3
    assert result["aggregation"]["all_assistant"]["hard"]["estimate_group_equal"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mask", [[False, False, False], None])
def test_unknown_or_empty_action_support_is_na(mask):
    case, records = fixture_case(action_mask=mask)
    result = report(case, records)
    action = result["cases"][0]["scopes"]["explicit_action"]
    assert action["comparisons"]["random"]["difference"] is None
    assert action["descriptive_candidate_rank"] is None


def test_same_source_cases_do_not_create_independent_ci():
    cases, rows = [], []
    for index in range(10):
        case, records = fixture_case("case" + str(index), "one_recording")
        cases.append(case)
        rows.extend(records)
    result = summarize_scores({"cases": cases}, rows, n_resamples=30)
    aggregate = result["aggregation"]["all_assistant"]["random"]
    assert aggregate["n_cases_paired"] == 10
    assert aggregate["n_source_groups_observed"] == 1
    assert aggregate["interval_group_equal"] is None
    assert aggregate["interval_status"] == "insufficient_independent_source_groups"


def test_equal_group_primary_does_not_silently_case_weight_large_source():
    cases, rows = [], []
    for case_id, group, difference in [("a", "large", 0), ("b", "large", 0),
                                        ("c", "large", 0), ("d", "small", 12)]:
        case, records = fixture_case(case_id, group, difference)
        cases.append(case)
        rows.extend(records)
    manifest = {"cases": cases}
    one = summarize_scores(manifest, rows, n_resamples=200, seed=9)
    two = summarize_scores(manifest, rows, n_resamples=200, seed=9)
    assert one == two
    aggregate = one["aggregation"]["all_assistant"]["random"]
    assert aggregate["estimate_group_equal"] == 2
    assert aggregate["estimate_case_equal"] == 1
    assert aggregate["interval_group_equal"] == [0, 4]
    assert aggregate["n_source_groups_expected"] == 2


def test_missing_cases_and_conditions_retained_without_zero_imputation():
    one, rows1 = fixture_case("complete", "g1")
    two, rows2 = fixture_case("missing_random", "g2")
    three, _ = fixture_case("no_records", "g3")
    rows2 = [row for row in rows2 if row["condition_id"] != "random"]
    result = summarize_scores({"cases": [one, two, three]}, rows1 + rows2, n_resamples=20)
    assert result["status"] == "incomplete_records"
    coverage = result["coverage"]
    assert coverage["n_cases_expected"] == 3 and coverage["n_cases_with_all_records"] == 1
    assert coverage["n_records_expected"] == 12 and coverage["n_records_observed"] == 7
    assert coverage["n_records_missing"] == 5
    random = result["aggregation"]["all_assistant"]["random"]
    hard = result["aggregation"]["all_assistant"]["hard"]
    assert random["n_cases_paired"] == 1 and hard["n_cases_paired"] == 2
    assert random["n_cases_missing_or_unsupported"] == 2
    assert len(result["cases"]) == 3
    missing = next(row for row in result["cases"] if row["case_id"] == "no_records")
    assert missing["scopes"]["all_assistant"]["condition_mean_nll"]["true"] is None


def test_no_records_returns_all_manifest_denominators_and_na_identity():
    case, _ = fixture_case()
    result = report(case, [])
    assert result["identity"] is None
    assert result["coverage"]["n_records_missing"] == 4
    assert result["aggregation"]["all_assistant"]["random"]["estimate_group_equal"] is None


def test_multiple_candidates_of_same_kind_rejected_without_inflating_denominator():
    case, rows = fixture_case()
    case["conditions"].append({"condition_id": "random2", "kind": "random", "audio_id": "r2"})
    with pytest.raises(ScoringError, match="duplicate condition kind"):
        report(case, rows)


def test_reviewed_subset_excludes_unreviewed_and_known_positive_candidates():
    case, rows = fixture_case()
    case["conditions"][1]["mismatch_review"] = "reviewed_mismatch"
    case["conditions"][2]["mismatch_review"] = "known_positive"
    rows[2]["token_nll"] = [0, 0, 0]  # Positive alternative wins, never penalize subset accuracy.
    result = report(case, rows)
    scope = result["cases"][0]["scopes"]["all_assistant"]
    assert scope["descriptive_candidate_rank"]["true_rank_min"] == 2
    assert scope["descriptive_rank_is_accuracy"] is False
    assert scope["reviewed_subset_rank"]["n_candidates"] == 2
    assert scope["reviewed_subset_rank"]["true_unique_best"] is True
    assert scope["known_positive_condition_ids"] == ["hard"]
    summary = result["reviewed_subset_ranking"]["all_assistant"]
    assert summary["unique_top1_rate_scored_reviewed_subset"] == 1
    assert summary["n_cases_with_reviewed_mismatch_subset"] == 1


def test_ties_exposed_and_silence_not_a_rank_candidate():
    case, rows = fixture_case(difference=0)
    case["conditions"][1]["mismatch_review"] = "reviewed_mismatch"
    rows[3]["token_nll"] = [0, 0, 0]
    result = report(case, rows)
    rank = result["cases"][0]["scopes"]["all_assistant"]["descriptive_candidate_rank"]
    assert rank["n_candidates"] == 3 and rank["true_rank_min"] == 1 and rank["true_rank_max"] == 3
    assert rank["true_unique_best"] is False and rank["true_best_including_ties"] is True
    summary = result["reviewed_subset_ranking"]["all_assistant"]
    assert summary["unique_top1_rate_scored_reviewed_subset"] == 0
    assert summary["best_including_ties_rate_scored_reviewed_subset"] == 1


def test_tie_tolerance_is_explicit_and_negative_difference_not_success():
    case, rows = fixture_case(difference=-0.003)
    precise = report(case, rows)
    tolerant = report(case, rows, tie_tolerance=0.01)
    assert precise["cases"][0]["scopes"]["all_assistant"]["descriptive_candidate_rank"]["true_rank_min"] == 3
    assert tolerant["cases"][0]["scopes"]["all_assistant"]["descriptive_candidate_rank"]["true_rank_min"] == 1


def test_missing_reviewed_ranking_is_pending_not_failure_or_success():
    case, rows = fixture_case()
    case["conditions"][1]["mismatch_review"] = "reviewed_mismatch"
    result = report(case, [row for row in rows if row["condition_id"] != "random"])
    ranking = result["reviewed_subset_ranking"]["all_assistant"]
    assert ranking["n_cases_with_reviewed_mismatch_subset"] == 1
    assert ranking["n_cases_pending"] == 1 and ranking["n_cases_scored"] == 0
    assert ranking["unique_top1_rate_scored_reviewed_subset"] is None


def test_manifest_true_positive_and_same_source_silence_are_valid():
    case, rows = fixture_case()
    case["conditions"][0]["mismatch_review"] = "known_positive"
    result = report(case, rows)
    assert result["status"] == "complete_records"
    assert result["cases"][0]["scopes"]["all_assistant"]["known_positive_condition_ids"] == []


def test_unavailable_hard_stays_in_predeclared_denominator_and_rank_is_partial():
    case, rows = fixture_case()
    case["conditions"] = [condition for condition in case["conditions"] if condition["kind"] != "hard"]
    case["unavailable_conditions"] = [{"kind": "hard", "reason": "no_explicit_hard_pair"}]
    rows = [row for row in rows if row["condition_id"] != "hard"]
    result = report(case, rows)
    coverage = result["coverage"]
    assert result["status"] == "complete_available_records_with_unavailable_conditions"
    assert coverage["n_conditions_predeclared"] == 4
    assert coverage["n_conditions_unavailable"] == 1
    assert coverage["n_records_expected"] == 3 and coverage["n_records_missing"] == 0
    assert coverage["n_cases_with_all_records"] == 1
    assert coverage["n_cases_with_all_predeclared_conditions_and_records"] == 0
    hard = result["aggregation"]["all_assistant"]["hard"]
    assert hard["n_cases_expected"] == 1 and hard["n_cases_paired"] == 0
    assert hard["estimate_group_equal"] is None
    scope = result["cases"][0]["scopes"]["all_assistant"]
    assert scope["comparisons"]["hard"]["n_conditions_expected"] == 1
    assert scope["comparisons"]["hard"]["n_conditions_available"] == 0
    rank = scope["descriptive_candidate_rank"]
    assert rank["n_candidates"] == 2
    assert rank["condition_kind_availability"]["hard"] is False
    assert rank["all_predeclared_candidate_kinds_available"] is False


def test_no_donor_available_does_not_make_trivial_one_candidate_rank_success():
    case, rows = fixture_case()
    case["conditions"] = [condition for condition in case["conditions"]
                          if condition["kind"] in ("true", "silence")]
    case["unavailable_conditions"] = [{"kind": kind, "reason": "no_eligible_donor"}
                                      for kind in ("random", "hard")]
    rows = [row for row in rows if row["condition_id"] in ("true", "silence")]
    result = report(case, rows)
    assert result["cases"][0]["scopes"]["all_assistant"]["descriptive_candidate_rank"] is None
    assert result["aggregation"]["all_assistant"]["silence"]["n_cases_paired"] == 1
    assert result["coverage"]["n_conditions_unavailable"] == 2


@pytest.mark.parametrize("unavailable", [
    [], [{"kind": "hard", "reason": ""}], [{"kind": "true", "reason": "missing"}],
    [{"kind": "hard", "reason": "missing"}, {"kind": "hard", "reason": "duplicate"}],
    [{"kind": "hard", "reason": "missing"}, {"kind": "random", "reason": "but_present"}],
])
def test_unavailable_candidate_declarations_must_exactly_match_absent_kinds(unavailable):
    case, rows = fixture_case()
    case["conditions"] = [condition for condition in case["conditions"] if condition["kind"] != "hard"]
    case["unavailable_conditions"] = unavailable
    with pytest.raises(ScoringError):
        report(case, [])


def test_silence_cannot_reference_an_unrelated_audio_identity():
    case, rows = fixture_case()
    case["conditions"][-1]["audio_id"] = "different-audio"
    with pytest.raises(ScoringError, match="silence must transform"):
        report(case, rows)


@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf"),
                                   -float("inf"), -0.1, 10 ** 400, 2 ** 60 + 1])
def test_invalid_nll_values_rejected(value):
    case, rows = fixture_case()
    rows[0]["token_nll"][0] = value
    with pytest.raises(ScoringError):
        report(case, rows)


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0].update(case_id="unknown"),
    lambda rows: rows[0].update(condition_id="unknown"),
    lambda rows: rows[0].update(target_sha256="0" * 64),
    lambda rows: rows[1].update(model_identity="different-adapter"),
    lambda rows: rows[1].update(tokenizer_identity="other-tokenizer"),
    lambda rows: rows[0].update(target_token_ids=[10, 12, 99]),
    lambda rows: rows[0].update(target_token_ids=[10, 99, 99]),
    lambda rows: rows[0].update(target_token_ids=[10, 11, 98]),
    lambda rows: rows[0].update(target_token_ids=[True, 11, 99]),
    lambda rows: rows[0].update(target_token_ids=[-1, 11, 99]),
    lambda rows: rows[0].update(token_nll=[1, 1]),
    lambda rows: rows[0].update(eos_token_id=True),
    lambda rows: rows[0].update(typo_nll=[]),
    lambda rows: rows[0].pop("model_identity"),
])
def test_invalid_or_inconsistent_records_rejected(mutate):
    case, rows = fixture_case()
    mutate(rows)
    with pytest.raises(ScoringError):
        report(case, rows)


@pytest.mark.parametrize("bad", [[1, 0, 0], [True, False], [True, False, True], None])
def test_invalid_action_mask_rejected(bad):
    case, rows = fixture_case()
    rows[0]["action_mask"] = bad
    with pytest.raises(ScoringError):
        report(case, rows)


def test_action_support_cannot_change_or_disappear_across_audio_conditions():
    case, rows = fixture_case(action_mask=[True, False, False])
    rows[1]["action_mask"] = [False, True, False]
    with pytest.raises(ScoringError, match="support must be identical"):
        report(case, rows)
    rows[1].pop("action_mask")
    with pytest.raises(ScoringError, match="support must be identical"):
        report(case, rows)


@pytest.mark.parametrize("mutate", [
    lambda case: case.update(target_text="different content"),
    lambda case: case.update(source_group_id=""),
    lambda case: case["conditions"].append(copy.deepcopy(case["conditions"][0])),
    lambda case: case["conditions"][0].update(kind="random"),
    lambda case: case["conditions"].pop(2),
    lambda case: case["conditions"][1].update(kind="negative"),
    lambda case: case["conditions"][1].update(mismatch_review="approved"),
    lambda case: case["conditions"][0].update(mismatch_review="reviewed_mismatch"),
    lambda case: case["conditions"][1].update(audio_id=""),
    lambda case: case["conditions"][1].update(audio_id=case["conditions"][0]["audio_id"]),
])
def test_manifest_contract_violations_rejected(mutate):
    case, rows = fixture_case()
    mutate(case)
    with pytest.raises(ScoringError):
        report(case, rows)


def test_duplicate_case_and_nonfinite_metadata_rejected():
    case, rows = fixture_case()
    with pytest.raises(ScoringError, match="duplicate"):
        summarize_scores({"cases": [case, copy.deepcopy(case)]}, rows)
    with pytest.raises(ScoringError, match="nonfinite"):
        summarize_scores({"cases": [case], "metadata": float("nan")}, rows)


@pytest.mark.parametrize("options", [
    {"seed": True}, {"seed": -1}, {"seed": 2 ** 63}, {"n_resamples": True},
    {"n_resamples": 0}, {"n_resamples": 100001}, {"confidence": 0},
    {"confidence": 1}, {"confidence": True}, {"tie_tolerance": -0.1},
    {"tie_tolerance": float("inf")},
])
def test_invalid_summary_options_rejected(options):
    case, rows = fixture_case()
    with pytest.raises(ScoringError):
        summarize_scores({"cases": [case]}, rows, **options)


def test_large_finite_values_mean_and_bootstrap_do_not_overflow():
    case1, rows1 = fixture_case("c1", "g1")
    case2, rows2 = fixture_case("c2", "g2")
    for row in rows1:
        row["token_nll"] = [0, 0, 0] if row["condition_id"] == "true" else [1e308] * 3
    for row in rows2:
        row["token_nll"] = [1e308] * 3 if row["condition_id"] == "true" else [0, 0, 0]
    result = summarize_scores({"cases": [case1, case2]}, rows1 + rows2, n_resamples=50)
    aggregate = result["aggregation"]["all_assistant"]["random"]
    assert aggregate["estimate_group_equal"] == 0
    assert aggregate["interval_group_equal"] == [-1e308, 1e308]
    json.dumps(result, allow_nan=False)


def test_input_hashes_bind_target_metadata_and_records():
    case, rows = fixture_case()
    first = report(case, rows)
    rows[0]["token_nll"][0] += 1
    second = report(case, rows)
    assert first["manifest_content_sha256"] == second["manifest_content_sha256"]
    assert first["records_content_sha256"] != second["records_content_sha256"]
    case["source_group_id"] = "different-component"
    third = report(case, rows)
    assert second["manifest_content_sha256"] != third["manifest_content_sha256"]


def test_bridge_metadata_is_bound_without_confusing_raw_and_canonical_manifest_hashes():
    case, rows = fixture_case()
    for row in rows:
        row.update(schema="captionlab.token_nll.v1", manifest_sha256="a" * 64,
                   runtime_config_sha256="b" * 64,
                   runtime_details={"input_tokens": 76, "cache_parity": "not_verified",
                                    "attention_mask_sha256": "c" * 64})
    first = report(case, rows)
    assert first["record_declared_bindings"]["manifest_sha256"] == "a" * 64
    assert first["record_declared_bindings"]["manifest_sha256"] != first["manifest_content_sha256"]
    rows[0]["runtime_details"]["input_tokens"] += 1
    second = report(case, rows)
    assert first["records_content_sha256"] != second["records_content_sha256"]


@pytest.mark.parametrize("mutate", [
    lambda rows: rows[0].update(manifest_sha256="bad"),
    lambda rows: rows[0].update(runtime_config_sha256="b" * 64),
    lambda rows: rows[0].pop("schema"),
    lambda rows: rows[0].update(runtime_details=[]),
    lambda rows: rows[0].update(runtime_details={"nll": float("nan")}),
])
def test_inconsistent_or_invalid_bridge_metadata_rejected(mutate):
    case, rows = fixture_case()
    for row in rows:
        row.update(schema="captionlab.token_nll.v1", manifest_sha256="a" * 64,
                   runtime_config_sha256="a" * 64, runtime_details={})
    mutate(rows)
    with pytest.raises(ScoringError):
        report(case, rows)
