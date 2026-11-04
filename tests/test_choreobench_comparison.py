"""Paired comparison regressions: missingness, false N, and numerical integrity."""

import copy
import json

import pytest

from choreobench.comparison import compare_methods
from choreobench.schema import ValidationError


def pair(song_id, group_id, a, b, reviewer_id=None):
    rows = [{"song_id": song_id, "group_id": group_id, "method_id": method, "value": value}
            for method, value in (("A", a), ("B", b))]
    if reviewer_id is not None:
        for row in rows:
            row["reviewer_id"] = reviewer_id
    return rows


def test_equal_song_pairing_does_not_weight_songs_by_number_of_ratings():
    rows = pair("many_ratings", "source1", 0, 2) * 100 + pair("few_ratings", "source2", 10, 0)
    snapshot = copy.deepcopy(rows)
    result = compare_methods(rows, "A", "B", n_resamples=100)
    assert result["estimate"] == -4
    assert result["method_a_mean"] == 5 and result["method_b_mean"] == 1
    assert result["n_songs"] == 2 and result["n_independent_groups"] == 2
    assert rows == snapshot
    assert "p_value" not in result and "significant" not in result


def test_all_songs_in_one_source_group_do_not_create_an_independent_ci():
    rows = sum((pair("song_" + str(index), "one_original_recording", 1, 2)
                for index in range(20)), [])
    result = compare_methods(rows, "A", "B")
    assert result["estimate"] == 1
    assert result["n_songs"] == 20 and result["n_independent_groups"] == 1
    assert result["interval"] is None
    assert result["status"] == "insufficient_independent_groups"


def test_cluster_resampling_retains_whole_unequal_groups_and_exposes_weighting():
    rows = pair("s1", "large_source", 0, 0) + pair("s2", "large_source", 0, 0)
    rows += pair("s3", "large_source", 0, 0) + pair("s4", "small_source", 0, 10)
    result = compare_methods(rows, "A", "B", n_resamples=400, seed=2)
    assert result["estimate"] == 2.5
    assert result["equal_group_mean_difference"] == 5
    assert result["interval"] == [0, 10]
    assert [group["n_songs"] for group in result["groups"]] == [3, 1]
    assert "equal weight" in result["comparison_unit"]


def test_strict_missing_method_rejected_and_intersection_reports_omissions():
    rows = pair("complete", "g1", 1, 2)
    rows += [{"song_id": "failed_b", "group_id": "g2", "method_id": "A", "value": 5}]
    with pytest.raises(ValidationError, match="failed_b=B"):
        compare_methods(rows, "A", "B")
    result = compare_methods(rows, "A", "B", allow_incomplete=True)
    assert result["estimate"] == 1 and result["n_songs"] == 1
    assert result["coverage"]["paired_fraction"] == 0.5
    assert result["coverage"]["omitted_requested_method_rows"] == 1
    assert result["coverage"]["omitted_songs"] == [
        {"song_id": "failed_b", "group_id": "g2", "present_methods": ["A"], "missing_methods": ["B"]}]
    assert not result["coverage"]["complete_for_observed_song_universe"]


def test_other_method_only_songs_are_not_silently_dropped_from_denominator():
    rows = pair("complete", "g1", 1, 2)
    rows += [{"song_id": "seen_only_for_c", "group_id": "g2", "method_id": "C", "value": 5}]
    with pytest.raises(ValidationError, match="seen_only_for_c=A,B"):
        compare_methods(rows, "A", "B")
    result = compare_methods(rows, "A", "B", allow_incomplete=True)
    assert result["coverage"]["input_songs"] == 2
    assert result["coverage"]["ignored_method_ids"] == ["C"]
    assert result["coverage"]["ignored_method_rows"] == 1
    assert result["coverage"]["omitted_songs"][0]["missing_methods"] == ["A", "B"]


def test_no_intersection_never_returns_zero_as_an_invented_effect():
    rows = [{"song_id": "a_only", "group_id": "g1", "method_id": "A", "value": 5},
            {"song_id": "b_only", "group_id": "g2", "method_id": "B", "value": 1}]
    result = compare_methods(rows, "A", "B", allow_incomplete=True)
    assert result["status"] == "no_paired_songs"
    assert result["estimate"] is None and result["interval"] is None
    assert result["coverage"]["omitted_song_count"] == 2


def test_empty_rows_rejected_by_default_and_explicitly_reported_in_intersection_mode():
    with pytest.raises(ValidationError):
        compare_methods([], "A", "B")
    result = compare_methods([], "A", "B", allow_incomplete=True)
    assert result["n_songs"] == 0 and result["coverage"]["paired_fraction"] is None
    assert result["status"] == "no_paired_songs" and result["estimate"] is None


@pytest.mark.parametrize("bad", [None, True, "4", float("nan"), float("inf"), -float("inf"), 10 ** 400])
def test_missing_boolean_nonfinite_and_unrepresentable_ratings_rejected(bad):
    rows = pair("s", "g", 1, bad)
    with pytest.raises(ValidationError):
        compare_methods(rows, "A", "B")


def test_difference_overflow_rejected_even_when_each_rating_is_finite():
    with pytest.raises(ValidationError, match="difference"):
        compare_methods(pair("s", "g", -1.7e308, 1.7e308), "A", "B")


def test_large_integer_precision_loss_cannot_silently_turn_a_difference_into_zero():
    with pytest.raises(ValidationError, match="integer precision"):
        compare_methods(pair("s", "g", 2 ** 53, 2 ** 53 + 1), "A", "B")


def test_aggregation_and_percentile_do_not_overflow_for_large_finite_values():
    rows = pair("s1", "g1", 0, 1.7e308) * 2 + pair("s2", "g2", 0, 1.7e308) * 2
    result = compare_methods(rows, "A", "B", n_resamples=31)
    assert result["estimate"] == 1.7e308 and result["interval"] == [1.7e308, 1.7e308]
    json.dumps(result, allow_nan=False)


def test_group_identity_must_agree_for_every_song_across_methods():
    rows = pair("s", "g", 1, 2)
    rows[1]["group_id"] = "pretend_independent_group"
    with pytest.raises(ValidationError, match="inconsistent group_id"):
        compare_methods(rows, "A", "B")


@pytest.mark.parametrize("field,value", [("song_id", ""), ("group_id", None), ("method_id", []), ("reviewer_id", None)])
def test_missing_or_invalid_identity_fields_are_not_inferred(field, value):
    rows = pair("s", "g", 1, 2)
    rows[0][field] = value
    with pytest.raises(ValidationError):
        compare_methods(rows, "A", "B")


def test_repeated_identified_reviewer_in_a_cell_is_rejected_even_for_equal_values():
    rows = pair("s", "g", 1, 2, "rater1")
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValidationError, match="duplicate reviewer_id"):
        compare_methods(rows, "A", "B")


def test_same_reviewer_can_rate_both_methods_and_other_songs_without_false_n():
    rows = pair("s1", "g1", 1, 2, "rater1") + pair("s2", "g2", 1, 3, "rater1")
    result = compare_methods(rows, "A", "B", n_resamples=30)
    assert result["n_songs"] == 2 and result["n_independent_groups"] == 2
    assert all(song["reviewer_pairing_status"] == "same_identified_set" for song in result["song_pairs"])
    assert result["coverage"]["rows_with_reviewer_id"] == 4


def test_distinct_or_missing_reviewer_sets_are_reported_not_claimed_paired():
    rows = pair("s1", "g1", 1, 2)
    rows[0]["reviewer_id"], rows[1]["reviewer_id"] = "rater1", "rater2"
    rows += pair("s2", "g2", 1, 2)
    result = compare_methods(rows, "A", "B", n_resamples=10)
    assert result["song_pairs"][0]["reviewer_pairing_status"] == "different_identified_sets"
    assert result["song_pairs"][1]["reviewer_pairing_status"] == "not_fully_identified"
    assert result["coverage"]["rows_without_reviewer_id"] == 2


def test_omitted_rating_origins_remain_explicitly_unknown():
    result = compare_methods(pair("s", "g", 1, 2), "A", "B")
    assert result["rating_origin"] == "unknown"
    assert result["coverage"]["selected_rows_without_declared_origin"] == 2


@pytest.mark.parametrize("origin", ["human", "model_judge"])
def test_homogeneous_declared_rating_origin_is_preserved(origin):
    rows = pair("s", "g", 1, 2)
    for row in rows:
        row["origin"] = origin
    result = compare_methods(rows, "A", "B")
    assert result["rating_origin"] == origin
    assert result["coverage"]["selected_rows_with_declared_origin"] == 2


def test_human_and_model_judge_ratings_cannot_be_averaged_together():
    rows = pair("s", "g", 1, 2)
    rows[0]["origin"], rows[1]["origin"] = "human", "model_judge"
    with pytest.raises(ValidationError, match="mix human"):
        compare_methods(rows, "A", "B")


def test_declared_and_unknown_origins_cannot_be_mixed():
    rows = pair("s", "g", 1, 2)
    rows[0]["origin"] = "human"
    with pytest.raises(ValidationError, match="declared origin and unknown"):
        compare_methods(rows, "A", "B")


@pytest.mark.parametrize("origin", [None, True, "assistant_delegated", "", []])
def test_invalid_selected_rating_origins_rejected(origin):
    rows = pair("s", "g", 1, 2)
    rows[0]["origin"] = origin
    with pytest.raises(ValidationError, match="origin"):
        compare_methods(rows, "A", "B")


def test_unselected_method_origin_does_not_mix_into_selected_comparison():
    rows = pair("s", "g", 1, 2)
    for row in rows:
        row["origin"] = "human"
    rows.append({"song_id": "s", "group_id": "g", "method_id": "C", "value": 4, "origin": "model_judge"})
    result = compare_methods(rows, "A", "B")
    assert result["rating_origin"] == "human" and result["coverage"]["ignored_method_rows"] == 1


def test_intersection_cannot_hide_mixed_origin_on_an_omitted_selected_method_song():
    rows = pair("complete", "g1", 1, 2)
    for row in rows:
        row["origin"] = "human"
    rows.append({"song_id": "a_only", "group_id": "g2", "method_id": "A", "value": 4, "origin": "model_judge"})
    with pytest.raises(ValidationError, match="mix human"):
        compare_methods(rows, "A", "B", allow_incomplete=True)


def test_metric_selection_and_optional_bounds_are_explicit():
    rows = pair("s", "g", 999, 999)
    rows[0]["musical_fit"], rows[1]["musical_fit"] = 1, 4
    result = compare_methods(rows, "A", "B", metric="musical_fit", value_bounds=[1, 5])
    assert result["estimate"] == 3 and result["metric"] == "musical_fit"
    assert result["value_bounds"] == [1, 5]
    with pytest.raises(ValidationError, match="outside"):
        compare_methods(rows, "A", "B", value_bounds=[1, 5])


@pytest.mark.parametrize("kwargs", [
    {"method_a": "A", "method_b": "A"}, {"method_a": "", "method_b": "B"},
    {"metric": "song_id"}, {"metric": "missing_score"}, {"n_resamples": 0},
    {"n_resamples": True}, {"confidence": float("nan")}, {"confidence": 1},
    {"confidence": True}, {"seed": False}, {"seed": -1}, {"allow_incomplete": "true"},
    {"value_bounds": [5, 1]}, {"value_bounds": [0, True]}, {"value_bounds": [0, float("inf")]},
])
def test_bad_comparison_configuration_rejected(kwargs):
    arguments = dict(method_a="A", method_b="B", n_resamples=10)
    arguments.update(kwargs)
    with pytest.raises(ValidationError):
        compare_methods(pair("s", "g", 1, 2), **arguments)


def test_deterministic_results_ignore_input_order_except_input_identity_hash():
    rows = pair("s1", "g1", 1, 2, "r1") + pair("s1", "g1", 2, 4, "r2")
    rows += pair("s2", "g2", 3, 2, "r1") + pair("s3", "g1", 2, 3, "r1")
    first = compare_methods(rows, "A", "B", n_resamples=111, seed=81)
    again = compare_methods(rows, "A", "B", n_resamples=111, seed=81)
    reversed_result = compare_methods(list(reversed(rows)), "A", "B", n_resamples=111, seed=81)
    assert first == again
    assert first.pop("rows_sha256") != reversed_result.pop("rows_sha256")
    assert first == reversed_result
