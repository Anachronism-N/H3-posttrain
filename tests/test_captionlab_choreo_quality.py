"""Offline rubric integrity, missingness, source grouping and semantic boundaries."""

import copy
import json
import subprocess
import sys

import pytest

from captionlab.choreo_quality import (
    DIMENSIONS, QualityError, main, make_review_item, mechanical_diagnostics,
    review_template, rubric, summarize_quality, validate_review,
)


def item(case_id="c1", method="A", group="g1", **kwargs):
    settings = dict(audio_sha256="a" * 64, reference_text="One wrist moves left; the elbow angle increases.",
                    duration_seconds=5.0)
    settings.update(kwargs)
    text = settings.pop("text", "Dance style: unknown\n[0.000-5.000s] Raise one arm, then lower it.")
    return make_review_item(case_id, group, method, text, **settings)


def reviewed(bound, reviewer_id="r1", score=3, *, origin="human", judge_identity=None, reference_score=None):
    result = review_template(bound, reviewer_id, origin=origin, judge_identity=judge_identity)
    result["review_status"] = "completed"
    result["evidence"].update(text_seen=True, text_quotes=[bound["text"]] if bound["text"].strip() else [],
                              evidence_scope="I inspected the bound text and the audio ranges below; this is a declaration.")
    for dimension in DIMENSIONS:
        result["ratings"][dimension] = {"status": "rated", "score": score, "reason": "Independent rubric judgment."}
    if bound["audio_sha256"]:
        result["evidence"].update(audio_seen=True, audio_sha256=bound["audio_sha256"],
                                  audio_intervals_seconds=[[0, bound["duration_seconds"] or 5]])
    else:
        result["ratings"]["music_fit"] = {"status": "not_applicable", "score": None, "reason": "No audio evidence."}
    result["reference_fidelity"] = {"status": "not_applicable", "score": None, "reason": "Reference agreement not requested."}
    if reference_score is not None:
        result["reference_fidelity"] = {"status": "rated", "score": reference_score, "reason": "Agreement only, not free choreography quality."}
        result["evidence"].update(reference_seen=True, reference_quotes=[bound["reference_text"]])
    return result


def summarize(items, reviews, **kwargs):
    return summarize_quality(items, reviews, n_resamples=40, **kwargs)


def test_mechanical_empty_and_header_are_not_high_quality():
    for text in ("", "   ", "Dance style: contemporary\n[0-5s] "):
        report = mechanical_diagnostics(text, duration_seconds=5)
        assert report["empty_body"]
        assert report["semantic_quality"] == report["music_fit"] == "not_measured"
        assert report["normalized_duplicate_fraction"] is None


def test_mechanical_unicode_untimed_is_not_automatically_invalid_or_semantically_scored():
    report = mechanical_diagnostics("手臂向上伸展，然后放下。")
    assert report["format_status"] == "untimed"
    assert not report["window_errors"]
    assert report["timed_coverage_fraction"] is None
    assert report["physical_execution"] == "not_measured"


def test_mechanical_overlap_union_and_repetition_are_literal_only():
    report = mechanical_diagnostics("[0-3s] Lift one arm.\n[2-5s] Lift one arm.", duration_seconds=5)
    assert report["timed_coverage_fraction"] == 1
    assert report["normalized_duplicate_units"] == 1
    assert report["normalized_duplicate_fraction"] == .5
    assert not report["window_errors"]


@pytest.mark.parametrize("text", ["[-1-2s] Move.", "[2-1s] Move.", "[0-6s] Move.", "[0-0s] Move."])
def test_timed_errors_do_not_become_scores(text):
    result = mechanical_diagnostics(text, duration_seconds=5)
    assert result["window_errors"]
    assert result["timed_coverage_fraction"] is None
    assert result["semantic_quality"] == "not_measured"


def test_numeric_overflow_in_literal_window_is_safe_json():
    result = mechanical_diagnostics("[0-" + "9" * 400 + "s] Move.")
    assert result["recognized_windows"][0]["end"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("duration", [True, False, 0, -1, float("nan"), float("inf"), "5", 10 ** 1000])
def test_duration_validation(duration):
    with pytest.raises(QualityError):
        item(duration_seconds=duration)


def test_item_binds_exact_utf8_and_reference_without_input_mutation():
    provenance = {"dataset": "avagelion/dance-data", "folder": "sol61_captions"}
    bound = item(text="抬手。", provenance=provenance)
    provenance["folder"] = "changed"
    assert bound["provenance"]["folder"] == "sol61_captions"
    assert bound["text_sha256"] != item(text="抬手。 ")["text_sha256"]
    assert bound["reference_role"] == "training_reference_not_unique_gold"
    assert "image-space" in rubric()["reference_fidelity"]["sol61_scope"]
    assert "not timestamps" in rubric()["reference_fidelity"]["sol61_scope"]
    one = rubric()
    one["dimensions"].clear()
    assert rubric()["dimensions"]


@pytest.mark.parametrize("field", ["text", "reference_text", "audio_sha256", "source_group_id"])
def test_item_mutation_rejected(field):
    bound = item()
    bound[field] = "b" * 64 if field == "audio_sha256" else "tampered"
    with pytest.raises(QualityError, match="SHA"):
        review_template(bound, "r1")


def test_bool_numeric_equality_cannot_bypass_actual_item_hash():
    bound = item()
    assert bound["mechanical"]["recognized_windows"][0]["line"] == 2
    bound["mechanical"]["empty_body"] = 0  # False == 0, but JSON/content SHA differs.
    with pytest.raises(QualityError, match="SHA"):
        review_template(bound, "r1")


def test_pending_template_cannot_count_as_completed_review():
    bound = item()
    pending = review_template(bound, "r1")
    result = summarize([bound], [pending], expected_reviewer_ids=["r1"])
    dimension = result["origins"]["human"]["methods"]["A"]["action_clarity"]
    assert dimension["rating_status_counts"] == {"pending": 1}
    assert dimension["estimate_group_equal"] is None
    assert result["coverage"]["pending_reviews"] == 1
    pending["ratings"]["action_clarity"] = {"status": "rated", "score": 5, "reason": "Not independently inspected."}
    with pytest.raises(QualityError, match="pending"):
        validate_review(pending, bound)


def test_fully_valid_review_roundtrip_has_no_side_effect_and_hash_is_checked():
    bound = item()
    review = reviewed(bound)
    before = copy.deepcopy((bound, review))
    validated = validate_review(review, bound)
    assert (bound, review) == before
    assert validated["review_sha256"]
    assert validate_review(validated, bound) == validated
    validated["ratings"]["action_clarity"]["score"] = 4
    with pytest.raises(QualityError, match="review SHA"):
        validate_review(validated, bound)


@pytest.mark.parametrize("score", [True, False, 0, 6, 3.0, float("nan"), float("inf"), "3", None, 10 ** 1000])
def test_semantic_rating_scale_is_strict(score):
    bound = item()
    review = reviewed(bound)
    review["ratings"]["action_clarity"]["score"] = score
    with pytest.raises(QualityError):
        validate_review(review, bound)


@pytest.mark.parametrize("field", ["item_sha256", "rubric_sha256"])
def test_review_binding(field):
    bound = item()
    review = reviewed(bound)
    review[field] = "b" * 64
    with pytest.raises(QualityError, match="binding"):
        validate_review(review, bound)


@pytest.mark.parametrize("mutation", ["unknown_key", "missing_dimension", "new_dimension", "bad_quote", "empty_scope", "unseen_text", "bad_status", "list_status"])
def test_review_format_and_text_evidence_are_checked(mutation):
    bound = item()
    review = reviewed(bound)
    if mutation == "unknown_key":
        review["automatic_pass"] = True
    elif mutation == "missing_dimension":
        del review["ratings"]["music_fit"]
    elif mutation == "new_dimension":
        review["ratings"]["physical_feasibility"] = review["ratings"]["action_clarity"]
    elif mutation == "bad_quote":
        review["evidence"]["text_quotes"] = ["Invented action not in the output."]
    elif mutation == "empty_scope":
        review["evidence"]["evidence_scope"] = " "
    elif mutation == "unseen_text":
        review["evidence"]["text_seen"] = False
    elif mutation == "bad_status":
        review["ratings"]["action_clarity"]["status"] = "passed"
    else:
        review["ratings"]["action_clarity"]["status"] = []
    with pytest.raises(QualityError):
        validate_review(review, bound)


def test_no_audio_is_na_not_a_low_or_lexical_music_score():
    bound = item(audio_sha256=None)
    review = reviewed(bound, score=5)
    result = summarize([bound], [review])
    music = result["origins"]["human"]["methods"]["A"]["music_fit"]
    assert music["estimate_group_equal"] is None
    assert music["rating_status_counts"] == {"not_applicable": 1}
    assert result["origins"]["human"]["methods"]["A"]["action_clarity"]["estimate_group_equal"] == 5
    review["ratings"]["music_fit"] = {"status": "rated", "score": 5, "reason": "Guessed from music words."}
    with pytest.raises(QualityError, match="audio evidence"):
        validate_review(review, bound)


@pytest.mark.parametrize("mutation", ["wrong_hash", "unheard", "empty_ranges", "partial", "outside", "nonfinite", "bool"])
def test_audio_evidence_binding_ranges_and_coverage(mutation):
    bound = item()
    review = reviewed(bound)
    evidence = review["evidence"]
    if mutation == "wrong_hash":
        evidence["audio_sha256"] = "b" * 64
    elif mutation == "unheard":
        evidence.update(audio_seen=False, audio_sha256=None, audio_intervals_seconds=[])
    elif mutation == "empty_ranges":
        evidence["audio_intervals_seconds"] = []
    else:
        evidence["audio_intervals_seconds"] = {"partial": [[0, 2]], "outside": [[0, 6]],
                                                 "nonfinite": [[0, float("inf")]], "bool": [[False, 5]]}[mutation]
    with pytest.raises(QualityError):
        validate_review(review, bound)


def test_audio_union_can_cover_full_clip_but_hash_is_not_authenticated_listening():
    bound = item()
    review = reviewed(bound)
    review["evidence"]["audio_intervals_seconds"] = [[2, 5], [0, 3]]
    validate_review(review, bound)
    result = summarize([bound], [review])
    assert any("not independent proof of listening" in text for text in result["limitations"])


@pytest.mark.parametrize("status,text", [("failed", ""), ("truncated", "Raise one arm"),
                                         ("unknown", ""), ("not_run", ""), ("ok", "Dance style: contemporary")])
def test_generation_failures_remain_in_denominator_without_made_up_semantic_scores(status, text):
    bound = item(generation_status=status, text=text)
    roster = {"r1": {"origin": "human", "judge_identity": None}}
    result = summarize([bound], [], expected_reviewer_ids=roster)
    dimension = result["origins"]["human"]["methods"]["A"]["action_clarity"]
    assert dimension["expected_items"] == 1
    assert dimension["rating_status_counts"] == {"generation_not_evaluable": 1}
    assert dimension["estimate_group_equal"] is None
    assert result["coverage"]["registered_items"] == 1
    assert result["coverage"]["reviewable_items"] == 0
    with pytest.raises(QualityError, match="cannot receive"):
        validate_review(reviewed(bound), bound)


def test_reference_fidelity_is_optional_separate_and_not_unique_quality():
    bound = item(text="Turn slowly and extend both arms.")
    review = reviewed(bound, score=5, reference_score=1)
    result = summarize([bound], [review])
    dimensions = result["origins"]["human"]["methods"]["A"]
    assert dimensions["action_clarity"]["estimate_group_equal"] == 5
    assert dimensions["reference_fidelity"]["estimate_group_equal"] == 1
    assert result["overall_quality_score"] is None
    review["evidence"]["reference_quotes"] = ["Not in this training reference"]
    with pytest.raises(QualityError, match="exact"):
        validate_review(review, bound)


def test_reference_fidelity_cannot_be_scored_without_reference():
    bound = item(reference_text=None)
    review = reviewed(bound)
    review["reference_fidelity"] = {"status": "rated", "score": 5, "reason": "Unsupported agreement claim"}
    with pytest.raises(QualityError, match="reference"):
        validate_review(review, bound)


def test_completed_review_cannot_hide_missing_audio_evidence_as_semantic_unknown():
    bound = item(audio_sha256=None)
    review = reviewed(bound)
    review["ratings"]["music_fit"] = {"status": "unknown", "score": None, "reason": "Did not hear audio."}
    with pytest.raises(QualityError, match="not_applicable"):
        validate_review(review, bound)


def test_missing_unknown_pending_and_failed_are_distinct_without_imputation():
    items = [item("c" + str(i), group="g" + str(i), generation_status="failed" if i == 3 else "ok") for i in range(4)]
    unknown = reviewed(items[0])
    unknown["ratings"]["action_clarity"] = {"status": "unknown", "score": None, "reason": "Ambiguous body relation."}
    pending = review_template(items[1], "r1")
    result = summarize(items, [unknown, pending], expected_reviewer_ids=["r1"])
    dim = result["origins"]["human"]["methods"]["A"]["action_clarity"]
    assert dim["expected_items"] == dim["expected_rating_slots"] == 4
    assert dim["rating_status_counts"] == {"unknown": 1, "pending": 1, "missing_review": 1, "generation_not_evaluable": 1}
    assert dim["estimate_group_equal"] is None
    assert len(result["coverage"]["missing_review_slots"]) == 1


def test_human_and_model_judge_are_not_pooled_or_anonymously_relabelled():
    bound = item()
    reviews = [reviewed(bound, "person", 5), reviewed(bound, "judge", 1, origin="model_judge", judge_identity="frozen-model-and-prompt-sha")]
    result = summarize([bound], reviews)
    assert result["origins"]["human"]["methods"]["A"]["action_clarity"]["estimate_group_equal"] == 5
    assert result["origins"]["model_judge"]["methods"]["A"]["action_clarity"]["estimate_group_equal"] == 1
    assert result["overall_quality_score"] is None


@pytest.mark.parametrize("mutation", ["origin", "judge"])
def test_same_reviewer_id_cannot_drift_origin_or_model(mutation):
    first, second = item(), item("c2", group="g2")
    one = reviewed(first, "judge", origin="model_judge", judge_identity="model1")
    two = reviewed(second, "judge", origin="human" if mutation == "origin" else "model_judge",
                   judge_identity=None if mutation == "origin" else "model2")
    with pytest.raises(QualityError, match="identity drift"):
        summarize([first, second], [one, two])


def test_unknown_roster_origin_is_reported_not_assumed_human():
    bound = item()
    result = summarize([bound], [], expected_reviewer_ids=["unseen-person"])
    assert result["coverage"]["unclassified_reviewer_ids"] == ["unseen-person"]
    assert len(result["coverage"]["missing_review_slots"]) == 1
    assert not result["origins"]["human"]["reviewer_ids"]
    assert not result["origins"]["model_judge"]["reviewer_ids"]


def test_pairing_group_equal_primary_and_no_false_independent_n():
    items, reviews = [], []
    for case_id, group, b_score in [("a", "large", 1), ("b", "large", 1), ("c", "large", 1), ("d", "small", 5)]:
        for method, score in (("A", 1), ("B", b_score)):
            bound = item(case_id, method, group)
            items.append(bound)
            reviews.append(reviewed(bound, score=score))
    original = copy.deepcopy((items, reviews))
    result = summarize(items, reviews, method_a="A", method_b="B", seed=19)
    pair = result["paired_comparisons"]["human"]["action_clarity"]
    assert pair["estimate_group_equal"] == 2
    assert pair["estimate_case_equal"] == 1
    assert pair["n_source_groups_observed"] == 2
    assert pair["paired_cases"] == 4
    assert pair["interval_group_equal"] is not None
    assert result == summarize(items, reviews, method_a="A", method_b="B", seed=19)
    assert (items, reviews) == original
    json.dumps(result, allow_nan=False)


def test_many_cases_in_one_source_have_no_independent_interval():
    items = [item("c" + str(i), group="one-recording") for i in range(8)]
    result = summarize(items, [reviewed(bound) for bound in items])
    dim = result["origins"]["human"]["methods"]["A"]["action_clarity"]
    assert dim["n_cases_rated"] == 8
    assert dim["n_source_groups_observed"] == 1
    assert dim["interval_group_equal"] is None


def test_paired_estimate_uses_only_same_named_raters_and_lists_omitted_cases():
    a, b = item(), item(method="B")
    missing = item("c2", "A", "g2")
    reviews = [reviewed(a, "r1", 1), reviewed(a, "r2", 5), reviewed(b, "r1", 3), reviewed(missing, "r1", 5)]
    result = summarize([a, b, missing], reviews, method_a="A", method_b="B")
    pair = result["paired_comparisons"]["human"]["action_clarity"]
    assert pair["estimate_group_equal"] == 2  # r2's unpaired A=5 cannot change it.
    assert pair["expected_cases"] == 2 and pair["paired_cases"] == 1
    assert pair["case_pairs"][0]["paired_reviewer_ids"] == ["r1"]
    assert pair["omitted_cases"] == [{"case_id": "c2", "reason": "method_not_registered"}]


def test_disjoint_reviewers_never_form_a_fake_pair():
    a, b = item(), item(method="B")
    result = summarize([a, b], [reviewed(a, "r1"), reviewed(b, "r2")], method_a="A", method_b="B")
    pair = result["paired_comparisons"]["human"]["action_clarity"]
    assert pair["paired_cases"] == 0
    assert pair["estimate_group_equal"] is None


def test_bad_registrations_duplicate_reviews_and_case_audio_drift_are_rejected():
    bound = item()
    with pytest.raises(QualityError, match="duplicate case"):
        summarize([bound, bound], [])
    review = reviewed(bound)
    with pytest.raises(QualityError, match="duplicate item/reviewer"):
        summarize([bound], [review, review])
    with pytest.raises(QualityError, match="binding differs"):
        summarize([bound, item(method="B", audio_sha256="b" * 64)], [])
    with pytest.raises(QualityError, match="unregistered"):
        summarize([bound], [reviewed(item("other"))])


@pytest.mark.parametrize("kwargs", [{"n_resamples": True}, {"n_resamples": 0}, {"seed": -1}, {"seed": True},
                                    {"confidence": float("nan")}, {"confidence": 1}, {"method_a": "A"},
                                    {"method_a": "A", "method_b": "A"}, {"expected_reviewer_ids": ["r1", "r1"]}])
def test_summary_config_rejected(kwargs):
    options = {"n_resamples": 10, **kwargs}
    with pytest.raises(QualityError):
        summarize_quality([item()], [], **options)


def test_cli_templates_and_missing_review_summary_are_offline_and_exclusive(tmp_path):
    items_path = tmp_path / "items.json"
    template_path, report_path = tmp_path / "review.json", tmp_path / "summary.json"
    items_path.write_text(json.dumps({"items": [item()]}), encoding="utf-8")
    main(["review-template", "--items", str(items_path), "--reviewer-id", "r1", "--out", str(template_path)])
    result = main(["summarize", "--items", str(items_path), "--reviews", str(template_path),
                   "--expected-reviewer-id", "r1", "--resamples", "10", "--out", str(report_path)])
    assert result["coverage"]["pending_reviews"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["overall_quality_score"] is None
    with pytest.raises(SystemExit):
        main(["review-template", "--items", str(items_path), "--reviewer-id", "r1", "--out", str(template_path)])


def test_cli_rejects_duplicate_json_keys_and_requires_model_identity(tmp_path):
    broken = tmp_path / "duplicate.json"
    broken.write_text('{"items": [], "items": []}', encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["review-template", "--items", str(broken), "--reviewer-id", "r1", "--out", str(tmp_path / "x")])
    valid = tmp_path / "items.json"
    valid.write_text(json.dumps([item()]), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["review-template", "--items", str(valid), "--reviewer-id", "judge", "--origin", "model_judge",
              "--out", str(tmp_path / "x")])


def test_python_module_rubric_entrypoint_needs_no_torch_or_network():
    result = subprocess.run([sys.executable, "-m", "captionlab.choreo_quality", "rubric"],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    assert json.loads(result.stdout)["composite_score"] is None
