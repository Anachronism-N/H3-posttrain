import copy

import pytest

from captionlab.choreo_quality import make_review_item, review_template
from captionlab.metric_validation import quadratic_kappa, spearman, validate_metric


def fixture():
    items, reviews = [], []
    for index in range(5):
        item = make_review_item(str(index), "g" + str(index), "base", "Raises the arms.")
        items.append(item)
        for reviewer in ("h1", "h2", "j"):
            row = review_template(item, reviewer, origin="human" if reviewer != "j" else "model_judge",
                                  judge_identity=None if reviewer != "j" else "frozen-judge")
            row["review_status"] = "completed"
            row["evidence"].update(text_seen=True, text_quotes=[item["text"]], evidence_scope="Text read.")
            for dimension in ("action_clarity", "temporal_organization"):
                row["ratings"][dimension] = {"status": "rated", "score": index + 1, "reason": "Test fixture."}
            row["ratings"]["music_fit"] = {"status": "not_applicable", "score": None, "reason": "No audio."}
            reviews.append(row)
    return items, reviews


@pytest.mark.parametrize("a,b,expected", [([1, 2, 3], [1, 2, 3], 1), ([1, 2, 3], [3, 2, 1], -1),
    ([1, 1, 3], [2, 2, 4], 1), ([], [], None), ([1], [2], None), ([1, 1], [2, 3], None)])
def test_rank_ties_and_degenerate(a, b, expected):
    result = spearman(a, b)
    assert result is None if expected is None else result == pytest.approx(expected)


def test_kappa_scale_and_degenerate():
    assert quadratic_kappa([1, 3, 5], [1, 3, 5]) == 1
    assert quadratic_kappa([1, 5], [5, 1]) == -1
    assert quadratic_kappa([2, 2], [2, 2]) is None
    assert quadratic_kappa([1], [2]) is None
    assert quadratic_kappa([1, 1], [2, 2]) == 0
    with pytest.raises(ValueError):
        quadratic_kappa([True], [1])
    with pytest.raises(ValueError):
        quadratic_kappa([1], [1, 2])
    with pytest.raises(ValueError):
        spearman([1], [])


def test_bound_reviews_full_coverage_and_deterministic_bootstrap():
    items, reviews = fixture()
    args = dict(humans=["h1", "h2"], judge="j", n_resamples=60)
    result = validate_metric(items, reviews, **args)
    assert result == validate_metric(items, reviews, **args)
    clarity = result["dimensions"]["action_clarity"]
    assert clarity["human_pair_coverage"] == 1
    assert clarity["human_quadratic_kappa"]["estimate"] == 1
    assert clarity["judge_vs_human_mean_spearman"]["estimate"] == 1
    assert result["dimensions"]["music_fit"]["judge_human_pair_coverage"] == 0
    assert result["dimensions"]["music_fit"]["human_quadratic_kappa"]["estimate"] is None


def test_missing_reviews_never_imputed():
    items, reviews = fixture()
    reviews = [row for row in reviews if not (row["item_sha256"] == items[0]["item_sha256"] and row["reviewer_id"] == "h2")]
    report = validate_metric(items, reviews, humans=["h1", "h2"], judge="j", n_resamples=10)
    dim = report["dimensions"]["action_clarity"]
    assert dim["registered_outputs"] == 5
    assert dim["human_pair_coverage"] == .8
    assert dim["judge_human_pair_coverage"] == .8


def test_failed_output_stays_denominator():
    items, reviews = fixture()
    items.append(make_review_item("failure", "gf", "base", "", generation_status="unknown"))
    report = validate_metric(items, reviews, humans=["h1", "h2"], n_resamples=10)
    assert report["dimensions"]["action_clarity"]["human_pair_coverage"] == 5 / 6


def test_one_source_does_not_invent_confidence_interval():
    items, reviews = fixture()
    changed_items, changed_reviews = [], []
    for item in items:
        replacement = make_review_item(item["case_id"], "one-source", item["method_id"], item["text"])
        changed_items.append(replacement)
        for row in reviews:
            if row["item_sha256"] == item["item_sha256"]:
                row = copy.deepcopy(row)
                row["item_sha256"] = replacement["item_sha256"]
                changed_reviews.append(row)
    report = validate_metric(changed_items, changed_reviews, humans=["h1", "h2"], n_resamples=10)
    metric = report["dimensions"]["action_clarity"]["human_quadratic_kappa"]
    assert metric["estimate"] == 1
    assert metric["interval_95"] is None
    assert metric["source_groups"] == 1


@pytest.mark.parametrize("change", ["duplicate", "forged_quote", "model_as_human", "human_as_judge"])
def test_invalid_evidence_or_identity_rejected(change):
    items, reviews = fixture()
    if change == "duplicate":
        reviews.append(reviews[0])
    elif change == "forged_quote":
        reviews[0]["evidence"]["text_quotes"] = ["never occurred"]
    elif change == "model_as_human":
        reviews[0].update(origin="model_judge", judge_identity="j")
    else:
        reviews[2].update(origin="human", judge_identity=None)
    with pytest.raises(ValueError):
        validate_metric(items, reviews, humans=["h1", "h2"], judge="j", n_resamples=10)


@pytest.mark.parametrize("humans", [["h1"], ["h1", "h1"], ["h1", ""]])
def test_invalid_roster(humans):
    with pytest.raises(ValueError):
        validate_metric(*fixture(), humans=humans, n_resamples=10)
