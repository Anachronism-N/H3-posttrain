"""Review-bound quality metrics; empty or unknown evidence never passes."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
from ms_body_quality import check_occurrence_coverage, evaluate_review, metrics, numeric_gates_pass


def data():
    event = {"id": "e1", "start": .2, "end": 1.2, "body_part": "left_elbow", "action": "bends",
             "evidence": {"measurement_name": "joint_interior_angle"}}
    facts = {"events": [event], "summary": {"source_events": [copy.deepcopy(event)],
        "selected_events": [copy.deepcopy(event)], "groups": [{"source_ids": ["e1"], "occurrences": [{"event_id": "e1"}]}]}}
    t = np.arange(31) / 20
    arrays = {"tracks": {"time": t}, "signals": {"body_signal_names": np.array(["left_elbow.joint_interior_angle"]),
                                                "body_signal_valid": np.ones((31, 1), bool)}}
    row = {"id": "case", "video_sha256": "videosha", "media_probe": {"time": t.tolist()}}
    review = {"schema": "native-body-review-template-v1", "id": "case", "video_sha256": "videosha", "facts_sha256": "factssha",
        "review_completed": True, "reference_built_from_rgb_before_reading_candidates": True,
        "reviewer": "synthetic-test-review-not-real-human-evidence", "reviewer_type": "assistant",
        "absolute_3d_scale_or_depth_certified": False,
        "reference_occurrences": [{"id": "r1", "start": .2, "end": 1.2, "body_part": "left_elbow",
                                   "action": "bends", "measurement_name": "joint_interior_angle", "decidable": True}],
        "event_decisions": [{"event_id": "e1", "verdict": "supported", "reference_id": "r1", "reason": "synthetic contract only"}],
        "decidable_intervals": [{"body_part": "left_elbow", "measurement_name": "joint_interior_angle", "start": .2, "end": 1.2}],
        "critical_source_time_mapping_errors": [], "incorrect_removed_occurrence_ids": []}
    return facts, arrays, row, review


def test_review_metrics_are_bound_to_actual_reference_and_support():
    facts, arrays, row, review = data()
    result = evaluate_review(review, facts, arrays, "factssha", row)
    assert result["tp"] == 1 and result["fp"] == result["fn"] == result["unknown"] == 0
    assert result["support_valid"] == result["support_total"] == 21
    values = metrics([result])
    assert values["boundary_error_p95_seconds"] == 0
    assert numeric_gates_pass(values)


def test_visible_abstention_and_unknown_are_not_true_positives():
    facts, arrays, row, review = data()
    review["event_decisions"][0]["verdict"] = "unknown"
    arrays["signals"]["body_signal_valid"][:] = False
    result = evaluate_review(review, facts, arrays, "factssha", row)
    assert result["tp"] == 0 and result["unknown"] == 1 and result["fn"] == 1
    values = metrics([result])
    assert values["false_positive_fraction"] is None
    assert values["false_negative_fraction"] == 1
    assert values["decidable_support_fraction"] == 0
    assert not numeric_gates_pass(values)


def test_empty_or_uncompleted_reviews_do_not_create_green_quality():
    assert not numeric_gates_pass(metrics([]))
    facts, arrays, row, review = data()
    review["review_completed"] = False
    assert evaluate_review(review, facts, arrays, "factssha", row) is None


@pytest.mark.parametrize("mutation", [
    lambda review: review.update(video_sha256="wrong"),
    lambda review: review.update(facts_sha256="wrong"),
    lambda review: review.update(event_decisions=[]),
    lambda review: review["reference_occurrences"][0].update(body_part="left_wrist"),
    lambda review: review.update(absolute_3d_scale_or_depth_certified=True),
])
def test_wrong_binding_primitive_or_unsupported_scale_claim_is_rejected(mutation):
    facts, arrays, row, review = data()
    mutation(review)
    with pytest.raises(ValueError):
        evaluate_review(review, facts, arrays, "factssha", row)


def test_grouping_must_cover_every_source_occurrence():
    facts, _, _, _ = data()
    check_occurrence_coverage(facts)
    facts["summary"]["groups"][0]["occurrences"] = []
    with pytest.raises(ValueError, match="disappeared"):
        check_occurrence_coverage(facts)
