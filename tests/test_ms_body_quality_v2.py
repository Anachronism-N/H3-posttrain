"""Adversarial verifier contracts; synthetic reviewers are not real human evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
from ms_body_quality_v2 import GEOMETRY_CHECKS, SCHEMA, contract_review_ids, evaluate_review, review_disagreements
from ms_body_quality import metrics, numeric_gates_pass
from test_ms_body_quality import data as original_fixture


def fixture():
    facts, arrays, row, review = original_fixture()
    facts["events"][0]["direction"] = "decreasing_joint_angle"
    facts["summary"]["source_events"] = copy.deepcopy(facts["events"])
    facts["summary"]["selected_events"] = copy.deepcopy(facts["events"])
    facts["signal_metadata"] = {"signals": [{"body_part": "left_elbow", "measurement_name": "joint_interior_angle"}]}
    review["schema"] = SCHEMA
    review["reference_inventory_complete"] = True
    review["reference_occurrences"][0]["direction"] = "decreasing_joint_angle"
    review["channel_assessments"] = [{"body_part": "left_elbow", "measurement_name": "joint_interior_angle",
        "reference_inventory_complete": True,
        "intervals": [{"start": 0., "end": 1.5, "observability": "decidable", "reason": "synthetic full-channel contract"}]}]
    review["geometry_checks"] = [{"name": name, "verdict": "pass", "reason": "synthetic geometry contract only"}
                                 for name in GEOMETRY_CHECKS]
    return facts, arrays, row, review


def test_one_to_one_review_keeps_reference_misses_and_correct_denominators():
    facts, arrays, row, review = fixture()
    result = evaluate_review(review, facts, arrays, "factssha", row)
    assert result["tp"] == result["references"] == 1
    assert result["fn"] == result["fp"] == result["unknown"] == 0
    assert result["support_total"] == result["support_valid"] == 31
    assert result["unknown_operator_fraction"] == 0
    assert result["geometry_checks_passed"]
    assert numeric_gates_pass(metrics([result]))


def test_two_candidates_cannot_share_one_reference_to_inflate_precision():
    facts, arrays, row, review = fixture()
    second = copy.deepcopy(facts["events"][0])
    second["id"] = "e2"
    facts["events"].append(second)
    facts["summary"]["source_events"].append(copy.deepcopy(second))
    facts["summary"]["selected_events"].append(copy.deepcopy(second))
    facts["summary"]["groups"].append({"source_ids": ["e2"], "occurrences": [{"event_id": "e2"}]})
    review["event_decisions"].append({"event_id": "e2", "verdict": "supported", "reference_id": "r1", "reason": "synthetic duplicate"})
    with pytest.raises(ValueError, match="multiple candidate"):
        evaluate_review(review, facts, arrays, "factssha", row)
    review["event_decisions"][-1]["verdict"] = "false_positive"
    result = evaluate_review(review, facts, arrays, "factssha", row)
    assert result["tp"] == result["fp"] == 1
    assert metrics([result])["false_positive_fraction"] == .5


def test_copied_reference_under_a_new_id_cannot_increase_denominators():
    facts, arrays, row, review = fixture()
    repeated = copy.deepcopy(review["reference_occurrences"][0])
    repeated["id"] = "r2"
    review["reference_occurrences"].append(repeated)
    with pytest.raises(ValueError, match="Duplicated reference"):
        evaluate_review(review, facts, arrays, "factssha", row)


def test_wrong_direction_cannot_pass_merely_because_action_and_operator_match():
    facts, arrays, row, review = fixture()
    review["reference_occurrences"][0]["direction"] = "increasing_joint_angle"
    with pytest.raises(ValueError, match="direction"):
        evaluate_review(review, facts, arrays, "factssha", row)


def test_omitted_operator_assessment_is_rejected_instead_of_shrinking_denominator():
    facts, arrays, row, review = fixture()
    facts["signal_metadata"]["signals"].append({"body_part": "right_elbow", "measurement_name": "joint_interior_angle"})
    arrays["signals"]["body_signal_names"] = np.array(["left_elbow.joint_interior_angle", "right_elbow.joint_interior_angle"])
    arrays["signals"]["body_signal_valid"] = np.ones((31, 2), bool)
    with pytest.raises(ValueError, match="Every measured channel"):
        evaluate_review(review, facts, arrays, "factssha", row)


@pytest.mark.parametrize("intervals", [
    [{"start": .1, "end": 1.5, "observability": "decidable", "reason": "missing start"}],
    [{"start": 0., "end": 1.4, "observability": "decidable", "reason": "missing end"}],
    [{"start": 0., "end": .7, "observability": "decidable", "reason": "first"},
     {"start": .8, "end": 1.5, "observability": "decidable", "reason": "hidden gap"}],
])
def test_unreviewed_time_holes_are_rejected(intervals):
    facts, arrays, row, review = fixture()
    review["channel_assessments"][0]["intervals"] = intervals
    with pytest.raises(ValueError, match="Channel observability"):
        evaluate_review(review, facts, arrays, "factssha", row)


def test_incomplete_reference_inventory_does_not_produce_metrics():
    facts, arrays, row, review = fixture()
    review["reference_inventory_complete"] = False
    assert evaluate_review(review, facts, arrays, "factssha", row) is None


def test_unknown_source_support_cannot_back_a_supported_reference():
    facts, arrays, row, review = fixture()
    review["channel_assessments"][0]["intervals"][0]["observability"] = "unknown"
    with pytest.raises(ValueError, match="unknown source support"):
        evaluate_review(review, facts, arrays, "factssha", row)
    review["reference_occurrences"][0]["decidable"] = False
    review["event_decisions"][0]["verdict"] = "unknown"
    result = evaluate_review(review, facts, arrays, "factssha", row)
    assert result["unknown_operator_fraction"] == 1
    assert result["tp"] == 0 and result["unknown"] == 1
    assert not numeric_gates_pass(metrics([result]))


def test_visible_motion_abstention_counts_as_miss_and_support_loss():
    facts, arrays, row, review = fixture()
    review["event_decisions"][0]["verdict"] = "unknown"
    arrays["signals"]["body_signal_valid"][:] = False
    result = evaluate_review(review, facts, arrays, "factssha", row)
    assert result["fn"] == result["unknown"] == 1
    assert result["support_valid"] == 0 and result["support_total"] == 31


def test_geometry_checks_cannot_be_missing_or_implicitly_passed():
    facts, arrays, row, review = fixture()
    review["geometry_checks"] = []
    with pytest.raises(ValueError, match="Every geometry"):
        evaluate_review(review, facts, arrays, "factssha", row)
    review["geometry_checks"] = [{"name": name, "verdict": "unknown", "reason": "not reviewed"} for name in GEOMETRY_CHECKS]
    assert not evaluate_review(review, facts, arrays, "factssha", row)["geometry_checks_passed"]


def test_independent_boundary_and_reference_inventory_disagreement_is_preserved():
    _, _, _, main = fixture()
    human = copy.deepcopy(main)
    human["reviewer_type"] = "human"
    human["reference_occurrences"][0]["id"] = "independent_r1"
    human["event_decisions"][0]["reference_id"] = "independent_r1"
    assert review_disagreements(main, human) == []
    human["reference_occurrences"][0]["start"] = .6
    assert any(item["kind"] == "reference_boundary" for item in review_disagreements(main, human))
    human["reference_occurrences"].append({"id": "r2", "body_part": "left_elbow", "measurement_name": "joint_interior_angle",
        "action": "extends", "direction": "increasing_joint_angle", "start": 1.2, "end": 1.5, "decidable": True})
    assert any(item["kind"] == "additional_independent_references" for item in review_disagreements(main, human))


def test_same_verdict_with_different_reference_assignment_is_a_disagreement():
    _, _, _, main = fixture()
    other_reference = copy.deepcopy(main["reference_occurrences"][0])
    other_reference.update(id="r2", start=.7, end=1.5)
    main["reference_occurrences"].append(other_reference)
    human = copy.deepcopy(main)
    human["event_decisions"][0]["reference_id"] = "r2"
    assert any(item["kind"] == "supported_reference_assignment" for item in review_disagreements(main, human))


def test_independent_observability_disagreement_is_not_hidden_by_identical_verdicts():
    _, _, _, main = fixture()
    human = copy.deepcopy(main)
    human["channel_assessments"][0]["intervals"][0]["observability"] = "unknown"
    assert any(item["kind"] == "source_observability" for item in review_disagreements(main, human))


def contract_fixture(tmp_path):
    supplement = {"schema": "native-body-supplemental-geometry-review-v1", "original_independent_panel_sha256": "panelsha",
                  "cases": [{"id": "base", "source_video_sha256": "basesha"},
                            {"id": "extra", "source_video_sha256": "extrasha"}]}
    path = tmp_path / "supplement.json"
    path.write_text(json.dumps(supplement))
    contract = {"schema": "native-body-independent-review-contract-v1", "selection_sha256": "selectionsha",
                "original_independent_panel_sha256": "panelsha", "required_independent_ids": ["base", "extra"],
                "supplemental_panels": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return contract_path, path, contract, supplement


def test_known_supplemental_review_is_required_without_double_counting_base_sources(tmp_path):
    path, _, _, _ = contract_fixture(tmp_path)
    ids, digest, sources = contract_review_ids(path, {"base"}, {"base": "basesha", "extra": "extrasha"}, "selectionsha", "panelsha")
    assert ids == {"base", "extra"}
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(sources) == 1


def test_review_contract_cannot_omit_a_known_extra_dispute(tmp_path):
    path, _, contract, _ = contract_fixture(tmp_path)
    contract["required_independent_ids"] = ["base"]
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="omitted"):
        contract_review_ids(path, {"base"}, {"base": "basesha", "extra": "extrasha"}, "selectionsha", "panelsha")


def test_altered_supplement_is_rejected(tmp_path):
    path, supplement_path, _, _ = contract_fixture(tmp_path)
    supplement_path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        contract_review_ids(path, {"base"}, {"base": "basesha", "extra": "extrasha"}, "selectionsha", "panelsha")


@pytest.mark.parametrize("field,value", [("id", "unregistered"), ("source_video_sha256", "wrong")])
def test_supplemental_case_must_match_frozen_source_id_and_video(tmp_path, field, value):
    path, supplement_path, contract, supplement = contract_fixture(tmp_path)
    supplement["cases"][1][field] = value
    supplement_path.write_text(json.dumps(supplement))
    contract["supplemental_panels"][0]["sha256"] = hashlib.sha256(supplement_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="not bound"):
        contract_review_ids(path, {"base"}, {"base": "basesha", "extra": "extrasha"}, "selectionsha", "panelsha")
