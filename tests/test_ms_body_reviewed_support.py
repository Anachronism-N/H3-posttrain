"""Distinct support causes must not relabel joints or silently convert warnings."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
from ms_body_reviewed_support import REVIEW_KINDS, preprocess_reviewed_support, validated_review_mask
from motionscript_v2.body_events import extract_body_events
from test_ms_body_identity_support import raw, review_fixture


def test_geometric_uncertainty_is_a_separate_hard_cut_not_identity_relabeling():
    source = raw()
    source["points_m"][8, 2] = source["points_m"][8, 1] + [.002, .005, 0.]
    mask = np.zeros((21, 6), bool)
    mask[8, [1, 2]] = True
    before = copy.deepcopy(source)
    tracks, quality = preprocess_reviewed_support(source, mask, kind="geometry", smooth_seconds=0)
    assert not tracks["point_valid"][8, [1, 2]].any()
    assert not tracks["point_interpolated"][8, [1, 2]].any()
    assert np.isnan(tracks["points_m"][8, [1, 2]]).all()
    assert (tracks["raw_mask_reason"][8, [1, 2]] & 64).all()
    assert not (tracks["raw_mask_reason"][8, [1, 2]] & 32).any()
    assert "raw_geometry_excluded" in tracks and "raw_identity_excluded" not in tracks
    assert quality["raw_mask_reason_bits"]["geometry_support_unknown"] == 64
    assert "identity_continuity_unknown" not in quality["raw_mask_reason_bits"]
    assert quality["geometry_excluded_native_frames"] == 1
    assert not quality["geometry_exclusion_filled_by_interpolation"]
    assert not quality["identity_side_relabeling_performed"]
    assert tracks["point_valid"][:, [0, 3, 4, 5]].all()
    np.testing.assert_array_equal(source["points_m"], before["points_m"])
    np.testing.assert_array_equal(tracks["raw_points_m"], source["points_m"])


def test_no_review_mask_does_not_automatically_turn_geometry_warning_into_exclusion():
    source = raw()
    source["points_m"][8, 2] = source["points_m"][8, 1] + [.002, .005, 0.]
    tracks, _ = preprocess_reviewed_support(source, np.zeros((21, 6), bool), kind="geometry", smooth_seconds=0)
    assert tracks["point_valid"].all()
    assert not tracks["raw_geometry_excluded"].any()


def test_geometry_review_blocks_smoothing_leakage_and_preserves_opposite_side():
    source = raw()
    source["points_m"][8, 2] += [10., 10., 0.]
    mask = np.zeros((21, 6), bool)
    mask[8, 2] = True
    tracks, quality = preprocess_reviewed_support(source, mask, kind="geometry", smooth_seconds=.0625)
    assert np.linalg.norm(tracks["points_m"][7, 2] - source["points_m"][7, 2]) < .1
    assert np.linalg.norm(tracks["points_m"][9, 2] - source["points_m"][9, 2]) < .1
    assert not quality["geometry_exclusion_crossed_by_smoothing"]
    assert tracks["point_valid"][:, 3:].all()
    events = extract_body_events(tracks)
    assert all(not (event["start"] < .4 and event["end"] > .4) for event in events
               if "left_wrist" in event["evidence"]["required_point_names"])


def test_geometry_gap_hidden_by_coarse_grid_still_has_independent_segment_ids():
    source = raw(count=6, fps=100)
    mask = np.zeros((6, 6), bool)
    mask[1:4, 2] = True
    tracks, _ = preprocess_reviewed_support(source, mask, kind="geometry", fps=20, smooth_seconds=0)
    assert tracks["point_valid"][:, 2].all()
    assert tracks["point_segment_id"][:, 2].tolist() == [0, 1]
    assert extract_body_events(tracks) == []


def test_geometry_source_review_requires_matching_kind_and_reason(tmp_path):
    source, review = review_fixture(tmp_path)
    schema, reason, _, _ = REVIEW_KINDS["geometry"]
    review["schema"] = schema
    review["exclusions"][0]["reason_code"] = reason
    mask = validated_review_mask(review, raw(), source, kind="geometry")
    assert mask.sum() == 2
    with pytest.raises(ValueError, match="exact source"):
        validated_review_mask(review, raw(), source, kind="identity")
    review["exclusions"][0]["reason_code"] = "anatomical_identity_continuity_unknown"
    with pytest.raises(ValueError, match="declared local review kind"):
        validated_review_mask(review, raw(), source, kind="geometry")


def test_unknown_support_cause_is_rejected():
    with pytest.raises(ValueError, match="identity or geometry"):
        preprocess_reviewed_support(raw(), np.zeros((21, 6), bool), kind="guess")
