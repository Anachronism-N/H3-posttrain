"""Identity uncertainty is a hard local support boundary, never an invented side."""
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
from ms_body_identity_support import SCHEMA, preprocess_identity_support, review_mask
from motionscript_v2.body_events import extract_body_events
from motionscript_v2.landmarks import preprocess_landmarks


def raw(count=21, fps=20):
    theta = np.deg2rad(np.linspace(40, 130, count))
    p = np.broadcast_to(np.array([[0., -.3, 3.], [0., 0., 3.], [0., 0., 3.],
                                  [1., -.3, 3.], [1., 0., 3.], [1.3, 0., 3.]]), (count, 6, 3)).copy()
    p[:, 2, 0] = .3 * np.sin(theta)
    p[:, 2, 1] = -.3 * np.cos(theta)
    return {"names": np.array(["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]),
            "points_m": p, "valid": np.ones((count, 6), bool), "points2d_px": np.full((count, 6, 2), 20.),
            "frame_size": np.array([100, 100]), "timestamps": np.arange(count) / fps, "fps": fps,
            "frame_indices": np.arange(count), "pts": np.arange(count), "time_base": np.array([1, fps]),
            "coordinate_frame": np.array("camera_xright_ydown_zforward"), "units": np.array("m")}


def test_explicit_identity_gap_is_not_filled_even_when_ordinary_short_gap_would_be():
    source = raw()
    ordinary = copy.deepcopy(source)
    ordinary["valid"][8:10, 2] = False
    ordinary_tracks, _ = preprocess_landmarks(ordinary, max_gap_seconds=.2, smooth_seconds=0)
    assert ordinary_tracks["point_valid"][8:10, 2].all()
    mask = np.zeros((21, 6), bool)
    mask[8:10, 2] = True
    before = copy.deepcopy(source)
    tracks, quality = preprocess_identity_support(source, mask, max_gap_seconds=.2, smooth_seconds=0)
    assert not tracks["point_valid"][8:10, 2].any()
    assert not tracks["point_observed"][8:10, 2].any()
    assert not tracks["point_interpolated"][8:10, 2].any()
    assert np.isnan(tracks["points_m"][8:10, 2]).all()
    assert tracks["point_segment_id"][7, 2] != tracks["point_segment_id"][10, 2]
    assert tracks["point_valid"][:, [0, 1, 3, 4, 5]].all()
    assert not quality["identity_exclusion_filled_by_interpolation"]
    assert not quality["identity_side_relabeling_performed"]
    np.testing.assert_array_equal(source["points_m"], before["points_m"])
    np.testing.assert_array_equal(tracks["raw_points_m"], source["points_m"])


def test_bad_identity_samples_do_not_leak_into_neighbor_smoothing():
    source = raw()
    source["points_m"][8:10, 2] += [10., 10., 0.]
    mask = np.zeros((21, 6), bool)
    mask[8:10, 2] = True
    tracks, _ = preprocess_identity_support(source, mask, smooth_seconds=.0625)
    assert np.linalg.norm(tracks["points_m"][7, 2] - source["points_m"][7, 2]) < .1
    assert np.linalg.norm(tracks["points_m"][10, 2] - source["points_m"][10, 2]) < .1
    assert np.isnan(tracks["points_m"][8:10, 2]).all()
    baseline, _ = preprocess_landmarks(source, smooth_seconds=.0625)
    np.testing.assert_array_equal(tracks["points_m"][:, 5], baseline["points_m"][:, 5])


def test_hidden_identity_gap_still_splits_a_coarse_output_grid():
    source = raw(count=6, fps=100)
    mask = np.zeros((6, 6), bool)
    mask[1:4, 2] = True
    tracks, _ = preprocess_identity_support(source, mask, fps=20, smooth_seconds=0)
    np.testing.assert_allclose(tracks["time"], [0, .05])
    assert tracks["point_valid"][:, 2].all()
    assert tracks["point_observed"][:, 2].all()
    assert tracks["point_segment_id"][:, 2].tolist() == [0, 1]
    assert extract_body_events(tracks) == []


def test_events_never_connect_across_explicit_identity_exclusions():
    source = raw()
    mask = np.zeros((21, 6), bool)
    mask[8:10, :3] = True
    tracks, _ = preprocess_identity_support(source, mask, smooth_seconds=0)
    events = extract_body_events(tracks)
    assert events
    assert all(not (event["start"] < .4 and event["end"] > .45) for event in events)
    assert tracks["point_valid"][:, 3:].all()


def test_whole_missing_side_does_not_drop_opposite_side_support():
    source = raw()
    mask = np.zeros((21, 6), bool)
    mask[:, 2] = True
    tracks, quality = preprocess_identity_support(source, mask)
    assert not tracks["point_valid"][:, 2].any()
    assert tracks["point_valid"][:, 3:].all()
    assert quality["identity_excluded_native_frames"] == 21


def review_fixture(tmp_path):
    evidence = tmp_path / "evidence.jpg"
    evidence.write_bytes(b"synthetic-evidence-binding-only")
    source = {"id": "fixture", "source_sha256": {"video": "videosha"}, "raw_sha256": "rawsha"}
    review = {"schema": SCHEMA, "id": "fixture", "video_sha256": "videosha", "raw_sha256": "rawsha",
              "reviewer": "synthetic-contract-not-a-human-review", "reviewer_type": "assistant",
              "exclusions": [{"point_names": ["left_wrist"], "frame_indices": [8, 9],
                 "reason_code": "anatomical_identity_continuity_unknown", "reason": "synthetic support contract",
                 "evidence": [{"path": str(evidence), "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]}]}
    return source, review


def test_exclusion_review_is_bound_to_native_frames_and_source_evidence(tmp_path):
    source, review = review_fixture(tmp_path)
    mask = review_mask(review, raw(), source)
    assert mask.shape == (21, 6) and mask.dtype == np.bool_
    assert np.flatnonzero(mask[:, 2]).tolist() == [8, 9]
    assert mask.sum() == 2


@pytest.mark.parametrize("mutation", [
    lambda review: review.update(video_sha256="wrong"),
    lambda review: review.update(raw_sha256="wrong"),
    lambda review: review["exclusions"][0].update(point_names=["invented_joint"]),
    lambda review: review["exclusions"][0].update(frame_indices=[True]),
    lambda review: review["exclusions"][0].update(frame_indices=[999]),
    lambda review: review["exclusions"][0]["evidence"][0].update(sha256="wrong"),
])
def test_wrong_review_bindings_or_frame_names_are_rejected(tmp_path, mutation):
    source, review = review_fixture(tmp_path)
    mutation(review)
    with pytest.raises(ValueError):
        review_mask(review, raw(), source)
