"""Synthetic checks of coordinate conventions, temporal support and rotations."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts_omnidance.exp.ms.motionscript_v2.coordinates import preprocess


def template():
    joints = np.zeros((52, 3))
    joints[:22] = [
        [0, 0, 0], [.1, -.1, 0], [-.1, -.1, 0], [0, .1, 0],
        [.1, -.5, 0], [-.1, -.5, 0], [0, .3, 0], [.1, -.9, 0],
        [-.1, -.9, 0], [0, .5, 0], [.1, -1, .1], [-.1, -1, .1],
        [0, .65, 0], [.1, .6, 0], [-.1, .6, 0], [0, .85, 0],
        [.2, .6, 0], [-.2, .6, 0], [.5, .4, 0], [-.5, .4, 0],
        [.7, .25, 0], [-.7, .25, 0],
    ]
    parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
                       + [20] * 15 + [21] * 15)
    for index in range(22, 52):
        wrist = 20 if index < 37 else 21
        joints[index] = joints[wrist] + [.03 * (index % 3 + 1), 0, .02]
    return dict(J_template=joints, parents=parents)


def nlf(*, count=21, source_fps=20, yaw=None, pelvis=None, timestamps=None):
    rest = template()["J_template"][:24]
    yaw = np.zeros(count) if yaw is None else np.asarray(yaw)
    rotation = Rotation.from_euler("y", yaw[:, None], degrees=True).as_matrix()
    camera_axes = np.diag([1., -1., -1.])
    pose = np.zeros((count, 72))
    pose[:, :3] = Rotation.from_matrix(camera_axes @ rotation).as_rotvec()
    pelvis = np.tile([0., 1., -3.], (count, 1)) if pelvis is None else np.asarray(pelvis)
    joints_y_up = np.einsum("tij,kj->tki", rotation, rest) + pelvis[:, None]
    result = dict(pose=pose, j3d=joints_y_up @ camera_axes.T,
                  valid=np.ones(count, dtype=bool), fps=source_fps)
    if timestamps is not None:
        result["timestamps"] = np.asarray(timestamps)
    return result


def test_static_body_masks_layout_and_metadata():
    arrays, metadata = preprocess(nlf(), template(), smooth_seconds=0)
    assert arrays["body"].shape == (21, 22, 3)
    assert arrays["coords"].shape == (21, 54, 3)
    assert arrays["root_rotation"].shape == (21, 3, 3)
    for key in ("body", "coords", "root_rotation", "global_joints", "translation", "yaw_deg"):
        assert arrays[key].dtype == np.float32
    for key in ("time", "source_time"):
        assert arrays[key].dtype == np.float64
    for key in ("valid", "joint_valid", "observed", "interpolated", "joint_observed",
                "joint_interpolated", "source_valid", "source_detection_valid", "source_joint_valid"):
        assert arrays[key].dtype == np.bool_
    assert arrays["segment_id"].dtype == np.int32
    assert arrays["joint_segment_id"].dtype == np.int32
    assert arrays["joint_segment_id"].shape == (21, 22)
    assert arrays["valid"].all() and arrays["joint_valid"].all()
    assert arrays["observed"].all() and not arrays["interpolated"].any()
    np.testing.assert_allclose(arrays["body"][:, 0, 1], 1)
    np.testing.assert_allclose(arrays["body"][:, [10, 11], 1], 0, atol=1e-12)
    np.testing.assert_allclose(arrays["coords"][:, :22], arrays["body"])
    np.testing.assert_allclose(arrays["coords"][:, 53], arrays["translation"])
    assert metadata["hand_fingers_observed"] is False
    assert metadata["uncertainty_gate_enabled"] is False
    json.dumps(metadata, allow_nan=False)


def test_slerp_and_rotation_smoothing_cross_180_without_false_reverse():
    angles = np.linspace(170, 190, 21)
    arrays, _ = preprocess(nlf(yaw=angles), template(), smooth_seconds=.0625)
    yaw = arrays["yaw_deg"]
    assert yaw[0] == pytest.approx(0)
    assert np.all(np.diff(yaw) > 0)
    assert yaw[-1] > 18
    assert np.abs(np.diff(arrays["coords"][:, 52, 2])).max() < 2
    rotation = arrays["root_rotation"]
    np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rotation.shape), atol=2e-6)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=2e-6)
    # Per-frame heading removal leaves a stable body, while global joints rotate.
    np.testing.assert_allclose(arrays["body"], np.broadcast_to(arrays["body"][0], arrays["body"].shape), atol=2e-6)
    assert np.linalg.norm(arrays["global_joints"][-1, 20] - arrays["global_joints"][0, 20]) > .2


def test_gap_rotations_take_short_arc_instead_of_axis_angle_midpoint():
    record = nlf(count=3, yaw=[179, 180, 181])
    record["valid"][1] = False
    record["pose"][1] = np.nan
    record["j3d"][1] = np.nan
    arrays, _ = preprocess(record, template(), max_gap_seconds=.05, smooth_seconds=0)
    assert arrays["valid"].all()
    assert arrays["observed"].tolist() == [True, False, True]
    assert arrays["interpolated"].tolist() == [False, True, False]
    np.testing.assert_allclose(arrays["yaw_deg"], [0, 1, 2], atol=1e-10)


def test_long_gap_and_missing_edges_are_not_filled_or_smoothed_across():
    record = nlf(count=21, yaw=np.r_[np.zeros(10), np.full(11, 90.)])
    record["valid"][[0, 1, 7, 8, 9, 10, 11, 19, 20]] = False
    arrays, metadata = preprocess(record, template(), max_gap_seconds=.1, smooth_seconds=.1)
    np.testing.assert_array_equal(arrays["valid"], record["valid"])
    assert np.isnan(arrays["body"][~arrays["valid"]]).all()
    assert np.isnan(arrays["root_rotation"][~arrays["valid"]]).all()
    np.testing.assert_allclose(arrays["yaw_deg"][2:7], 0, atol=1e-10)
    np.testing.assert_allclose(arrays["yaw_deg"][12:19], 90, atol=1e-10)
    assert metadata["long_gap_count"] == 1
    assert metadata["output_valid_segments"] == 2


def test_exact_output_period_and_source_observation_flags():
    arrays, _ = preprocess(nlf(count=81, source_fps=16), template(), smooth_seconds=0)
    assert len(arrays["time"]) == 101
    np.testing.assert_allclose(np.diff(arrays["time"]), .05, rtol=0, atol=1e-15)
    assert arrays["time"][-1] <= arrays["source_time"][-1]
    assert arrays["observed"].sum() == 21  # Every 0.25 seconds both grids coincide.
    assert arrays["valid"].all()
    np.testing.assert_array_equal(arrays["interpolated"], arrays["valid"] & ~arrays["observed"])
    generated, _ = preprocess(nlf(count=124, source_fps=24), template(), smooth_seconds=0)
    assert len(generated["time"]) == 103
    assert generated["time"][-1] == pytest.approx(5.1)
    np.testing.assert_allclose(np.diff(generated["time"]), .05, rtol=0, atol=1e-15)


def test_explicit_timestamps_drive_interpolation_and_keep_time_origin():
    timestamps = np.array([.2, .3, .5])
    pelvis = np.column_stack((timestamps, np.ones(3), np.full(3, -3.)))
    record = nlf(count=3, source_fps=10, pelvis=pelvis, timestamps=timestamps)
    arrays, metadata = preprocess(record, template(), fps=20, max_gap_seconds=.15, smooth_seconds=0)
    np.testing.assert_allclose(arrays["time"][:3], [.2, .25, .3])
    np.testing.assert_allclose(arrays["translation"][:, 0], arrays["time"] - .2, atol=2e-6)
    assert metadata["source_timestamps"] == "provided_seconds"
    assert arrays["time"][-1] <= .5


def test_translation_uses_first_usable_heading_once():
    heading = 90.
    base = np.tile([0., 1., -3.], (5, 1))
    # In fixed camera-Y-up axes, a person facing +X travels +X.
    base[:, 0] += np.arange(5) * .1
    arrays, _ = preprocess(nlf(count=5, yaw=np.full(5, heading), pelvis=base), template(), smooth_seconds=0)
    np.testing.assert_allclose(arrays["translation"][:, 0], 0, atol=1e-12)
    np.testing.assert_allclose(arrays["translation"][:, 2], np.arange(5) * .1, atol=2e-6)
    np.testing.assert_allclose(arrays["yaw_deg"], 0, atol=1e-12)
    np.testing.assert_allclose(arrays["global_joints"][:, 0], base, atol=2e-6)


def test_uncertainty_gate_is_optional_and_propagates_to_descendants():
    record = nlf(count=9)
    record["unc"] = np.zeros((9, 24))
    record["unc"][:, 18] = 2  # Left elbow, so left wrist also becomes unsupported.
    plain, _ = preprocess(record, template(), smooth_seconds=0)
    gated, metadata = preprocess(record, template(), smooth_seconds=0, max_joint_uncertainty=1)
    assert plain["joint_valid"].all()
    assert gated["valid"].all()
    assert not gated["joint_valid"][:, [18, 20]].any()
    assert gated["joint_valid"][:, [19, 21]].all()
    assert np.isnan(gated["body"][:, 20]).all()
    assert metadata["uncertainty_is_calibrated_confidence"] is False
    del record["unc"]
    with pytest.raises(ValueError, match="unc is required"):
        preprocess(record, template(), max_joint_uncertainty=1)


def test_finite_masks_prevent_unsupported_joint_values_leaking():
    record = nlf(count=9)
    record["pose"][:, 18 * 3:19 * 3] = np.nan
    arrays, _ = preprocess(record, template(), smooth_seconds=0)
    assert arrays["valid"].all()
    assert not arrays["joint_valid"][:, 18].any()
    assert not arrays["joint_valid"][:, 20].any()
    assert np.isnan(arrays["coords"][:, 22:37]).all()


def test_source_quality_provenance_is_preserved_separately():
    record = nlf(count=9)
    record["valid"][0] = False
    record["pose"][1, :3] = np.nan
    record["unc"] = np.arange(9 * 24, dtype=np.float32).reshape(9, 24)
    record["box"] = np.ones((9, 5), dtype=np.float32)
    arrays, metadata = preprocess(record, template(), smooth_seconds=0)
    np.testing.assert_array_equal(arrays["source_detection_valid"], record["valid"])
    assert arrays["source_detection_valid"][1] and not arrays["source_valid"][1]
    np.testing.assert_array_equal(arrays["source_uncertainty"], record["unc"])
    np.testing.assert_array_equal(arrays["source_boxes"], record["box"])
    assert arrays["source_joint_valid"].shape == (9, 22)
    assert metadata["source_uncertainty_preserved"] and metadata["source_boxes_preserved"]


def test_joint_observation_masks_include_joint_parent_quality():
    record = nlf(count=9)
    record["pose"][4, 18 * 3:19 * 3] = np.nan
    arrays, _ = preprocess(record, template(), smooth_seconds=0)
    assert arrays["observed"][4]  # Root observation says nothing about elbow quality.
    assert arrays["joint_valid"][4, [18, 20]].all()
    assert not arrays["joint_observed"][4, [18, 20]].any()
    assert arrays["joint_interpolated"][4, [18, 20]].all()
    assert not arrays["source_joint_valid"][4, 20]
    assert arrays["joint_observed"][4, 19]
    np.testing.assert_array_equal(arrays["joint_interpolated"], arrays["joint_valid"] & ~arrays["joint_observed"])
    upsampled, _ = preprocess(nlf(count=5, source_fps=10), template(), smooth_seconds=0)
    assert not upsampled["joint_observed"][1::2].any()
    assert upsampled["joint_interpolated"][1::2].all()


def test_low_destination_fps_does_not_reconnect_hidden_long_gap():
    pelvis = np.array([[0., 1., -3.]] * 6)
    pelvis[-1, 1] = 3
    # Root height jumps across a missing interval; floor must not borrow either side.
    record = nlf(count=6, source_fps=10, yaw=[170, 0, 0, 0, 0, -170], pelvis=pelvis)
    record["valid"][1:5] = False
    arrays, metadata = preprocess(record, template(), fps=2, max_gap_seconds=.1, smooth_seconds=.1)
    assert arrays["valid"].tolist() == [True, True]
    assert arrays["segment_id"].tolist() == [0, 1]
    assert metadata["output_valid_segments"] == 2
    np.testing.assert_allclose(arrays["body"][:, 0, 1], 1, atol=2e-6)
    # A >180-degree branch jump remains a gap, rather than being unwrapped as a turn.
    another = nlf(count=19, source_fps=10,
                  yaw=np.r_[0., np.zeros(8), 170., np.zeros(8), -170.])
    another["valid"][:] = False
    another["valid"][[0, 9, 18]] = True
    coarse, _ = preprocess(another, template(), fps=10 / 9, max_gap_seconds=.1, smooth_seconds=0)
    assert coarse["segment_id"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(coarse["yaw_deg"], [0, 170, -170], atol=2e-6)


def test_joint_segments_preserve_a_gap_hidden_between_output_samples():
    record = nlf(count=6, source_fps=100)
    record["pose"][1:4, 18 * 3:19 * 3] = np.nan
    arrays, _ = preprocess(record, template(), fps=20, max_gap_seconds=0, smooth_seconds=0)
    np.testing.assert_allclose(arrays["time"], [0, .05], rtol=0, atol=1e-15)
    assert arrays["valid"].all() and arrays["joint_valid"].all()
    assert arrays["segment_id"].tolist() == [0, 0]
    assert arrays["joint_segment_id"][:, 18].tolist() == [0, 1]
    assert arrays["joint_segment_id"][:, 20].tolist() == [0, 1]  # Parent-chain support.
    assert arrays["joint_segment_id"][:, 19].tolist() == [0, 0]
    assert arrays["joint_observed"].all()  # Both endpoints really are observations.


def test_all_invalid_and_single_frame_are_well_formed():
    record = nlf(count=5)
    record["valid"][:] = False
    arrays, metadata = preprocess(record, template())
    assert not arrays["valid"].any()
    assert not arrays["observed"].any()
    assert metadata["reference_time"] is None
    assert np.isnan(arrays["coords"]).all()
    json.dumps(metadata, allow_nan=False)
    single, _ = preprocess(nlf(count=1), template())
    assert single["valid"].tolist() == [True]


@pytest.mark.parametrize("argument,value", [("fps", 0), ("fps", np.nan), ("fps", True),
                                           ("max_gap_seconds", -1), ("smooth_seconds", np.inf),
                                           ("max_joint_uncertainty", -1)])
def test_invalid_options_are_rejected(argument, value):
    with pytest.raises(ValueError):
        preprocess(nlf(), template(), **{argument: value})


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(fps=0),
    lambda x: x.update(pose=np.zeros((21, 66))),
    lambda x: x.update(j3d=np.zeros((20, 24, 3))),
    lambda x: x.update(valid=np.ones(20, dtype=bool)),
    lambda x: x.update(valid=np.full(21, 2)),
    lambda x: x.update(timestamps=np.zeros(21)),
    lambda x: x.update(timestamps=np.r_[np.arange(20), np.nan]),
    lambda x: x.update(unc=np.zeros((21, 22))),
])
def test_invalid_source_shapes_or_times_are_rejected(mutation):
    record = nlf()
    mutation(record)
    with pytest.raises(ValueError):
        preprocess(record, template())


def test_invalid_template_tree_is_rejected():
    rest = template()
    rest["parents"][4] = 4
    with pytest.raises(ValueError, match="parents"):
        preprocess(nlf(), rest)
