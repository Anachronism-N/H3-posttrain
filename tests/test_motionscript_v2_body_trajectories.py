"""Independent wrist/ankle trajectory frames and camera-yaw support tests."""
import copy
import json

import numpy as np
from scipy.spatial.transform import Rotation

from scripts_omnidance.exp.ms.motionscript_v2.body_events import extract_body_events, measure_body_signals, render_body_caption


def upper_body(count=41):
    t = np.linspace(0, 2, count)
    theta = np.linspace(0, np.pi, count)
    names = ["left_shoulder", "right_shoulder", "neck", "spine3", "left_elbow", "left_wrist", "right_wrist"]
    points = np.broadcast_to(np.array([[-.3, 0., 3.], [.3, 0., 3.], [0., -.2, 3.],
                                      [0., .1, 3.], [-.3, .3, 3.], [-.3, .4, 3.], [.3, .4, 3.]]), (count, 7, 3)).copy()
    points[:, 5] = points[:, 0] + np.column_stack([.4 * np.cos(theta), np.zeros(count), .4 * np.sin(theta)])
    return {"time": t, "names": names, "points_m": points,
            "point_valid": np.ones((count, 7), bool), "point_segment_id": np.zeros((count, 7), int),
            "point_observed": np.ones((count, 7), bool), "point_interpolated": np.zeros((count, 7), bool),
            "coordinate_frame": "camera_xright_ydown_zforward", "units": "m"}


def test_wrist_position_keeps_motion_with_constant_wrist_shoulder_distance():
    data = upper_body()
    events = extract_body_events(data)
    assert not any(event["evidence"]["measurement_name"] == "wrist_to_shoulder_distance" for event in events)
    local = [event for event in events if "wrist_position_along_" in event["evidence"]["measurement_name"]]
    assert local and all(event["body_part"] == "left_wrist" for event in local)
    assert any(abs(event["measurement"]["delta"]) > .7 for event in local)
    assert all("pelvis" not in event["evidence"]["required_point_names"] for event in local)
    assert all("foot" not in name for event in local for name in event["evidence"]["required_point_names"])
    assert not any(event["evidence"]["plane_normal_is_verified_body_front"] for event in local)
    json.dumps(events, allow_nan=False)
    assert all("local" in line for line in render_body_caption(local))


def test_local_trajectories_are_invariant_to_time_varying_camera_rigid_transform():
    data = upper_body()
    moved = copy.deepcopy(data)
    rotations = Rotation.from_euler("xyz", np.column_stack([data["time"] * .6, data["time"], data["time"] * .2])).as_matrix()
    moved["points_m"] = np.einsum("tij,tkj->tki", rotations, data["points_m"]) + data["time"][:, None, None]
    a, metadata = measure_body_signals(data)
    b, _ = measure_body_signals(moved)
    columns = [i for i, item in enumerate(metadata["signals"]) if item["common_so3_translation_invariant"]]
    np.testing.assert_allclose(a["body_signal_values"][:, columns], b["body_signal_values"][:, columns], atol=1e-6)


def test_missing_other_wrist_does_not_delete_left_local_trajectory():
    data = upper_body()
    data["point_valid"][:, 6] = False
    data["points_m"][:, 6] = np.nan
    signals, _ = measure_body_signals(data)
    columns = [i for i, name in enumerate(signals["body_signal_names"]) if name.startswith("left_wrist.wrist_position_along_")]
    assert len(columns) == 3
    assert signals["body_signal_valid"][:, columns].all()


def test_degenerate_shoulder_belt_abstains_from_local_frame_without_dropping_elbow():
    data = upper_body()
    data["points_m"][:, 1] = data["points_m"][:, 0]
    signals, _ = measure_body_signals(data)
    names = signals["body_signal_names"].tolist()
    columns = [i for i, name in enumerate(names) if "_position_along_" in name or name.endswith("shoulder_plane_camera_yaw")]
    assert not signals["body_signal_valid"][:, columns].any()
    assert signals["body_signal_valid"][:, names.index("left_elbow.joint_interior_angle")].all()


def test_trajectory_reversal_and_hidden_point_segment_are_preserved():
    data = upper_body(81)
    data["points_m"][:, 5, 0] = np.r_[np.linspace(-.7, .1, 41), np.linspace(.1, -.7, 41)[1:]]
    events = [event for event in extract_body_events(data) if event["evidence"]["measurement_name"] == "wrist_position_along_shoulder_axis"]
    assert len(events) == 2
    assert np.sign(events[0]["measurement"]["delta"]) == -np.sign(events[1]["measurement"]["delta"])
    data["point_segment_id"][41:, 5] = 1
    split = [event for event in extract_body_events(data) if event["evidence"]["measurement_name"] == "wrist_position_along_shoulder_axis"]
    assert all(not (event["start"] < data["time"][41] < event["end"]) for event in split)


def test_pelvis_camera_displacement_does_not_require_ground_or_claim_world_motion():
    count = 41
    points = np.column_stack([np.linspace(0, .5, count), np.zeros(count), np.full(count, 3.)])[:, None]
    data = {"time": np.linspace(0, 2, count), "names": ["pelvis"], "points_m": points,
            "point_valid": np.ones((count, 1), bool), "point_observed": np.ones((count, 1), bool),
            "point_interpolated": np.zeros((count, 1), bool), "point_segment_id": np.zeros((count, 1), int),
            "coordinate_frame": "camera_xright_ydown_zforward", "units": "m"}
    events = extract_body_events(data)
    assert len(events) == 1
    assert events[0]["evidence"]["measurement_name"] == "pelvis_camera_x"
    assert events[0]["measurement"]["delta"] == .5
    assert not events[0]["evidence"]["camera_axes_are_world_or_gravity"]
    assert events[0]["evidence"]["required_point_names"] == ["pelvis"]
    assert "not world translation or ground height" in render_body_caption(events)[0]
    data["point_valid"][:] = False
    assert extract_body_events(data) == []


def test_camera_yaw_unwraps_short_arc_but_never_bridges_required_point_segments():
    data = upper_body()
    theta = np.linspace(170, 230, len(data["time"]))
    origin = np.array([0., 0., 3.])
    rotations = Rotation.from_euler("y", theta, degrees=True).as_matrix()
    data["points_m"] = np.einsum("tij,tkj->tki", rotations, data["points_m"] - origin) + origin
    events = [event for event in extract_body_events(data) if event["evidence"]["measurement_name"] == "shoulder_plane_camera_yaw"]
    assert len(events) == 1
    assert abs(events[0]["measurement"]["delta"]) < 70
    assert not events[0]["evidence"]["common_so3_translation_invariant"]
    assert "camera yaw" in render_body_caption(events)[0]
    data["point_segment_id"][21:, 0] = 1
    split = [event for event in extract_body_events(data) if event["evidence"]["measurement_name"] == "shoulder_plane_camera_yaw"]
    assert all(not (event["start"] < data["time"][21] < event["end"]) for event in split)
