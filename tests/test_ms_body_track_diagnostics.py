"""Continuity warnings never become automatic side labels or error ground truth."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
from ms_body_track_diagnostics import diagnose


def raw(count=5):
    names = ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
    point = np.array([[-.3, 0., 3.], [-.6, .2, 3.], [-.8, .4, 3.], [.3, 0., 3.], [.6, .2, 3.], [.8, .4, 3.]])
    return {"names": np.array(names), "points_m": np.broadcast_to(point, (count, 6, 3)).copy(),
            "points2d_px": np.full((count, 6, 2), 20.), "valid": np.ones((count, 6), bool),
            "timestamps": np.arange(count) / 20, "frame_indices": np.arange(count), "pts": np.arange(count),
            "time_base": np.array([1, 20]), "frame_size": np.array([100, 100]),
            "units": np.array("m"), "coordinate_frame": np.array("camera_xright_ydown_zforward")}


def test_static_partial_body_has_no_swap_or_bone_change_warning():
    result = diagnose(raw())
    assert result["side_swap_candidates"] == result["relative_bone_length_change_candidates"] == []
    assert result["side_correspondence_support"]["arms"]["assessable_intervals"] == 4
    assert result["side_correspondence_support"]["legs"]["assessable_intervals"] == 0
    assert result["common_scale_available_frames"] == 5
    assert not result["semantic_quality_gate_passed"]


def test_hypothetical_whole_arm_side_swap_is_flagged_without_relabeling():
    source = raw()
    source["points_m"][2:] = source["points_m"][2:, [3, 4, 5, 0, 1, 2]]
    before = copy.deepcopy(source)
    result = diagnose(source)
    assert len(result["side_swap_candidates"]) == 1
    warning = result["side_swap_candidates"][0]
    assert warning["frame_start"] == 1 and warning["frame_end"] == 2
    assert warning["swapped_to_direct_cost_ratio"] == pytest.approx(0)
    assert not warning["confirmed_side_swap"]
    assert not result["automatic_relabeling"] and not result["candidate_events_deleted"]
    np.testing.assert_array_equal(source["points_m"], before["points_m"])


def test_true_rigid_half_turn_can_trigger_hypothesis_but_is_not_declared_impossible():
    source = raw()
    rotation = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    origin = np.array([0., 0., 3.])
    source["points_m"][2:] = (source["points_m"][2:] - origin) @ rotation.T + origin
    result = diagnose(source)
    assert result["side_swap_candidates"]
    assert all(item["true_turn_or_crossing_is_possible"] for item in result["side_swap_candidates"])
    assert not result["physical_impossibility_inferred"]
    assert result["relative_bone_length_change_candidates"] == []


def test_common_camera_model_translation_and_scale_do_not_create_relative_bone_warning():
    source = raw()
    origin = np.array([0., 0., 3.])
    source["points_m"][2:] = (source["points_m"][2:] - origin) * 1.8 + origin + np.array([.2, .1, .2])
    result = diagnose(source)
    assert result["relative_bone_length_change_candidates"] == []
    assert not result["semantic_quality_gate_passed"]
    assert all(item["speed_includes_camera_and_model_motion"] for item in result["point_displacement_peaks"])


def test_single_bone_discontinuity_is_a_review_warning_not_confirmed_joint_error():
    source = raw()
    source["points_m"][2, 2, 0] -= 1.
    result = diagnose(source)
    flagged = [item for item in result["relative_bone_length_change_candidates"] if item["bone"] == "left_elbow--left_wrist"]
    assert flagged
    assert {item["frame_end"] for item in flagged} == {2, 3}
    assert all(not item["confirmed_joint_error"] for item in flagged)


def test_detection_gap_does_not_connect_opposite_side_predictions():
    source = raw()
    source["valid"][2] = False
    source["points_m"][3:] = source["points_m"][3:, [3, 4, 5, 0, 1, 2]]
    result = diagnose(source)
    assert result["side_swap_candidates"] == []
    assert result["side_correspondence_support"]["arms"]["assessable_intervals"] == 2
    assert not result["source_gaps_interpolated_or_stitched"]


def test_offscreen_hand_does_not_drop_other_independent_point_diagnostics():
    source = raw()
    source["points2d_px"][:, 2, 0] = 101.
    result = diagnose(source)
    assert result["side_correspondence_support"]["arms"]["assessable_intervals"] == 0
    peaks = {item["point"]: item for item in result["point_displacement_peaks"]}
    assert peaks["left_wrist"]["peak_camera_displacement_speed_m_per_s"] is None
    assert peaks["right_wrist"]["assessable_intervals"] == 4
    assert not result["quality_mask_is_observability_ground_truth"]


def test_vfr_peak_uses_actual_pts_intervals_not_nominal_fps():
    source = raw(count=4)
    source["timestamps"] = np.array([.2, .233, .283, .383])
    source["pts"] = np.array([1200, 1233, 1283, 1383])
    source["time_base"] = np.array([1, 1000])
    source["points_m"][:, :, 0] += np.arange(4)[:, None] * .02
    result = diagnose(source)
    wrist = next(item for item in result["point_displacement_peaks"] if item["point"] == "left_wrist")
    assert wrist["peak_camera_displacement_speed_m_per_s"] == pytest.approx(.02 / .033)
    assert wrist["pts_start"] == 1200 and wrist["pts_end"] == 1233
    assert wrist["seconds_start"] == .2 and wrist["seconds_end"] == .233


@pytest.mark.parametrize("mutation", [
    lambda source: source.update(units=np.array("mm")),
    lambda source: source.update(coordinate_frame=np.array("world")),
    lambda source: source.update(frame_indices=np.array([0, 1, 3, 4, 5])),
    lambda source: source.update(pts=np.array([0, 1, 2, 3, 5])),
])
def test_wrong_units_or_source_timing_is_rejected(mutation):
    source = raw()
    mutation(source)
    with pytest.raises(ValueError):
        diagnose(source)
