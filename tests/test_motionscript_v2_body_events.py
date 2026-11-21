"""Deterministic CPU-only tests of direct-point 3D body candidates, not accuracy."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_omnidance.exp.ms.motionscript_v2.body_events import (
    extract_body_events, measure_body_signals, render_body_caption,
)
from scripts_omnidance.exp.ms.motionscript_v2.landmark_events import (
    extract_landmark_events, measure_landmark_signals,
)
from scripts_omnidance.exp.ms.motionscript_v2.landmarks import preprocess_landmarks


FRAME = "camera_xright_ydown_zforward"
SOURCE = "nlf_direct_rgb_body_rules"


def tracks(points, time=None, n=41):
    time = np.arange(n, dtype=float) / 20 if time is None else np.asarray(time, float)
    names = list(points)
    positions = np.stack([np.broadcast_to(np.asarray(points[name], float), (len(time), 3))
                          for name in names], axis=1) if names else np.empty((len(time), 0, 3))
    shape = (len(time), len(names))
    return {"time": time, "points_m": positions.copy(), "names": names if names else np.asarray([], dtype="U"),
            "point_valid": np.ones(shape, bool), "point_observed": np.ones(shape, bool),
            "point_interpolated": np.zeros(shape, bool), "point_segment_id": np.zeros(shape, int),
            "coordinate_frame": FRAME, "units": "m"}


def interior(angles=None, *, part="elbow", side="left", time=None):
    angles = np.linspace(120., 60., 41) if angles is None else np.asarray(angles, float)
    theta = np.radians(angles)
    a, pivot, b = ("shoulder", "elbow", "wrist") if part == "elbow" else ("hip", "knee", "ankle")
    return tracks({side + "_" + a: [-.4, 0., 0.], side + "_" + pivot: [0., 0., 0.],
                   side + "_" + b: np.column_stack([-.3 * np.cos(theta), .3 * np.sin(theta), np.zeros(len(theta))])},
                  time=time, n=len(theta))


def distance(values=None, *, distal="wrist", time=None):
    values = np.linspace(.2, .6, 41) if values is None else np.asarray(values, float)
    proximal = "shoulder" if distal == "wrist" else "hip"
    return tracks({"left_" + distal: np.column_stack([values, np.zeros((len(values), 2))]),
                   "left_" + proximal: [0., 0., 0.]}, time=time, n=len(values))


def elevation(angles=None):
    angles = np.linspace(0., 60., 41) if angles is None else np.asarray(angles, float)
    theta = np.radians(angles)
    return tracks({"left_shoulder": [.5, 0., 0.],
                   "left_elbow": np.column_stack([.5 + .4 * np.sin(theta), .4 * np.cos(theta), np.zeros(len(theta))]),
                   "neck": [0., -.6, 0.], "spine3": [0., -.3, 0.]}, n=len(theta))


def coarse_foot(angles=None):
    angles = np.linspace(30., 80., 41) if angles is None else np.asarray(angles, float)
    theta = np.radians(angles)
    return tracks({"left_ankle": [0., 0., 0.], "left_foot": np.column_stack([
        .25 * np.sin(theta), -.25 * np.cos(theta), np.zeros(len(theta))]),
        "left_knee": [0., -.5, 0.]}, n=len(theta))


def joined(*items):
    result = copy.deepcopy(items[0])
    result["names"] = [name for item in items for name in item["names"]]
    for key in ("points_m", "point_valid", "point_observed", "point_interpolated", "point_segment_id"):
        result[key] = np.concatenate([item[key] for item in items], axis=1)
    return result


def rigid_motion(data):
    result = copy.deepcopy(data)
    t = data["time"]
    x, y = .7 * t + .2, 1.3 * t + .4
    rx = np.broadcast_to(np.eye(3), (len(t), 3, 3)).copy()
    ry = rx.copy()
    rx[:, 1, 1] = rx[:, 2, 2] = np.cos(x)
    rx[:, 1, 2], rx[:, 2, 1] = -np.sin(x), np.sin(x)
    ry[:, 0, 0] = ry[:, 2, 2] = np.cos(y)
    ry[:, 0, 2], ry[:, 2, 0] = np.sin(y), -np.sin(y)
    rotation = rx @ ry
    shift = np.column_stack([np.sin(t), .5 * t, -.4 * t])
    result["points_m"] = np.einsum("nij,nkj->nki", rotation, data["points_m"]) + shift[:, None]
    return result


def select(events, measurement, part=None):
    return [e for e in events if e["evidence"]["measurement_name"] == measurement
            and (part is None or e["body_part"] == part)]


def trace(data, name):
    arrays, metadata = measure_body_signals(data)
    column = list(arrays["body_signal_names"]).index(name)
    return arrays["body_signal_values"][:, column], arrays["body_signal_valid"][:, column], metadata["signals"][column]


def source_vfr():
    data = distance([.2, .25, .4, .55], time=[.05, .15, .25, .35])
    data.update({"raw_time": np.array([0., .04, .1, .22, .4, .55]),
                 "raw_frame_indices": np.array([7, 9, 12, 18, 25, 31]),
                 "raw_pts": np.array([9000, 9040, 9100, 9220, 9400, 9550]),
                 "time_base": np.asarray("1/1000"), "raw_quality_valid": np.ones((6, 2), bool)})
    data["raw_quality_valid"][[1, 3], 0] = False
    return data


class BodyEventsTest(unittest.TestCase):
    def test_half_body_preprocessing_needs_no_pelvis_or_feet(self):
        data = interior()
        raw = {"points_m": data["points_m"], "valid": data["point_valid"],
               "timestamps": data["time"], "names": np.asarray(data["names"]),
               "coordinate_frame": FRAME, "units": "m", "fps": 20.,
               "points2d_px": np.full((41, 3, 2), 20.)}
        processed, _ = preprocess_landmarks(raw, smooth_seconds=0, frame_size=[100, 80])
        event = select(extract_body_events(processed), "joint_interior_angle")[0]
        self.assertEqual(event["body_part"], "left_elbow")
        self.assertEqual(event["action"], "bends")
        self.assertEqual(event["evidence"]["required_point_names"], ["left_shoulder", "left_elbow", "left_wrist"])
        self.assertNotIn("source_frames", event["evidence"])
        self.assertFalse(any(e["body_part"] in ("torso", "root") for e in extract_body_events(processed)))

    def test_masked_unrelated_pelvis_feet_and_hand_do_not_suppress_elbow(self):
        data = joined(interior(), tracks({"pelvis": [np.nan] * 3, "left_foot": [np.nan] * 3,
                                         "right_wrist": [np.nan] * 3}))
        data["point_valid"][:, 3:] = False
        data["point_segment_id"][:, 3:] = -1
        events = select(extract_body_events(data), "joint_interior_angle")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["evidence"]["required_joints"], [0, 1, 2])
        self.assertEqual(events[0]["evidence"]["continuity_scope"], "independent_required_point_segments")
        self.assertNotIn("continuity_segment_id", events[0]["evidence"])
        data["point_segment_id"][:, 3:] = np.arange(41)[:, None]
        self.assertEqual(events, select(extract_body_events(data), "joint_interior_angle"))

    def test_missing_one_hand_does_not_drop_other_side(self):
        data = joined(interior(side="left"), interior(side="right"))
        for mode in ("missing_name", "invalid", "not_supported"):
            with self.subTest(mode=mode):
                candidate = copy.deepcopy(data)
                if mode == "missing_name":
                    candidate["names"][2] = "unconfigured_left_wrist"
                elif mode == "invalid":
                    candidate["point_valid"][:, 2] = False
                else:
                    candidate["point_observed"][:, 2] = False
                events = select(extract_body_events(candidate), "joint_interior_angle")
                self.assertEqual([e["body_part"] for e in events], ["right_elbow"])
                self.assertEqual(events[0]["evidence"]["required_joints"], [3, 4, 5])

    def test_elbow_and_knee_interior_angles_bend_and_extend_on_both_sides(self):
        for part in ("elbow", "knee"):
            for side in ("left", "right"):
                for first, last, action in ((120, 60, "bends"), (60, 120, "extends")):
                    with self.subTest(part=part, side=side, action=action):
                        data = interior(np.linspace(first, last, 41), part=part, side=side)
                        event = select(extract_body_events(data), "joint_interior_angle")[0]
                        self.assertEqual((event["body_part"], event["action"]), (side + "_" + part, action))
                        self.assertAlmostEqual(event["measurement"]["start"], first)
                        self.assertAlmostEqual(event["measurement"]["end"], last)
                        self.assertIn("3D interior angle", render_body_caption([event])[0])

    def test_elevation_zero_downward_and_only_neck_spine3_axis(self):
        data = elevation()
        values, good, description = trace(data, "left_upper_arm.upper_arm_elevation_relative_to_torso")
        self.assertTrue(good.all())
        self.assertAlmostEqual(values[0], 0.)
        self.assertAlmostEqual(values[-1], 60.)
        self.assertEqual(description["required_point_names"], ["left_shoulder", "left_elbow", "neck", "spine3"])
        event = extract_body_events(data)[0]
        self.assertEqual(event["action"], "raises")
        self.assertIn("neck-spine3 torso axis", render_body_caption([event])[0])
        for missing in ("neck", "spine3"):
            candidate = copy.deepcopy(data)
            candidate["names"][candidate["names"].index(missing)] = "unconfigured_" + missing
            candidate = joined(candidate, tracks({"pelvis": [0., .5, 0.]}))
            self.assertEqual(select(extract_body_events(candidate), "upper_arm_elevation_relative_to_torso"), [])
            arrays, _ = measure_body_signals(candidate)
            self.assertFalse(any("upper_arm_elevation" in name for name in arrays["body_signal_names"]))
        for key in ("point_valid", "point_observed"):
            candidate = copy.deepcopy(data)
            candidate[key][:, 3] = False
            _, good, _ = trace(candidate, "left_upper_arm.upper_arm_elevation_relative_to_torso")
            self.assertFalse(good.any())
            self.assertEqual(extract_body_events(candidate), [])
        self.assertEqual(extract_body_events(elevation(np.linspace(60, 0, 41)))[0]["action"], "lowers")

    def test_pair_distance_m_changes_not_world_translation(self):
        for distal, proximal in (("wrist", "shoulder"), ("ankle", "hip")):
            for first, last in ((.2, .6), (.6, .2)):
                with self.subTest(distal=distal, first=first):
                    event = extract_body_events(distance(np.linspace(first, last, 41), distal=distal))[0]
                    self.assertEqual(event["measurement"]["unit"], "m")
                    self.assertAlmostEqual(event["measurement"]["delta"], last - first)
                    self.assertEqual(event["direction"], ("farther_from_" if last > first else "closer_to_") + proximal)
                    text = render_body_caption([event])[0]
                    self.assertIn("distance between these points", text)
                    self.assertNotIn("moves", text)
                    self.assertNotIn("world", text)

    def test_coarse_foot_requires_ankle_foot_knee_not_toes(self):
        data = coarse_foot()
        event = extract_body_events(data)[0]
        self.assertEqual(event["evidence"]["measurement_name"], "foot_long_axis_to_shin_angle")
        self.assertEqual(event["evidence"]["required_point_names"], ["left_ankle", "left_foot", "left_knee"])
        self.assertAlmostEqual(event["measurement"]["delta"], 50.)
        extra = tracks({"left_big_toe": [np.nan] * 3, "left_little_toe": [np.nan] * 3,
                        "left_heel": [np.nan] * 3})
        extra["point_valid"][:] = False
        self.assertEqual(extract_body_events(joined(data, extra)), [event])
        self.assertEqual(extract_landmark_events(data), [])
        for missing in data["names"]:
            candidate = copy.deepcopy(data)
            candidate["names"][candidate["names"].index(missing)] = "unconfigured_" + missing
            self.assertEqual(extract_body_events(candidate), [])

    def test_torso_is_fixed_camera_up_orientation_not_absolute_bend(self):
        data = tracks({"neck": [0., -.5, 0.], "pelvis": [0., 0., 0.]})
        theta = np.radians(np.linspace(0., 120., 41))
        data["points_m"][:, 0] = np.column_stack([np.zeros(41), -.5 * np.cos(theta), -.5 * np.sin(theta)])
        values, good, desc = trace(data, "torso.torso_camera_up_angle")
        np.testing.assert_allclose(values, np.linspace(0, 120, 41), atol=1e-6)
        self.assertTrue(good.all())
        self.assertFalse(desc["common_so3_translation_invariant"])
        self.assertIn("camera_Y_up=(0,-1,0)", desc["operator"])
        event = extract_body_events(data)[0]
        self.assertEqual(event["action"], "changes_camera_coordinate_orientation")
        text = render_body_caption([event])[0]
        self.assertIn("camera Y-up", text)
        self.assertIn("camera-coordinate orientation change", text)
        self.assertFalse(any(word in text.lower() for word in ("gravity", "bend", "ground", "contact")))
        shifted = copy.deepcopy(data)
        shifted["points_m"] += np.column_stack([data["time"], data["time"] ** 2, -data["time"]])[:, None]
        np.testing.assert_allclose(trace(shifted, "torso.torso_camera_up_angle")[0], values, atol=1e-6)

    def test_time_varying_3d_rigid_motion_preserves_local_signals_and_changes(self):
        for data in (interior(), interior(part="knee"), distance(), distance(distal="ankle"), elevation(), coarse_foot()):
            with self.subTest(names=data["names"]):
                before, _ = measure_body_signals(data)
                after, _ = measure_body_signals(rigid_motion(data))
                np.testing.assert_array_equal(before["body_signal_names"], after["body_signal_names"])
                np.testing.assert_array_equal(before["body_signal_valid"], after["body_signal_valid"])
                np.testing.assert_allclose(before["body_signal_values"], after["body_signal_values"], atol=1e-6)
                a, b = extract_body_events(data), extract_body_events(rigid_motion(data))
                self.assertEqual(len(a), len(b))
                for original, transformed in zip(a, b):
                    for key in ("start", "end", "action", "direction", "body_part"):
                        self.assertEqual(original[key], transformed[key])
                    for key in ("start", "end", "delta", "path_change", "net_speed_per_s", "mean_speed_per_s", "peak_speed_per_s"):
                        self.assertAlmostEqual(original["measurement"][key], transformed["measurement"][key], places=6)

    def test_static_or_pure_rigid_motion_produces_no_local_events(self):
        for data in (interior(), interior(part="knee"), distance(), elevation(), coarse_foot()):
            data["points_m"][:] = data["points_m"][:1]
            self.assertEqual(extract_body_events(data), [])
            self.assertEqual(extract_body_events(rigid_motion(data)), [])
        self.assertEqual(extract_body_events(interior([30, 35, 30, 35, 30, 35, 30])), [])

    def test_repeated_reversals_have_separate_deterministic_events(self):
        angles = np.concatenate([np.linspace(30, 90, 11), np.linspace(90, 30, 11)[1:],
                                 np.linspace(30, 90, 11)[1:], np.linspace(90, 30, 11)[1:]])
        data = interior(angles)
        original = copy.deepcopy(data)
        events = extract_body_events(data)
        elbow = select(events, "joint_interior_angle")
        self.assertEqual([e["action"] for e in elbow], ["extends", "bends", "extends", "bends"])
        self.assertEqual([(e["start"], e["end"]) for e in elbow], [(0., .5), (.5, 1.), (1., 1.5), (1.5, 2.)])
        self.assertEqual(events, extract_body_events(data))
        self.assertEqual([e["id"] for e in events], [f"body_event_{i:04d}" for i in range(1, len(events) + 1)])
        for key in data:
            np.testing.assert_array_equal(data[key], original[key])

    def test_duration_excludes_flat_holds_and_thresholds_are_independent(self):
        data = interior([30, 90, 90, 90, 90, 90, 90, 90])
        self.assertEqual(extract_body_events(data), [])
        self.assertEqual(extract_body_events(interior([30, 90])), [])
        self.assertEqual(extract_body_events(interior(), min_angle_change_degrees=70, min_distance_change_m=10), [])
        self.assertEqual(extract_body_events(distance(), min_distance_change_m=.5), [])
        self.assertEqual(len(extract_body_events(distance(), min_angle_change_degrees=180)), 1)

    def test_degrees_speeds_path_change_and_peak_interval_midpoint(self):
        time = [.0, .1, .3, .5, .7]
        for angles, sign in (([30, 34, 60, 58, 90], 1), ([90, 86, 60, 62, 30], -1)):
            event = select(extract_body_events(interior(angles, time=time)), "joint_interior_angle")[0]
            m = event["measurement"]
            self.assertEqual(m["unit"], "degrees")
            self.assertEqual(m["speed_unit"], "degrees/s")
            self.assertAlmostEqual(m["duration_seconds"], .7)
            self.assertAlmostEqual(event["duration_seconds"], .7)
            self.assertAlmostEqual(m["delta"], sign * 60)
            self.assertAlmostEqual(m["path_change"], 64.)
            self.assertAlmostEqual(m["net_speed_per_s"], 60 / .7)
            self.assertAlmostEqual(m["mean_speed_per_s"], 64 / .7)
            self.assertAlmostEqual(m["peak_speed_per_s"], 160.)
            self.assertAlmostEqual(m["peak_speed_time"], .6)

    def test_metre_speeds_use_varying_intervals_and_first_exact_peak_tie(self):
        event = extract_body_events(distance([.2, .24, .5, .48, .8], time=[1., 1.1, 1.3, 1.5, 1.7]))[0]
        m = event["measurement"]
        self.assertEqual(m["unit"], "m")
        self.assertEqual(m["speed_unit"], "m/s")
        self.assertAlmostEqual(m["path_change"], .64)
        self.assertAlmostEqual(m["net_speed_per_s"], .6 / .7)
        self.assertAlmostEqual(m["mean_speed_per_s"], .64 / .7)
        self.assertAlmostEqual(m["peak_speed_per_s"], 1.6)
        self.assertAlmostEqual(m["peak_speed_time"], 1.6)
        tied = extract_body_events(distance([.25, .5, .75], time=[0, .25, .5]))[0]["measurement"]
        self.assertEqual(tied["peak_speed_per_s"], 1.)
        self.assertEqual(tied["peak_speed_time"], .125)

    def test_long_timestamp_gap_is_not_joined(self):
        data = interior([40, 45, 50, 110, 115, 120], time=[0, .05, .1, 1.1, 1.15, 1.2])
        self.assertEqual(select(extract_body_events(data), "joint_interior_angle"), [])
        data["time"] = np.arange(6) * .05
        self.assertEqual(len(select(extract_body_events(data), "joint_interior_angle")), 1)

    def test_required_segment_tuple_blocks_hidden_gap_even_when_min_max_unchanged(self):
        data = interior([40, 45, 50, 110, 115, 120], time=np.arange(6) * .5)
        data["point_segment_id"][:3] = [0, 1, 2]
        data["point_segment_id"][3:] = [1, 0, 2]
        self.assertEqual(select(extract_body_events(data), "joint_interior_angle"), [])
        values, good, metadata = trace(data, "left_elbow.joint_interior_angle")
        self.assertTrue(good.all())
        self.assertEqual(metadata["required_joint_segment_ids"], [[0, 1, 2]] * 3 + [[1, 0, 2]] * 3)
        self.assertGreater(np.diff(values).max(), 50)
        data["point_segment_id"][:] = 0
        self.assertEqual(len(select(extract_body_events(data), "joint_interior_angle")), 1)

    def test_invalid_unsupported_and_minus_one_runs_are_not_crossed(self):
        for mode in ("point_valid", "point_segment_id", "point_observed", "nonfinite"):
            with self.subTest(mode=mode):
                data = interior()
                if mode == "nonfinite":
                    data["points_m"][10:31, 2] = np.nan
                else:
                    data[mode][10:31, 2] = -1 if mode == "point_segment_id" else False
                self.assertEqual(select(extract_body_events(data), "joint_interior_angle"), [])
                values, good, _ = trace(data, "left_elbow.joint_interior_angle")
                self.assertFalse(good[10:31].any())
                self.assertTrue(np.isnan(values[10:31]).all())

    def test_zero_length_vectors_reject_every_affected_operator(self):
        cases = [(interior(), 0, 1, "left_elbow.joint_interior_angle"),
                 (interior(), 2, 1, "left_elbow.joint_interior_angle"),
                 (interior(part="knee"), 0, 1, "left_knee.joint_interior_angle"),
                 (elevation(), 2, 3, "left_upper_arm.upper_arm_elevation_relative_to_torso"),
                 (elevation(), 0, 1, "left_upper_arm.upper_arm_elevation_relative_to_torso"),
                 (distance(), 0, 1, "left_wrist.wrist_to_shoulder_distance"),
                 (distance(distal="ankle"), 0, 1, "left_ankle.ankle_to_hip_distance"),
                 (coarse_foot(), 1, 0, "left_foot.foot_long_axis_to_shin_angle"),
                 (coarse_foot(), 2, 0, "left_foot.foot_long_axis_to_shin_angle"),
                 (tracks({"neck": [0, -.5, 0], "pelvis": [0, 0, 0]}), 0, 1, "torso.torso_camera_up_angle")]
        for data, a, b, name in cases:
            with self.subTest(name=name, a=a, b=b):
                data["points_m"][:, a] = data["points_m"][:, b]
                values, good, _ = trace(data, name)
                self.assertFalse(good.any())
                self.assertTrue(np.isnan(values).all())
                self.assertEqual(select(extract_body_events(data), name.split(".")[1], name.split(".")[0]), [])
        data = interior()
        data["points_m"][20, 0] = data["points_m"][20, 1]
        _, good, _ = trace(data, "left_elbow.joint_interior_angle")
        self.assertFalse(good[20])
        self.assertTrue(good[19] and good[21])

    def test_full_signal_units_metadata_safe_npz_and_subthreshold_traces(self):
        points = {"neck": [0., -.7, 0.], "spine3": [0., -.4, 0.], "pelvis": [0., .3, 0.]}
        for side, sign in (("left", -1), ("right", 1)):
            for joint, point in {"shoulder": [.3, -.5, 0.], "elbow": [.55, -.1, .12],
                                 "wrist": [.5, .2, .3], "hip": [.18, .3, 0.], "knee": [.2, .8, .1],
                                 "ankle": [.2, 1.3, 0.], "foot": [.2, 1.35, .25]}.items():
                points[side + "_" + joint] = [sign * point[0], point[1], point[2]]
        data = tracks(points)
        arrays, metadata = measure_body_signals(data)
        self.assertEqual(arrays["body_signal_values"].shape, (41, 29))
        self.assertTrue(arrays["body_signal_valid"].all())
        self.assertEqual(metadata["units"], ["degrees", "m"])
        self.assertEqual(len(metadata["signals"]), 29)
        for index, description in enumerate(metadata["signals"]):
            for key in ("name", "operator", "coordinate_frame", "required_joints", "required_point_names", "required_joint_segment_ids", "unit"):
                self.assertIn(key, description)
            self.assertEqual(description["coordinate_frame"], FRAME)
            self.assertEqual(description["unit"], arrays["body_signal_units"][index])
            self.assertEqual(len(description["required_joint_segment_ids"]), 41)
            self.assertEqual(description["circular"], description["measurement_name"] == "shoulder_plane_camera_yaw")
        json.dumps(metadata, allow_nan=False)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        subtle = interior(np.linspace(60, 65, 41))
        self.assertEqual(extract_body_events(subtle), [])
        self.assertAlmostEqual(np.ptp(trace(subtle, "left_elbow.joint_interior_angle")[0]), 5.)

    def test_evidence_quality_is_local_uncalibrated_and_json_finite(self):
        data = interior()
        data["point_observed"][5:10, 2] = False
        data["point_interpolated"][5:10, 2] = True
        # Extra estimates are neither interpreted nor promoted to visibility.
        data["uncertainty"] = np.full((41, 3), np.inf)
        data["visibility"] = np.zeros((41, 3))
        event = select(extract_body_events(data), "joint_interior_angle")[0]
        evidence = event["evidence"]
        self.assertAlmostEqual(evidence["observed_fraction"], 36 / 41)
        self.assertAlmostEqual(evidence["interpolated_fraction"], 5 / 41)
        self.assertEqual(evidence["required_joint_segment_ids"], [0, 0, 0])
        self.assertEqual(evidence["point_quality"][0]["observed_fraction"], 1.)
        self.assertAlmostEqual(evidence["point_quality"][2]["observed_fraction"], 36 / 41)
        self.assertFalse(evidence["quality_is_probability"])
        self.assertFalse(evidence["calibrated"])
        self.assertEqual(event["source"], SOURCE)
        self.assertEqual(event["semantic_status"], "measurement_derived_candidate")
        self.assertFalse(event["semantic_approval"])
        json.dumps(extract_body_events(data), allow_nan=False)
        self.assertIn("Candidate:", render_body_caption([event])[0])

    def test_native_vfr_frames_nonzero_pts_brackets_and_all_point_support(self):
        data = source_vfr()
        event = extract_body_events(data)[0]
        evidence = event["evidence"]["source_frames"]
        self.assertEqual(evidence["time_base"], "1/1000")
        self.assertEqual(evidence["pts_time_offset_seconds"], 9.)
        self.assertEqual([f["raw_index"] for f in evidence["start_bracket"]], [1, 2])
        self.assertEqual([f["raw_index"] for f in evidence["end_bracket"]], [3, 4])
        self.assertEqual([f["frame_index"] for f in evidence["start_bracket"]], [9, 12])
        self.assertEqual([f["pts"] for f in evidence["end_bracket"]], [9220, 9400])
        self.assertEqual(evidence["range_start"], {"raw_index": 0, "frame_index": 7, "pts": 9000, "raw_time": 0.})
        self.assertEqual(evidence["range_end"]["raw_index"], 4)
        self.assertEqual(evidence["frame_indices"], [7, 9, 12, 18, 25])
        self.assertEqual(evidence["required_point_support"][0]["raw_quality_valid_support_indices"], [0, 2, 4])
        self.assertEqual(evidence["required_point_support"][1]["raw_quality_valid_support_indices"], [0, 1, 2, 3, 4])
        self.assertEqual(evidence["required_point_support"][0]["start_support_bracket"], [0, 2])
        self.assertEqual(evidence["required_point_support"][0]["end_support_bracket"], [2, 4])
        json.dumps(event, allow_nan=False)
        _, metadata = measure_body_signals(data)
        self.assertTrue(metadata["source_frame_evidence_available"])
        absolute = copy.deepcopy(data)
        absolute["time"] += 9
        absolute["raw_time"] += 9
        absolute_evidence = extract_body_events(absolute)[0]["evidence"]["source_frames"]
        self.assertEqual(absolute_evidence["pts_time_offset_seconds"], 0.)
        self.assertEqual(absolute_evidence["pts"], evidence["pts"])
        self.assertEqual(absolute_evidence["frame_indices"], evidence["frame_indices"])

    def test_exact_source_brackets_and_missing_quality_never_assumed_valid(self):
        data = source_vfr()
        data["time"] = np.array([0., .1, .22, .4])
        del data["raw_quality_valid"]
        evidence = extract_body_events(data)[0]["evidence"]["source_frames"]
        self.assertEqual([f["raw_index"] for f in evidence["start_bracket"]], [0, 0])
        self.assertEqual([f["raw_index"] for f in evidence["end_bracket"]], [4, 4])
        self.assertFalse(evidence["raw_quality_valid_available"])
        self.assertTrue(all(p["raw_quality_valid_support_indices"] is None for p in evidence["required_point_support"]))
        data["raw_quality_valid"] = np.zeros((6, 2), bool)
        evidence = extract_body_events(data)[0]["evidence"]["source_frames"]
        self.assertTrue(all(p["raw_quality_valid_support_indices"] == [] for p in evidence["required_point_support"]))
        self.assertTrue(all(p["start_support_bracket"] is None for p in evidence["required_point_support"]))

    def test_no_source_frame_invention_without_complete_native_information(self):
        for raw_only in (False, True):
            data = distance()
            if raw_only:
                data["raw_time"] = data["time"].copy()
                data["raw_quality_valid"] = np.ones((41, 2), bool)
            event = extract_body_events(data)[0]
            self.assertNotIn("source_frames", event["evidence"])
            self.assertNotIn("frame_index", event["evidence"])
            self.assertFalse(measure_body_signals(data)[1]["source_frame_evidence_available"])
        for missing in ("raw_time", "raw_frame_indices", "raw_pts", "time_base"):
            data = source_vfr()
            del data[missing]
            for function in (measure_body_signals, extract_body_events):
                with self.assertRaisesRegex(ValueError, "supplied together"):
                    function(data)

    def test_source_validation_rejects_bad_ranges_shapes_pts_and_time_bases(self):
        changes = [("raw_time", [0, .1, .1, .2, .3, .4]),
                   ("raw_time", [0, .1, .2, .3, .4, np.inf]),
                   ("raw_time", [0, .04, .1, .22, .41, .55]),
                   ("raw_frame_indices", [7, 9, 12, 18, 25, 25]),
                   ("raw_frame_indices", [-1, 9, 12, 18, 25, 31]),
                   ("raw_frame_indices", np.arange(6, dtype=float)),
                   ("raw_pts", [9000, 9040, 9100, 9220, 9400]),
                   ("raw_pts", np.arange(6, dtype=float)),
                   ("raw_pts", [9000, 9040, 9100, 9220, 9400, 9400]),
                   ("raw_quality_valid", np.ones((6, 3), bool)),
                   ("raw_quality_valid", np.full((6, 2), np.nan)),
                   ("raw_quality_valid", np.full((6, 2), 2)),
                   ("time", [-.1, .15, .25, .35]), ("time", [.05, .15, .25, .65])]
        changes += [("time_base", value) for value in ("0", "1/0", "unknown", True, np.nan, np.inf, [1], [1, 0])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                data = source_vfr()
                data[key] = value
                for function in (measure_body_signals, extract_body_events):
                    with self.assertRaises(ValueError):
                        function(data)
        for base in (.001, [1, 1000], np.asarray("1/1000")):
            data = source_vfr()
            data["time_base"] = base
            self.assertEqual(extract_body_events(data)[0]["evidence"]["source_frames"]["time_base"], "1/1000")

    def test_input_boundaries_units_and_thresholds_are_validated(self):
        changes = [("point_valid", np.full((41, 3), np.nan)), ("point_observed", np.full((41, 3), 2)),
                   ("point_interpolated", np.zeros((41, 2))), ("point_segment_id", np.full((41, 3), .5)),
                   ("point_segment_id", np.full((41, 3), -2)), ("point_segment_id", np.full((41, 3), np.inf)),
                   ("names", ["left_wrist"] * 3), ("names", ["left_shoulder", "", "left_wrist"]),
                   ("points_m", np.zeros((41, 3, 2))), ("points_m", np.zeros((41, 3, 3), complex)),
                   ("points_m", np.full((41, 3, 3), "1")), ("coordinate_frame", "world_y_up"),
                   ("coordinate_frame", [FRAME]), ("units", "mm"), ("units", "degrees"),
                   ("time", np.zeros(41)), ("time", np.full(41, np.inf)), ("time", np.arange(41).astype(str)),
                   ("time", np.arange(41).reshape(41, 1))]
        for key, value in changes:
            with self.subTest(key=key):
                data = interior()
                data[key] = value
                for function in (measure_body_signals, extract_body_events):
                    with self.assertRaisesRegex(ValueError, key):
                        function(data)
        for key in ("time", "points_m", "point_observed", "point_segment_id", "units"):
            data = interior()
            del data[key]
            with self.assertRaisesRegex(ValueError, key):
                extract_body_events(data)
        for option in ("min_duration_seconds", "min_angle_change_degrees", "min_distance_change_m"):
            for value in (0, -1, np.inf, np.nan, True, "1", [1]):
                with self.subTest(option=option, value=value):
                    with self.assertRaisesRegex(ValueError, option):
                        extract_body_events(interior(), **{option: value})

    def test_empty_missing_operators_single_sample_and_renderer_validation(self):
        for data in (tracks({}, n=0), tracks({}, n=5), tracks({"pelvis": [0, 0, 0]}), interior([60])):
            self.assertEqual(extract_body_events(data), [])
            arrays, metadata = measure_body_signals(data)
            self.assertEqual(arrays["body_signal_values"].shape[0], len(data["time"]))
            json.dumps(metadata, allow_nan=False)
        self.assertEqual(render_body_caption([]), [])
        events = extract_body_events(interior())
        self.assertEqual(render_body_caption(events), render_body_caption(list(reversed(events))))
        events[0]["evidence"]["measurement_name"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "Unsupported body measurement"):
            render_body_caption(events)

    def test_legacy_micro_events_remain_separate_with_no_body_aliasing(self):
        data = tracks({"left_shoulder": [-.3, 0, 0], "left_elbow": [0, 0, 0], "left_wrist": [0, .3, 0],
                       "left_index_mcp": [.04, .38, 0], "left_little_mcp": [-.04, .38, 0]})
        theta = np.radians(np.linspace(0, 60, 41))
        offset = data["points_m"][:, 3:] - data["points_m"][:, 2:3]
        rotation = np.broadcast_to(np.eye(3), (41, 3, 3)).copy()
        rotation[:, 1, 1] = rotation[:, 2, 2] = np.cos(theta)
        rotation[:, 1, 2], rotation[:, 2, 1] = -np.sin(theta), np.sin(theta)
        data["points_m"][:, 3:] = data["points_m"][:, 2:3] + np.einsum("nij,nkj->nki", rotation, offset)
        old_events, old_signals = extract_landmark_events(data), measure_landmark_signals(data)
        self.assertTrue(old_events)
        self.assertEqual(extract_body_events(data), [])
        measure_body_signals(data)
        self.assertEqual(extract_landmark_events(data), old_events)
        after = measure_landmark_signals(data)
        for key in old_signals[0]:
            np.testing.assert_array_equal(old_signals[0][key], after[0][key])
        self.assertEqual(old_signals[1], after[1])


if __name__ == "__main__":
    unittest.main()
