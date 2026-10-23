"""Model-free tests for 32-point palm/foot detail and rigid-motion invariance."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_omnidance.exp.ms.motionscript_v2.events import JOINT_NAMES
from scripts_omnidance.exp.ms.motionscript_v2.landmark_events import (
    extract_landmark_events, measure_landmark_signals, render_landmark_caption,
)


def arrays(points, n=41, fps=20.):
    names = list(points)
    positions = np.stack([np.broadcast_to(np.asarray(points[name], float), (n, 3)) for name in names], axis=1)
    return {"time": np.arange(n) / fps, "names": names, "points_m": positions,
            "point_valid": np.ones((n, len(names)), bool), "point_segment_id": np.zeros((n, len(names)), int),
            "point_observed": np.ones((n, len(names)), bool), "point_interpolated": np.zeros((n, len(names)), bool),
            "coordinate_frame": "camera_xright_ydown_zforward", "units": "m"}


def rotate_x(theta):
    theta = np.asarray(theta)
    rotation = np.broadcast_to(np.eye(3), (len(theta), 3, 3)).copy()
    rotation[:, 1, 1] = rotation[:, 2, 2] = np.cos(theta)
    rotation[:, 1, 2], rotation[:, 2, 1] = -np.sin(theta), np.sin(theta)
    return rotation


def rotate_y(theta):
    theta = np.asarray(theta)
    rotation = np.broadcast_to(np.eye(3), (len(theta), 3, 3)).copy()
    rotation[:, 0, 0] = rotation[:, 2, 2] = np.cos(theta)
    rotation[:, 0, 2], rotation[:, 2, 0] = np.sin(theta), -np.sin(theta)
    return rotation


def rigid_motion(data):
    result = copy.deepcopy(data)
    t = data["time"]
    rotation = rotate_x(.7 * t) @ rotate_y(1.3 * t + .4)
    shift = np.stack([np.sin(t), .5 * t, -.4 * t], axis=-1)
    result["points_m"] = np.einsum("nij,nkj->nki", rotation, data["points_m"]) + shift[:, None]
    return result


def palm(n=41, mode="tilt", start=0., end=60., angles=None):
    data = arrays({"left_shoulder": [-.3, 0, 0], "left_elbow": [0, 0, 0], "left_wrist": [0, .3, 0],
                   "left_index_mcp": [.04, .38, 0], "left_little_mcp": [-.04, .38, 0]}, n)
    angle = np.radians(np.linspace(start, end, n) if angles is None else angles)
    rotation = rotate_x(angle) if mode == "tilt" else rotate_y(angle)
    wrist = data["points_m"][:, 2:3]
    data["points_m"][:, 3:] = wrist + np.einsum("nij,nkj->nki", rotation, data["points_m"][:, 3:] - wrist)
    return data


def foot(n=41):
    data = arrays({"left_knee": [0, -.5, 0], "left_ankle": [0, 0, 0], "left_heel": [0, .05, -.1],
                   "left_big_toe": [.04, .05, .15], "left_little_toe": [-.04, .05, .15]}, n)
    rotation = rotate_x(np.radians(np.linspace(20, 60, n)))
    data["points_m"][:, 2:] = np.einsum("nij,nkj->nki", rotation, data["points_m"][:, 2:])
    return data


def full32():
    points = {name: np.zeros((41, 3)) for name in JOINT_NAMES}
    for detail in (palm(), foot()):
        for index, name in enumerate(detail["names"]):
            points[name] = detail["points_m"][:, index].copy()
            mirrored = np.broadcast_to(detail["points_m"][:1, index], (41, 3)).copy()
            mirrored[:, 0] = 2. - mirrored[:, 0]
            points[name.replace("left_", "right_")] = mirrored
    return arrays(points)


def select(events, measurement):
    return [event for event in events if event["evidence"]["measurement_name"] == measurement]


class LandmarkEventsTest(unittest.TestCase):
    def test_default32_full_traces_survive_subthreshold_motion_and_safe_npz(self):
        data = full32()
        signals, metadata = measure_landmark_signals(data)
        self.assertEqual(signals["landmark_signal_values"].shape, (41, 8))
        self.assertEqual(signals["landmark_signal_valid"].shape, (41, 8))
        self.assertEqual(signals["landmark_signal_names"].shape, (8,))
        self.assertEqual(signals["landmark_signal_names"].dtype.kind, "U")
        self.assertTrue(signals["landmark_signal_valid"].all())
        self.assertFalse(metadata["calibrated"])
        self.assertEqual(len(metadata["signals"]), 8)
        self.assertTrue(all(item["required_points"] and item["operator"] for item in metadata["signals"]))
        buffer = io.BytesIO()
        np.savez(buffer, **signals)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["landmark_signal_names"], signals["landmark_signal_names"])
            np.testing.assert_array_equal(saved["landmark_signal_values"], signals["landmark_signal_values"])
        subtle = palm(start=0., end=5.)
        self.assertEqual(extract_landmark_events(subtle), [])
        traces, _ = measure_landmark_signals(subtle)
        index = list(traces["landmark_signal_names"]).index("left_palm.palm_normal_to_forearm_angle")
        self.assertAlmostEqual(np.ptp(traces["landmark_signal_values"][:, index]), 5.)

    def test_full_trace_missing_point_degenerate_geometry_and_validity(self):
        data = palm()
        data["point_valid"][7, 3] = False
        data["points_m"][10, 4] = data["points_m"][10, 3]
        data["point_segment_id"][12, 3] = -1
        signals, _ = measure_landmark_signals(data)
        self.assertTrue(np.isnan(signals["landmark_signal_values"][[7, 10, 12]]).all())
        self.assertFalse(signals["landmark_signal_valid"][[7, 10, 12]].any())
        self.assertTrue(signals["landmark_signal_valid"][[0, 6, 8, 9, 11, 13]].all())
        data["names"][1] = "unconfigured_elbow"
        signals, metadata = measure_landmark_signals(data)
        self.assertEqual(signals["landmark_signal_values"].shape, (41, 0))
        self.assertEqual(signals["landmark_signal_valid"].shape, (41, 0))
        self.assertEqual(signals["landmark_signal_names"].dtype.kind, "U")
        self.assertEqual(metadata["signals"], [])

    def test_circular_full_trace_retains_branch_and_point_segment_boundaries(self):
        data = palm(mode="roll", start=150., end=210.)
        data["point_segment_id"][21:, 3] = 1
        signals, _ = measure_landmark_signals(data)
        index = list(signals["landmark_signal_names"]).index("left_palm.palm_roll_relative_to_arm_plane")
        roll = signals["landmark_signal_values"][:, index]
        self.assertAlmostEqual(roll[0], 150.)
        self.assertAlmostEqual(roll[-1], -150.)
        self.assertTrue(((roll >= -180.) & (roll <= 180.)).all())
        self.assertTrue(signals["landmark_signal_valid"][:, index].all())
        self.assertLess(np.diff(roll).min(), -350.)

    def test_full_traces_are_rigid_transform_invariant(self):
        data = full32()
        before, _ = measure_landmark_signals(data)
        after, _ = measure_landmark_signals(rigid_motion(data))
        np.testing.assert_array_equal(before["landmark_signal_names"], after["landmark_signal_names"])
        np.testing.assert_array_equal(before["landmark_signal_valid"], after["landmark_signal_valid"])
        # Compare circular branches as angles; +180 and -180 are equivalent.
        difference = (before["landmark_signal_values"] - after["landmark_signal_values"] + 180.) % 360. - 180.
        np.testing.assert_allclose(difference, 0., atol=1e-6)

    def test_default_32_points_support_palm_and_foot_without_other_detail(self):
        data = full32()
        self.assertEqual(len(data["names"]), 32)
        events = extract_landmark_events(data)
        self.assertEqual(len(select(events, "palm_normal_to_forearm_angle")), 1)
        self.assertEqual(len(select(events, "sole_normal_to_shin_angle")), 1)
        self.assertEqual(len(select(events, "foot_long_axis_to_shin_angle")), 1)
        self.assertEqual({event["body_part"] for event in events}, {"left_palm", "left_foot"})

    def test_extra_points_reveal_motion_with_wrist_and_arm_unchanged(self):
        data = palm()
        before = data["points_m"][:, :3]
        np.testing.assert_allclose(before, np.broadcast_to(before[:1], before.shape))
        events = select(extract_landmark_events(data), "palm_normal_to_forearm_angle")
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["measurement"]["delta"], 60)
        text = render_landmark_caption(events)[0]
        self.assertIn("palm-normal-to-forearm angle", text)
        self.assertNotIn("wrist turns", text)

    def test_signed_roll_unwraps_and_does_not_claim_isolated_wrist_turn(self):
        data = palm(mode="roll", start=150., end=210.)
        events = select(extract_landmark_events(data), "palm_roll_relative_to_arm_plane")
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["measurement"]["delta"], 60)
        self.assertEqual(events[0]["direction"], "positive_roll")
        self.assertIn("upper-arm/forearm plane", render_landmark_caption(events)[0])

    def test_straight_arm_and_parallel_normal_abstain_from_roll(self):
        data = palm(mode="roll")
        data["points_m"][:, 0] = [0, -.3, 0]
        self.assertEqual(select(extract_landmark_events(data), "palm_roll_relative_to_arm_plane"), [])
        data = palm(start=90., end=90.)
        self.assertEqual(select(extract_landmark_events(data), "palm_roll_relative_to_arm_plane"), [])

    def test_sole_and_long_axis_report_relative_geometry_without_contact(self):
        events = extract_landmark_events(foot())
        sole = select(events, "sole_normal_to_shin_angle")
        axis = select(events, "foot_long_axis_to_shin_angle")
        self.assertEqual((len(sole), len(axis)), (1, 1))
        self.assertAlmostEqual(sole[0]["measurement"]["delta"], 40)
        self.assertAlmostEqual(axis[0]["measurement"]["delta"], -40)
        text = " ".join(render_landmark_caption(events)).lower()
        self.assertFalse(any(word in text for word in ("contact", "landing", "jump", "clap", "dance step")))

    def test_time_varying_so3_and_translation_preserve_events_and_measurements(self):
        for data in (palm(), palm(mode="roll", start=150., end=210.), foot(), full32()):
            with self.subTest(names=data["names"]):
                before, after = extract_landmark_events(data), extract_landmark_events(rigid_motion(data))
                self.assertEqual(len(before), len(after))
                for a, b in zip(before, after):
                    self.assertEqual((a["start"], a["end"], a["body_part"], a["action"], a["direction"]),
                                     (b["start"], b["end"], b["body_part"], b["action"], b["direction"]))
                    for key in ("start", "end", "delta"):
                        self.assertAlmostEqual(a["measurement"][key], b["measurement"][key], places=6)

    def test_pure_rigid_motion_creates_no_local_event(self):
        for data in (palm(start=30., end=30.), foot(), full32()):
            data["points_m"][:] = data["points_m"][:1]
            self.assertEqual(extract_landmark_events(rigid_motion(data)), [])

    def test_missing_forearm_or_shin_anchor_has_no_camera_fallback(self):
        data = palm()
        data["names"][1] = "unconfigured_elbow"
        self.assertEqual(extract_landmark_events(data), [])
        data = foot()
        data["names"][0] = "unconfigured_knee"
        self.assertEqual(extract_landmark_events(data), [])

    def test_required_point_segment_changes_block_gap_even_at_low_fps(self):
        data = palm(n=6, angles=[0, 5, 10, 70, 75, 80])
        data["time"] = np.arange(6) * .5
        data["point_segment_id"][3:, 3] = 1
        self.assertEqual(extract_landmark_events(data), [])
        data["point_segment_id"][:] = 0
        self.assertEqual(len(extract_landmark_events(data)), 1)

    def test_invalid_gap_and_minus_one_are_not_crossed(self):
        for mode in ("valid", "segment"):
            with self.subTest(mode=mode):
                data = palm()
                if mode == "valid":
                    data["point_valid"][10:31, 3] = False
                else:
                    data["point_segment_id"][10:31, 3] = -1
                self.assertEqual(extract_landmark_events(data), [])

    def test_quality_is_per_required_point_and_anatomical_side_is_from_name(self):
        data = palm()
        data["names"] = [name.replace("left_", "right_") for name in data["names"]]
        data["points_m"] += [-10., 0, 0]
        data["point_observed"][5:10, 3] = False
        data["point_interpolated"][5:10, 3] = True
        event = extract_landmark_events(data)[0]
        self.assertEqual(event["body_part"], "right_palm")
        self.assertEqual(event["source"], "nlf_direct_rgb_landmarks")
        evidence = event["evidence"]
        self.assertAlmostEqual(evidence["observed_fraction"], 36 / 41)
        self.assertAlmostEqual(evidence["interpolated_fraction"], 5 / 41)
        self.assertEqual(evidence["coverage_scope"], "required_point_timestamp_support")
        self.assertFalse(evidence["quality_is_probability"])
        self.assertEqual(evidence["point_quality"][0]["observed_fraction"], 1.)
        self.assertAlmostEqual(evidence["point_quality"][1]["observed_fraction"], 36 / 41)
        json.dumps(event, allow_nan=False)

    def test_degenerate_plane_and_nonfinite_required_point_abstain(self):
        data = palm()
        data["points_m"][:, 4] = data["points_m"][:, 3]
        self.assertEqual(extract_landmark_events(data), [])
        data = palm()
        data["points_m"][:, 3] = np.nan
        self.assertEqual(extract_landmark_events(data), [])

    def test_short_or_subthreshold_changes_do_not_generate_events(self):
        self.assertEqual(extract_landmark_events(palm(), min_angle_change_degrees=70), [])
        self.assertEqual(extract_landmark_events(palm(n=3)), [])

    def test_empty_or_only_old_body_names_produce_no_invented_detail(self):
        self.assertEqual(extract_landmark_events(arrays({"pelvis": [0, 0, 0]}, n=0)), [])
        self.assertEqual(extract_landmark_events(arrays({"left_wrist": [0, 0, 0], "left_elbow": [0, .3, 0]})), [])
        self.assertEqual(render_landmark_caption([]), [])

    def test_masks_segments_units_and_names_are_validated(self):
        changes = (
            ("point_valid", np.full((41, 5), np.nan)),
            ("point_observed", np.full((41, 5), 2)),
            ("point_interpolated", np.zeros((41, 4))),
            ("point_segment_id", np.full((41, 5), .5)),
            ("point_segment_id", np.full((41, 5), -2)),
            ("point_segment_id", np.full((41, 5), np.inf)),
            ("names", ["left_wrist"] * 5),
            ("coordinate_frame", "unknown"), ("units", "mm"),
        )
        for key, value in changes:
            with self.subTest(key=key):
                data = palm()
                data[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    extract_landmark_events(data)


if __name__ == "__main__":
    unittest.main()
