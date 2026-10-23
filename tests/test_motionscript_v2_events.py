"""CPU-only synthetic geometry tests; run with unittest, no model assets needed."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_omnidance.exp.ms.motionscript_v2.events import (  # noqa: E402
    extract_events, make_llm_payload, render_caption,
)


def skeleton(n=61, fps=20.0):
    pose = np.array([
        [0, 1, 0], [.12, .9, 0], [-.12, .9, 0], [0, 1.15, 0],
        [.12, .5, 0], [-.12, .5, 0], [0, 1.3, 0], [.12, .1, 0],
        [-.12, .1, 0], [0, 1.4, 0], [.12, .05, .15], [-.12, .05, .15],
        [0, 1.55, 0], [.1, 1.5, 0], [-.1, 1.5, 0], [0, 1.75, 0],
        [.22, 1.5, 0], [-.22, 1.5, 0], [.48, 1.5, 0], [-.48, 1.5, 0],
        [.74, 1.5, 0], [-.74, 1.5, 0],
    ], dtype=float)
    body = np.broadcast_to(pose, (n, 22, 3)).copy()
    return {
        "time": np.arange(n) / fps, "body": body,
        "global_joints": body.copy(), "translation": np.zeros((n, 3)),
        "yaw_deg": np.zeros(n), "valid": np.ones(n, bool),
        "joint_valid": np.ones((n, 22), bool), "observed": np.ones(n, bool),
        "interpolated": np.zeros(n, bool),
    }


def elbow_angle(arrays, side, angle):
    elbow, wrist, sign = (18, 20, 1) if side == "left" else (19, 21, -1)
    theta = np.radians(np.broadcast_to(angle, len(arrays["time"])))
    vector = np.stack([-sign * np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
    arrays["body"][:, wrist] = arrays["body"][:, elbow] + .26 * vector


def select(events, part=None, measure=None):
    return [event for event in events
            if (part is None or event["body_part"] == part)
            and (measure is None or event["evidence"]["measurement_name"] == measure)]


class MotionScriptEventsTest(unittest.TestCase):
    def test_static_and_empty_abstain(self):
        for arrays in (skeleton(), skeleton(0)):
            events = extract_events(arrays)
            self.assertEqual(events, [])
            self.assertEqual(render_caption(events), [])
            payload = make_llm_payload(events)
            self.assertTrue(payload["abstain"])
            self.assertEqual(payload["events"], [])
            self.assertEqual(payload["fallback_captions"], [])

    def test_left_right_alternation_keeps_side_and_time(self):
        arrays = skeleton()
        elbow_angle(arrays, "left", np.r_[np.linspace(180, 90, 31), np.full(30, 90)])
        elbow_angle(arrays, "right", np.r_[np.full(30, 180), np.linspace(180, 90, 31)])
        events = extract_events(arrays)
        left, right = select(events, "left_elbow"), select(events, "right_elbow")
        self.assertEqual((len(left), len(right)), (1, 1))
        self.assertEqual(left[0]["action"], "bends")
        self.assertAlmostEqual(left[0]["measurement"]["delta"], -90)
        self.assertAlmostEqual(left[0]["end"], 1.5)
        self.assertAlmostEqual(right[0]["start"], 1.5)
        self.assertEqual(len({e["id"] for e in events}), len(events))
        captions = render_caption(list(reversed(events)))
        self.assertIn("Left elbow bends", captions[0])
        self.assertIn("Right elbow bends", captions[1])
        self.assertFalse(any(word in " ".join(captions).lower()
                             for word in (" then ", "simultaneous", "jump", "contact", "clap")))

    def test_small_threshold_jitter_does_not_create_events(self):
        arrays = skeleton(201)
        elbow_angle(arrays, "left", 120 + 4 * np.sin(np.arange(201) * .7))
        self.assertEqual(extract_events(arrays), [])

    def test_reversal_deadband_preserves_large_slow_movement(self):
        arrays = skeleton(81)
        elbow_angle(arrays, "left", np.linspace(170, 80, 81) + .6 * np.sin(np.arange(81)))
        events = select(extract_events(arrays), "left_elbow")
        self.assertEqual(len(events), 1)
        self.assertLess(events[0]["measurement"]["delta"], -89)

    def test_flexion_then_extension_are_separate_measured_events(self):
        arrays = skeleton()
        elbow_angle(arrays, "left", np.r_[np.linspace(180, 90, 31), np.linspace(90, 180, 31)[1:]])
        events = select(extract_events(arrays), "left_elbow")
        self.assertEqual([e["action"] for e in events], ["bends", "extends"])
        self.assertAlmostEqual(events[0]["end"], events[1]["start"])
        self.assertAlmostEqual(events[1]["measurement"]["delta"], 90)

    def test_long_invalid_gap_is_never_bridged(self):
        arrays = skeleton()
        elbow_angle(arrays, "left", np.r_[np.linspace(180, 140, 21), np.full(19, 120), np.linspace(100, 60, 21)])
        arrays["valid"][21:40] = False
        arrays["observed"][21:40] = False
        events = select(extract_events(arrays), "left_elbow")
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e["end"] <= 1 or e["start"] >= 2 for e in events))
        self.assertTrue(all(abs(e["measurement"]["delta"] + 40) < 1e-6 for e in events))

    def test_unmarked_timestamp_hole_is_not_bridged(self):
        arrays = skeleton(42)
        arrays["time"][21:] += 5
        elbow_angle(arrays, "left", np.r_[np.linspace(180, 170, 21), np.linspace(100, 90, 21)])
        self.assertEqual(select(extract_events(arrays), "left_elbow"), [])

    def test_source_segment_boundary_survives_low_fps_sampling(self):
        arrays = skeleton(6, fps=2)
        # Both source segments survive, but none of the invalid samples between
        # them do. Timestamp spacing alone cannot detect this original gap.
        arrays["segment_id"] = np.array([0, 0, 0, 1, 1, 1])
        elbow_angle(arrays, "left", np.array([180, 175, 170, 100, 95, 90]))
        self.assertEqual(select(extract_events(arrays), "left_elbow"), [])
        legacy = {key: value for key, value in arrays.items() if key != "segment_id"}
        self.assertEqual(len(select(extract_events(legacy), "left_elbow")), 1)

    def test_segment_minus_one_is_invalid_even_if_boolean_mask_missed_it(self):
        arrays = skeleton(21)
        elbow_angle(arrays, "left", np.linspace(180, 90, 21))
        arrays["segment_id"] = np.full(21, -1)
        self.assertEqual(extract_events(arrays), [])

    def test_events_record_source_segment_without_merging(self):
        arrays = skeleton(42)
        arrays["segment_id"] = np.r_[np.full(21, 3), np.full(21, 7)]
        elbow_angle(arrays, "left", np.r_[np.linspace(180, 140, 21), np.linspace(100, 60, 21)])
        events = select(extract_events(arrays), "left_elbow")
        self.assertEqual([event["evidence"]["continuity_segment_id"] for event in events], [3, 7])
        self.assertTrue(all(abs(event["measurement"]["delta"] + 40) < 1e-6 for event in events))

    def test_joint_gap_between_output_samples_does_not_break_other_signals(self):
        arrays = skeleton(6, fps=2)
        arrays["segment_id"] = np.zeros(6, dtype=int)
        arrays["joint_segment_id"] = np.zeros((6, 22), dtype=int)
        # Root stays continuous. A left elbow gap exists in the source between
        # the third and fourth output samples, with both retained poses usable.
        arrays["joint_segment_id"][3:, 18] = 1
        elbow_angle(arrays, "left", [180, 175, 170, 100, 95, 90])
        theta = np.radians(np.linspace(180, 90, 6))
        arrays["body"][:, 7] = arrays["body"][:, 4] + .4 * np.stack(
            [np.zeros(6), np.cos(theta), np.sin(theta)], -1)
        arrays["translation"][:, 2] = np.linspace(0, .5, 6)
        events = extract_events(arrays)
        self.assertEqual(select(events, "left_elbow"), [])
        self.assertEqual(len(select(events, "left_knee")), 1)
        root = select(events, "root", "pelvis_translation_z")
        self.assertEqual(len(root), 1)
        self.assertEqual(root[0]["evidence"]["required_joint_segment_ids"], [0])
        self.assertEqual(root[0]["evidence"]["continuity_scope"], "root_and_required_joint_segments")
        legacy = {key: value for key, value in arrays.items() if key != "joint_segment_id"}
        self.assertEqual(len(select(extract_events(legacy), "left_elbow")), 1)

    def test_required_joint_minus_one_overrides_joint_valid(self):
        arrays = skeleton(21)
        elbow_angle(arrays, "left", np.linspace(180, 90, 21))
        elbow_angle(arrays, "right", np.linspace(180, 90, 21))
        arrays["joint_segment_id"] = np.zeros((21, 22), dtype=int)
        arrays["joint_segment_id"][:, 20] = -1
        events = extract_events(arrays)
        self.assertEqual(select(events, "left_elbow"), [])
        self.assertEqual(len(select(events, "right_elbow")), 1)

    def test_required_joint_id_tuple_changes_even_when_maximum_is_unchanged(self):
        arrays = skeleton(6, fps=2)
        arrays["joint_segment_id"] = np.zeros((6, 22), dtype=int)
        arrays["joint_segment_id"][:, 16] = 9
        arrays["joint_segment_id"][3:, 18] = 1
        elbow_angle(arrays, "left", [180, 175, 170, 100, 95, 90])
        self.assertEqual(select(extract_events(arrays), "left_elbow"), [])

    def test_missing_wrist_suppresses_just_its_elbow_evidence(self):
        arrays = skeleton(21)
        elbow_angle(arrays, "left", np.linspace(180, 90, 21))
        elbow_angle(arrays, "right", np.linspace(180, 90, 21))
        arrays["joint_valid"][:, 20] = False
        events = extract_events(arrays)
        self.assertEqual(select(events, "left_elbow"), [])
        self.assertEqual(len(select(events, "right_elbow")), 1)

    def test_interpolation_is_reported_as_coverage_not_probability(self):
        arrays = skeleton(21)
        elbow_angle(arrays, "left", np.linspace(180, 90, 21))
        arrays["observed"][5:10] = False
        arrays["interpolated"][5:10] = True
        event = select(extract_events(arrays), "left_elbow")[0]
        evidence = event["evidence"]
        self.assertAlmostEqual(evidence["observed_fraction"], 16 / 21)
        self.assertAlmostEqual(evidence["interpolated_fraction"], 5 / 21)
        self.assertFalse(evidence["quality_is_probability"])
        self.assertEqual(evidence["coverage_scope"], "root_timestamp_only")
        self.assertEqual(evidence["required_joints"], [16, 18, 20])

    def test_joint_coverage_uses_all_required_observed_and_any_interpolated(self):
        arrays = skeleton(21)
        elbow_angle(arrays, "left", np.linspace(180, 90, 21))
        elbow_angle(arrays, "right", np.linspace(180, 90, 21))
        arrays["joint_observed"] = np.ones((21, 22), bool)
        arrays["joint_interpolated"] = np.zeros((21, 22), bool)
        # Root is fully observed; the left wrist has its own interpolation.
        arrays["joint_observed"][5:10, 20] = False
        arrays["joint_interpolated"][5:10, 20] = True
        # A separate required shoulder has two different interpolated samples.
        arrays["joint_observed"][10:12, 16] = False
        arrays["joint_interpolated"][10:12, 16] = True
        # An unrelated left ankle must not affect either elbow's coverage.
        arrays["joint_observed"][:, 7] = False
        arrays["joint_interpolated"][:, 7] = True
        events = extract_events(arrays)
        left = select(events, "left_elbow")[0]["evidence"]
        right = select(events, "right_elbow")[0]["evidence"]
        self.assertEqual(left["coverage_scope"], "required_joint_timestamp_support")
        self.assertAlmostEqual(left["observed_fraction"], 14 / 21)
        self.assertAlmostEqual(left["interpolated_fraction"], 7 / 21)
        self.assertEqual(right["observed_fraction"], 1.)
        self.assertEqual(right["interpolated_fraction"], 0.)
        self.assertIn("any required joint", left["interpolated_fraction_definition"])
        payload = make_llm_payload(events)
        self.assertTrue(any("not probability or confidence" in rule for rule in payload["constraints"]))
        self.assertTrue(any("never upgrade its scope" in rule for rule in payload["constraints"]))

    def test_twist_unwraps_across_180_inside_valid_run(self):
        arrays = skeleton(41)
        theta = np.radians(np.linspace(150, 210, 41))
        center = np.array([0, 1.5, 0])
        line = np.stack([.22 * np.cos(theta), np.zeros(41), -.22 * np.sin(theta)], -1)
        arrays["body"][:, 16] = center + line
        arrays["body"][:, 17] = center - line
        events = select(extract_events(arrays), measure="shoulder_yaw_relative_to_hips")
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["measurement"]["delta"], 60)
        self.assertEqual(events[0]["direction"], "positive_yaw")

    def test_root_uses_continuous_relative_yaw_without_wrapping(self):
        arrays = skeleton(101)
        arrays["yaw_deg"] = np.linspace(0, 450, 101)
        events = select(extract_events(arrays), "root", "continuous_root_yaw_relative_to_first_valid_frame")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["measurement"], {"start": 0., "end": 450., "delta": 450., "unit": "degrees"})
        self.assertEqual(events[0]["direction"], "positive_yaw")
        self.assertNotIn("spin", render_caption(events)[0])

    def test_knees_and_root_are_not_truncated_by_upper_body_events(self):
        arrays = skeleton(41)
        theta = np.radians(np.linspace(180, 90, 41))
        arrays["body"][:, 7] = arrays["body"][:, 4] + .4 * np.stack([np.zeros(41), np.cos(theta), np.sin(theta)], -1)
        arrays["translation"][:, 2] = np.linspace(0, .5, 41)
        elbow_angle(arrays, "left", np.linspace(180, 90, 41))
        elbow_angle(arrays, "right", np.linspace(180, 90, 41))
        events = extract_events(arrays)
        self.assertEqual(len(select(events, "left_knee")), 1)
        root = select(events, "root", "pelvis_translation_z")
        self.assertEqual(len(root), 1)
        self.assertAlmostEqual(root[0]["measurement"]["delta"], .5)
        self.assertEqual(root[0]["direction"], "positive_z")

    def test_arm_elevation_relative_to_torso(self):
        arrays = skeleton(41)
        theta = np.radians(np.linspace(0, 90, 41))
        arm = .26 * np.stack([np.sin(theta), -np.cos(theta), np.zeros(41)], -1)
        arrays["body"][:, 18] = arrays["body"][:, 16] + arm
        arrays["body"][:, 20] = arrays["body"][:, 16] + 2 * arm
        event = select(extract_events(arrays), "left_upper_arm")[0]
        self.assertEqual(event["action"], "raises")
        self.assertAlmostEqual(event["measurement"]["delta"], 90)

    def test_rigid_body_lean_is_not_relative_arm_raise(self):
        for axis, measure in ((0, "torso_forward_tilt"), (2, "torso_side_tilt")):
            with self.subTest(axis=axis):
                arrays = skeleton(41)
                angles = np.radians(np.linspace(0, 35, 41))
                rotations = np.broadcast_to(np.eye(3), (41, 3, 3)).copy()
                a, b = ((1, 2) if axis == 0 else (0, 1))
                rotations[:, a, a] = rotations[:, b, b] = np.cos(angles)
                rotations[:, a, b] = -np.sin(angles)
                rotations[:, b, a] = np.sin(angles)
                pelvis = arrays["body"][:, :1].copy()
                arrays["body"] = pelvis + np.einsum("nij,nkj->nki", rotations, arrays["body"] - pelvis)
                events = extract_events(arrays)
                self.assertEqual(len(select(events, "torso", measure)), 1)
                self.assertEqual(select(events, "left_upper_arm"), [])
                self.assertEqual(select(events, "right_upper_arm"), [])

    def test_short_spike_or_jump_then_hold_does_not_pass_duration(self):
        for angles in (np.r_[np.full(20, 180), 90, np.full(20, 180)],
                       np.r_[np.full(20, 180), np.full(21, 90)]):
            arrays = skeleton(41)
            elbow_angle(arrays, "left", angles)
            self.assertEqual(select(extract_events(arrays), "left_elbow"), [])

    def test_nan_or_degenerate_bones_do_not_generate_angle_events(self):
        arrays = skeleton(21)
        elbow_angle(arrays, "left", np.linspace(180, 90, 21))
        arrays["body"][:, 16] = arrays["body"][:, 18]
        arrays["body"][:, 20] = np.nan
        self.assertEqual(select(extract_events(arrays), "left_elbow"), [])

    def test_payload_is_local_json_serializable_and_keeps_provenance(self):
        arrays = skeleton(21)
        elbow_angle(arrays, "left", np.linspace(180, 90, 21))
        events = extract_events(arrays)
        payload = make_llm_payload(events)
        self.assertFalse(payload["abstain"])
        self.assertEqual(payload["events"], events)
        self.assertIn("motionscript_v2_rules", payload["fallback_captions"][0])
        json.dumps(payload, allow_nan=False)
        payload["events"][0]["measurement"]["delta"] = 0
        self.assertNotEqual(events[0]["measurement"]["delta"], 0)

    def test_bad_contract_and_thresholds_fail_explicitly(self):
        arrays = skeleton(21)
        arrays["time"][5] = arrays["time"][4]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            extract_events(arrays)
        with self.assertRaisesRegex(ValueError, "joint_valid"):
            extract_events({**skeleton(21), "joint_valid": np.ones((21, 21), bool)})
        with self.assertRaisesRegex(ValueError, "positive"):
            extract_events(skeleton(), min_angle_change_degrees=0)

    def test_nonbinary_or_nonfinite_masks_are_rejected_before_bool_cast(self):
        for key, shape in (("valid", (21,)), ("observed", (21,)),
                           ("interpolated", (21,)), ("joint_valid", (21, 22)),
                           ("joint_observed", (21, 22)), ("joint_interpolated", (21, 22))):
            for invalid in (np.nan, np.inf, -1, 2):
                with self.subTest(key=key, invalid=invalid):
                    arrays = skeleton(21)
                    arrays["joint_observed"] = np.ones((21, 22), bool)
                    arrays["joint_interpolated"] = np.zeros((21, 22), bool)
                    raw = np.zeros(shape)
                    raw.flat[0] = invalid
                    arrays[key] = raw
                    with self.assertRaisesRegex(ValueError, key):
                        extract_events(arrays)

    def test_optional_masks_and_segment_shapes_are_checked(self):
        for extra, message in (
            ({"segment_id": np.zeros((21, 1))}, "segment_id"),
            ({"segment_id": np.full(21, np.nan)}, "segment_id"),
            ({"segment_id": np.full(21, .5)}, "segment_id"),
            ({"segment_id": np.full(21, -2)}, "segment_id"),
            ({"joint_segment_id": np.zeros((21, 21))}, "joint_segment_id"),
            ({"joint_segment_id": np.full((21, 22), np.nan)}, "joint_segment_id"),
            ({"joint_segment_id": np.full((21, 22), np.inf)}, "joint_segment_id"),
            ({"joint_segment_id": np.full((21, 22), .5)}, "joint_segment_id"),
            ({"joint_segment_id": np.full((21, 22), -2)}, "joint_segment_id"),
            ({"joint_observed": np.ones((21, 22))}, "supplied together"),
            ({"joint_observed": np.ones((21, 21)),
              "joint_interpolated": np.zeros((21, 22))}, "joint_observed"),
            ({"joint_observed": np.ones((21, 22)),
              "joint_interpolated": np.zeros((21, 21))}, "joint_interpolated"),
        ):
            with self.subTest(extra=list(extra)):
                with self.assertRaisesRegex(ValueError, message):
                    extract_events({**skeleton(21), **extra})


if __name__ == "__main__":
    unittest.main()
