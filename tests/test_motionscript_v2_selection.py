"""Selection must compress duplication without discarding real motion evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_omnidance.exp.ms.motionscript_v2.selection import make_summary_llm_payload, summarize_events


def event(name, start=0., end=1., action="bends", part="left_elbow"):
    bends = action == "bends"
    return {
        "id": name, "start": start, "end": end, "body_part": part,
        "action": action, "direction": "decreasing_joint_angle" if bends else "increasing_joint_angle",
        "measurement": {"start": 160. if bends else 120., "end": 120. if bends else 160.,
                        "delta": -40. if bends else 40., "unit": "degrees"},
        "evidence": {
            "measurement_name": "joint_interior_angle", "coordinate_frame": "body_yaw_removed_y_up_m",
            "required_joints": [16, 18, 20], "continuity_segment_id": 0,
            "required_joint_segment_ids": [0, 0, 0], "continuity_scope": "root_and_required_joint_segments",
            "observed_fraction": 1., "interpolated_fraction": 0., "quality_is_probability": False,
            "sample_start": round(start * 20), "sample_end": round(end * 20),
            "sample_count": round((end - start) * 20) + 1,
        }, "source": "motionscript_v2_rules",
    }


def alternation(offset=0., prefix="e", part="left_elbow"):
    return [event(f"{prefix}{i}", offset + i, offset + i + 1.,
                  "bends" if i % 2 == 0 else "extends", part) for i in range(4)]


class MotionScriptSelectionTest(unittest.TestCase):
    def test_empty_abstains_and_explains_closed_vocabulary(self):
        result = summarize_events([])
        self.assertTrue(result["abstain"])
        self.assertEqual(result["groups"], [])
        self.assertEqual(result["compact_caption"], [])
        self.assertFalse(result["vocabulary_coverage"]["open_vocabulary"])
        self.assertIn("Missing descriptions", result["vocabulary_coverage"]["notice"])

    def test_only_nearly_identical_high_overlap_copy_is_removed(self):
        original = event("a")
        duplicate = copy.deepcopy(original)
        duplicate.update(id="b", start=.005, end=1.005)
        result = summarize_events([duplicate, original])
        self.assertEqual(result["source_event_count"], 2)
        self.assertEqual(result["selected_event_count"], 1)
        self.assertEqual(result["duplicate_map"], {"b": "a"})
        self.assertEqual(result["removed_duplicate_ids"], ["b"])
        self.assertEqual(result["groups"][0]["source_ids"], ["a", "b"])
        self.assertEqual(result["groups"][0]["source_times"][1]["end"], 1.005)

    def test_same_primitive_at_different_times_is_not_a_duplicate(self):
        events = [event(f"e{i}", i * 2., i * 2. + 1.) for i in range(4)]
        result = summarize_events(events)
        self.assertEqual(result["selected_event_count"], 4)
        self.assertEqual(result["removed_duplicate_ids"], [])
        self.assertEqual(result["group_count"], 4)

    def test_partial_overlap_or_different_measurement_is_not_removed(self):
        for changed in (dict(start=.2, end=1.2),
                        dict(measurement={"start": 165., "end": 125., "delta": -40., "unit": "degrees"})):
            with self.subTest(changed=changed):
                duplicate = event("b")
                duplicate.update(changed)
                self.assertEqual(summarize_events([event("a"), duplicate])["selected_event_count"], 2)

    def test_distinct_source_side_action_direction_and_channel_are_preserved(self):
        for path, value in (
            (("source",), "other_rules"), (("body_part",), "right_elbow"),
            (("action",), "extends"), (("direction",), "other_direction"),
            (("evidence", "measurement_name"), "other_angle"),
            (("evidence", "coordinate_frame"), "other_frame"),
            (("evidence", "continuity_segment_id"), 1),
            (("evidence", "required_joint_segment_ids"), [0, 1, 0]),
            (("evidence", "unknown_provenance"), "different"),
        ):
            with self.subTest(path=path):
                second = event("b")
                target = second
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertEqual(summarize_events([event("a"), second])["selected_event_count"], 2)

    def test_materially_different_observation_evidence_is_preserved(self):
        second = event("b")
        second["evidence"]["observed_fraction"] = .5
        self.assertEqual(summarize_events([event("a"), second])["selected_event_count"], 2)

    def test_four_alternating_occurrences_get_one_reversible_caption(self):
        events = alternation()
        result = summarize_events(events)
        self.assertEqual(result["selected_event_count"], 4)
        self.assertEqual(result["removed_duplicate_ids"], [])
        self.assertEqual(result["group_count"], 1)
        group = result["groups"][0]
        self.assertEqual(group["kind"], "repeated_alternation")
        self.assertEqual(group["source_ids"], [e["id"] for e in events])
        self.assertEqual([(e["start"], e["end"]) for e in group["occurrences"]],
                         [(e["start"], e["end"]) for e in events])
        self.assertEqual(result["compact_caption"], ["0.00-4.00 s: Left elbow repeatedly bends and extends."])
        self.assertEqual(result["source_events"], events)

    def test_three_occurrences_do_not_assert_both_actions_repeat(self):
        result = summarize_events(alternation()[:3])
        self.assertEqual(result["group_count"], 3)
        self.assertFalse(any("repeatedly" in line for line in result["compact_caption"]))

    def test_duplicate_within_cycle_keeps_every_original_id_and_time(self):
        events = alternation()
        duplicate = copy.deepcopy(events[0])
        duplicate.update(id="duplicate", start=.005, end=1.005)
        result = summarize_events(events + [duplicate])
        self.assertEqual((result["source_event_count"], result["selected_event_count"], result["group_count"]), (5, 4, 1))
        group = result["groups"][0]
        self.assertEqual(group["occurrences"][0]["source_ids"], ["e0", "duplicate"])
        self.assertEqual(len(group["source_times"]), 5)
        self.assertEqual(result["source_events"][-1], duplicate)

    def test_separate_repetitions_remain_separate_time_spans(self):
        result = summarize_events(alternation() + alternation(10., "later"))
        self.assertEqual(result["group_count"], 2)
        self.assertEqual([(g["start"], g["end"]) for g in result["groups"]], [(0., 4.), (10., 14.)])
        self.assertEqual(result["selected_event_count"], 8)

    def test_left_and_right_are_not_merged_into_a_bilateral_action(self):
        result = summarize_events(alternation() + alternation(.2, "right", "right_elbow"))
        self.assertEqual(result["group_count"], 2)
        self.assertEqual([g["body_part"] for g in result["groups"]], ["left_elbow", "right_elbow"])
        self.assertFalse(any("both" in line for line in result["compact_caption"]))

    def test_source_or_segment_boundary_prevents_repetition_summary(self):
        for changed in ("source", "root", "joint"):
            with self.subTest(changed=changed):
                events = alternation()
                for item in events[2:]:
                    if changed == "source":
                        item["source"] = "other_rules"
                    elif changed == "root":
                        item["evidence"]["continuity_segment_id"] = 1
                    else:
                        item["evidence"]["required_joint_segment_ids"] = [0, 1, 0]
                result = summarize_events(events)
                self.assertEqual(result["group_count"], 4)
                self.assertEqual(result["removed_duplicate_ids"], [])

    def test_unknown_continuity_is_not_assumed_for_repeated_occurrences(self):
        events = alternation()
        for item in events:
            item["evidence"]["continuity_segment_id"] = None
            item["evidence"]["required_joint_segment_ids"] = None
        self.assertEqual(summarize_events(events)["group_count"], 4)
        exact_copy = copy.deepcopy(events[0])
        exact_copy["id"] = "copy"
        self.assertEqual(summarize_events([events[0], exact_copy])["selected_event_count"], 1)
        exact_copy["start"] = .005
        self.assertEqual(summarize_events([events[0], exact_copy])["selected_event_count"], 2)

    def test_overlapping_or_discontinuous_measurements_are_not_a_cycle(self):
        for mode in ("overlap", "gap", "measurement", "same_direction"):
            with self.subTest(mode=mode):
                events = alternation()
                if mode == "overlap":
                    events[1]["start"] = .9
                elif mode == "gap":
                    for item in events[2:]:
                        item["start"] += 1.
                        item["end"] += 1.
                elif mode == "measurement":
                    events[1]["measurement"].update(start=130., end=170.)
                else:
                    events[1]["direction"] = "decreasing_joint_angle"
                self.assertEqual(summarize_events(events)["group_count"], 4)

    def test_summary_and_source_copies_do_not_mutate_input(self):
        events = alternation()
        before = copy.deepcopy(events)
        result = summarize_events(events)
        self.assertEqual(events, before)
        result["source_events"][0]["measurement"]["delta"] = 0
        result["selected_events"][0]["measurement"]["delta"] = 1
        result["groups"][0]["occurrences"][0]["measurement"]["delta"] = 2
        self.assertEqual(events, before)
        json.dumps(summarize_events(events), allow_nan=False)

    def test_compact_payload_preserves_all_ids_times_and_allows_verified_summary(self):
        events = alternation()
        duplicate = copy.deepcopy(events[0])
        duplicate.update(id="duplicate", start=.005, end=1.005)
        summary = summarize_events(events + [duplicate])
        before = copy.deepcopy(summary)
        payload = make_summary_llm_payload(summary)
        self.assertEqual(len(payload["fallback_sentences"]), 1)
        sentence = payload["fallback_sentences"][0]
        self.assertIn("repeatedly bends and extends", sentence["text"])
        self.assertEqual(set(sentence["source_ids"]), {item["id"] for item in events + [duplicate]})
        self.assertEqual(sentence["source_times"], summary["groups"][0]["source_times"])
        self.assertTrue(any("exactly once" in rule for rule in payload["constraints"]))
        self.assertTrue(any("independent visual evidence" in rule for rule in payload["constraints"]))
        self.assertTrue(any("not synonym substitution" in rule for rule in payload["constraints"]))
        json.dumps(payload, allow_nan=False)
        payload["groups"][0]["source_ids"].clear()
        self.assertEqual(summary, before)

    def test_compact_payload_empty_abstains_without_a_static_pose(self):
        payload = make_summary_llm_payload(summarize_events([]))
        self.assertTrue(payload["abstain"])
        self.assertEqual(payload["fallback_sentences"], [])
        self.assertEqual(payload["fallback_captions"], [])

    def test_compact_payload_rejects_unverified_groups_or_missing_source_coverage(self):
        for mode in ("missing_id", "invented_action", "duplicate_map", "missing_source"):
            with self.subTest(mode=mode):
                summary = summarize_events(alternation())
                if mode == "missing_id":
                    summary["groups"][0]["source_ids"].pop()
                elif mode == "invented_action":
                    summary["groups"][0]["text"] = "Performs a dance step."
                elif mode == "duplicate_map":
                    summary["duplicate_map"]["e1"] = "e0"
                else:
                    del summary["source_events"]
                with self.assertRaisesRegex(ValueError, "summary|Summary"):
                    make_summary_llm_payload(summary)

    def test_duplicate_ids_bad_measurements_and_nonfinite_data_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_events([event("a"), event("a")])
        malformed = event("a")
        malformed["measurement"]["delta"] = 0
        with self.assertRaisesRegex(ValueError, "delta"):
            summarize_events([malformed])
        malformed = event("a")
        malformed["start"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_events([malformed])


if __name__ == "__main__":
    unittest.main()
