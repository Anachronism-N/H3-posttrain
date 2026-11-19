"""Targeted CPU declaration contracts; do not weaken the frozen native gate."""
import copy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
import caption_supported_part_declaration_v1 as recipe
bar, _ = recipe.direct.native()


def manifest():
    return dict(window=dict(window_id="w00000", start=0., end=2.),
                frames=[dict(index=i, time=i * .5) for i in range(4)])


def raw(text="The dancer raises the left arm.", parts=None):
    value = bar.empty_observation("unknown")
    value.update(uncertainties=[], abstain=False, events=[dict(id="e1", description=text,
                 parts=["torso"] if parts is None else parts, frame_indices=[0, 1])])
    for row in value["part_support"]:
        row.update(frame_indices=[0, 1, 2, 3], reason="visible")
    return value


def support(value, part, frames):
    next(s for s in value["part_support"] if s["part"] == part)["frame_indices"] = frames


class DeclarationContracts(unittest.TestCase):
    def test_missing_declared_part_fails_full_original_native_gate(self):
        value = raw()
        before = copy.deepcopy(value)
        result = recipe.derive(value, manifest())
        self.assertFalse(result["original_native_validate_passed"])
        self.assertIn("correct part support", result["original_native_validate_error"])
        self.assertEqual(result["declared_parts_delta"], [])
        self.assertEqual(result["raw_original"], result["declared_observation"])
        self.assertEqual(result["original_raw_filter"], result["new_checked_filter"])
        self.assertEqual(result["new_checked_filter"]["status"], "abstain")
        self.assertEqual(value, before)

    def test_eligible_analysis_is_not_applied_or_semantically_approved(self):
        value = raw()
        additions, _ = recipe.required_declarations(value["events"][0], value, manifest(), {})
        self.assertEqual(additions, ["left_arm"])
        result = recipe.derive(value, manifest())
        self.assertEqual(result["declared_parts_delta"], [])
        self.assertFalse(result["semantic_approval"])
        self.assertFalse(result["training_eligible"])

    def test_already_valid_event_is_identical_noop(self):
        value = raw(parts=["left_arm"])
        result = recipe.derive(value, manifest())
        self.assertTrue(result["original_native_validate_passed"])
        self.assertEqual(result["new_checked_filter"]["status"], "candidate")
        self.assertEqual(result["raw_original"], value)
        self.assertEqual(result["declared_observation"], value)
        self.assertEqual(result["declared_parts_delta"], [])

    def test_both_hands_need_two_same_event_frames_on_each_side(self):
        value = raw("The dancer moves both hands forward.")
        additions, _ = recipe.required_declarations(value["events"][0], value, manifest(), {})
        self.assertEqual(set(additions), {"left_hand", "right_hand"})
        for frames in ([0], [2, 3]):
            support(value, "right_hand", frames)
            self.assertEqual(recipe.required_declarations(value["events"][0], value, manifest(), {})[0], [])
        self.assertEqual(recipe.derive(value, manifest())["declared_parts_delta"], [])

    def test_singular_hand_side_not_guessed_even_both_support_available(self):
        value = raw("The dancer moves a hand forward.")
        additions, reason = recipe.required_declarations(value["events"][0], value, manifest(), {})
        self.assertEqual(additions, [])
        self.assertEqual(reason, "anatomical_side_unknown")

    def test_hand_shape_never_repairs_or_deletes_compound_body_clause(self):
        value = raw("The dancer raises the left arm while the left fingers form a V-shape.")
        additions, reason = recipe.required_declarations(value["events"][0], value, manifest(), {})
        self.assertEqual(additions, [])
        self.assertIn("fine_anatomy", reason)
        result = recipe.derive(value, manifest())
        self.assertEqual(result["declared_observation"], value)
        self.assertIn("V-shape", result["raw_original"]["events"][0]["description"])

    def test_support_reason_and_side_uncertainty_not_overridden(self):
        value = raw()
        next(s for s in value["part_support"] if s["part"] == "left_arm")["reason"] = "uncertain anatomical side"
        self.assertEqual(recipe.required_declarations(value["events"][0], value, manifest(), {})[0], [])
        value["uncertainties"] = ["Left/right hand identity is unknown."]
        self.assertEqual(recipe.required_declarations(value["events"][0], value, manifest(), {})[0], [])
        value = raw("The dancer raises the screen-left arm.")
        self.assertEqual(recipe.required_declarations(value["events"][0], value, manifest(), {})[0], [])

    def test_hardmask_event_or_support_forbidden_reject(self):
        value = raw()
        self.assertEqual(recipe.required_declarations(value["events"][0], value, manifest(), {"left_arm": {1}})[0], [])
        result = recipe.derive(value, manifest(), {"left_arm": {1}})
        self.assertEqual(result["original_raw_filter"]["status"], "native_rejected")
        self.assertEqual(result["original_raw_filter"], result["new_checked_filter"])
        self.assertEqual(result["declared_parts_delta"], [])

    def test_duplicate_event_frames_support_and_parts_keep_native_rejection(self):
        for kind in ("event_frames", "support_frames", "event_id", "parts"):
            value = raw()
            if kind == "event_frames":
                value["events"][0]["frame_indices"] = [0, 0]
            elif kind == "support_frames":
                support(value, "left_arm", [0, 0])
            elif kind == "event_id":
                value["events"].append(copy.deepcopy(value["events"][0]))
            else:
                value["events"][0]["parts"] = ["torso", "torso"]
            result = recipe.derive(value, manifest())
            self.assertFalse(result["original_native_validate_passed"])
            self.assertEqual(result["declared_parts_delta"], [])
            self.assertEqual(result["original_raw_filter"], result["new_checked_filter"])

    def test_timestamp_outside_half_open_window_not_fixed(self):
        m = manifest()
        m["frames"][-1]["time"] = m["window"]["end"]
        result = recipe.derive(raw(), m)
        self.assertEqual(result["original_raw_filter"]["status"], "native_rejected")
        self.assertEqual(result["declared_parts_delta"], [])

    def test_no_frame_expansion_no_static_substitution_no_forearm_inference(self):
        value = raw()
        value["events"][0]["frame_indices"] = [0]
        result = recipe.derive(value, manifest())
        self.assertEqual(result["declared_observation"], value)
        self.assertEqual(result["new_checked_filter"]["status"], "abstain")
        forearm = raw("The dancer moves the left forearm.")
        self.assertEqual(bar.claimed_part_groups(forearm["events"][0]["description"]), [])
        self.assertEqual(recipe.required_declarations(forearm["events"][0], forearm, manifest(), {})[0], [])

    def test_nonpart_problem_and_independent_reliable_body_left_to_native_filter(self):
        value = raw("The dancer raises the left arm. Her outfit moves in the breeze.", ["left_arm"])
        result = recipe.derive(value, manifest())
        self.assertEqual(result["declared_observation"], value)
        self.assertEqual(result["new_checked_filter"], result["original_raw_filter"])
        accepted = result["new_checked_filter"]["json"]["events"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["description"], "The dancer raises the left arm.")
        self.assertIn("outfit", result["raw_original"]["events"][0]["description"])

    def test_real_fixed_three_sources_native_replay_and_zero_gain(self):
        report = recipe.pilot3()
        self.assertEqual(report["source_count"], 3)
        self.assertEqual(report["window_count"], 8)
        self.assertEqual(report["applied_part_declarations"], 0)
        self.assertEqual(report["recovered_window_program_upper_bound"], 0)
        self.assertFalse(report["media_or_weights_content_read"])
        for source in report["sources"]:
            self.assertEqual(source["original_candidate_windows"], source["window_count"])
            for window in source["windows"]:
                audit = window["audit"]
                self.assertEqual(audit["raw_original"], audit["declared_observation"])
                self.assertEqual(audit["original_raw_filter"], audit["new_checked_filter"])


if __name__ == "__main__":
    unittest.main()
