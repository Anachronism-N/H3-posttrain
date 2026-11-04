"""Source binding and initial-prefix validation; synthetic unit fixtures only."""
import copy
import json
import unittest

from choreobench.context import build_boundary_context
from choreobench.io import object_sha
from choreobench.schema import ValidationError


def prefix(events=None):
    return {"schema": "choreobench.plan.v1", "song_id": "context-fixture", "sample_rate": 16000,
            "num_samples": 100, "motifs": [{"motif_id": "m", "description": "Declared theme."}],
            "events": [] if events is None else events,
            "provenance": {"private_source_marker": "SECRET_PREFIX_PROVENANCE"}}


def event():
    return {"event_id": "e", "start": 0, "end": 20, "track": "body", "action": "Step left.",
            "attributes": {}, "entry": {}, "exit": {"support": None}, "motif_id": "m", "evidence": []}


class ContextBindingTests(unittest.TestCase):
    def test_empty_prefix_still_binds_complete_song_and_clock(self):
        value = prefix()
        old = copy.deepcopy(value)
        result = build_boundary_context(value, 0)
        self.assertEqual(result["source_prefix_sha256"], object_sha(value))
        self.assertEqual((result["song_id"], result["sample_rate"], result["num_samples"]),
                         ("context-fixture", 16000, 100))
        self.assertEqual(value, old)
        self.assertFalse(any(row["latest_event"] for row in result["body_parts"].values()))
        self.assertNotIn("SECRET_PREFIX_PROVENANCE", json.dumps(result))
        self.assertNotIn("context-validation-only", json.dumps(result))

    def test_empty_prefix_requires_every_header_field(self):
        for key in prefix():
            with self.subTest(missing=key):
                value = prefix()
                del value[key]
                with self.assertRaises(ValidationError):
                    build_boundary_context(value, 0)

    def test_empty_prefix_rejects_invalid_header_and_unknown_keys(self):
        for key, value in (("schema", "wrong"), ("song_id", "bad:filename"),
                           ("sample_rate", True), ("sample_rate", 0), ("num_samples", 0),
                           ("motifs", {}), ("provenance", [])):
            with self.subTest(key=key, value=value):
                source = prefix()
                source[key] = value
                with self.assertRaises(ValidationError):
                    build_boundary_context(source, 0)
        value = prefix()
        value["hidden_header"] = "unexpected"
        with self.assertRaises(ValidationError):
            build_boundary_context(value, 0)

    def test_initial_motifs_are_validated_despite_having_no_events(self):
        invalid = [[{"motif_id": "m", "description": ""}],
                   [{"motif_id": "bad id", "description": "x"}],
                   [{"motif_id": "m", "description": "x"}, {"motif_id": "m", "description": "y"}],
                   [{"motif_id": "m", "description": "x", "extra": "y"}]]
        for motifs in invalid:
            with self.subTest(motifs=motifs), self.assertRaises(ValidationError):
                value = prefix()
                value["motifs"] = motifs
                build_boundary_context(value, 0)

    def test_source_hash_changes_with_song_clock_or_prefix_content(self):
        first = prefix([event()])
        initial = build_boundary_context(first, 20)
        for key, value in (("song_id", "another-song"), ("sample_rate", 24000), ("num_samples", 200)):
            second = copy.deepcopy(first)
            second[key] = value
            result = build_boundary_context(second, 20)
            self.assertNotEqual(result["source_prefix_sha256"], initial["source_prefix_sha256"])
            self.assertEqual(result[key], value)
        second = copy.deepcopy(first)
        second["events"][0]["action"] = "A distinct planned action."
        self.assertNotEqual(build_boundary_context(second, 20)["source_prefix_sha256"],
                            initial["source_prefix_sha256"])

    def test_bound_context_preserves_unknown_state_and_detaches_event_values(self):
        value = prefix([event()])
        original = copy.deepcopy(value)
        result = build_boundary_context(value, 30)
        leaf = result["body_parts"]["left_leg"]
        self.assertEqual(leaf["gap_samples"], 10)
        self.assertFalse(leaf["state_at_boundary_verified"])
        self.assertFalse(leaf["touches_boundary"])
        self.assertIsNone(leaf["latest_event"]["exit"]["support"])
        leaf["latest_event"]["exit"]["support"] = "changed context"
        self.assertEqual(value, original)

    def test_empty_prefix_only_at_zero_and_future_events_still_fail(self):
        with self.assertRaises(ValidationError):
            build_boundary_context(prefix(), 1)
        for boundary in (0, 19, 100, True, -1, 1.5):
            with self.subTest(boundary=boundary), self.assertRaises(ValidationError):
                build_boundary_context(prefix([event()]), boundary)

    def test_finite_metadata_and_history_budget_are_required(self):
        value = prefix()
        value["provenance"]["invalid"] = float("inf")
        with self.assertRaises(ValidationError):
            build_boundary_context(value, 0)
        for budget in (0, 9, True, 1.0):
            with self.subTest(budget=budget), self.assertRaises(ValidationError):
                build_boundary_context(prefix(), 0, motif_examples=budget)


if __name__ == "__main__":
    unittest.main()
