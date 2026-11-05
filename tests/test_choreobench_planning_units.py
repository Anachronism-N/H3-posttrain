"""Synthetic clock/partition checks, not measured music-analysis quality."""
import copy
import json
import random
import unittest

from choreobench.audio import make_windows
from choreobench.io import object_sha
from choreobench.planning_units import build_planning_units, validate_planning_units
from choreobench.schema import ValidationError


def song_fixture(count=30000, downbeats=(), cuts=(), rate=1000):
    bounds = [0] + list(cuts) + [count]
    beats = set(downbeats)
    for left, right in zip(downbeats, downbeats[1:]):
        beats.add((left + right) // 2)
    return {"song_id": "synthetic-clock", "split": "dev", "song_group_id": None,
        "artist_group_id": None, "near_duplicate_group_id": None, "brief": "Synthetic fixture only",
        "audio": {"path": "not-read.wav", "sha256": "a" * 64, "sample_rate": rate, "num_samples": count},
        "music": {"source": "synthetic fixture; no real audio analysis", "beats": sorted(beats),
            "downbeats": list(downbeats), "sections": [
                {"section_id": "s%03d" % i, "start": left, "end": right,
                 "label": "unknown", "repeat_group": None}
                for i, (left, right) in enumerate(zip(bounds, bounds[1:]))]}}


class PlanningUnitTests(unittest.TestCase):
    def assert_partition(self, document, song):
        cursor = 0
        sections = {s["section_id"]: s for s in song["music"]["sections"]}
        for unit in document["units"]:
            self.assertEqual(unit["start"], cursor)
            self.assertGreater(unit["end"], cursor)
            self.assertLessEqual(unit["end"] - unit["start"], document["provenance"]["capacity_samples"])
            section = sections[unit["section_id"]]
            self.assertLessEqual(section["start"], unit["start"])
            self.assertLessEqual(unit["end"], section["end"])
            self.assertFalse(unit["semantic_phrase_verified"])
            cursor = unit["end"]
        self.assertEqual(cursor, song["audio"]["num_samples"])
        self.assertIs(validate_planning_units(document, song), document)

    def test_default_meter_gate_is_explicit_even_for_synthetic_known_beats(self):
        song = song_fixture(downbeats=range(0, 30000, 2000))
        result = build_planning_units(song)
        self.assertEqual([(u["start"], u["end"]) for u in result["units"]],
                         [(0, 10125), (10125, 20250), (20250, 30000)])
        self.assertTrue(all(u["kind"] == "time_block" for u in result["units"]))
        self.assertTrue(all("meter_unverified" in u["reasons"] for u in result["units"]))
        self.assertEqual(result["provenance"]["meter_status_basis"], "author_declaration_not_confidence")
        self.assert_partition(result, song)

    def test_fifty_millisecond_time_tail_merges_without_crossing_public_section(self):
        song = song_fixture(40000, cuts=[20300])
        result = build_planning_units(song)
        section = [u for u in result["units"] if u["section_id"] == "s000"]
        self.assertEqual([(u["start"], u["end"]) for u in section], [(0, 10125), (10125, 20300)])
        self.assertIn("short_time_tail_merged_with_previous", section[-1]["reasons"])
        self.assertFalse(any(u["start"] < 20300 < u["end"] for u in result["units"]))
        self.assert_partition(result, song)

    def test_short_tail_balances_when_merging_would_exceed_capacity(self):
        song = song_fixture(20250)
        result = build_planning_units(song, config={"fallback_seconds": 20, "max_unit_seconds": 20})
        self.assertEqual([(u["start"], u["end"]) for u in result["units"]], [(0, 10125), (10125, 20250)])
        self.assertTrue(all("short_time_tail_balanced_with_previous" in u["reasons"] for u in result["units"]))
        self.assert_partition(result, song)

    def test_short_source_section_is_preserved_not_merged_across_scope(self):
        song = song_fixture(12000, cuts=[500])
        result = build_planning_units(song)
        self.assertEqual((result["units"][0]["start"], result["units"][0]["end"]), (0, 500))
        self.assertIn("source_interval_shorter_than_time_target", result["units"][0]["reasons"])
        self.assert_partition(result, song)

    def test_variable_tempo_uses_actual_downbeat_intervals(self):
        song = song_fixture(20000, [0, 1000, 2500, 4500, 7000, 10000, 14000, 17000])
        result = build_planning_units(song, config={"meter_status": "reviewed"})
        first = result["units"][0]
        self.assertEqual((first["kind"], first["start"], first["end"], first["bars"]),
                         ("metrical_block", 0, 7000, 4))
        self.assertEqual(first["bar_intervals"], [{"start": a, "end": b}
                         for a, b in [(0, 1000), (1000, 2500), (2500, 4500), (4500, 7000)]])
        self.assert_partition(result, song)

    def test_pickup_remaining_bars_and_unclosed_tail_are_not_fabricated_bars(self):
        song = song_fixture(12000, [720, 2720, 4720, 6720, 8720, 10720])
        result = build_planning_units(song, config={"meter_status": "reviewed"})
        units = result["units"]
        self.assertEqual([(u["kind"], u["start"], u["end"]) for u in units], [
            ("partial_fragment", 0, 720), ("metrical_block", 720, 8720),
            ("partial_fragment", 8720, 10720), ("partial_fragment", 10720, 12000)])
        self.assertEqual(units[0]["reasons"], ["before_first_downbeat"])
        self.assertEqual(units[2]["complete_bar_count"], 1)
        self.assertIsNone(units[2]["bars"])
        self.assertEqual(units[-1]["reasons"], ["unclosed_tail_after_last_downbeat"])
        self.assert_partition(result, song)

    def test_sections_cutting_a_bar_are_exact_and_never_snapped_or_crossed(self):
        song = song_fixture(12000, [720, 2720, 4720, 6720, 8720, 10720], [7900])
        result = build_planning_units(song, config={"meter_status": "reviewed"})
        self.assertIn((6720, 7900), [(u["start"], u["end"]) for u in result["units"]])
        self.assertIn((7900, 8720), [(u["start"], u["end"]) for u in result["units"]])
        self.assertFalse(any(u["start"] < 7900 < u["end"] for u in result["units"]))
        self.assert_partition(result, song)

    def test_section_wholly_inside_predicted_bar_remains_a_fragment(self):
        song = song_fixture(12000, [0, 4000, 8000], [1000, 2000])
        result = build_planning_units(song, config={"meter_status": "reviewed"})
        middle = [u for u in result["units"] if u["section_id"] == "s001"]
        self.assertEqual(len(middle), 1)
        self.assertEqual((middle[0]["kind"], middle[0]["complete_bar_count"]), ("partial_fragment", 0))
        self.assert_partition(result, song)

    def test_empty_and_single_downbeat_fall_back_instead_of_extrapolating(self):
        for downbeats in ([], [1000]):
            with self.subTest(downbeats=downbeats):
                song = song_fixture(downbeats=downbeats)
                result = build_planning_units(song, config={"meter_status": "reviewed"})
                self.assertTrue(all(u["kind"] == "time_block" for u in result["units"]))
                self.assertTrue(all("insufficient_downbeats_for_closed_bar" in u["reasons"] for u in result["units"]))
                self.assert_partition(result, song)

    def test_capacity_reduces_bar_count_at_real_downbeats(self):
        song = song_fixture(43000, list(range(0, 43000, 6000)))
        result = build_planning_units(song, config={"meter_status": "reviewed"})
        blocks = [u for u in result["units"] if u["kind"] == "metrical_block"]
        self.assertEqual([(u["start"], u["end"], u["bars"]) for u in blocks], [(0, 18000, 3), (18000, 36000, 3)])
        self.assertTrue(all(u["reasons"] == ["bar_count_reduced_to_capacity"] for u in blocks))
        self.assert_partition(result, song)

    def test_one_overlong_predicted_bar_uses_bounded_time_fallback(self):
        song = song_fixture(55000, [0, 25000, 50000])
        result = build_planning_units(song, config={"meter_status": "reviewed"})
        first = result["units"][0]
        self.assertEqual((first["kind"], first["start"], first["end"]), ("time_block", 0, 10125))
        self.assertIn("predicted_bar_exceeds_capacity", first["reasons"])
        self.assertTrue(all(u["bars"] is None for u in result["units"]))
        self.assert_partition(result, song)

    def test_overlong_unclosed_fragment_uses_bounded_time_fallback(self):
        song = song_fixture(50000, [0, 2000])
        result = build_planning_units(song, config={"meter_status": "reviewed"})
        tail = result["units"][1:]
        self.assertTrue(all(u["kind"] == "time_block" for u in tail))
        self.assertTrue(all("fragment_exceeds_capacity" in u["reasons"] for u in tail))
        self.assert_partition(result, song)

    def test_maximum_duration_caps_fallback_and_never_rounds_capacity_up(self):
        song = song_fixture(count=20000)
        result = build_planning_units(song, config={"max_unit_seconds": 3.9999})
        self.assertEqual(result["provenance"]["capacity_samples"], 3999)
        self.assertTrue(all(u["end"] - u["start"] <= 3999 for u in result["units"]))
        self.assertIn("fallback_reduced_to_capacity", result["units"][0]["reasons"])
        self.assert_partition(result, song)

    def test_intersections_preserve_one_window_decision_across_blocks(self):
        song = song_fixture(count=17000)
        result = build_planning_units(song, config={"fallback_seconds": 2})
        windows = make_windows(17000, 1000)
        for window in windows:
            ownership = []
            for unit in result["units"]:
                for ref in unit["window_refs"]:
                    if ref["window_index"] == window["index"]:
                        expected_start = max(unit["start"], window["observation_start"])
                        expected_end = min(unit["end"], window["observation_end"])
                        self.assertEqual(ref["observation_overlap"], {"start": expected_start, "end": expected_end})
                        if ref["ownership_overlap"] is not None:
                            ownership.append(ref["ownership_overlap"])
            self.assertEqual(ownership[0]["start"], window["ownership_start"])
            self.assertEqual(ownership[-1]["end"], window["ownership_end"])
            self.assertTrue(all(a["end"] == b["start"] for a, b in zip(ownership, ownership[1:])))
        self.assertEqual(result["provenance"]["decision_scope"], "observation_window")
        self.assertTrue(any(ref["ownership_overlap"] is None for u in result["units"] for ref in u["window_refs"]))
        self.assert_partition(result, song)

    def test_custom_stride_is_bound_without_changing_expert_windows(self):
        song = song_fixture(count=17000)
        windows = make_windows(17000, 1000, stride_seconds=2)
        before = object_sha(windows)
        result = build_planning_units(song, windows)
        self.assertEqual(result["windows_sha256"], before)
        self.assertIs(validate_planning_units(result, song, windows), result)
        self.assertEqual(before, object_sha(windows))
        with self.assertRaises(ValidationError):
            validate_planning_units(result, song)

    def test_output_is_json_and_does_not_mutate_inputs(self):
        song = song_fixture(downbeats=range(0, 30000, 2000))
        before = object_sha(song)
        cfg = {"meter_status": "reviewed"}
        result = build_planning_units(song, config=cfg)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertEqual(before, object_sha(song))
        self.assertEqual(cfg, {"meter_status": "reviewed"})

    def test_bad_config_is_rejected_including_bool_nonfinite_and_unknown(self):
        song = song_fixture()
        for config in ({"bars_per_block": True}, {"bars_per_block": 0}, {"bars_per_block": 4.0},
                       {"max_unit_seconds": False}, {"fallback_seconds": float("nan")},
                       {"fallback_seconds": float("inf")}, {"max_unit_seconds": "20"},
                       {"fallback_seconds": .0001}, {"meter_status": True}, {"unknown": 1}):
            with self.subTest(config=config), self.assertRaises(ValidationError):
                build_planning_units(song, config=config)

    def test_bad_song_clock_beat_and_section_contracts_fail(self):
        song = song_fixture(downbeats=[0, 2000, 4000])
        changes = [("clock_bool", lambda s: s["audio"].update(num_samples=True)),
                   ("duplicate", lambda s: s["music"].update(downbeats=[0, 0])),
                   ("not_beat", lambda s: s["music"].update(downbeats=[1])),
                   ("section_gap", lambda s: s["music"]["sections"][0].update(start=1))]
        for name, mutate in changes:
            changed = copy.deepcopy(song)
            mutate(changed)
            with self.subTest(name=name), self.assertRaises(ValidationError):
                build_planning_units(changed)

    def test_short_song_has_no_padded_expert_window_fallback(self):
        with self.assertRaisesRegex(ValidationError, "shorter"):
            build_planning_units(song_fixture(count=1000))

    def test_tampered_units_sources_types_and_extra_fields_fail_recomputation(self):
        song = song_fixture(downbeats=range(0, 30000, 2000))
        result = build_planning_units(song, config={"meter_status": "reviewed"})
        mutations = [lambda d: d["units"][0].update(end=d["units"][0]["end"] + 1),
                     lambda d: d["units"][0].update(semantic_phrase_verified=0),
                     lambda d: d.update(audio_sha256="b" * 64),
                     lambda d: d.update(extra=True),
                     lambda d: d["units"][0]["window_refs"][0].update(ownership_overlap=None),
                     lambda d: d["units"][0].update(bars=4.0)]
        for mutate in mutations:
            altered = copy.deepcopy(result)
            mutate(altered)
            with self.assertRaises(ValidationError):
                validate_planning_units(altered, song)
        changed_song = copy.deepcopy(song)
        changed_song["brief"] = "Changed source identity"
        with self.assertRaises(ValidationError):
            validate_planning_units(result, changed_song)

    def test_malformed_or_retimed_windows_are_rejected(self):
        song = song_fixture()
        windows = make_windows(30000, 1000)
        for mutate in (lambda w: w[0].update(index=False),
                       lambda w: w[0].update(ownership_end=w[0]["ownership_end"] - 1),
                       lambda w: w[0].update(boundaries_are_action_boundaries=0),
                       lambda w: w[0].update(extra=1)):
            changed = copy.deepcopy(windows)
            mutate(changed)
            with self.assertRaises(ValidationError):
                build_planning_units(song, changed)

    def test_deterministic_varied_grids_keep_capacity_partition_and_anchor_support(self):
        rng = random.Random(411)
        for case in range(40):
            count = rng.randrange(10000, 45000)
            downbeats = []
            anchor = rng.randrange(0, 1800)
            while anchor < count:
                downbeats.append(anchor)
                anchor += rng.randrange(500, 8500)
            cuts = sorted(rng.sample(range(1, count), 3))
            song = song_fixture(count, downbeats, cuts)
            with self.subTest(case=case):
                result = build_planning_units(song, config={"meter_status": "reviewed",
                    "bars_per_block": rng.choice([1, 2, 4, 8]), "max_unit_seconds": rng.choice([3, 10, 20])})
                self.assert_partition(result, song)
                real_bars = set(zip(downbeats, downbeats[1:]))
                for unit in result["units"]:
                    for bar in unit["bar_intervals"]:
                        self.assertIn((bar["start"], bar["end"]), real_bars)
                        self.assertLessEqual(unit["start"], bar["start"])
                        self.assertLessEqual(bar["end"], unit["end"])


if __name__ == "__main__":
    unittest.main()
