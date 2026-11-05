"""Synthetic literal schedule checks, not measured choreography quality."""
import copy

import pytest

from choreobench.io import object_sha
from choreobench.schedule_diagnostics import schedule_diagnostics
from choreobench.schema import ValidationError


def event(name, start, end, track="body"):
    return dict(event_id=name, start=start, end=end, track=track, action="Synthetic movement phrase.",
                attributes={}, entry={}, exit={}, motif_id=None, evidence=[])


def documents(bounds=(0, 120), rate=10, events=None):
    units = [dict(unit_id="u%03d" % i, start=start, end=end)
             for i, (start, end) in enumerate(zip(bounds, bounds[1:]))]
    doc = dict(schema="choreobench.planning_units.v1", song_id="synthetic", sample_rate=rate,
               num_samples=bounds[-1], units=units)
    plan = dict(schema="choreobench.plan.v1", song_id="synthetic", sample_rate=rate, num_samples=bounds[-1],
                motifs=[], events=events or [event("e0", 0, bounds[-1])], provenance={})
    return plan, doc


def balanced(**changes):
    result = dict(event_time_format="boundary_ids", event_schedule_policy="balanced_full_unit",
                  event_grid_seconds=1., min_event_seconds=.5, max_events_per_unit=3)
    result.update(changes)
    return result


def test_75_of_76_saturation_counts_actual_events_and_preserves_inputs():
    events = []
    bounds = list(range(0, 76 * 120 + 1, 120))
    for i, left in enumerate(bounds[:-1]):
        cuts = [left, left + 60, left + 120] if i == 38 else [left, left + 40, left + 80, left + 120]
        events.extend(event("e%03d_%d" % (i, j), start, end)
                      for j, (start, end) in enumerate(zip(cuts, cuts[1:])))
    plan, units = documents(bounds, events=events)
    cfg = balanced()
    before = copy.deepcopy((plan, units, cfg))
    report = schedule_diagnostics(plan, units, cfg)
    assert (plan, units, cfg) == before
    assert report["unit_count"] == 76 and report["event_count"] == 227
    assert report["units_at_configured_maximum"] == 75
    assert report["configured_maximum_saturation_fraction"] == 75 / 76
    assert report["configured_event_capacity"] == 228
    assert report["total_event_capacity_fraction"] == 227 / 228
    assert report["units_at_allowed_maximum"] == 75
    assert report["units"][38]["event_count"] == 2
    assert all(row["allowed_event_counts"] == [1, 2, 3] and row["matches_permitted_schedule"]
               for row in report["units"])
    assert report["track_counts"]["body"] == 227
    assert report["event_durations"]["minimum_seconds"] == 4
    assert report["event_durations"]["median_seconds"] == 4
    assert report["saturation_denominator"] == dict(unit="planning_unit", count=76)
    assert report["plan_sha256"] == object_sha(plan) and report["units_sha256"] == object_sha(units)
    assert not report["musical_or_physical_quality_verified"]


def test_real_off_grid_tail_constrains_allowed_count_separately_from_config_cap():
    # The last grid gap is 0.05 seconds. It must not create a second >=0.6s slot.
    plan, units = documents((0, 105), rate=100, events=[event("tail", 0, 105)])
    report = schedule_diagnostics(plan, units, balanced(event_grid_seconds=.5, min_event_seconds=.6))
    row = report["units"][0]
    assert row["allowed_event_counts"] == [1]
    assert row["matches_permitted_schedule"]
    assert row["end"] == 105 and row["event_durations"]["minimum_seconds"] == 1.05
    assert not row["at_configured_maximum"] and row["at_allowed_maximum"]
    assert report["configured_maximum_saturation_fraction"] == 0
    assert report["allowed_maximum_saturation_fraction"] == 1


def test_free_mode_does_not_claim_configured_counts_are_feasible():
    plan, units = documents((0, 120), events=[event("e0", 0, 50), event("e1", 50, 120)])
    report = schedule_diagnostics(plan, units, {})
    row = report["units"][0]
    assert report["event_schedule_policy"] == "free" and report["max_events_per_unit"] == 3
    assert row["configured_event_count_range"] == dict(minimum=1, maximum=3)
    assert row["allowed_event_counts"] is None and row["maximum_allowed_event_count"] is None
    assert row["matches_permitted_schedule"] is None and row["count_in_allowed_set"] is None
    assert row["allowed_counts_basis"] == "not_enumerated_configured_range_only"
    assert report["units_at_allowed_maximum"] is None and report["allowed_maximum_saturation_fraction"] is None
    assert report["event_durations"] == dict(minimum_samples=50, median_samples=60., minimum_seconds=5., median_seconds=6.)


def test_parallel_tracks_are_counted_as_events_not_invented_independent_units():
    plan, units = documents(events=[event("e1", 0, 120, "left_arm"), event("e2", 0, 120, "right_leg")])
    report = schedule_diagnostics(plan, units, {})
    assert report["unit_count"] == 1 and report["event_count"] == 2
    assert report["track_counts"]["left_arm"] == report["track_counts"]["right_leg"] == 1
    assert report["track_counts"]["body"] == 0
    assert report["units"][0]["event_ids"] == ["e1", "e2"]


def test_empty_unit_and_count_over_cap_are_reported_not_dropped():
    plan, units = documents((0, 60, 120), events=[event("one", 0, 30), event("two", 30, 60)])
    report = schedule_diagnostics(plan, units, dict(max_events_per_unit=1))
    assert report["event_count"] == 2 and report["unit_count"] == 2
    assert [row["event_count"] for row in report["units"]] == [2, 0]
    assert all(not row["count_within_configured_range"] for row in report["units"])
    assert report["units_with_no_events"] == 1
    assert report["units_outside_configured_count_range"] == 2
    assert report["units"][1]["event_durations"]["minimum_seconds"] is None
    assert report["units"][1]["all_events_meet_minimum_duration"] is None


def test_balanced_count_match_does_not_conceal_wrong_timing_or_short_action():
    plan, units = documents(events=[event("e0", 0, 1), event("e1", 1, 120)])
    report = schedule_diagnostics(plan, units, balanced())
    row = report["units"][0]
    assert row["count_in_allowed_set"]
    assert row["matches_permitted_schedule"] is False
    assert row["all_events_meet_minimum_duration"] is False
    assert row["event_durations"]["minimum_seconds"] == .1


def test_cross_unit_event_is_rejected_without_silent_reassignment():
    plan, units = documents((0, 60, 120), events=[event("crosses", 59, 61)])
    with pytest.raises(ValidationError, match="uniquely contained"):
        schedule_diagnostics(plan, units, {})


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(song_id="other"), lambda d: d.update(sample_rate=20),
    lambda d: d.update(num_samples=121), lambda d: d.update(sample_rate=True),
    lambda d: d.update(schema="other"), lambda d: d.update(units=[]),
    lambda d: d["units"][0].update(start=1), lambda d: d["units"][0].update(end=119),
    lambda d: d["units"][0].update(start=False), lambda d: d["units"][0].update(end=120.),
])
def test_mismatched_or_incomplete_partition_is_rejected(mutation):
    plan, units = documents()
    mutation(units)
    with pytest.raises(ValueError):
        schedule_diagnostics(plan, units, {})


@pytest.mark.parametrize("mutation", [
    lambda d: d["units"][1].update(start=59), lambda d: d["units"][1].update(start=61),
    lambda d: d["units"][1].update(unit_id="u000"), lambda d: d["units"].reverse(),
])
def test_overlap_gap_duplicate_id_and_reordered_units_are_rejected(mutation):
    plan, units = documents((0, 60, 120))
    mutation(units)
    with pytest.raises(ValueError):
        schedule_diagnostics(plan, units, {})


@pytest.mark.parametrize("config", [None, [], dict(max_events_per_unit=True), dict(max_events_per_unit=0),
    dict(max_events_per_unit=9), dict(event_schedule_policy="other"), dict(event_time_format="seconds"),
    dict(event_schedule_policy="balanced_full_unit"), dict(min_event_seconds=True), dict(min_event_seconds=-1),
    dict(event_grid_seconds=0), dict(event_grid_seconds=float("inf")), dict(min_event_seconds=float("nan"))])
def test_invalid_config_is_rejected(config):
    plan, units = documents()
    with pytest.raises(ValueError):
        schedule_diagnostics(plan, units, config)
