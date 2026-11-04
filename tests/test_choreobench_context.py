import copy

import pytest

from choreobench.context import build_boundary_context


def event(name, start, end, track, motif=None, exit=None):
    return dict(event_id=name, start=start, end=end, track=track, action="A planned movement",
                attributes={}, entry={}, exit=exit or {}, motif_id=motif, evidence=[])


def plan(events):
    return dict(schema="choreobench.plan.v1", song_id="song", sample_rate=16000, num_samples=100,
                motifs=[dict(motif_id="m", description="Declared step theme")],
                events=sorted(events, key=lambda e: (e["start"], e["track"], e["event_id"])), provenance={})


def test_boundary_uses_latest_end_per_body_part_not_last_two_starts():
    prefix = plan([event("arm", 0, 40, "left_arm", exit={"height": "raised"}),
                   event("leg1", 0, 10, "left_leg"), event("leg2", 10, 20, "left_leg"),
                   event("leg3", 20, 40, "left_leg")])
    context = build_boundary_context(prefix, 40)
    arm = context["body_parts"]["left_arm"]
    assert arm["latest_event"]["event_id"] == "arm"
    assert arm["latest_event"]["exit"] == {"height": "raised"}
    assert arm["touches_boundary"] and arm["gap_samples"] == 0
    assert not arm["state_at_boundary_verified"]
    assert context["body_parts"]["right_leg"]["latest_event"] is None


def test_body_scope_and_gaps_do_not_invent_pose_persistence():
    prefix = plan([event("whole", 0, 10, "body", exit={"support": None}),
                   event("upper", 10, 20, "upper")])
    before = copy.deepcopy(prefix)
    context = build_boundary_context(prefix, 30)
    assert context["body_parts"]["left_arm"]["latest_event"]["event_id"] == "upper"
    leg = context["body_parts"]["left_leg"]
    assert leg["latest_event"]["event_id"] == "whole" and leg["gap_samples"] == 20
    assert leg["latest_event"]["exit"]["support"] is None
    assert not leg["touches_boundary"]
    context["body_parts"]["left_leg"]["latest_event"]["exit"]["support"] = "changed"
    assert prefix == before


def test_motif_history_is_bounded_and_explicitly_unverified():
    prefix = plan([event("a", 0, 10, "body", "m"), event("b", 10, 20, "body", "m"),
                   event("c", 20, 30, "body", "m")])
    memory = build_boundary_context(prefix, 30)["motif_memory"][0]
    assert memory["prior_event_count"] == 3
    assert [e["event_id"] for e in memory["recent_examples"]] == ["b", "c"]
    assert memory["last_end"] == 30 and not memory["semantic_recurrence_verified"]
    assert build_boundary_context(plan([]), 0)["body_parts"]["head"]["latest_event"] is None


@pytest.mark.parametrize("boundary", [True, -1, 1.5, 101])
def test_bad_boundary_rejected(boundary):
    with pytest.raises(ValueError):
        build_boundary_context(plan([]), boundary)


def test_future_or_crossing_events_are_rejected():
    with pytest.raises(ValueError, match="crosses"):
        build_boundary_context(plan([event("future", 20, 40, "body")]), 30)
    with pytest.raises(ValueError, match="empty prefix"):
        build_boundary_context(plan([]), 30)
