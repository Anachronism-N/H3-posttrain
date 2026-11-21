"""Co-support groups retain occurrences and never claim one physical action."""
import copy

from scripts_omnidance.exp.ms.motionscript_v2.selection import co_supported_groups


def event(sid, name, start, end, body="left_wrist", segment=0):
    return {"id": sid, "start": start, "end": end, "body_part": body, "action": "changes_local_position",
            "direction": "positive_axis", "source": "synthetic_rules",
            "measurement": {"start": 0., "end": .3, "delta": .3, "unit": "m"},
            "evidence": {"measurement_name": name, "coordinate_frame": "local_test_frame",
                         "required_joints": [0, 1], "required_joint_segment_ids": [segment, segment],
                         "continuity_scope": "independent_required_point_segments"}}


def test_overlap_groups_keep_full_original_intervals_and_do_not_delete_source():
    source = [event("x", "x_position", 0., 1.), event("y", "y_position", .4, 1.4)]
    original = copy.deepcopy(source)
    groups = co_supported_groups(source)
    assert len(groups) == 1
    assert groups[0]["start"] == .4 and groups[0]["end"] == 1.
    assert groups[0]["event_ids"] == ["x", "y"]
    assert not groups[0]["single_action_or_strict_simultaneity_certified"]
    assert not groups[0]["source_occurrences_deleted"]
    assert source == original
    assert groups[0]["channels"][0]["source_start"] == 0.
    assert groups[0]["channels"][1]["source_end"] == 1.4


def test_touching_events_and_same_channel_copies_are_not_compound_movements():
    assert co_supported_groups([event("a", "x", 0., 1.), event("b", "y", 1., 2.)]) == []
    assert co_supported_groups([event("a", "x", 0., 1.), event("b", "x", 0., 1.)]) == []


def test_other_sides_other_limbs_and_mismatched_segments_are_not_joined():
    a = event("a", "x", 0., 1.)
    for b in (event("b", "y", 0., 1., body="right_wrist"),
              event("b", "y", 0., 1., body="left_ankle"),
              event("b", "y", 0., 1., segment=1)):
        assert co_supported_groups([a, b]) == []


def test_repeated_occurrences_remain_separate_groups():
    source = [event("a", "x", 0., 1.), event("b", "y", 0., 1.),
              event("c", "x", 2., 3.), event("d", "y", 2., 3.)]
    groups = co_supported_groups(source)
    assert len(groups) == 2
    assert groups[0]["event_ids"] == ["a", "b"]
    assert groups[1]["event_ids"] == ["c", "d"]
