"""Synthetic caption audit fixtures; no media, model or cluster is accessed."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts_omnidance.exp.ms import ms_caption_audit as audit


def write(root, name, record):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def legacy(sid, sentences, **extra):
    return {"sid": sid, "ms": sentences, "source": "legacy_rules", **extra}


def v2(sid, *, caption=None, events=None, **extra):
    return {"sid": sid, "caption": ["Left elbow bends."] if caption is None else caption,
            "events": [] if events is None else events, "source": "v2_rules",
            "semantic_approval": False, **extra}


def event(*, source="measured_rules", action="bends", part="left_elbow", direction="flexion"):
    return {"source": source, "body_part": part, "action": action, "direction": direction}


def visual(video_hash, *, reviewed=False, descriptions=None):
    descriptions = ["Executes a grapevine sequence."] if descriptions is None else descriptions
    return {"schema": "motionscript_open_visual_v1",
            "source": "independent_video_open_vocabulary" + ("_reviewed" if reviewed else ""),
            "motion_caption": "A dancer executes a grapevine sequence.",
            "events": [{"id": f"event_{index}", "description": text, "frame_indices": [0, 1]}
                       for index, text in enumerate(descriptions)],
            "video_sha256": video_hash, "semantic_approval": reviewed}


def test_normalization_preserves_sides_numbers_and_time_ranges():
    assert audit.normalize_sentence("  LEFT  hand rises at 1.20–2.30 s. ") == "left hand rises at 1.20–2.30 s."
    values = ["Left hand rises at 1.0-2.0 s.", "Right hand rises at 1.0-2.0 s.",
              "Left hand rises at 2.0-3.0 s.", "Left hand rises 2 times.", "Left hand rises 3 times."]
    assert len({audit.normalize_sentence(value) for value in values}) == 5


def test_shared_sentence_is_not_duplicate_video_content(tmp_path):
    write(tmp_path, "a.json", legacy("a", ["Left arm raises."] * 3))
    write(tmp_path, "b.json", legacy("b", ["left arm raises.", "Right knee bends."]))
    report = audit.audit_directory(tmp_path)
    assert report["sample_count"] == 2
    assert report["independent_source_groups"] == 2
    assert report["matching_source_content_hashes"] == []
    summary = report["by_source"]["legacy_rules"]
    sentence = summary["top_sentences"][0]
    assert sentence["value"] == "left arm raises."
    assert sentence["sentence_occurrences"] == 4
    assert sentence["sample_presence"] == 2 and sentence["independent_group_presence"] == 2
    first = report["samples"][0]
    assert first["repeat_fraction"] == pytest.approx(2 / 3)
    assert first["longest_consecutive_run"] == 3
    assert summary["micro_repeat_fraction"] == pytest.approx(2 / 5)


def test_full_caption_equality_does_not_establish_source_content_duplicate(tmp_path):
    write(tmp_path, "a.json", legacy("a", ["Left elbow bends."], input_sha256="a" * 64))
    write(tmp_path, "b.json", legacy("b", ["Left elbow bends."], input_sha256="b" * 64))
    report = audit.audit_directory(tmp_path)
    assert report["independent_source_groups"] == 2
    assert report["matching_source_content_hashes"] == []


def test_shared_code_hash_cannot_group_unrelated_media(tmp_path):
    write(tmp_path, "a.json", legacy("a", ["Left elbow bends."], source_sha256="a" * 64))
    write(tmp_path, "b.json", legacy("b", ["Right knee bends."], source_sha256="a" * 64))
    report = audit.audit_directory(tmp_path)
    assert report["independent_source_groups"] == 2
    assert report["matching_source_content_hashes"] == []


def test_shared_sid_or_content_hash_forms_transitive_groups(tmp_path):
    write(tmp_path, "one/a.json", legacy("same_sid", ["Left elbow bends."]))
    write(tmp_path, "two/a.json", v2("same_sid", input_sha256="a" * 64))
    write(tmp_path, "three/b.json", legacy("other_sid", ["Right knee bends."], input_sha256="a" * 64))
    report = audit.audit_directory(tmp_path)
    assert report["unique_sid_count"] == 2
    assert report["independent_source_groups"] == 1
    assert report["source_groups"][0]["record_count"] == 3
    assert len(report["matching_source_content_hashes"]) == 1
    summary = report["by_source"]["legacy_rules"]
    assert summary["sample_count"] == 2 and summary["independent_source_groups"] == 1
    # These two distinct clauses coexist in one group; their group prevalence
    # does not claim that two independent source videos were seen.
    assert all(row["independent_group_presence"] == 1 for row in summary["top_sentences"])


def test_explicit_source_identity_groups_different_windows_without_caption_merge(tmp_path):
    write(tmp_path, "a.json", legacy("window_a", ["Left elbow bends."], source_id="video_1"))
    write(tmp_path, "b.json", legacy("window_b", ["Right knee bends."], video_id="video_1"))
    report = audit.audit_directory(tmp_path)
    assert report["independent_source_groups"] == 1
    assert report["matching_source_content_hashes"] == []
    assert report["samples"][0]["sentence_count"] == report["samples"][1]["sentence_count"] == 1


def test_source_and_event_domains_are_not_pooled_and_predicates_use_presence(tmp_path):
    write(tmp_path, "legacy.json", legacy("legacy", ["Left elbow bends."] * 4))
    write(tmp_path, "v1.json", v2("v1", events=[event()] * 5 + [event(source="other_sensor")]))
    write(tmp_path, "v2.json", v2("v2", events=[event()] * 7,
                                 semantic_approval=True))
    report = audit.audit_directory(tmp_path)
    assert set(report["by_source"]) == {"legacy_rules", "v2_rules"}
    assert "top_action_predicates" not in report  # No mixed-domain action frequency.
    source = report["by_source"]["v2_rules"]
    assert source["semantic_approval_counts"] == {"unapproved": 1, "approved": 1}
    measured = source["structured_events_by_source"]["measured_rules"]
    assert measured["event_count"] == 12
    assert measured["top_action_predicates"][0]["sample_presence"] == 2
    assert measured["top_action_predicates"][0]["body_part"] == "left_elbow"
    other = source["structured_events_by_source"]["other_sensor"]
    assert other["event_count"] == 1
    assert other["top_action_predicates"][0]["sample_presence"] == 1
    assert source["structured_events_by_source"]["measured_rules"]["body_part_coverage"][0]["value"] == "left_elbow"


def test_lexical_counts_are_sample_presence_and_retain_body_side(tmp_path):
    write(tmp_path, "a.json", legacy("a", ["The left elbow bends."] * 9 + ["The right elbow extends."]))
    summary = audit.audit_directory(tmp_path)["by_source"]["legacy_rules"]
    actions = {row["value"]: row["sample_presence"] for row in summary["lexical_action_mentions"]}
    assert actions == {"bend": 1, "extend": 1}
    parts = {row["value"] for row in summary["lexical_body_part_coverage"]}
    assert parts == {"left_elbow", "right_elbow"}


def test_body_coverage_denominator_includes_abstention_records(tmp_path):
    write(tmp_path, "one.json", v2("one", events=[event()]))
    write(tmp_path, "abstain.json", v2("abstain", events=[], caption=[]))
    summary = audit.audit_directory(tmp_path)["by_source"]["v2_rules"]
    coverage = summary["structured_events_by_source"]["measured_rules"]["body_part_coverage"][0]
    assert coverage["sample_presence"] == 1
    assert coverage["sample_fraction"] == .5


def test_sentence_segmentation_does_not_split_decimal_times_or_join_blanks(tmp_path):
    write(tmp_path, "a.json", legacy("a", ["Left hand moves from 1.25 to 2.50 s. Right hand moves.",
                                            "", "Right hand moves."]))
    report = audit.audit_directory(tmp_path)
    row = report["samples"][0]
    assert row["sentence_count"] == 3
    assert row["unique_sentence_count"] == 2
    assert row["longest_consecutive_run"] == 1
    assert row["blank_sentence_entries"] == 1


def test_empty_unknown_and_malformed_records_are_reported(tmp_path):
    write(tmp_path, "empty.json", legacy("empty", None))
    write(tmp_path, "blank.json", v2("blank", caption=["  "]))
    write(tmp_path, "manifest.json", {"records": [{"sid": "not_a_sample"}]} )
    write(tmp_path, "bad_schema.json", legacy("bad", [1]))
    (tmp_path / "bad_json.json").write_text("{", encoding="utf-8")
    report = audit.audit_directory(tmp_path)
    assert report["sample_count"] == report["empty_description_count"] == 2
    assert report["ignored_json_files"] == ["manifest.json"]
    assert len(report["parse_errors"]) == 2
    assert report["by_source"]["legacy_rules"]["top_sentences"] == []
    json.dumps(report, allow_nan=False)


def test_top_k_is_per_source_and_invalid_options_fail(tmp_path):
    write(tmp_path, "a.json", legacy("a", ["Left arm bends.", "Right arm bends."]))
    report = audit.audit_directory(tmp_path, top_k=1)
    assert len(report["by_source"]["legacy_rules"]["top_sentences"]) == 1
    for value in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            audit.audit_directory(tmp_path, top_k=value)


def test_cli_refuses_input_tree_and_existing_file(tmp_path):
    source = tmp_path / "input"
    original = write(source, "a.json", legacy("a", ["Left arm bends."]))
    before = original.read_bytes()
    args = argparse.Namespace(input=str(source), out=str(source / "audit.json"), top_k=20)
    with pytest.raises(ValueError, match="outside the input tree"):
        audit.run(args)
    args.out = str(original)
    with pytest.raises(ValueError, match="outside the input tree"):
        audit.run(args)
    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    args.out = str(existing)
    with pytest.raises(FileExistsError, match="already exists"):
        audit.run(args)
    assert original.read_bytes() == before
    assert existing.read_text() == "keep"


def test_cli_runs_without_site_packages_and_keeps_inputs_unchanged(tmp_path):
    source = tmp_path / "input"
    original = write(source, "a.json", legacy("a", ["Left arm bends."] * 3))
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    output = tmp_path / "report.json"
    completed = subprocess.run([sys.executable, "-S", str(Path(audit.__file__).resolve()),
                                "--input", str(source), "--out", str(output), "--top-k", "5"],
                               text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["sample_count"] == 1
    assert report["training_weights_emitted"] is False
    assert report["semantic_approval_granted"] is False
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    assert report["input_json_sha256"] == [{"path": "a.json", "sha256": digest}]


def test_audit_is_deterministic_and_does_not_infer_semantic_approval(tmp_path):
    write(tmp_path, "a.json", legacy("a", ["The arm does not bend."]))
    before = audit.audit_directory(tmp_path)
    after = audit.audit_directory(tmp_path)
    assert before == after
    assert before["by_source"]["legacy_rules"]["semantic_approval_counts"] == {"unspecified": 1}
    assert "not positive action facts" in before["definitions"]["lexical_limitations"]
    assert "Synonym expansion does not add new actions" in before["training_recommendations"][-1]


def test_visual_bundle_missing_sid_uses_full_relative_path(tmp_path):
    write(tmp_path, "bundle_a/candidate.json", visual("a" * 64))
    write(tmp_path, "bundle_b/candidate.json", visual("b" * 64))
    write(tmp_path, "bundle_a/manifest.json", {"schema": "motionscript_open_visual_v1", "video_sha256": "a" * 64})
    report = audit.audit_directory(tmp_path)
    assert report["sample_count"] == 2
    assert report["independent_source_groups"] == 2
    assert {row["sid"] for row in report["samples"]} == {"bundle_a/candidate.json", "bundle_b/candidate.json"}
    assert all(row["sid_origin"] == "relative_record_path_fallback" for row in report["samples"])
    assert report["ignored_json_files"] == ["bundle_a/manifest.json"]
    assert not report["parse_errors"]


def test_visual_candidate_and_reviewed_export_link_only_by_video_hash(tmp_path):
    write(tmp_path, "bundle/candidate.json", visual("a" * 64))
    write(tmp_path, "bundle/export.json", visual("a" * 64, reviewed=True))
    report = audit.audit_directory(tmp_path)
    assert report["sample_count"] == 2
    assert report["independent_source_groups"] == 1
    assert len(report["by_source"]) == 2
    assert report["by_source"]["independent_video_open_vocabulary"]["semantic_approval_counts"] == {"unapproved": 1}
    reviewed = report["by_source"]["independent_video_open_vocabulary_reviewed"]
    assert reviewed["semantic_approval_counts"] == {"approved": 1}
    assert len(report["matching_source_content_hashes"]) == 1
    assert report["semantic_approval_granted"] is False  # Merely reports the input flag.


def test_open_event_text_is_counted_without_closed_predicate_fields(tmp_path):
    description = "Executes a grapevine sequence."
    write(tmp_path, "a/candidate.json", visual("a" * 64, descriptions=[description] * 5))
    write(tmp_path, "b/candidate.json", visual("b" * 64, descriptions=[description]))
    report = audit.audit_directory(tmp_path)
    summary = report["by_source"]["independent_video_open_vocabulary"]
    assert summary["structured_events_by_source"] == {}
    opened = summary["open_event_descriptions_by_source"]["independent_video_open_vocabulary"]
    assert opened["event_count"] == 6
    assert opened["top_descriptions"][0]["sample_presence"] == 2
    assert opened["top_descriptions"][0]["independent_group_presence"] == 2
    assert opened["top_descriptions"][0]["value"] == description.lower()
    assert opened["lexical_action_mentions"] == []
    assert all(row["skipped_events"] == 0 for row in report["samples"])
    assert "not semantic action-class diversity or open-vocabulary coverage" in opened["coverage_limitation"]


def test_free_description_sources_and_side_time_differences_are_retained(tmp_path):
    item = visual("a" * 64, descriptions=["Left hand circles at 1-2 s.", "Right hand circles at 1-2 s.",
                                           "Left hand circles at 2-3 s."])
    item["events"].append({"id": "external", "description": "Left hand circles at 1-2 s.",
                           "frame_indices": [0, 1], "source": "another_observer"})
    write(tmp_path, "candidate.json", item)
    summary = audit.audit_directory(tmp_path)["by_source"]["independent_video_open_vocabulary"]
    buckets = summary["open_event_descriptions_by_source"]
    assert len(buckets) == 2
    assert buckets["independent_video_open_vocabulary"]["unique_normalized_descriptions"] == 3
    assert buckets["another_observer"]["unique_normalized_descriptions"] == 1


def test_compact_caption_is_paired_comparison_without_extra_samples(tmp_path):
    write(tmp_path, "one.json", v2("one", caption=["Left elbow bends."] * 4,
                                  compact_caption=["Left elbow repeatedly bends."]))
    write(tmp_path, "two.json", v2("two", caption=["Right knee bends."]))
    report = audit.audit_directory(tmp_path)
    assert report["sample_count"] == report["independent_source_groups"] == 2
    summary = report["by_source"]["v2_rules"]
    assert summary["sentence_count"] == 5
    compared = summary["compact_caption_comparison"]
    assert compared["paired_sample_count"] == 1
    assert compared["additional_independent_samples"] == 0
    assert compared["original_caption"]["sample_count"] == compared["compact_caption"]["sample_count"] == 1
    assert compared["original_caption"]["micro_repeat_fraction"] == .75
    assert compared["compact_caption"]["micro_repeat_fraction"] == 0
    assert report["samples"][0]["compact_caption"]["sentence_count"] == 1
    assert report["samples"][0]["sentence_count"] == 4
    assert "compact_caption" not in report["samples"][1]


def test_empty_compact_caption_is_not_silently_dropped_from_comparison(tmp_path):
    write(tmp_path, "one.json", v2("one", compact_caption=[]))
    report = audit.audit_directory(tmp_path)
    compared = report["by_source"]["v2_rules"]["compact_caption_comparison"]
    assert compared["paired_sample_count"] == 1
    assert compared["compact_caption"]["empty_description_count"] == 1
    assert report["sample_count"] == 1
