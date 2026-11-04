"""Independent regression probes for submissions and the journaled edit runner.

All audio/providers are tiny synthetic test fixtures. No external API is called.
"""

import copy
import json
import wave
from pathlib import Path

import pytest

from choreobench import baseline, edit_runner, submissions
from choreobench.io import file_sha, json_text, object_sha, read_json, write_json


def fixture_song(directory, song_id="song_a"):
    wav = directory / (song_id + ".wav")
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(100)
        stream.writeframes(b"\0\0" * 18000)
    return {"song_id": song_id, "split": "test", "song_group_id": "sg_" + song_id,
            "artist_group_id": "ag_" + song_id, "near_duplicate_group_id": "ng_" + song_id,
            "brief": "Synthetic fixture, not a scientific result.",
            "audio": {"path": wav.name, "sha256": file_sha(wav), "sample_rate": 100, "num_samples": 18000},
            "music": {"source": "unavailable", "beats": [], "downbeats": [], "sections": [
                {"section_id": "s0", "start": 0, "end": 18000, "label": "unknown", "repeat_group": None}]}}


def fixture_plan(song):
    def event(event_id, start, end):
        return {"event_id": event_id, "start": start, "end": end, "track": "body",
                "action": "Hold still.", "attributes": {"side": "right", "rest": True},
                "entry": {}, "exit": {"facing": "front"}, "motif_id": None,
                "evidence": ["PRIVATE_EVIDENCE_NOT_FOR_EDITOR"]}
    return {"schema": "choreobench.plan.v1", "song_id": song["song_id"],
            "sample_rate": 100, "num_samples": 18000, "motifs": [],
            "events": [event("e0", 0, 9000), event("e1", 9000, 18000)],
            "provenance": {"private_path": "PRIVATE_PROVENANCE_NOT_FOR_EDITOR"}}


def fixture_dataset(songs):
    # synthetic=False is deliberately a declaration for testing the eligibility
    # combination. These unit-test fixtures must never be reported as real data.
    return {"schema": "choreobench.dataset.v1", "dataset_id": "submission_review_fixture",
            "synthetic": False, "grouping": {"status": "reviewed", "evidence": "Unit-test declaration only"},
            "songs": songs}


def write_submission(directory, dataset_path, song, *, synthetic=False):
    plan_path = directory / "prediction.json"
    write_json(plan_path, fixture_plan(song))
    path = directory / "submission.json"
    write_json(path, {"schema": "choreobench.submission.v1", "method_id": "fixture_method",
                      "dataset_sha256": file_sha(dataset_path), "synthetic": synthetic,
                      "predictions": [{"song_id": song["song_id"], "path": plan_path.name,
                                       "sha256": file_sha(plan_path)}]})
    return path, plan_path


def test_missing_and_invalid_predictions_stay_in_song_denominator(tmp_path):
    songs = [fixture_song(tmp_path, song_id) for song_id in ("song_a", "song_b")]
    dataset_path = tmp_path / "dataset.json"
    write_json(dataset_path, fixture_dataset(songs))
    submission_path, plan_path = write_submission(tmp_path, dataset_path, songs[0])
    report = submissions.evaluate_submission(dataset_path, submission_path)
    assert report["selected_songs"] == 2 and report["valid_plan_rate"] == 0.5
    assert report["songs"][1]["status"] == "missing"
    plan_path.write_text('{"invalid":"not a plan"}', encoding="utf-8")
    report = submissions.evaluate_submission(dataset_path, submission_path)
    assert report["selected_songs"] == 2 and report["valid_plan_rate"] == 0
    assert report["missing_or_invalid_plans"] == 2


def test_dataset_mutation_cannot_bind_old_denominator_to_new_manifest_hash(tmp_path, monkeypatch):
    first, second = fixture_song(tmp_path), fixture_song(tmp_path, "song_b")
    dataset_path = tmp_path / "dataset.json"
    old_dataset, new_dataset = fixture_dataset([first]), fixture_dataset([first, second])
    write_json(dataset_path, new_dataset)
    submission_path, _ = write_submission(tmp_path, dataset_path, first)
    dataset_path.write_text(json_text(old_dataset), encoding="utf-8")
    original_snapshot = submissions.read_json_snapshot

    def mutate_after_snapshot(path):
        consumed = original_snapshot(path)
        if Path(path).resolve() == dataset_path.resolve():
            dataset_path.write_text(json_text(new_dataset), encoding="utf-8")
        return consumed

    monkeypatch.setattr(submissions, "read_json_snapshot", mutate_after_snapshot)
    try:
        report = submissions.evaluate_submission(dataset_path, submission_path)
    except ValueError:
        return  # Rejecting the change is also a valid outcome.
    assert report["dataset_sha256"] == file_sha(dataset_path)
    assert report["selected_songs"] == 2, "a two-song manifest SHA must not carry a one-song denominator"


def test_prediction_mutation_cannot_evaluate_unsubmitted_bytes(tmp_path, monkeypatch):
    selected_song = fixture_song(tmp_path)
    dataset_path = tmp_path / "dataset.json"
    write_json(dataset_path, fixture_dataset([selected_song]))
    submission_path, plan_path = write_submission(tmp_path, dataset_path, selected_song)
    original_plan = read_json(plan_path)
    modified_plan = copy.deepcopy(original_plan)
    modified_plan["events"][0]["action"] = "Unsubmitted action after file hashing."
    original_sha = submissions.file_sha
    changed = []

    def mutate_after_hash(path):
        result = original_sha(path)
        if Path(path).resolve() == plan_path.resolve() and not changed:
            changed.append(True)
            plan_path.write_text(json_text(modified_plan), encoding="utf-8")
        return result

    monkeypatch.setattr(submissions, "file_sha", mutate_after_hash)
    try:
        report = submissions.evaluate_submission(dataset_path, submission_path)
    except ValueError:
        return
    if report["songs"][0]["valid"]:
        assert report["songs"][0]["diagnostics"]["plan_sha256"] == object_sha(original_plan), \
            "diagnostics must describe the bytes bound by the submitted file hash"


def test_synthetic_predictions_block_scientific_release_even_for_real_declared_data(tmp_path):
    selected_song = fixture_song(tmp_path)
    dataset_path = tmp_path / "dataset.json"
    write_json(dataset_path, fixture_dataset([selected_song]))
    submission_path, _ = write_submission(tmp_path, dataset_path, selected_song, synthetic=True)
    report = submissions.evaluate_submission(dataset_path, submission_path)
    assert report["synthetic"] is True
    assert not report["release_contract"]["eligible_by_declared_contract"], \
        "synthetic predictions cannot inherit scientific eligibility from the dataset"


def edit_inputs(directory):
    song = fixture_song(directory)
    source = fixture_plan(song)
    contract = {"schema": "choreobench.edit_task.v1", "task_id": "edit_fixture", "song_id": song["song_id"],
                "source_plan_sha256": object_sha(source), "instruction": "Use the left arm in the first event.",
                "allowed_event_ids": ["e0"], "repair_event_ids": [],
                "protected": [{"event_id": "*", "field": "exit"}],
                "goals": [{"event_id": "e0", "field": "attributes.side", "op": "eq", "value": "PRIVATE_LITERAL_GOAL"}],
                "semantic_goals": ["PRIVATE_SEMANTIC_GOAL"]}
    patch = {"schema": "choreobench.patch.v1", "base_plan_sha256": object_sha(source),
             "instruction": contract["instruction"], "allowed_event_ids": ["e0"], "repair_event_ids": [],
             "protected": copy.deepcopy(contract["protected"]),
             "operations": [{"event_id": "e0", "field": "attributes.side", "value": "left"}]}
    source_path, task_path = directory / "source.json", directory / "task.json"
    write_json(source_path, source)
    write_json(task_path, contract)
    return source_path, task_path, patch


class FixtureEditor:
    def __init__(self, proposal, identity_suffix="one"):
        self.identity = {"kind": "review_fixture", "model": "fixture-" + identity_suffix, "synthetic": True}
        self.proposal = proposal
        self.requests = []

    def generate(self, request):
        self.requests.append(copy.deepcopy(request))
        return {"text": json.dumps(self.proposal), "identity": copy.deepcopy(self.identity), "synthetic": True,
                "raw": {"fixture": True}, "stop_reason": "stop",
                "costs": {"output_tokens": 100, "input_tokens": 200, "latency_seconds": 0.0}}


def test_edit_runner_hides_private_goals_and_source_metadata_and_reuses_raw(tmp_path, monkeypatch):
    source_path, task_path, proposal = edit_inputs(tmp_path)
    provider = FixtureEditor(proposal)
    monkeypatch.setattr(edit_runner, "create_provider", lambda *args, **kwargs: provider)
    output = tmp_path / "edit_run"
    result = edit_runner.propose_edit(source_path, task_path, {}, output)
    assert result["complete"] and result["semantic_success"] is None
    assert len(provider.requests) == 1
    payload = json.dumps(provider.requests[0])
    for secret in ("PRIVATE_LITERAL_GOAL", "PRIVATE_SEMANTIC_GOAL", "PRIVATE_PROVENANCE_NOT_FOR_EDITOR", "PRIVATE_EVIDENCE_NOT_FOR_EDITOR"):
        assert secret not in payload
    again = edit_runner.propose_edit(source_path, task_path, {}, output, resume=True)
    assert result == again and len(provider.requests) == 1


def test_edit_runner_records_expanded_scope_failure_without_hidden_retry(tmp_path, monkeypatch):
    source_path, task_path, proposal = edit_inputs(tmp_path)
    proposal["allowed_event_ids"].append("e1")
    provider = FixtureEditor(proposal)
    monkeypatch.setattr(edit_runner, "create_provider", lambda *args, **kwargs: provider)
    output = tmp_path / "edit_run"
    for resume in (False, True):
        with pytest.raises(ValueError, match="frozen"):
            edit_runner.propose_edit(source_path, task_path, {}, output, resume=resume)
    assert len(provider.requests) == 1
    assert read_json(output / "results.json")["complete"] is False
    assert len(list((output / "calls").glob("*.raw.json"))) == 1


def test_edit_runner_resume_rejects_changed_private_task_or_provider(tmp_path, monkeypatch):
    source_path, task_path, proposal = edit_inputs(tmp_path)
    provider = FixtureEditor(proposal)
    monkeypatch.setattr(edit_runner, "create_provider", lambda *args, **kwargs: provider)
    output = tmp_path / "edit_run"
    edit_runner.propose_edit(source_path, task_path, {}, output)
    changed_provider = FixtureEditor(proposal, "different")
    monkeypatch.setattr(edit_runner, "create_provider", lambda *args, **kwargs: changed_provider)
    with pytest.raises(ValueError, match="identity changed"):
        edit_runner.propose_edit(source_path, task_path, {}, output, resume=True)
    assert not changed_provider.requests
    monkeypatch.setattr(edit_runner, "create_provider", lambda *args, **kwargs: provider)
    changed_task = read_json(task_path)
    changed_task["goals"][0]["value"] = "POST_HOC_PRIVATE_GOAL"
    task_path.write_text(json_text(changed_task), encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        edit_runner.propose_edit(source_path, task_path, {}, output, resume=True)
    assert len(provider.requests) == 1


def test_unknown_outcome_after_provider_call_is_not_automatically_retried(tmp_path, monkeypatch):
    source_path, task_path, proposal = edit_inputs(tmp_path)
    provider = FixtureEditor(proposal)
    monkeypatch.setattr(edit_runner, "create_provider", lambda *args, **kwargs: provider)
    original_write = baseline.write_json

    def interrupt_raw_write(path, *args, **kwargs):
        if str(path).endswith(".raw.json"):
            raise RuntimeError("simulated crash before raw publication")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(baseline, "write_json", interrupt_raw_write)
    output = tmp_path / "edit_run"
    with pytest.raises(RuntimeError, match="simulated crash"):
        edit_runner.propose_edit(source_path, task_path, {}, output)
    monkeypatch.setattr(baseline, "write_json", original_write)
    with pytest.raises(ValueError, match="Pending/unknown"):
        edit_runner.propose_edit(source_path, task_path, {}, output, resume=True)
    assert len(provider.requests) == 1
