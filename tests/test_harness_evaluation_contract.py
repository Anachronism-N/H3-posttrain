"""Regression cases for evaluation contracts; no model or cluster access."""
import json
import wave

import pytest

from experiments.structured_dev_20261114 import (
    controls, structure_adherence, energy_alignment, human_analyze_compact,
    report_multi, evaluation_contract as ec,
)


def event(text="Raise the right arm.", start=0, end=4):
    return dict(event_id="e0", start=start, end=end, track="body", action=text, motif_id="private")


@pytest.mark.parametrize("module", [controls, structure_adherence])
def test_summary_control_is_a_derangement_and_preserves_evidence(module):
    source = dict(blocks=[
        dict(block_id=str(i), start=i, end=i + 1, relative_energy_db=i,
             onset_rate_per_second=i * 2, energy_trend=str(i)) for i in range(5)], relations=[])
    before = json.dumps(source)
    changed = module.shuffled(source)
    assert json.dumps(source) == before
    assert sorted(b["relative_energy_db"] for b in changed["blocks"]) == list(range(5))
    assert all(i != j for i, j in enumerate(changed["_control_permutation"]))
    assert [b["start"] for b in changed["blocks"]] == list(range(5))
    assert module.shuffled(dict(blocks=[], relations=[]))["_control_identifiable"] is False


def test_spearman_uses_average_tied_ranks():
    assert energy_alignment.spearman([1, 1, 2, 3], [1, 2, 2, 3]) == pytest.approx(5 / 6)
    assert energy_alignment.spearman([1, 1], [1, 2]) is None
    assert energy_alignment.spearman([], []) is None


def test_kappa_uses_separate_marginals_and_retains_undefined():
    assert human_analyze_compact.cohen_kappa([True] * 4 + [False], [True] + [False] * 4) == pytest.approx(2 / 17)
    assert human_analyze_compact.cohen_kappa([], []) is None
    assert human_analyze_compact.cohen_kappa([True], [True]) is None


def test_full_text_and_explicit_global_seconds():
    text = "A long movement description. " * 200
    row = ec.compact([event(text)], limit=0)[0]
    assert row["action"] == text and "motif" not in row
    assert ec.compact([event(text)], limit=12)[0]["action"] == text[:12]
    with pytest.raises(ValueError):
        ec.compact([event()], limit=-1)
    pair = dict(pair_id="p", song_id="s", comparison="x", A=[event()], B=[event()])
    with pytest.raises(ValueError, match="time_unit"):
        ec.check_pair(pair)
    ec.check_pair(dict(pair, time_unit="seconds"))


def test_audio_crop_binds_frame_duration_offset_and_hash(tmp_path):
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(b"\0\0" * 16000)
    scope = ec.audio_scope(path, offset=30, expected_end=32)
    assert scope["audio_global_start_seconds"] == 30
    assert scope["audio_global_end_seconds"] == 32
    assert scope["server_encoding_verified"] is False
    visible = ec.compact([event(start=29, end=31)], scope=scope)[0]
    assert (visible["start"], visible["end"]) == (29, 31)
    assert visible["observed_interval_seconds"] == [30, 31]
    assert ec.compact([event(start=0, end=4)], scope=scope) == []
    with pytest.raises(ValueError, match="frames"):
        ec.audio_scope(path, offset=30, expected_end=33)


def scores():
    return dict(**{c: 3 for c in ec.CRITERIA}, reason="e0 specifies a sustained action.")


@pytest.mark.parametrize("bad", [True, 0, 6, 3.0, "3", None])
def test_invalid_scores_are_not_silently_accepted(bad):
    value = scores()
    value["musical_fit"] = bad
    with pytest.raises(ValueError):
        ec.validate_pairwise(dict(A=value, B=scores(), preferred="A"))


def test_order_mapping_and_tie_consistency():
    value = dict(A=scores(), B=scores(), preferred="A")
    assert ec.swapped(value)["preferred"] == "B"
    other = ec.swapped(value)
    other["A"]["musical_fit"] = 2
    assert not ec.consistency(value, other)["musical_fit"]["consistent"]
    with pytest.raises(ValueError):
        ec.strict_json('{"A":1,"A":2}')
    with pytest.raises(ValueError):
        ec.response_text(dict(choices=[dict(finish_reason="length", message=dict(content="{}"))]))


def test_audit_logs_keep_bad_raw_responses_and_forbid_overwrite(tmp_path, monkeypatch):
    class Reply:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b"not-json-from-server"
    monkeypatch.setattr(ec.urllib.request, "urlopen", lambda *a, **k: Reply())
    out = tmp_path / "review.json"
    audit = ec.AuditLog(out)
    with pytest.raises(ValueError):
        audit.request("http://example.invalid/v1", "fake", [], max_tokens=10, seed=42,
                      call_id="p/0", safe_payload={"time_unit": "seconds"})
    logged = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert logged[1]["raw_response"] == "not-json-from-server"
    with pytest.raises(FileExistsError):
        ec.AuditLog(out)


def test_terminal_execution_is_not_success_and_zero_cost_is_kept(tmp_path):
    run = tmp_path / "s"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(dict(complete=True, full_song_succeeded=False,
        total_calls=0, calls_used=99, input_tokens=0, output_tokens=0)))
    result = report_multi.facts(tmp_path, "s")
    assert result["status"] == "failed"
    assert result["full_song_succeeded"] is False
    assert result["calls"] == 0


def test_success_without_plan_is_recorded_as_failure(tmp_path):
    run = tmp_path / "s"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(dict(complete=True, full_song_succeeded=True)))
    result = report_multi.facts(tmp_path, "s")
    assert result["status"] == "invalid_plan" and not result["full_song_succeeded"]


def test_new_rubric_enforces_frozen_applicability_and_evidence():
    from experiments.structured_dev_20261114 import judge_rubric_v2 as rubric
    payload = dict(dimensions={"D6": rubric.DIMENSIONS["local6"]["D6"]},
                   D6_applicable=False, events=[event()],
                   audio_scope=dict(audio_global_start_seconds=0, audio_global_end_seconds=4))
    row = dict(status="not_applicable", score=None, reason="Frozen stable-audio observation.",
               quote="", event_ids=[], music_evidence=[])
    assert rubric.validate_ratings(dict(ratings={"D6": row}), payload)
    row.update(status="scored", score=5, quote="Raise", event_ids=["e0"],
               music_evidence=[dict(start=0, end=2, observation="Audible rise.")])
    with pytest.raises(ValueError, match="applicability"):
        rubric.validate_ratings(dict(ratings={"D6": row}), payload)
    payload["D6_applicable"] = True
    assert rubric.validate_ratings(dict(ratings={"D6": row}), payload)
    row["music_evidence"][0]["end"] = 5
    with pytest.raises(ValueError, match="outside"):
        rubric.validate_ratings(dict(ratings={"D6": row}), payload)
    row["music_evidence"][0]["end"] = 2
    row["quote"] = "Invented turn"
    with pytest.raises(ValueError, match="Quote"):
        rubric.validate_ratings(dict(ratings={"D6": row}), payload)
