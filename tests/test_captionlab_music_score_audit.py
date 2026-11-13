"""Synthetic stored-score fixtures test rubric logic, not music quality."""
import hashlib
import json

import pytest

from captionlab.music_score_audit import audit_ratings, main, proposed_score


def row(index, *, conflict="none", correspondence="explicit", full=False, score=4, status="valid", method="m"):
    return dict(item_sha256=hashlib.sha256(str(index).encode()).hexdigest(), case_id="case-" + str(index),
                method_id=method, status=status,
                rating=dict(conflict=conflict, correspondence=correspondence, full_support=full, music_fit=score))


def write_ratings(tmp_path, rows):
    source = tmp_path / "ratings.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return source


@pytest.mark.parametrize("correspondence,old", [("absent", 2), ("partial", 3), ("explicit", None)])
def test_all_uncertain_combinations_remain_unknown(tmp_path, correspondence, old):
    source = write_ratings(tmp_path, [row(1, conflict="uncertain", correspondence=correspondence, score=old)])
    original = source.read_bytes()
    result = audit_ratings(source, tmp_path / "audit")
    method = result["methods"]["m"]
    assert method["registered"] == 1 and method["proposed_valid_count"] == 0
    assert method["uncertain_mis_scored_count"] == int(old is not None)
    assert method["original_mean"] == old and method["proposed_mean"] is None
    assert source.read_bytes() == original
    assert result["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["model_calls"] == 0 and not result["quality_validated"]
    assert not result["historical_code_verified"] and not result["raw_model_responses_verified"]
    assert not result["expert_selection_change_established"]
    assert not result["generator_identities_verified"] and not result["generation_inventory_verified"]
    assert "blind alias" in result["grouping_scope"]


@pytest.mark.parametrize("correspondence", ["absent", "partial", "explicit"])
def test_clear_conflict_has_priority(correspondence):
    assert proposed_score(dict(conflict="clear", correspondence=correspondence, full_support=True)) == 1


@pytest.mark.parametrize("correspondence,full,expected", [
    ("absent", False, 2), ("partial", False, 3), ("explicit", False, 4), ("explicit", True, 5)])
def test_normal_scores_unchanged(tmp_path, correspondence, full, expected):
    source = write_ratings(tmp_path, [row(1, correspondence=correspondence, full=full, score=expected)])
    summary = audit_ratings(source, tmp_path / "audit")
    method = summary["methods"]["m"]
    assert method["original_mean"] == method["proposed_mean"] == expected
    assert method["changed_score_count"] == 0


def test_failures_unknown_and_missing_rating_keep_registered_slots(tmp_path):
    rows = [row(1), row(2, status="failed"), row(3, status="unknown"), row(4, method="n")]
    rows[3].pop("rating")
    source = write_ratings(tmp_path, rows)
    output = tmp_path / "audit"
    result = audit_ratings(source, output)
    method = result["methods"]["m"]
    assert result["registered"] == 4 and method["registered"] == 3
    assert method["original_valid_count"] == method["proposed_valid_count"] == 1
    assert method["proposed_unscored_count"] == 2
    assert result["methods"]["n"]["missing_rating_rows"] == 1
    stored = [json.loads(line) for line in (output / "row_audit.jsonl").read_text().splitlines()]
    assert [record["source_row"] for record in stored] == rows
    assert all(record["retained_in_registered_denominator"] for record in stored)
    assert stored[1]["proposed_score"] is None and not stored[1]["eligible_for_recompute"]


def test_duplicate_id_rejected_before_creating_output(tmp_path):
    source = write_ratings(tmp_path, [row(1), row(1, method="other")])
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="Duplicate item_sha256"):
        audit_ratings(source, output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    source = write_ratings(tmp_path, [row(1)])
    output = tmp_path / "audit"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        audit_ratings(source, output)
    assert sentinel.read_text() == "original"


@pytest.mark.parametrize("field,value", [("item_sha256", "invalid"), ("method_id", ""), ("status", "invented")])
def test_bad_basic_structure_rejected(tmp_path, field, value):
    record = row(1)
    record[field] = value
    source = write_ratings(tmp_path, [record])
    with pytest.raises(ValueError):
        audit_ratings(source, tmp_path / "audit")


def test_boolean_score_and_duplicate_json_key_rejected(tmp_path):
    source = write_ratings(tmp_path, [row(1, score=True)])
    with pytest.raises(ValueError, match="original music_fit"):
        audit_ratings(source, tmp_path / "audit")
    source.write_text('{"status":"valid","status":"unknown"}\n')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        audit_ratings(source, tmp_path / "audit")


def test_cli_writes_hypothetical_report(tmp_path, capsys):
    source = write_ratings(tmp_path, [row(1, conflict="uncertain", correspondence="partial", score=3)])
    output = tmp_path / "audit"
    assert main(["--ratings", str(source), "--out-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["hypothetical_rescoring"]
    assert json.loads((output / "summary.json").read_text())["original_scores_modified"] is False
