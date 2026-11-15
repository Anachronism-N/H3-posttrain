"""CPU contracts for the batch runner; no model calls or external services."""
import json

import pytest

from choreobench import bar_agent_experiment as runner


def test_revision_views_keep_prefix_cost_and_plan_even_if_final_fails(tmp_path):
    runner.write(tmp_path / "before_rewrite.json", {"events": []})
    runner.write(tmp_path / "before_rewrite_costs.json", {
        "full_song_succeeded": True, "calls_used": 7, "input_tokens": 80, "output_tokens": 20})
    rows = runner.logical_rows({"arm": "W", "song_id": "s"}, tmp_path,
                               {"full_song_succeeded": False, "calls_used": 9}, True)
    assert [(r["arm"], r["complete"]) for r in rows] == [("W", True), ("W-plusR", False)]
    assert rows[0]["costs"]["calls_used"] == 7
    assert rows[1]["costs"]["calls_used"] == 9
    assert rows[0]["plan"].endswith("before_rewrite.json")
    assert rows[1]["plan"] is None


def test_missing_ablation_remains_in_registered_output(tmp_path):
    rows = runner.logical_rows({"arm": "O", "song_id": "s"}, tmp_path, {}, True)
    assert {r["arm"] for r in rows} == {"O", "O-noR"}
    assert all(not r["complete"] for r in rows)


def test_dry_run_does_not_dispatch_and_execute_requires_server_receipt(tmp_path, monkeypatch):
    contract = {"expected_runs": 4, "readiness": {k: False for k in runner.READY},
                "jobs": [], "songs": ["s"], "input_hashes": {}}
    monkeypatch.setattr(runner, "prepare", lambda *a: contract)
    result = runner.run("inventory", "provider", tmp_path / "run")
    assert result["dry_run"] is True
    assert not (tmp_path / "run" / "registry.json").exists()
    with pytest.raises(ValueError, match="preflight"):
        runner.run("inventory", "provider", tmp_path / "run", execute=True, resume=True)
