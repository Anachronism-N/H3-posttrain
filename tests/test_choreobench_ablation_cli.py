"""Exercise the public preparation CLI without contacting providers."""
import json
from pathlib import Path

import pytest

from choreobench.cli import main
from choreobench.experiments import verify_ablation_experiment
from choreobench.harness import run_harness
from choreobench.io import read_json
from test_choreobench_experiments import setup_experiment, forbid_providers
from test_choreobench_ablations import ablation_planner


@pytest.mark.parametrize("changes,expected", [({}, 0), ({"max_calls": 0}, 2), ({"min_event_seconds": 100}, 2)])
def test_prepare_cli_exit_status_has_source_bound_manifests(tmp_path, monkeypatch, capsys, changes, expected):
    source, bank, config, _ = setup_experiment(tmp_path, **changes)
    forbid_providers(monkeypatch)
    output = tmp_path / "prepared"
    assert main(["prepare-ablation", "--dataset", str(source), "--candidates", str(bank),
                 "--config", str(config), "--out-dir", str(output)]) == expected
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["prepared"] and not manifest["executed"]
    assert manifest["ready_to_run"] is (expected == 0)
    assert verify_ablation_experiment(output) == manifest
    assert all(not Path(row["run_dir"]).exists() for row in manifest["variants"])


def test_prepared_four_arm_configs_execute_with_stub_and_resume_without_calls(tmp_path, monkeypatch):
    source, bank, config, _ = setup_experiment(tmp_path)
    output = tmp_path / "prepared"
    assert main(["prepare-ablation", "--dataset", str(source), "--candidates", str(bank),
                 "--config", str(config), "--out-dir", str(output)]) == 0
    manifest = verify_ablation_experiment(output)
    for row in manifest["variants"]:
        requests = []
        ablation_planner(monkeypatch, requests)
        cfg = read_json(row["config_path"])
        result = run_harness(source, bank, row["run_dir"], cfg)
        assert result["all_songs_succeeded"]
        assert result["calls_used"] == row["expected_new_text_calls"]
        assert result["ablation"] == dict(use_outline=row["use_outline"], use_history=row["use_history"])
        assert run_harness(source, bank, row["run_dir"], cfg, resume=True) == result
        assert len(requests) == result["calls_used"]
