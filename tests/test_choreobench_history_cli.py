"""CPU-only integration through the public command entry points."""
from choreobench import harness
from choreobench.cli import main
from choreobench.io import read_json, write_json
from test_choreobench_ablations import ablation_planner, cfg
from test_choreobench_harness import fixture
from test_choreobench_review_analysis import material, A, B


def test_public_history_cli_prepares_no_calls_then_audits_actual_pair(tmp_path, monkeypatch, capsys):
    source, _, bank = fixture(tmp_path)
    requests = []
    ablation_planner(monkeypatch, requests)
    producer, shared = tmp_path / "producer", tmp_path / "shared.json"
    harness.run_harness(source, bank, producer, cfg())
    original_calls = len(requests)
    assert main(["export-outlines", "--run-dir", str(producer), "--out", str(shared)]) == 0
    config, experiment = tmp_path / "config.json", tmp_path / "experiment"
    write_json(config, cfg())
    assert main(["prepare-history", "--dataset", str(source), "--candidates", str(bank),
        "--config", str(config), "--outlines", str(shared), "--out-dir", str(experiment)]) == 0
    assert main(["verify-history", "--experiment", str(experiment)]) == 0
    assert len(requests) == original_calls
    missing = tmp_path / "missing.json"
    assert main(["audit-history", "--experiment", str(experiment), "--out", str(missing)]) == 2
    assert read_json(missing)["missing_or_failed_pair_count"] == 1
    commands = read_json(experiment / "commands.json")
    for row in commands["runs"]:
        assert row["ready_to_run"]
        assert main(row["argv"][3:]) == 0  # Omit python -m choreobench.
    report = tmp_path / "audit.json"
    assert main(["audit-history", "--experiment", str(experiment), "--out", str(report)]) == 0
    assert read_json(report)["paired_control_verified"]
    assert len(requests) == original_calls + 4
    assert all(r["response_kind"] == "coordination_unit" for r in requests[original_calls:])
    capsys.readouterr()


def test_public_review_analysis_cli_preserves_dataset_denominator(tmp_path, capsys):
    data, public, private, reviews = material(tmp_path, reviewed=False,
        include={("song1", A), ("song1", B), ("song2", A), ("song2", B)})
    output = tmp_path / "analysis.json"
    assert main(["analyze-reviews", "--dataset", str(data), "--public", str(public), "--private", str(private),
        "--reviews", str(reviews), "--method-a", A, "--method-b", B, "--criterion", "musical_fit",
        "--split", "dev", "--origin", "human", "--out", str(output)]) == 0
    result = read_json(output)
    assert result["coverage"]["expected_cells"] == 6
    assert result["coverage"]["rated_cells"] == 4
    assert len(result["coverage"]["missing_or_unrated_cells"]) == 2
    assert not result["grouping"]["usable_for_group_comparison"]
    assert result["comparison"]["result"] is None
    assert all(row["group_id"] is None for row in result["rows"])
    capsys.readouterr()
