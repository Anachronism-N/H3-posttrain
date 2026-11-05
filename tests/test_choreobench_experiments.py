"""Preparation is CPU/file work only; fixtures never imply model quality."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from choreobench import baseline, coordination, experiments, harness, providers
from choreobench.candidates import export_candidates
from choreobench.io import file_sha, object_sha, read_json, write_json
from test_choreobench_harness import fixture
from test_choreobench_coordination import sidecar
from test_choreobench_baseline import dataset_file


def setup_experiment(tmp_path, *, structure=True, **changes):
    folder = tmp_path / "inputs with spaces"
    folder.mkdir()
    source, song, bank = fixture(folder)
    cfg = dict(mode="coordinated", planner_provider=dict(kind="chat_completions",
        base_url="http://127.0.0.1:8123/v1", model="declared-fixture-not-a-live-service",
        api_key_env="CHOREOBENCH_EXPERIMENT_TEST_KEY", temperature=.2, json_mode=True),
        max_calls=10, seed=17, max_output_tokens=512, outline_max_output_tokens=1024,
        event_time_format="boundary_ids", event_schedule_policy="balanced_full_unit",
        event_grid_seconds=.25, min_event_seconds=.5)
    public = None
    if structure:
        public, _ = sidecar(folder, source, song)
        cfg["public_structure_path"] = public.name  # Runtime resolves from the dataset, not config directory.
    cfg.update(changes)
    config = folder / "different config directory" / "coordinated.json"
    write_json(config, cfg)
    return source, bank, config, public


def forbid_providers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Experiment preparation must not instantiate a provider or call a model")
    for module in (baseline, coordination, harness, providers):
        monkeypatch.setattr(module, "create_provider", forbidden)


def test_four_configs_share_inputs_seed_model_timing_and_keep_run_paths_new(tmp_path, monkeypatch):
    source, bank, config, public = setup_experiment(tmp_path)
    before = {p: file_sha(p) for p in (source, bank, config, public)}
    forbid_providers(monkeypatch)
    monkeypatch.setenv("CHOREOBENCH_EXPERIMENT_TEST_KEY", "fixture-secret-must-not-appear")
    output = tmp_path / "new experiment with spaces"
    manifest = experiments.prepare_ablation_experiment(source, bank, config, output)
    assert manifest["prepared"] and manifest["ready_to_run"] and manifest["all_budgets_sufficient"]
    assert manifest["all_preparations_valid"] and manifest["known_preparation_failures"] == []
    assert "experiments.py" in manifest["code_sha256"] and "coordination.py" in manifest["code_sha256"]
    assert manifest["python"]["executable"] and manifest["python"]["version"]
    assert not manifest["executed"] and not manifest["results_verified"] and not manifest["model_or_service_verified"]
    assert not manifest["training_performed"] and manifest["input_synthetic"]
    assert manifest["actual_cost"] is manifest["actual_input_tokens"] is manifest["actual_output_tokens"] is None
    expected = [("main", True, True), ("no_outline", False, True),
                ("no_history", True, False), ("no_outline_no_history", False, False)]
    assert [(v["variant_id"], v["use_outline"], v["use_history"]) for v in manifest["variants"]] == expected
    invariant = []
    for row in manifest["variants"]:
        variant = read_json(row["config_path"])
        assert harness._config(variant) == variant
        assert variant["public_structure_path"] == str(public.resolve())
        assert variant["max_calls"] == 10 and variant["seed"] == 17
        assert row["config_sha256"] == file_sha(row["config_path"])
        assert not Path(row["run_dir"]).exists()
        invariant.append({k: v for k, v in variant.items() if k not in ("use_outline", "use_history")})
    assert all(value == invariant[0] for value in invariant)
    units = manifest["song_workload"][0]["planning_unit_count"]
    assert [v["expected_new_text_calls"] for v in manifest["variants"]] == [units + 1, units, units + 1, units]
    assert manifest["original_collection"]["recorded_calls"] == 6
    assert all(v["new_local_model_calls"] == 0 for v in manifest["variants"])
    commands = read_json(output / "commands.json")
    for command, row in zip(commands["runs"], manifest["variants"]):
        assert isinstance(command["argv"], list)
        assert command["argv"][-1] == row["run_dir"]
        assert str(source.resolve()) in command["argv"] and str(bank.resolve()) in command["argv"]
    assert "fixture-secret-must-not-appear" not in json.dumps(manifest) + json.dumps(commands)
    assert {p: file_sha(p) for p in before} == before
    assert experiments.verify_ablation_experiment(output) == manifest
    preflight = subprocess.run(commands["preflight_argv"], cwd=commands["cwd"], capture_output=True, text=True)
    assert preflight.returncode == 0, preflight.stderr
    assert '"executed": false' in preflight.stdout
    with pytest.raises(FileExistsError):
        experiments.prepare_ablation_experiment(source, bank, config, output)


def test_budget_shortfall_is_explicit_and_never_increased(tmp_path):
    source, bank, config, _ = setup_experiment(tmp_path, structure=False, max_calls=2)
    output = tmp_path / "insufficient"
    manifest = experiments.prepare_ablation_experiment(source, bank, config, output)
    assert manifest["prepared"] and not manifest["ready_to_run"] and not manifest["all_budgets_sufficient"]
    assert [v["expected_new_text_calls"] for v in manifest["variants"]] == [3, 2, 3, 2]
    assert [v["budget_sufficient"] for v in manifest["variants"]] == [False, True, False, True]
    assert all(read_json(v["config_path"])["max_calls"] == 2 for v in manifest["variants"])
    with pytest.raises(SystemExit) as error:
        experiments.main(["--verify", str(output)])
    assert error.value.code == 2
    assert experiments.verify_ablation_experiment(output) == manifest


@pytest.mark.parametrize("role", ["dataset", "candidates", "base_config", "public_structure", "audio:song1"])
def test_preflight_rejects_changed_input_bytes_even_when_json_meaning_is_unchanged(tmp_path, role):
    source, bank, config, _ = setup_experiment(tmp_path)
    output = tmp_path / "experiment"
    manifest = experiments.prepare_ablation_experiment(source, bank, config, output)
    path = Path(next(r["path"] for r in manifest["sources"] if r["role"] == role))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source bytes changed"):
        experiments.verify_ablation_experiment(output)


@pytest.mark.parametrize("artifact", ["config", "commands", "manifest"])
def test_preflight_rejects_changed_preparation_artifacts(tmp_path, artifact):
    source, bank, config, _ = setup_experiment(tmp_path)
    output = tmp_path / "experiment"
    manifest = experiments.prepare_ablation_experiment(source, bank, config, output)
    if artifact == "config":
        path = Path(manifest["variants"][0]["config_path"])
    else:
        path = output / (artifact + ".json")
    if artifact == "manifest":
        changed = read_json(path)
        changed["ready_to_run"] = False
        path.write_text(json.dumps(changed), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed|differs"):
        experiments.verify_ablation_experiment(output)


def test_no_outline_removes_theme_planning_not_only_a_call_and_history_cost_is_not_assumed_equal(tmp_path):
    source, bank, config, _ = setup_experiment(tmp_path, use_outline=False, use_history=False)
    manifest = experiments.prepare_ablation_experiment(source, bank, config, tmp_path / "experiment")
    assert manifest["variants"][0]["use_outline"] and manifest["variants"][0]["use_history"]
    confounds = " ".join(manifest["confounds"])
    assert "motif/theme planning" in confounds and "prompt length" in confounds and "not imply identical billed cost" in confounds
    assert manifest["block_seeds_identical_across_variants"]
    cfg = read_json(manifest["variants"][0]["config_path"])
    for binding in manifest["song_workload"][0]["block_call_seeds"]:
        assert binding["seed"] == baseline._request(binding["call_id"], cfg)["seed"]


def test_wrong_mode_or_no_selected_songs_does_not_publish_anything(tmp_path):
    source, bank, config, _ = setup_experiment(tmp_path)
    for value, match, directory in [({"mode": "index0"}, "coordinated", "mode"),
                                   ({"mode": "coordinated", "planner_provider": {"kind": "fixture"}, "splits": ["test"]},
                                    "No songs", "split")]:
        config.write_text(json.dumps(value), encoding="utf-8")
        output = tmp_path / directory
        with pytest.raises(ValueError, match=match):
            experiments.prepare_ablation_experiment(source, bank, config, output)
        assert not output.exists()


def test_dataset_byte_identity_must_still_match_original_candidate_collection(tmp_path):
    source, bank, config, _ = setup_experiment(tmp_path)
    source.write_bytes(source.read_bytes() + b" ")
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="identity differs"):
        experiments.prepare_ablation_experiment(source, bank, config, output)
    assert not output.exists()


def test_known_bad_local_prefix_remains_in_manifest_with_zero_new_calls(tmp_path, monkeypatch):
    source, _ = dataset_file(tmp_path)
    original_create = baseline.create_provider
    def malformed(spec, role, base_dir):
        provider = original_create(spec, role, base_dir)
        generate = provider.generate
        def response(request):
            result = generate(request)
            result["text"] = "original malformed local response"
            return result
        provider.generate = response
        return provider
    monkeypatch.setattr(baseline, "create_provider", malformed)
    baseline.run_baseline(source, tmp_path / "collection", {"continue_on_model_error": True})
    bank = tmp_path / "failed-bank.json"
    export_candidates(tmp_path / "collection", bank)
    config = tmp_path / "config.json"
    write_json(config, dict(mode="coordinated", planner_provider={"kind": "fixture"}, max_calls=0))
    forbid_providers(monkeypatch)
    manifest = experiments.prepare_ablation_experiment(source, bank, config, tmp_path / "experiment")
    assert manifest["selected_songs"] == ["song1"]
    assert len(manifest["song_workload"]) == 1
    assert "failed prefix" in manifest["song_workload"][0]["preparation_failure"]
    assert all(row["expected_new_text_calls"] == 0 for row in manifest["variants"])
    assert manifest["original_collection"]["recorded_calls"] == 1
    assert manifest["original_collection"]["expected_calls"] == 3
    assert manifest["all_budgets_sufficient"] and not manifest["all_preparations_valid"] and not manifest["ready_to_run"]
    assert manifest["known_preparation_failures"][0]["song_id"] == "song1"


def test_timing_infeasibility_keeps_song_and_does_not_predict_billed_calls(tmp_path):
    source, bank, config, _ = setup_experiment(tmp_path, structure=False, min_event_seconds=7)
    manifest = experiments.prepare_ablation_experiment(source, bank, config, tmp_path / "experiment")
    assert manifest["selected_songs"] == ["song1"]
    assert "shorter than min_event_seconds" in manifest["song_workload"][0]["preparation_failure"]
    assert all(row["expected_new_text_calls"] == 0 for row in manifest["variants"])
    assert manifest["actual_cost"] is None
    assert not manifest["all_preparations_valid"] and not manifest["ready_to_run"]


def test_preflight_rejects_code_closure_and_python_drift_without_model_calls(tmp_path, monkeypatch):
    source, bank, config, _ = setup_experiment(tmp_path)
    output = tmp_path / "experiment"
    manifest = experiments.prepare_ablation_experiment(source, bank, config, output)
    original = experiments._code_hashes
    changed = dict(manifest["code_sha256"], **{"hypothetical_new_module.py": "a" * 64})
    monkeypatch.setattr(experiments, "_code_hashes", lambda: changed)
    with pytest.raises(ValueError, match="implementation closure changed"):
        experiments.verify_ablation_experiment(output)
    monkeypatch.setattr(experiments, "_code_hashes", original)
    monkeypatch.setattr(experiments, "_python_identity", lambda: dict(manifest["python"], version="another-version"))
    with pytest.raises(ValueError, match="Python executable/version changed"):
        experiments.verify_ablation_experiment(output)
