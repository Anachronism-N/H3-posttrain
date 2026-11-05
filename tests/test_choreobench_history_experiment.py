"""Synthetic fixed-outline provenance and actual-run negative audits only."""
import copy
import json
from pathlib import Path

import pytest

from choreobench import coordination, experiments, harness, history_experiment as history
from choreobench.baseline import _sealed
from choreobench.io import atomic_write, file_sha, object_sha, read_json, write_json
from choreobench.outlines import export_shared_outlines
from test_choreobench_coordination import fake_planner
from test_choreobench_harness import fixture


def grid_reply(answer, request, payload):
    if request["response_kind"] == "coordination_unit":
        for event in answer["events"]:
            event.pop("start")
            event.pop("end")
            event.update(start_boundary=0, end_boundary=len(payload["event_clock"]["boundary_samples"]) - 1)


def setup(tmp_path, monkeypatch, **changes):
    folder = tmp_path / "inputs with spaces"
    folder.mkdir()
    source, _, bank = fixture(folder)
    cfg = dict(mode="coordinated", planner_provider=dict(kind="fixture"), max_calls=20, seed=17,
        max_output_tokens=512, event_time_format="boundary_ids", event_schedule_policy="balanced_full_unit",
        min_event_seconds=.5, event_grid_seconds=.25, structured_unit_output=True)
    source_requests = []
    fake_planner(monkeypatch, source_requests, grid_reply)
    source_result = harness.run_harness(source, bank, folder / "source-run", cfg)
    assert source_result["all_songs_succeeded"]
    outlines = folder / "shared-outlines.json"
    export_shared_outlines(folder / "source-run", outlines)
    cfg.update(changes)
    config_path = folder / "config.json"
    write_json(config_path, cfg)
    return source, bank, config_path, outlines


def prepare(tmp_path, monkeypatch, **changes):
    source, bank, cfg, outlines = setup(tmp_path, monkeypatch, **changes)
    out = tmp_path / "history experiment"
    manifest = history.prepare_history_experiment(source, bank, cfg, outlines, out)
    return out, manifest


def execute(manifest, monkeypatch, mutate=None):
    requests = []
    def customize(answer, request, payload):
        grid_reply(answer, request, payload)
        if mutate:
            mutate(answer, request, payload)
    fake_planner(monkeypatch, requests, customize)
    sources = {row["role"]: row["path"] for row in manifest["sources"]}
    for arm in manifest["variants"]:
        harness.run_harness(sources["dataset"], sources["candidates"], arm["run_dir"], read_json(arm["config_path"]))
    return requests


def test_prepare_verify_two_arms_no_provider_and_historical_costs_once(tmp_path, monkeypatch):
    source, bank, cfg, outlines = setup(tmp_path, monkeypatch)
    def no_provider(*args, **kwargs):
        pytest.fail("Preparation or verification contacted a provider")
    monkeypatch.setattr(coordination, "create_provider", no_provider)
    before = {path: file_sha(path) for path in (source, bank, cfg, outlines)}
    out = tmp_path / "new history experiment"
    result = history.prepare_history_experiment(source, bank, cfg, outlines, out)
    assert result["schema"] == "choreobench.history_experiment.v1"
    assert result["ready_to_run"] and result["prepared"]
    assert not result["executed"] and not result["results_verified"] and not result["model_or_service_verified"]
    assert result["shared_outline_source_costs"]["recorded_calls"] == 1
    assert result["original_collection"]["recorded_calls"] == 6
    assert result["input_synthetic"] and result["actual_cost"] is None
    configs = [read_json(row["config_path"]) for row in result["variants"]]
    assert configs[0]["use_history"] is True and configs[1]["use_history"] is False
    assert all(cfg["use_outline"] is True and cfg["shared_outline_path"] == str(outlines.resolve()) for cfg in configs)
    assert {k: v for k, v in configs[0].items() if k != "use_history"} == {
        k: v for k, v in configs[1].items() if k != "use_history"}
    assert all(row["expected_outline_calls"] == 0 and row["expected_new_text_calls"] == 2 for row in result["variants"])
    assert all(not Path(row["run_dir"]).exists() for row in result["variants"])
    assert {path: file_sha(path) for path in before} == before
    assert history.verify_history_experiment(out) == result
    assert history.audit_history_experiment(out)["missing_or_failed_pair_count"] == 1
    assert not history.audit_history_experiment(out)["paired_control_verified"]
    assert all(not Path(row["run_dir"]).exists() for row in result["variants"])
    with pytest.raises(FileExistsError):
        history.prepare_history_experiment(source, bank, cfg, outlines, out)


def test_actual_pair_audit_separates_event_count_time_track_and_text(tmp_path, monkeypatch):
    out, manifest = prepare(tmp_path, monkeypatch)
    def different(answer, request, payload):
        if request["response_kind"] == "coordination_unit" and not payload["history_available"]:
            answer["events"][0].update(track="upper", action="A different declared movement phrase.")
    requests = execute(manifest, monkeypatch, different)
    assert len(requests) == 4 and all(r["response_kind"] == "coordination_unit" for r in requests)
    report = history.audit_history_experiment(out)
    assert report["paired_control_verified"] and report["content_controlled_pair_count"] == 1
    assert report["selected_song_count"] == report["complete_pair_count"] == 1
    pair = report["pairs"][0]
    assert pair["actual_shared_outline_equal"] and pair["actual_common_inputs_menus_seeds_equal"]
    assert pair["event_count_equal"] and pair["time_boundaries_equal"]
    assert pair["track_sequence_equal"] is False and pair["action_sequence_equal"] is False
    assert pair["time_track_sequence_equal"] is False
    assert report["new_outline_calls"] == 0 and report["input_synthetic"]
    assert report["quality_status"] == "pending_independent_review" and not report["model_identity_independently_verified"]
    assert [report["arms"][name]["calls_used"] for name in ("main", "no_history")] == [2, 2]
    assert history.verify_history_experiment(out)["executed"] is False  # Preparation is immutable.


def test_known_model_failure_preserves_pair_denominator_and_costs(tmp_path, monkeypatch):
    out, manifest = prepare(tmp_path, monkeypatch)
    def bad(answer, request, payload):
        if not payload["history_available"]:
            answer["events"][0]["end_boundary"] = 0
    requests = execute(manifest, monkeypatch, bad)
    report = history.audit_history_experiment(out)
    assert not report["paired_control_verified"] and report["selected_song_count"] == 1
    assert report["complete_pair_count"] == 0 and report["missing_or_failed_pair_count"] == 1
    assert report["pairs"][0]["no_history_status"] == "failed"
    assert report["pairs"][0]["time_boundaries_equal"] is None
    assert report["arms"]["no_history"]["calls_used"] == 1 and len(requests) == 3
    assert report["arms"]["no_history"]["songs"]["song1"]["failure"]["retained_in_evaluation_denominator"]


def test_equal_event_count_can_have_different_boundaries_with_same_public_free_clock(tmp_path, monkeypatch):
    out, manifest = prepare(tmp_path, monkeypatch, event_schedule_policy="free")
    def shorten(answer, request, payload):
        if not payload["history_available"]:
            answer["events"][0]["end_boundary"] -= 1
    execute(manifest, monkeypatch, shorten)
    pair = history.audit_history_experiment(out)["pairs"][0]
    assert pair["content_control_verified"] and pair["event_count_equal"]
    assert pair["time_boundaries_equal"] is False
    assert pair["track_sequence_equal"] and pair["action_sequence_equal"]


def test_missing_results_and_unknown_raw_outcome_are_explicit_and_read_only(tmp_path, monkeypatch):
    out, manifest = prepare(tmp_path, monkeypatch)
    requests = []
    fake_planner(monkeypatch, requests, grid_reply, infrastructure_error=True)
    sources = {row["role"]: row["path"] for row in manifest["sources"]}
    arm = manifest["variants"][0]
    with pytest.raises(ValueError, match="Recorded provider failure"):
        harness.run_harness(sources["dataset"], sources["candidates"], arm["run_dir"], read_json(arm["config_path"]))
    root = Path(arm["run_dir"])
    report = history.audit_history_experiment(out)
    assert not report["paired_control_verified"]
    assert report["arms"]["main"]["unknown_outcomes_count"] == 1
    assert report["selected_song_count"] == report["missing_or_failed_pair_count"] == 1
    (root / "results.json").unlink()
    next((root / "calls").glob("*.raw.json")).unlink()
    before = {path: file_sha(path) for path in root.rglob("*.json")}
    report = history.audit_history_experiment(out)
    assert report["arms"]["main"]["status"] == "missing_results"
    assert report["arms"]["main"]["unknown_outcomes_count"] == 1
    assert {path: file_sha(path) for path in root.rglob("*.json")} == before


def test_timing_failure_stays_in_preparation_denominator(tmp_path, monkeypatch):
    out, manifest = prepare(tmp_path, monkeypatch, min_event_seconds=100)
    assert not manifest["ready_to_run"] and manifest["selected_songs"] == ["song1"]
    assert len(manifest["known_preparation_failures"]) == 1
    assert all(row["expected_new_text_calls"] == 0 for row in manifest["variants"])
    assert manifest["shared_outline_source_costs"]["recorded_calls"] == 1


def test_all_published_plans_do_not_hide_an_incomplete_run_marker(tmp_path, monkeypatch):
    out, manifest = prepare(tmp_path, monkeypatch)
    execute(manifest, monkeypatch)
    path = Path(manifest["variants"][0]["run_dir"]) / "results.json"
    result = read_json(path)
    result.update(complete=False, all_songs_succeeded=False, error=dict(type="OSError", message="Final receipt interrupted"))
    result.pop("record_sha256")
    atomic_write(path, _sealed(result))
    report = history.audit_history_experiment(out)
    assert report["complete_pair_count"] == 1
    assert not report["paired_control_verified"] and not report["run_completion_verified"]
    assert report["incomplete_run_variants"] == ["main"]


@pytest.mark.parametrize("role", ["dataset", "candidates", "base_config", "shared_outlines", "audio:song1"])
def test_verify_rejects_changed_source_bytes(tmp_path, monkeypatch, role):
    out, manifest = prepare(tmp_path, monkeypatch)
    path = Path(next(row["path"] for row in manifest["sources"] if row["role"] == role))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source bytes changed"):
        history.verify_history_experiment(out)


def test_call_budget_does_not_include_historical_outline_generation(tmp_path, monkeypatch):
    out, manifest = prepare(tmp_path, monkeypatch, max_calls=1)
    assert not manifest["ready_to_run"] and not manifest["all_budgets_sufficient"]
    assert all(row["expected_new_text_calls"] == 2 and row["expected_outline_calls"] == 0 for row in manifest["variants"])
    with pytest.raises(SystemExit) as error:
        history.main(["--verify", str(out)])
    assert error.value.code == 2


@pytest.mark.parametrize("artifact", ["manifest", "config", "commands"])
def test_preparation_artifact_changes_rejected(tmp_path, monkeypatch, artifact):
    out, manifest = prepare(tmp_path, monkeypatch)
    path = Path(manifest["variants"][0]["config_path"]) if artifact == "config" else out / (artifact + ".json")
    if artifact == "manifest":
        value = read_json(path)
        value["ready_to_run"] = False
        atomic_write(path, value)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        history.verify_history_experiment(out)


@pytest.mark.parametrize("artifact", ["plan", "raw", "outline", "contract", "producer_python"])
def test_audit_rejects_actual_evidence_tampering(tmp_path, monkeypatch, artifact):
    out, manifest = prepare(tmp_path, monkeypatch)
    execute(manifest, monkeypatch)
    root = Path(manifest["variants"][0]["run_dir"])
    result = read_json(root / "results.json")
    if artifact == "plan":
        path = root / result["plans"][0]["path"]
    elif artifact == "raw":
        path = next((root / "calls").glob("*.raw.json"))
    elif artifact == "outline":
        path = next((root / "outlines").glob("*.json"))
    else:
        path = root / "contract.json"
    if artifact in ("contract", "producer_python"):
        changed = read_json(path)
        if artifact == "producer_python":
            changed["software"]["python"] = "A different producer Python declaration"
        else:
            changed["config"]["use_history"] = False
        atomic_write(path, changed)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    # Raw JSON whitespace alone preserves its semantic seal; mutate its identity.
    if artifact == "raw":
        changed = read_json(path)
        changed["success"] = False
        atomic_write(path, changed)
    with pytest.raises(ValueError):
        history.audit_history_experiment(out)


def test_history_input_leak_cannot_be_hidden_by_resealing_journal(tmp_path, monkeypatch):
    out, manifest = prepare(tmp_path, monkeypatch)
    execute(manifest, monkeypatch)
    root = Path(manifest["variants"][1]["run_dir"])
    start_path = next((root / "calls").glob("*.started.json"))
    start = read_json(start_path)
    payload = json.loads(start["request"]["messages"][-1]["content"])
    payload["boundary_context"] = {"private_unrequested_history": "leak"}
    start["request"]["messages"][-1]["content"] = json.dumps(payload)
    start["request_sha256"] = object_sha(start["request"])
    start.pop("record_sha256")
    start = _sealed(start)
    atomic_write(start_path, start)
    raw_path = start_path.with_name(start_path.name.replace(".started.json", ".raw.json"))
    raw = read_json(raw_path)
    raw["started_sha256"] = start["record_sha256"]
    raw.pop("record_sha256")
    raw = _sealed(raw)
    atomic_write(raw_path, raw)
    result = read_json(root / "results.json")
    result["call_records"][start["call_id"]] = raw["record_sha256"]
    result.pop("record_sha256")
    atomic_write(root / "results.json", _sealed(result))
    with pytest.raises(ValueError, match="history fields"):
        history.audit_history_experiment(out)


def test_verify_environment_drift_and_audit_do_not_claim_current_reconstruction(tmp_path, monkeypatch):
    out, manifest = prepare(tmp_path, monkeypatch)
    execute(manifest, monkeypatch)
    monkeypatch.setattr(experiments, "_code_hashes", lambda: dict(manifest["code_sha256"], imaginary_module="a" * 64))
    with pytest.raises(ValueError, match="implementation changed"):
        history.verify_history_experiment(out)
    report = history.audit_history_experiment(out)
    assert not report["current_implementation_matches_preparation"]
    assert not report["paired_control_verified"] and report["complete_pair_count"] == 1
