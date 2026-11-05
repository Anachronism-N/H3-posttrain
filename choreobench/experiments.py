"""Prepare fixed-input inference ablations. Never load models or execute runs."""
import argparse
import copy
import json
from pathlib import Path
import platform
import sys

from .audio import make_windows
from .baseline import _request
from .candidates import load_candidates, match_candidates
from .dataset import resolve_audio, verify_dataset_audio
from .harness import _config
from .io import file_sha, object_sha, read_json, read_json_snapshot, write_json
from .planning_units import build_planning_units
from .public_structure import load_public_structure
from .schema import validate_dataset
from .timing import TimingInfeasibleError, event_clock


VARIANTS = (
    ("main", True, True), ("no_outline", False, True),
    ("no_history", True, False), ("no_outline_no_history", False, False),
)


def _seal(value):
    value = copy.deepcopy(value)
    value["record_sha256"] = object_sha(value)
    return value


def _source(path, digest, role):
    return dict(role=role, path=str(Path(path).resolve()), sha256=digest)


def _assert_sources(sources):
    for source in sources:
        if file_sha(source["path"]) != source["sha256"]:
            raise ValueError("Ablation source bytes changed: " + source["role"])


def _shared_config(config):
    return {k: v for k, v in config.items() if k not in ("use_outline", "use_history")}


def _code_hashes():
    return {path.name: file_sha(path) for path in sorted(Path(__file__).resolve().parent.glob("*.py"))}


def _python_identity():
    return dict(executable=str(Path(sys.executable).resolve()), version=sys.version,
                implementation=platform.python_implementation())


def _workload(originals, effective, bank, config):
    source_cfg = bank.data["source"]["contract"]["config"]
    states = {row["song_id"]: row for row in bank.states}
    rows = []
    for original, song in zip(originals, effective):
        windows = make_windows(song["audio"]["num_samples"], song["audio"]["sample_rate"],
                               source_cfg["window_seconds"], source_cfg["stride_seconds"])
        units = build_planning_units(song, windows, config["unit_config"])
        failure = None
        try:
            for unit in units["units"]:
                event_clock(unit, song["audio"]["sample_rate"], config)
        except TimingInfeasibleError as error:
            failure = str(error)
        if states[original["song_id"]]["status"] != "complete":
            failure = "Recorded local model failure; original failed prefix remains in the song denominator"
        calls = [dict(call_id=song["song_id"] + "/coordinate/" + unit["unit_id"],
            seed=_request(song["song_id"] + "/coordinate/" + unit["unit_id"], config)["seed"])
            for unit in units["units"]]
        rows.append(dict(song_id=song["song_id"], planning_unit_count=len(units["units"]),
            planning_units_sha256=object_sha(units), preparation_failure=failure,
            expected_block_calls=0 if failure else len(units["units"]), block_call_seeds=calls))
    return rows


def prepare_ablation_experiment(dataset_path, candidates_path, base_config_path, output_dir):
    """Publish four new configs, command arrays and a sealed preparation manifest.

    Existing output paths are refused. Run directories are reserved in the
    mapping but deliberately NOT created: run-harness requires fresh paths.
    ready_to_run requires all selected songs locally preparable and sufficient
    declared max_calls, never proving service availability or model quality.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(str(output))
    dataset_path, candidates_path, base_config_path = [Path(p).resolve() for p in
                                                      (dataset_path, candidates_path, base_config_path)]
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    base, base_sha = read_json_snapshot(base_config_path)
    config = _config(base)
    if config["mode"] != "coordinated":
        raise ValueError("Ablation preparation requires coordinated mode")
    songs = [s for s in dataset["songs"] if s["split"] in config["splits"]]
    if not songs:
        raise ValueError("No songs in selected ablation splits")
    bank = load_candidates(candidates_path)
    bank_info = match_candidates(bank, dataset_sha, songs, {s["song_id"]: s["audio"] for s in songs},
                                 bank.data["source"]["contract"]["config"])
    structure_path = config["public_structure_path"]
    if structure_path is not None:
        structure_path = Path(structure_path).expanduser()
        structure_path = structure_path.resolve() if structure_path.is_absolute() else (dataset_path.parent / structure_path).resolve()
        config["public_structure_path"] = str(structure_path)
    effective, structure, structure_sha = load_public_structure(structure_path, dataset_sha, songs)
    sources = [_source(dataset_path, dataset_sha, "dataset"), _source(candidates_path, bank.bytes_sha, "candidates"),
               _source(base_config_path, base_sha, "base_config")]
    if structure_path is not None:
        sources.append(_source(structure_path, structure_sha, "public_structure"))
    # Runtime validates all dataset waveforms, including unselected splits.
    sources.extend(_source(resolve_audio(s, dataset_path), s["audio"]["sha256"], "audio:" + s["song_id"])
                   for s in dataset["songs"])
    workload = _workload(songs, effective, bank, config)
    common_sha = object_sha(_shared_config(config))
    blocks = sum(s["expected_block_calls"] for s in workload)
    eligible_songs = sum(s["preparation_failure"] is None for s in workload)
    variants, configs = [], {}
    for name, outline, history in VARIANTS:
        variant = _config(dict(config, use_outline=outline, use_history=history))
        if object_sha(_shared_config(variant)) != common_sha:
            raise ValueError("Ablation normalization changed a non-ablated parameter")
        configs[name] = variant
        outlines = eligible_songs if outline else 0
        calls = blocks + outlines
        variants.append(dict(variant_id=name, use_outline=outline, use_history=history,
            config_path=str(output / "configs" / (name + ".json")), run_dir=str(output / "runs" / name),
            expected_new_text_calls=calls, expected_block_calls=blocks, expected_outline_calls=outlines,
            max_calls=variant["max_calls"], budget_sufficient=calls <= variant["max_calls"],
            new_local_model_calls=0, output_token_cap=blocks * variant["max_output_tokens"] +
                outlines * variant["outline_max_output_tokens"]))
    _assert_sources(sources)
    bank.assert_unchanged(full=True)
    output.mkdir(parents=True, exist_ok=False)
    for row in variants:
        path = Path(row["config_path"])
        write_json(path, configs[row["variant_id"]])
        row["config_sha256"] = file_sha(path)
    sufficient = all(row["budget_sufficient"] for row in variants)
    failures = [dict(song_id=row["song_id"], reason=row["preparation_failure"])
                for row in workload if row["preparation_failure"] is not None]
    preparations_valid = not failures
    code_sha, python_identity = _code_hashes(), _python_identity()
    cwd = str(Path(__file__).resolve().parents[1])
    commands = dict(schema="choreobench.ablation_commands.v1", cwd=cwd,
        preflight_argv=[sys.executable, "-m", "choreobench.experiments", "--verify", str(output)],
        runs=[dict(variant_id=row["variant_id"], ready_to_run=row["budget_sufficient"] and preparations_valid, argv=[sys.executable,
            "-m", "choreobench", "run-harness", "--dataset", str(dataset_path), "--candidates", str(candidates_path),
            "--config", row["config_path"], "--out-dir", row["run_dir"]]) for row in variants],
        execution_guidance="Not executed. Use subprocess.run(argv, cwd=cwd, check=True, shell=False); "
            "run preflight before each variant. Supply credentials through the provider's named environment variable. "
            "After execution compare each actual run contract/results to this preparation; input hashes are not a model attestation.")
    commands_path = output / "commands.json"
    write_json(commands_path, commands)
    manifest = _seal(dict(schema="choreobench.ablation_experiment.v1", prepared=True, executed=False,
        results_verified=False, model_or_service_verified=False, training_performed=False,
        all_budgets_sufficient=sufficient, all_preparations_valid=preparations_valid,
        known_preparation_failures=failures, ready_to_run=sufficient and preparations_valid,
        readiness_scope="All selected songs locally preparable; source binding and declared max_calls valid; no model/service/token-context acceptance",
        code_sha256=code_sha, python=python_identity,
        sources=sources, selected_songs=[s["song_id"] for s in songs], song_workload=workload,
        shared_config_sha256=common_sha, planner_config_sha256=object_sha(config["planner_provider"]),
        planner_identity_evidence="Configuration declaration only; no service contacted or weights loaded",
        input_synthetic=bool(dataset["synthetic"] or bank.data["synthetic"] or
            (structure and any(s["synthetic"] for s in structure["songs"]))),
        original_collection=dict(recorded_calls=bank_info["original_collection_calls"],
            expected_calls=bank_info["expected_local_calls"], costs_sha256=object_sha(bank_info["original_collection_costs"]),
            scope="One shared local collection; never multiplied by four or claimed free"),
        block_seeds_identical_across_variants=True, call_seed_policy="Existing global seed plus stable per-block call_id",
        actual_input_tokens=None, actual_output_tokens=None, actual_cost=None, variants=variants,
        commands=dict(path=str(commands_path), sha256=file_sha(commands_path)),
        confounds=[
            "no_outline removes the outline call AND motif/theme planning, replacing section intent with an explicit neutral design; not merely one free call removed",
            "no_history removes prior committed text, declared exits, motif counts and prefix hashes from model input; this changes prompt length and token cost",
            "Identical max_calls/seed/model/time policy does not imply identical billed cost or input context",
            "Shared local predictions are fallible; preparation, full coverage and valid schemas do not establish musical or physical quality",
        ]))
    _assert_sources(sources)
    if code_sha != _code_hashes():
        raise ValueError("Ablation implementation changed during preparation")
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_ablation_experiment(output_dir):
    """Recheck preparation hashes without running a provider or changing files."""
    output = Path(output_dir).resolve()
    manifest = read_json(output / "manifest.json")
    if (not isinstance(manifest, dict) or manifest.get("schema") != "choreobench.ablation_experiment.v1"
            or manifest.get("record_sha256") != object_sha({k: v for k, v in manifest.items() if k != "record_sha256"})):
        raise ValueError("Ablation preparation manifest hash/schema differs")
    _assert_sources(manifest["sources"])
    if manifest.get("code_sha256") != _code_hashes():
        raise ValueError("Ablation implementation closure changed; prepare a new experiment")
    if manifest.get("python") != _python_identity():
        raise ValueError("Ablation Python executable/version changed")
    expected_names = [v[0] for v in VARIANTS]
    if [v["variant_id"] for v in manifest["variants"]] != expected_names:
        raise ValueError("Ablation variant inventory differs")
    for row, (_, outline, history) in zip(manifest["variants"], VARIANTS):
        path = output / "configs" / (row["variant_id"] + ".json")
        if (str(path) != row["config_path"] or str(output / "runs" / row["variant_id"]) != row["run_dir"]
                or file_sha(path) != row["config_sha256"]):
            raise ValueError("Prepared ablation configuration/path changed")
        config = _config(read_json(path))
        if (config["mode"] != "coordinated" or config["use_outline"] is not outline or config["use_history"] is not history
                or object_sha(_shared_config(config)) != manifest["shared_config_sha256"]):
            raise ValueError("A non-ablated parameter or ablation flag changed")
    commands_path = output / "commands.json"
    if str(commands_path) != manifest["commands"]["path"] or file_sha(commands_path) != manifest["commands"]["sha256"]:
        raise ValueError("Prepared ablation commands changed")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify prepared ablation input/configuration hashes; no inference")
    parser.add_argument("--verify", required=True, metavar="EXPERIMENT_DIR")
    args = parser.parse_args(argv)
    report = verify_ablation_experiment(args.verify)
    print(json.dumps(dict(prepared=report["prepared"], ready_to_run=report["ready_to_run"], executed=False)))
    if not report["ready_to_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
