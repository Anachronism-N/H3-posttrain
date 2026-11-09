"""Prepare a frozen-draft revision comparison without calling any model."""
import copy
from pathlib import Path
import sys

from .baseline import _sealed
from .io import file_sha, object_sha, read_json_snapshot, write_json
from .revision import _config, _inputs


def prepare_revision_experiment(dataset_path, plan_path, config_path, out_dir):
    """Publish configs and argv arrays; run directories remain uncreated.

    One text-only diagnosis is shared by targeted/listening, random/listening
    and targeted/text-only arms, all starting from the same original plan.
    Preparation checks source bytes/audio and declared call bounds, never
    provider availability, loaded weights, actual costs or revision quality.
    """
    output = Path(out_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(str(output))
    dataset_path, plan_path, config_path = [Path(p).expanduser().resolve()
                                          for p in (dataset_path, plan_path, config_path)]
    base, config_sha = read_json_snapshot(config_path)
    cfg = _config(base)
    if not cfg["listen"] or not isinstance(cfg["observer_provider"], dict):
        raise ValueError("Revision preparation requires listen=true and an explicit observer_provider")
    if cfg["planner_provider"].get("kind") not in ("chat_completions", "replay"):
        raise ValueError("Revision text planner must explicitly declare chat_completions or replay")
    if cfg["observer_provider"].get("kind") not in ("native", "native_service", "replay"):
        raise ValueError("Revision observer must explicitly declare native, native_service or replay")
    if cfg["max_calls"] < 3 * cfg["max_repairs"]:
        raise ValueError("max_calls must cover 3*max_repairs for the complete listening-arm budget")
    dataset, song, before, audio, source_hashes = _inputs(dataset_path, plan_path)
    source_hashes[str(config_path)] = config_sha
    sources = [{"role": role, "path": str(path), "sha256": source_hashes[str(path)]}
               for role, path in (("dataset", dataset_path), ("plan", plan_path), ("base_config", config_path))]
    sources.append({"role": "audio", "path": audio["path"], "sha256": audio["sha256"]})

    def unchanged():
        for path, digest in source_hashes.items():
            if file_sha(path) != digest:
                raise ValueError("Revision preparation source bytes changed: " + path)

    targeted = _config(dict(cfg, selection_policy="targeted"))
    configs = {
        "diagnosis": _config(dict(targeted, listen=False, observer_provider=None, max_calls=1)),
        "targeted": targeted,
        "random": _config(dict(targeted, selection_policy="random")),
        "text_only": _config(dict(targeted, listen=False, observer_provider=None,
                                  max_calls=2 * cfg["max_repairs"])),
    }
    order = ["diagnosis", "targeted", "random", "text_only"]
    paths = {name: str(output / "configs" / (name + ".json")) for name in order}
    run_dirs = {name: str(output / "diagnosis" if name == "diagnosis" else output / "runs" / name)
                for name in order}
    diagnosis_path = str(output / "diagnosis" / "diagnosis.json")
    commands = {"schema": "choreobench.revision_commands.v1",
                "cwd": str(Path(__file__).resolve().parents[1]), "execution_order": order, "steps": []}
    for name in order:
        argv = [sys.executable, "-m", "choreobench", "diagnose-plan" if name == "diagnosis" else "revise-plan",
                "--dataset", str(dataset_path), "--plan", str(plan_path), "--config", paths[name],
                "--out-dir", run_dirs[name]]
        if name != "diagnosis":
            argv.extend(["--diagnosis", diagnosis_path])
        commands["steps"].append({"stage": name, "argv": argv})
    commands["execution_guidance"] = (
        "Not executed. Run steps in order with subprocess.run(argv,cwd=cwd,shell=False,check=True). "
        "Reuse the one bound diagnosis; do not regenerate it per arm or reuse one arm's revised plan as another input."
    )
    unchanged()
    output.mkdir(parents=True, exist_ok=False)
    products = {}
    for name in order:
        write_json(paths[name], configs[name])
        products[name] = {"path": paths[name], "sha256": file_sha(paths[name])}
    commands_path = output / "commands.json"
    write_json(commands_path, commands)
    manifest = _sealed({
        "schema": "choreobench.revision_experiment.v1", "prepared": True, "executed": False,
        "deployment_verified": False, "model_or_service_verified": False, "quality_verified": False,
        "training_performed": False, "model_weights_updated": False,
        "readiness_scope": "CPU source/audio validation and declared call budgets only; no provider or deployment checks",
        "sources": sources, "song_id": song["song_id"], "source_plan_sha256": object_sha(before),
        "dataset_synthetic": dataset["synthetic"], "source_plan_synthetic_declaration": cfg["source_plan_synthetic"],
        "configs": products, "commands": {"path": str(commands_path), "sha256": file_sha(commands_path)},
        "execution_order": order, "run_directories": run_dirs, "shared_diagnosis_path": diagnosis_path,
        "original_baseline": {"kind": "unchanged_original_plan", "path": str(plan_path),
                              "file_sha256": source_hashes[str(plan_path)], "plan_sha256": object_sha(before),
                              "new_model_calls": 0},
        "call_upper_bounds": {"diagnosis_once": 1, "targeted": 3 * cfg["max_repairs"],
                              "random": 3 * cfg["max_repairs"], "text_only": 2 * cfg["max_repairs"]},
        "same_frozen_initial_plan": True, "same_shared_diagnosis": True,
        "same_selection_count_targeted_random": True,
        "same_listening_call_cap_targeted_random": True,
        "actual_calls": None, "actual_cost": None,
        "limitations": [
            "Shared diagnosis and internal judge verdicts are unverified proposals, not independent quality evidence.",
            "Targeted/random use the same min(max_repairs,issue_count) target count and call cap; actual usage/costs may differ.",
            "Text-only removes audio-observer calls and has a different call cap; it is not an equal-cost control.",
            "Original plan is the zero-new-call baseline; no new baseline generation or training is performed.",
            "Configuration and source hashes are not independent verification of provider deployment, weights or model execution.",
        ],
    })
    unchanged()
    for reference in list(products.values()) + [manifest["commands"]]:
        if file_sha(reference["path"]) != reference["sha256"]:
            raise ValueError("Revision preparation product changed before commit")
    write_json(output / "manifest.json", manifest)
    return copy.deepcopy(manifest)
