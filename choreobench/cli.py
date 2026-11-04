"""Command line entry points. Execute from the repository root."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from importlib import metadata

from .dataset import (assemble_dataset, import_structure, load_dataset, prepare_audio,
                      release_report, resolve_audio, verify_dataset_audio)
from .io import file_sha, object_sha, read_json, read_json_snapshot, write_json
from .schema import validate_dataset, validate_plan


def _song(dataset, song_id):
    matches = [song for song in dataset["songs"] if song["song_id"] == song_id]
    if len(matches) != 1:
        raise ValueError("Unknown song_id: " + song_id)
    return matches[0]


def _emit(value, path=None):
    if path:
        write_json(path, value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def doctor():
    packages = {}
    for name in ("numpy", "scipy", "soundfile", "torch", "transformers", "peft", "safetensors"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    nvidia_smi = shutil.which("nvidia-smi")
    gpu = None
    if nvidia_smi:
        try:
            process = subprocess.run([nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10, check=False)
            gpu = {"returncode": process.returncode, "inventory": process.stdout.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            gpu = {"error": type(error).__name__}
    return {"schema": "choreobench.environment.v1", "python": platform.python_version(), "platform": platform.platform(),
            "core_python_supported": sys.version_info >= (3, 9), "packages": packages, "gpu": gpu,
            "native_inference_verified": False,
            "note": "Inventory only; no model loaded, API contacted, weights downloaded or memory-fit guarantee."}


def resources(dataset_path, config_path):
    from .audio import make_windows
    from .baseline import _config
    data, cfg = load_dataset(dataset_path), _config(read_json(config_path))
    rows = []
    for song in data["songs"]:
        if song["split"] not in cfg["splits"]:
            continue
        count = len(make_windows(song["audio"]["num_samples"], song["audio"]["sample_rate"],
                                 cfg["window_seconds"], cfg["stride_seconds"]))
        outer = 0 if cfg["mode"] == "concat" else 1 if cfg["mode"] == "global" else 1 + len(song["music"]["sections"])
        rows.append({"song_id": song["song_id"], "seconds": song["audio"]["num_samples"] / song["audio"]["sample_rate"],
                     "observation_windows": count, "local_calls": count * cfg["candidates_per_window"], "outer_calls": outer})
    calls = sum(row["local_calls"] + row["outer_calls"] for row in rows)
    return {"schema": "choreobench.resource_estimate.v1", "songs": rows, "expected_generation_calls": calls,
            "max_calls": cfg["max_calls"], "budget_sufficient_for_declared_work": calls <= cfg["max_calls"],
            "local_output_token_limit_per_call": cfg["max_output_tokens"],
            "planner_output_token_limit_per_call": cfg.get("planner_max_output_tokens", cfg["max_output_tokens"]),
            "output_token_cap_if_each_call_reaches_limit": sum(
                row["local_calls"] * cfg["max_output_tokens"] +
                row["outer_calls"] * cfg.get("planner_max_output_tokens", cfg["max_output_tokens"]) for row in rows),
            "additional_costs_not_in_this_estimate": ["input tokens", "music analysis", "editing", "independent review", "model load"],
            "measured_gpu_memory": None, "measured_runtime": None, "monetary_cost": None,
            "note": "Compute/API rates and latency must come from the actual provider and a real pilot."}


def _review_pack(args):
    from .baseline import _verify
    from .reviews import build_review_pack, blank_review_template
    dataset, dataset_sha = read_json_snapshot(args.dataset)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, args.dataset)
    songs = [s for s in dataset["songs"] if s["split"] == args.split and (not args.song_id or s["song_id"] in args.song_id)]
    if not songs:
        raise ValueError("No requested review songs")
    if args.song_id and set(args.song_id) != {s["song_id"] for s in songs}:
        raise ValueError("Requested song outside selected split")
    items, labels = [], set()
    for value in args.run:
        if "=" not in value:
            raise ValueError("--run must be METHOD_ID=RUN_DIRECTORY")
        label, location = value.split("=", 1)
        if not label or label in labels:
            raise ValueError("Review method labels must be nonempty and unique")
        labels.add(label)
        root = Path(location).resolve()
        result, contract = _verify(read_json(root / "results.json")), read_json(root / "contract.json")
        if result.get("contract_sha256") != object_sha(contract) or contract["dataset_sha256"] != dataset_sha:
            raise ValueError("Review run/dataset contract differs")
        refs = {row["song_id"]: row for row in result["plans"]}
        for song in songs:
            if song["song_id"] not in refs:
                raise ValueError("Missing plan for a selected review song; report failure in submission evaluation, do not silently drop it")
            ref = refs[song["song_id"]]
            path = root / ref["path"]
            plan, plan_sha = read_json_snapshot(path)
            if plan_sha != ref["sha256"]:
                raise ValueError("Review plan changed")
            song = dict(song, audio=dict(song["audio"], path=str(resolve_audio(song, args.dataset))))
            items.append({"song": song, "plan": plan, "method_id": label})
    result = build_review_pack(items, seed=args.seed)
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "public" / "pack.json", result["public"])
    write_json(output / "private" / "bindings.json", result["private"])
    write_json(output / "public" / "reviews.template.json", blank_review_template(result["public"]))
    if args.copy_audio:
        for row in result["private"]["items"]:
            source = Path(row["audio_source_path"])
            destination = output / "public" / row["public_audio_path"]
            if file_sha(source) != row["audio_sha256"]:
                raise ValueError("Review audio changed before copying")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with source.open("rb") as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            if file_sha(destination) != row["audio_sha256"]:
                raise ValueError("Copied review audio identity differs")
    return {"public_pack": str(output / "public" / "pack.json"), "private_bindings": str(output / "private" / "bindings.json"),
            "review_template": str(output / "public" / "reviews.template.json"), "audio_copied": args.copy_audio,
            "notice": "Distribute ONLY public/ to raters. Private bindings contain method identities."}


def build_parser():
    parser = argparse.ArgumentParser(description="Music-to-text choreography benchmark and reference baselines")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate synthetic smoke fixtures, never scientific results")
    demo.add_argument("--out-dir", required=True, type=Path)
    demo.add_argument("--seconds", type=float, default=12.)
    demo.add_argument("--run", action="store_true", help="Run all three mock baselines and evaluate manual edit")
    sub.add_parser("doctor", help="Read-only CPU/package/GPU inventory; no model/API call")
    prep = sub.add_parser("prepare-audio", help="Explicitly decode/downmix/resample to canonical PCM16")
    for name in ("source", "out-wav", "out-song"):
        prep.add_argument("--" + name, required=True, type=Path)
    prep.add_argument("--song-id", required=True)
    prep.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    prep.add_argument("--sample-rate", type=int, default=16000)
    for name in ("song-group-id", "artist-group-id", "near-duplicate-group-id"):
        prep.add_argument("--" + name)
    prep.add_argument("--brief", default="")
    assemble = sub.add_parser("assemble-data", help="Combine prepared song JSONs without claiming grouping review")
    assemble.add_argument("--songs", nargs="+", required=True, type=Path)
    assemble.add_argument("--out", required=True, type=Path)
    assemble.add_argument("--dataset-id", required=True)
    assemble.add_argument("--synthetic", action="store_true")
    valid = sub.add_parser("validate-data")
    valid.add_argument("--dataset", required=True, type=Path)
    valid.add_argument("--release-check", action="store_true", help="Exit nonzero if declared release conditions fail")
    valid.add_argument("--track", choices=("local", "full_song"), default="full_song")
    structure = sub.add_parser("import-structure")
    structure.add_argument("--dataset", required=True, type=Path)
    structure.add_argument("--song-id", required=True)
    structure.add_argument("--structure", required=True, type=Path)
    structure.add_argument("--audio-sha256", required=True)
    structure.add_argument("--format", choices=("allin1", "samples"), default="allin1")
    structure.add_argument("--out", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--dataset", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--resume", action="store_true")
    estimate = sub.add_parser("resources")
    estimate.add_argument("--dataset", required=True, type=Path)
    estimate.add_argument("--config", required=True, type=Path)
    for name in ("validate-plan", "evaluate-plan"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--plan", required=True, type=Path)
        cmd.add_argument("--dataset", required=True, type=Path)
        cmd.add_argument("--out", type=Path)
    for name in ("apply-edit", "undo-edit"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--plan", required=True, type=Path, help="Original source plan")
        cmd.add_argument("--task", required=True, type=Path)
        cmd.add_argument("--patch", required=True, type=Path)
        cmd.add_argument("--out", required=True, type=Path)
        if name == "undo-edit":
            cmd.add_argument("--after", required=True, type=Path)
    propose = sub.add_parser("propose-edit")
    for name in ("plan", "task", "provider", "out-dir"):
        propose.add_argument("--" + name, required=True, type=Path)
    propose.add_argument("--seed", type=int, default=7)
    propose.add_argument("--max-output-tokens", type=int, default=4096)
    propose.add_argument("--resume", action="store_true")
    public = sub.add_parser("public-edit-task", help="Export instruction/scope only; strips all private goals")
    public.add_argument("--task", required=True, type=Path)
    public.add_argument("--out", required=True, type=Path)
    edit_eval = sub.add_parser("evaluate-edit")
    for name in ("before", "after", "task"):
        edit_eval.add_argument("--" + name, required=True, type=Path)
    edit_eval.add_argument("--out", type=Path)
    review = sub.add_parser("review-pack")
    review.add_argument("--dataset", required=True, type=Path)
    review.add_argument("--run", action="append", required=True, metavar="METHOD=RUN_DIR")
    review.add_argument("--split", choices=("train", "dev", "test"), default="test")
    review.add_argument("--song-id", action="append")
    review.add_argument("--out-dir", required=True, type=Path)
    review.add_argument("--seed", type=int, default=7)
    review.add_argument("--copy-audio", action="store_true")
    review_import = sub.add_parser("review-import")
    for name in ("public", "private", "reviews"):
        review_import.add_argument("--" + name, required=True, type=Path)
    review_import.add_argument("--out", required=True, type=Path)
    export = sub.add_parser("export-submission")
    export.add_argument("--run-dir", required=True, type=Path)
    export.add_argument("--method-id", required=True)
    export.add_argument("--out", required=True, type=Path)
    score = sub.add_parser("evaluate-submission")
    score.add_argument("--dataset", required=True, type=Path)
    score.add_argument("--submission", required=True, type=Path)
    score.add_argument("--split", choices=("train", "dev", "test"), default="test")
    score.add_argument("--track", choices=("local", "full_song"), default="full_song")
    score.add_argument("--out", type=Path)
    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--rows", required=True, type=Path)
    bootstrap.add_argument("--metric", default="value")
    bootstrap.add_argument("--seed", type=int, default=7)
    bootstrap.add_argument("--resamples", type=int, default=2000)
    bootstrap.add_argument("--out", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return dispatch(args)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


def dispatch(args):
    cmd = args.command
    if cmd == "doctor":
        _emit(doctor())
    elif cmd == "demo":
        from .fixtures import create_demo
        result = create_demo(args.out_dir, seconds=args.seconds)
        if args.run:
            from .baseline import run_baseline
            from .editing import apply_task_patch
            from .evaluation import evaluate_edit
            root = args.out_dir
            for mode in ("concat", "global", "hierarchical"):
                run_baseline(root / "dataset.json", root / ("run-" + mode), read_json(root / (mode + ".json")))
            before, task = read_json(root / "source_plan.json"), read_json(root / "edit_task.private.json")
            after = apply_task_patch(before, read_json(root / "example_patch.json"), task)
            write_json(root / "edited_plan.json", after)
            write_json(root / "edit_evaluation.json", evaluate_edit(before, after, task))
            result["mock_runs_complete"] = True
        _emit(result)
    elif cmd == "prepare-audio":
        _emit(prepare_audio(args.source, args.out_wav, args.out_song, song_id=args.song_id, split=args.split,
                           sample_rate=args.sample_rate, song_group_id=args.song_group_id,
                           artist_group_id=args.artist_group_id, near_duplicate_group_id=args.near_duplicate_group_id,
                           brief=args.brief))
    elif cmd == "assemble-data":
        _emit(assemble_dataset(args.songs, args.out, dataset_id=args.dataset_id, synthetic=args.synthetic))
    elif cmd == "validate-data":
        dataset = load_dataset(args.dataset)
        report = release_report(dataset, args.track)
        _emit({"valid": True, "songs": len(dataset["songs"]), "release_contract": report})
        if args.release_check and not report["eligible_by_declared_contract"]:
            return 2
    elif cmd == "import-structure":
        _emit(import_structure(args.dataset, args.song_id, args.structure, args.out, audio_sha256=args.audio_sha256, format=args.format))
    elif cmd == "run":
        from .baseline import run_baseline
        _emit(run_baseline(args.dataset, args.out_dir, read_json(args.config), resume=args.resume))
    elif cmd == "resources":
        _emit(resources(args.dataset, args.config))
    elif cmd in ("validate-plan", "evaluate-plan"):
        from .evaluation import evaluate_plan
        plan, dataset = read_json(args.plan), load_dataset(args.dataset)
        song = _song(dataset, plan["song_id"])
        validate_plan(plan, song)
        result = {"valid": True, "plan_sha256": object_sha(plan)} if cmd == "validate-plan" else evaluate_plan(plan, song)
        _emit(result, args.out)
    elif cmd in ("apply-edit", "undo-edit"):
        from .editing import apply_task_patch
        before = read_json(args.plan)
        after = apply_task_patch(before, read_json(args.patch), read_json(args.task))
        if cmd == "undo-edit":
            if object_sha(read_json(args.after)) != object_sha(after):
                raise ValueError("Undo target differs from this exact edit; refusing stale undo")
            _emit(before, args.out)
        else:
            _emit(after, args.out)
    elif cmd == "propose-edit":
        from .edit_runner import propose_edit
        _emit(propose_edit(args.plan, args.task, read_json(args.provider), args.out_dir,
                           seed=args.seed, max_output_tokens=args.max_output_tokens, resume=args.resume))
    elif cmd == "public-edit-task":
        from .editing import public_edit_task
        _emit(public_edit_task(read_json(args.task)), args.out)
    elif cmd == "evaluate-edit":
        from .evaluation import evaluate_edit
        _emit(evaluate_edit(read_json(args.before), read_json(args.after), read_json(args.task)), args.out)
    elif cmd == "review-pack":
        _emit(_review_pack(args))
    elif cmd == "review-import":
        from .reviews import import_reviews
        _emit(import_reviews(read_json(args.public), read_json(args.private), read_json(args.reviews)), args.out)
    elif cmd == "export-submission":
        from .submissions import export_submission
        _emit(export_submission(args.run_dir, args.out, args.method_id))
    elif cmd == "evaluate-submission":
        from .submissions import evaluate_submission
        _emit(evaluate_submission(args.dataset, args.submission, args.split, args.track), args.out)
    elif cmd == "bootstrap":
        from .evaluation import song_bootstrap
        _emit(song_bootstrap(read_json(args.rows), metric=args.metric, n_resamples=args.resamples, seed=args.seed), args.out)
    return 0
