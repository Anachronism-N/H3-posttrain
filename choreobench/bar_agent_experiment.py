"""Registered four-strategy experiment. Default is validation/dry-run, never inference.

Inputs are precomputed native variable-duration catalogs, acoustic summaries and
a global style prior. This runner does not start models or fabricate readiness.
"""
import argparse
import hashlib
import json
from pathlib import Path

ARMS = {"W": "workflow", "SF": "self_refine", "RA": "react", "O": "music_structured"}
READY = ("variable_audio_consumption_verified", "local_decoding_frozen",
         "model_identities_frozen", "source_split_verified", "budgets_dev_checked")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Fill every required deployment path before validation")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def prepare(inventory_path, provider_path, with_ablations=False):
    inventory_path = Path(inventory_path).resolve()
    inventory = read(inventory_path)
    if inventory.get("schema") != "choreobench.bar_agent_inventory.v1":
        raise ValueError("Unknown inventory schema; protocol.json is a plan, not an inventory")
    songs = inventory.get("songs")
    if not isinstance(songs, list) or not songs or len(songs) != inventory.get("expected_songs"):
        raise ValueError("Inventory must contain every pre-registered song")
    if inventory.get("panel") not in ("dev", "new_holdout", "historical_retest"):
        raise ValueError("Declare the panel, including historical retests")
    ids = [row.get("song_id") for row in songs]
    import re
    if any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Song identifiers must be unique and safe path components")
    base = inventory_path.parent
    source = _path(inventory.get("source"), base)
    style = _path(inventory.get("style_prior"), base)
    provider = Path(provider_path).resolve()
    source_contract = source.parent / "contract.json"
    catalog = read(source_contract)
    if catalog.get("schema") != "choreobench.untimed_catalog_contract.v2" or catalog.get("audio_contract") != "variable_duration_v1":
        raise ValueError("Main experiment requires a real bar-window v2 catalog")
    if not catalog.get("window_manifest_sha256"):
        raise ValueError("Catalog has no frozen bar-window identity")
    from .public_style_prior import load_public_style_prior
    prior = load_public_style_prior(style, catalog["dataset_sha256"], catalog["dataset_snapshot"]["songs"])
    if prior is None:
        raise ValueError("All arms require the same global style condition")
    hashes = {str(p): sha(p) for p in (inventory_path, source, source_contract, style, provider)}
    source_ids = {s["song_id"] for s in catalog["dataset_snapshot"]["songs"]}
    jobs = []
    for i, row in enumerate(songs):
        if row["song_id"] not in source_ids:
            raise ValueError("Registered song missing from shared catalog")
        summary = _path(row.get("music_summary"), base)
        hashes[str(summary)] = sha(summary)
        # Rotate method order by song. Registration never depends on outputs.
        order = list(ARMS)
        order = order[i % len(order):] + order[:i % len(order)]
        for arm in order:
            jobs.append(dict(song_id=row["song_id"], arm=arm, method=ARMS[arm],
                             source=str(source), music_summary=str(summary), style_prior=str(style),
                             rewrite_repeats=arm == "O" or (with_ablations and arm == "W")))
    limits = dict(seed=42, max_output_tokens=4096, outline_max_output_tokens=8192,
                  max_request_characters=300000, max_input_tokens=2000000,
                  max_total_output_tokens=150000, repeat_groups=5)
    overrides = inventory.get("limits", {})
    if set(overrides) - set(limits):
        raise ValueError("Unknown limit field")
    for key, value in overrides.items():
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError("Limits must be positive integers; seed may be zero")
    limits.update(overrides)
    return dict(schema="choreobench.bar_agent_matrix.v1", panel=inventory["panel"],
                songs=ids, expected_runs=len(jobs), jobs=jobs, limits=limits,
                provider=str(provider), input_hashes=hashes,
                code_hashes={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                readiness={k: inventory.get("readiness", {}).get(k) is True for k in READY},
                with_ablations=with_ablations,
                note="Equal caps, not equal actual cost; readiness declarations require server evidence.")


def logical_rows(job, run_dir, result, with_ablations):
    """A coupled W+R dispatch exposes W before revision without running W twice."""
    run_dir = Path(run_dir)
    complete = result.get("full_song_succeeded") is True
    costs = {k: result.get(k) for k in ("calls_used", "input_tokens", "output_tokens")}
    final = dict(arm=job["arm"], song_id=job["song_id"], complete=complete,
                 plan=str(run_dir / "plan.json") if complete else None,
                 physical_run=str(run_dir), costs=costs, status=result.get("stop_reason"))
    if not with_ablations or job["arm"] not in ("W", "O"):
        return [final]
    before = run_dir / "before_rewrite.json"
    before_costs = run_dir / "before_rewrite_costs.json"
    if not before.is_file() or not before_costs.is_file():
        missing = dict(final, arm="W-plusR" if job["arm"] == "W" else "O-noR",
                       complete=False, plan=None, costs=None, status="pre_rewrite_plan_unavailable")
        return [final, missing]
    prefix = read(before_costs)
    before_row = dict(final, arm="W" if job["arm"] == "W" else "O-noR",
                      complete=prefix.get("full_song_succeeded") is True, plan=str(before),
                      costs={k: prefix.get(k) for k in costs}, status="pre_rewrite_snapshot")
    if job["arm"] == "W":
        final["arm"] = "W-plusR"
    return [before_row, final]


def export_reviews(output, contract, rows):
    """Export the logical W prefix, not W+R's final plan, for blinded review."""
    import random
    rng = random.Random(contract["limits"]["seed"])
    logical = {(r["song_id"], r["arm"]): r for run in rows for r in run["logical_results"]}
    catalog = read(Path(contract["jobs"][0]["source"]).parent / "contract.json")
    originals = {s["song_id"]: s for s in catalog["dataset_snapshot"]["songs"]}
    pairs, bindings, cases, case_bindings, unavailable = [], [], [], [], []
    for song_id in contract["songs"]:
        events = {}
        arms = ["W", "SF", "RA", "O"] + (["O-noR", "W-plusR"] if contract["with_ablations"] else [])
        rng.shuffle(arms)
        for arm in arms:
            row = logical.get((song_id, arm))
            if not row or not row.get("complete") or not row.get("plan"):
                unavailable.append(dict(song_id=song_id, arm=arm, status="no_complete_plan"))
                continue
            plan = read(row["plan"])
            rate = plan["sample_rate"]
            events[arm] = [dict(event_id=e["event_id"], start=e["start"] / rate,
                                end=e["end"] / rate, track=e.get("track"), action=e["action"])
                           for e in plan["events"]]
            original = originals[song_id]
            audio = Path(original["audio"]["path"])
            if not audio.is_absolute():
                audio = Path(catalog["dataset_path"]).parent / audio
            duration = original["audio"]["num_samples"] / original["audio"]["sample_rate"]
            case_id = "c%05d" % len(cases)
            cases.append(dict(case_id=case_id, song_id=song_id, time_unit="seconds", events=events[arm],
                              audio_path=str(audio), audio_sha256=original["audio"]["sha256"],
                              audio_global_start_seconds=0, audio_global_end_seconds=duration,
                              song_duration_seconds=duration))
            case_bindings.append(dict(case_id=case_id, song_id=song_id, arm=arm, source_plan=row["plan"]))
        for baseline in ("W", "SF", "RA"):
            if baseline not in events or "O" not in events:
                continue
            a, b = (baseline, "O") if rng.randrange(2) else ("O", baseline)
            pair_id = "p%05d" % len(pairs)
            pairs.append(dict(pair_id=pair_id, song_id=song_id, comparison="anonymous_pair",
                              time_unit="seconds", A=events[a], B=events[b]))
            bindings.append(dict(pair_id=pair_id, song_id=song_id, A=a, B=b))
    root = Path(output) / "review"
    write(root / "pairs.public.json", dict(schema="choreobench.review_pairs.v2", pairs=pairs))
    write(root / "bindings.private.json", dict(pairs=bindings, cases=case_bindings))
    write(root / "whole3.cases.json", dict(cases=cases))
    write(root / "unavailable.json", dict(registered_songs=contract["songs"], records=unavailable))


def _run(inventory, provider, out_dir, *, execute=False, with_ablations=False, resume=False):
    contract = prepare(inventory, provider, with_ablations)
    if execute and not all(contract["readiness"].values()):
        raise ValueError("Formal dispatch blocked: complete the declared server preflight first")
    output = Path(out_dir).resolve()
    contract_path = output / "matrix_contract.json"
    if output.exists():
        if not resume or not contract_path.is_file() or read(contract_path) != contract:
            raise ValueError("Output exists; resume requires identical inputs, code and configuration")
    else:
        output.mkdir(parents=True)
        write(contract_path, contract)
    registry_path = output / "registry.json"
    rows = read(registry_path).get("runs", []) if registry_path.exists() else []
    existing = {(r["song_id"], r["arm"]): r for r in rows}
    if not execute:
        return dict(dry_run=True, expected_runs=contract["expected_runs"],
                    readiness=contract["readiness"], contract=str(contract_path))
    from .composition_workflows import run_composition
    for job in contract["jobs"]:
        key = (job["song_id"], job["arm"])
        if key in existing:
            continue  # Terminal failures are retained, never resampled on resume.
        if any(sha(path) != expected for path, expected in contract["input_hashes"].items()):
            raise ValueError("Frozen input changed during the experiment")
        run_dir = output / "runs" / job["arm"] / job["song_id"]
        record = dict(song_id=job["song_id"], arm=job["arm"], run_dir=str(run_dir))
        try:
            if run_dir.exists():
                raise ValueError("Interrupted run exists without terminal registry; inspect it, do not overwrite")
            result = run_composition(job["source"], job["song_id"], contract["provider"], run_dir,
                                     method=job["method"], controller_version="v2",
                                     music_summary_path=job["music_summary"], style_prior=job["style_prior"],
                                     acoustic_guidance=job["arm"] == "O",
                                     rewrite_repeats=job["rewrite_repeats"], **contract["limits"])
            record.update(status="terminal", result=result,
                          logical_results=logical_rows(job, run_dir, result, with_ablations))
        except Exception as error:
            record.update(status="dispatch_error", error_type=type(error).__name__, error=str(error),
                          logical_results=logical_rows(job, run_dir, {}, with_ablations))
        rows.append(record)
        existing[key] = record
        write(registry_path, dict(schema="choreobench.bar_agent_registry.v1", expected_runs=contract["expected_runs"],
                                 registered_songs=contract["songs"], runs=rows))
    export_reviews(output, contract, rows)
    return dict(dry_run=False, registered=contract["expected_runs"], recorded=len(rows), registry=str(registry_path))


def run(inventory, provider, out_dir, *, execute=False, with_ablations=False, resume=False):
    from .io import file_lock
    output = Path(out_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(output.parent / ("." + output.name + ".matrix.lock")):
        return _run(inventory, provider, output, execute=execute,
                    with_ablations=with_ablations, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--out-dir", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Dispatch model requests after server preflight")
    action.add_argument("--dry-run", action="store_true", help="Validate only (default)")
    parser.add_argument("--with-ablations", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.inventory, args.provider, args.out_dir, execute=args.execute,
                         with_ablations=args.with_ablations, resume=args.resume), ensure_ascii=False))


if __name__ == "__main__":
    main()
