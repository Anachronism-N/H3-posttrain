"""CPU preparation and bounded, inference-only music-caption probes."""
import argparse
from pathlib import Path

from .io import loads, object_sha, read_json, write_json
from .manifest import prepare_probe, TARGET_MODES
from .targets import convert_record, build_crop_manifest
from .scoring import summarize_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze validation cases and audio candidates; no model calls")
    prep.add_argument("--sources", type=Path, required=True)
    prep.add_argument("--case-ids", type=Path)
    prep.add_argument("--hard-pairs", type=Path)
    prep.add_argument("--target-mode", choices=TARGET_MODES, default="raw")
    prep.add_argument("--max-cases", type=int, default=256)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--out", type=Path, required=True)
    targets = commands.add_parser("targets", help="Keep raw source; export actions/JEV and optional crop plan")
    targets.add_argument("--sources", type=Path, required=True)
    targets.add_argument("--keep-timestamps", action="store_true")
    targets.add_argument("--out", type=Path, required=True)
    targets.add_argument("--crop-out", type=Path)
    summary = commands.add_parser("summarize", help="Validate and summarize fixed-target per-token scores")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--scores", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    summary.add_argument("--resamples", type=int, default=2000)
    summary.add_argument("--seed", type=int, default=42)
    run = commands.add_parser("run", help="Score through an audited original-runtime factory; never train")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--runtime-config", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("prepare", "targets"):
        dataset = read_json(args.sources)
        if dataset.get("schema") != "captionlab.sources.v1":
            raise ValueError("Expected captionlab.sources.v1")
        # Freeze resolved source paths so moving the manifest does not redirect audio.
        for record in dataset["records"]:
            path = Path(record["audio"]["path"]).expanduser()
            if not path.is_absolute():
                record["audio"]["path"] = str((args.sources.resolve().parent / path).resolve())
        if args.command == "prepare":
            value = prepare_probe(dataset["records"], genre_classes=dataset["genre_classes"],
                                  target_mode=args.target_mode, max_cases=args.max_cases, seed=args.seed,
                                  case_ids=read_json(args.case_ids) if args.case_ids else None,
                                  hard_pairs=read_json(args.hard_pairs) if args.hard_pairs else ())
            write_json(args.out, value)
        else:
            from .targets import validate_source_records
            isolation = validate_source_records(dataset["records"])
            rows = [convert_record(row, keep_timestamps=args.keep_timestamps,
                                   genre_classes=dataset["genre_classes"]) for row in dataset["records"]]
            # Validate all products before creating outputs.
            crops = build_crop_manifest(rows) if args.crop_out else None
            if args.out.exists() or (args.crop_out and args.crop_out.exists()):
                raise FileExistsError("Use new output paths")
            if args.crop_out and args.crop_out.resolve() == args.out.resolve():
                raise ValueError("Target and crop outputs must differ")
            write_json(args.out, {"schema": "captionlab.targets.v1", "records": rows, "isolation": isolation})
            if crops is not None:
                write_json(args.crop_out, crops)
    elif args.command == "summarize":
        manifest = read_json(args.manifest)
        records = [loads(line) for line in args.scores.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(r.get("manifest_sha256", object_sha(manifest)) != object_sha(manifest) for r in records):
            raise ValueError("Scores refer to a different manifest")
        write_json(args.out, summarize_scores(manifest, records, n_resamples=args.resamples, seed=args.seed))
    else:
        from .runner import run_probe
        run_probe(args.manifest, args.runtime_config, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
