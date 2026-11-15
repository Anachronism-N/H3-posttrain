#!/usr/bin/env python3
"""Run the structured-composition dev matrix: 4 methods x 3 dev songs.

Contract is fixed across methods: same frozen candidates, same public music
evidence, seed 42, same decoder limits and the same total budget (1+4U calls,
2M input / 150k output tokens per song). Historical runs are never overwritten:
an existing completed run is reported as reused, anything else is dispatched
into a fresh directory.

No model is started here; --provider must point at an already verified
text-only service. Credentials stay in the environment.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

METHODS = ("workflow", "react", "choreography", "structured")
CONTROLLER = {"workflow": "v1", "react": "v2", "choreography": "v2", "structured": "v2"}


def completed(run_dir):
    path = Path(run_dir) / "results.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-root", required=True, help="directory holding one song source per song id")
    parser.add_argument("--song-id", action="append", required=True, help="exactly three dev song ids")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.song_id) != 3:
        raise SystemExit("exactly three dev songs are required")
    methods = [m for m in args.methods.split(",") if m]
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit("unknown methods: " + ",".join(sorted(unknown)))
    root = Path(args.dev_root)
    out_root = Path(args.out_root)
    rows = []
    for song in args.song_id:
        source = root / "views" / (song + ".json")
        if not source.is_file():
            source = root / (song + ".json")
        for method in methods:
            run_dir = out_root / song / method
            previous = completed(run_dir)
            status = "reused" if previous and previous.get("complete") else "run"
            row = dict(song_id=song, method=method, run_dir=str(run_dir), action=status)
            if previous and previous.get("complete"):
                row.update(full_song_succeeded=previous.get("full_song_succeeded"),
                           calls_used=previous.get("calls_used"), input_tokens=previous.get("input_tokens"),
                           output_tokens=previous.get("output_tokens"))
                rows.append(row)
                print("reuse", song, method, row["full_song_succeeded"], flush=True)
                continue
            argv = [sys.executable, "-m", "choreobench", "run-composition",
                    "--source", str(source), "--song-id", song, "--provider", args.provider,
                    "--out-dir", str(run_dir), "--method", method,
                    "--controller-version", CONTROLLER[method], "--seed", "42",
                    "--max-request-characters", "300000",
                    "--max-input-tokens", "2000000", "--max-total-output-tokens", "150000"]
            row["argv"] = argv
            rows.append(row)
            if args.dry_run:
                print("dry-run", " ".join(argv), flush=True)
                continue
            if run_dir.exists():
                raise SystemExit("refusing to overwrite an incomplete non-resumable run: " + str(run_dir))
            print("run", song, method, flush=True)
            code = subprocess.call(argv, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            row["exit_code"] = code
            result = completed(run_dir) or {}
            row.update(full_song_succeeded=result.get("full_song_succeeded"),
                       calls_used=result.get("calls_used"), input_tokens=result.get("input_tokens"),
                       output_tokens=result.get("output_tokens"), stop_reason=result.get("stop_reason"))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "matrix.json").write_text(json.dumps(dict(
        methods=methods, songs=args.song_id, seed=42, controller=CONTROLLER,
        max_request_characters=300000, max_input_tokens=2000000, max_total_output_tokens=150000,
        calls_per_song="1+4U", runs=rows), ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(runs=len(rows), dry_run=args.dry_run), ensure_ascii=False))


if __name__ == "__main__":
    main()
