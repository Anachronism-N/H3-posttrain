#!/usr/bin/env python3
"""Preflight and collect new audio-only SFT v3 candidates without altering old journals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from choreobench.audio import make_windows
from choreobench.io import read_json_snapshot, write_json
from choreobench.schema import validate_dataset
from choreobench.untimed_catalog import run_catalog
from checkpoint_provider import create, DECODING, MAX_NEW_TOKENS


def validate_clock(dataset):
    cases = []
    for song in dataset["songs"]:
        audio = song["audio"]
        if audio["sample_rate"] != 16000 or audio["num_samples"] < 81000:
            raise ValueError("Preflight requires 16000 Hz audio with at least one complete SFT v3 window")
        for window in make_windows(audio["num_samples"], audio["sample_rate"]):
            for candidate in range(2):
                case_id = song["song_id"] + "/untimed/%04d/%02d" % (window["index"], candidate)
                cases.append(dict(case_id=case_id, seed=int(hashlib.sha256(
                    ("sol61-select:" + case_id).encode()).hexdigest()[:8], 16)))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--provider", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset, dataset_sha = read_json_snapshot(args.dataset.resolve())
    validate_dataset(dataset)
    provider, provider_sha = read_json_snapshot(args.provider.resolve())
    if (provider.get("kind") != "native" or provider.get("family") != "omni_captioner"
            or provider.get("portable_checkpoint") is not True or provider.get("task_prompt")):
        raise ValueError("Require explicit audio-only portable SFT v3 provider configuration")
    cases = validate_clock(dataset)
    manifest = dict(schema="choreobench.sft_v3_collection_preflight.v2", dataset_sha256=dataset_sha,
                    provider_config_sha256=provider_sha, provider_factory="checkpoint_provider.create",
                    candidates_per_window=2, decoding=dict(DECODING, max_new_tokens=MAX_NEW_TOKENS),
                    canonical_case_ids=cases, expected_calls=len(cases),
                    inferred_audio_boundary=False, model_calls_sent=0)
    if args.manifest_out is not None:
        write_json(args.manifest_out, manifest)
    if args.prepare_only:
        print(json.dumps(dict(manifest, canonical_case_ids=len(cases)), ensure_ascii=False), flush=True)
        return
    result = run_catalog(args.dataset, args.provider, args.out_dir, seed=42,
                         candidates_per_window=2, max_output_tokens=MAX_NEW_TOKENS,
                         max_calls=len(cases), resume=args.resume, provider_factory=create)
    print(json.dumps({k: result[k] for k in ("complete", "calls_used", "expected_calls",
        "complete_proposals", "failed_proposals", "unknown_usage_count", "unknown_outcomes_count")},
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
