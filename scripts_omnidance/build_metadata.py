#!/usr/bin/env python
"""Build OmniDance training/val metadata CSVs for Wan2.2-TI2V-5B-Audio fullSFT.

Sources:
  - mainline_input.csv        (video, input_audio, duration_seconds, has_audio)
  - temporal_analysis.csv     (preprocess_action: keep/trim/review_stop/skip)
  - OmniDance/text/<id>.txt   (5-paragraph choreography-aware annotation)

Filters: keep + has_audio + duration>=5.1s (all portrait, 16fps already).
Output columns: video, input_audio, prompt  (absolute paths; prompt = full 5-paragraph text).
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

MAINLINE = Path("/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed/pose_preprocess_outputs/mainline")
TEXT_DIR = Path("/apdcephfs_gy4/share_302533218/cedricnie/OmniDance/text")


def load_prompt(video_path: str) -> str:
    stem = Path(video_path).stem
    txt = TEXT_DIR / f"{stem}.txt"
    if not txt.is_file():
        return ""
    return txt.read_text(encoding="utf-8", errors="ignore").strip().replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--val-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="optional cap on train rows (smoke tests)")
    args = ap.parse_args()

    mi = pd.read_csv(MAINLINE / "mainline_input.csv")
    ta = pd.read_csv(MAINLINE / "temporal_analysis.csv", usecols=["source_row_index", "preprocess_action"])
    df = mi.merge(ta, on="source_row_index", how="left")
    df = df[
        (df.preprocess_action == "keep")
        & (df.has_audio == 1)
        & (df.duration_seconds >= 5.1)
    ].reset_index(drop=True)
    print(f"[build] pool after keep+audio+duration filter: {len(df)}", file=sys.stderr)

    # deterministic shuffle so val split is stable across rebuilds
    df = df.sample(frac=1.0, random_state=20260822).reset_index(drop=True)

    val = df.head(args.val_size).copy()
    train = df.iloc[args.val_size:].copy()
    if args.limit:
        train = train.head(args.limit)

    for name, part in (("train", train), ("val", val)):
        prompts = [load_prompt(p) for p in part.video]
        out = pd.DataFrame({
            "video": part.video,
            "input_audio": part.input_audio,
            "prompt": prompts,
        })
        n_missing = int((out.prompt == "").sum())
        out = out[out.prompt != ""].reset_index(drop=True)
        print(f"[build] {name}: {len(out)} rows (dropped {n_missing} missing text)", file=sys.stderr)
        out.to_csv(args.out_train if name == "train" else args.out_val, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
