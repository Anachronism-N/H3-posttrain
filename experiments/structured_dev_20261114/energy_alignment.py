#!/usr/bin/env python3
"""Objective audio-vs-plan alignment, computed from raw audio (no judge, no summary).

The acoustic summary that steered the music_structured arm is not used here, so a
gain cannot be explained by the same numbers that produced the plan.

Per one-second bin:
  onset  = mean librosa onset strength (audible activity)
  starts = number of events whose start falls in the bin (how often the plan moves)
  active = seconds covered by at least one event
Report Spearman correlations per song:
  starts~onset  : does new movement start where audible activity is high
  active~rms    : is movement present in louder passages
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np

HOP = 512
SR = 16000


def curve(audio, seconds):
    signal, _ = librosa.load(str(audio), sr=SR, mono=True)
    onset = librosa.onset.onset_strength(y=signal, sr=SR, hop_length=HOP)
    frames_per_second = SR / HOP
    bins = np.arange(0, seconds, 1.0)
    starts = np.array([b * frames_per_second for b in bins], dtype=int)
    ends = np.minimum(starts + int(frames_per_second), len(onset))
    onset_bin = np.array([onset[a:b].mean() if b > a else 0.0 for a, b in zip(starts, ends)])
    rms = librosa.feature.rms(y=signal, hop_length=HOP)[0]
    rms_bin = np.array([rms[a:b].mean() if b > a else 0.0 for a, b in zip(starts, ends)])
    return bins, onset_bin, 20 * np.log10(np.maximum(rms_bin, 1e-6))


def plan_curves(plan, bins, rate):
    starts = np.zeros(len(bins))
    active = np.zeros(len(bins))
    for event in plan["events"]:
        first = int(event["start"] / rate)
        last = int(np.ceil(event["end"] / rate))
        if 0 <= first < len(bins):
            starts[first] += 1
        for index in range(max(first, 0), min(last, len(bins))):
            active[index] += 1
    return starts, (active > 0).astype(float)


def spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--runs", required=True, help="directory with <arm>/<song>/plan.json")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for arm in args.arms:
        for plan_path in sorted((Path(args.runs) / arm).glob("*/plan.json")):
            song = plan_path.parent.name
            if not (Path(args.audio_root) / (song + ".wav")).is_file():
                print("skip", arm, song, "no audio", flush=True)
                continue
            plan = json.loads(plan_path.read_text())
            rate = plan.get("sample_rate", SR)
            duration = plan["events"][-1]["end"] / rate if plan["events"] else 0
            bins, onset_bin, rms_db = curve(Path(args.audio_root) / (song + ".wav"), duration)
            starts, active = plan_curves(plan, bins, rate)
            rows.append(dict(arm=arm, song=song, seconds=round(duration, 1), events=len(plan["events"]),
                             starts_onset=spearman(starts, onset_bin),
                             active_rms=spearman(active, rms_db),
                             start_rate=round(float(starts.mean()), 3),
                             active_share=round(float(active.mean()), 3)))
    payload = dict(schema="choreobench.energy_alignment.v1", hop_seconds=1.0, rows=rows,
                   note="Raw-audio onset strength and RMS vs the finished plan; no judge, no acoustic summary.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    for row in rows:
        print(row["arm"], row["song"], "starts~onset", None if row["starts_onset"] is None else round(row["starts_onset"], 3),
              "active~rms", None if row["active_rms"] is None else round(row["active_rms"], 3),
              "starts/s", row["start_rate"], flush=True)


if __name__ == "__main__":
    main()
