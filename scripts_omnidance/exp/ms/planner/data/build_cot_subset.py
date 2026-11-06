#!/usr/bin/env python3
"""Build a CoT-lite subset: deterministic music analysis prefix + dance caption."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np


def analysis(path):
    signal, rate = librosa.load(path, sr=16000, mono=True, duration=5.0625)
    envelope = librosa.onset.onset_strength(y=signal, sr=rate)
    tempo, beats = librosa.beat.beat_track(onset_envelope=envelope, sr=rate)
    times = librosa.times_like(envelope, sr=rate)
    third = 5.0625 / 3
    energy = [float(envelope[(times >= i * third) & (times < (i + 1) * third)].mean()) for i in range(3)]
    peak = max(energy) or 1.0
    energy = [round(value / peak, 2) for value in energy]
    accents = int(sum(1 for time in librosa.frames_to_time(beats, sr=rate) if time < 5.0625))
    return 'Music analysis: %d BPM; onset energy %s over three equal parts; %d accents.\n' % (
        round(float(np.atleast_1d(tempo)[0])), '/'.join('%.2f' % value for value in energy), accents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='full-v4 directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train', type=int, default=5000)
    parser.add_argument('--val', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('exists')
    args.out.mkdir(parents=True)
    (args.out / 'audio').mkdir()

    def select(split, quota):
        rows = [json.loads(line) for line in (args.data / (split + '.jsonl')).read_text().splitlines() if line.strip()]
        rows.sort(key=lambda row: hashlib.sha256((str(args.seed) + ':' + row['id']).encode()).hexdigest())
        return rows[:quota]

    for split, quota in (('train', args.train), ('val', args.val)):
        output = []
        for row in select(split, quota):
            source = args.data / row['audio_path']
            text = analysis(source) + row['target_text']
            output.append(dict(row, split=split, target_text=text, caption_version='m2d-cot-lite-v1'))
            os.link(source, args.out / 'audio' / (row['id'] + '.wav'))
        (args.out / (split + '.jsonl')).write_text(
            ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in output), encoding='utf-8')
        print(split, len(output))
    contract = dict(schema='music-planner-caption-contract-v1', caption_version='m2d-cot-lite-v1',
                    task_instruction='Listen to the music, analyse it, then describe the dance.',
                    audio_min_seconds=1.0, audio_max_seconds=10.0,
                    required_markers=[], require_target_windows=False)
    (args.out / 'caption_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
