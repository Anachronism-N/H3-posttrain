#!/usr/bin/env python3
"""Teacher-forced token NLL on the official full-v4 test split."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

MS = Path('/ssd/2/niezichao.nzc/artifacts/h3-jev-worktree-120fe6d/scripts_omnidance/exp/ms')
sys.path.insert(0, str(MS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--rows', type=int, default=500)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-map')
    parser.add_argument('--device')
    args = parser.parse_args()

    import torch
    from music_planner_adapters import load_planner_adapter

    adapter = load_planner_adapter(family='omni_captioner', model_path=args.model,
                                   checkpoint=args.checkpoint, device_map=args.device_map,
                                   device=args.device, dtype=torch.bfloat16)
    rows = []
    with (args.data / (args.split + '.jsonl')).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = Path(row['audio_path'])
            if not audio.is_absolute():
                audio = (args.data / audio).resolve()
            rows.append(dict(row, audio_path=str(audio)))
    rows.sort(key=lambda row: hashlib.sha256(('nll:' + row['id']).encode()).hexdigest())
    rows = rows[:args.rows]
    args.out.mkdir(parents=True, exist_ok=False)
    total = count = 0.0
    records = []
    with (args.out / 'rows.jsonl').open('w', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            result = adapter.teacher_forced(row)
            total += result['sum']
            count += result['token_count']
            records.append(dict(id=row['id'], nll=result['mean_nll'], tokens=result['token_count']))
            stream.write(json.dumps(records[-1], ensure_ascii=False) + '\n')
            stream.flush()
            if (index + 1) % 50 == 0:
                print('done %d/%d mean_nll=%.4f' % (index + 1, len(rows), total / count), flush=True)
    summary = dict(rows=len(rows), tokens=int(count), mean_nll=total / count, split=args.split,
                   checkpoint=str(args.checkpoint))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
