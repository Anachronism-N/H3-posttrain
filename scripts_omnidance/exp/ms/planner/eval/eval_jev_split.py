#!/usr/bin/env python3
"""Measure JEV head top-1/top-3 accuracy on the balanced held-out validation split."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

MS = Path('/ssd/2/niezichao.nzc/artifacts/h3-jev-worktree-120fe6d/scripts_omnidance/exp/ms')
sys.path.insert(0, str(MS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--jev', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', default='val', choices=['val','test'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    import torch
    from music_planner_adapters import load_planner_adapter
    from music_planner_jev import load_jev, predict_jev

    adapter = load_planner_adapter(family='omni_captioner', model_path=args.model,
                                   checkpoint=args.checkpoint, device_map=args.device_map,
                                   dtype=torch.bfloat16)
    load_jev(adapter, args.jev)
    labels = {}
    for line in (args.data / (args.split + '-genres.jsonl')).open(encoding='utf-8'):
        row = json.loads(line)
        labels[row['id']] = row['genre']
    rows = []
    with (args.data / (args.split + '.jsonl')).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['id'] not in labels:
                continue
            audio = Path(row['audio_path'])
            if not audio.is_absolute():
                audio = (args.data / audio).resolve()
            rows.append(dict(row, audio_path=str(audio)))
    if args.limit:
        rows = rows[:args.limit]
    args.out.mkdir(parents=True, exist_ok=False)
    confusion = Counter()
    correct1 = correct3 = 0
    with (args.out / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            head = predict_jev(adapter, row)
            probs = head['probabilities'][0].tolist()
            ranked = sorted(zip(head['classes'], probs), key=lambda item: -item[1])
            expected = labels[row['id']]
            top1, top3 = ranked[0][0], [name for name, _ in ranked[:3]]
            correct1 += top1 == expected
            correct3 += expected in top3
            confusion[(expected, top1)] += 1
            stream.write(json.dumps(dict(id=row['id'], expected=expected, top1=top1,
                                         top1_prob=ranked[0][1], top3=top3), ensure_ascii=False) + '\n')
            stream.flush()
    summary = dict(rows=len(rows), top1=correct1, top1_accuracy=correct1 / len(rows),
                   top3=correct3, top3_accuracy=correct3 / len(rows),
                   per_class={cls: dict(total=sum(v for (e, _), v in confusion.items() if e == cls),
                                        correct=confusion[(cls, cls)])
                              for cls in sorted({e for e, _ in confusion})})
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
