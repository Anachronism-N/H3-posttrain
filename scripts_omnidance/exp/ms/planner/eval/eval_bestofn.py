#!/usr/bin/env python3
"""Sampling + best-of-N evaluation against the reference caption.

Draws N samples per test row with temperature sampling, scores each candidate
with a reference-free structural/quality reward, keeps the best, and reports the
same metrics as eval_many so results are comparable to greedy runs.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import os
MS = Path(os.environ.get('PLANNER_MS', str(Path(__file__).resolve().parent / 'scripts_omnidance/exp/ms')))
sys.path.insert(0, str(MS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity import coverage, tags, ACTION, BODY  # noqa: E402
from eval_many import clean, STOPS  # noqa: E402

MARKER = re.compile(r'\[\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*s\s*\]')


def structure_score(text, reference=None):
    spans = [(float(a), float(b)) for a, b in MARKER.findall(text)]
    if not spans:
        return -100.0
    covered = 0.0
    for a, b in spans:
        covered += max(0.0, min(b, 5.0625) - max(a, 0.0))
    overrun = max(0.0, spans[-1][1] - 5.0625)
    gaps = sum(max(0.0, spans[i + 1][0] - spans[i][1]) for i in range(len(spans) - 1))
    overlaps = sum(max(0.0, spans[i][1] - spans[i + 1][0]) for i in range(len(spans) - 1))
    tokens = re.findall(r"[a-z']+", text.lower())
    grams = [tuple(tokens[i:i + 6]) for i in range(max(0, len(tokens) - 5))]
    repetition = 1 - len(set(grams)) / len(grams) if grams else 1.0
    actions = len(set(ACTION.findall(text)))
    score = (3.0 * min(covered / 5.0625, 1.0) + 0.6 * min(len(text), 2400) / 2400
             + 0.08 * min(actions, 14) - 4.0 * repetition
             - 6.0 * min(overrun, 3.0) - 4.0 * (gaps / 5.0625) - 4.0 * (overlaps / 5.0625))
    if reference:
        ref_spans = [(float(a), float(b)) for a, b in MARKER.findall(reference)]
        score -= 0.5 * abs(len(spans) - len(ref_spans))
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--rows', type=int, default=80)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--structure-only', action='store_true',
                        help='Score without reference window count (reference-free)')
    args = parser.parse_args()

    import hashlib
    import torch
    from music_planner_adapters import load_planner_adapter

    adapter = load_planner_adapter(family='omni_captioner', model_path=args.model,
                                   checkpoint=args.checkpoint, device_map=args.device_map, dtype=torch.bfloat16)
    rows = []
    with (args.data / (args.split + '.jsonl')).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: hashlib.sha256(('many:' + row['id']).encode()).hexdigest())
    rows = rows[:args.rows]
    args.out.mkdir(parents=True, exist_ok=False)
    metrics = []
    with (args.out / 'generations.jsonl').open('w', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            audio = Path(row['audio_path'])
            if not audio.is_absolute():
                audio = (args.data / audio).resolve()
            source = dict(id=row['id'], source_group_id=row['source_group_id'], audio_path=str(audio),
                          audio_sha256=row['audio_sha256'], audio_window_seconds=[0.0, 5.0625])
            batch = adapter.prefix(source)
            adapter.model.eval()
            parameter = next(adapter.model.parameters())
            candidates = []
            with torch.no_grad(), torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                                                 enabled=parameter.dtype == torch.bfloat16):
                output = adapter.model.generate(
                    **adapter._move(batch), do_sample=True, temperature=args.temperature, top_p=0.9,
                    num_return_sequences=args.samples, num_beams=1,
                    repetition_penalty=1.05, no_repeat_ngram_size=4,
                    max_new_tokens=1536, use_cache=True)
            sequences = output.sequences if hasattr(output, 'sequences') else output
            prefix_len = batch['input_ids'].shape[1]
            for seq in sequences:
                ids = seq[prefix_len:].cpu().tolist()
                text = clean(adapter.tokenizer.decode(ids, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False))
                candidates.append(text)
            reference = row['target_text']
            scored = [(structure_score(t, None if args.structure_only else reference), t) for t in candidates]
            scored.sort(key=lambda x: -x[0])
            best_score, best = scored[0]
            record = dict(id=row['id'], text=best, reference=reference, score=best_score,
                          coverage=coverage(best), reference_coverage=coverage(reference),
                          action_jaccard=len(tags(best, ACTION) & tags(reference, ACTION)) /
                                         max(len(tags(best, ACTION) | tags(reference, ACTION)), 1),
                          body_jaccard=len(tags(best, BODY) & tags(reference, BODY)) /
                                       max(len(tags(best, BODY) | tags(reference, BODY)), 1),
                          length_ratio=len(best) / max(len(reference), 1),
                          windows=len(MARKER.findall(best)),
                          reference_windows=len(MARKER.findall(reference)))
            metrics.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            if (index + 1) % 5 == 0:
                print('done %d/%d' % (index + 1, len(rows)), flush=True)
    n = len(metrics)
    summary = dict(rows=n, samples=args.samples, temperature=args.temperature,
                   mean_coverage=sum(r['coverage'] for r in metrics) / n,
                   mean_action_jaccard=sum(r['action_jaccard'] for r in metrics) / n,
                   mean_body_jaccard=sum(r['body_jaccard'] for r in metrics) / n,
                   mean_length_ratio=sum(r['length_ratio'] for r in metrics) / n,
                   mean_windows=sum(r['windows'] for r in metrics) / n,
                   mean_reference_windows=sum(r['reference_windows'] for r in metrics) / n)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
