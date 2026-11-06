#!/usr/bin/env python3
"""Aggregate planner benchmark10 generations into a comparison table."""
import difflib
import json
import sys
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in Path(path).open(encoding='utf-8')]
    return rows


def pairwise_similarity(texts):
    if len(texts) < 2:
        return 0.0
    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            scores.append(difflib.SequenceMatcher(None, texts[i][:600], texts[j][:600]).ratio())
    return sum(scores) / len(scores)


def summarize(run):
    rows = load(run['generations'])
    texts = [row.get('raw_text', '') for row in rows]
    out = dict(run=run['name'], cases=len(rows),
               failures=sum(1 for row in rows if row.get('failure')),
               mean_windows=sum(row.get('metrics', {}).get('window_count', 0) for row in rows) / len(rows),
               mean_chars=sum(row.get('metrics', {}).get('char_length', 0) for row in rows) / len(rows),
               mean_repetition=sum(row.get('metrics', {}).get('ngram8_repetition', 0) for row in rows) / len(rows),
               mean_unique_sentence=sum(row.get('metrics', {}).get('unique_sentence_ratio', 0) for row in rows) / len(rows),
               styled=sum(1 for row in rows if row.get('metrics', {}).get('style_header')),
               eos_stops=sum(1 for row in rows if row.get('stop_reason') == 'eos'),
               pairwise_similarity=round(pairwise_similarity(texts), 4))
    jev = [row for row in rows if row.get('jev')]
    if jev:
        out['jev_top1_match'] = sum(1 for row in jev if row['jev']['top1'] == row['genre'])
        out['jev_scored'] = len(jev)
        out['jev_top1'] = {row['id']: dict(expected=row['genre'], predicted=row['jev']['top1'],
                                           probability=round(row['jev']['top1_prob'], 4)) for row in jev}
    return out


def main():
    specs = json.loads(Path(sys.argv[1]).read_text())
    summary = [summarize(spec) for spec in specs]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = Path(sys.argv[2])
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
