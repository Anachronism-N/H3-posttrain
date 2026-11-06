#!/usr/bin/env python3
"""Offline ROUGE-L / lexical overlap scoring for eval_many generations."""
import json
import re
import sys
from pathlib import Path


def toks(text):
    return re.findall(r"[a-z]+(?:['’][a-z]+)?|\d+(?:\.\d+)?s?", text.lower())


def rouge_l(generated, reference):
    a, b = toks(generated), toks(reference)
    prev = [0] * (len(b) + 1)
    for token in a:
        cur = [0]
        for index, other in enumerate(b):
            cur.append(prev[index] + 1 if token == other else max(prev[index + 1], cur[index]))
        prev = cur
    lcs = prev[-1]
    precision = lcs / max(len(a), 1)
    recall = lcs / max(len(b), 1)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    rows = [json.loads(line) for line in Path(sys.argv[1]).open(encoding='utf-8')]
    scores = [rouge_l(row['text'], row['reference']) for row in rows]
    summary = dict(rows=len(rows), mean_rouge_l=sum(scores) / len(scores),
                   min_rouge_l=min(scores), max_rouge_l=max(scores),
                   mean_length_ratio=sum(row['length_ratio'] for row in rows) / len(rows))
    print(json.dumps(summary, indent=2))
    Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
