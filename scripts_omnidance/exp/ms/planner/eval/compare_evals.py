#!/usr/bin/env python3
"""Compare two eval_many generation files on length/density/ROUGE."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_many import rouge_l  # noqa: E402

MARKER = re.compile(r'\[\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*s?\]')


def stats(path):
    rows = [json.loads(line) for line in Path(path).open(encoding='utf-8')]
    windows = []; clauses = []
    for row in rows:
        parts = [part for part in MARKER.split(row['text']) if part.strip()]
        sentences = [s for part in parts for s in re.split(r'[;。]', part) if s.strip()]
        windows.append(len(parts))
        clauses.append(len(sentences) / max(len(parts), 1))
    return dict(rows=len(rows),
                mean_chars=sum(len(row['text']) for row in rows) / len(rows),
                mean_windows=sum(windows) / len(windows),
                mean_clauses_per_window=sum(clauses) / len(clauses),
                mean_rouge_l=sum(rouge_l(row['text'], row['reference']) for row in rows) / len(rows),
                mean_length_ratio=sum(row['length_ratio'] for row in rows) / len(rows))


def main():
    left, right = stats(sys.argv[1]), stats(sys.argv[2])
    print(json.dumps(dict(first=left, second=right), indent=2))


if __name__ == '__main__':
    main()
