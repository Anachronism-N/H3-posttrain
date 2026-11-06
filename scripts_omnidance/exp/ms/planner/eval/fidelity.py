#!/usr/bin/env python3
"""Compare generated captions with reference captions on shared structural metrics."""
import json
import re
import sys
from pathlib import Path

MARKER = re.compile(r'\[(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*s?\]')
ACTION = {
    'wrist': r'\bwrist\w*', 'elbow': r'\belbow\w*', 'knee': r'\bknee\w*', 'ankle': r'\bankle\w*',
    'shoulder': r'\bshoulder\w*', 'hip': r'\bhips?\b', 'head': r'\bhead\w*', 'torso': r'\btorso\w*',
    'up': r'\bup\b', 'down': r'\bdown\b', 'left': r'\bleft\b', 'right': r'\bright\b',
    'increases': r'\bincreas\w*', 'decreases': r'\bdecreas\w*', 'cross': r'\bcross\w*',
    'extend': r'\bextend\w*', 'bend': r'\bbend\w*', 'turn': r'\bturn\w*', 'sway': r'\bsway\w*',
    'reach': r'\breach\w*', 'raise': r'\brais\w*', 'lower': r'\blower\w*',
}
BODY = {
    'arms': r'\b(?:arms?|elbows?|shoulders?)\b', 'hands': r'\b(?:hands?|palms?|wrists?|fingers?)\b',
    'legs': r'\b(?:legs?|knees?|thighs?)\b', 'feet': r'\b(?:feet|foot|toes?|heels?)\b',
    'torso': r'\b(?:torso|hips?|waist|chest|spine)\b', 'head': r'\b(?:head|neck|chin)\b',
}


def tags(text, patterns):
    return {name for name, pattern in patterns.items() if re.search(pattern, text, re.I)}


def coverage(text, duration=5.0625):
    ends = [float(end) for _, end in MARKER.findall(text)]
    return min(max(ends, default=0.0), duration) / duration


def main():
    generations = [json.loads(line) for line in Path(sys.argv[1]).open(encoding='utf-8')]
    manifest = {row['id']: row for row in json.loads(Path(sys.argv[2]).read_text())}
    rows = []
    for record in generations:
        reference = manifest[record['id']].get('reference_caption')
        if not reference:
            continue
        generated = record['raw_text']
        rows.append(dict(
            id=record['id'], genre=record['genre'],
            action_jaccard=len(tags(generated, ACTION) & tags(reference, ACTION)) /
                           max(len(tags(generated, ACTION) | tags(reference, ACTION)), 1),
            body_jaccard=len(tags(generated, BODY) & tags(reference, BODY)) /
                         max(len(tags(generated, BODY) | tags(reference, BODY)), 1),
            coverage_generated=round(coverage(generated), 3),
            coverage_reference=round(coverage(reference), 3),
            length_ratio=round(len(generated) / max(len(reference), 1), 3),
        ))
    summary = dict(cases=len(rows),
                   mean_action_jaccard=sum(r['action_jaccard'] for r in rows) / max(len(rows), 1),
                   mean_body_jaccard=sum(r['body_jaccard'] for r in rows) / max(len(rows), 1),
                   mean_coverage_generated=sum(r['coverage_generated'] for r in rows) / max(len(rows), 1),
                   mean_coverage_reference=sum(r['coverage_reference'] for r in rows) / max(len(rows), 1),
                   mean_length_ratio=sum(r['length_ratio'] for r in rows) / max(len(rows), 1),
                   rows=rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
