#!/usr/bin/env python3
"""Per-body-part mention rates for planner generations, against the reference.

The aggregate body recall in audit_sol61.py mixes six groups; a leg-weighted
training run has to move the *legs* row specifically. Reports, per file, the
rate at which each group is named, plus recall/precision against the reference
for that group.

Usage:
    python body_breakdown.py gen_a.jsonl gen_b.jsonl [--reference-column reference]
"""
import argparse
import json
import re

BODY = {
    'arms': r'\b(?:arms?|elbows?|shoulders?)\b',
    'hands': r'\b(?:hands?|palms?|wrists?|fingers?)\b',
    'legs': r'\b(?:legs?|knees?|thighs?|shins?|calves?|calf|hamstrings?|quads?)\b',
    'feet': r'\b(?:feet|foot|toes?|heels?|ankles?)\b',
    'torso': r'\b(?:torso|hips?|waist|chest|spine)\b',
    'head': r'\b(?:head|neck|chin)\b',
}


def has(text, pattern):
    return bool(re.search(pattern, text or '', re.I))


def rows(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score(items):
    out = {}
    for name, pattern in BODY.items():
        gen = [has(row.get('text'), pattern) for row in items]
        ref = [has(row.get('reference'), pattern) for row in items]
        tp = sum(1 for g, r in zip(gen, ref) if g and r)
        out[name] = dict(
            generated=round(sum(gen) / len(items), 3),
            reference=round(sum(ref) / len(items), 3),
            recall=round(tp / max(sum(ref), 1), 3),
            precision=round(tp / max(sum(gen), 1), 3),
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    header = 'file'.ljust(30) + ''.join(name.rjust(22) for name in BODY)
    print(header)
    for path in args.files:
        items = rows(path)
        result = score(items)
        cells = ''.join(('%s/%s' % (result[n]['generated'], result[n]['recall'])).rjust(22)
                        for n in BODY)
        print(path.split('/')[-1][:29].ljust(30) + cells)
    print('\ncell = mention_rate (generated) / recall vs reference')
    for path in args.files:
        print(path, json.dumps(score(rows(path)), indent=1))


if __name__ == '__main__':
    main()
