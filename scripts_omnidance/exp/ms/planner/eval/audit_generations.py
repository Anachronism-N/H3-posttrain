#!/usr/bin/env python3
"""Offline audit of planner generations: window structure, density, diversity.

Reads one or more ``<name>_gen.jsonl`` files (each row: id, text, reference) and
prints the metrics used for checkpoint selection, so comparisons do not depend on
the training/eval job that produced them.

Usage:
    python audit_generations.py docs/sft198k_5515_gen.jsonl docs/fullv4_2000_gen.jsonl
"""
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

WINDOW = re.compile(r'\[\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*s\s*\]')
TARGET_SECONDS = 5.0625


def spans(text):
    return [(float(a), float(b)) for a, b in WINDOW.findall(text)]


def clauses(text):
    parts = [p for p in WINDOW.split(text) if p.strip()]
    return sum(len([s for s in re.split(r'[;。]', p) if s.strip()]) for p in parts)


def ngrams(text, n=3):
    words = re.findall(r'[a-z]+', text.lower())
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def reference_recall(row):
    ref = ngrams(row['reference'])
    return len(ngrams(row['text']) & ref) / max(len(ref), 1)


def kl(gen, ref):
    keys = set(gen) | set(ref)
    total_g, total_r = sum(gen.values()), sum(ref.values())
    return sum((gen[k] / total_g) * math.log((gen[k] / total_g) / ((ref[k] + 1e-9) / total_r))
               for k in keys if gen[k])


def audit(path):
    rows = [json.loads(line) for line in Path(path).open(encoding='utf-8') if line.strip()]
    if not rows:
        return dict(rows=0, error='empty generation file', verdict=dict(passed=False, checks={}))
    gen_counts = collections.Counter()
    ref_counts = collections.Counter()
    window_totals = []
    durations, per_window_clauses, lengths = [], [], []
    openers = set()
    recalls = []
    overruns = gaps = overlaps = 0
    for row in rows:
        g, r = spans(row['text']), spans(row['reference'])
        gen_counts[len(g)] += 1
        ref_counts[len(r)] += 1
        window_totals.append(len(g))
        lengths.append(len(row['text']) / max(len(row['reference']), 1))
        openers.add(re.sub(r'\s+', ' ', row['text'][:60]))
        recalls.append(reference_recall(row))
        if not g:
            overruns += 1
            continue
        if g[-1][1] > TARGET_SECONDS + 0.3:
            overruns += 1
        durations.extend(b - a for a, b in g)
        parts = [p for p in WINDOW.split(row['text']) if p.strip()]
        per_window_clauses.extend(len([s for s in re.split(r'[;。]', p) if s.strip()]) for p in parts)
        gaps += sum(1 for i in range(len(g) - 1) if g[i + 1][0] > g[i][1] + 1e-6)
        overlaps += sum(1 for i in range(len(g) - 1) if g[i + 1][0] < g[i][1] - 1e-6)
    return dict(
        rows=len(rows),
        gen_window_dist=dict(sorted(gen_counts.items())),
        ref_window_dist=dict(sorted(ref_counts.items())),
        window_kl=round(kl(gen_counts, ref_counts), 3),
        mean_windows=round(statistics.mean(window_totals), 2) if window_totals else 0.0,
        mean_window_duration=round(statistics.mean(durations), 3) if durations else 0.0,
        mean_clauses_per_window=round(statistics.mean(per_window_clauses), 3) if per_window_clauses else 0.0,
        mean_length_ratio=round(statistics.mean(lengths), 3),
        distinct_openers=len(openers),
        mean_ref_3gram_recall=round(statistics.mean(recalls), 3) if recalls else 0.0,
        cases_with_overrun=overruns,
        window_gaps=gaps,
        window_overlaps=overlaps,
    )


TARGETS = dict(window_kl_max=0.10, clauses_per_window_min=1.6, window_duration_max=2.1,
               distinct_openers_min=20, overrun_max=0)


def proximity(m):
    """Continuous 0-1 score: how far each gate is from satisfied, averaged.

    Lets partially-improved checkpoints be ranked before any of them passes all
    gates (the binary verdict saturates at 0 for a long time).
    """
    def ratio(value, target, higher_better):
        x = value / target if target else 0.0
        return min(x, 1.0) if higher_better else min(target / max(value, 1e-9), 1.0)
    parts = [
        ratio(m['window_kl'], TARGETS['window_kl_max'], higher_better=False),
        ratio(m['mean_clauses_per_window'], TARGETS['clauses_per_window_min'], higher_better=True),
        ratio(m['mean_window_duration'], TARGETS['window_duration_max'], higher_better=False),
        ratio(m['distinct_openers'], TARGETS['distinct_openers_min'], higher_better=True),
        1.0 if m['cases_with_overrun'] <= TARGETS['overrun_max'] else 0.0,
    ]
    return round(sum(parts) / len(parts), 4)


def verdict(m):
    """Selection-protocol pass/fail for one checkpoint."""
    checks = {
        'window_kl': m['window_kl'] <= TARGETS['window_kl_max'],
        'clauses_per_window': m['mean_clauses_per_window'] >= TARGETS['clauses_per_window_min'],
        'window_duration': m['mean_window_duration'] <= TARGETS['window_duration_max'],
        'distinct_openers': m['distinct_openers'] >= TARGETS['distinct_openers_min'],
        'no_overrun': m['cases_with_overrun'] <= TARGETS['overrun_max'],
    }
    return dict(passed=all(checks.values()), checks=checks)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print('%-22s %5s %7s %7s %7s %7s %6s %6s %7s %6s' % (
        'name', 'rows', 'winKL', 'winDur', 'clause/win', 'lenRatio', 'openers', 'recall', 'overrun', 'prox'))
    for path in sys.argv[1:]:
        name = Path(path).name.replace('_gen.jsonl', '')
        m = audit(path)
        if m.get('rows', 0) == 0:
            print('%-22s EMPTY (%s)' % (name, m.get('error')))
            continue
        print('%-22s %5d %7.3f %7.2f %7.2f %7.2f %6d %6.3f %7d %6.3f' % (
            name, m['rows'], m['window_kl'], m['mean_window_duration'],
            m['mean_clauses_per_window'], m['mean_length_ratio'], m['distinct_openers'],
            m['mean_ref_3gram_recall'], m['cases_with_overrun'], proximity(m)))
    for path in sys.argv[1:]:
        name = Path(path).name.replace('_gen.jsonl', '')
        metrics = audit(path)
        print(name, json.dumps(dict(metrics, proximity=proximity(metrics), verdict=verdict(metrics)),
                               ensure_ascii=False))


if __name__ == '__main__':
    main()
