#!/usr/bin/env python3
"""Selection audit for timestampless (sol61-style) planner captions.

Implements the gates in docs/M2D_PLANNER_SELECTION_PROTOCOL_SOL61.md:
action recall and body-part recall are the primary signals, detail density only
has to fall in a sane band, and repetition collapse disqualifies a checkpoint.

Input rows: {"id", "text" (or "generated"), "reference"}.
Usage:
    python audit_sol61.py gen_a.jsonl gen_b.jsonl
"""
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ACTION = {
    'wrist': r'\bwrist', 'elbow': r'\belbow', 'knee': r'\bknee', 'ankle': r'\bankle',
    'shoulder': r'\bshoulder', 'hip': r'\bhips?\b', 'head': r'\bhead', 'torso': r'\btorso',
    'up': r'\bup\b', 'down': r'\bdown\b', 'left': r'\bleft\b', 'right': r'\bright\b',
    'increase': r'\bincreas', 'decrease': r'\bdecreas', 'cross': r'\bcross',
    'extend': r'\bextend', 'bend': r'\bbend', 'turn': r'\bturn', 'sway': r'\bsway',
    'reach': r'\breach', 'raise': r'\brais', 'lower': r'\blower', 'jump': r'\bjump',
    'spin': r'\bspin', 'kick': r'\bkick', 'step': r'\bstep', 'roll': r'\broll',
    'wave': r'\bwav', 'shake': r'\bshak', 'bounce': r'\bbounc', 'lean': r'\blean',
    'arch': r'\barch', 'pop': r'\bpop', 'slide': r'\bslid', 'grab': r'\bgrab',
    'clap': r'\bclap', 'stomp': r'\bstomp', 'circle': r'\bcirc', 'lift': r'\blift',
}
BODY = {
    'arms': r'\b(?:arms?|elbows?|shoulders?)\b', 'hands': r'\b(?:hands?|palms?|wrists?|fingers?)\b',
    'legs': r'\b(?:legs?|knees?|thighs?)\b', 'feet': r'\b(?:feet|foot|toes?|heels?)\b',
    'torso': r'\b(?:torso|hips?|waist|chest|spine)\b', 'head': r'\b(?:head|neck|chin)\b',
}
# Body parts the captioner has no audio evidence for; a planner that keeps
# inserting them is describing a generic dance rather than this track.
HALLUCINATED = re.compile(r'\b(?:smil\w+|grin\w+|eyes?|gaz\w+|fac(?:e|ial)|express\w+|'
                          r'tail|wing\w*|wheelchair|animal\w*|shadow\w*)\b', re.I)
CLAUSE = re.compile(r'[.;,](?=\s|$)|\band\b|\bwhile\b|\bthen\b')
WINDOW = re.compile(r'\[\s*\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*s\s*\]')

TARGETS = dict(
    action_recall_min=0.55,
    body_recall_min=0.75,
    density_ratio=(0.8, 1.2),      # generated vs reference clauses per 100 chars
    distinct_openings_min=25,
    self_repetition_max=0.10,
    hallucinated_part_rate_max=0.05,
)
DISQUALIFY = dict(self_repetition=0.35, distinct_openings=15, density_ratio=(0.5, 1.6))


def tags(text, patterns):
    return {name for name, pattern in patterns.items() if re.search(pattern, text, re.I)}


def clauses(text):
    return len([c for c in CLAUSE.split(WINDOW.sub(' ', text)) if c.strip()])


def sentences(text):
    return [s.strip() for s in re.split(r'[.;]', WINDOW.sub(' ', text)) if s.strip()]


def repetition(text):
    """Share of 8-word shingles that occur more than once inside one caption."""
    words = re.findall(r"[a-z']+", text.lower())
    size = min(8, max(len(words) // 2, 1))
    shingles = [tuple(words[i:i + size]) for i in range(max(0, len(words) - size + 1))]
    if not shingles:
        return 0.0
    counts = collections.Counter(shingles)
    return sum(c for c in counts.values() if c > 1) / len(shingles)


def rec(xs):
    return statistics.mean(xs) if xs else 0.0


def audit(path):
    rows = [json.loads(line) for line in Path(path).open(encoding='utf-8') if line.strip()]
    if not rows:
        return dict(rows=0, error='empty generation file')
    action_hits, action_tot, body_hits, body_tot = 0, 0, 0, 0
    density_g, density_r, char_ratio = [], [], []
    openers, reps, halluc = set(), [], 0
    sentence_lengths = []
    for row in rows:
        gen = row.get('text') or row.get('generated') or ''
        ref = row.get('reference') or ''
        ga, ra = tags(gen, ACTION), tags(ref, ACTION)
        action_hits += len(ga & ra)
        action_tot += max(len(ra), 1)
        gb, rb = tags(gen, BODY), tags(ref, BODY)
        body_hits += len(gb & rb)
        body_tot += max(len(rb), 1)
        if HALLUCINATED.search(gen):
            halluc += 1
        g_clause = clauses(gen)
        density_g.append(100.0 * g_clause / max(len(gen), 1))
        density_r.append(100.0 * clauses(ref) / max(len(ref), 1))
        char_ratio.append(len(gen) / max(len(ref), 1))
        first = sentences(gen)
        if first:
            openers.add(' '.join(first[0].lower().split()[:4]))
        reps.append(repetition(gen))
        sentence_lengths.extend(len(s.split()) for s in first)
    mean_dg, mean_dr = rec(density_g), rec(density_r)
    m = dict(
        rows=len(rows),
        action_recall=round(action_hits / max(action_tot, 1), 4),
        body_recall=round(body_hits / max(body_tot, 1), 4),
        detail_density=round(mean_dg, 2),
        reference_density=round(mean_dr, 2),
        density_ratio=round(mean_dg / max(mean_dr, 1e-9), 3),
        mean_char_ratio=round(rec(char_ratio), 3),
        mean_sentence_words=round(rec(sentence_lengths), 2),
        distinct_openings=len(openers),
        distinct_opening_rate=round(len(openers) / len(rows), 3),
        self_repetition=round(rec(reps), 4),
        hallucinated_part_rate=round(halluc / len(rows), 4),
    )
    return m


def disqualifiers(m):
    reasons = []
    if m['self_repetition'] > DISQUALIFY['self_repetition']:
        reasons.append('self_repetition>%.2f' % DISQUALIFY['self_repetition'])
    if m['distinct_openings'] < DISQUALIFY['distinct_openings']:
        reasons.append('distinct_openings<%d' % DISQUALIFY['distinct_openings'])
    lo, hi = DISQUALIFY['density_ratio']
    if not lo <= m['density_ratio'] <= hi:
        reasons.append('density_ratio outside %.1f-%.1f' % (lo, hi))
    return reasons


def verdict(m):
    lo, hi = TARGETS['density_ratio']
    checks = {
        'action_recall': m['action_recall'] >= TARGETS['action_recall_min'],
        'body_recall': m['body_recall'] >= TARGETS['body_recall_min'],
        'density_ratio': lo <= m['density_ratio'] <= hi,
        'distinct_openings': m['distinct_openings'] >= TARGETS['distinct_openings_min'],
        'self_repetition': m['self_repetition'] <= TARGETS['self_repetition_max'],
        'hallucinated_parts': m['hallucinated_part_rate'] <= TARGETS['hallucinated_part_rate_max'],
    }
    disq = disqualifiers(m)
    return dict(passed=all(checks.values()) and not disq, checks=checks, disqualified=disq)


def selection_score(m):
    """Primary ranking value from the protocol: action recall, discounted by defects."""
    score = m['action_recall'] + 0.3 * m['body_recall']
    score *= 1.0 - min(m['self_repetition'], 0.5)
    if m['distinct_openings'] < TARGETS['distinct_openings_min']:
        score *= m['distinct_openings'] / float(TARGETS['distinct_openings_min'])
    return round(score, 4)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print('%-28s %5s %7s %6s %7s %6s %8s %6s %6s %6s' % (
        'name', 'rows', 'actRec', 'bodyR', 'densRat', 'lenRat', 'openings', 'selfRp', 'halluc', 'score'))
    for path in sys.argv[1:]:
        name = Path(path).name.replace('_gen.jsonl', '')
        m = audit(path)
        if not m.get('rows'):
            print('%-28s EMPTY (%s)' % (name, m.get('error')))
            continue
        print('%-28s %5d %7.3f %6.3f %7.2f %6.2f %8d %6.3f %6.3f %6.3f' % (
            name, m['rows'], m['action_recall'], m['body_recall'], m['density_ratio'],
            m['mean_char_ratio'], m['distinct_openings'], m['self_repetition'],
            m['hallucinated_part_rate'], selection_score(m)))
        print('    ' + json.dumps(dict(m, verdict=verdict(m), score=selection_score(m)), ensure_ascii=False))


if __name__ == '__main__':
    main()
