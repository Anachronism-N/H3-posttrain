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
import math
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


def _tok(text):
    return re.findall(r"[a-z']+", text.lower())


def _ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def bleu(gen, ref, n=4):
    """Corpus-free sentence BLEU with add-epsilon smoothing and brevity penalty."""
    g, r = _tok(gen), _tok(ref)
    if len(g) < n or not r:
        return 0.0
    logs = []
    for k in range(1, n + 1):
        gc, rc = collections.Counter(_ngrams(g, k)), collections.Counter(_ngrams(r, k))
        hit = sum(min(c, rc[x]) for x, c in gc.items())
        logs.append(math.log((hit or 1e-9) / max(sum(gc.values()), 1)))
    penalty = min(0.0, 1 - len(r) / max(len(g), 1))
    return math.exp(sum(logs) / n + penalty)


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(cur[j - 1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l(gen, ref):
    g, r = _tok(gen), _tok(ref)
    if not g or not r:
        return 0.0
    common = _lcs(g, r)
    p, rec_ = common / len(g), common / len(r)
    return 0.0 if p + rec_ == 0 else 2 * p * rec_ / (p + rec_)


def token_f1(gen, ref):
    g, r = collections.Counter(_tok(gen)), collections.Counter(_tok(ref))
    hit = sum((g & r).values())
    if not hit:
        return 0.0
    p, rec_ = hit / sum(g.values()), hit / sum(r.values())
    return 2 * p * rec_ / (p + rec_)


def distinct_n(texts, n):
    """Type/token ratio of n-grams over the whole sample; low values mean templating."""
    grams, total = set(), 0
    for text in texts:
        for gram in _ngrams(_tok(text), n):
            grams.add(gram)
            total += 1
    return len(grams) / max(total, 1)


def opener_entropy(texts, k=3):
    """Shannon entropy over the first-k words of each caption, in bits."""
    heads = collections.Counter(' '.join(_tok(t)[:k]) for t in texts)
    total = sum(heads.values())
    return -sum((c / total) * math.log2(c / total) for c in heads.values())


def top_opener_share(texts, k=3):
    heads = collections.Counter(' '.join(_tok(t)[:k]) for t in texts)
    return max(heads.values()) / max(sum(heads.values()), 1)


def rec(xs):
    return statistics.mean(xs) if xs else 0.0


def audit(path):
    rows = [json.loads(line) for line in Path(path).open(encoding='utf-8') if line.strip()]
    if not rows:
        return dict(rows=0, error='empty generation file')
    action_hits, action_tot, body_hits, body_tot = 0, 0, 0, 0
    density_g, density_r, char_ratio = [], [], []
    openers, reps, halluc = set(), [], 0
    gen_texts, ref_texts = [], []
    b1, b4, rl, tf = [], [], [], []
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
        gen_texts.append(gen)
        ref_texts.append(ref)
        b1.append(bleu(gen, ref, 1))
        b4.append(bleu(gen, ref, 4))
        rl.append(rouge_l(gen, ref))
        tf.append(token_f1(gen, ref))
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
        distinct_1=round(distinct_n(gen_texts, 1), 4),
        distinct_2=round(distinct_n(gen_texts, 2), 4),
        opener_entropy=round(opener_entropy(gen_texts), 3),
        top_opener_share=round(top_opener_share(gen_texts), 4),
        bleu1=round(rec(b1), 4),
        bleu4=round(rec(b4), 4),
        rouge_l=round(rec(rl), 4),
        token_f1=round(rec(tf), 4),
    )
    return m


def opening_floor(m, absolute, ratio):
    """Absolute counts only apply to samples large enough to reach them."""
    return min(absolute, max(1, int(ratio * m['rows'])))


def disqualifiers(m):
    reasons = []
    if m['self_repetition'] > DISQUALIFY['self_repetition']:
        reasons.append('self_repetition>%.2f' % DISQUALIFY['self_repetition'])
    floor = opening_floor(m, DISQUALIFY['distinct_openings'], 0.15)
    if m['distinct_openings'] < floor:
        reasons.append('distinct_openings<%d' % floor)
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
        'distinct_openings': m['distinct_openings'] >= opening_floor(
            m, TARGETS['distinct_openings_min'], 0.4),
        'self_repetition': m['self_repetition'] <= TARGETS['self_repetition_max'],
        'hallucinated_parts': m['hallucinated_part_rate'] <= TARGETS['hallucinated_part_rate_max'],
    }
    disq = disqualifiers(m)
    return dict(passed=all(checks.values()) and not disq, checks=checks, disqualified=disq)


def selection_score(m):
    """Primary ranking value from the protocol: action recall, discounted by defects."""
    score = m['action_recall'] + 0.3 * m['body_recall']
    score *= 1.0 - min(m['self_repetition'], 0.5)
    floor = opening_floor(m, TARGETS['distinct_openings_min'], 0.4)
    if m['distinct_openings'] < floor:
        score *= m['distinct_openings'] / float(floor)
    return round(score, 4)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print('%-28s %5s %7s %6s %7s %6s %8s %6s %6s %6s %6s %6s' % (
        'name', 'rows', 'actRec', 'bodyR', 'densRat', 'lenRat', 'openings', 'selfRp', 'halluc',
        'B1', 'B4', 'RGL'))
    for path in sys.argv[1:]:
        name = Path(path).name.replace('_gen.jsonl', '')
        m = audit(path)
        if not m.get('rows'):
            print('%-28s EMPTY (%s)' % (name, m.get('error')))
            continue
        print('%-28s %5d %7.3f %6.3f %7.2f %6.2f %8d %6.3f %6.3f %6.3f %6.3f %6.3f' % (
            name, m['rows'], m['action_recall'], m['body_recall'], m['density_ratio'],
            m['mean_char_ratio'], m['distinct_openings'], m['self_repetition'],
            m['hallucinated_part_rate'], m['bleu1'], m['bleu4'], m['rouge_l']))
        print('    ' + json.dumps(dict(m, verdict=verdict(m), score=selection_score(m)), ensure_ascii=False))


if __name__ == '__main__':
    main()
