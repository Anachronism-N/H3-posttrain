# sol61 captioner: what automatic metrics cannot decide (human review plan)

The automatic audit (`eval/audit_sol61.py`) covers action recall, body-part
recall, density, length, distinct-n, opener entropy, self-repetition,
hallucinated parts, BLEU-1..4, ROUGE-1/2/L, chrF and token-F1. Everything below
needs a human who listens to the audio; no automatic score can stand in for it.

## Blinding

Each item shows one caption plus the 5.0625 s audio, with the model identity and
checkpoint step removed. Order is shuffled per reviewer. Reference captions are
hidden in the main-quality stage so they cannot anchor the judgement; the
reference-consistency stage is a separate pass that shows them explicitly.

## Dimensions (1-5 each, `null` + reason when the evidence is insufficient)

| dimension | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Musical fit | contradicts audible tempo, energy or accents | roughly compatible, generic response | action placement and force track audible changes |
| Long-range organisation | isolated clauses, no development | theme and contrast present, weak ending | clear statement, return, variation, contrast and close |
| Planned executability | ambiguous, contradictory, or impossible in the time | mostly understandable, some order/body relations unclear | a dancer could rehearse it directly: actions, order and simultaneity are unambiguous |
| Boundary quality | adjacent actions contradict or jar | most transitions work, some need unstated conditions | transitions have the preparation and resolution they need |

Do not reward literal repetition, and do not punish a slow or sparse passage;
5 does not require a new action every beat.

## Per-item evidence the reviewer must supply

- audio SHA-256 and the listened range (`[[0, 5.0625]]`; `music_fit` must be
  `null` if the audio was not actually played)
- one exact quoted span from the caption that justifies each score
- for the organisation dimension, the two clauses being compared
- unchanged `review_status` until all three dimensions are filled

## Pairwise preference (separate pass)

For five fixed checkpoint pairs, ask "which plan would you rather rehearse?" with
options A / B / both usable / both unusable / cannot tell. Report the descriptive
win rate `(win + 0.5 * both_usable) / valid comparisons` and always report the two
tie categories separately — a tie between two bad plans must not read as success.

## What to review, in order

1. `sol61_full_20261112_161243/checkpoint-0001000` (current best) — 30 cases.
2. `sol61_full_20261112_161243/checkpoint-0001527` — 30 cases, same items where
   possible, to see whether the extra updates cost leg coverage.
3. High-rate run `sol61hlr_full_20261113_064338/checkpoint-0001000` — 30 cases,
   the candidate that replaces (1) if its audio fit is at least equal.
4. Reference-consistency stage for (1) only: show the stripped reference caption
   and score how much of its stated content the candidate reproduces, without
   letting that score affect the main-quality judgement.

## Agreement reporting

Quadratic-weighted Cohen's kappa per dimension between the two reviewers;
undefined (not 1.0) when fewer than two pairs or when expected disagreement is
zero. Spearman between reviewer mean and any automatic proxy, with source-group
bootstrap intervals, reported separately from the kappa.
