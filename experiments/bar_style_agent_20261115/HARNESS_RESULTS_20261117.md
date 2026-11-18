# Harness comparison and ablations (2026-11-17)

Panel: 20 songs (11 core + 9 extension), 1004 bar windows, per-song SOL61 v3 catalogs,
shared Qwen3-VL-30B outer model (4 replicas), failures counted in every denominator.

## Main comparison

| arm | complete/registered | rate | mean calls (completed) | mean latency | mean input tokens | duplicate rate |
|---|---|---|---|---|---|---|
| W | 2/20 | 0.10 | 48.0 | 757.6 s | 913269 | 0.509 |
| SF | 2/20 | 0.10 | 119.5 | 1312.9 s | 2565518 | 0.598 |
| RA | 14/20 | 0.70 | 99.0 | 1618.2 s | 2379984 | 0.513 |
| O (v3) | 12/20 | 0.60 | 46.8 | 516.8 s | 857395 | 0.410 |

O versions: v1 11/20, v2 11/20 (4096 truncation repair), v3 12/20 (8192 repair),
v4 11/20 (8192 output + deterministic schedule snapping). Snap/long-output did not improve yield.

Strict completion (coverage >= 0.99, duplicate rate <= 0.55, events >= 0.7 * units):
W 1/20, SF 1/20, RA 9/20, O 7/20.

Matched-budget RA (<= 50 calls) was attempted; the first runs did not actually apply the cap
(`max_calls_cap` was not forwarded), so those numbers are invalid and are not reported here.
A corrected run is in flight.

## L1-L3 (Qwen3-Omni-30B, text-only judge)

Absolute scores (1-5):

| arm | n | L1 | L2 | L3 |
|---|---|---|---|---|
| W | 2 | 5.00 | 3.00 | 5.00 |
| SF | 2 | 5.00 | 4.00 | 5.00 |
| RA | 14 | 4.86 | 3.71 | 4.86 |
| O | 12 | 5.00 | 4.45 | 5.00 |

Anonymous pairwise (common completed songs):
O vs RA (9 pairs): L1 1 win / 5 tie / 3 loss; L2 5 win / 4 loss; L3 5 win / 4 loss.
O vs W (2 pairs) and O vs SF (2 pairs): mixed, too small to conclude.
L1/L3 saturate under the text-only rubric; L2 is the only discriminative dimension.

## Ablations on the same 20 songs

| variant | complete/registered | rate | L2 (text judge) |
|---|---|---|---|
| O main (style + acoustic + motif) | 11/20 (v1) | 0.55 | 4.45 |
| no_style | 16/20 | 0.80 | 3.88 |
| no_acoustic | 9/20 | 0.45 | 3.67 |
| no_style_no_acoustic | 17/20 | 0.85 | 3.12 |
| no_motif | 11/20 | 0.55 | 4.18 |
| no_repair | 11/20 | 0.55 | not scored |
| motif_variation | 11/20 | 0.55 | not scored |

Removing the JEV style condition raises deliverability but lowers whole-song organization (L2).
Removing acoustic guidance raises structural failures. Repair toggles do not change yield;
the dominant remaining failure is provider output truncation.

## Limitations

- Expanded panel reuses songs seen in earlier development; not a fresh holdout.
- Two cells per arm failed at dispatch (`no CELL`), an infrastructure failure counted in denominators.
- L1-L3 are text-only model judgments, not human-validated and not audio-based.
- A 50-song panel (43 FMA + 7 AIOZ research-licensed short tracks) is prepared but not yet run.

## Update: matched-budget RA, semantic L1/L3, and newer O versions

Full-budget RA reaches 14/20 but uses 99 calls on average. With max_calls_cap=50 (verified applied,
max calls seen 50), RA drops to **3/20 (15%)**, while O-v3 reaches 12/20 at ~47 calls.
This supports a same-budget comparison instead of a same-capability claim: O dominates at matched budget,
RA dominates only when allowed roughly twice the calls.

O versions: v1 11/20, v2 11/20, v3 12/20, v4 11/20; v5 (continuity prompt + motif variation) running.

Semantic L1 (body part + direction + level + motion continuity), completed plans:

| arm | semantic continuity | explicit contradiction | no shared state |
|---|---:|---:|---:|
| W | 0.696 | 0.067 | 0.000 |
| SF | 0.426 | 0.220 | 0.000 |
| RA | 0.684 | 0.130 | 0.016 |
| O | 0.405 | 0.169 | 0.104 |

L3 structural metrics:

| arm | recurrences per action | varied share | verbatim share |
|---|---:|---:|---:|
| W | 13.8 | 0.275 | 0.196 |
| SF | 9.4 | 0.392 | 0.607 |
| RA | 8.8 | 0.418 | 0.543 |
| O | 8.0 | 0.312 | 0.500 |

Absolute 1-5 model scores saturate for L1/L3 (nearly all 5) even with anchored strict prompts;
they are not used for conclusions. Pairwise O vs RA: L2 5:4, L3 5:4, L1 1 win / 5 tie / 3 loss.

Extra ablations on 20 songs: no_repair 11/20, motif_variation 11/20; repair toggles are not the bottleneck.

Status: 50-song panel (43 FMA + 7 AIOZ research-licensed tracks, 2124 windows) prepared but not yet run.

## Iteration log after the first 20-song report

Goal: raise O's completion above RA's 14/20 without weakening baselines.

Failed attempts (all on the same panel and counted in denominators):
- 8192 block/outline cap (O-v3): 12/20 (best so far).
- schedule snapping after repair (O-v4): 11/20.
- continuity + motif-variation prompts (O-v5b): 10/20; semantic L1 worsened to 0.344.
- duplicate style-prior removal (O-v6/O-v7): 12/20, unchanged.
- 16384 output cap on the eight failed songs (O8c): 0/8 conversions.
- outline compression (one 60-char candidate per window) + two repairs (O9b): 1/8.
- repetition penalty 1.05 + no_repeat_ngram 8 (O11b): 0/8 and JSON-LD degeneration.
- music-only global outline (no window captions, O10): 10/20, but semantic L1 0.519 and duplicate rate 0.355 (quality direction improved).

Interpretation: the remaining failures on this service are model output-instability (malformed JSON, JSON-LD degeneration, loops), not a token-budget limit. Raising caps or adding prompt constraints does not fix them. Matched-budget O still dominates RA (60% vs 15% at <=50 calls).

Expansion status: 23 additional FMA songs and 7 AIOZ research-licensed short tracks are running through the same four-arm protocol; results will be appended when the matrices finish.
