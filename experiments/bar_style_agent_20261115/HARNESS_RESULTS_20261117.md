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
