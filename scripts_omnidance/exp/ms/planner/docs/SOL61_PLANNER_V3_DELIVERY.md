# sol61 planner v3 (clause-weighted, step 1000) — the model of record

LoRA adapter for the Qwen3-Omni-Captioner base. **This is the best measured
checkpoint** and replaces v1 (`sol61_full_20261112_161243/checkpoint-0001000`)
as the planner of record. Selected on the 80-case val panel *before* the frozen
test384 run.

All numbers are from the **Xingyun (Nebula) L20Z cluster** with the frozen
contract: audio-only user turn, exactly 5.0625 s, `temperature 0.7, top_p 0.9,
top_k 20, num_beams 1, repetition_penalty 1.05, no_repeat_ngram 4`, explicit chat
EOS, per-case seed.

## Frozen test384 (384 cases, 384/384 ok, 384/384 stopped on EOS)

| metric | v1 @1000 | **v3 @1000** |
| --- | --- | --- |
| action recall | 0.838 | **0.859** |
| body-part recall | 0.927 | **0.970** |
| corpus BLEU-4 (raw sacrebleu) | 8.98 | **9.66** |
| ROUGE-L F1 (rouge_score, stemming) | 0.3050 | **0.3087** |
| BERTScore F1 | 0.8975 | 0.8974 |
| length ratio | **1.006** | 1.073 |

Paired over 384 source groups, 4,000 bootstrap resamples:

| metric | delta | 95% CI | win / loss / tie |
| --- | --- | --- | --- |
| action recall | +0.0187 | [0.0022, 0.0354] | 41.1% / 31.5% / 27.3% |
| body-part recall | +0.0382 | [0.0234, 0.0530] | 16.4% / 4.9% / 78.6% |

Both intervals exclude zero, and all three text metrics are equal or better
(BLEU-4 9.66 vs 8.98, ROUGE-L 0.3087 vs 0.3050, BERTScore tied). The one cost is
length: captions run ~7% longer than the reference.

Val panel (80 cases, temperature 0.7, per-case seeds): action 0.875 / body 0.995
/ 57 distinct openings / length 1.02 — best of the ten configurations tested
(v1 at 500/1000/1527, v3 at 500/1000, and four decoding variants).

## What it is

Same data and schedule as v1 (sol61-v1, 1e-5 LoRA, 32 GPU, accum 4, global batch
128), plus `PLANNER_LEG_WEIGHT=1.5`: the CE of every **clause** naming a leg part
(`leg/knee/thigh/shin/calf/hamstring/quad`) is multiplied by 1.5. Token-level
weighting was measured first and rejected — leg words are <0.5% of assistant
tokens and moved nothing.

Limitation to state alongside any use: this is a single intervention on the one
body group where the frozen references have mass and the baseline does not
(legs 0.993 reference coverage vs 0.787 baseline; arms/hands are already 1.00 and
feet/torso/head never appear in the references). The random-clause control that
would separate "leg semantics" from "extra emphasis on saturated clauses" has
not been run.

## Provenance

`sol61leg3_full_20261113_193635` / attempt `...-attempt194115` /
`checkpoint-0001000`. Adapter files are copied byte-for-byte from the run store;
the frozen test384 generation is
`sol61_eval_test384_leg31000_20261114_025608`.
