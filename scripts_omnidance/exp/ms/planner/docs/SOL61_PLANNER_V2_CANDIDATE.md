# sol61 planner v2 candidate (clause-level leg weighting, step 500)

LoRA adapter for the Qwen3-Omni-Captioner base that writes one dance caption per
5.0625 s music window. This is the current best measured checkpoint; **v1
(`../sol61_planner_v1`, `sol61_full_20261112_161243/checkpoint-0001000`) remains
the model named in the frozen benchmark plan until test384 is run.**

All numbers below were produced on the **Xingyun (Nebula) L20Z cluster**
(79 GB/GPU, `future_hz_qwenvideo`), 80 SHA-ordered sol61-v1 val cases, decoding
`temperature=0.7, top_p=0.9, num_beams=1, repetition_penalty=1.05,
no_repeat_ngram_size=4`.

| metric | v2 @500 | v1 @1000 | reference |
| --- | --- | --- | --- |
| action recall | **0.880** | 0.871 | 1.000 |
| body-part recall | **0.968** | 0.945 | 1.000 |
| leg mentions | **0.875** | 0.787 | 0.993 |
| detail density ratio | 0.98 | 1.02 | 1.00 |
| length ratio | **1.04** | 0.93 | 1.00 |
| distinct openings | **57** | 54 | — |
| opener entropy | **5.01** | 4.80 | 5.57 |
| self-repetition | 0.000 | 0.000 | 0.000 |
| BLEU-1 / BLEU-4 | **0.535 / 0.072** | 0.516 / 0.055 | — |
| ROUGE-L | 0.292 | 0.290 | — |
| token-F1 | **0.598** | 0.594 | — |

## What changed

Same data, same schedule (1e-5, 32 GPU, accum 4, global batch 128), plus
`PLANNER_LEG_WEIGHT=1.5`: the CE of every clause naming a leg part
(`leg/knee/thigh/shin/calf/hamstring/quad`) is multiplied by 1.5. Token-level
weighting was tried first and abandoned — leg words are <0.5% of assistant
tokens, so it barely moved the loss.

Provenance: run `sol61leg3_full_20261113_193635`, attempt
`...-attempt194115`, `checkpoint-000500`; adapter files match the run store
byte-for-byte (copy, not rebuild).

## Usage

Identical to v1: audio-only user turn, exactly 5.0625 s, and an explicit chat
EOS (the checkpoint's `generation_config` does not carry `<|im_end|>`, so without
it generation runs past the caption into symbol spam). See
`../sol61_planner_v1/README.md` for the verified call and the
`planner/eval_harness/generate_caption.py` entry point.

## Caveats

- Only the 500-step checkpoint is measured. 1,000 and 1,527 were not evaluated
  (those evaluations were paused to concentrate compute on the frozen test384
  run), so "500 is the best of this run" is an inference from the v1 curve, not
  a measurement.
- Selection used the same 80 val cases that tuned the decoding temperature, so
  the numbers carry mild selection optimism; the frozen test384 run is the
  unbiased number of record.
- Leg coverage is still 12 points below the reference (0.875 vs 0.993).
