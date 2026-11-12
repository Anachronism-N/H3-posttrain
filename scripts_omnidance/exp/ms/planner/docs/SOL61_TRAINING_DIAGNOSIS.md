# sol61 captioner: training evidence and diagnosis

Status 2026-11-13. Everything below is measured, not projected; items that are
still running say so.

## 1. Data

`sol61-v1`: 217,373 rows from `avagelion/dance-data` → `sol61_captions`
(revision `1ca58b4`), gate-false rows (842) and rows without verified audio
(2,452) dropped, `Dance style:` header stripped. Splits train 195,367 /
val 10,938 / test 11,068, zero cross-split groups and zero id overlap. Mean 768
chars, no timestamps. Contract `caption_contract_nostyle.json`,
`caption_version=m2d-sol61-v1`.

## 2. Completed runs

Global batch 128 (micro 1), LoRA r32/α64 on the last four audio-tower layers plus
projector, ZeRO-3, max length 8,192, one epoch = 1,527 updates.

| run | GPUs | token weight | val NLL @500 / 1000 / 1500 / 1527 |
| --- | --- | --- | --- |
| `sol61_full_20261112_102148` | 32 | 1.0 | 1.581 / 1.531 / 1.523 / **1.522** |
| `sol61trick_full_20261112_093656` | 32 | 1.3 | 2.056 / 1.993 / 1.981 / 1.981 |

Second batch (`..._161243`, `..._161536`) saved and validated every 100 updates:
16 checkpoints each, same plateau.

Dense curve from the 64-GPU run `sol61_full_20261113_010434` (per 100 updates):
1.901, 1.717, 1.646, 1.605, 1.578, 1.560, 1.547. Per-100 improvement falls
−0.184 → −0.072 → −0.041 → −0.026 → −0.018 → −0.013, i.e. the first epoch is
saturating. Validation never rose, so nothing overfit; the budget is simply small
(1,527 optimizer updates for a 43M-parameter adapter).

## 3. Measured generation quality

`eval/audit_sol61.py`, hz 1527, 20 val rows, greedy, `--max-new-tokens 2048`:

| metric | value | gate |
| --- | --- | --- |
| action recall | 0.838 | ≥0.55 pass |
| body-part recall | 0.944 | ≥0.75 pass |
| density ratio | 1.081 | 0.8–1.2 pass |
| self-repetition | 0.000 | ≤0.10 pass |
| hallucinated parts | 0.000 | ≤0.05 pass |
| distinct openings | 9/20 | size-scaled pass |

Shape matches the references: 5.45 vs 5.50 sentences per caption, 13/20 identical
sentence counts, mean length ratio 0.865.

## 4. Defects found (with evidence)

### 4.1 Template collapse across samples

`self_repetition` measures within-caption 8-gram reuse and reports 0.000, which
hid the real problem. Cross-sample statistics on the same 90 generations:

| metric | 500 | 1000 | 1500 | reference |
| --- | --- | --- | --- | --- |
| distinct-2 | 0.193 | 0.210 | 0.207 | **0.353** |
| opener entropy (bits) | 1.71 | 3.02 | 2.92 | **4.44** |
| most common 3-word opener | 0.667 | 0.333 | 0.300 | **0.133** |

At 500 updates two thirds of the captions open with the same three words. The
training data's most frequent opener is only 4.4% of captions, so this is model
degeneration, not the data distribution. `The dancer lowers one hand.` occurs 21
times in 90 generations.

### 4.2 Leg coverage degrades with training

| checkpoint | action recall | body recall | missing `legs` | extra `legs` |
| --- | --- | --- | --- | --- |
| 500 | 0.821 | 0.938 | 5 | 6 |
| 1000 | 0.783 | 0.889 | 9 | 6 |
| 1500 | 0.810 | 0.815 | 15 | 3 |

The entire body-recall drop is legs/knees: references say "the knee on that side
softens" and the later checkpoint stops mentioning knees at all. Other body parts
are stable. More updates concentrate the output on high-frequency arm vocabulary.

### 4.3 Sentence-level complexity

Generated sentences average 22.2 words / 1.04 commas vs reference 26.9 / 1.51;
direction words per caption 4.6/4.6 vs 5.9/5.7. Missing vocabulary concentrates
in `bend`, `reach`, `up`, `elbow`, `knee` while `raise`, `lower`, `down` appear
instead, so most of the recall gap is wording preference (ordinal references such
as `the first/second hand` are used 4x more often than the references do).

### 4.4 Text-overlap metrics rank differently from NLL

30-row BLEU/ROUGE on hz 500/1000/1500:

| checkpoint | BLEU-1 | BLEU-4 | ROUGE-L | token-F1 |
| --- | --- | --- | --- | --- |
| 500 | 0.528 | 0.083 | 0.318 | 0.613 |
| 1000 | 0.494 | 0.052 | 0.301 | 0.590 |
| 1500 | 0.503 | 0.081 | 0.307 | 0.601 |

Text overlap prefers 500 while validation NLL prefers 1527. The 500-step model
wins on overlap partly because its collapsed opener is also the reference's most
common opener, which is why the protocol forbids using overlap as the sole key.

## 5. Working hypothesis

The dominant cause is **learning rate too low for LoRA** (`lora_lr=1e-5`;
common LoRA ranges are 1e-4–3e-4). Evidence: loss plateaus while long-tail
vocabulary (legs, reach, bend) and structural diversity are never learned, and
`sentences per caption` is already correct while `words per sentence` is 18% low.
A 1e-4 run with warmup 100 was started and then stopped to give the main run
bandwidth; it must be re-run to confirm.

## 6. In flight

- `sol61-2ep-64gpu` (`sol61_full_20261113_010434`): 2 epochs, 8 nodes × 8 GPUs,
  global batch still 128 with accum 2, save + validate every 100 updates.
- `sol61-eval-80x7`: 7 checkpoints × 80 fixed val rows with the full audit.
  200 rows was planned first and abandoned: at ~2 min/row on a 30B single-GPU
  shard it needed 9 hours.

## 7. E2 baseline readiness

Weights for six E2 baselines are downloaded and mirrored to
`oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/models/<name>/`.

| model | load | generate | obstacle |
| --- | --- | --- | --- |
| Qwen3-Omni-30B-A3B-Instruct | — | — | local copy had 1/15 shards; now complete and re-uploaded |
| Qwen2.5-Omni-7B | ok (20.9 GB) | partial | needs audio resampling 24k→16k (fixed), some rows still error |
| MOSS-Music-8B | ok | — | `modeling_moss_music.py` uses relative imports; needs package-style load |
| Music-Flamingo-2601 | ok | 0 rows | processor field mismatch after dtype fix |
| Audio-Flamingo-3 | ok | 0 rows | same family as above |
| MOSS-Audio-8B | fail | — | upstream repo ships no modeling file and transformers 5.18 has no native `moss_audio` |

The probe harness is `artifacts/sol61-e2-baselines-job/e2_baselines.py`; the
first-pass comparison runner is `sol61-e2-baselines-job/entry.py`.
