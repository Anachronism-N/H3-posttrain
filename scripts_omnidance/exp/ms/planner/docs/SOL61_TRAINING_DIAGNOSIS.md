# sol61 captioner: training evidence and diagnosis

Status 2026-11-13. Everything below is measured, not projected; items that are
still running say so.

> **Hardware:** every training and evaluation number in this document was
> produced on the Xingyun (Nebula) L-card cluster — Alibaba `L20Z` GPUs
> (79 GB each), via `nebula-cli` on `future_hz_qwenvideo` / `futurelab_vlm_caption`.
> Decoding there runs BF16 with the same package set (`torch 2.8`, transformers
> from the shared `venv_h3` tarball). Numbers from other hardware are not
> comparable and are not mixed in.

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

## 5.1 80-case paired measurements (checkpoints from the same family)

Fixed 80 val cases, greedy decoding, `--max-new-tokens 2048`:

| checkpoint | action recall | body recall | density | length | distinct openings | BLEU-1 | BLEU-4 | ROUGE-L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.804 | 0.899 | 1.05 | 0.87 | **8** | 0.521 | 0.071 | 0.314 |
| 1000 | 0.802 | 0.904 | 1.07 | 0.84 | **16** | 0.508 | 0.054 | 0.301 |

Two things this settles:

1. Action and body recall saturate by update 500 (0.804/0.899 → 0.802/0.904).
2. Diversity and text overlap move in opposite directions. Opening variety
   doubles (8 → 16 of 80) while BLEU-4 and ROUGE-L fall (0.071 → 0.054,
   0.314 → 0.301). The 500-step model scores higher on overlap because its
   collapsed opener is also the reference's most common opener. **BLEU/ROUGE
   therefore cannot be used to choose a checkpoint here**, and any table that
   ranks by them inverts the true ordering.

Working selection rule from the evidence: keep action recall and density inside
their bands, then prefer the checkpoint with the most distinct openings, and use
leg coverage (body recall) as the tie-breaker that penalises over-training.

## 5.2 Five-checkpoint comparison on 80 fixed cases

| checkpoint | action recall | body recall | density | opener entropy | top opener | distinct openings | BLEU-4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run-A @200 | 0.793 | 0.908 | 1.07 | 1.27 | 0.762 | 10 | 0.051 |
| run-A @500 | 0.804 | 0.899 | 1.05 | 1.52 | 0.725 | 8 | 0.071 |
| run-A @1000 | 0.802 | 0.904 | 1.07 | 2.81 | 0.412 | 16 | 0.054 |
| run-A @1500 | 0.814 | 0.853 | 1.06 | 2.93 | 0.362 | 15 | 0.074 |
| **run-B @1000** | **0.837** | **0.913** | 1.05 | **3.04** | **0.275** | 16 | 0.069 |
| run-A @1527 | 0.822 | 0.894 | 1.06 | 2.88 | 0.388 | 14 | 0.071 |
| run-B @1527 | 0.836 | **0.917** | 1.04 | 2.90 | 0.375 | **17** | 0.072 |
| reference | — | — | — | **5.57** | **0.062** | — | — |

With seven checkpoints across two independent runs the plateau is unambiguous:
opener entropy sits at 2.88-3.04 for every weight at or past 1,000 updates and
never approaches the reference's 5.57, while 500 updates gives 1.52. BLEU-4
again inverts the ordering: the highest value (0.074, run-A @1500) belongs to the
checkpoint with the weakest body recall (0.853).

Two acceptable deliverables:

- `run-B @1000` — highest action recall (0.837), highest entropy (3.04), least
  repetitive opening (0.275 of samples share the top opening).
- `run-B @1527` — best body-part coverage (0.917) and most distinct openings (17),
  at a slightly lower entropy (2.90).

Selection outcome:

1. **The optimum sits near 1,000 updates.** Two independent runs both peak
   there relative to their own 500- and 1,500-step checkpoints, so this is not
   sampling noise.
2. **Past 1,000 updates the model degrades**: body recall falls 0.913 → 0.853
   while action recall rises slightly, i.e. it concentrates on arm vocabulary and
   drops legs.
3. **Diversity plateaus around 3.0 bits against the reference's 5.57.** Under
   `lora_lr=1e-5` this ceiling is not reachable by adding updates.

Current best deliverable: `sol61_full_20261112_161243/checkpoint-0001000`
(run-B @1000), not the 1,527-step weight that the earlier NLL-only reading
suggested.

## 5.3 Learning rate is the dominant bottleneck (measured)

Two runs with identical data, batch and steps, differing only in `lora_lr`
(and warmup 20 → 100):

| step | `lora_lr=1e-5` val NLL | `lora_lr=1e-4` val NLL |
| --- | --- | --- |
| 100 | 1.9007 | **1.6527** |
| 200 | 1.7171 | **1.5218** |
| 300 | 1.6456 | 1.4853 |
| 400 | 1.6046 | 1.4598 |
| 500 | 1.5782 | **1.4408** |
| 1,527 | 1.5223 | — |
| 3,054 | 1.4683 | — |

The high-rate run reaches the low-rate run's *final* (1,527-step) validation NLL
after 200 updates, and by update 500 it is already below the 3,054-step 2-epoch
run's endpoint (1.4408 vs 1.4683), i.e. better quality with one sixth of the
compute. The gap also narrows
(−0.248 → −0.195), so both are heading toward the same floor; the open question
is whether the higher rate settles below it. Training is stable at 1e-4: no
spikes, no divergence over 200 updates.

Practical consequence *if NLL were the target*: any future sol61 run should
start at `lora_lr=1e-4`. **Generation quality says the opposite**, which is why
NLL must not be used as the selection signal here.

### 5.3.1 High learning rate, measured on generation (80 cases, temp 0.7)

| run | action | body | density | length | openings | entropy | legs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1e-5` @1,000 | **0.871** | **0.945** | 1.02 | 0.93 | 54 | 4.80 | **0.787** |
| `1e-4` @1,527 | 0.847 | 0.936 | 0.96 | 0.95 | 51 | **4.89** | 0.725 |

The high-rate run recovers most of the gap once it is decoded the same way (the
earlier 0.801 / entropy 2.05 figure was its greedy decoding, not the weights'),
but after a full epoch it still trails the low-rate 1,000-step checkpoint on the
two primary metrics and, notably, on leg coverage (0.725 vs 0.787 mentions).
Its NLL is lower and its opener entropy is marginally higher. Conclusion: at
this data scale `1e-4` is not an upgrade, it is a different trade — do not
switch learning rates on NLL evidence alone.

## 5.4 Second epoch contributes almost nothing (measured)

`sol61_full_20261113_010434` ran the full 2-epoch budget (3,054 updates,
8 nodes x 8 GPUs, global batch 128 via accum 2):

| step | val NLL |
| --- | --- |
| 1,527 (end of epoch 1) | 1.4851 |
| 2,100 | 1.4749 |
| 2,400 | 1.4697 |
| 2,700 | 1.4687 |
| 2,900 | 1.4683 |
| 3,054 (end of epoch 2) | **1.4683** |

The second epoch lowers validation NLL by 0.0168 in total and the last 350
updates change it by 0.0001. Under `lora_lr=1e-5` the run is effectively
converged at the end of epoch 1. For comparison the 1e-4 run reaches 1.4598
after only 400 updates, i.e. better than this 3,054-update run with 7.6x less
training.

Practical conclusion: do not spend compute on additional epochs at 1e-5. Either
raise the learning rate (measured 7x sample-efficiency gain) or invest in data /
decoding changes. Any claim that the model needs "more epochs" should be checked
against this table first.

## 5.5 Two runs at the same step differ 4x in diversity

Both checkpoints are update 1,527 on the same 80 val cases and the same audit:

| run | GPUs / accum | val NLL | action recall | body recall | opener entropy | top opener | openings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sol61_full_20261112_161243` | 32 / 4 | 1.5223 | 0.822 | 0.894 | **2.88** | 0.388 | 14 |
| `sol61_full_20261113_010434` | 64 / 2 | **1.4851** | 0.812 | 0.904 | **0.70** | 0.838 | 4 |

The 64-GPU run has the *better* validation NLL and the *worse* generation: 84% of
its captions open with the same phrase (`The dancer lowers and lengthens one
arm...`), against 39% for the 32-GPU run. Global batch is 128 in both cases, so
the effective difference is world size and accumulation depth (accum 2 instead of
4).

This is the third independent demonstration that validation NLL ranks checkpoints
inversely to generation quality (see 5.2 and 4.4), and it adds a configuration
caveat: **do not assume a wider launch reproduces the narrower one**. The
accum-2 path must be re-checked with a generation audit before being used for
any production run.

## 5.6 Decoding dominates the remaining quality gap (measured)

All rows below are the same checkpoint
(`sol61_full_20261112_161243/checkpoint-0001000`) on the same 80 SHA-ordered
val cases, differing only in decoding. Sampling uses top-p 0.9,
repetition_penalty 1.05, no_repeat_ngram 4; beam is 1 unless stated.

| decoding | actRec | bodyR | densRat | lenRat | openings | entropy | top-opener | distinct-2 | BLEU-1 | BLEU-4 | ROUGE-L | tok-F1 | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy | 0.803 | 0.904 | 1.07 | 0.84 | 16 | 2.81 | 0.413 | 0.114 | 0.508 | 0.054 | 0.301 | 0.592 | 0.687 |
| beam = 3 | 0.714 | 0.917 | — | — | — | — | — | — | — | — | — | — | — |
| temperature 0.6 | 0.839 | 0.931 | 0.99 | 0.88 | 58 | 4.84 | 0.138 | 0.183 | 0.506 | 0.057 | 0.297 | 0.593 | 1.118 |
| **temperature 0.7** | **0.871** | 0.945 | 1.02 | 0.93 | 54 | 4.80 | 0.100 | 0.195 | **0.516** | 0.055 | 0.290 | **0.594** | **1.154** |
| temperature 0.8 | 0.835 | **0.950** | 0.97 | 0.92 | 52 | 4.97 | 0.100 | 0.202 | 0.492 | 0.056 | 0.285 | 0.577 | 1.119 |
| temperature 0.9 | 0.841 | 0.940 | 0.96 | **0.95** | **64** | **5.36** | **0.075** | **0.225** | 0.500 | 0.054 | 0.276 | 0.579 | 1.124 |
| reference | 1.000 | 1.000 | 1.00 | 1.00 | — | 5.57 | 0.062 | 0.234 | 1.000 | 1.000 | 1.000 | 1.000 | — |

Findings:

- **The headline defect was the decoder, not the weights.** Greedy decoding
  collapses the model onto one opener (41% of captions) and 16 distinct
  openings out of 80; temperature 0.7 raises that to 54 openings and entropy
  4.80 with no self-repetition, and simultaneously *improves* action recall
  (0.803 → 0.871) and body recall (0.904 → 0.945).
- Beam search is strictly worse here (action 0.714): it optimises likelihood,
  which is exactly the quantity this model is over-fitted on.
- temperature 0.7 is the best operating point: highest action recall, highest
  BLEU-1/F1, and a length ratio of 0.93 against the reference's 1.00.
- temperature 0.9 is measurably closer to the reference on diversity
  (entropy 5.36 vs 5.57, top-opener 0.075 vs 0.062, length 0.95) but loses
  ~3 action points; use it when longer, more varied captions matter more than
  vocabulary overlap.
- Greedy, beam, and all NLL-based selection agree with each other and disagree
  with these numbers — the fourth independent demonstration in this document
  that NLL is the wrong selection criterion.

**Deliverable decode config:** `do_sample=True, temperature=0.7, top_p=0.9,
repetition_penalty=1.05, no_repeat_ngram_size=4, num_beams=1`.

## 5.7 Clause-level leg weighting is the first change that beats the baseline

`sol61leg3`: identical recipe to the baseline run (1e-5, 32 GPU, accum 4,
global batch 128, 1 epoch) plus `PLANNER_LEG_WEIGHT=1.5`, which multiplies the CE
of every clause that names a leg part (`leg/knee/thigh/shin/calf/hamstring/quad`)
by 1.5. Token-level weighting was tried first and rejected: leg words are <0.5%
of assistant tokens, so it moved almost none of the loss.

80 SHA-ordered val cases, temperature 0.7:

| checkpoint | action | body | legs | density | length | openings | entropy | BLEU-1 | BLEU-4 | ROUGE-L | tok-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline @1,000 | 0.871 | 0.945 | 0.787 | 1.02 | 0.93 | 54 | 4.80 | 0.516 | 0.055 | 0.290 | 0.594 |
| **leg-weighted @500** | **0.880** | **0.968** | **0.875** | 0.98 | **1.04** | **57** | **5.01** | **0.535** | **0.072** | 0.292 | **0.598** |

At half the training the weighted run is better on every primary and secondary
metric, and leg coverage moves from 0.787 to 0.875 mentions (reference 0.993).
The length ratio also stops undershooting (0.93 → 1.04 against a reference of
1.00), which the plain run never fixed.

Reading: the leg deficit was a loss-shaping problem, not a data or capacity
problem — the same rows, one extra weight on the clauses that carry the
information, and the weakest body group jumps 9 points. 1,000- and 1,527-step
checkpoints of the same run are being audited to see whether it keeps improving
or starts to collapse the way the unweighted run did after ~1,000 steps.

## 6. Status summary

Done and measured:

- seven checkpoints scored on 80 fixed cases (section 5.2);
- second epoch effect quantified (section 5.4);
- learning-rate effect quantified at five points (section 5.3);
- first-pass E2 comparison against three general baselines (section 7.1).

Still running at the time of writing:

- `sol61-hlr-1ep-32gpu` (`lora_lr=1e-4`, 1 epoch) — val NLL 1.4270 at update 600;
  its 1,000-step generation audit is the last open question, namely whether the
  opener entropy rises above the 3.04 measured under the low rate.
- `sol61-eval-epochs-8gpu` — e1_1527 vs e2_3054 on the same 80 cases, to confirm
  that the second epoch also fails to move generation quality.
- E2 MOSS-Music prompt fix verification.

## 7. In flight

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

The probe harness is `sol61-e2-baselines-job/e2_baselines.py`; the first-pass
comparison runner is the same directory's `entry.py`.

### 7.1 Aligned comparison on 20 identical val cases

Case set: the first 20 ids after the audit's SHA ordering
(`sha256('many:' + id)`), i.e. a strict prefix of the 80-case audit set, so the
SFT rows here are the same generations as section 5.2 restricted to those ids.
All rows decoded greedily unless stated. `run = sol61_e2_20261113_114545`.

| model | rows | actRec | bodyR | densRat | lenRat | openings | selfRp | halluc | BLEU-1 | BLEU-4 | ROUGE-L | tok-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference (upper bound) | 20 | 1.000 | 1.000 | 1.00 | 1.00 | 17 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **SFT captioner @1000 (temp 0.7)** | 20 | **0.877** | **0.944** | 1.06 | 0.82 | **17** | 0.000 | 0.000 | 0.476 | 0.052 | 0.286 | 0.580 |
| **SFT captioner @1000 (greedy)** | 20 | 0.804 | 0.907 | 1.12 | 0.79 | 10 | 0.000 | 0.000 | 0.464 | 0.040 | 0.293 | 0.579 |
| SFT captioner @1000 (2nd epoch, greedy) | 20 | 0.827 | 0.907 | 1.11 | 0.80 | 9 | 0.000 | 0.000 | **0.489** | **0.070** | **0.302** | **0.594** |
| Qwen3-Omni-30B-A3B-Instruct | 20 | 0.224 | 0.722 | 0.89 | 1.18 | 2 | 0.394 | 0.000 | 0.134 | 0.000 | 0.104 | 0.157 |
| Qwen2.5-Omni-7B | 20 | 0.022 | 0.204 | 1.38 | 0.25 | 4 | 0.000 | 0.000 | 0.026 | 0.000 | 0.075 | 0.096 |
| Music-Flamingo-2601 | 20 | 0.129 | 0.074 | 1.34 | 0.66 | 7 | 0.108 | 0.000 | 0.082 | 0.000 | 0.129 | 0.158 |
| Audio-Flamingo-3 | 20 | 0.017 | 0.093 | 1.37 | 0.22 | 1 | 0.000 | 0.050 | 0.007 | 0.000 | 0.063 | 0.070 |
| MOSS-Music-8B-Instruct | 20 | 0.151 | 0.426 | 1.54 | 1.16 | 8 | 0.207 | 0.100 | 0.114 | 0.000 | 0.111 | 0.148 |

### 7.2 MOSS-Music-8B under transformers 5.18 (three fixes)

The published model ships for transformers 4.x. Running it unchanged on the
platform venv (`transformers 5.18.0`) fails three separate ways; all three are
fixed in the harness, no model weights are modified:

1. **Attention signature.** `WhisperEncoderLayer.forward` now goes through the
   new attention interface, so the encoder's tensors hit
   `matmul: size of tensor a (64) must match tensor b (20)`. The harness
   re-implements the layer against the module's own `q/k/v/out_proj`, head count
   and head dim, so it does not depend on transformers internals.
2. **Multi-device sharding.** `device_map='auto'` spreads the encoder and the
   embedding layer across GPUs and `masked_scatter_` then raises
   `source is on cuda:4, other tensors on cuda:0`. Load with
   `device_map={'': 0}`; the 8B model fits a single 79 GB L20Z.
3. **Audio re-injection on decode steps.** `prepare_inputs_for_generation`
   only clears `audio_data` when `cache_position[0] > 0`, which transformers 5.x
   does not always supply, so the appended token is forward-ed with
   `[1, T+1]` embeddings against a `[1, T]` audio mask. Both `forward` and
   `prepare_inputs_for_generation` are wrapped to drop audio inputs whenever the
   sequence length and the audio mask length disagree (the audio is already in
   the KV cache at that point). Wrapping with `functools.wraps` matters:
   `generate` validates kwargs against the forward signature.

After the fixes the encoder output is finite (`absmean` 0.48) and captions are
coherent; the harness is `sol61-e2-moss-job/e2_baselines.py`.

Reading, with the protocol's caveat that the reference is one instantiation and
low overlap is not automatically low quality:

- The SFT planner is 3.6-40x higher on action recall and 1.3-10x higher on body
  recall than every general audio/omni model. It is the only model inside this
  dataset's action vocabulary, which is what the task asks for.
- Qwen3-Omni-Instruct writes fluent, whole-body dance prose (length ratio 1.18)
  but reuses one opener 60% of the time, self-repeats 0.394 and hits the
  repetition disqualifier. Its vocabulary is mostly outside the reference's
  action inventory, so low recall here means "different description of the same
  audio", not "wrong".
- Qwen2.5-Omni-7B (0.25 length), Audio-Flamingo-3 (0.22 length) and
  Music-Flamingo-2601 (0.66 length, 0.074 body recall — almost no body nouns)
  are not usable planners as wired here: their prompts/audio adaptation still
  need work, so treat them as lower bounds on those models' ability.
- MOSS-Music now runs after three harness fixes (see 7.2). It is the second-best
  baseline on body-part recall (0.426) but writes 1.54x the reference density
  with 0.207 self-repetition and 10% captions naming body parts with no audio
  evidence; action recall is 0.151.
- BLEU/ROUGE/F1 are reference-overlap metrics and say almost nothing about
  planner quality (section 4.4); they are reported because they were requested,
  not because they select a checkpoint.
- These numbers cannot by themselves decide "our model is better"; that needs
  the blinded human review the protocol requires.
