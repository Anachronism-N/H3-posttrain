# M2D planner training: complete record

Everything needed to reproduce, audit or continue the audio-to-caption planner work:
data lineage, every training run, evaluation protocol and measured results.

---

## 1. Data lineage

### 1.1 Upstream delivery

| item | location |
| --- | --- |
| Captions (Qwen-polished, 220,667) | `https://huggingface.co/datasets/avagelion/dance-data` → `qwen35_captions/` |
| Same data on OSS (SG) | `oss://futurelab-cognwm-sg/niezichao.nzc/m2d_output/` |
| Dance-genre labels (198,315) | `avagelion/dance-data` → `qwen35_captions/genres/` |

Delivered caption shape: `Dance style: <genre>` header followed by
`[start-end s] ...` windows. Mean 1,838 chars, 2.83 windows. Style header present
on 100% of rows; 56% of labels are `unknown`, 37% `contemporary`.

### 1.2 Training sets built from the delivery

All builders live in `data/`; every set validates 24 kHz mono 121,500-frame audio
and its SHA-256 before the row is admitted.

| set | builder | rows (train/val) | What it changes |
| --- | --- | --- | --- |
| `full-v3` | `build_full_v3.py` | 176,480 / 10,990 | first 82k+ delivery, style header stripped |
| `full-v4` | `build_full_v4.py` | 196,097 / 10,990 (test 11,119) | final 220,667 delivery, style header stripped, source-group + audio-SHA isolated splits, 2,459 geometry rejects dropped |
| `dense-v1` | `build_dense_subset.py` | 10,000 / 500 | longest 10k rows (mean 3,443 chars) |
| `mid-v1` | (mid builder) | 16,904 / 500 | 1.8–2.6k chars, 8–12 clauses/window |
| `mid2-v1` | inline builder | 49,431 / 500 | 1.2–2.8k chars, 2–4 windows, 8–20 clauses/window |
| `nohard-v1` | inline builder | 20,000 / 500 | excludes len>3000 & density>15 |
| `mix-v1` | inline builder | 20,000 / 500 | 50:50 dense + random |
| `cot-lite` / `cot-v2` | `build_cot_subset.py` | 5k / 20k | deterministic librosa analysis prefix |
| `strat-v1` | `build_strat.sh` | 117,280 / 800 | full-v4 stratified by window count (bins 1/2/3/4/5+) |
| `perwin-v1` | `build_perwindow.py` | 293,045 / 2,443 | each window becomes its own sample, `Window k/n: [a-b s] ...` |
| `strat-jev` | inline builder | 116,251 / 1,029 | strat-v1 + genre labels, val split by `source_group_id` (0 group overlap) |
| `vlm-jev-v2` | `build_vlm_jev.py` | 4,200 / 210 | 600/class balanced |
| `vlm-jev-v3` | `build_vlm_jev.py` | 10,500 / 210 | 1,500/class balanced |

Archives: `oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/data/<name>.tar`.

---

## 2. Training

Base model: Qwen3-Omni-30B-A3B-Captioner (Thinker only). LoRA r=32, α=64 on the
last 4 audio-tower layers plus the projector; BF16 base, FP32 trainables; DeepSpeed
ZeRO-3, micro 1, accum 4, global batch 32 on 8×L20Z; cosine schedule, warmup 20;
`loss-chunk-size 128`; max length 8192.

Every run below is a directory under `jobs/` with its `entry.py` (Nebula entry),
`mode.txt` and cluster spec. All entries pass `tools/check_entry.py`.

### 2.1 Main runs and outcomes

| run | data | epochs | extra objective | steps | best val NLL |
| --- | --- | --- | --- | --- | --- |
| sft (82k) | full-v3 early | 1 | – | ~2,000 | 0.3375 (test) |
| sft198k | full-v3 | 1 | – | 5,515 | 0.3254 |
| full-v4 SFT | full-v4 | 1 | – | 6,129 | 0.3223 |
| **strat** | strat-v1 | 1 | – | 3,665 | **0.3033** |
| stratinc | strat-v1 | 1 | 100-step saves | in progress | – |
| stratdense | strat-v1 | 1 | token weight 2.0 | in progress | – |
| stratdense3 | strat-v1 | 1 | token weight 3.0 | in progress | – |
| stratft | strat-v1 | 2 | token weight 1.5 | in progress | – |
| fullmild | full-v4 | 1 | token weight 1.3 | in progress | – |
| dens | full-v3 | 1 | token weight 1.3 | in progress | – |
| stratcount | strat-v1 | 1 | window-count head 0.05 | in progress | – |
| stratjev | strat-v1 + labels | 1 | JEV genre CE 0.1 | in progress | – |
| perwin | perwin-v1 | 1 | – | in progress | – |
| dense-sft | dense-v1 | 1 | – | 412 | 0.3168 (val) |
| mid-sft | mid-v1 | 1 | – | 529 | 0.3168 (val) |
| mid2count | mid2-v1 | 1 | count head 0.05 | 1,545 | 0.3090 |
| mid2jev | mid2-v1 | 1 | count + JEV | 1,530 | 0.3090 |
| struct | mid2-v1 | 1 | count + last-window end | 1,545 | 0.3093 |
| nohard | nohard-v1 | 1 | – | 625 | 0.3435 (val) |
| mix | mix-v1 | 1 | – | – | ~0.34 |
| cot / cot2 | cot-lite / cot-v2 | 1 | analysis prefix | – | ~0.34 |
| full4count | full-v4 | 1 | count head 0.05 | 6,129 | 0.3228 |

Key observation: **NLL ranks data sets, not generation quality.** full-v4 at step
6129 has the lowest NLL of its family but the worst captions; step 2000 is better.

### 2.2 Loss variants implemented

| variant | flag | module |
| --- | --- | --- |
| JEV genre classifier on the native audio prefix | `--genre-*`, `--enable-jev` | `trainer/music_planner_jev.py` |
| Window-count regression (`log1p` count + last-window end) | `--count-weight`, `--count-head-dim` | `trainer/music_planner_count.py` |
| Token density weighting of assistant CE | `PLANNER_TOKEN_WEIGHT` env | `trainer/music_planner_token_loss.py` |
| MusicFM encoder fusion | `--musicfm-cache`, `--fusion-*` | MOSS only, blocked for this family |

`tests/test_token_weight_loss.py` pins the weighting contract (weighted CE equals
unweighted CE times the weight, identical token counts, finite gradients).

---

## 3. Evaluation

Protocol: `docs/M2D_PLANNER_SELECTION_PROTOCOL.md`. Fixed 80 official test rows,
greedy decode with `repetition_penalty=1.05, no_repeat_ngram_size=4`.

| gate | threshold |
| --- | --- |
| window-count KL vs reference | ≤ 0.10 |
| clauses per window | ≥ 1.6 (reference 1.92) |
| mean window duration | ≤ 2.1 s |
| distinct openings | ≥ 20 of 80 |
| timestamp overruns | 0 |

`eval/audit_generations.py` computes all metrics plus a continuous **proximity**
score (mean of per-gate closeness) so partially passing checkpoints can be ranked.
`eval/collect_results.py` re-scores every evaluation run directly from OSS.

### 3.1 Measured standings (2026-11-06)

| checkpoint | KL | dur | clauses/win | len ratio | openers | recall | prox |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **sft198k_5515** | 0.068 | 1.75 | 1.48 | 0.58 | 28 | 0.677 | **0.985** |
| fullv4_4000 | 0.042 | 1.81 | 1.47 | 0.54 | 29 | 0.660 | 0.984 |
| fullv4_6129 | 0.095 | 1.90 | 1.54 | 0.52 | 36 | 0.652 | – |
| strat@3665 | 0.515 | 2.30 | **2.60** | 0.71 | 12 | 0.673 | 0.742 |
| mid2count | 0.644 | 2.52 | 3.39 | 0.98 | 8 | 0.712 | 0.678 |
| struct | 0.644 | 2.52 | 3.44 | 0.94 | 4 | 0.696 | 0.676 |

Contradiction resolution: the caption of every checkpoint before `strat` writes
1.45–1.54 clauses per window while the reference has 1.92. Window **count** already
matches (KL 0.04–0.07); only per-window content lags. `strat` data fixes density but
collapses window count and opener diversity, so it trades three gates for one.

### 3.2 Auxiliary results

| item | value |
| --- | --- |
| JEV v3 genre head (1500/class) | test top-1 48.1%, top-3 88.6% |
| JEV v2 (600/class) | test top-1 42.9%, top-3 82.9% |
| counterfactual audio swap | caption similarity 0.207 (model follows audio) |
| RSI-lite best-of-3 | +2.6% over greedy on 3 cases |
| timestamp syntax | monotonic, starts at 0, mean last end 4.87–5.07 s |

---

## 4. Operational lessons (encode these in any new job)

1. A Nebula task in `RUNNING` proves nothing; the platform restarts crashes. Grep
   logs for `Traceback`, `Unknown mode`, `trainer exited`, `Size does not match`.
2. `run_config.json` in the run directory is the authoritative "training started"
   signal; it records `token_weight`, `epochs`, `save_every_updates` and module SHAs.
3. Plain jobs upload at the end, so an empty run directory mid-training is normal.
   Jobs that need mid-run visibility must sync `out/` every few minutes.
4. Whole-file rewrites only: slice-and-replace patches once truncated
   `music_planner_token_loss.py` and silently broke six jobs.
5. `check_entry.py` before every submission (return contracts and `mode.txt`
   whitelist). It has caught three silent-failure configurations.
6. Verify the deployed code, not the local file: a best-of-N job burned 3h23m
   because the task had been submitted before the `ACTION.findall` fix landed.

---

## 5. Where things live

| artifact | path |
| --- | --- |
| Trainer, adapters, loss modules | `trainer/` |
| Data builders | `data/` |
| Evaluation scripts | `eval/` |
| Nebula job entries (31) | `jobs/` |
| Static checker | `tools/check_entry.py` |
| Unit tests | `tests/` |
| Consolidated metrics | `docs/results_summary.json` |
| Protocol / audit / handoff | `docs/M2D_PLANNER_*.md` |

---

## 6. sol61 timestampless run (2026-11-12/13)

### 6.1 Data

`sol61-v1`: 217,373 rows built by `data/build_sol61.py` from
`avagelion/dance-data` → `sol61_captions` (revision `1ca58b4`). Gate-false rows
(842) and rows without a verified audio match (2,452) are dropped; the
`Dance style:` header is stripped. Splits: train 195,367 / val 10,938 /
test 11,068, zero cross-split groups, zero id overlap. Mean 768 chars, no
timestamps. Contract `caption_contract_nostyle.json` declares
`caption_version=m2d-sol61-v1` and requires no windows.

### 6.2 Runs completed (1 epoch, 1,527 updates each)

Both used 4 nodes × 8 GPUs, global batch 128 (micro 1 × accum 4), LoRA r32/α64 on
the last four audio-tower layers plus projector, ZeRO-3, max length 8,192.

| run | token weight | val NLL 500 | 1000 | 1500 | 1527 |
| --- | --- | --- | --- | --- | --- |
| `sol61_full_20261112_102148` (hz) | 1.0 | 1.581 | 1.531 | 1.523 | **1.522** |
| `sol61trick_full_20261112_093656` (sg) | 1.3 | 2.056 | 1.993 | 1.981 | **1.981** |

Second batch of runs saved and validated every 100 updates:
`sol61_full_20261112_161243` (hz, 16 checkpoints) and
`sol61trick_full_20261112_161536` (sg, 16 checkpoints). Train loss by 200-update
block on the hz run: 2.145 → 1.678 → 1.610 → 1.581 → 1.562 → 1.552 → 1.549 →
1.547. The last four blocks each gain ~0.002, so one epoch is saturated but
validation NLL never rose: no overfitting, and the budget is too small to call
the model converged (1,527 optimizer updates for a 43M-parameter adapter).

### 6.3 First measured generation quality (hz 1527, 20 val rows)

`eval/audit_sol61.py`, 20 rows, greedy decoding, `--max-new-tokens 2048`:

| metric | value | gate | pass |
| --- | --- | --- | --- |
| action recall | 0.838 | ≥0.55 | yes |
| body-part recall | 0.944 | ≥0.75 | yes |
| density ratio | 1.081 | 0.8–1.2 | yes |
| self-repetition | 0.000 | ≤0.10 | yes |
| hallucinated parts | 0.000 | ≤0.05 | yes |
| distinct openings | 9 / 20 | scaled | yes |

Shape matches the references closely: 5.45 vs 5.50 sentences per caption, 13 of
20 captions with identical sentence counts, mean length ratio 0.865. The gap is
inside sentences: 131 vs 160 chars per sentence and 4.6/4.6 left/right mentions
against 5.9/5.7, i.e. the model emits the right action skeleton with ~20% less
directional and joint detail. Missing vocabulary concentrates in
`bend`, `reach`, `up`, `elbow`, `knee` while `raise`, `lower`, `down` appear
instead, so most of the recall gap is wording preference rather than missing
content. No template collapse and no invented body parts, which were the defects
that sank the earlier timestamped planner checkpoints.

The protocol's absolute opening threshold (25 distinct) applies to 200-row
samples; `audit_sol61.py` now scales it down with the row count so 20-row smoke
runs are not misjudged.

### 6.4 What degrades as training proceeds: leg coverage

The same 30 val rows were generated from the 500, 1000 and 1500 checkpoints:

| checkpoint | action recall | body-part recall | missing `legs` | extra `legs` |
| --- | --- | --- | --- | --- |
| 500 | 0.821 | 0.938 | 5 | 6 |
| 1000 | 0.783 | 0.889 | 9 | 6 |
| 1500 | 0.810 | 0.815 | 15 | 3 |

Action recall is flat within noise (30 rows is roughly +-0.03) but body-part
recall falls monotonically, and the entire drop is `legs`: references say "the
knee on that side softens" or "one knee bends and straightens", and the later
checkpoint stops mentioning knees at all. Arms/hands/torso/head/feet are stable.
So more updates concentrate the output on the high-frequency arm vocabulary and
drop the long tail, which is also why `bend`/`knee` dominate the missing-action
counts. Checkpoint choice must therefore depend on whether downstream review
cares about lower-body coverage, and 30 rows cannot resolve differences of this
size: paired comparison on 200 fixed cases is required.

### 6.5 Continuing training

A resume from `checkpoint-0001527` could not start: containers mounted the run
store over FUSE and `stat` on the deep checkpoint path failed for all 12 retries,
while the same directory is readable through a parent listing from the eval job.
The continuation therefore restarts from the base weights with `epochs=2`
(`sol61-2ep-64gpu`, 8 nodes × 8 GPUs, global batch still 128 with accum 2), saving
and validating every 100 updates to expose the overfitting point.
