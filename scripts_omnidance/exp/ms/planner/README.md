# M2D audio-to-caption planner + H3 video pipeline

Audio (5.0625 s, 24 kHz mono) -> Qwen3-Omni-30B-A3B-Captioner planner -> timed
motion caption -> MiniMax H3 Ref2VA video.

## Layout

| Path | Contents |
| --- | --- |
| `trainer/` | Planner training/inference code: `music_planner_train.py`, adapters, JEV genre head, window-count auxiliary head (`music_planner_count.py`), schema/contract validation, MusicFM fusion, H3 video helpers. |
| `data/` | Dataset builders for the delivered splits (`build_full_v4.py`), dense/mid/no-hard/CoT subsets, and the fixed benchmark case sets. |
| `eval/` | Evaluation utilities: teacher-forced NLL, generation on fixed cases, JEV accuracy, ROUGE-L, sanitization, aggregation, RSI best-of-N, counterfactual audio swap. |
| `jobs/` | Nebula entry scripts and cluster/deploy configs per experiment (SFT, JEV v2/v3, full-v4, count-aux, dense/mid/no-hard/mix/CoT, 30B labeling, MACE/OmniDance/Wan-Dancer baselines, H3 render batches). |
| `render/` | H3 prompt-package builders and contact/compare sheet generators. |
| `docs/` | Setup notes, H3-JEV notes and `results_summary.json` (consolidated metrics). |

## Data contract

- Audio: 24 kHz mono, exactly 121,500 frames (5.0625 s), SHA256 pinned per row.
- Captions: `[start-end s] ...` windows covering the full clip, no `Dance style:` header
  unless the JEV variant is explicitly requested.
- Splits isolate both `source_group_id` and `audio_sha256`; no cross-split leakage.
- Delivered corpus: full-v4 = 196,097 train / 10,990 val / 11,119 test.

## Training

```bash
python -m scripts_omnidance.exp.ms.music_planner_train --family omni_captioner \
  --model <local Qwen3-Omni-Captioner> --train train.jsonl --val val.jsonl \
  --caption-contract caption_contract_nostyle.json --epochs 1 --batch 32 --micro 1 --accum 4 \
  --rank 32 --alpha 64 --audio-last-n 4 --lora-lr 1e-5 --projector-lr 5e-5 \
  --backend deepspeed --no-enable-jev
```

Auxiliary heads (all on the last native audio-prefix token, FP32):

- JEV genre classifier: `--genre-classes classes.json --train-genre-labels ... --enable-jev --genre-weight 0.1`
- Window-count regression (anti under-generation): `--count-weight 0.05 --count-head-dim 128`
- Exported alongside the checkpoint under `<checkpoint>/jev` and `<checkpoint>/count`.

## Evaluation

`eval/eval_nll.py` (teacher-forced NLL), `eval/eval_many.py` (greedy with
`repetition_penalty=1.05`, `no_repeat_ngram_size=4`), `eval/score_many.py`
(ROUGE-L), `eval/eval_jev_split.py` (top-1/top-3), `eval/counterfactual.py`
(audio-swap sensitivity). Consolidated numbers live in `docs/results_summary.json`.

## Levers under test (2026-11-06)

Cross-run audit of 20+ checkpoints (`eval/audit_generations.py`, `eval/collect_results.py`)
shows every checkpoint fails the same gate: **1.45-1.54 clauses per window against
a 1.92 reference**. Window count already matches (KL 0.04-0.07). Token-level CE does
not reward writing more distinct motion inside a window, so four levers are running:

| lever | implementation | status |
| --- | --- | --- |
| window-count stratified data | `data/build_strat.sh` -> `strat-v1` (117,280 rows) | baseline finished, val NLL **0.3033** (best so far) |
| token density weight | `PLANNER_TOKEN_WEIGHT` in `trainer/music_planner_token_loss.py` | weights 1.5 / 2.0 / 3.0 training |
| per-window supervision | `perwin-v1`: 293,045 samples of the form `Window k/n: [a-b s] ...` | training |
| density-aware best-of-N | `eval/eval_bestofn.py` | running |
| genre head (JEV) | stratified data + group-isolated labels | training |

`eval/audit_generations.py` prints a pass/fail verdict against
`docs/M2D_PLANNER_SELECTION_PROTOCOL.md`; `tools/check_entry.py` statically verifies
Nebula entry return contracts before submission.

## Selected results

| Run | Data | Metric |
| --- | --- | --- |
| SFT 82k | 82k | test NLL 0.3375 |
| SFT 198k | 198k | test NLL 0.3311 |
| Dense-SFT | 10k longest | density 2.73→7.87 / frame, length ratio 1.79, ROUGE-L 0.369 |
| Mix 50:50 | 20k | ROUGE-L 0.383, length ratio 0.674 |
| CoT-2 | 20k | ROUGE-L 0.431 vs baseline 0.444 |
| JEV v3 | 10.5k (1,500/class) | genre top-1 48.1%, top-3 88.6% |

Generation remains under-length (length ratio ~0.58) with correct timestamp
syntax and audio conditioning (audio-swap caption similarity 0.207); the
window-count head plus coverage-constrained decoding targets this gap.
