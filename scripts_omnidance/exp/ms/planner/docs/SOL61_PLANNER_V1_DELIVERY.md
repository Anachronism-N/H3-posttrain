# sol61 dance-caption planner (v1)

LoRA adapter that turns a 5 s music window into one sol61-style dance caption.
All numbers below were measured on the **Xingyun (Nebula) L20Z cluster**
(79 GB/GPU, `future_hz_qwenvideo`).

## Files

| path | content |
| --- | --- |
| `adapter/` | PEFT LoRA adapter + planner metadata + caption contract + tokenizer files |
| `../sol61_planner_v1/README.md` | this file |

Source checkpoint:
`runs/sol61_full_20261112_161243/attempts/sol61_full_20261112_161243-attempt161716/checkpoint-0001000`.

## Base model

`Qwen3-Omni-30B-A3B-Captioner-full` (Thinker only; audio tower + projector + LLM,
no visual tower). Adapter targets the last 4 audio-tower layers
(`audio_tower.layers.26-29`) plus the projector, rank 32 / alpha 64.

## Input contract

- audio: mono, 16 kHz, 1.0–10.0 s per window; training used 5.0625 s windows
- prompt: one user turn containing **only** the audio, no text instruction:

```python
conversation = [{'role': 'user', 'content': [{'type': 'audio', 'audio': wav_path}]}]
prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
```

- output: free text, one caption, no timestamps, no style labels

## Decoding (validated configuration)

```python
generate(
    do_sample=True, temperature=0.7, top_p=0.9,
    repetition_penalty=1.05, no_repeat_ngram_size=4, num_beams=1,
    max_new_tokens=2048,
)
```

Greedy decoding is **not** recommended: it collapses 41% of captions onto one
opening phrase (16 distinct openings in 80 samples vs 54 at temperature 0.7).

## Measured quality

80 SHA-ordered val cases, audited with `experiments/m2d_planner/eval/audit_sol61.py`:

| decoding | action recall | body recall | length ratio | distinct openings | opener entropy | self-repetition |
| --- | --- | --- | --- | --- | --- | --- |
| greedy | 0.803 | 0.904 | 0.84 | 16 | 2.81 | 0.000 |
| **temp 0.7 (recommended)** | **0.871** | **0.945** | 0.93 | 54 | 4.80 | 0.000 |
| temp 0.9 (max diversity) | 0.841 | 0.940 | 0.95 | 64 | 5.36 | 0.000 |
| reference captions | 1.000 | 1.000 | 1.00 | — | 5.57 | 0.000 |

Against every general audio/omni model tested on the same 20 cases, this
planner is 3.6–40× higher on action recall and 1.3–10× higher on body-part
recall (see `SOL61_TRAINING_DIAGNOSIS.md` §7.1).

## Verified usage

`planner/eval_harness/generate_caption.py` was run on the platform (L20Z) against
the published adapter and the base model, on three held-out windows:

```
$ python generate_caption.py --model .../Qwen3-Omni-30B-A3B-Captioner-full \
      --adapter .../deliverables/sol61_planner_v1/adapter --audio window.wav
"As one hand moves left, the dancer folds both arms and lifts that hand while the
 opposite knee straightens. The opposite arm extends as its knee bends, then both
 hands rise before that arm curls right and the other drops into a tighter fold."
```

Two details matter for reproducing this:

- the checkpoint's `generation_config` does not carry the chat terminator, so
  `eos_token_id` must be set explicitly (`<|im_end|>`); without it the model keeps
  writing after the caption and degenerates into symbol spam;
- the window must be exactly 5.0625 s — the loader rejects other lengths rather
  than padding.

Three sample windows are cached at `samples/` next to this README so the check
runs in minutes instead of re-streaming the 53 GB data tar.

## Known limits

- Leg/knee/thigh coverage is below the reference; a class-weighted CE variant
  (`PLANNER_LEG_WEIGHT`) is being trained.
- Captions are ~7% shorter than the reference at temperature 0.7 (length ratio
  0.93); temperature 0.9 closes that gap at the cost of ~3 action points.
- Trained on 195k sol61 rows (one epoch, 1,527 updates). More steps past ~1,000
  reduce opener diversity; validation NLL is not a usable selection signal.
- Text-overlap metrics (BLEU/ROUGE/F1) are reported for completeness only and
  must not be used to select checkpoints.
