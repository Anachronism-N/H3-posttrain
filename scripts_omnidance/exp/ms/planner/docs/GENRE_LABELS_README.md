# Dance genre labels (Qwen3-VL-30B-A3B-Instruct)

Dance-style labels for the `qwen35_captions` split of
[avagelion/dance-data](https://huggingface.co/datasets/avagelion/dance-data).
Produced with Qwen3-VL-30B-A3B-Instruct watching 16 evenly spaced frames per clip.

## Files

| file | rows | notes |
| --- | --- | --- |
| `train_genres.jsonl` | 198,315 | one row per caption id in `train_captions.jsonl` |
| `summary.json` | — | genre / confidence distributions, model and prompt versions |

`valtest_genres.jsonl` is not produced yet: the 22,352 validation/test captions
have no labels.

## Row schema

```json
{"id": "<caption id>", "genre": "k-pop", "confidence": "high",
 "secondary": "hip-hop", "evidence": "one sentence citing observed movement",
 "parse_ok": true, "raw": "<model output>", "model": "Qwen3-VL-30B-A3B-Instruct",
 "prompt_version": "vlm-genre-22-v1", "video_bytes": 1234567}
```

`genre` is one of 22 fixed classes (`chinese classical`, `chinese folk`, `ballet`,
`contemporary`, `lyrical`, `modern jazz`, `jazz funk / heels`, `k-pop`, `hip-hop`,
`popping`, `locking`, `breaking`, `house`, `krump`, `waacking`, `street jazz`,
`latin`, `belly dance`, `indian / bollywood`, `other ethnic / folk`, `pop / casual`,
`freestyle / other`).

## Distribution

| genre | rows |
| --- | --- |
| pop / casual | 90,054 |
| k-pop | 43,016 |
| hip-hop | 29,645 |
| chinese classical | 17,501 |
| chinese folk | 6,022 |
| jazz funk / heels | 4,118 |
| contemporary | 3,961 |
| ballet | 3,701 |
| remaining 14 classes | < 3,000 each |

Confidence: 139,432 high, 58,883 medium. All rows parsed to valid JSON
(`parse_ok = true`).

## Usage notes

- These are **visual** labels from a VLM watching frames, not from music. They are
  intended as an auxiliary target for a genre head trained on the audio prefix.
- The distribution is highly imbalanced; downstream training should class-balance
  or resample, otherwise `pop / casual` and `k-pop` dominate.
- Training sets built from these labels use group-isolated splits so that no
  `source_group_id` appears in both train and validation.
