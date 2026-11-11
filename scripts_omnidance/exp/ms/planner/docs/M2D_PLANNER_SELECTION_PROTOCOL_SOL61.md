# Planner selection protocol (timestampless captions, sol61)

The original protocol in `M2D_PLANNER_SELECTION_PROTOCOL.md` scores window
structure (KL, window duration, clauses per window, timestamp overruns). Those
metrics require `[start-end s]` markers. The `sol61` captions have none: each
caption is one choreography paragraph. This document replaces the structural
metrics for that family; everything else (fixed evaluation rows, no test
leakage, decision by generation not NLL) still applies.

## Evaluation set

- 200 official `test` rows, fixed by hash order, never used for fitting.
- Audio verified with the same loader as training (24 kHz mono, 121,500 frames).

## Metrics

| metric | how | target |
| --- | --- | --- |
| **action recall** | `tags(gen, ACTION) ∩ tags(ref, ACTION)` / `|tags(ref)|` | higher; primary |
| **body recall** | same for the curated body-part vocabulary | higher |
| **detail density** | distinct action tags in the generation per second of clip | close to reference |
| **compression ratio** | generated chars / reference chars | 0.8–1.2 (no drift to one-liners or padding) |
| **distinct openings** | unique first-60-char prefixes over the eval set | >= 25 of 200 |
| **self-repetition** | 1 − unique/总 6-gram count inside one caption | <= 0.10 |
| **hallucinated limbs** | body parts absent from the reference but present in the output | lower |
| **ROUGE-L** | report only | never decisive alone |

Timestamps and window counts are not scored for this family.

## Disqualifying conditions

- self-repetition > 0.35 (the template-collapse failure mode seen in mid2).
- distinct openings < 15 of 200.
- compression ratio outside 0.5–1.6.

## Decision rule

1. Disqualify on the three conditions above.
2. Among survivors, pick the highest **action recall**.
3. Break ties with body recall, then the lower hallucinated-limb count, then
   higher ROUGE-L.
4. NLL is reported for training diagnostics only; it is never the selection key
   (full-v4 step 6129 had the best NLL of its family and the worst captions).

## Mandatory companions for the chosen checkpoint

- 30-case human/VLM spot check (`eval/eval_many.py` output + render via H3).
- Audio-swap counterfactual (`eval/counterfactual.py`): caption similarity
  should be clearly below the same-audio baseline, proving the caption follows
  the music rather than the prompt.

## Implementation status

`eval/audit_generations.py` computes the window-based metrics used for the
timestamped corpora. Timestampless (sol61) checkpoints are scored with
`eval/audit_sol61.py`, which implements every metric in this protocol:

```bash
python eval/audit_sol61.py runs/<run>/eval/gen2000_gen.jsonl runs/<run>/eval/gen4000_gen.jsonl
```

It prints one table row per checkpoint (action recall, body recall, density
ratio vs. reference, length ratio, distinct openings, self-repetition,
hallucinated-part rate, selection score) followed by the full JSON including
pass/fail verdict and disqualifiers. Input rows only need
`{"id", "text", "reference"}`.
