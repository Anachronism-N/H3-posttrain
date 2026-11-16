# Manuscript and qualitative-example revision

The paper retains the local/global decomposition: source-linked supervision
supports a local movement expert; a stateful harness organizes the whole song;
evaluation separates reference agreement, versioned ratings and musical response.

## Final editorial decisions

- The active qualitative figure is v4, with six original video frames and a
  measured audio waveform. It shows the initial leg pose and final knee lift.
  Two internal action phases share one two-bar unit; the ending is a fragment.
  Phrase-level colors distinguish body, action, spatial range and coordination.
  The hand-off specifies written state, not a measured pose trajectory.
- D1-D3 are transcribed from the public refine-v1 release, with exact per-case
  recomputation. r4 and v2.1 appear separately on the same 2,000 outputs.
  D4-D6 remain unreported. Refinement is identified as an extra editor step.
- In the manuscript, the method is named **Ours** and score cells contain only
  means. Valid counts and the complete inference configuration are concentrated
  in the appendix. The automatic-rating table remains distinguished from the
  reference-agreement table's original SFT output stage.
- Missing assessor metadata and anchor definitions prevent interpreting source
  rating IDs as a validated application of the manuscript's proposed rubric.
  No baseline comparison, human agreement or musical superiority is inferred.
- The style result uses only the 822-clip population, all seven zero-shot
  models and all three supervised objectives.
- E2, versioned ratings, style and preliminary whole-song results are main-text
  tables. Older reliability and exploratory tables remain supplementary.
- Repeated historical diagnostics are shortened in the main text, retaining
  their full numbers and sensitivity analysis in the appendix.

## Claim-evidence map

| Claim | Evidence | Status |
| --- | --- | --- |
| Dedicated SFT has higher reference agreement | Frozen E2 2,000 aggregate | Supported within that panel; training provenance still needed |
| Supervised style heads specialize against video pseudo-labels | Matched 822-clip comparison | Supported for this scope; downstream benefit untested |
| Refine-v1 has published ratings | 2,000 exact output/hash-matched records | Supported as versioned ratings; anchors/assessor metadata absent |
| Higher v2.1 D2 means the model improved | Same outputs, changed rating version | Unsupported; explicitly described as protocol sensitivity |
| Latest harness outperforms other controllers | Historical unequal-evidence comparisons | Not established; controlled latest results pending |
| The example demonstrates measured good musical choreography | Real media with author-composed instructions | Not claimed; it explains text information and continuity |

## Five-dimension self-review

1. Contribution: four deliverables serve the local/global composition story.
2. Clarity: raw SFT, SFT plus editor, style heads and whole-song controllers
   have distinct names and evaluation scopes.
3. Evidence: no missing score is imputed; no post-hoc protocol change is
   presented as a method gain; no results are fabricated for D4-D6.
4. Evaluation: assessor/anchor release, listener agreement and controlled
   audio-conditioned comparisons remain material outstanding requirements.
5. Design: source-clock windows, persistent history and bounded patching
   remain explicit. Illustrative pose states do not imply physical validation.

No model inference or training was run during this manuscript revision.
