# Writing and figure revision audit

This is an editorial audit, not part of the submitted paper. No model
training or benchmark inference was run during this revision.

## Guidance consulted

- User-supplied `CVPR2026-Introduction-逻辑拆解.pdf` and
  `CVPR2026-正文写作指导.pdf`: task-first introductions, explicit
  challenge-to-module correspondence, and contribution-to-evidence structure.
- [Research-Paper-Writing-Skills, pinned SKILL.md](https://github.com/Master-cai/Research-Paper-Writing-Skills/blob/77e7c2c1ba06f7d71844873147665437a03aac1b/research-paper-writing/SKILL.md),
  plus its Introduction, Method, Experiments and Paper Review references.
- [learning_research, pinned README](https://github.com/pengsida/learning_research/blob/1ee561e5019fcc7d102b9a647591372871e40204/README.md)
  and the two available research-guidance pages. This repository has no
  SKILL.md. Its external Notion paper template was unavailable and was
  not used as an inspected source.

## Reverse outline

1. Abstract: define the timed textual-score task; introduce four connected
   deliverables; explain local/global separation; report reference
   agreement; bound the current whole-song evidence.
2. Introduction: task and interface; three requirements (specific local
   content, long-form organization, non-unique valid outputs); mechanism;
   deliverables; existing results and pending validation.
3. Supervision: paired inventory; actual 2D rule-log rewrite lineage;
   geometry extension and evidence consolidation; descriptive statistics.
   The real-data example illustrates information content and attribution.
4. Method: local proposals and stateful harness; bar timeline; shared
   musical conditions; outline/history-conditioned composition; bounded
   revision. Each module specifies an operation tied to a task requirement.
5. Evaluation: reference agreement versus quality; movement content,
   musical response, whole-song composition; evidence contracts;
   applicable-case coverage and song-level aggregation.
6. Experiments: caption adaptation; supervised style specialization and
   population-dependent rankings; historical reliability/cost; preliminary
   text-judge composition evidence; bounded mechanism diagnostics.

## Major claim–evidence map

| Claim | Evidence | Status |
| --- | --- | --- |
| A source-linked annotation resource exists | Inventory/distribution audit, paired records, sentence-event metadata | Supported as a resource description; annotation quality remains uncalibrated |
| SFT improves reference agreement | Frozen 2,000 aggregate; same 1,861 valid BERTScore cases | Supported for the reported panel; unseen-data claims await training provenance |
| Local prediction is separated from stateful whole-song composition | Implemented interfaces, cached proposals, committed history, validators and bounded patches | Supported as implementation; latest server execution/quality remains pending |
| Additional iteration does not guarantee completion | Historical 24-song policy completion and dispatch counts | Supported within that historical protocol |
| Structured composition improves musical quality | Same-family acoustic-summary text judging; cross-family interval includes zero | Not established; no general quality superiority claim |
| The nine-dimensional audio/text rubric is a validated metric | Contract implementation and written anchors only | Needs listener agreement, audio coverage and perturbation validation |
| Evidence consolidation is a new complete motion-description framework | MotionScript-derived extraction plus added validation, overlap merge, alternation summarization and provenance | Unsupported as a wholesale-invention claim; source method remains cited |
| The real example is verified excellent choreography | Actual audio/caption/source links; no video or listening quality review | Not claimed; it illustrates informative language and coarse attribution |

## Five-dimension self-review

1. **Contribution.** The organizing idea is local/global decomposition
   for an inspectable whole-song score. Four deliverables are named;
   renaming a stage is not used as evidence of novelty. Harness superiority
   still requires the planned controlled experiment.
2. **Clarity.** The opening defines input/output; requirements map to
   components; motion primitives mean open-vocabulary proposals. The
   figure follows execution order, and extraction is distinguished from
   our Motion Evidence Consolidation.
3. **Experimental strength.** Existing numerical evidence supports
   annotation agreement and limited historical diagnostics. Negative
   results, same-family judge limitations, and failed runs remain visible.
4. **Evaluation completeness.** Independent audio-grounded and human
   assessment, controlled agent comparisons, and final training provenance
   remain missing. The draft does not silently replace these with surface
   measures, pseudo-label accuracy or an illustrative case.
5. **Method soundness.** Duration, state, budget and patch permissions
   remain explicit. Checks concern written plans; physical executability,
   anatomical laterality and exact action/music timing are not inferred.

## Figure revisions

- `annotation_pipeline_v2.png`: built-in GPT Image edit of the previous
  artwork. Removed global title/subtitle; consolidated the palette;
  renamed stage 4 Motion Evidence Consolidation; expanded validation,
  overlap deduplication, alternation summarization and source tracing.
  MotionScript remains attributed in the manuscript caption and text.
  The final prompt is saved beside the figure sources.
- `choreography_example.pdf` / `.svg` / `.png`: scientific vector drawing
  from a real verified WAV and exact 25-word dataset excerpt. Colors
  expose body/action/relation information. Source interval 0–3.38 s is
  coarse attribution, not action timing. The waveform was not generated
  or redrawn by an image model. No video frames, beat labels, quality
  scores or structured model-input claims were invented.
