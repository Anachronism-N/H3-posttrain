# HereWeChoreography manuscript snapshot

The current paper is `main.tex` / `HereWeChoreography.pdf` (eight content pages
plus references); supplementary material is `appendix.tex` / `appendix.pdf`.
`main.pdf` is an identical alias so cross-document PDF links remain valid.
The existing separation between paper and supplement is retained.

## Current editable figures

Use [the current figure package](figures/revision_20261117_v3/README.md).
It includes both overview alternatives, F2a/F2b, F3, F4 and two supplementary
diagrams as SVG masters and an eight-slide native editable PowerPoint.
Paper PDFs and PNGs come from the same native slide objects; the PNG crop
checks are in `paper_ppt_consistency.json`. Only six real video photographs
are raster objects. Older `figures/editable` files are historical versions.

Figure 1 defaults to `figures/revision_20261117_v3/paper/overview_method.pdf`.
Uncomment `\def\OverviewDataBenchmark{1}` in `main.tex` for the benchmark
alternative. The two figures are alternatives for one overview slot.

## ChoreoScribe and the updated result table

The trained music-to-text choreographer is now named **ChoreoScribe**.
The full framework remains HereWeChoreography. The model generates directly
from SFT, without an additional LLM editor, according to the author-confirmed
correction of historical export labels. Frozen source records are unchanged.

`tables/captioner_unified.tex` combines reference agreement, local text
quality Q1--Q3, and local music compatibility M1--M3 into one grouped main
table. Rebuild with `python results/build_unified_caption_table.py`.
The new music results come from commit `44d4290`, stored verbatim under
`results/music_compatibility_v3/`. They use the revised affective M3.
Old event-response D6 scores are not reused.

Direct SFT and deterministic music-rules-v5 have separate rows; the latter
does not use an LLM but changes the text. Its unmeasured columns remain
missing. The archived LLM-editing condition is excluded from the main table.
Groups retain separate cohorts/protocols and are not averaged into a total.
The receipt records common-valid denominators, half-up rounding, the invalid
judgment, raw-audio hashes, and the mismatch between declared and JSON-wrapper
protocol versions. Independent human calibration remains pending.

The 822-clip style experiment and whole-song results are preserved. No new
training or model inference was run. The snapshot also retains the current
completed V3 figure exports referenced by the local manuscript; their
editable master package is under `figures/revision_20261117_v3/`.

## Build

Run from this directory with a TeX distribution containing CVPR dependencies:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build appendix.tex
```

`latexmkrc` prepares cross-document labels in `ref-build/`. Keep both source
documents together. Detailed rubric anchors are in the supplement; primary
experiment tables and figures remain in the main paper. The internal strict
review is in `review/strict_review_20261117_zh.md`.

## Reproduction and provenance

- `python results/build_d123_refinement_table.py` recomputes legacy D1--D3
  means from released judgments and emits the Q/M display table.
- `python results/build_style_prior_822.py` rebuilds the style table.
- Current figure rebuild and font/runtime instructions are in the figure
  package README. Statistical panels use frozen source audit values.
- The case uses unchanged, uniformly sampled real frames and a measured
  waveform. Its instructions are author-composed illustrations, not scored
  model outputs or evidence of measured synchronization.
- Current labels follow 2D rule-event text refinement. The NLF 3D geometry
  extension is kept separate from current experimental-label provenance.
- Raw MP4/WAV files are not redistributed. Full case regeneration requires
  restoring the media identified in the original source-case audit; the
  included six stills suffice for rendering the delivered SVG and PPT.
- Prior GPT Image concept prompts and original frozen receipts remain in
  historical source directories. Final active diagrams use editable vectors.

The snapshot manifest records copied file hashes. The figure validation receipt
checks package structure and geometry; it does not claim testing in Microsoft
PowerPoint or semantic validation of the proposed rubric.
