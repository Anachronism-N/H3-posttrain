# HereWeChoreography manuscript snapshot

The current paper is `main.tex` / `HereWeChoreography.pdf` (eight content pages
plus references); supplementary material is `appendix.tex` / `appendix.pdf`.
`main.pdf` is an identical alias so cross-document PDF links remain valid.
The existing separation between paper and supplement is retained.

## Current editable figures

Use [the revised figure package](figures/revision_20261117_v3/README.md).
It includes both overview alternatives, F2a/F2b, F3, F4 and two supplementary
diagrams as SVG masters and an eight-slide native editable PowerPoint.
Paper PDFs and PNGs come from the same native slide objects; the PNG crop
checks are in `paper_ppt_consistency.json`. Only real source-video frames are raster objects. Older `figures/editable` and `figures/revision_20261117` files are historical versions.

Figure 1 defaults to `figures/revision_20261117_v3/paper/overview_method.pdf`.
Uncomment `\def\OverviewDataBenchmark{1}` in `main.tex` for the benchmark
alternative. The two figures are alternatives for one overview slot.

## Evaluation names and unchanged results

Q1--Q3 denote local text quality, M1--M3 local music compatibility, and
G1--G3 global composition. Frozen records retain their D/L field names;
existing D1--D3 scores are displayed as Q1--Q3 without changing values.
M3 is the newly defined affective compatibility criterion: old event-response
D6 scores cannot be reused. Music-aware ratings remain unreported pending
calibration and execution. The mapping is in
`results/paper_dimension_map_20261117.json`.

The style table uses the 822-clip population. Detailed scope and provenance
remain in `results/`. No new training, benchmark inference, human labels or
quality scores were produced by this figure revision.

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

## Vivid foreground revision

The latest figures separate pale backgrounds from vivid foreground graphics.
The case, evaluation figure and both overview layouts have been redrawn.
This revision updates figure assets and captions only; pre-existing sections,
tables and experimental records in this branch are preserved. Concurrent
manuscript/result edits in the local workspace are not merged by this task.
`snapshot_build_audit.json` is the current compilation receipt. Earlier
build receipts under `results/` are retained as historical records.
