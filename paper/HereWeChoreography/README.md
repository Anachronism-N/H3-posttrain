# HereWeChoreography manuscript snapshot

This directory contains the active CVPR manuscript, figures, source tables, and
frozen result receipts. `HereWeChoreography.pdf` is the compiled preview.
The experiment section reports the available D1-D3 evidence; D4-D6 remain
unreported until their evaluation is complete. The style comparison uses only
the 822-clip population. Detailed result scopes and limitations are in `results/`.

The proposed local D4-D6 redesign is documented in
[the style/affect rubric proposal](../../notes/62_local_music_rubric_style_affect_proposal_20261117_zh.md).
It requires calibration and has not been applied to the manuscript's rubric,
evaluation code, or reported scores.

## Overview variants and expert supervision

The default Figure 1 uses `figures/overview_method_v3.png`, emphasizing the
trained expert, JEV, controller/tool feedback, persistent state and targeted
revision. The alternative `figures/overview_data_benchmark_v2.png` emphasizes
paired supervision and the proposed benchmark. Uncomment the
`\def\OverviewDataBenchmark{1}` line in `main.tex` to select the latter.
The two images are alternatives for one overview slot. Source prompts and a
semantic review receipt are in `figures/source/overview_*`.

The expert's training description now uses joint-aware clause reweighting,
instantiated for lower-limb clauses at weight 1.5. Public implementation and
checkpoint-report evidence are frozen in `results/joint_clause_weighting/`;
the objective uses the count of valid assistant tokens as its denominator.
No new training, benchmark inference or experimental result was added.

## Build

Run from this directory with a TeX distribution containing the CVPR dependencies:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The checked-in figures and tables are sufficient to compile the complete paper.
Compiler caches and temporary logs are excluded. The enclosing branch preserves
the bar-style implementation described by the manuscript; the snapshot manifest
records all copied files and hashes.
The directory's Git attributes preserve exact file bytes for those hash checks.

## Figure and table reproduction

- `python results/build_d123_refinement_table.py` recomputes the versioned
  D1-D3 means and valid counts from the released per-case judgments.
- `python results/build_style_prior_822.py` rebuilds the active style table from
  frozen aggregates and audit assertions, without model inference.
- `python figures/source/plot_annotation_distributions_v3.py` rebuilds the
  dataset statistics from `dataset_distribution_audit.json` (NumPy/Matplotlib).
- The qualitative figure's active plotting source and unchanged source stills
  are under `figures/source/`. Its text is an author-composed illustration;
  timestamps and audio derive from actual paired media. It is not a scored model
  prediction or a demonstration of verified audio-motion synchronization.
- Raw MP4/WAV files are not redistributed in this snapshot. Full regeneration
  of the qualitative figure requires restoring case `wJPptuhclZmKAEWs` under
  `figures/source/case_example/`. The paired-data audit identifies the public
  dataset `avagelion/dance-data`, source revisions, media hashes, and bounded tar
  member locations. `extract_source_video_range.py wJPptuhclZmKAEWs 48 4024`
  documents the range-download route; verify both restored media hashes before
  running the plotting source. No GPU or new model inference is required.
- Conceptual GPT Image prompts and provenance are retained for the active
  generated figures. Source motion-description components remain attributed.

This snapshot does not run training or benchmark inference during export.
