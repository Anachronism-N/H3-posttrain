# V3: vivid foregrounds, pale backgrounds

This is the current figure revision. The earlier `revision_20261117` directory
is retained for comparison. Four figures have new layouts: the choreography
case, the evaluation protocol, the method overview, and the benchmark overview.
F2 and the two supplementary diagrams retain their structure with revised colors.

## Design

Background panels use near-white colors (`#FAFCFD`, `#FAFCFB`, `#FBFAFE`,
`#FFFCF7`). Foregrounds use teal `#008F95`, blue `#2474B5`, purple `#7852A9`,
orange `#E69A28`, and coral `#D45D64`. Text uses Times New Roman and dark ink.
The foreground and background palettes are controlled independently.

- **Method overview:** trained Qwen3-Omni + LoRA, shared musical evidence,
  a central composer/controller/tool loop, persistent memory, and a clearly
  marked keep/edit/keep example. The task brief sits inside the harness.
  Hierarchical output is evaluated outside generation.
- **Benchmark overview:** genuine source frames, existing 2D rule-event
  refinement, paired resources, local and whole-song tasks, and our compact
  baseline. Music bypasses the text refiner. NLF is a separate extension.
  Reference agreement is a diagnostic, not an input to generation.
- **Case:** six uniformly spaced source frames, measured audio, predicted
  bars, and Hands/Pose/Legs instruction fields. Colors link the fields to
  authored prose. A shared elbow-state band makes continuity explicit.
- **Evaluation:** one aligned unit is enlarged for Q/M criteria, while G
  inspects the whole song. Small diagrams explain body/space, local timing,
  rhythm, dynamics, affect and motifs. Five anchors and the evidence sequence
  make the rating requirements visible.

## Editable files and consistency

The eight SVG masters and `powerpoint/HereWeChoreography_figures_revision_v3.pptx`
are editable. The deck contains 594 native text objects, 2361 native shapes
and 9 real-frame image objects (six unique source frames). No diagram is
wrapped as a full-slide bitmap. Statistical bars are editable shapes rather
than spreadsheet-linked charts.

`paper/` contains the active paper PDF/PNG exports. Every standalone preview
is copied from the same native slide render. `paper_ppt_consistency.json`
records exact PNG cropping and same-slide PDF export for all eight figures.
Package, layout and import validation are recorded in `powerpoint/validation.json`
in the GitHub snapshot. This does not claim a desktop PowerPoint opening test.

## Evidence boundaries

No training, inference, new human labels or experimental scores were added.
The six data-audit panels retain the frozen data and denominators documented
in `source/annotation_audit_definitions.md`. Automatic flags are not human
correctness. Source rule windows are not musical bars.

Q1--Q3, M1--M3 and G1--G3 keep the existing display definitions. M3 is affective
compatibility and has no local-event gate. Old event-response D6 scores cannot
be reused for M3. Calibration remains pending.

The case photos and waveform are genuine. Its prose and field bands are
illustrative instructions, not model predictions, measured action intervals,
motion trajectories or proof of synchronization. The current label route is
2D rule-event rewriting; the 3D extension is not attributed as its production history.

## Rebuild

The `source/build_*.py` files rebuild the corresponding SVGs. After editing:

```sh
python source/svg_to_native_scene.py
node source/build_native_pptx.mjs
python source/export_paper_from_ppt_master.py
node source/finalize_native_pptx.mjs
```

Python requires Pillow, lxml, fontTools and PyMuPDF. Node uses the bundled
Artifact Tool; set `ARTIFACT_NODE_MODULES` on another installation. Set
`FIGURE_FONT_DIR` to a directory containing `times.ttf`, `timesbd.ttf`,
`timesi.ttf`, and `timesbi.ttf`. The finalizer also has local skill/runtime
paths that need adjustment on another machine. `pdf_standard_fonts.cjs`
adapts a missing import map without changing the installed runtime.

The finalizer refuses to overwrite an existing delivery; use a new filename
for a subsequent revision. Raw MP4/WAV are needed to regenerate the source
case waveform and stills; the delivered SVG already embeds the stills and
can be converted to PPT without raw media or model calls.

`render_svg_previews.mjs` produces drafts under `.draft/`. These drafts are
excluded from delivery. Use the native slide exports for final comparison.
If editing the PPT directly, export the paper figures from that edited deck.
Running the old SVG pipeline would overwrite those PowerPoint-only changes.
