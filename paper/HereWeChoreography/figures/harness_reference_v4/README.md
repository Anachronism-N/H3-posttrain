# Harness overview from the supplied reference

This revision emphasizes the learned local expert and the stateful agent
harness. The task brief is inside the harness. The right-hand output contains
a global outline and local bar-unit instructions, followed by separate local
Q/M and whole-song G evaluation outside generation.

There are two explicitly different artifact forms:

- `harness_overview_gpt_v4.png`: GPT Image redraw of the supplied reference.
  This is a raster illustration. The built-in image tool produced the concept
  and a targeted connector/content correction; both prompts are in `source/`.
- `overview_method.svg`, `paper/overview_method.png`, `paper/overview_method.pdf`
  and `powerpoint/HereWeChoreography_harness_reference_v4.pptx`: the editable
  vector reconstruction. PNG/PDF and PPT in this group share one native master.
  The vector version has independently editable labels, icons, poses and
  connectors. It follows the same design but is not a pixel tracing of the GPT
  raster; do not use the raster to audit PowerPoint rendering consistency.

The foreground palette is vivid, while panels remain near white. Times New
Roman is the native font. The output poses and text excerpts are schematic
examples, not measured motion, model predictions or synchronization results.
The evidence/proposal cache contains text. Source audio feeds the local expert
and musical analysis. The composer consumes their cached proposals/summaries.
Structural checks return to the controller; benchmark scores do not.

No training or benchmark inference was run. The drawing does not introduce a
learned compatibility critic, physical transition renderer, or inference-time
weight updates. The current design uses bar groups; archived fixed-window
results do not validate the new duration configuration.

To rebuild the editable form, run `source/build_harness_reference.py`, then
`source/svg_to_native_scene.py`, `source/build_native_pptx.mjs`,
`source/export_paper_from_ppt_master.py`, and `source/finalize_native_pptx.mjs`.
The native helpers use the same bundled Artifact Tool and Times New Roman
setup as `../revision_20261117_v3/README.md`. Choose a new final PPT filename
before rerunning its finalizer. Validation covers structure and geometry,
not opening the file in desktop PowerPoint.
