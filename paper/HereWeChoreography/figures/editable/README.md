# Editable paper figures

The collection contains eight major figure panels, including both overview
alternatives. These are editable reconstructions, rather than PNG images
embedded in slides. The decorative title icon is outside this collection.

## Open and edit

- Open `powerpoint/HereWeChoreography_editable_figures.pptx` in PowerPoint.
  Text boxes, panel geometry, arrows, skeletons, chart bars and waveform paths
  are native objects. The deck contains 571 native text boxes and 3,117 native
  shapes including those text boxes. Use the Selection Pane to select named
  objects, and group related objects after editing if desired.
- Open each `.editable.svg` in Inkscape, Illustrator, or another SVG editor.
  Text remains `<text>`; shapes and connections remain vector objects.
- The six photographs in the qualitative example are unchanged source-video
  frames and remain independent image objects. The waveform and annotations
  are editable. Reconstructing a diagram does not turn its illustrative
  captions into measured model outputs.

| Slide | Figure source | Role |
|---|---|---|
| 1 | `overview_method.editable.svg` | Method-focused overview |
| 2 | `overview_data_benchmark.editable.svg` | Data/benchmark-focused alternative |
| 3 | `annotation_pipeline.editable.svg` | Six-stage geometric annotation framework |
| 4 | `evaluation_rubric.editable.svg` | Current evaluation dimensions and anchors |
| 5 | `agent_policy_comparison.editable.svg` | Four implemented control-policy adaptations |
| 6 | `dataset_construction.editable.svg` | Current 2D production route and separate 3D extension |
| 7 | `annotation_distributions.editable.svg` | Frozen paired-data statistics |
| 8 | `choreography_example.editable.svg` | Real frames/audio with illustrative instructions |

The six conceptual diagrams were rebuilt from the raster originals with
native primitives. Their semantics and major layout are retained, with some
illustrations simplified. The statistical and qualitative SVGs are copied
from the existing vector sources without changing data or source imagery.
In PowerPoint, translucent SVG areas use equivalent pale color tints; original
SVGs retain their alpha settings. This is not a pixel-identical conversion.

## Checks and scientific scope

`editability_manifest.json` records file hashes, native object counts and the
six independent photographs. `validation_receipt.json` records successful PPTX
package, geometry, font and first-party import checks. All eight slide renders
were inspected; `pptx_preview/` contains those renders. Two rotated-axis
textbox bounding-box warnings on the statistics slide were checked visually:
their displayed glyphs do not overlap. The deck was rendered with Artifact
Tool; it has not been interactively tested in the PowerPoint application.

The evaluation figure retains the manuscript's current D6 local-development
definition. The separate proposed affect rubric has not been substituted.
The benchmark grader stays outside the generation loop. No new training,
generation experiment, evaluation score or data-distribution claim was added.

## Rebuild

Generators and conversion code are under `source/`. Rebuild the conceptual
SVGs with `build_overviews.py`, `build_core_diagrams.py`, and
`build_additional_figures.py`. The last script copies the two existing native
SVGs from the parent figure directory; their plotting sources and frozen
data are under `../source/`.

`svg_to_native_scene.py` converts the eight SVGs into a derived scene JSON;
it requires Python with lxml, Pillow and fontTools and uses Windows Arial/
Times font files for text measurement. `build_native_pptx.mjs` uses the
bundled `@oai/artifact-tool` Node runtime to create native PowerPoint objects.
Set `ARTIFACT_NODE_MODULES` to the supplied runtime package directory when its
path differs. `finalize_native_pptx.mjs` validates a new final export; set
`PRESENTATIONS_SKILL_DIR` and `RUNTIME_NODE_MODULES` for a different runtime.
The finalizer intentionally refuses to overwrite an existing final file.

Private build files are excluded from the distribution. Figure reconstruction
does not modify the manuscript; the strict review is delivered separately so
the next revision can address its findings explicitly.
