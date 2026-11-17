# Experiment sources and reproduction

The manuscript includes an abstract, introduction, verified related work,
data provenance, method, evaluation protocol, completed-result analysis,
conclusion, and supplementary material. Dataset construction and compact
statistics appear in `sec/2a_dataset.tex`. `figures/dataset_overview.tex`
combines the richer six-stage framework and compact vector statistics
as two labeled panels, retaining the original framework artwork. The
explicit current-label provenance diagram is in the appendix. Their
different scope remains stated in the main text and captions.
The evaluation figure, nine-dimension core-anchor table, and per-song
aggregation formula are in `sec/3a_evaluation.tex`. Four result tables
are now in the main paper: full-2,000 E2; matched D1-D3 automatic
ratings; the 822-clip style comparison; and preliminary whole-song
composition. Historical controller reliability/cost and repaired
four-song exploratory ablations are supplementary. Wide result floats
are declared early beside the experiment
discussion. Main-text analysis also reports keyword precision/recall,
connective saturation, and historical repetition repair with its limits.
The redundant composition-hierarchy draft is excluded from manuscript
inputs. Diagnostics, implementation details, provenance, sensitivity
limits, and full rubric anchors remain in `sec/8_supplementary.tex`.
Pending experiments have no numerical scores.

## Published D1-D3 refinement ratings (2026-11-17)

- Source: H3-posttrain `dc6dfd9894fb702acb857524f930481efec829df`, branch `results/sft-refine-v1-2000-20261116`, directory `results/m2d/sft_v3_rp105_refine_v1_2000_20261116/`.
- Exact public manifest, outputs and judgments are preserved under `results/d123_refine_v1/`; the provenance receipt records byte hashes and all denominators.
- `python results/build_d123_refinement_table.py` recomputes the active `tables/refinement_d123.tex`. All 2,000 case identities and caption hashes were checked; invalid/null scores remain missing.
- The manuscript displays the supplied prediction set as **Ours**, without internal version branding or per-cell sample counts. Coverage is documented in the appendix; source identities and denominators remain unchanged in these receipts.
- Baseline release `92d6dcfd590d520cff40c7ec1cb04da48970e11e` supplies seven models under r4 holdout and older v1.2 full-run scoring. Exact exports and a joint audit are under `results/d123_baselines/`.
- The main table selects **r4**, the recorded scoring family available for all eight methods. Each dimension uses its all-method valid intersection on the common holdout; no scoring versions are mixed. The main table has no rating-version column or per-cell sample counts. Ours is 4.97 / 3.33 / 4.80; Audio-Flamingo-Next leads D2 at 3.74. All seven comparator scores are filled. Counts, available-case sensitivity and full case hashes remain in the appendix and audit.
- The author reports that the D1-D3 outputs are direct SFT predictions without an additional editor. This conflicts with the frozen export's processing-stage metadata; `d123_generation_clarification_20261117.json` records the unresolved provenance discrepancy. The manuscript does not assert either processing-stage account as independently verified. Frozen manifests, predictions, judgments and numerical scores are unchanged.
- r4 means/counts: D1 4.96698/1,999; D2 3.33467/1,990; D3 4.80190/1,999. v2.1: 4.92200/2,000; 4.50102/1,966; 4.91446/1,999. Versions score the same outputs and are never pooled.
- The exports lack judge identity, acoustic-input metadata, anchor definitions and evidence quotations. The baseline r4 version name is given, but its rubric content hash is absent; exact prompt-hash parity cannot be verified. Dimension IDs are retained as source IDs, not certified as our nine-dimensional rubric. v2.1 is excluded from the primary comparison. The observed comparison does not establish musical fit or semantic faithfulness.
- D4-D6 are displayed as unreported (`--`), not zero. Edited-output reference metrics have not been recomputed; the E2 agreement table retains the original SFT outputs.

## E2 caption evaluation

- Table source: `e2_full2000.source.json`, copied from the audited E2 full-2000 aggregate.
- Original repository: `Anachronism-N/H3-posttrain`.
- Original data file: `scripts_omnidance/exp/ms/planner/docs/SOL61_SELECTED436_RESULTS_20261114.json`.
- Frozen file SHA256: `ee3a8e8beb2d900543f0834f1e51921b808052abadf20724a477782eebd598c1`.
- Reference results use `population2000_context`, not the selected 436 cases.
- BERTScore uses `retain436_common_1861`; all other main-table columns use 2,000 cases. The original per-method available-case means and coverage remain in the diagnostic table.
- TC counts and duplicate excess are exact restatements of archived ratios, not newly evaluated semantic metrics.
- Included files: `tables/e2_reference_agreement.tex`, `tables/e2_diagnostics.tex`.

## MuQ-JEV dance-style ranking: active 822-clip experiment

- The manuscript reports only the 822-clip experiment, per the latest author instruction. Older result files, tables, and builders remain archived but are not manuscript inputs.
- Run `python results/build_style_prior_822.py` to regenerate `tables/style_prior_822.tex` and `style_prior_822.source.json` from the frozen comparison and audits.
- All seven zero-shot audio-language comparators and the CE, weighted-CE, and focal MuQ heads are retained on the same 822 clips. No new inference or resampling is performed.
- Rankings use 22 labels; Macro-F1 and balanced accuracy average over 13 observed classes. Only 15 classes have positive training support; eight evaluation classes have 100 cases each and the remaining five have 1, 1, 1, 3, and 16.
- Focal was chosen on validation before the preview. Its paired Top-k intervals against CE include zero. Video-VLM pseudo-label agreement does not establish human music suitability or downstream harness benefits.
- Supervised MuQ-JEV and the E2 SFT captioner are separate experts. The main table clearly separates supervised and zero-shot access.

## Expanded-label pilot and whole-song harness

- Expanded-style pilot sources: `baselines822-comparison-v1.json`, `preview-audit-v1.json`, and `pilot_final_status_v1.json` from sound-caption `a61bb1b`. These are a 22-output, 15-training-supported-class pilot, not a completed 22-class benchmark.
- `harness/` preserves the four published summary files used in `tables/harness_preliminary.tex`. Original cohort: 11 attempted songs, 6 completed pairs; extension: 9 attempted songs, 8 completed pairs. The extension JSON's stale top-level `n=6` is not the final paired denominator.
- Earlier whole-song judges consumed text plus acoustic summaries, not audio. Both Qwen judges share the outer planner's family; the independent MOSS attempt does not confirm the improvement.
- Code description follows H3-posttrain `083d1d0` on `codex/bar-style-agent-20261115`. The latest real-bar, shared-evidence four-policy comparison has not run. Its 12-song manifest is not yet registered, and server input/coverage checks remain pending.
- `e2_full2000_protocol_and_selection.source.md` identifies the v3 model as Qwen3-Omni-30B-A3B-Captioner with LoRA and clause CE weighting, and records the case-level decoding seed/settings. The separately frozen `joint_clause_weighting/` implementation and delivery report document joint-aware clause reweighting instantiated for lower-limb descriptions at weight 1.5, with valid-token-count normalization. Original training artifacts are still needed to verify training overlap, checkpoint selection, exact trainable scope and the full optimizer recipe. Only the full-2,000 aggregate is used in the main table; the source document's outcome-selected 436 cases are not the main evaluation population.

## Main-text evidence selection (2026-11-16)

- `build_style_prior_822.py` generates the active `tables/style_prior_822.tex`; `style_prior_822.source.json` records hashes and all 40 numeric cells. The older multi-population builder/table is archival only and is no longer included in the manuscript.
- `tables/harness_controller_history.tex` reports all 24 attempts per historical strategy, including failures. `harness/controller_history.provenance.json` records the original mechanical summary and independent inventory cross-check. Calls are total outer dispatches divided by 24; their mean is not the prospective equal-budget bar experiment.
- `tables/harness_exploratory_ablation.tex` uses corrected Section 17 of note 55, corroborated by `paper_result_v7.ablation_final`. Earlier timing-unit numbers and their intervals are superseded. `harness/exploratory_ablation_repaired.provenance.json` preserves the excerpt, hashes, and limits. Unequal acoustic access and development reuse prevent pure component or held-out claims.
- `harness/repeat_rewrite_effect.json` and `harness/truncation_audit.json` preserve the historical normalized-exact-text repair and presentation sensitivity results. Five valid pairs of six registered songs support the 73.14% to 36.82% descriptive change. The six-song 400-character comparison does not establish an advantage for the stricter rewriter (0.00, interval [-2.00, 2.00]). `mechanism_diagnostics.provenance.json` separates the three distinct repetition definitions and algorithm versions.
- Main text now explains keyword precision/recall and the saturated connective-presence metric. The larger surface-diagnostics table remains supplementary because it does not measure music alignment or choreographic quality. Original standalone style tables are preserved as source artifacts but not duplicated in manuscript inputs.
- The old 87/87 saturated quality assessment is disclosed as non-discriminating. No simulated author review is called human evaluation. Neither historical text judging nor classifier pseudo-label agreement validates the new audio-conditioned rubric.

## Figures

- **Active annotation diagram:** `figures/annotation_pipeline_v5.png`, edited with built-in GPT Image from the authors' preferred v2 layout. Card 2 has full-body geometry, palm ROI, and leg detail; wrist and knee callouts are separate. Matching 4/5 badges replace long expansion arrows. Stage 3 is Temporal Event Extraction, while MotionScript remains cited in the paper. The exact edit prompt is `figures/source/annotation_pipeline_v5.prompt.txt`. Earlier v2/v3/v4 assets are historical alternatives.
- **Active qualitative example:** `figures/choreography_example_v4.pdf` (editable SVG and PNG companions), generated from actual `wJPptuhclZmKAEWs` media and archived Beat This predictions. Six source frames (0, 20, 36, 56, 68, 80) show the opening bent-leg pose, hand movements, lowering, and final knee lift. A common display viewport removes empty background while retaining the entire person. One two-bar unit contains two unboxed action phases; the ending is an explicit fragment. Phrase roles distinguish body, action, spatial range and coordination. The hand-off reads the author-composed instructions rather than measured poses. Audio shows measured 2-ms extrema with one global scale, replacing the earlier RMS drawing; no waveform peaks are invented. `figures/source/plot_choreography_example_v4.py` and `choreography_example_v4.audit.json` preserve source hashes, frame indices and scope. Media pairing does not independently establish exact AV synchronization. Earlier versions remain inactive.

- `figures/annotation_pipeline_v2.png` is the historical design base for active v5. MotionScript attribution remains in the caption and data section; the original edit prompt and artwork are preserved.
- `figures/choreography_example.pdf` is the historical single-column illustration; v4 is the active example. Its original caption/source-event provenance remains archived.
- `results/writing_revision_20261116.md` records the writing-guidance sources, reverse outline, major claim/evidence map and five-dimension editorial self-review. None of these editorial artifacts is inserted into the submission body.

- `figures/annotation_pipeline.png`: preserved source artwork for the active v2 edit. The active caption distinguishes component integration from the actual 2D reference-label production route.
- `figures/dataset_construction_gpt_v2.png`: appendix provenance diagram. Solid arrows identify the current 2D-rule-log reference route; the dashed geometric extension is not established as their production history. Paired music bypasses the text-only refiner.
- `figures/evaluation_rubric_gpt_v3.png`: active GPT Image evaluation illustration. D3 is labeled Local timing and score 5 Fully supported, matching the written rubric. The frozen audio-only D6 decision is explicit. This is a protocol diagram without measured scores; calibration is pending.
- `figures/annotation_distributions_v2.pdf` (also SVG/PNG): historical vector plot, reproduced by `figures/source/plot_annotation_distributions_v2.py` from the frozen audit JSON. Active v3 retains the same 22,352 paired clips and units; body-part statistics count event occurrences. Earlier versions remain preserved.
- Prompts for the new dataset and evaluation artwork are saved in `figures/source/figure_prompts_20261116.json`. Numerical plots use standard plotting code, not generated values.
- `figures/harness_runtime_gpt_v4.png`: active GPT Image model-labeled architecture. The explicit bar-window arrow connects Beat This grouping to the local expert; tools use readable operation labels. Feedback, persistent state, and budgets remain visible. Generation receives no benchmark judge scores, and one local backend is fixed per run.
- `figures/annotation_distributions_v3.pdf` (SVG/PNG companions): active compact statistical panels, reproduced by `figures/source/plot_annotation_distributions_v3.py`. Inner panels use (i)--(iii) to distinguish them from the construction/statistics subfigure labels. All source counts and bins are unchanged.
- Latest figure edit prompts and plotting provenance are in `figures/source/core_figure_refinement_20261116.json`; earlier versions remain available but are not active manuscript inputs.
- `figures/agent_policy_comparison_gpt.png`: GPT Image control-policy schematic, included in the supplementary material. It compares control flow and reports no quality results.
- Generation prompts are preserved under `figures/source/method_image_prompts*.json`. The original colorful overview and canceled movement-card draft are excluded from all manuscript inputs.
- `figures/annotation_pipeline.png`: previously generated sequential GPT figure, copied without image modification.
- `figures/annotation_distributions.pdf`: vector version sized for a CVPR full-width figure. It uses the same 22,352 paired validation/test records and retains every automatic quality flag.
- Figure data, rendering script and image-generation prompt are preserved under `figures/source/`.
- The pipeline diagram describes proposed integration. Existing E2 references are documented as 2D rule-log rewrites; the full NLF-to-optimized-MotionScript chain must not be represented as their established production lineage.

## Compile

From the project root, use the usual Overleaf compile button or run:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Build products are ignored by Git. Source tables and figures use standard CVPR packages, with no external file paths required to compile. Absolute paths in provenance JSON identify audited source material only.
