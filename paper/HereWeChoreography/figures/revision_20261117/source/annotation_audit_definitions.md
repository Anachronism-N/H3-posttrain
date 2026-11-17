# Annotation audit: definitions and research basis

The new panel is a descriptive audit of the same strictly paired original/refined records. Every number comes from the frozen `figures/source/dataset_distribution_audit.json` at source revision `1ca58b4ebc7f9457fb7dc7db587c05f3e79b767a`, SHA-256 `bef81f195f1fc88372ee6b3a8206818d043e28721fff97f9ab4a077254637052`. The selected fields and exact counts are frozen in `annotation_audit.source.json`; calculated percentages are in `annotation_audit.derived.json`.

All panels use all 22,352 strictly paired validation/test records. No producer-gate filtering or outcome-based case selection is applied. This population is the paired-data audit, not a count of independently verified songs or performers.

| Panel | Metric and denominator | Meaning and limit |
|---|---|---|
| (a) Source event count | Nonempty Chinese-full-stop-separated source statements, counted once across target windows; each bin count / 22,352 clips | Measures source annotation density. It does not count distinct actions or 3D motion primitives. Bins are categorical frequencies, with >=80 as the overflow bin. |
| (b) Refined caption length | ASCII Latin body-word count after removing the dance-style prefix; each bin count / 22,352 clips | Measures language length; source statements and English words have different units, so no numeric compression or diversity gain is claimed. >=180 is the overflow bin. |
| (c) Body-part information | Wrist/elbow/knee source-event occurrences / 1,055,138 total source-event occurrences | Exposes the information imbalance of the source rule vocabulary. It is not per-clip presence, annotation correctness, or whole-body motion coverage. |
| (d) Source temporal granularity | Number of existing rule-event windows per clip; each bin count / 22,352 clips | Identifies source temporal organization. These are not newly inferred bars, phrases, or the harness's 2–4-bar planning units. All audited clips have duration 5.0625 s. |
| (e) Source-style metadata | Existing unknown/contemporary/other label counts / 22,352 clips | Describes available metadata, preserving unknown. Labels were not independently reclassified; other is the sum of all non-contemporary, known source labels. |
| (f) Annotation audit | Producer `gate=true` and `gate=false` counts / 22,352 clips; paired IDs and mismatch counts are reported separately | Automatic pass/flagged status is producer-reported, not a human or independent VLM correctness rate. Paired IDs establish record reconciliation, not source-event factual coverage. Source/split mismatch and unchanged style fields are metadata checks only. |

## Why these panels are useful

The first three panels retain the original source-density, language-length and body-part diagnostics. The added panels characterize source segmentation, missing style metadata and the scope of automatic filtering. They explain what supervision actually contains, rather than making an unsupported improvement claim from free prose or nearly perfect automatic pass rates.

The design takes three ideas from primary dataset resources:

- [HumanML3D, CVPR 2022](https://github.com/EricGuo5513/HumanML3D) explicitly documents clip duration, language annotation multiplicity and average description length. The transferable lesson is to report language and motion characteristics in their own units. The present figure does not compare our caption lengths or annotation quality numerically against HumanML3D.
- [BABEL, CVPR 2021](https://babel.is.tue.mpg.de/data.html) distinguishes sequence labels, dense temporal labels and extra annotations, including their temporal support. The transferable lesson is to make source annotation granularity explicit, which motivates panel (d); our rule windows are not treated as equivalent to BABEL's human-labeled action intervals.
- [FineDance, ICCV 2023](https://github.com/li-ronghui/FineDance) documents music/motion alignment, style metadata and separate dancer/genre split protocols. It motivates making body/genre information and split assumptions inspectable. Our three-part lexical source vocabulary is not equivalent to FineDance's full-body joint observations.

## Unavailable measurements, reserved for collection

The frozen audit does not establish sentence-to-event attribution recall, unsupported-claim rate, human factual correctness, actual video/audio synchronization, independent song/dancer counts, cross-split audio fingerprints, or verified action-class diversity. These require raw annotation/event linkage audits, media inspection or independent human annotation. They are not plotted as zeros, perfect scores or surrogate automatic-pass rates.

## Reproduction and editing

Run `python source/build_annotation_pipeline.py` and `python source/build_annotation_audit.py` from the revision directory, or use their absolute paths. Both outputs use native Times New Roman `<text>` and individual vector shapes. `annotation_pipeline.svg` retains six stages and the detailed stage-4/5 panels, with filled human contours and editable joint/hand/leg landmarks. All poses and waveforms in that workflow are schematic; the statistical bars in `annotation_audit.svg` are source-bound measurements.
