# MotionScript-style 3D body front end — status snapshot (2026-11-20)

This directory documents the 3D-first MotionScript-style body front end,
the 200-source acceptance protocol, and the source-audit pipeline that
validates measured body events. It is a status snapshot, not a claim of
completed quality validation.

## Scope

Implemented and frozen before acceptance inference:

- official NLF v0.2.0 native RGB -> 3D body joints (`model_sha256`
  `2090b041ccebbd395018033152628c4f24afe519aed3a299861621547582cdad`,
  reference source commit `29046738a8101c15dc366ab45945b45e40debf17`);
- versioned joint mapping `smpl24-body20-no-toes-v1`
  (`e21c2877aa56fdab6842ad1402934762c2b6d9264be2f9b4a8529efef2ea5222`);
- native decoded frame indices, integer PTS/time_base and actual seconds
  (no frame-index/fps substitution);
- partial-body handling: upper body remains measurable without floor,
  pelvis or feet; long gaps are not interpolated; toes are not required;
- measured primitives: joint interior angles, relative elevation,
  wrist/ankle position along local axes, distances, speed/path change;
  every event keeps source frames and seconds;
- deterministic coarse evidence; no fine finger shape and no LLM caption
  polishing in this stage.

## Acceptance protocol (pre-registered)

- 20 development sources + 200 disjoint acceptance sources; selection was
  frozen before native inference, stratified from source RGB only.
- Every source is registered; failures and unknowns are retained.
- Independent review: source RGB first, freeze proximal-chain references,
  then compare candidates. Assistant diagnostics are not human reviews.
- Engineering gates (preregistered, not paper-universal thresholds):
  zero source/time/unit/mapping critical errors; false positives <= 5%;
  decidable-primitive misses <= 15%; boundary error P95 <= 0.20 s;
  zero real occurrences removed by dedup; decidable support coverage
  >= 90%, with stratified and per-video bootstrap reporting.
- A quality gate requires all 200 source reviews and at least 40
  independent human reviews, with no pending/invalid/disputed sources.

## Current status (honest)

- Native inference and postprocessing: complete for 200 acceptance sources
  (16,200 native source frames).
- Completed assistant source audits: **21 / 200** (batch 7, contract v8).
- Pending assistant audits: 179. Invalid bindings: 0.
- Independent human reviews: **0 / 62 required**. Assistant audits are not
  human accuracy evidence.
- Quality report: `docs/motionscript_3d_front/quality_report.v2-012.json`
  (status `pending_or_insufficient_review`, gate false).
- Partial metrics over the 21 audited sources (not 200-source results):
  tp 296, fp 0, fn 217, unknown 2311, decidable references 513;
  false-negative fraction 0.423, boundary-error P95 0.40 s,
  unknown operator-time fraction 0.929, decidable support fraction 0.985.
  These already exceed the preregistered miss/boundary limits, so the
  current pipeline is NOT validated as high quality. They are reported to
  expose the gaps rather than tuned away.
- Not started: original MotionScript real-video comparison (SMPL-H template
  and licence admission pending), LLM polishing, music association,
  training or deployment. No original annotations were modified.

## Source-audit pipeline

`tools/motionscript_source_audit/` contains the compilers and diagnostics
that produced the audits. They expect the artifact layout used during
development (workspace root with `panel-200-v1/`, `acceptance-panel-v1-postprocess/`
and `acceptance-review-workspace-v1/`); paths are resolved relative to the
script location's grandparent, so keep the original directory relationship
when reproducing.

- `compile_source_review_batch_v1.py` — first-principles reference compiler
  (source frame/pts/time-base validation, one-reference-one-candidate).
- `compile_source_review_batch_v2.py` / `_v3.py` — further source batches
  with interval compilation and disclosure fields.
- `compile_source_audit_batch.py` — current batch compiler: validates the
  frozen inventory SHA, source RGB SHA, facts SHA, native PTS binding,
  compiles references, writes per-candidate decisions, builds the
  supplemental source package and the next independent-review contract.
- `dev_uncertainty_geometry_diagnostic.py`,
  `dev_torso_axis_decomposition.py`, `coherent_pose_dev_diagnostic.py` —
  development-only diagnostics (no thresholds were selected from them).
- `download_nlf.py` — records the official asset download and SHA checks.
- `measurement_protocol.v1.json` — frozen measurement/parameter protocol.
- `batches/` — frozen batch definitions for batches 4–7.
- `source_observations/` — frozen per-source reference inventories
  (coarse, source-first tracks and explicit unknown reasons).

Verifier: `scripts_omnidance/exp/ms/ms_body_quality_v2.py` (in this repo).
It requires full-source channel coverage, decidable references inside
decidable intervals, exact channel/action/direction matching, one
reference supporting at most one candidate, and separate human reviews.

## Constraints preserved

- No credentials, media, model weights, caches or private paths are
  included in this directory.
- Unknown is not a true positive, not stillness; source projection and
  camera axes are not calibrated world/gravity/ground truth.
