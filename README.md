# H3-posttrain

MiniMax-H3 post-training for music-driven dance video (OmniDance): bidirectional LoRA SFT (DeCFG adapters,
guidance loss, Hybrid BF16 base), AR/causal experiments, music-condition injection, evaluation and review tooling.

## Layout
- `scripts_omnidance/exp/h3ar/` — H3 training / sampling / caching: `train_ha2.sh`, `q2_20261010.sh` (job launcher),
  `sample_h3_ar.py` (bi / block-causal AR sampler), `h3_causal.py`, `text_recache.py` / `nega_cache.py` (rich-prompt and
  unconditional sidecars), `fuse_adapter.py` (DeCFG / ostris adapter fusion, Hybrid merge), `h3_music.py` (MERT / MoE),
  `h3_dpo.py`, `eval_sched.sh` + `eval_jobs*.txt` (evaluation queue).
- `scripts_omnidance/exp/h3/` — H3 zero-shot baseline (E3) and OOS T2VA.
- `scripts_omnidance/exp/review_web/` — human-review web page (`python3 server.py --port 8765`).
- `scripts_omnidance/exp/{e6,e7,e9,e10,e11,dmd,g2,eval}` — earlier Wan2.2 experiments and metrics.
- `notes/` — design notes and experiment logs (19: H3 plan/log, 20: music condition ideas, 21: CFG / quality research).
- `diffsynth_patch/` — changes on top of DiffSynth-Studio commit `974cfa37f27ac55eba3b6d10efa21f876900572d`
  (`git apply DiffSynth-Studio_974cfa3.patch`); `files/` holds the full patched files
  (loss.py guidance modes, runner.py step cap / music lr, H3 DiT, H3 pipeline, dataset loader sidecars, train.py).

Paths, node addresses and dataset locations are those of the original cluster and must be adapted.
Access tokens were removed (`<HF_TOKEN_REDACTED>`); supply your own via environment variables.
Model weights, caches, data and generated videos are not included.

## Music-to-text choreography benchmark

[ChoreoBench](choreobench/README.md) provides separate benchmark contracts and reference
baselines for short-window captions, whole-song choreography scores and constrained
text edits. It includes CPU smoke fixtures, native planner integration, external
text-planner interfaces, strict edit checks, method-independent submission evaluation
and blinded review exchange. Formal datasets and real-model benchmark results are
not bundled; see its [protocol](choreobench/docs/PROTOCOL_zh.md) and
[server resource guide](choreobench/docs/RESOURCES_zh.md).

The [current harness implementation guide (中文, final section)](choreobench/docs/HARNESS_DESIGN_zh.md)
explains frozen audio proposals, whole-song design, causal block composition, public
scheduling constraints, bounded reinspection/revision, recovery and independent
evaluation boundaries, with the actual v11/v22 settings separated from defaults.

## Music-to-caption training analysis (separate from ChoreoBench)

[2026-11-09 中文训练复盘与下一阶段计划](notes/34_music_caption_analysis_20261109_zh.md)
records the completed Qwen Captioner SFT experiments, closed-vocabulary action evaluation,
JEV classification, measured training capacity/throughput, and proposed timestamp-free
and short-audio mapping experiments. Its trainer references refer to `main@02f478a`;
this documentation does not modify the ChoreoBench protocol, candidates or frozen results.

[CaptionLab](captionlab/README.md) implements separate, inference-only paired-audio
diagnostics and reversible target preparation for that analysis. See the
[8a833c9 review and next experiments](notes/35_captionlab_experiments_20261109_zh.md)
for execution order, resources, and remaining original-checkpoint integration.

The [parallel harness / Captioner experiment decisions](notes/36_harness_captioner_parallel_experiments_20261109_zh.md)
separate whole-song planning evaluation from local expert training, with independent
acceptance criteria and a later controlled integration experiment.

The [Captioner comparison and harness progress report](notes/37_captioner_comparison_and_harness_progress_20261110_zh.md)
summarizes the evidence as of `c350213`, explains the running pipeline, and separates
implemented controls from unverified model-quality claims.

The [bounded music reinspection and revision guide](notes/39_bounded_music_reinspection_and_revision_20261110_zh.md)
documents the shared-diagnosis experiment, audio reinspection, action-only repairs,
matched selection controls, deployment requirements, and remaining quality evaluation.

The [609c27e review and next experiments](notes/40_609c27e_review_and_next_experiments_20261110_zh.md)
separates the real v21/v22 execution evidence from unmeasured quality, records the
reused-diagnosis bound fix, and specifies independent candidate review, time-view
controls, new-source evaluation, and parallel inference-only Captioner comparisons.

The [671fc25 review and revised experiment plan](notes/42_latest_review_and_experiment_plan_20261110_zh.md)
incorporates the completed five-recipe Captioner pilot, prioritizes reuse of existing
paired scores and independent harness review, and defines a 40-output generation
pilot plus a bounded next-token prefix diagnostic before further training.

The [v23/v24 candidate-review audit and next experiments](notes/43_candidate_review_audit_and_next_experiments_20261110_zh.md)
updates the completed review tooling, separates timing-policy rejection from dance
quality, and specifies the two-reviewer target/candidate workflow before new model calls.
Section 8 records the real v24 source-trace re-audit and randomized review-package
deployment. Section 9 adds the assistant’s substantive review of all 227 original
events and six candidates: history-associated copying, overloaded short events and
minimal revision gains. Model review is not human or direct-audio evaluation; obvious
content issues no longer wait for a complete human roster before development proceeds.

The [CVPR experiment matrix](notes/44_cvpr_experiment_matrix_20261110_zh.md)
specifies five required experiments: dataset/pipeline quality, the SFT model against
open audio-instruction models, whole-song agent comparison, instruction editing,
and metric validity. Seven optional experiments cover agent mechanisms, music
cross-evaluation, pipeline ablations, expert-by-agent comparisons, robustness,
fixed visual execution, and controlled label-training comparisons. Required
experiments use existing weights; controls, samples, evaluation, resources and
execution order are specified separately from optional extensions.
The historical [v25 execution record](notes/47_memory_replay_v25_execution_record_20261110_zh.md)
is preserved separately from this forward-looking protocol.

The [latest main caption review and next experiments](notes/45_latest_main_caption_audit_and_next_experiments_20261110_zh.md)
reviews `ce4fc96`, reproduces failure cases in its lexical tier heuristic, separates
reported template prevalence from visual correctness, and prioritizes source/label
audits before bounded rewriting experiments. Harness memory development remains a
parallel track with its existing candidate bank frozen.

The [fixed-prefix replay review and content experiments](notes/46_memory_replay_review_and_duration_experiment_20261110_zh.md)
reviews `891e0f5` and records v26–v29: 21 real bounded calls for brief/duration
controls, adaptive slots and compact boundary memory, untimed base-Omni observation
anchors, and explicit oracle edits. Failures, costs and residual content problems
remain visible; the future task-trained untimed captioner is not yet loaded.
The manuscript draft includes these methods, negative results and oracle examples,
without claiming held-out quality, human ratings or a validated novelty advantage.
Section 10 additionally records v30: a typed primary/concurrent/finish representation
versus same-budget free text, six real calls, retained word-budget failures and
residual sequential overload—not a validated representation improvement.
