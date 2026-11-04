# Related work and benchmark positioning

Scope checked on 2026-11-04. This package targets **music → open-language choreography plans**, full-song organization, and constrained text editing. Its methods are baselines for the benchmark. It does not yet establish a released scientific dataset, calibrated quality metrics, a public leaderboard, or state-of-the-art dance generation.

The table records relevant overlap and evaluation implications, not a cross-benchmark ranking. Research prototypes, advertised capabilities and reproduced functionality must be distinguished. A method that outputs 3D motion is not directly scored against text plans without a disclosed, frozen conversion and its limitations.

## Direct task and representation precedents

| Primary source | Relevant overlap | Implication for this benchmark |
|---|---|---|
| [MAESTRO project](https://midotronn.github.io/MAESTRO/), [reviewed code snapshot](https://github.com/midotronn/MAESTRO/tree/cfc77a3a1e607dfaebd0c381573d6a943616e11a) | Public positioning includes music structure, language storyboarding, motif return and window editing. The earlier audit did not locate the complete advertised editing pipeline in that snapshot. | A direct novelty precedent. Cite its stated scope; do not invent baseline scores or present our reconstruction as the authors' released implementation. |
| [AtomicDance paper](https://arxiv.org/abs/2607.13978), [official code](https://github.com/oceanflowlab/AtomicDance) | Plans action categories, locations and durations before variation and transition generation. | Planning layers and transitions have precedents. Test whether an open-language score and protected editing add measurable capability. Verify current weight/code availability before promising reproduction. |
| [STREAM paper](https://arxiv.org/abs/2606.22726), [official code](https://github.com/SeongJong-Yoo/STREAM) | Text/music conditioning, long-horizon motion and timeline control; music-swap evaluation is relevant. | Compare similar control requests, and use matched wrong-music/time-shift tests. Beat alignment alone is insufficient evidence of musical dependence. |
| [DanceChat paper](https://arxiv.org/abs/2506.10574) | Language motion instructions assist music-to-dance. | An LLM or language intermediate alone is not a new contribution. Separate the benchmark contribution from baseline architecture. |
| [MotionFix paper](https://arxiv.org/abs/2408.00712), [official data/code](https://github.com/atnikos/motionfix) | Source motion, an editing instruction and a target motion. | Borrow editing taxonomy and source-preservation reasoning. It is not a ready-made full-song music-to-choreography-text dataset. |
| [STMC official project](https://mathis.petrovich.fr/stmc/) | Temporal and body-part composition from multiple text descriptions. | Multi-track time/body control has precedents. Public format contracts are useful; its 3D denoising algorithm is not a drop-in text evaluator. |
| [LODGE official paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Lodge_A_Coarse_to_Fine_Diffusion_Network_for_Long_Dance_CVPR_2024_paper.html), [EDGE project](https://edge-dance.github.io/) | Long/coarse-to-fine dance and constraints on motion generation. | Extending duration is not enough for novelty; evaluate structure, editing, generalization and cost. Their 3D metrics are not text-plan musicality scores. |

## Stateful evaluation, harnesses and self-improvement

| Primary source | What it contributes | Boundary for our use |
|---|---|---|
| [ClawMark](https://arxiv.org/abs/2604.23781) | Stateful multi-turn coworker-agent evaluation with deterministic task checks. | Useful for edit histories and immutable checks, not a dance-aesthetic judge. |
| [DeepSeek prototype](https://github.com/deepseek-ai/dsh-superagent-prototype), [fixed architecture](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/docs/007-architecture.md) | Persistent tickets, asynchronous workers, recovery and versioned prompts. The README explicitly limits it to a research prototype, with some essay designs unimplemented. | Reuse state/recovery patterns; it is neither a choreography baseline nor demonstrated model RSI. |
| [Meta-Harness](https://arxiv.org/abs/2603.28052), [Agentic Harness Engineering](https://arxiv.org/abs/2604.25850) | Harness-code optimization from scores/traces; explicit editable components and predicted/verified edit effects. | Frozen-model harness optimization is a valid later experiment. Freeze the evaluation and report failed changes as well as accepted ones. |
| [Harness²](https://arxiv.org/abs/2610.14224) | Task-level probing plus experience that informs later harness adaptation. | Distinguish within-song revision from persistent changes used on later songs. |
| [Co-Harness](https://arxiv.org/abs/2607.22688) | Alternates harness improvement with model training on generated trajectories. | Joint weight/harness updates require a distinct data and compute protocol, not a mandatory benchmark dependency. |
| [RSI-Env](https://arxiv.org/abs/2611.01736), [full text](https://arxiv.org/html/2611.01736v1) | Persistent model/harness research state and an external controller; successor systems can conduct subsequent improvement. | Its intermediate acceptance checks validity/executability, not guaranteed capability gain. Weight updates are permitted, not the definition of every form of RSI. No author code entry was verified in the cited audit. |
| [NeuroClaw](https://arxiv.org/abs/2604.24696), [full text](https://arxiv.org/html/2604.24696v3) | Domain-aware skill dependencies, checkpoints, artifact checks and audit logs. | Borrow workflow evidence. Paper-reported report-rubric scores are not proof of all live executions; the author repository/homepage returned 404 during the audit, so current reproducibility remains unverified. |
| [Learning-to-Memorize](https://arxiv.org/abs/2611.01858) | Learns a router for historical video-frame KV retention with a frozen video backbone. | A long-video memory study, not an LLM-agent RSI method. Its router/cache integration is not already available in our caption planner. |

Repeated self-critique is inference-time refinement; a retained chat is memory. Persistent harness changes used in subsequent tasks can be studied without updating the base model. If an updated system recursively conducts future improvement, report that mechanism separately from whether held-out performance improved. Neither the benchmark nor its first baseline requires RSI, DPO, RL or fresh model weights.

## What a defensible benchmark contribution must demonstrate

- Real, licensed and group-audited songs; local clips and edit derivatives remain grouped with their source songs. Fixtures and mock outputs are explicitly excluded from scientific claims.
- Public audio/brief/tool predictions and private references/goals are isolated. Unknown group provenance, semantic review or usage is reported as unknown, not converted into a pass or zero cost.
- Task definitions distinguish creative quality, literal conformance, semantic goals, preservation and cost. Metric calibration includes reasonable alternate choreography and adversarial failures such as empty/no-op plans, wrong timing and repetitive templates.
- Baselines share input evidence and comparable budgets; a strong outer model or extra sampling is controlled. Failure and abstention remain in the original request accounting.
- Full-song planning, multi-turn editing and music dependence are assessed on held-out groups with human calibration. Optional RGB results are reported with a fixed renderer and cannot silently replace the text benchmark.

See [PROTOCOL_zh.md](PROTOCOL_zh.md) for release phases and [RESOURCES_zh.md](RESOURCES_zh.md) for actual model-loading and resource boundaries. These documents define a proposed protocol and runnable-software requirements; they do not claim the formal dataset or leaderboard already exists.
