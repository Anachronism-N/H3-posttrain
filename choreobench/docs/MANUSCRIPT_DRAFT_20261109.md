# ChoreoBench: Auditable Music-to-Text Choreography Planning and Bounded Revision

*Preliminary task-and-baseline manuscript. Empirical benchmark quality calibration and held-out evaluation remain incomplete.*

**Working manuscript — 2026-11-09.** Latest research direction is benchmark-first: task definitions, source-isolated task/annotation data, calibrated evaluation and baseline comparisons; the harness is a baseline, not the entire contribution. The present draft is still a method-oriented starting point. New nine-plan review material and task inventories do not constitute completed benchmark data or quality results. Refer to PROGRESS_20261109_zh.md for the current evidence boundary. This is a submission-oriented draft, not a completed empirical paper. Numerical results below are limited to verified engineering/development experiments. No quality improvement, statistical significance, human-study result or first-of-its-kind claim is asserted. Authors, venue and final manuscript format remain unset. Square-bracketed experiment placeholders are not results.

## Abstract

Music-conditioned dance systems often deliver motion or video, while an author may first need an interpretable score that specifies what to do, when to do it, and which parts can be revised. We investigate long-form music-conditioned choreography planning as the generation of an editable, open-vocabulary event score rather than a measured motion sequence. Our harness separates local audio-conditioned movement proposals, whole-song motif design, causal block composition, and deterministic constraint execution. The representation uses a canonical audio clock, explicit body scopes, motif references, and scoped field edits; private evaluation targets are not exposed to generators. We distinguish mechanically verifiable validity from musical and choreographic quality. In a three-song development study, six complete scores pass structural checks, and a frozen-outline history comparison verifies identical shared conditioning while retaining all requests and costs. A matched-schedule independent-block baseline also completes all three songs, showing that validity and coverage alone do not isolate a planning benefit. A three-task scoped-edit pilot changes each selected action while preserving all other event fields and exactly recovering the source through undo. Real bounded-revision pilots further expose a truncated diagnostic and internally accepted timing claims that conflict with the fixed-time contract. Fixed-prefix probes subsequently reveal history-associated copying, no clear benefit from duration wording beyond common brevity constraints, and retained adaptive-slot failures; explicit oracle edits demonstrate conditional content simplification. An untimed whole-window observation-anchor probe prepares an interface for a future captioner without claiming trained-expert or musical-quality gains. The current experiments establish executable, auditable workflows and inspectable failure controls, but do not yet establish perceptual quality gains. We identify a concrete limitation of near-equal scheduling menus and outline a held-out evaluation of musical relevance, long-range organization, edit satisfaction and preservation. This draft treats a successful system trace as necessary evidence of execution, not sufficient evidence of effective choreography.

**Final abstract action:** replace development-only wording with the actual supported main-study findings after frozen-baseline evaluation and independent review; retain negative/neutral results and do not claim professional physical execution from text scores.

## 1. Introduction

A choreography workflow is not only the production of a sequence of poses. Authors organize phrases across musical changes, reuse and vary motifs, and revise selected parts while preserving the rest of a score. Music-to-motion generation, however, can make the intermediate authoring decisions difficult to inspect independently of rendering quality. We study a complementary output: a timed, open-vocabulary choreography score that is itself an editable artifact.

A natural-language list is not sufficient for this purpose. A usable planning interface must state the audio clock, expose the affected body scope, preserve stable identities during edits, and distinguish a proposed transition from an observed physical state. Conversely, a well-formed score need not be musically useful: a decoder can enforce coverage without understanding a song, and a rich movement paragraph can be too dense or unsupported by the provided music. These observations motivate both a constrained planning harness and an evaluation that keeps validity separate from creative quality.

Our study asks: **What does explicit whole-song planning and causal composition contribute beyond a frozen local proposal model and a shared constrained output interface?** A second question examines whether a localized text instruction can change an existing score without changing protected fields. The goal is not to reproduce one reference dance exactly, generate verified human motion, or replace expert evaluation with schema checks.

### Proposed contributions and their evidence status

1. **Task and representation:** an auditable music-conditioned, long-form score with explicit event timing, body scopes and local edit permissions. This representation is implemented. Its practical value relative to existing authoring representations still needs evaluation.
2. **Constraint-aware execution:** separation of local musical proposals, whole-song design, causal composition and mechanical validation, with immutable traces and explicit failure accounting. Real development executions are available.
3. **Controlled analysis:** a frozen-outline comparison that removes generated-outline variation from the history intervention, plus matched scheduling controls and scoped-edit/no-op calibration. History execution and all three new independent-block/scoped-edit pilots are verified. No perceptual advantage is assumed.

The final paper should retain only contributions supported by experiments and related-work distinctions. Logging, JSON syntax and the number of unit tests are reproducibility mechanisms, not standalone claims of scientific novelty.

## 2. Related Work and Positioning

### Music-to-motion and structure-aware planning

Music-to-motion research already addresses rhythm, motion diversity and long-sequence organization. In *Music-to-Dance Generation via Atomic Movements* (arXiv:2607.13978), interpretable atomic movements are planned with types, duration and placement, followed by motion completion. This is closely related high-level structural planning; our paper cannot claim that symbolic dance allocation or hierarchical music-conditioned planning is new in general. The distinction to test is an open-vocabulary textual score as the independently delivered and editable artifact, with explicit execution and evaluation contracts, rather than an atomic vocabulary feeding a motion generator.

### Interactive choreography and editing

*CustomDance: Customized 3D Dance Generation with Coarse-to-Fine Human-Centered Interactive Control* (arXiv:2608.06722) uses multimodal music analysis, JSON anchors/creative cues, phrase authoring slots, motion retrieval and local editing to deliver 3D motion. It overlaps strongly with music-conditioned motif planning and authoring. Our draft differs in studying automatically composed text-event scores and permission-bounded edits without claiming a rendered-motion result. This difference is not sufficient by itself: the evaluation must demonstrate a reason to inspect and edit the intermediate score.

*DanceEditor: Towards Iterative Editable Music-driven Dance Generation with Open-Vocabulary Descriptions* (arXiv:2508.17342) studies iterative music/text-conditioned motion editing and introduces paired editing data. Our scoped-edit task concerns existing score fields and exact preservation boundaries, not motion-space reconstruction. We do not claim arbitrary open-vocabulary motion editing, nor directly compare text-score validity with motion FID or beat-alignment scores.

### Language-model planning and constrained execution

The general separation of proposal generation and symbolic validation is established in language-model planning. The camera-ready version needs a systematic related-work review of constrained decoding, program/plan repair, and music or motion authoring interfaces. **Pending bibliography expansion:** primary sources, evaluation scopes and strong planning baselines. No priority claim should be made from a limited search.

| Family | Deliverable | Planning/editing already present | Our evaluated distinction |
|---|---|---|---|
| Atomic-movement planning | Symbolic allocation and completed motion | Music-conditioned movement placement | Open-vocabulary score delivered independently; no motion equivalence claimed |
| CustomDance | Interactive 3D dance | Music anchors, motif cues, slots and local edits | Automatic text-score composition and auditable field-preservation contract |
| DanceEditor | Edited dance motion | Iterative music/text-conditioned edits | Explicit existing-event text edits; target satisfaction separated from exact preservation |
| Direct language-model scoring | Timed text/JSON | Prompt-based generation | Added whole-song/history mechanisms must be measured under shared constraints and costs |

## 3. Task Definition

Let a canonical mono waveform be x with sample rate r and duration N samples. The input consists of public musical observations, an optional author brief, and task-specific permissions. A score P contains motifs M and events E. Each event is

    e = (id, start, end, body_scope, action_text, motif_id, attributes, entry, exit),

where start/end are integer canonical samples with 0 <= start < end <= N. Intervals are half-open. Actions are open-vocabulary text; body scopes come from a declared track vocabulary. Within overlapping intervals, identical tracks or incompatible hierarchical scopes cannot conflict. For example, a whole-body event occupies the body parts represented by narrower tracks.

Unknown attributes or physical boundary states remain empty or unknown. A null support foot is not replaced by a guessed state. Event-level validity refers to declared text scheduling and scope, not verified biomechanical feasibility.

We keep three distinct evaluation tracks:

- **Local proposal:** a short music window maps to movement text. Existing observations use complete 5.0625-second inputs; a two-part output protocol is not a two-second audio crop.
- **Full-song planning:** the original complete audio duration maps to a coherent score. Failed songs remain in the declared inventory even if a valid prefix exists.
- **Scoped editing:** a fixed source score and instruction map to permitted replacements of existing fields. The instruction, source binding, allowed/repair event IDs and protections are fixed before execution. No event addition/deletion or implicit global retiming is allowed in the current task.

The dataset and published score need not contain a single uniquely correct choreography. Quality is evaluated through musical relevance, organization, clarity and editable control, not by requiring exact caption reproduction. Machine-generated captions are fallible proposals, not expert reference truth.

## 4. Harness

### 4.1 Public local proposals

A frozen audio-language model produces K=2 local proposals per observation window. The original development collection uses a base instruction model with a task prompt, not a task-trained choreography expert. Proposals and their parsing identities are frozen and reused across planner variants. A citation denotes which proposal was available to the generator; it does not certify that newly composed movements are semantically entailed or musically correct.

### 4.2 Whole-song design and causal composition

A text planner creates motif definitions and section-level intent from the public observations. The controller then visits immutable planning units in order. A block request contains the shared musical information, eligible proposals and, when enabled, the committed-prefix context. Only a validated block is added to the prefix. The no-history intervention withholds committed action text/state summaries, not overlapping observations already shared across variants.

For history-only analysis, both arms load the exact same externally bound whole-song outline. A seed alone is insufficient to fix intermediate content: the earlier independent-outline pilot returned different outlines for identical requests. We therefore validate actual loaded outlines and reconstruct shared block inputs.

    freeze observations and optionally import a bound outline
    for unit u in the complete immutable partition:
        construct public music/proposal/design input
        include validated committed-prefix context only when enabled
        request a candidate score under the declared decoder contract
        validate time, scope, theme permission and source references
        commit only on success; otherwise retain the failure and full cost
    publish a complete score only if the song finishes successfully

### 4.3 Scheduling and constraints

The current development configuration supplies a finite set of complete-unit schedules: one near-equal contiguous partition for each feasible event count up to three. A boundary grid and minimum event duration constrain this menu. Structured decoder alternatives fix the selected partition; the controller independently checks it. Complete coverage is therefore a public engineering constraint, not a measured creative improvement.

This choice reduces invalid timing but restricts expression. Once the event count is chosen, the internal cuts are determined without reading beats. A future beat-informed intervention must change the menu before generation and keep other conditions fixed; snapping a finished score after the fact would conflate planning with postprocessing.

### 4.4 Scoped text editing

The editor receives a public source score and a fixed permission-bearing instruction, not private evaluation goals. It proposes replacements of existing event fields. The controller rejects unauthorized targets, duplicate operations, changes to protected fields or invalid resulting scores. An empty patch is an explicit abstention; it is not automatically target success. Provenance records the applied patch, and undo checks exact recovery of the source object.

Mechanical preservation can be checked exactly. Whether a replacement actually reduces energy or improves a transition is a separate semantic judgment. The no-op control calibrates this distinction: it preserves everything while generally failing an unmet change request.

### 4.5 Bounded music reinspection and revision

We add a frozen-draft revision baseline without changing the initial generator or model weights. A text-only diagnostic pass proposes suspected event-level issues from the full score and public music predictions. Its output is shared by three conditions: targeted selection with music reinspection, matched-count random selection with reinspection, and targeted text-only revision. Selected event sets are fixed before revision and processed in original event order; no repaired output from one arm becomes another arm's input.

For music reinspection, the base Omni observer receives only a complete 5.0625-second PCM excerpt and a fixed music-observation prompt. The window centers on the event midpoint and shifts at song edges without padding. It receives no choreography, diagnostic reason or private evaluation goal. Its response is fallible musical evidence, not a verified beat annotation. The text editor can change only one existing action field; event timing, body scope, motif, state objects and every other action stay fixed. A seeded anonymous A/B comparison by the same text model may accept the candidate, keep the source or return uncertainty. This is an internal selection policy, not an independent benchmark judge.

For m selected events, listening arms allow up to 3m calls (listen, edit, compare); the text-only arm permits at most 2m. A no-op skips comparison, so equal caps need not mean equal actual cost. Empty diagnoses, no-ops, rejected candidates, truncated calls and unattempted events are all retained. Completion and acceptance are reported separately; weights are never updated and no RSI claim is made.

The first real revision pilot exposed both a lengthy diagnostic that exhausted its output budget and a time-unit hallucination accepted by the internal judge. A new revision version limits the diagnosis to at most eight suspected events and 400 characters per reason in both decoder schema and controller. Prompts state that event/beat timestamps are samples and equal adjacent half-open boundaries are valid. An action-only candidate is rejected if it introduces a new recognized numeric seconds/millisecond literal absent from the source action. This narrow mechanical guard protects a fixed-time contract; it does not verify all language-level timing statements or certify existing source claims.

### 4.6 Reproducibility and privacy

Input bytes, code versions, model/configuration declarations and request/response journals are bound to independent runs. New code/configuration uses new directories; unknown outcomes are not silently retried. Public review packs omit private method mappings, goals and citations. Original review files, a declared allocation design and optional issue sidecars are kept separate. These controls constrain evaluation leakage, but they do not independently authenticate remote model tensors, reviewer identities or listening.

### 4.7 Fixed-prefix content and memory interventions

To separate content organization from rollout feedback, a development probe replays one preselected block per recording from the same historical committed prefix. Full history exposes recent actions, body-part last events and motif examples; metadata-only removes event.action from all three paths while retaining timing, declared states, motif counts and the unchanged outline/proposals. It is not semantically empty conditioning. Both arms retain the same source seed, prompt and token cap; only validated new block events are produced. They do not become a new full-song trajectory.

We next compare brief expression (B) and duration-sensitive organization (D) under the exact same three-slot schedule, common 12–24 word target and hard 32-word/240-character ceiling. D adds a hypothesis that a sub-second slot should contain a primary movement phase with concurrent modifiers rather than several complete successive state changes. This is an engineering assumption, not a physiological law. Word and character limits differ; violations are retained without truncation or resampling.

A separate adaptive-slot probe admits the maximum number of slots satisfying a two-second scheduling minimum, then fixes that schedule across metadata-only and compact-boundary conditions. The latter presents only events ending exactly at the incoming boundary, with each stable event/action once rather than duplicated body-part examples. The adaptive probe changes the schedule relative to B/D and cannot serve as its isolated duration-prompt effect. Finally, an oracle edit supplies an explicit simplification target for one action, with all other fields protected; this tests conditional editor capability, not automatic diagnosis or selection.

### 4.8 Untimed whole-window proposals and musical observation anchors

A planned untimed captioner is expected to consume roughly five seconds of music and produce whole-window choreography prose. Until its actual checkpoint/input contract is available, we use the original Omni instruction model with a fixed untimed choreography prompt as an explicitly labeled interface substitute. No task-trained untimed expert result is claimed.

A predicted-downbeat observation policy shifts a complete 5.0625-second waveform to a nearby predicted downbeat without padding, time stretching or global-BPM extrapolation. The endpoint is not guaranteed to be a downbeat: this is start anchoring, not verified bar/phrase segmentation. For coverage manifests, backward anchoring keeps consecutive full observations connected; their ownership intervals partition the recording. Untimed prose is not split into invented action timestamps, and ownership is not evidence that all source actions fit after cropping. A three-recording local probe holds prompt/seed/decoding fixed and changes only the observation interval; prepared full-song manifests do not imply full-song generation was executed.

### 4.9 Typed primary phases, concurrency and closure

We test a more explicit action representation after oracle edits show that an instruction targeting the primary motion can change substantive content. A phase has a primary ongoing motion (main), zero to two simultaneous modifiers (concurrent), and an optional endpoint/hold (finish). Fixed slots of at most 1.5 seconds admit one primary phase; longer slots admit up to two. This capacity rule is an engineering hypothesis, not a biomechanical limit.

A text-control arm receives the same phase-budget instructions but emits a free action string. The typed arm emits closed phase objects, with field/count/character decoding constraints and post-generation component-word checks. Both use the same historical prefix, local proposals, outline, actual slot intervals, seed and total action budget. The decoder spaces and compilation step necessarily differ; this is the intended representation intervention, not a claim of identical decoding schemas.

The deterministic compiler joins main and concurrent text with while/and and appends finish, adding Then only between primary phases. It invents no support/contact state or extra motion facts. Typed raw responses remain available separately from their compiled score actions. A short field can still contain multiple hidden operations, and a finish can describe an implausible new maneuver; field validity alone therefore cannot certify semantic stage separation, musical fit or physical feasibility. Completed artifacts are audited by recompiling recorded raw responses, not by resampling missing calls.

### 4.10 Shared-tool instruction-editing policy comparison

A bounded action-observation editor compares a generic tool-use policy with a choreography-specific decision policy using the same frozen source score, public instruction, target/neighbors, model, seed and tool permissions. Both can inspect context, check a cumulative patch, submit a patch or abstain. Checks do not commit; every proposal is evaluated against the original score, preventing cumulative scope drift through rebasing. Only allowed action fields may change. The execution code, timing/metadata protection and tool-result interface are shared; the intervention is the system policy, not an external agent framework or a privileged executor.

The choreography policy emphasizes requested motion facts, within-target semantic preservation, successive versus concurrent movement, duration burden and local closure. The generic policy receives the same complete evidence and tools and may stop early. Each task/policy allows at most three model calls of 2,048 output tokens. A structurally valid patch is not semantic success, and budget exhaustion without submission is separately retained. Private evaluation goals are excluded from all prompts and tool feedback, and completed trajectories are reconstructed from recorded raw responses without resampling. This interface permits a matched editing study, but the small repeatedly seen development set cannot establish the final held-out editing result.

### 4.11 Source-span permissions and evidence-bound textual fact review

After a within-target preservation failure, we study a narrower editing subtrack in which the author specifies an exact source-action character span that can be replaced. A full-action response and a replacement-only response receive the same source score, instruction, span permissions and seed. Both must keep the surrounding text unchanged; the replacement-only response is compiled by literal source-span substitution. This preserves explicit unedited wording rather than accepting a promise to preserve whatever actions existed. It is not automatic semantic localization and restricts valid paraphrases outside the span.

The corresponding review protocol uses a different local text checkpoint, from the same model family, to inspect the instruction and source-evidence-bound motion facts. It distinguishes retained/equivalent, omitted, contradicted and unknown facts, and verifies that reported evidence is an exact candidate-text span. Model-only control tests include literal retention, deletion replaced by a preservation promise, explicit negation, and equivalent paraphrase. Known control recipes and method identities are omitted from the review prompt. These controls are assistant-constructed development material, not expert-validated or held-out gold data; the reviewer has no audio/video and does not become a human evaluation.

## 5. Experimental Protocol

### 5.1 Development data and models

The pilot uses three complete recordings of 252.9698125, 230.0343125 and 217.2081875 seconds. Source licenses and canonical audio preparation are recorded; the group/near-duplicate audit is unverified. These songs have been used repeatedly during development and must not be called a held-out test set.

The shared local collection contains 278 base audio-model calls. Whole-song composition uses a frozen 32B instruction model in text-only BF16 inference. The complete planning partitions contain 27, 26 and 23 units. The current menu has a 0.25-second grid, a 0.5-second minimum and a three-event maximum per unit. No task-specific model training or video rendering is performed in these harness pilots.

### 5.2 Baselines and intervention matrix

| Variant | Local music proposals | Whole-song outline | Committed history | Same menu family | Current evidence |
|---|---|---|---|---|---|
| Main | Shared | Frozen/imported | Yes | Yes | Verified historical v11 |
| No-history | Shared | Same frozen content | No | Yes | Verified v11 controlled pair |
| Independent constrained blocks | Shared | No | No | Yes | Verified v17 three-song development pilot |
| Direct full-song prompt | Same permitted observations to be specified | Native prompt behavior | Not causal blocks | Must specify equivalent permissions | Earlier prototype exists; matched main-study comparison pending |
| Scoped editor / no-op | Same source score and task | Source motifs unchanged | Source score | Original event times protected | Verified v17 patch/preservation/undo; semantic review pending |

The independent-block baseline tests the joint contribution of global design and committed history, not a pure history intervention. It also removes motif definitions and associated IDs. We will not rank motif-count differences as quality. Its v17 execution is compared descriptively with immutable v11 scores; service sessions differ and model stochasticity is not eliminated. A final main study needs contemporaneous repeated runs, explicit input equality where applicable and cost-aware reporting.

### 5.3 Evaluation

- **Validity/coverage:** complete valid-song fraction over the full declared inventory, scope conflicts and timing-contract satisfaction. Show that a baseline with the same decoder can also satisfy these constraints.
- **Score quality:** blinded musical fit, long-range organization, planned executability and boundary clarity. These are judgments about textual scores, not measured bodies or synchronized video.
- **Editing:** semantic instruction satisfaction, exact preservation of protected/non-target fields, new local problems and abstention. No-op is a calibration control, not a claim of strong editing quality.
- **Diagnostics:** retain localized unintended-repeat/ambiguity/music-mismatch observations, positive motif recurrence counterexamples and uncertainty. Counted declarations are not calibrated quality metrics.
- **Efficiency:** actual input/output tokens, model calls and failed/unknown outcomes; shared collection and historical outlines counted once. Tokens do not capture loading, idle rental or complete GPU-hours.

Actual reviewers have not yet been provided, and there are no submitted quality scores. Reviewer roster/design checks only verify declared allocation and completeness. Model judging may support exploratory analysis if separately labeled and calibrated; it cannot be reported as an expert human study. Missing ratings remain missing, never zero or implicit success.

### 5.4 Held-out study required for the final paper

The latest executable protocol separates five required studies (pipeline/data quality, SFT versus strong audio baselines, fixed workflow versus general tool agent versus our harness, instruction editing, and metric validity) from optional mechanism studies. The fixed-prefix and untimed substitute experiments below are development evidence, not completion of those required studies; the new SFT captioner, general-agent main comparison and calibrated independent evaluations remain separate deliverables.

Before adding test songs, freeze the selected hypothesis, prompts, menu, rubric, model settings and edit-task coverage. Audit source/artist/near-duplicate/audio relationships and licensing; retain the existing three songs as development. Determine scope from a pilot's variability and rating disagreement rather than declaring a sample size guarantees significance. Use repeated decoding to measure session/output variability, and group-level rather than event-level comparisons. We do not claim population generalization from the current three songs.

## 6. Results Available for the Draft

### 6.1 Frozen-outline history pilot (verified v11)

| Recording | Main events | No-history events | Calls per arm | Minimum event duration | Complete validity per pair |
|---|---:|---:|---:|---:|---:|
| Development song 1 | 81 | 81 | 27 | 1.25 s | 2/2 |
| Development song 2 | 78 | 78 | 26 | 0.75 s | 2/2 |
| Development song 3 | 68 | 69 | 23 | 1.5 s | 2/2 |

All three pairs satisfy actual shared-outline/common-input control. Main selects the maximum three events in 75/76 units; no-history does so in 76/76. All complete scores cover the canonical duration under the supplied schedule constraint. Action and track sequences differ in all pairs, while pure time boundaries differ only for song 3. Outputs differ between the verified intervention arms; these single-run observations do not establish a stable causal effect or a quality benefit from history.

| Arm | New calls | Input tokens | Output tokens |
|---|---:|---:|---:|
| Main | 76 | 490,835 | 33,953 |
| No-history | 76 | 272,351 | 33,724 |

The history input incurs additional context. A cheaper no-history variant may be preferable if independent ratings show no meaningful benefit. The original three imported outline calls and audio collection are historical shared costs, not additional per-arm calls.

### 6.2 Independent-block baseline (v17)

The selected independent-block arm completes all three predeclared songs, with no recorded model failures or unknown outcomes. Every actual request omits generated whole-song design and committed history. The reused v11 main and the new arm have identical candidate/music observations, unit menus, block seeds/output caps and nine audited generation-core module hashes. No baseline outline call is made.

| Recording | Complete valid baseline | Events | Calls | Coverage | Independent quality |
|---|---|---|---|---|---|
| Development song 1 | 1/1 | 81 | 27 | 1.0 | Not submitted |
| Development song 2 | 1/1 | 78 | 26 | 1.0 | Not submitted |
| Development song 3 | 1/1 | 69 | 23 | 1.0 | Not submitted |

The independent baseline chooses three events in all 76 units, emits no motif definitions and uses only null motif references. It consumes 226,520 input and 34,741 output tokens across 76 new calls. This valid baseline is a useful control: complete coverage and validity cannot establish that the global/causal machinery improves choreography. Its lower input burden is a cost observation, not a quality preference.

**Comparison caveat:** immutable v11 main was executed in a previous service session. The baseline is matched in observations, planning units, schedule family and declared decoding caps, but is not a concurrent repeated causal comparison. Relevant planner implementation bytes and inputs are audited explicitly.

### 6.3 Scoped-edit/no-op pilot (v17)

One first-event low-energy instruction per development song is frozen before generation without inspecting new outputs. Protect timing, track, motif/state fields and all other events. All three editor attempts return a valid, non-abstaining action replacement; checks verify preservation of every other event field and exact undo. No-op keeps events unchanged; an unchanged unmet instruction is not semantic success. The replacement content still needs independent review even when a different action string is emitted.

| Task | Editor returns valid patch | Non-target fields preserved | Exact undo | Semantic satisfaction | No-op target satisfaction |
|---|---|---|---|---|---|
| Song 1, first event | Yes | Exact | Exact | Not reviewed | Not reviewed; not presumed successful |
| Song 2, first event | Yes | Exact | Exact | Not reviewed | Not reviewed; not presumed successful |
| Song 3, first event | Yes | Exact | Exact | Not reviewed | Not reviewed; not presumed successful |

The editor uses three new requests, 38,053 input tokens and 1,207 output tokens; the no-op control uses no model call. Overall v17 costs are 79 calls, 264,573 input and 35,948 output tokens, with zero new audio/outline calls. Historical main, outline and audio collection costs remain separate.

Selected first-event examples retain the source duration: song 1 changes wording from sharp/staccato lifts and hip drops toward “small, in-place steps” and “gentle, relaxed arm movements”; song 3 replaces percussive stomps/downward strikes with small steps, gentle arms and a soft torso sway. These are inspectable text differences, not independent judgments that the target, musical relevance or physical feasibility is satisfied. The second replacement still includes a low crouch/subtle knee lifts, illustrating why string change or soft-energy vocabulary cannot replace semantic review.

Separate blinded three-case packages contain six main/baseline plans and six editor/no-op plans, with the public source and instruction shown for edits. HTTP/audio ranges and actual browser interactions are verified; temporary browser test grades are discarded and all submitted human ratings remain absent.

### 6.4 Scheduling expressivity diagnostic (v16)

| Recording | Main internal cuts | Median distance to nearest predicted beat | Within 50 ms | Same-count beat-cut feasible units |
|---|---:|---:|---:|---:|
| Development song 1 | 54 | 170 ms | 18/54 | 27/27 |
| Development song 2 | 52 | 82.5 ms | 13/52 | 25/26 |
| Development song 3 | 45 | 165 ms | 6/45 | 22/23 |

These are representability/nearest-distance diagnostics, not a beat-synchrony score. Predicted beats are not audited ground truth, and text event cuts do not measure movement accents. With fixed unit endpoints, actual event count and minimum duration, 74/76 main units admit a different predicted-beat partition. The two infeasible units are retained. This identifies a testable limitation of the scheduling interface, not an evaluated replacement method.

### 6.5 Real revision pilot and retained failures (v21)

The pilot preselects the same three original main scores and m=1 per song. A single diagnosis is frozen per song, with no resampling after failure. Targeted/random have a listening cap of three calls and text-only two. All arms share the initial score, public music evidence and diagnosis; targeted/text-only share the event, while random samples from the complete source event inventory. The additional music channel has a different cost and is not an equal-cost treatment.

| Development song | Diagnostic outcome | Targeted/listen | Random/listen | Targeted/text-only |
|---|---|---|---|---|
| 1 | Completed, 49 suspected items | Candidate internally accepted; 3 calls | No-op, 2 calls | Candidate internally accepted; 2 calls |
| 2 | 4096-token truncation | Not attempted | Not attempted | Not attempted |
| 3 | Completed, 18 suspected items | Internal uncertainty; source kept; 3 calls | No-op, 2 calls | Internal uncertainty; source kept; 2 calls |

Actual v21 cost is 17 calls (three diagnoses, four audio observations and ten edit/comparison calls), 311,024 input and 12,767 output tokens. The failed diagnostic is included; the three unattempted song-2 conditions are not replaced by new targets or silently shown as successful original outputs. The package contains nine legal displayed plans from twelve predeclared slots, and the failure remains in the experiment inventory.

A source timestamp of 640 samples at 16 kHz equals 40 ms. The song-1 accepted revision nevertheless appended a statement about 640 ms. Its text-only counterpart made the same unit error. In another selected issue, equal end/start boundaries were treated as evidence of an abrupt cut, although adjacency alone violates no score rule. These inspectable inconsistencies demonstrate that internal acceptance does not certify an improvement. The source score, suspect diagnostics, candidate texts and choice reasons remain available as separate traces. We neither report accepted_count as repair accuracy nor claim music reinspection outperforms random or text-only revision.

Version v22 is an explicitly separate bounded-diagnosis/clock-guard deployment, not a hidden retry of the v21 failed song. All three songs are preselected again with the same one-event budget. Its actual execution results are recorded below only after verification, with v21 failures retained and costs not pooled as one successful run.

### 6.6 Bounded-diagnosis revision deployment (v22)

The separately frozen v22 pilot limits diagnoses to eight items/400-character reasons, explains sample units and valid adjacency, and guards recognized numeric seconds/millisecond additions to action prose. The original three songs are all preselected before its outputs. All three diagnoses finish with eight suspected items (940, 916 and 893 output tokens), and all nine revision-arm inventories are processed without a recorded model failure or additional repair sampling. Exact same-condition resumes add no new model calls.

| Development song | Targeted/listen | Random/listen | Targeted/text-only | New calls including one shared diagnosis |
|---|---|---|---|---:|
| 1 | Internal uncertainty, original kept | No-op | Internal uncertainty, original kept | 8 |
| 2 | Candidate internally accepted | No-op | Candidate internally accepted | 8 |
| 3 | Internal uncertainty, original kept | No-op | Internal uncertainty, original kept | 8 |

The full twelve-slot review package is present: one original plus three strategy outputs per song. Repeated originals are policy outcomes, not additional independent choreographies or successful repairs. All event times, body scopes, motifs, state fields and nonselected action/evidence fields remain unchanged. The observer receives complete 81,000-sample PCM windows; their bytes match the corresponding original slices. Six real music observations and eighteen text-model calls (including three shared diagnoses) constitute the 24-call v22 total.

| v22 stage | Calls | Input tokens | Output tokens |
|---|---:|---:|---:|
| Shared diagnoses | 3 | 58,563 | 2,749 |
| Targeted/listen | 9 | 158,759 | 2,403 |
| Random/listen | 6 | 64,265 | 1,420 |
| Targeted/text-only | 6 | 157,360 | 2,147 |
| Total | 24 | 438,947 | 8,719 |

The internal decisions do not establish a reinspection benefit: targeted/listen and text-only each accept one candidate, while the three random attempts are no-ops. More importantly, inspection of the accepted song-2 action reveals a remaining loophole. The listening candidate adds a bare numeric downbeat reference, and the text-only candidate adds a downbeat at sample 54,400 although the event ends at 52,000. The first guard recognized seconds/millisecond units, not unqualified beat timestamps. We therefore extend the narrow literal guard to explicit samples and bare numeric beat/onset/timestamp phrases. An offline replay of these two original candidates rejects both; this is deterministic guard validation, not a newly executed quality-improved model. The v22 result files and internal acceptance counts remain immutable.

Across v21 and v22, 41 model calls consume 749,971 input and 21,486 output tokens, including ten new music observations. These are new revision costs, not a replacement for or repeated accounting of the historical local collection/main generation. No original full-song candidate collection or model training is repeated. We preserve the failed v21 diagnosis and unattempted arms, and do not pool the two protocol versions as one favorable effect estimate. Both revision packages pass actual browser/HTTP range checks and retain null human ratings.

### 6.7 Failure analysis

Early whole-song global variants reproduced concatenated content and hierarchical variants mainly changed motif labels. A later free-boundary grid pilot failed on body-scope overlap, and a time-first pilot failed on sub-minimum events. Full relational decoding solved those specific violations but produced low coverage; complete-unit menus restored contractual coverage while restricting timing freedom. These failures motivate decomposition and clearer control, but are not themselves evidence of semantic gains. Changed prompts/contracts are development revisions, not clean single-variable quality ablations.

### 6.8 Fixed-prefix history copying (v25)

Three preselected development blocks share their historical prefix, candidates, outline, schedule and source seed. All six requests complete with valid event arrays. Full history yields four exact copies among nine new actions, while metadata-only yields none; action characters total 3,466 versus 1,588. Exact equality is measured against the entire historical prefix, and the later source-intersection audit confirms all four exact copies were visible in the request. It does not identify a unique causal source or establish semantic novelty when strings differ. Three pairs consume 33,490 input and 2,525 output tokens. Text inspection suggests fewer redundant successive operations after history removal, but the 0.75-second rock slot remains overloaded; no whole-song quality advantage is established.

### 6.9 Briefness, duration organization and oracle simplification (v26/v28/v29)

| Independent development stage | Calls | Valid outputs / declared attempts | Input / output tokens | Supported observation |
|---|---:|---:|---:|---|
| B/D, fixed three slots (v26) | 6 | 4/6 | 30,533 / 2,207 | D shows no clear additional benefit; two outputs exceed the shared word ceiling |
| Adaptive slots, metadata/compact boundary (v28) | 6 | 0/6 | 31,561 / 1,353 | All exceed the word ceiling; one response contains a malformed mixed-language fragment |
| Explicit oracle action edit (v29) | 3 | 3/3 | 25,434 / 1,453 | Single-event content changes while every other event field is preserved |

B/D use the exact three intervals observed in v25, the same prefix and metadata-only evidence, and the same model/seed/length ceiling. The rock middle slot has identical action text in both conditions: a spin landing on one leg with knee lift and coordinated arms/torso within 0.75 seconds. This gives no clear support for the additional duration wording. Both acoustic-recording outputs obey the character ceiling but exceed 32 words per action (36/40/41 for B and 36/34/34 for D); they remain failures, not shorter postprocessed scores.

The adaptive stage changes slot counts to 1/1/3 for the three blocks, with a common two-second scheduling minimum and either no action history or one deduplicated boundary action. All six violate the same word ceiling. Fewer slots therefore do not, by themselves, make the generator obey content constraints. The raw electronic compact-boundary output ends with an incomplete mixed fragment; closed JSON and character bounds do not certify clear movement text. These failures cannot be pooled with the earlier valid outputs as a favorable quality effect.

Oracle targets are supplied by development review, not an automatic critic. The editor simplifies the electronic 1.25-second chain from roll/isolation/lunge/pivot/sweep into a continuous body wave with concurrent shoulder/arm modulation; the 0.75-second rock chain becomes a small continuous hip rotation with arm circling, without a full spin or landing; the acoustic finish no longer restarts with taps/thrusts after stillness. These are substantive inspectable changes, unlike merely appending a beat reference. Residual ambiguity and real execution remain unverified. The oracle uses a different source/task/decoding contract and a soft length preference, so 3/3 validity is not a same-condition repair-rate improvement over B/D or adaptive-slot failures.

### 6.10 Untimed start-anchoring interface probe (v27)

| Recording | Anchored start shift | Fixed / anchored prose words | Whole-song fixed / anchored manifest windows |
|---|---:|---:|---:|
| Development song 1 | -0.405 s | 115 / 61 | 50 / 68 |
| Development song 2 | -0.0625 s | 133 / 127 | 46 / 53 |
| Development song 3 | -0.140 s | 125 / 106 | 43 / 49 |

All six base-Omni calls stop normally and return prose without recognized timestamp formatting. Both sides consume the same 81,000-sample waveform length with the same untimed task prompt and sampling seed, but different source intervals. The acoustic anchored prose uses a more continuous torso/arm movement than its fixed-window counterpart, while both electronic outputs retain a similar knee/arm inventory. This is exploratory text inspection, not direct listening or evidence that anchoring improves musical fit. The prepared coverage manifests require more overlapping observations, exposing a cost trade-off; only six local captions were actually generated, not a new full-song score.

Actual provider-reported cost is 1,038 input and 804 output tokens; the native input-token count is not a measure of full audio-encoder computation. The task-trained untimed captioner has not been supplied or loaded. Across v26–v29, the new bounded experiments total 21 calls, 88,566 input and 5,817 output tokens. Historical local-proposal/outline/v25 costs remain separate, and no failed condition is resampled. All model services are stopped and used GPUs restored to unique holders. These experiments add executable methods, matched content controls and explicit negative/oracle findings to the writing evidence, not held-out benchmark quality or a human-study result.

### 6.11 Typed movement-phase probe (v30)

The typed-phase and same-budget text-control arms use the same three fixed blocks/prefixes/proposals/design/seed, with a one-phase cap for slots of at most 1.5 seconds and at most two phases otherwise. Each primary string targets at most eight words, concurrent modifiers four words each, and closure six words, while total compiled action remains capped at 32 words/240 characters. Schema constraints enforce fields/counts/characters; word limits are checked afterward.

All six calls stop normally. Two text-control blocks pass the overall mechanical checks, while all three typed blocks violate component-word budgets; the acoustic text-control block violates total word count. The electronic typed response also contains a mixed-language fragment. None is resampled or truncated. Different typed/free schemas impose different detectable constraints, so these validity fractions are not a semantic winner comparison. Text inspection still finds full spin-and-landing in a 0.75-second typed phase, and additional rise/glide operations placed inside closure fields. Merely naming a field primary or concurrent does not certify that it contains one stage or truly simultaneous motion.

v30 consumes 32,015 input and 2,427 output tokens across six calls. Completed raw/artifact inventories are reconstructed by a read-only audit, and same-condition resume issues zero new calls. Across v26–v30, the development probes total 27 new requests, 120,581 input and 8,244 output tokens; historical shared audio/design/v25 costs remain separate. All associated services are stopped and GPUs restored to unique holders. The current evidence supports an inspectable failure analysis and conditional oracle-editor capability, not a typed-representation quality gain or completion of the required held-out studies.

### 6.12 Same-tool instruction-editing comparison (v31)

We predeclare four tasks per previously seen development recording: density reduction, within-target lower-body preservation while changing upper-body coordination, action replacement or ending, and an already-satisfied/conflicting control. Both generic and choreography policies receive the same complete source/task/target context, tools, initial seed and three-call ceiling. All 24 task-policy runs complete; each policy submits nine modifications and abstains on all three controls. Exact non-target fields and original event identities are preserved throughout. Same-condition resume and recorded tool-action reconstruction issue no new calls.

| Policy | Task runs | Submitted edits | Unchanged controls | Model calls | Input / output tokens |
|---|---:|---:|---:|---:|---:|
| Generic | 12 | 9 | 3 | 29 | 546,225 / 5,289 |
| Choreography-specific | 12 | 9 | 3 | 28 | 535,432 / 5,456 |

Six of twelve pairs have identical final event content, including the controls. Source/task/payload/schema/permissions equality is verified at the initial request; only policy differs, with subsequent histories allowed to diverge. This is a shared-executor policy comparison, not a verified implementation of a specific external general-agent framework. Its high input burden includes repeatedly transmitting the entire original score and must not be hidden by quoting only action tokens.

Assistant text inspection identifies partial density changes that still retain several successive operations despite calling them continuous. In the acoustic upper-body task, the generic edit explicitly retains the original percussive stomps, whereas the choreography policy replaces that fact with a generic promise to preserve whatever lower-body actions existed. Mechanical scope checks accept both; within-target semantic preservation is not certified by field equality. Ending changes in the acoustic recording visibly remove tap/thrust restarts after stillness. The rock ending task, however, already ended in a hold; its planned unmet classification is not independently validated and is separately flagged without changing the preregistered inventory.

No stable quality advantage of the choreography policy is established. The nine submitted edits are not nine joint successes; controls, partial/unknown judgments and task-authoring ambiguity remain separate. These are non-blinded assistant text reviews, not direct music listening or human consensus. The current three recordings remain unverified-grouping development data, and the user confirms that formal model/test artifacts are not yet available. The framework and development comparison therefore do not complete the required held-out E4 study, pipeline/data evaluation, SFT baseline or metric calibration. All services are stopped and used GPUs restored; the complete 57-call trace and its 1,081,657 input/10,745 output tokens are retained.

### 6.13 Source-span editing and factual-review controls (v32/v33)

Three explicit upper-body text spans are edited in the same source scores, comparing full-action output with replacement-only output under identical span permissions. All six calls compile to valid, non-abstaining action edits; unchanged source words directly retain stomps, knee lifts, hip drops and crouch/weight-shift content. The rock and acoustic outputs are identical between formats; the electronic pair differs only in the connective wording. The explicit-span subtrack is narrower than broad semantic E4 editing and does not demonstrate automatic localization or a quality advantage of replacement-only formatting. Costs are 75,480 input tokens and 364 output tokens, with full-action versus replacement-only outputs of 202 versus 162 tokens.

A different Qwen3-32B text checkpoint, from the same model family as the generation checkpoint, then reviews 12 assistant-constructed retain/omit/negate/paraphrase controls, six actual span outputs and the prior actual acoustic preservation failure. Method/control identities and expected labels are hidden from the reviewer, which has no audio/video. All 19 calls finish normally; four predictions fail exact candidate-quote validation. On the full 12-control denominator, eight predictions agree with the constructed labels, nine yield valid evidence-bound records; the three omission-promise controls all incorrectly claim retention in raw output and invent supporting quotations. They are blocked by quote membership, not correctly classified omissions. One explicit-negation control yields legal quotes from an earlier positive clause while overlooking a later instruction to keep knees/pelvis fixed and never shift weight.

The literal all-source-evidence substring baseline agrees with seven of twelve constructed controls, rejecting all three equivalent paraphrases and missing two explicit contradictions. These results show different failure modes, not formal E5 validity or statistical superiority. The six actual span edits are rated retained by the model, but the prior actual omission is again misrated with a fabricated quote and blocked. Source quote membership alone is insufficient for comprehensive semantic preservation, and a different checkpoint is not independent ground truth.

v32/v33 add 25 calls, 85,402 input and 3,916 output tokens. Source/request/response/compiled-edit chains, label hiding, failed record retention and zero-new-call resume are verified; services are stopped and used GPUs restored. Original development sources still fail documented grouping/near-duplicate release gates. Formal model/test manifests and independent evaluation are unavailable, so these experiments contribute source-scope mechanisms and calibration failure analysis without completing the required main studies or human joint-success evaluation.

### 6.14 Joint-edit evaluation contract and zero-call evidence audit (v34)

A v2 evaluation record adds an explicit protected-semantics verdict to every original goal, local boundary verdict and required review interval. A separate task-authoring document binds every registered task and source score, records whether a change is genuinely needed, and checks that private evaluation requirements do not exceed the public instruction. Two specified human reviewers must agree on the complete joint outcome. Unknown, incomplete review and unresolved disagreements remain U; mechanical/literal execution failures remain F. Each method uses the same author-valid change-task denominator N, with S/N and bounds [S/N,(S+U)/N]; missing author validation is retained separately rather than used to manufacture N. Control tasks and historical costs are separate. Observer and pre-execution-freeze declarations do not authenticate human identity, timing or truth.

The zero-call audit maps all 24 v31 task-policy states (12 registered tasks, three development songs) to anonymous before/after requirements: 18 submissions and six explicit abstentions. Abstention binds the exact unchanged source without fabricating an executed empty patch or changing its provenance. All 24 states are mechanically valid, but all 48 required human records, 144 goal verdicts, 48 protected-semantics verdicts and 48 boundary verdicts remain missing. None of the 12 tasks has independent human author validation; the questionable rock-ending task remains unresolved, not retrospectively relabeled as a satisfied control. Consequently no formal E4 success rate is reported. Prior assistant prose is not converted into per-goal human labels.

The factual reviewer now requires nonempty exact candidate evidence for retained/contradicted facts; a known omission or contradiction is not hidden by another fact's unknown status. Offline re-parsing of the original 19 v33 responses still yields 15 protocol-valid records and four failures, with no changed validity labels. This is a parser audit, not a new experiment under the revised prompt, and it does not repair the previously observed legal-quote/negation error. The audit uses no new model, audio or training calls and does not complete the held-out E4/E5 studies.

### 6.15 Causal whole-song tool policies (v35; complete development experiment)

The shared executor provides inspect_context, check_block, commit_block and finish. Both policies receive the same frozen base-Omni proposal catalog, historical outline, public clock and permissions, text checkpoint, seed and budgets. Each starts with an empty event prefix and composes the song using only its own committed events. Inspection reads the full catalog; checking does not commit; committing appends only the current unit; early finish or budget exhaustion is incomplete rather than a full-song success. The choreography policy adds duration-aware continuing primary movement, concurrent modulation, motif development and settled-ending instructions. This is a policy comparison within a bounded current-unit tool subset, not an end-to-end held-out E3 comparison or proof of autonomous global search/audio reinspection.

The first completed song uses 27 units per policy, and each trajectory stops after 27 direct commits plus finish (28 calls). Neither requests inspection or an explicit check. Its original requests, raw tool responses and all committed prefixes have been reconstructed. Generic/choreography produce 62/67 events, with 23/11 action sentences respectively appearing verbatim in earlier committed history; none of these complete copied sentences occurs verbatim within the current local raw proposals. This is a source-intersection observation, not identification of a unique causal source or a music-quality advantage. Both still overload short events: at 1.8 seconds, the generic phrase lists floor sliding, shoulder waves, arm thrusts, a turn and a lunge, while the choreography phrase reaches a freeze and then restarts with body rolling and isolations. A narrower policy has not eliminated the content problem.

The first two states consume 56 calls, 702,914 input and 27,925 output tokens. The full six-state/three-song inventory subsequently completes with 158 calls, 2,022,028 input and 83,393 output tokens; every actual request, tool response and committed prefix is reconstructed. Rock generic/choreography produce 65/69 events with 25/20 prior full-sentence copies, while acoustic produces 52/55 events with 9/10 copies. Every trajectory uses only direct commits and finish, so this run does not demonstrate an adaptive inspection/checking advantage. Direct assistant review of all 370 events finds persistent movement overload, repeated long chains and restarts after stated stillness; all six are unsuitable as final choreography text in that model opinion. Musical fit and actual execution remain unknown, with no human ratings or formal E3 completion. Shared historical outline/candidate acquisition costs remain distinct from new execution costs.

At the user's request, the assistant separately reviews all 24 v31 editing states using actual before/after instructions and neighbors, producing explicit per-goal/protection/boundary declarations marked model_judge, nonblind and text-only. Under its retrospective classification of eight valid unmet-change tasks, model-declared joint passes are 5/8 for generic and 4/8 for choreography; each method also has three unchanged-source controls and one unresolved-authoring rock-ending case. These are neither independent human joint-success rates nor calibrated model-quality estimates. Original human-pending records remain unchanged.

The follow-up v36 mechanism experiment removes action text from all four history paths while keeping the authoritative score outside the model, fixed current proposals/outline/clock, policy, seed, checkpoint and tools. New full-mode code reconstructs all 79 original choreography-full requests exactly, allowing those outputs to serve as controls without resampling. Three metadata-only songs are launched from empty histories with their own causal trajectories; completion and quality effects are not assumed before actual outputs are audited. The intervention trades visible historical semantics for less copying exposure and cannot infer physical continuity from metadata alone.

## 7. Discussion and Limitations

A text score allows controlled editing without requiring motion reconstruction, but usefulness must be measured rather than asserted. Open-vocabulary action paragraphs may compress several movements into a short event. Physical feasibility, contact, facing and support cannot be verified from empty states or text alone. Music-language proposals and structural detectors can be wrong; an apparently musically descriptive phrase may simply repeat a generic rhythmic template.

The schedule menu trades valid execution for expressive freedom and currently lacks verified beat-aware or phrase-aware timing. Whole-body context may be redundant, and history can increase cost without improving perceived coherence. The small, repeatedly used development collection does not support generalization, and one decode per arm cannot distinguish stable intervention effects from service variability. Scope-preserving edits are simpler than arbitrary authoring requests. The current text task excludes event addition/deletion, open-ended retiming and uncontrolled references.

Our evaluation separates mechanical validity, score quality and editing preservation. A blinded review is necessary for claims about music or usability; optional rating infrastructure is not a substitute for participation. Public evidence omits sensitive private bindings and evaluation goals. Data licenses and participant consent must be resolved before releasing a benchmark or study materials. A final manuscript should make the development/held-out boundary explicit and avoid invalid cross-modal comparisons.

## 8. Conclusion (Development-Only Version)

We formulate music-conditioned choreography planning as an editable, auditable text-event score and implement a harness that separates proposal generation, whole-song design, causal composition and constrained execution. The current development evidence verifies complete executions and controlled history inputs, while exposing limitations of temporal menus and unresolved perceptual evaluation. The intended submission contribution is the supported planning/control trade-off under matched baselines, not a blanket claim of professional choreography quality. Final conclusions must follow the completed independent evaluation, including neutral or negative module effects.

## Submission Checklist — Not Additional Claimed Results

- [x] Implemented task/representation, run contracts, explicit validity and edit boundaries.
- [x] Real full-song execution and actual frozen-outline history control.
- [x] Manuscript skeleton with methods, primary neighboring work and supported development numbers.
- [x] Verified v17 baseline and edit tables; session caveats retained; semantic ratings still absent.
- [x] Real v21/v22 revision comparisons with targeted/random/text-only controls, retained failures, immutable internal decisions and an offline timing-guard recheck.
- [x] Initial-writing evidence: task/representation, runnable baselines, original-vs-edit controls, bounded revision trace, costs and explicit limitations. This is not submission-ready evidence of benchmark quality or method superiority.
- [ ] Matched direct-generation baseline and cost/permission description.
- [ ] Actual independent quality/semantic reviews with calibration and disagreements.
- [ ] Frozen, source-isolated test collection and decoding-variability analysis.
- [ ] Related-work expansion and a final supported novelty statement.
- [ ] Final venue format, author information, consent/release policy and claim audit.

## Primary Sources Checked for Positioning

1. *Music-to-Dance Generation via Atomic Movements*, arXiv:2607.13978. Abstract/primary HTML checked for symbolic atomic movement planning and completion.
2. *CustomDance: Customized 3D Dance Generation with Coarse-to-Fine Human-Centered Interactive Control*, arXiv:2608.06722. Primary HTML checked for multimodal music analysis, JSON anchor/cue planning, phrase authoring and local motion editing.
3. *DanceEditor: Towards Iterative Editable Music-driven Dance Generation with Open-Vocabulary Descriptions*, arXiv:2508.17342. Primary paper excerpt checked for iterative music/text-conditioned dance editing.

This list is an initial positioning set, not an exhaustive survey. Populate full bibliographic metadata from original records during manuscript formatting; do not invent author lists or publication venues.
