# ChoreoBench: Auditable Music-to-Text Choreography Planning and Bounded Revision

*Preliminary task-and-baseline manuscript. Empirical benchmark quality calibration and held-out evaluation remain incomplete.*

**Working manuscript — 2026-11-09.** Latest research direction is benchmark-first: task definitions, source-isolated task/annotation data, calibrated evaluation and baseline comparisons; the harness is a baseline, not the entire contribution. The present draft is still a method-oriented starting point. New nine-plan review material and task inventories do not constitute completed benchmark data or quality results. Refer to PROGRESS_20261109_zh.md for the current evidence boundary. This is a submission-oriented draft, not a completed empirical paper. Numerical results below are limited to verified engineering/development experiments. No quality improvement, statistical significance, human-study result or first-of-its-kind claim is asserted. Authors, venue and final manuscript format remain unset. Square-bracketed experiment placeholders are not results.

## Abstract

Music-conditioned dance systems often deliver motion or video, while an author may first need an interpretable score that specifies what to do, when to do it, and which parts can be revised. We investigate long-form music-conditioned choreography planning as the generation of an editable, open-vocabulary event score rather than a measured motion sequence. Our harness separates local audio-conditioned movement proposals, whole-song motif design, causal block composition, and deterministic constraint execution. The representation uses a canonical audio clock, explicit body scopes, motif references, and scoped field edits; private evaluation targets are not exposed to generators. We distinguish mechanically verifiable validity from musical and choreographic quality. In a three-song development study, six complete scores pass structural checks, and a frozen-outline history comparison verifies identical shared conditioning while retaining all requests and costs. A matched-schedule independent-block baseline also completes all three songs, showing that validity and coverage alone do not isolate a planning benefit. A three-task scoped-edit pilot changes each selected action while preserving all other event fields and exactly recovering the source through undo. Real bounded-revision pilots further expose a truncated diagnostic and internally accepted timing claims that conflict with the fixed-time contract. The current experiments establish executable, auditable workflows and inspectable failure controls, but do not yet establish perceptual quality gains. We identify a concrete limitation of near-equal scheduling menus and outline a held-out evaluation of musical relevance, long-range organization, edit satisfaction and preservation. This draft treats a successful system trace as necessary evidence of execution, not sufficient evidence of effective choreography.

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
