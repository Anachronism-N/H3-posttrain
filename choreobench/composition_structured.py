"""Structured section design plus targeted revision (new composition method).

This is the fourth adaptation. It reuses the same evidence, permissions, clocks
and total budget as the other methods; what changes is that the whole-song
outline must state, per public section, which primary motifs it uses, how they
develop, how the movement amount changes and how the section connects to the
previous one -- each field either backed by an explicitly listed public
evidence source or declared unknown.

Block generation receives that requirement plus the previous block's last
action. A deterministic scan then reports only four kinds of concrete problem;
when none is found the draft is committed immediately with no extra call and no
generic reflection. At most one targeted revision is allowed, and a reasonable
motif recurrence is never treated as a defect.
"""
import copy
import json

from .composition_controller_v2 import CompositionStateV2
from .io import object_sha
from .schema import identifier, keys, text
from .coordination import COMMON, LIMITS, _bounded_text, _candidates, _guard_request, _messages

OUTLINE_SCHEMA = "choreobench.structured_outline.v1"
DEVELOPMENT = ("introduce", "repeat", "vary", "contrast", "resolve", "unknown")
QUANTITY = ("increase", "decrease", "steady", "unknown")
TRANSITION = ("continue", "return", "shift", "unknown")
EVIDENCE_SOURCES = (
    "local_caption_density",   # public per-window caption count/length, not audio energy
    "public_beat_density",     # public beat predictions inside the section
    "public_downbeat_density",
    "public_section_span",     # section duration only
    "brief",
)
ISSUE_CODES = (
    "missing_required_motif",
    "action_timing_contradiction",
    "missing_transition",
    "mechanical_repetition",
)
MAX_ISSUES = 3
CLAUSE_MARKERS = (";", " then ", ", then ", " followed by ", " and then ")
MIN_SECONDS_PER_CLAUSE = 0.5

INSTRUCTION = (
    "Design a whole-song structured plan over the supplied PUBLIC music evidence only. "
    "Return exactly {motifs:[{motif_id,description}],sections:[...]}. Each section needs exactly "
    "section_id, primary_motifs, development, quantity_change, transition_from_previous, evidence, uncertainty. "
    "primary_motifs is a list of motif_id values declared in motifs (may be empty only when development='unknown'). "
    "development is one of introduce/repeat/vary/contrast/resolve/unknown: how the primary motifs are used here. "
    "quantity_change is increase/decrease/steady/unknown: the planned change in movement amount, NOT measured audio energy. "
    "transition_from_previous is {kind,description}, kind one of continue/return/shift/unknown; the first section must be unknown. "
    "evidence lists {source,note}; source must be one of " + ", ".join(EVIDENCE_SOURCES) + ". "
    "Every non-unknown field must be supported by at least one evidence entry whose source is actually available in this request. "
    "If a section's label is unknown, no beat exists, or the public captions do not support a claim, use unknown and say so in uncertainty. "
    "Never claim chorus/verse detection, a musical phrase, beat synchronization or audio energy you were not given. "
    "Assign each supplied section exactly once, in order. At most eight motifs. Do not output events or a score."
)


def section_evidence(song, options, section):
    """Deterministic public evidence summary for one section.

    Caption density is a count of supplied texts, not a measurement of audio
    energy; beats are tool predictions. Nothing here is labelled as truth.
    """
    span = section["end"] - section["start"]
    beats = [b for b in song["music"].get("beats") or [] if section["start"] <= b < section["end"]]
    downbeats = [d for d in song["music"].get("downbeats") or [] if section["start"] <= d < section["end"]]
    captions = 0
    chars = 0
    for row in options["windows"]:
        window = row["window"]
        if window["observation_start"] < section["end"] and window["observation_end"] > section["start"]:
            captions += len(row["candidates"])
            chars += sum(len(c.get("raw_text", "")) for c in row["candidates"])
    available = {"public_section_span", "brief"}
    if beats:
        available.add("public_beat_density")
    if downbeats:
        available.add("public_downbeat_density")
    if captions:
        available.add("local_caption_density")
    return dict(section_id=section["section_id"], label=section["label"],
                duration_seconds=round(span / 16000.0, 2),
                beat_count=len(beats), downbeat_count=len(downbeats), local_caption_count=captions,
                mean_caption_characters=(round(chars / captions, 1) if captions else None),
                label_is_tool_prediction=True, audio_energy_measured=False,
                available_evidence_sources=sorted(available))


def outline_request(song, options, units, cfg, style_prior=None):
    sections = song["music"]["sections"]
    payload = dict(
        song=dict(song_id=song["song_id"], brief=song["brief"], sample_rate=song["audio"]["sample_rate"],
                  num_samples=song["audio"]["num_samples"]),
        planning_units=[{k: u[k] for k in ("unit_id", "section_id", "start", "end", "kind")} for u in units["units"]],
        public_sections=[{k: s[k] for k in ("section_id", "start", "end", "label")} for s in sections],
        section_evidence=[section_evidence(song, options, s) for s in sections],
        local_proposals_are_unverified=True,
        evidence_policy=dict(
            unknown_is_expected=True,
            never_claim=["chorus/verse detection", "musical phrase or boundary", "beat synchronization",
                         "measured audio energy", "verified physical state"],
            energy_proxy_note="local caption density counts supplied texts; it is not measured audio energy."),
    )
    if style_prior is not None:
        payload["style_prior"] = style_prior.signal(song["song_id"])
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return _guard_request(song["song_id"] + "/structured/outline", cfg,
                          _messages(payload, INSTRUCTION), "structured_outline", cfg["outline_max_output_tokens"])


def read_outline(response, song, options):
    """Validate the structured outline, including evidence honesty."""
    from .providers import parse_json_response
    from .baseline import ModelOutputError
    try:
        value = parse_json_response(response)
        keys(value, ("motifs", "sections"), where="structured outline")
        motifs = {}
        if not isinstance(value["motifs"], list) or len(value["motifs"]) > LIMITS["motifs"]:
            raise ValueError("Structured outline requires at most eight motifs")
        for motif in value["motifs"]:
            keys(motif, ("motif_id", "description"), where="outline motif")
            identifier(motif["motif_id"], "motif_id")
            _bounded_text(motif["description"], "motif.description", LIMITS["design_characters"])
            if motif["motif_id"] in motifs:
                raise ValueError("Duplicate outline motif")
            motifs[motif["motif_id"]] = motif["description"]
        expected = song["music"]["sections"]
        if not isinstance(value["sections"], list) or len(value["sections"]) != len(expected):
            raise ValueError("Structured outline must assign every public section in order")
        sections = []
        for index, (entry, source) in enumerate(zip(value["sections"], expected)):
            keys(entry, ("section_id", "primary_motifs", "development", "quantity_change",
                         "transition_from_previous", "evidence", "uncertainty"), where="outline section")
            if entry["section_id"] != source["section_id"]:
                raise ValueError("Structured outline cannot rename, omit or reorder public sections")
            if entry["development"] not in DEVELOPMENT:
                raise ValueError("development must be one of " + "/".join(DEVELOPMENT))
            if entry["quantity_change"] not in QUANTITY:
                raise ValueError("quantity_change must be one of " + "/".join(QUANTITY))
            primary = entry["primary_motifs"]
            if (not isinstance(primary, list) or any(not isinstance(m, str) or m not in motifs for m in primary)
                    or len(set(primary)) != len(primary)):
                raise ValueError("primary_motifs must be declared, unique motifs")
            if entry["development"] != "unknown" and not primary:
                raise ValueError("A non-unknown development must name at least one primary motif")
            transition = entry["transition_from_previous"]
            keys(transition, ("kind", "description"), where="transition_from_previous")
            if transition["kind"] not in TRANSITION:
                raise ValueError("transition kind must be one of " + "/".join(TRANSITION))
            _bounded_text(transition["description"], "transition.description", LIMITS["design_characters"])
            _bounded_text(entry["uncertainty"], "uncertainty", LIMITS["design_characters"])
            if index == 0 and transition["kind"] != "unknown":
                raise ValueError("The first section has no previous section; transition must be unknown")
            available = section_evidence(song, options, source)["available_evidence_sources"]
            evidence = entry["evidence"]
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("Every section needs at least one evidence entry")
            for item in evidence:
                keys(item, ("source", "note"), where="section evidence")
                if item["source"] not in EVIDENCE_SOURCES:
                    raise ValueError("Unknown evidence source: " + str(item["source"]))
                if item["source"] not in available:
                    raise ValueError("Evidence source is not available for this section: " + item["source"])
                _bounded_text(item["note"], "evidence.note", 600)
            asserted = entry["development"] != "unknown" or entry["quantity_change"] != "unknown"
            if asserted and not evidence:
                raise ValueError("An asserted section design needs evidence")
            # motif_ids is the legacy alias used by the shared block validator:
            # this section may only reference its own primary motifs.
            sections.append(dict(entry, evidence=copy.deepcopy(evidence),
                                 motif_ids=list(entry["primary_motifs"])))
        return dict(motifs=value["motifs"], sections=sections)
    except (ValueError, TypeError, KeyError) as error:
        raise ModelOutputError(str(error)) from error


def _section_units(units, section_id):
    return [(index, u) for index, u in enumerate(units["units"]) if u["section_id"] == section_id]


def block_requirement(units, design, index):
    """Deterministic block role and the motifs this block is actually required to use."""
    unit = units[index]
    members = [i for i, u in enumerate(units) if u["section_id"] == unit["section_id"]]
    position = members.index(index)
    last = len(members) - 1
    role = "only" if len(members) == 1 else ("first" if position == 0 else "last" if position == last else "middle")
    required = []
    if role in ("first", "only") and design["development"] in ("introduce", "repeat", "vary"):
        required = list(design["primary_motifs"])
    elif role in ("last", "only") and design["development"] == "resolve":
        required = list(design["primary_motifs"])
    return dict(unit_id=unit["unit_id"], role=role, position=position, count=len(members),
                required_motifs=required, development=design["development"],
                quantity_change=design["quantity_change"],
                transition=copy.deepcopy(design["transition_from_previous"]),
                primary_motifs=list(design["primary_motifs"]),
                repetition_note=("Exact repeat is expected here." if design["development"] == "repeat"
                                 else "Repeat only with variation; avoid copying the previous block verbatim."))


def _previous_block_events(plan, units, index):
    if index == 0:
        return []
    previous_id = units[index - 1]["unit_id"]
    return [e for e in plan["events"] if e["event_id"].startswith(previous_id + "-")]


def _clauses(action):
    lowered = " " + action.lower() + " "
    count = 1
    for marker in CLAUSE_MARKERS:
        count += lowered.count(marker)
    return count


def _normalize(action):
    return " ".join(action.lower().split())


def detect_issues(events, unit, requirement, previous_events, sample_rate):
    """Exactly four deterministic problem kinds; no aesthetic or musical judgement."""
    issues = []
    if requirement["required_motifs"]:
        used = {e["motif_id"] for e in events}
        if not used & set(requirement["required_motifs"]):
            issues.append(dict(
                code="missing_required_motif",
                detail="This block must carry one of the section's primary motifs (%s) but none is used."
                       % ", ".join(requirement["required_motifs"]),
                evidence=dict(required=requirement["required_motifs"], used=sorted(m for m in used if m))))
    for event in events:
        seconds = (event["end"] - event["start"]) / float(sample_rate)
        clauses = _clauses(event["action"])
        if clauses >= 3 and seconds < clauses * MIN_SECONDS_PER_CLAUSE:
            issues.append(dict(
                code="action_timing_contradiction",
                detail="Event %s describes %d clauses in %.2f s." % (event["event_id"], clauses, seconds),
                evidence=dict(event_id=event["event_id"], clauses=clauses, seconds=round(seconds, 3))))
            break
    if previous_events and requirement["transition"]["kind"] in ("continue", "return"):
        previous_end = max(e["end"] for e in previous_events)
        gap = (unit["start"] - previous_end) / float(sample_rate)
        if unit["start"] - previous_end >= sample_rate:
            issues.append(dict(
                code="missing_transition",
                detail="The section asks to %s from the previous block, but this block starts %.2f s after it ends."
                       % (requirement["transition"]["kind"], gap),
                evidence=dict(previous_end=previous_end, unit_start=unit["start"], gap_seconds=round(gap, 3))))
    if previous_events and requirement["development"] != "repeat":
        previous_text = _normalize(previous_events[-1]["action"])
        for event in events:
            if _normalize(event["action"]) == previous_text:
                issues.append(dict(
                    code="mechanical_repetition",
                    detail="Event %s repeats the previous block's action verbatim, but this section does not ask for exact repetition."
                           % event["event_id"],
                    evidence=dict(event_id=event["event_id"], previous_event_id=previous_events[-1]["event_id"])))
                break
    return issues[:MAX_ISSUES]


def requirement_payload(requirement, design):
    return dict(
        section_requirement=dict(design=copy.deepcopy(design)),
        block_requirement=requirement,
        previous_block_last_action=None,
        revision_policy=dict(
            commit_directly_when_no_issue=True,
            allowed_revision_reasons=list(ISSUE_CODES),
            one_revision_per_block=True,
            reasonable_recurrence_is_not_a_defect=True,
            do_not_restate_whole_design=True),
    )


class StructuredState(CompositionStateV2):
    """v2 control plus structured requirement and targeted revision."""

    def __init__(self, source, outline):
        super().__init__(source, outline)
        self.designs = {s["section_id"]: s for s in outline["sections"]}
        self.issues_reported = None
        self.blocks_with_issues = 0
        self.revision_calls = 0

    def requirement(self):
        units = self.source["units"]["units"]
        if self.index >= len(units):
            return None
        unit = units[self.index]
        design = self.designs[unit["section_id"]]
        requirement = block_requirement(units, design, self.index)
        previous = _previous_block_events(self.plan, units, self.index)
        requirement["previous_block_last_action"] = (
            None if not previous else {k: previous[-1][k] for k in ("event_id", "start", "end", "track", "action", "motif_id")})
        return requirement, design

    def detect(self, events):
        requirement, _ = self.requirement()
        previous = requirement["previous_block_last_action"]
        previous_events = [] if previous is None else [previous]
        return detect_issues(events, self.source["units"]["units"][self.index], requirement,
                             previous_events, self.plan["sample_rate"])

    def execute(self, decision):
        """Stage any draft, scan it, then commit only when no concrete issue exists."""
        from .schema import keys as _keys, text as _text
        _keys(decision, ("tool", "proposal", "reason"), where="composition tool")
        _text(decision["reason"], "tool reason")
        if len(decision["reason"]) > 1200:
            raise ValueError("Tool reason budget exceeded")
        tool, proposal = decision["tool"], decision["proposal"]
        if tool not in self.available_tools():
            raise ValueError("Tool is not allowed in the current controller state: " + str(tool))
        if tool not in ("propose_block", "revise_block", "submit_block") and proposal is not None:
            raise ValueError("Read/check/commit/finish cannot mutate draft")
        if tool == "propose_block" and self.draft is not None:
            raise ValueError("Current draft already exists; use revise_block to change it or commit_block to advance")
        if tool == "revise_block" and self.draft is None:
            raise ValueError("No current draft to revise")
        if tool == "revise_block" and self.revisions_used >= 1:
            raise ValueError("Shared one-revision-per-unit limit reached; check, commit or finish")
        if tool in ("propose_block", "revise_block", "submit_block"):
            if tool == "revise_block" or (tool == "submit_block" and self.draft is not None):
                self.revisions_used += 1
                if self.issues_reported is not None:
                    self.revision_calls += 1
            self._store(proposal)
            observation = dict(tool=tool, status="drafted", draft_sha256=object_sha(self.draft),
                               revisions_used=self.revisions_used, revision_limit=1, quality_verified=False)
            try:
                events, _ = self.validate(self.draft)
            except (ValueError, TypeError, KeyError) as error:
                observation["structural_error"] = str(error)
                return observation
            issues = self.detect(events)
            if not issues:
                committed = self._commit("structured_auto_commit", "no_detected_issue")
                committed["issue_scan"] = dict(issues=[], revision_requested=False,
                                               note="No concrete issue detected; committed without an extra call.")
                return committed
            if self.issues_reported is None:
                self.blocks_with_issues += 1
            self.issues_reported = issues
            observation.update(status="drafted_with_issues", issue_scan=dict(
                issues=issues, revision_requested=True, revision_limit=1,
                note="Revise only these enumerated problems, or commit if one is not applicable. "
                     "A reasonable motif recurrence is not a defect."))
            return observation
        return super().execute(decision)

    def available_tools(self, remaining_calls=None):
        tools = super().available_tools(remaining_calls)
        if self.draft is None or self.issues_reported is None:
            return tools
        # After the single allowed revision, only a commit decision remains.
        if self.revisions_used >= 1:
            return [t for t in tools if t in ("commit_block", "check_block")]
        return [t for t in tools if t in ("revise_block", "commit_block", "check_block")]

    def after_decision(self, before):
        if self.progress_signature() != before:
            self.no_progress_streak = 0
            return None
        if self.draft is not None and self.draft_is_valid() and self.issues_reported is not None and self.revisions_used >= 1:
            return self._commit("controller_commit", "issues_reported_once")
        return super().after_decision(before)


def unit_request(state, unit, cfg):
    """The shared unit request plus the structured requirement, same evidence and permissions."""
    from .coordination import _unit_request
    base = _unit_request(state.source["song"], state.source["options"], state.source["units"],
                         unit, state.outline, state.plan, state.source["config"], state.source.get("style_prior"))
    payload = json.loads(base["messages"][-1]["content"])
    requirement, design = state.requirement()
    payload.update(requirement_payload(requirement, design))
    payload["previous_block_last_action"] = requirement.get("previous_block_last_action")
    system = base["messages"][0]["content"] + (
        " Honour section_requirement and block_requirement: they are binding design targets derived from the "
        "same public evidence, not new audio facts. block_requirement.required_motifs must appear in this block "
        "when non-empty. Continue from previous_block_last_action without restating it. "
        "Do not pad the block with extra actions to look busy; quantity_change is a design target, not measured energy. "
        "If the previous action is unclear, do not invent its state. "
        "When an issue_scan lists concrete problems, change only what those problems require."
    )
    request = dict(base, messages=[dict(role="system", content=system),
                                   dict(role="user", content=json.dumps(payload, ensure_ascii=False,
                                                                        sort_keys=True, allow_nan=False))])
    return _reguard(request, cfg)


def _reguard(request, cfg):
    """Re-check the public request budget after adding the structured fields."""
    from .request_budget import guard_request
    if sum(len(m["content"]) for m in request["messages"]) > cfg["max_request_characters"]:
        from .baseline import ModelOutputError
        raise ModelOutputError("Structured unit input exceeds max_request_characters; no call sent")
    return guard_request(request, cfg)


def contribution(result):
    """Facts for the report; never a quality claim."""
    stats = result.get("structured_statistics") or {}
    return dict(method=result.get("method"), full_song_succeeded=result.get("full_song_succeeded"),
                calls=result.get("calls_used"), input_tokens=result.get("input_tokens"),
                output_tokens=result.get("output_tokens"), **stats)
