"""Module S: music-structure-driven composition (same evidence, better binding).

The outline call is the existing one, extended with an acoustic summary and a
small machine-checkable ``section_metadata`` side channel, so the public plan
schema is unchanged. Each block request additionally carries the current
section's requirement and the acoustic summary of the blocks it covers.
"""
import copy
import json

from .io import object_sha
from .schema import identifier, keys, text
from .coordination import LIMITS, _bounded_text, _guard_request, _messages
from .music_structure_summary import action_profile as _action_profile
from .music_structure_summary import compact as compact_summary
from .music_structure_summary import direction as _acoustic_direction
from .timing import event_clock as _event_clock

DEVELOPMENT = ("introduce", "repeat", "vary", "contrast", "resolve", "unknown")
QUANTITY = ("increase", "decrease", "steady", "unknown")
TRANSITION = ("continue", "return", "shift", "unknown")
BODY_TOKENS = ("arm", "leg", "torso", "head", "knee", "foot", "feet", "hip", "shoulder", "wrist",
               "elbow", "spine", "chest", "hand", "neck")

INSTRUCTION = (
    "Design the whole-song plan from the supplied PUBLIC local proposals and the acoustic music summary. "
    "Return exactly {motifs, sections, section_metadata}. motifs and sections keep the original schema: "
    "motifs=[{motif_id,description}], sections=[{section_id,intent,motif_ids}] for every public section in order. "
    "section_metadata has one entry per section: {section_id, development, quantity_change, "
    "transition_from_previous:{kind,description}, music_blocks, evidence, uncertainty}. "
    "development is one of introduce/repeat/vary/contrast/resolve/unknown; quantity_change is "
    "increase/decrease/steady/unknown; transition kind is continue/return/shift/unknown and must be unknown "
    "for the first section. music_blocks lists block_id values from the supplied summary that this section uses. "
    "evidence lists short {source,note} items; source must be music_block or local_caption. "
    "Each intent must state a concrete action requirement: name a declared motif_id or an explicit body part, "
    "not only adjectives such as 'more powerful' or 'echo the music'. "
    "Map acoustic evidence to action: higher relative energy or onset rate may justify more/faster movement; "
    "a fall may justify sustained or reduced movement; a possible_recurrence may justify returning to a motif, "
    "allowed to change direction, amplitude, level or concurrency. Not every section has to add movement; "
    "sustained, sparse and closing sections are allowed. "
    "The summary's repeat/contrast entries are chroma-similarity guesses, not detected sections; never claim "
    "chorus/verse, a musical phrase, beat synchronization or measured audio energy used as action intention. "
    "Also return block_guidance with exactly one entry per planning unit: "
    "{unit_id, quantity_change, motif_id}. quantity_change is increase/steady/decrease and must follow the "
    "overlapping acoustic blocks (rising energy or onset rate -> increase, falling -> decrease, otherwise steady); "
    "motif_id must be a motif allowed for that unit's section or null. "
    "If evidence is insufficient, use steady and say so in uncertainty. At most eight motifs. "
    "A section that spans three or more planning units must name at least TWO distinct motifs in motif_ids, "
    "so the theme can be introduced, varied or contrasted instead of collapsed into one repeated phrase. "
    "Give at most TWO evidence items per section and keep each note under 240 characters; do not "
    "enumerate the local proposals one by one."
)


def outline_request(song, options, units, summary, cfg, require_guidance=True, style_prior=None):
    from .coordination import _candidates
    payload = dict(
        song=dict(song_id=song["song_id"], brief=song["brief"], sample_rate=song["audio"]["sample_rate"],
                  num_samples=song["audio"]["num_samples"]),
        planning_units=[{k: u[k] for k in ("unit_id", "section_id", "start", "end", "kind")}
                        for u in units["units"]],
        public_sections=[{k: s[k] for k in ("section_id", "start", "end", "label")}
                         for s in song["music"]["sections"]],
        music_summary=dict(blocks=summary["blocks"], relations=summary["relations"],
                           parameters=summary["parameters"],
                           functional_labels_verified=False, beat_or_meter_assumed=False),
        local_proposals=_candidates(options, char_limit=cfg.get("outline_proposal_char_limit"),
                                    per_window=cfg.get("outline_candidates_per_window")),
        local_proposals_are_unverified=True,
        proposal_budget=dict(char_limit=cfg.get("outline_proposal_char_limit"),
                             candidates_per_window=cfg.get("outline_candidates_per_window") or 1,
                             note="Deterministic input budget; not extra evidence."),
        evidence_policy=dict(
            summary_is_acoustic_only=True,
            unknown_is_expected=True,
            never_claim=["chorus/verse detection", "musical phrase", "beat synchronization", "measured audio energy as intention"]),
    )
    if style_prior is not None:
        payload["style_prior"] = style_prior.signal(song["song_id"])
    request = _guard_request(song["song_id"] + "/music-outline", cfg,
                             _messages(payload, INSTRUCTION), "music_outline", cfg["outline_max_output_tokens"])
    required_fields = ["motifs", "sections", "section_metadata"]
    if require_guidance:
        required_fields.append("block_guidance")
    request["response_schema"] = dict(
        type="object", additionalProperties=False, required=required_fields,
        properties=dict(
            motifs=dict(type="array", maxItems=8, items=dict(
                type="object", additionalProperties=False, required=["motif_id", "description"],
                properties=dict(motif_id=dict(type="string", minLength=1, maxLength=32),
                                description=dict(type="string", minLength=1, maxLength=240)))),
            sections=dict(type="array", items=dict(
                type="object", additionalProperties=False, required=["section_id", "intent", "motif_ids"],
                properties=dict(section_id=dict(type="string"), intent=dict(type="string", maxLength=600),
                                motif_ids=dict(type="array", maxItems=8, items=dict(type="string"))))),
            section_metadata=dict(type="array", items=dict(
                type="object", additionalProperties=False,
                required=["section_id", "development", "quantity_change", "transition_from_previous",
                          "music_blocks", "evidence", "uncertainty"],
                properties=dict(
                    section_id=dict(type="string"),
                    development=dict(type="string", enum=list(DEVELOPMENT)),
                    quantity_change=dict(type="string", enum=list(QUANTITY)),
                    transition_from_previous=dict(type="object", additionalProperties=False,
                                                  required=["kind", "description"],
                                                  properties=dict(kind=dict(type="string", enum=list(TRANSITION)),
                                                                  description=dict(type="string", maxLength=240))),
                    music_blocks=dict(type="array", maxItems=8, items=dict(type="string")),
                    evidence=dict(type="array", maxItems=2, items=dict(
                        type="object", additionalProperties=False, required=["source", "note"],
                        properties=dict(source=dict(type="string", enum=["music_block", "local_caption"]),
                                        note=dict(type="string", maxLength=240)))),
                    uncertainty=dict(type="string", maxLength=240)))),
            block_guidance=dict(type="array", items=dict(
                type="object", additionalProperties=False,
                required=["unit_id", "quantity_change", "motif_id"],
                properties=dict(unit_id=dict(type="string"),
                                quantity_change=dict(type="string", enum=list(QUANTITY)),
                                motif_id=dict(type="string"))))))
    return request


def read_outline(response, song, summary, units, require_guidance=True):
    from .baseline import ModelOutputError
    from .providers import parse_json_response
    try:
        value = parse_json_response(response)
        keys(value, ("motifs", "sections", "section_metadata"), optional=("block_guidance",),
             where="music outline")
        motifs = {}
        if not isinstance(value["motifs"], list) or len(value["motifs"]) > LIMITS["motifs"]:
            raise ValueError("Outline requires at most eight motifs")
        for motif in value["motifs"]:
            keys(motif, ("motif_id", "description"), where="outline motif")
            identifier(motif["motif_id"], "motif_id")
            _bounded_text(motif["description"], "motif.description", LIMITS["design_characters"])
            if motif["motif_id"] in motifs:
                raise ValueError("Duplicate outline motif")
            motifs[motif["motif_id"]] = motif["description"]
        expected = song["music"]["sections"]
        for key, label in (("sections", "sections"), ("section_metadata", "section_metadata")):
            if not isinstance(value[key], list) or len(value[key]) != len(expected):
                raise ValueError("Outline must cover every public section once: " + label)
        block_ids = {b["block_id"] for b in summary["blocks"]}
        warnings = []
        unit_counts = {}
        for unit in units["units"]:
            unit_counts[unit["section_id"]] = unit_counts.get(unit["section_id"], 0) + 1
        sections = []
        for index, (entry, meta, source) in enumerate(zip(value["sections"], value["section_metadata"], expected)):
            keys(entry, ("section_id", "intent", "motif_ids"), where="outline section")
            keys(meta, ("section_id", "development", "quantity_change", "transition_from_previous",
                        "music_blocks", "evidence", "uncertainty"), where="section_metadata")
            if entry["section_id"] != source["section_id"] or meta["section_id"] != source["section_id"]:
                raise ValueError("Outline cannot rename, omit or reorder public sections")
            _bounded_text(entry["intent"], "section.intent", LIMITS["design_characters"])
            ids = entry["motif_ids"]
            if not isinstance(ids, list):
                raise ValueError("motif_ids must be a list")
            declared = [i for i in ids if isinstance(i, str) and i in motifs]
            dropped = [i for i in ids if i not in declared]
            if len(set(declared)) != len(declared):
                raise ValueError("motif_ids must not repeat a declared motif")
            if dropped:
                # Recorded degradation, not a silent repair: the section keeps only
                # motifs that were actually declared, and loses its assertion if none remain.
                warnings.append(dict(section_id=source["section_id"], dropped_motif_ids=dropped,
                                     note="undeclared motif ids dropped; design downgraded"))
                ids = declared
            if meta["development"] != "unknown" and len(ids) == 0:
                meta["development"] = "unknown"
                meta["uncertainty"] = (meta.get("uncertainty") or "") + " [design downgraded: no declared motif]"
                warnings.append(dict(section_id=source["section_id"], downgraded="development->unknown"))
            if (unit_counts.get(source["section_id"], 0) >= 3 and meta["development"] != "unknown"
                    and len(ids) < 2):
                warnings.append(dict(section_id=source["section_id"],
                                     note="section spans >=3 blocks with <2 motifs; quota will cycle what exists"))
            lowered = entry["intent"].lower()
            if not any(m in entry["intent"] for m in ids) and not any(t in lowered for t in BODY_TOKENS):
                raise ValueError("section intent must name a motif or an explicit body part, not only adjectives")
            if meta["development"] not in DEVELOPMENT:
                raise ValueError("Unknown development value")
            if meta["quantity_change"] not in QUANTITY:
                raise ValueError("Unknown quantity_change value")
            transition = meta["transition_from_previous"]
            keys(transition, ("kind", "description"), where="transition_from_previous")
            if transition["kind"] not in TRANSITION:
                raise ValueError("Unknown transition kind")
            if index == 0 and transition["kind"] != "unknown":
                raise ValueError("The first section has no previous section; transition must be unknown")
            blocks = meta["music_blocks"]
            if (not isinstance(blocks, list) or any(not isinstance(b, str) or b not in block_ids for b in blocks)
                    or len(set(blocks)) != len(blocks)):
                raise ValueError("music_blocks must reference existing summary block ids")
            evidence = meta["evidence"]
            if not isinstance(evidence, list) or not evidence or len(evidence) > 2:
                raise ValueError("Every section needs one or two evidence entries")
            for item in evidence:
                keys(item, ("source", "note"), where="section evidence")
                if item["source"] not in ("music_block", "local_caption"):
                    raise ValueError("Unknown evidence source")
                _bounded_text(item["note"], "evidence.note", 240)
            _bounded_text(meta["uncertainty"], "uncertainty", LIMITS["design_characters"])
            sections.append(dict(section_id=source["section_id"], intent=entry["intent"], motif_ids=list(ids),
                                 metadata=copy.deepcopy(meta)))
        guidance = value.get("block_guidance")
        unit_ids = [u["unit_id"] for u in units["units"]]
        if not require_guidance:
            if guidance is not None:
                warnings.append(dict(note="model block_guidance ignored; harness derives it from acoustics"))
            return dict(motifs=value["motifs"], sections=sections, block_guidance=[],
                        outline_warnings=warnings)
        if not require_guidance and guidance is None:
            # The harness derives the block requirement from acoustics; no model guidance needed.
            return dict(motifs=value["motifs"], sections=sections, block_guidance=[],
                        outline_warnings=warnings)
        if not isinstance(guidance, list) or [g.get("unit_id") for g in guidance] != unit_ids:
            raise ValueError("block_guidance must cover every planning unit in order")
        section_of = {u["unit_id"]: u["section_id"] for u in units["units"]}
        allowed = {s["section_id"]: set(s["motif_ids"]) for s in sections}
        clean = []
        for item in guidance:
            keys(item, ("unit_id", "quantity_change", "motif_id"), where="block_guidance")
            if item["quantity_change"] not in QUANTITY:
                raise ValueError("Unknown block quantity_change")
            motif = item["motif_id"]
            if motif is not None and (not isinstance(motif, str)
                                      or motif not in allowed[section_of[item["unit_id"]]]):
                # Recorded degradation: keep the block, drop the unusable motif.
                warnings.append(dict(section_id=section_of[item["unit_id"]], unit_id=item["unit_id"],
                                     dropped_block_motif=motif))
                motif = None
            clean.append(dict(item, motif_id=motif))
        return dict(motifs=value["motifs"], sections=sections, block_guidance=clean,
                    outline_warnings=warnings)
    except (ValueError, TypeError, KeyError) as error:
        raise ModelOutputError(str(error)) from error


def assigned_motifs(units, index, design):
    """Deterministic per-block motif quota: cycling assignment, last block returns home.

    The block decoder only ever sees the assigned motif, so a section cannot
    collapse into one repeated phrase even if the model prefers it.
    """
    members = [i for i, u in enumerate(units) if u["section_id"] == design["section_id"]]
    if not members or not design["motif_ids"]:
        return []
    position = members.index(index)
    primary = list(design["motif_ids"])
    if len(primary) == 1:
        return primary
    if position == len(members) - 1 and len(members) >= 3:
        return [primary[0]]
    return [primary[position % len(primary)]]


def _strip_null_motif(node):
    """Remove the null option so the decoder can only emit an assigned motif."""
    if isinstance(node, dict):
        if isinstance(node.get("motif_id"), dict) and isinstance(node["motif_id"].get("enum"), list):
            node["motif_id"]["enum"] = [m for m in node["motif_id"]["enum"] if m is not None]
        for value in node.values():
            _strip_null_motif(value)
    elif isinstance(node, list):
        for value in node:
            _strip_null_motif(value)
    return node


def previous_motif_occurrence(events, motif_id):
    """Last committed event carrying this theme, for a variation requirement."""
    if not motif_id:
        return None
    earlier = [e for e in events if e.get("motif_id") == motif_id]
    if not earlier:
        return None
    last = earlier[-1]
    return dict(motif_id=motif_id, action=last["action"], start=last["start"], end=last["end"])


def unit_request(state, unit, summary, cfg):
    """Shared unit request plus the section requirement and overlapping acoustics."""
    from .coordination import _unit_request
    base = _unit_request(state.source["song"], state.source["options"], state.source["units"],
                         unit, {"motifs": state.outline["motifs"],
                                "sections": [{k: s[k] for k in ("section_id", "intent", "motif_ids")}
                                             for s in state.outline["sections"]]},
                         state.plan, state.source["config"], state.source.get("style_prior"))
    payload = json.loads(base["messages"][-1]["content"])
    design = next(s for s in state.outline["sections"] if s["section_id"] == unit["section_id"])
    units = state.source["units"]["units"]
    members = [i for i, u in enumerate(units) if u["section_id"] == unit["section_id"]]
    position = members.index(next(i for i, u in enumerate(units) if u["unit_id"] == unit["unit_id"]))
    units = state.source["units"]["units"]
    index = next(i for i, u in enumerate(units) if u["unit_id"] == unit["unit_id"])
    guidance = next((g for g in (state.outline.get("block_guidance") or [])
                     if g["unit_id"] == unit["unit_id"]), None)
    if cfg.get("acoustic_guidance"):
        # The harness, not the model, maps the acoustic summary to the block
        # requirement; the model only writes actions for it.
        direction = _acoustic_direction(summary, unit["start"], unit["end"])
        guidance = dict(unit_id=unit["unit_id"], quantity_change=direction,
                        motif_id=(guidance or {}).get("motif_id"))
        guidance["source"] = "deterministic_acoustic_direction"
        guidance["action_profile"] = _action_profile(direction)
    assigned = ([guidance["motif_id"]] if guidance and guidance.get("motif_id")
                else assigned_motifs(units, index, design))
    previous_occurrence = (previous_motif_occurrence(state.plan["events"], assigned[0])
                           if cfg.get("motif_variation") and assigned else None)
    payload.update(
        previous_motif_occurrence=previous_occurrence,
        section_requirement=dict(section_id=design["section_id"], intent=design["intent"],
                                 metadata=copy.deepcopy(design["metadata"])),
        assigned_motifs=assigned,
        block_guidance=guidance,
        section_position=dict(index=position + 1, total=len(members)),
        music_evidence=compact_summary(summary, unit["start"], unit["end"]),
        previous_committed_action=(None if not state.plan["events"] else
                                   {k: state.plan["events"][-1][k] for k in ("event_id", "start", "end", "track",
                                                                              "action", "motif_id")}),
        motif_policy=dict(assigned_for_this_block=assigned,
                          note="Only the assigned motif(s) are decodable for this block; this is an engineering quota, not a musical claim."),
        mapping_policy=dict(
            higher_energy_may_justify="more or faster movement",
            falling_energy_may_justify="sustained or reduced movement",
            possible_recurrence_may_justify="returning to a motif with changed direction/amplitude/level/concurrency",
            no_section_must_add_movement=True,
            unknown_physical_state=True))
    system = base["messages"][0]["content"] + (
        (" previous_motif_occurrence is the last time the assigned theme appeared. Treat this block as a "
         "return of that theme: change at least one of direction, level (high/mid/low), concurrency or which "
         "limb leads, and do not reuse its sentence." if previous_occurrence else "") +
        " Honour section_requirement: its intent and metadata are binding design targets. "
        "Use music_evidence only as acoustic measurements; its repeat/contrast entries are similarity guesses, "
        "not detected sections. Do not add adjectives instead of actions: every change must be expressed as a "
        "concrete body movement with timing. Do not pad the block to look busy; sustained and sparse blocks are allowed. "
        "Do not invent support-foot, facing or contact state. "
        "block_guidance.action_profile states exactly how much movement this block must contain "
        "(event_count and mean_duration); follow it literally and never add filler events to look busy."
    )
    from .coordination import unit_response_schema
    if not cfg.get("motif_quota", True):
        assigned = []
    if assigned:
        base["response_schema"] = unit_response_schema(
            unit, assigned, [c["call_id"] for c in payload["local_proposals"]],
            max_events=cfg["max_events_per_unit"],
            clock=_event_clock(unit, state.plan["sample_rate"], state.source["config"]),
            max_request_characters=cfg["max_request_characters"])
        _strip_null_motif(base["response_schema"])
        payload["output_contract"]["allowed_motif_ids"] = list(assigned)
    request = dict(base, messages=[dict(role="system", content=system),
                                   dict(role="user", content=json.dumps(payload, ensure_ascii=False,
                                                                        sort_keys=True, allow_nan=False))])
    from .request_budget import guard_request
    if sum(len(m["content"]) for m in request["messages"]) > cfg["max_request_characters"]:
        from .baseline import ModelOutputError
        raise ModelOutputError("Music-structured unit input exceeds max_request_characters; no call sent")
    return guard_request(request, cfg)


def contribution(outline):
    metas = [s["metadata"] for s in outline["sections"]]
    return dict(sections=len(metas), development={m["development"] for m in metas},
                quantity={m["quantity_change"] for m in metas},
                music_blocks_used=sum(len(m["music_blocks"]) for m in metas),
                outline_sha256=object_sha(dict(motifs=outline["motifs"],
                                               sections=[{k: s[k] for k in ("section_id", "intent", "motif_ids")}
                                                         for s in outline["sections"]])))
