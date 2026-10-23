"""Read-only caption repetition/coverage audit; standard library only.

Usage: python ms_caption_audit.py --input /path/to/reports --out /other/audit.json

Source IDs and source-content hashes can link related records. Shared captions
cannot establish duplicate videos. Lexical mentions are not verified actions,
and this tool neither rewrites labels nor produces training/sample weights.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


# source_sha256 is deliberately excluded: elsewhere in this project it can
# identify implementation code, which is shared across unrelated source clips.
_HASH_FIELDS = ("input_sha256", "content_sha256", "video_sha256", "source_content_sha256")
_ID_FIELDS = ("source_id", "source_group", "video_id")
_ACTIONS = {
    "bend": r"bend(?:s|ing)?|bent", "extend": r"extend(?:s|ed|ing)?",
    "raise": r"rais(?:e|es|ed|ing)", "lower": r"lower(?:s|ed|ing)?",
    "turn": r"turn(?:s|ed|ing)?", "twist": r"twist(?:s|ed|ing)?",
    "lean": r"lean(?:s|ed|ing)?", "move": r"mov(?:e|es|ed|ing)",
    "translate": r"translat(?:e|es|ed|ing)", "rotate": r"rotat(?:e|es|ed|ing)",
    "swing": r"swing(?:s|ing)?|swung", "step": r"step(?:s|ped|ping)?",
    "jump": r"jump(?:s|ed|ing)?", "crouch": r"crouch(?:es|ed|ing)?",
    "kick": r"kick(?:s|ed|ing)?", "clap": r"clap(?:s|ped|ping)?",
    "walk": r"walk(?:s|ed|ing)?", "run": r"run(?:s|ning)?|ran",
}
_ACTION_PATTERNS = {key: re.compile(r"\b(?:" + pattern + r")\b") for key, pattern in _ACTIONS.items()}
_PART_ALIASES = {
    "head": "head", "neck": "neck", "torso": "torso", "trunk": "torso",
    "spine": "spine", "pelvis": "pelvis", "hip": "hip", "hips": "hip",
    "arm": "arm", "arms": "arm", "shoulder": "shoulder", "shoulders": "shoulder",
    "elbow": "elbow", "elbows": "elbow", "hand": "hand", "hands": "hand",
    "wrist": "wrist", "wrists": "wrist", "leg": "leg", "legs": "leg",
    "knee": "knee", "knees": "knee", "ankle": "ankle", "ankles": "ankle",
    "foot": "foot", "feet": "foot",
}
_PART_PATTERN = re.compile(r"\b(?:(left|right|both)\s+)?("
                           + "|".join(sorted(_PART_ALIASES, key=len, reverse=True)) + r")\b")


def normalize_sentence(text: str) -> str:
    """Normalize Unicode, case and whitespace, retaining sides, times and numbers."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _sentences(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("caption fields must be null, a string or a list of strings")
    result = []
    for item in value:
        # A decimal point or a time range is never a sentence boundary. Preserve
        # blank entries as run separators; do not join A, <blank>, A into a run.
        result.extend(re.split(r"(?<=[.!?])\s+(?=[A-Za-z\[])|\n", item))
    return [normalize_sentence(item) for item in result]


def _lexical_parts(sentences):
    parts = set()
    for sentence in sentences:
        for match in _PART_PATTERN.finditer(sentence.replace("_", " ")):
            side, part = match.groups()
            base = _PART_ALIASES[part]
            parts.add(f"{side}_{base}" if side else base)
    return parts


def _text_view(value):
    sentences = _sentences(value)
    longest, current, previous = 0, 0, None
    for sentence in sentences:
        current = current + 1 if sentence and sentence == previous else int(bool(sentence))
        longest = max(longest, current)
        previous = sentence
    return {"sentences": [sentence for sentence in sentences if sentence],
            "blank_sentence_entries": sum(not sentence for sentence in sentences),
            "longest_consecutive_run": longest}


def _identities(data, sid, *, declared_sid):
    keys = {("sid" if declared_sid else "relative_record_path", sid)}
    evidence = []
    for field in _ID_FIELDS:
        value = data.get(field)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            keys.add(("source_id", str(value)))
            evidence.append({"field": field, "value": str(value)})
    for field in _HASH_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            keys.add(("content_sha256", value.lower()))
            evidence.append({"field": field, "value": value.lower()})
    return keys, evidence


def _sample(data, path, root):
    if not isinstance(data, dict):
        return None
    if data.get("schema") == "motionscript_open_visual_v1" and "motion_caption" in data:
        schema, text = "motionscript_open_visual_v1", data["motion_caption"]
    elif "ms" in data:
        schema, text = "legacy_motionscript", data["ms"]
    elif "caption" in data and ("events" in data or "sid" in data):
        schema, text = "motionscript_v2", data["caption"]
    else:
        return None
    relative_path = path.relative_to(root).as_posix()
    declared_sid = "sid" in data
    # Independent visual bundles commonly repeat candidate.json/export.json.
    # A filename stem cannot stand in for source identity across those bundles.
    sid = data.get("sid", relative_path)
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError("sid must be a nonempty string")
    source = data.get("source")
    if not isinstance(source, str) or not source.strip():
        source = schema + "_source_unspecified"
    text_view = _text_view(text)
    nonempty = text_view["sentences"]
    approval = data.get("semantic_approval")
    approval = "approved" if approval is True else "unapproved" if approval is False else "unspecified"
    keys, evidence = _identities(data, sid, declared_sid=declared_sid)
    events = data.get("events", [])
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ValueError("events must be a list of objects")
    structured = defaultdict(lambda: {"predicates": set(), "actions": set(), "parts": set(), "event_count": 0})
    open_descriptions = defaultdict(lambda: {"descriptions": set(), "event_count": 0})
    skipped_events = 0
    for event in events:
        part, action = event.get("body_part"), event.get("action")
        direction = event.get("direction", "unspecified")
        event_source = event.get("source", source)
        recognized = False
        if all(isinstance(value, str) and value.strip() for value in (part, action, direction, event_source)):
            bucket = structured[event_source]
            # Structured labels remain exact; no synonym or opposite-side merging.
            bucket["predicates"].add(json.dumps([part, action, direction], ensure_ascii=False))
            bucket["actions"].add(action)
            bucket["parts"].add(part)
            bucket["event_count"] += 1
            recognized = True
        description = event.get("description")
        if (isinstance(description, str) and description.strip()
                and isinstance(event_source, str) and event_source.strip()):
            bucket = open_descriptions[event_source]
            bucket["descriptions"].add(normalize_sentence(description))
            bucket["event_count"] += 1
            recognized = True
        if not recognized:
            skipped_events += 1
    return {
        "path": relative_path, "sid": sid, "source": source,
        "sid_origin": "declared" if declared_sid else "relative_record_path_fallback",
        "schema": schema, "semantic_approval": approval, **text_view,
        "lexical_actions": {action for action, pattern in _ACTION_PATTERNS.items()
                            if any(pattern.search(sentence) for sentence in nonempty)},
        "lexical_parts": _lexical_parts(nonempty), "structured": dict(structured),
        "open_descriptions": dict(open_descriptions),
        "compact_caption_view": _text_view(data["compact_caption"]) if "compact_caption" in data else None,
        "skipped_events": skipped_events, "identity_keys": keys, "identity_evidence": evidence,
    }


def _group_samples(samples):
    parents = list(range(len(samples)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owner = {}
    for index, sample in enumerate(samples):
        for key in sample["identity_keys"]:
            if key in owner:
                parents[find(index)] = find(owner[key])
            else:
                owner[key] = index
    members = defaultdict(list)
    for index, sample in enumerate(samples):
        members[find(index)].append(sample)
    groups = []
    for index, members_of_group in enumerate(sorted(members.values(), key=lambda group: min(row["path"] for row in group)), 1):
        group_id = f"source_group_{index:06d}"
        for sample in members_of_group:
            sample["group_id"] = group_id
        groups.append({
            "group_id": group_id, "record_count": len(members_of_group),
            "sids": sorted({row["sid"] for row in members_of_group}),
            "sources": sorted({row["source"] for row in members_of_group}),
            "records": sorted(row["path"] for row in members_of_group),
            "explicit_identity_evidence": [{"record": row["path"], **item}
                                           for row in members_of_group for item in row["identity_evidence"]],
        })
    return groups


def _presence(samples, sets, top_k, *, occurrences=None):
    counts = Counter()
    groups = defaultdict(set)
    for sample, values in zip(samples, sets):
        for value in set(values):
            counts[value] += 1
            groups[value].add(sample["group_id"])
    independent = len({sample["group_id"] for sample in samples})
    order = sorted(counts, key=lambda key: (-len(groups[key]), -counts[key], key))[:top_k]
    return [{
        "value": value, "sample_presence": counts[value],
        "sample_fraction": counts[value] / len(samples) if samples else 0.0,
        "independent_group_presence": len(groups[value]),
        "independent_group_fraction": len(groups[value]) / independent if independent else 0.0,
        **({"sentence_occurrences": occurrences[value]} if occurrences is not None else {}),
    } for value in order]


def _caption_statistics(samples, top_k):
    count = len(samples)
    sentence_counts = [len(sample["sentences"]) for sample in samples]
    duplicates = [len(sample["sentences"]) - len(set(sample["sentences"])) for sample in samples]
    occurrences = Counter(sentence for sample in samples for sentence in sample["sentences"])
    return {
        "sample_count": count, "empty_description_count": sum(not number for number in sentence_counts),
        "sentence_count": sum(sentence_counts), "unique_normalized_sentences": len(occurrences),
        "repeated_sentence_occurrences_within_samples": sum(duplicates),
        "micro_repeat_fraction": sum(duplicates) / sum(sentence_counts) if sum(sentence_counts) else 0.0,
        "mean_sample_repeat_fraction": sum(d / n if n else 0.0 for d, n in zip(duplicates, sentence_counts)) / count if count else 0.0,
        "longest_consecutive_run": max((sample["longest_consecutive_run"] for sample in samples), default=0),
        "top_sentences": _presence(samples, [sample["sentences"] for sample in samples], top_k, occurrences=occurrences),
    }


def _summary(samples, top_k):
    result = {
        **_caption_statistics(samples, top_k),
        "unique_sid_count": len({sample["sid"] for sample in samples}),
        "independent_source_groups": len({sample["group_id"] for sample in samples}),
        "semantic_approval_counts": dict(Counter(sample["semantic_approval"] for sample in samples)),
        "lexical_action_mentions": _presence(samples, [sample["lexical_actions"] for sample in samples], top_k),
        "lexical_body_part_coverage": _presence(samples, [sample["lexical_parts"] for sample in samples], len(_PART_ALIASES) * 4),
        "structured_events_by_source": {},
        "open_event_descriptions_by_source": {},
    }
    event_sources = sorted({source for sample in samples for source in sample["structured"]})
    for source in event_sources:
        annotated = [sample for sample in samples if source in sample["structured"]]
        buckets = [sample["structured"][source] for sample in annotated]
        predicates = _presence(samples, [sample["structured"].get(source, {}).get("predicates", set())
                                         for sample in samples], top_k)
        for row in predicates:
            row["body_part"], row["action"], row["direction"] = json.loads(row.pop("value"))
        result["structured_events_by_source"][source] = {
            "samples_with_events": len(annotated),
            "independent_source_groups_with_events": len({sample["group_id"] for sample in annotated}),
            "event_count": sum(bucket["event_count"] for bucket in buckets),
            "top_action_predicates": predicates,
            "action_presence": _presence(samples, [sample["structured"].get(source, {}).get("actions", set())
                                                    for sample in samples], top_k),
            "body_part_coverage": _presence(samples, [sample["structured"].get(source, {}).get("parts", set())
                                                      for sample in samples], 1000),
            "presence_denominator": "all records in this enclosing caption-source domain, including empty/no-event reports; absence measures annotation coverage, not true action absence",
        }
    open_sources = sorted({source for sample in samples for source in sample["open_descriptions"]})
    for source in open_sources:
        descriptions = [sample["open_descriptions"].get(source, {}).get("descriptions", set())
                        for sample in samples]
        result["open_event_descriptions_by_source"][source] = {
            "samples_with_events": sum(source in sample["open_descriptions"] for sample in samples),
            "event_count": sum(sample["open_descriptions"].get(source, {}).get("event_count", 0)
                               for sample in samples),
            "unique_normalized_descriptions": len(set().union(*descriptions)),
            "top_descriptions": _presence(samples, descriptions, top_k),
            "lexical_action_mentions": _presence(samples,
                [{action for action, pattern in _ACTION_PATTERNS.items() if any(pattern.search(text) for text in values)}
                 for values in descriptions], top_k),
            "lexical_body_part_mentions": _presence(samples, [_lexical_parts(values) for values in descriptions], len(_PART_ALIASES) * 4),
            "presence_denominator": "all records in this enclosing caption-source domain, including no-event/abstention reports",
            "coverage_limitation": "exact free-description string presence and finite-list lexical mentions; not semantic action-class diversity or open-vocabulary coverage",
        }
    paired = [sample for sample in samples if sample["compact_caption_view"] is not None]
    if paired:
        compact = [{**sample, **sample["compact_caption_view"]} for sample in paired]
        result["compact_caption_comparison"] = {
            "paired_sample_count": len(paired), "additional_independent_samples": 0,
            "original_caption": _caption_statistics(paired, top_k),
            "compact_caption": _caption_statistics(compact, top_k),
            "interpretation": "paired textual views of the same records; compression and reduced repetition do not establish new action coverage or semantic correctness",
        }
    return result


def audit_directory(input_dir: str | Path, *, top_k=20) -> dict:
    """Read legacy/v2/open-visual JSON recursively, without accessing media or writing."""
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise ValueError("--input must be an existing directory")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    samples, errors, ignored, inputs = [], [], [], []
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink():
            raise ValueError(f"input JSON symlinks are not supported: {path}")
        content = path.read_bytes()
        inputs.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(content).hexdigest()})
        try:
            data = json.loads(content.decode("utf-8-sig"))
            sample = _sample(data, path, root)
            if sample is None:
                ignored.append(path.relative_to(root).as_posix())
            else:
                samples.append(sample)
        except (UnicodeError, ValueError, TypeError) as error:
            errors.append({"path": path.relative_to(root).as_posix(), "error": str(error)})
    groups = _group_samples(samples)
    by_source = defaultdict(list)
    for sample in samples:
        by_source[sample["source"]].append(sample)
    content_duplicates = defaultdict(list)
    for sample in samples:
        for kind, value in sample["identity_keys"]:
            if kind == "content_sha256":
                content_duplicates[value].append(sample["path"])
    per_sample = []
    for sample in samples:
        total, unique = len(sample["sentences"]), len(set(sample["sentences"]))
        metrics = {key: sample[key] for key in ("path", "sid", "sid_origin", "source", "schema", "semantic_approval", "group_id", "longest_consecutive_run", "blank_sentence_entries", "skipped_events")}
        metrics.update({"sentence_count": total, "unique_sentence_count": unique,
                        "repeated_sentence_count": total - unique,
                        "repeat_fraction": (total - unique) / total if total else 0.0})
        if sample["compact_caption_view"] is not None:
            compact = sample["compact_caption_view"]
            compact_total, compact_unique = len(compact["sentences"]), len(set(compact["sentences"]))
            metrics["compact_caption"] = {"sentence_count": compact_total,
                "unique_sentence_count": compact_unique,
                "repeated_sentence_count": compact_total - compact_unique,
                "repeat_fraction": (compact_total - compact_unique) / compact_total if compact_total else 0.0,
                "longest_consecutive_run": compact["longest_consecutive_run"],
                "blank_sentence_entries": compact["blank_sentence_entries"]}
        per_sample.append(metrics)
    return {
        "schema": "motionscript_caption_audit_v1", "input_directory": str(root),
        "read_only": True, "semantic_approval_granted": False, "training_weights_emitted": False,
        "sample_count": len(samples), "empty_description_count": sum(not row["sentences"] for row in samples),
        "unique_sid_count": len({row["sid"] for row in samples}), "independent_source_groups": len(groups),
        "input_json_count": len(inputs), "ignored_json_files": ignored, "parse_errors": errors,
        "by_source": {source: _summary(rows, top_k) for source, rows in sorted(by_source.items())},
        "samples": per_sample, "source_groups": groups,
        "matching_source_content_hashes": [{"sha256": value, "records": sorted(paths)}
                                            for value, paths in sorted(content_duplicates.items()) if len(paths) > 1],
        "input_json_sha256": inputs,
        "definitions": {
            "sentence_normalization": "Unicode NFKC + casefold + whitespace collapse; retain left/right, digits, times, punctuation and event IDs",
            "repeat_fraction": "(nonempty sentence count - distinct normalized sentences) / nonempty sentence count; empty=0",
            "longest_consecutive_run": "maximum adjacent identical normalized nonempty sentences; blanks break a run",
            "presence": "at most one count per report or linked source group; repeated events/sentences cannot inflate presence",
            "grouping": "transitive shared declared SID, explicit source_id/source_group/video_id, or supplied SHA256 of source/input content; missing SID uses full relative record path, never filename stem; never caption equality",
            "grouping_limitations": "IDs/hashes are declared provenance, not verified media identity; input_sha256 identifies an NLF artifact, not a recovered video hash. Without provenance, media duplicates remain unknown. Source groups can include distinct windows of one video.",
            "lexical_limitations": "English finite-list word mentions, including negated or hypothetical text; not positive action facts, semantic correctness, or open-vocabulary coverage. Free event descriptions remain separately counted even when they contain no listed verbs or body parts.",
            "structured_limitations": "event predicates describe supplied candidate annotations; their frequency is not verified physical action frequency or label approval",
            "open_description_limitations": "normalized exact free-description strings retain facts and are never forced into the closed action/body_part enums; string variety is not semantic action-class diversity",
            "compact_caption": "paired variant of the same original record, not an extra sample/source group; raw-caption counts are not pooled with compact-caption counts",
            "source_separation": "no pooled cross-source action frequency; caption source and nested event source remain separate",
        },
        "training_recommendations": [
            "Review frequent/repeated labels against video and retained event times before changing supervision; true repeated actions must remain represented.",
            "Split/evaluate by linked source groups and verify missing source/media identity; shared caption wording alone never establishes duplicate videos.",
            "Compare sentence and reviewed event/body-part coverage within each source. Closed-vocabulary coverage and linguistic variety do not measure physical action diversity.",
            "Do not oversample unreviewed labels from these counts. After semantic review, evaluate any balancing proposal on held-out source groups; this audit emits no weights.",
            "If using constrained paraphrases, preserve side, timing, measurements and repeated-event facts. Synonym expansion does not add new actions or repair missing pose observations.",
        ],
    }


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--input", required=True, help="Legacy/v2/open-visual JSON directory (read only)")
    argument_parser.add_argument("--out", required=True, help="New JSON path outside the input tree")
    argument_parser.add_argument("--top-k", type=int, default=20)
    return argument_parser


def run(args):
    root, output = Path(args.input).resolve(), Path(args.out).resolve()
    if output == root or root in output.parents:
        raise ValueError("--out must be outside the input tree")
    if output.exists():
        raise FileExistsError("--out already exists; input files and existing reports must not be overwritten")
    report = audit_directory(root, top_k=args.top_k)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return report


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError) as error:
        argument_parser.exit(2, f"error: {error}\n")
    print(json.dumps({"samples": report["sample_count"], "source_groups": report["independent_source_groups"],
                      "parse_errors": len(report["parse_errors"]), "out": str(Path(args.out).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
