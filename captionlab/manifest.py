"""Freeze paired-audio probes without inference, training or negative-label guesses."""
import copy
from collections import Counter
import ntpath
import posixpath

from .io import object_sha, text_sha
from .targets import convert_record, validate_source_records


SCHEMA = "captionlab.probe.v1"
TARGET_MODES = ("raw", "actions_timed", "actions_untimed")


def source_components(records):
    """Connected components, including transitive source/song/file/PCM relationships."""
    parent = {row["id"]: row["id"] for row in records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for row in sorted(records, key=lambda r: r["id"]):
        path = row["audio"]["path"]
        path = ntpath.normcase(ntpath.normpath(path)) if ntpath.splitdrive(path)[0] or path.startswith("\\") else posixpath.normpath(path)
        for field, value in (("source", row["source_group_id"]),
                             ("song", row["song_id"]),
                             ("near", row["near_duplicate_group_id"]),
                             ("pcm", row["audio"]["pcm_sha256"]),
                             ("file", row["audio"]["sha256"]), ("path", path)):
            if value is None:
                continue
            key = (field, value)
            if key in owners:
                a, b = find(row["id"]), find(owners[key])
                parent[max(a, b)] = min(a, b)
            else:
                owners[key] = row["id"]
    members = {}
    for identifier in parent:
        members.setdefault(find(identifier), []).append(identifier)
    names = {root: "group-" + object_sha(sorted(ids))[:20] for root, ids in members.items()}
    return {identifier: names[find(identifier)] for identifier in parent}


def _same_geometry(first, second):
    # An unknown padding map stays unknown; equality is no proof of its accuracy.
    return all(first["audio"][k] == second["audio"][k]
               for k in ("sample_rate", "num_samples", "pad_intervals"))


def prepare_probe(records, *, genre_classes, target_mode="raw", case_ids=None,
                  max_cases=256, seed=42, hard_pairs=()):
    """Select validation cases; donors are validation-only, all labels retained.

    A hard pair is explicit, never inferred from genre or BPM. Missing donors stay
    visible. Random candidates remain unreviewed even after identity exclusions.
    """
    if target_mode not in TARGET_MODES:
        raise ValueError("Unknown target_mode")
    if type(max_cases) is not int or max_cases < 1 or type(seed) is not int:
        raise ValueError("Require a positive max_cases and integer seed")
    isolation = validate_source_records(records)
    by_id = {row["id"]: row for row in records}
    groups = source_components(records)
    pool = sorted((row for row in records if row["split"] == "val"), key=lambda r: r["id"])
    if not pool:
        raise ValueError("No validation rows; test/train may not be used for this development probe")
    if case_ids is None:
        selected = sorted(pool, key=lambda r: text_sha(f"{seed}:case:{r['id']}"))[:max_cases]
        selection = "deterministic-sha256-validation-subset"
    else:
        if (not isinstance(case_ids, list) or not case_ids or
                any(not isinstance(i, str) for i in case_ids) or len(set(case_ids)) != len(case_ids)):
            raise ValueError("case_ids must be unique nonempty validation IDs")
        if len(case_ids) > max_cases or any(i not in by_id or by_id[i]["split"] != "val" for i in case_ids):
            raise ValueError("Fixed case list exceeds budget or contains missing/non-validation IDs")
        selected = [by_id[i] for i in case_ids]
        selection = "explicit-frozen-validation-ids"
    selected_ids = {r["id"] for r in selected}
    hard = {}
    for pair in hard_pairs:
        if set(pair) != {"case_id", "donor_id", "mismatch_review", "review_evidence"}:
            raise ValueError("Hard pairs require case_id, donor_id, mismatch_review, review_evidence")
        case_id, donor = pair["case_id"], pair["donor_id"]
        if case_id not in selected_ids or case_id in hard or donor not in by_id:
            raise ValueError("Hard pair has unknown/duplicate case or unknown donor")
        if by_id[donor]["split"] != "val" or groups[case_id] == groups[donor]:
            raise ValueError("Hard pair crosses split or is in the same known positive/source component")
        if not _same_geometry(by_id[case_id], by_id[donor]):
            raise ValueError("Hard pair must match declared rate, sample count and padding geometry")
        if pair["mismatch_review"] not in ("unreviewed", "reviewed_mismatch"):
            raise ValueError("Known positive is not an eligible hard negative candidate")
        if not isinstance(pair["review_evidence"], str):
            raise ValueError("Review evidence must be text")
        if pair["mismatch_review"] == "reviewed_mismatch" and not pair["review_evidence"].strip():
            raise ValueError("Reviewed mismatch needs actual review evidence")
        hard[case_id] = pair
    cases = []
    for source in selected:
        identifier = source["id"]
        converted = convert_record(source, keep_timestamps=(target_mode != "actions_untimed"),
                                   genre_classes=genre_classes)
        target = source["raw_caption"] if target_mode == "raw" else converted["target_text"]

        def condition(kind, donor, review="unreviewed", evidence=""):
            return {"condition_id": kind, "kind": kind, "audio_id": donor["id"],
                    "audio": copy.deepcopy(donor["audio"]),
                    "donor_source_group_id": groups[donor["id"]],
                    "mismatch_review": review, "review_evidence": evidence,
                    "transform": "zero_waveform_same_geometry" if kind == "silence" else "identity"}

        conditions = [condition("true", source, "known_positive")]
        excluded = {identifier, hard[identifier]["donor_id"]} if identifier in hard else {identifier}
        eligible = [row for row in pool if groups[row["id"]] != groups[identifier] and _same_geometry(source, row)
                    and row["id"] not in excluded]
        unavailable = []
        if eligible:
            donor = min(eligible, key=lambda r: text_sha(f"{seed}:donor:{identifier}:{r['id']}"))
            conditions.append(condition("random", donor))
        else:
            unavailable.append({"kind": "random", "reason": "no_distinct_eligible_validation_donor"})
        if identifier in hard:
            pair = hard[identifier]
            conditions.append(condition("hard", by_id[pair["donor_id"]],
                                        pair["mismatch_review"], pair["review_evidence"]))
        else:
            unavailable.append({"kind": "hard", "reason": "no_explicit_hard_pair"})
        conditions.append(condition("silence", source))
        cases.append({"case_id": identifier, "source_group_id": groups[identifier], "split": "val",
                      "target_text": target, "target_sha256": text_sha(target),
                      "target_mode": target_mode, "target_source": copy.deepcopy(source),
                      "target_windows": [{"start": w["start"], "end": w["end"],
                                          "caption": w["action_text"].strip()} for w in converted["windows"]],
                      "padding_geometry_known": source["audio"]["pad_intervals"] is not None,
                      "conditions": conditions, "unavailable_conditions": unavailable})
    return {"schema": SCHEMA, "source_records_sha256": object_sha(records),
            "source_isolation": isolation, "selection": selection, "seed": seed,
            "case_budget": max_cases, "target_mode": target_mode,
            "genre_classes": list(genre_classes), "cases": cases,
            "condition_counts": dict(Counter(c["kind"] for r in cases for c in r["conditions"])),
            "forward_calls_per_checkpoint": sum(len(r["conditions"]) for r in cases),
            "negative_policy": "known source/song/near-duplicate/file/PCM components excluded; equal declared rate/sample/padding geometry; random remains unreviewed",
            "song_identity_complete": all(row["song_id"] is not None for row in records),
            "claim": "teacher-forced sensitivity diagnostic; not dance quality or unique choreography truth"}
