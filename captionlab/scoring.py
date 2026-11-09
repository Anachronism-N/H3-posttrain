"""CPU-only paired diagnostics for frozen music-to-caption likelihood records.

Inputs are observations supplied by a scoring bridge, not proof of model execution.
Every target ID is an unpadded, effective assistant token, including one terminal
EOS.  This module never tokenizes, parses captions, or infers action semantics.
"""

import hashlib
import json
import math
import random
from fractions import Fraction
from statistics import mean


class ScoringError(ValueError):
    """Invalid or internally inconsistent frozen diagnostic evidence."""


KINDS = ("true", "random", "hard", "silence")
REVIEWS = ("unreviewed", "reviewed_mismatch", "known_positive")
_REQUIRED_RECORD = {
    "case_id", "condition_id", "target_token_ids", "token_nll", "eos_token_id",
    "model_identity", "tokenizer_identity", "target_sha256",
}
_OPTIONAL_RECORD = {"action_mask", "provenance", "runtime_details", "schema",
                    "manifest_sha256", "runtime_config_sha256"}


def _require(test, message):
    if not test:
        raise ScoringError(message)


def _text(value, label):
    _require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def _number(value, label, nonnegative=False):
    _require(type(value) in (int, float), label + " must be a finite number, not boolean")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ScoringError(label + " exceeds the finite numeric range") from error
    _require(math.isfinite(result), label + " must be finite")
    if type(value) is int:
        _require(int(result) == value, label + " loses integer precision")
    _require(not nonnegative or result >= 0, label + " must be nonnegative")
    return result


def _mean(values, label):
    _require(bool(values), label + " has no support")
    try:
        return _number(mean(values), label)
    except (OverflowError, ValueError) as error:
        raise ScoringError(label + " cannot be represented finitely") from error


def _difference(a, b, label):
    # Fraction avoids an overflowing intermediate sum or subtraction.
    try:
        return _number(float(Fraction.from_float(a) - Fraction.from_float(b)), label)
    except OverflowError as error:
        raise ScoringError(label + " exceeds the finite numeric range") from error


def _finite_json(value, label):
    def visit(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            _require(math.isfinite(item), label + " contains nonfinite JSON")
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            _require(all(isinstance(key, str) for key in item), label + " has non-string keys")
            for child in item.values():
                visit(child)
            return
        raise ScoringError(label + " must contain JSON values only")
    visit(value)


def _sha(value):
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, OverflowError) as error:
        raise ScoringError("input cannot be encoded as canonical finite UTF-8 JSON") from error
    return hashlib.sha256(raw).hexdigest()


def _digest(value, label):
    _require(isinstance(value, str) and len(value) == 64 and
             all(character in "0123456789abcdef" for character in value),
             label + " must be lowercase SHA256 hex")
    return value


def _validate_manifest(manifest):
    _require(isinstance(manifest, dict), "manifest must be an object")
    _finite_json(manifest, "manifest")
    cases = manifest.get("cases")
    _require(isinstance(cases, list) and bool(cases), "manifest cases must be a nonempty list")
    by_case, expected = {}, {}
    for case in cases:
        _require(isinstance(case, dict), "manifest case must be an object")
        case_id = _text(case.get("case_id"), "case_id")
        _require(case_id not in by_case, "duplicate manifest case_id: " + case_id)
        _text(case.get("source_group_id"), "source_group_id")
        target = _text(case.get("target_text"), "target_text")
        try:
            target_sha = hashlib.sha256(target.encode("utf-8")).hexdigest()
        except UnicodeError as error:
            raise ScoringError("target_text must be valid UTF-8") from error
        _require(_digest(case.get("target_sha256"), "target_sha256") == target_sha,
                 "target_text SHA256 mismatch: " + case_id)
        conditions = case.get("conditions")
        _require(isinstance(conditions, list) and bool(conditions), "conditions must be nonempty")
        counts = dict.fromkeys(KINDS, 0)
        audio_ids = set()
        condition_by_kind = {}
        for condition in conditions:
            _require(isinstance(condition, dict), "condition must be an object")
            condition_id = _text(condition.get("condition_id"), "condition_id")
            key = (case_id, condition_id)
            _require(key not in expected, "duplicate condition_id in case: " + case_id)
            kind = condition.get("kind")
            _require(isinstance(kind, str) and kind in KINDS, "invalid condition kind")
            audio_id = _text(condition.get("audio_id"), "audio_id")
            if kind != "silence":
                _require(audio_id not in audio_ids, "duplicate candidate audio_id within a case")
                audio_ids.add(audio_id)
            review = condition.get("mismatch_review", "unreviewed")
            _require(isinstance(review, str) and review in REVIEWS, "invalid mismatch_review")
            _require(kind != "true" or review in ("unreviewed", "known_positive"),
                     "true condition cannot be declared a reviewed mismatch")
            _require(kind != "silence" or review == "unreviewed",
                     "silence is not a reviewed mismatch candidate")
            counts[kind] += 1
            _require(counts[kind] == 1, "duplicate condition kind within a case")
            condition_by_kind[kind] = condition
            expected[key] = condition
        _require(counts["true"] == 1 and counts["silence"] == 1,
                 "each case requires exactly one true and one silence condition")
        _require(condition_by_kind["silence"]["audio_id"] == condition_by_kind["true"]["audio_id"],
                 "silence must transform the true condition audio_id")
        unavailable = case.get("unavailable_conditions", [])
        _require(isinstance(unavailable, list), "unavailable_conditions must be a list")
        unavailable_kinds = set()
        for entry in unavailable:
            _require(isinstance(entry, dict), "unavailable condition must be an object")
            kind = entry.get("kind")
            _require(kind in ("random", "hard"), "only random/hard conditions may be unavailable")
            _text(entry.get("reason"), "unavailable condition reason")
            _require(kind not in unavailable_kinds, "duplicate unavailable condition kind")
            _require(counts[kind] == 0, "condition cannot be both available and unavailable")
            unavailable_kinds.add(kind)
        _require(unavailable_kinds == {kind for kind in ("random", "hard") if counts[kind] == 0},
                 "every absent random/hard condition requires an explicit unavailable reason")
        by_case[case_id] = case
    return by_case, expected


def _validate_records(records, by_case, expected):
    _require(isinstance(records, list), "records must be a list")
    _finite_json(records, "records")
    by_key, reference, identity = {}, {}, None
    for record in records:
        _require(isinstance(record, dict), "score record must be an object")
        _require(_REQUIRED_RECORD <= set(record), "score record is missing required fields")
        _require(not (set(record) - _REQUIRED_RECORD - _OPTIONAL_RECORD),
                 "score record has unknown fields")
        if "runtime_details" in record:
            _require(isinstance(record["runtime_details"], dict), "runtime_details must be an object")
        case_id = _text(record["case_id"], "record case_id")
        condition_id = _text(record["condition_id"], "record condition_id")
        key = (case_id, condition_id)
        _require(key in expected, "record names an unknown case/condition")
        _require(key not in by_key, "duplicate score record: " + case_id + "/" + condition_id)
        _require(record["target_sha256"] == by_case[case_id]["target_sha256"],
                 "record target_sha256 mismatch")
        tokens, nll = record["target_token_ids"], record["token_nll"]
        _require(isinstance(tokens, list) and bool(tokens) and
                 all(type(token) is int and token >= 0 for token in tokens),
                 "target_token_ids must be nonempty nonnegative integer IDs")
        eos = record["eos_token_id"]
        _require(type(eos) is int and eos >= 0, "eos_token_id must be a nonnegative integer")
        _require(tokens[-1] == eos and tokens.count(eos) == 1,
                 "effective target support must include exactly one terminal EOS")
        _require(isinstance(nll, list) and len(nll) == len(tokens),
                 "token_nll must match every effective target token, including EOS")
        for value in nll:
            _number(value, "token NLL", nonnegative=True)
        action_mask = record.get("action_mask")
        if "action_mask" in record:
            _require(isinstance(action_mask, list) and len(action_mask) == len(tokens) and
                     all(type(value) is bool for value in action_mask),
                     "action_mask must have one boolean per target token")
            _require(not action_mask[-1], "action_mask must exclude EOS")
        current_identity = (_text(record["model_identity"], "model_identity"),
                            _text(record["tokenizer_identity"], "tokenizer_identity"), eos)
        if identity is None:
            identity = current_identity
        _require(identity == current_identity,
                 "model_identity/tokenizer_identity/EOS must be consistent across records")
        support = (tokens, action_mask)
        if case_id in reference:
            _require(reference[case_id] == support,
                     "target token IDs and action support must be identical across conditions")
        else:
            reference[case_id] = support
        by_key[key] = record
    bindings = {}
    for name in ("schema", "manifest_sha256", "runtime_config_sha256"):
        declared = [record[name] for record in records if name in record]
        if declared:
            _require(len(declared) == len(records), name + " declarations must not mix with unknown bindings")
            for value in declared:
                if name == "schema":
                    _text(value, "record schema")
                else:
                    _digest(value, name)
            _require(all(value == declared[0] for value in declared), name + " differs across records")
            bindings[name] = declared[0]
    return by_key, identity, bindings


def _percentile(values, probability):
    position = (len(values) - 1) * probability
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    weight = Fraction.from_float(position - lower)
    return _number(float((1 - weight) * Fraction.from_float(values[lower]) +
                         weight * Fraction.from_float(values[upper])), "bootstrap percentile")


def _aggregate(values, cases, n_resamples, confidence, seed):
    """values maps case ID to a complete paired comparison or None."""
    group_values = {}
    for case_id, value in values.items():
        if value is not None:
            group_values.setdefault(cases[case_id]["source_group_id"], []).append(value)
    groups = [{"source_group_id": group_id, "n_cases": len(group_values[group_id]),
               "mean_difference": _mean(group_values[group_id], "source-group mean")}
              for group_id in sorted(group_values)]
    complete = [value for value in values.values() if value is not None]
    report = {
        "difference_direction": "condition_minus_true_nll",
        "n_cases_expected": len(cases), "n_cases_paired": len(complete),
        "n_cases_missing_or_unsupported": len(cases) - len(complete),
        "missing_or_unsupported_case_ids": sorted(case_id for case_id, value in values.items()
                                                   if value is None),
        "n_source_groups_expected": len({case["source_group_id"] for case in cases.values()}),
        "n_source_groups_observed": len(groups), "groups": groups,
        "estimate_group_equal": None, "estimate_case_equal": None,
        "interval_group_equal": None,
        "interval_status": "no_complete_pairs", "confidence": confidence,
        "bootstrap_resamples": n_resamples, "seed": seed,
        "estimand": "equal source-group weight; within group equal complete-case weight",
        "missingness_scope": "available complete case/kind pairs only; missing cases are not zero",
    }
    if not complete:
        return report
    group_means = [group["mean_difference"] for group in groups]
    report["estimate_group_equal"] = _mean(group_means, "equal-group estimate")
    report["estimate_case_equal"] = _mean(complete, "equal-case estimate")
    if len(groups) < 2:
        report["interval_status"] = "insufficient_independent_source_groups"
        return report
    rng = random.Random(seed)
    samples = sorted(_mean([group_means[rng.randrange(len(group_means))]
                            for _ in group_means], "cluster bootstrap mean")
                     for _ in range(n_resamples))
    tail = (1 - confidence) / 2
    report["interval_group_equal"] = [_percentile(samples, tail), _percentile(samples, 1 - tail)]
    report["interval_status"] = "available_complete_pairs_cluster_percentile"
    return report


def _rank(true_id, candidate_ids, scores, tolerance):
    if len(candidate_ids) < 2 or any(scores[condition_id] is None for condition_id in candidate_ids):
        return None
    true_score = scores[true_id]
    deltas = [_difference(scores[condition_id], true_score, "ranking difference")
              for condition_id in candidate_ids if condition_id != true_id]
    better = sum(value < -tolerance for value in deltas)
    ties = sum(abs(value) <= tolerance for value in deltas)
    return {"n_candidates": len(candidate_ids), "true_rank_min": 1 + better,
            "true_rank_max": 1 + better + ties, "n_wrong_strictly_better": better,
            "n_other_tied_with_true": ties, "true_unique_best": better == 0 and ties == 0,
            "true_best_including_ties": better == 0,
            "candidate_condition_ids": sorted(candidate_ids)}


def _ranking_summary(rows, scope):
    eligible = [row for row in rows if row["scopes"][scope]["reviewed_subset_eligible"]]
    scored = [row["scopes"][scope]["reviewed_subset_rank"] for row in eligible
              if row["scopes"][scope]["reviewed_subset_rank"] is not None]
    return {
        "n_cases_manifest": len(rows), "n_cases_with_reviewed_mismatch_subset": len(eligible),
        "n_cases_scored": len(scored), "n_cases_pending": len(eligible) - len(scored),
        "n_cases_without_reviewed_mismatch_subset": len(rows) - len(eligible),
        "n_cases_with_all_predeclared_candidate_kinds_available": sum(
            all(row["condition_kind_availability"].values()) for row in rows),
        "n_true_unique_best": sum(rank["true_unique_best"] for rank in scored),
        "n_true_best_including_ties": sum(rank["true_best_including_ties"] for rank in scored),
        "unique_top1_rate_scored_reviewed_subset":
            sum(rank["true_unique_best"] for rank in scored) / len(scored) if scored else None,
        "best_including_ties_rate_scored_reviewed_subset":
            sum(rank["true_best_including_ties"] for rank in scored) / len(scored) if scored else None,
        "rate_scope": "conditional on complete explicit reviewed-mismatch subsets; not population accuracy",
        "review_status_is_declaration_not_independent_semantic_verification": True,
    }


def summarize_scores(manifest, records, *, n_resamples=2000, confidence=0.95,
                     seed=0, tie_tolerance=0.0):
    """Validate and summarize fixed-target likelihood contrasts, without mutation.

    Required manifest cases have case_id, source_group_id, target_text (exact
    UTF-8 SHA256 target_sha256), and conditions. Each condition has condition_id,
    kind true/random/hard/silence and audio_id, optionally mismatch_review:
    unreviewed (default), reviewed_mismatch, or known_positive. There is exactly
    one true/silence and at most one random/hard per case. Absent random/hard
    conditions require an unavailable_conditions entry with kind/reason and stay
    in each kind's predeclared case denominator. Candidate audio IDs are unique;
    silence must refer to the true ID as a transformed condition. Different IDs
    do not establish different PCM or semantic mismatch. True may be declared
    known_positive; random/hard review labels are explicit declarations. Extra
    manifest metadata is preserved in its content hash.

    Required record fields are documented by _REQUIRED_RECORD; optional fields
    are action_mask, provenance, runtime_details, schema, manifest_sha256 and
    runtime_config_sha256. The optional hashes are raw-artifact declarations;
    callers must bind them to their actual files (canonical content hashes in
    this report are distinct). Missing records are retained in coverage,
    never fabricated as zero NLL/failure/success. An absent action_mask means NA,
    not an inferred parse. Identities are declarations, not model attestations.
    """
    _require(type(n_resamples) is int and 1 <= n_resamples <= 100000,
             "n_resamples must be an integer in [1,100000]")
    confidence = _number(confidence, "confidence")
    _require(0 < confidence < 1, "confidence must lie strictly between 0 and 1")
    _require(type(seed) is int and 0 <= seed < 2 ** 63, "seed must be an integer in [0,2**63)")
    tolerance = _number(tie_tolerance, "tie_tolerance", nonnegative=True)
    cases, expected = _validate_manifest(manifest)
    observed, identity, bindings = _validate_records(records, cases, expected)
    manifest_sha, records_sha = _sha(manifest), _sha(records)
    rows = []
    scopes = ("all_assistant", "explicit_action")
    values = {scope: {kind: {} for kind in KINDS[1:]} for scope in scopes}
    for case_id in sorted(cases):
        case = cases[case_id]
        conditions = case["conditions"]
        true_id = next(condition["condition_id"] for condition in conditions if condition["kind"] == "true")
        missing = [condition["condition_id"] for condition in conditions
                   if (case_id, condition["condition_id"]) not in observed]
        unavailable = case.get("unavailable_conditions", [])
        availability = {kind: any(condition["kind"] == kind for condition in conditions) for kind in KINDS}
        row = {"case_id": case_id, "source_group_id": case["source_group_id"],
               "target_sha256": case["target_sha256"],
               "condition_kind_availability": availability,
               "unavailable_conditions": [{"kind": entry["kind"], "reason": entry["reason"]}
                                          for entry in unavailable],
               "n_conditions_expected": len(conditions), "n_conditions_observed": len(conditions) - len(missing),
               "missing_condition_ids": sorted(missing), "scopes": {}}
        for scope in scopes:
            scores, support_counts = {}, {}
            for condition in conditions:
                condition_id = condition["condition_id"]
                record = observed.get((case_id, condition_id))
                mask = None if record is None else (
                    [True] * len(record["target_token_ids"]) if scope == "all_assistant"
                    else record.get("action_mask"))
                support_counts[condition_id] = sum(mask) if mask is not None else None
                selected = [] if mask is None else [float(value) for value, valid in
                                                    zip(record["token_nll"], mask) if valid]
                scores[condition_id] = _mean(selected, "condition token NLL") if selected else None
            true_score = scores[true_id]
            differences = {condition["condition_id"]:
                           (_difference(scores[condition["condition_id"]], true_score, "paired NLL difference")
                            if scores[condition["condition_id"]] is not None and true_score is not None else None)
                           for condition in conditions if condition["kind"] != "true"}
            by_kind = {}
            for kind in KINDS[1:]:
                ids = [condition["condition_id"] for condition in conditions if condition["kind"] == kind]
                paired = bool(ids) and all(differences[condition_id] is not None for condition_id in ids)
                value = _mean([differences[condition_id] for condition_id in ids], "case kind difference") if paired else None
                by_kind[kind] = {"difference": value, "n_conditions_expected": 1,
                                 "n_conditions_available": len(ids),
                                 "n_condition_pairs_observed": sum(differences[condition_id] is not None for condition_id in ids),
                                 "all_kind_condition_pairs_complete": paired}
                values[scope][kind][case_id] = value
            candidate_ids = [condition["condition_id"] for condition in conditions if condition["kind"] != "silence"]
            reviewed_ids = [condition["condition_id"] for condition in conditions
                            if condition.get("mismatch_review") == "reviewed_mismatch"]
            descriptive_rank = _rank(true_id, candidate_ids, scores, tolerance)
            if descriptive_rank is not None:
                descriptive_rank.update({"condition_kind_availability": dict(availability),
                                         "all_predeclared_candidate_kinds_available": all(availability.values()),
                                         "candidate_set_scope": "available true/random/hard only; silence excluded"})
            reviewed_rank = _rank(true_id, [true_id] + reviewed_ids, scores, tolerance)
            if reviewed_rank is not None:
                reviewed_rank.update({"condition_kind_availability": dict(availability),
                                      "all_predeclared_candidate_kinds_available": all(availability.values()),
                                      "candidate_set_scope": "explicit reviewed-mismatch subset only"})
            row["scopes"][scope] = {
                "condition_mean_nll": scores, "effective_token_counts": support_counts,
                "condition_minus_true_nll": differences, "comparisons": by_kind,
                "descriptive_candidate_rank": descriptive_rank,
                "descriptive_rank_is_accuracy": False,
                "reviewed_subset_eligible": bool(reviewed_ids),
                "reviewed_subset_rank": reviewed_rank,
                "reviewed_mismatch_condition_ids": sorted(reviewed_ids),
                "unreviewed_condition_ids": sorted(condition["condition_id"] for condition in conditions
                    if condition["kind"] in ("random", "hard") and
                    condition.get("mismatch_review", "unreviewed") == "unreviewed"),
                "known_positive_condition_ids": sorted(condition["condition_id"] for condition in conditions
                    if condition["kind"] in ("random", "hard") and
                    condition.get("mismatch_review") == "known_positive"),
            }
        rows.append(row)
    missing_records = [{"case_id": case_id, "condition_id": condition_id,
                        "kind": expected[(case_id, condition_id)]["kind"]}
                       for case_id, condition_id in sorted(set(expected) - set(observed))]
    unavailable_count = sum(len(case.get("unavailable_conditions", [])) for case in cases.values())
    return {
        "schema": "captionlab.frozen_mapping_scores.v1",
        "manifest_content_sha256": manifest_sha, "records_content_sha256": records_sha,
        "record_declared_bindings": bindings,
        "status": ("incomplete_records" if missing_records else
                   "complete_available_records_with_unavailable_conditions" if unavailable_count else "complete_records"),
        "identity": {"model_identity": identity[0], "tokenizer_identity": identity[1], "eos_token_id": identity[2]}
                    if identity is not None else None,
        "coverage": {"n_cases_expected": len(cases),
                     "n_cases_with_all_records": sum(not row["missing_condition_ids"] for row in rows),
                     "n_cases_with_all_predeclared_conditions_and_records": sum(
                         not row["missing_condition_ids"] and not row["unavailable_conditions"] for row in rows),
                     "n_conditions_predeclared": 4 * len(cases),
                     "n_conditions_unavailable": unavailable_count,
                     "n_records_expected": len(expected), "n_records_observed": len(observed),
                     "n_records_missing": len(missing_records), "missing_records": missing_records,
                     "n_source_groups_expected": len({case["source_group_id"] for case in cases.values()})},
        "aggregation": {scope: {kind: _aggregate(values[scope][kind], cases, n_resamples, confidence, seed)
                                 for kind in KINDS[1:]} for scope in scopes},
        "reviewed_subset_ranking": {scope: _ranking_summary(rows, scope) for scope in scopes},
        "tie_tolerance_nll": tolerance, "cases": rows,
        "limitations": [
            "Teacher-forced fixed-target likelihood is not generated-action accuracy or beat synchrony.",
            "All-assistant support includes EOS; explicit action support is caller-supplied and excludes EOS.",
            "Absent or empty action masks are not parsed or guessed; action diagnostics remain unavailable.",
            "Random/hard IDs alone do not establish semantic mismatches; multiple valid choreographies may exist.",
            "Reviewed-subset ranks exclude unreviewed/known-positive candidates and disclose incomplete coverage.",
            "Source-group IDs, masks, mismatch reviews and model/tokenizer identities are input declarations.",
            "Cluster intervals are conditional on complete pairs and supplied grouping, not proof of independence.",
            "Positive condition-minus-true NLL favors the true condition; silence response alone is not music discrimination.",
        ],
    }
