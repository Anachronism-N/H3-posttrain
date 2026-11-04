"""Paired, equal-song comparisons with source-group cluster bootstrap.

Input rows are observations, not independent experimental units. The primary
estimate first averages ratings within each song/method, then averages paired
B-minus-A differences equally across songs. Bootstrap replicates draw groups
with replacement and retain every paired song in each sampled group. Equal-group
weighting is reported separately and never silently replaces the estimand.
"""

import math
import random
from fractions import Fraction
from statistics import mean

from .io import finite_json, object_sha
from .schema import ValidationError


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _text(value, label):
    _require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")


def _number(value, label):
    _require(type(value) in (int, float), label + " must be a finite number, not a boolean or missing rating")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValidationError(label + " exceeds the finite floating-point range") from error
    _require(math.isfinite(result), label + " must be finite")
    if type(value) is int:
        _require(int(result) == value, label + " loses integer precision in the comparison's floating-point range")
    return result


def _finite_mean(values, label):
    try:
        return _number(mean(values), label)
    except (OverflowError, ValueError) as error:
        raise ValidationError(label + " cannot be represented as a finite number") from error


def _percentile(values, quantile):
    position = (len(values) - 1) * quantile
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return values[lower]
    # Interpolation is a convex combination; intermediate sums must not overflow.
    weight = Fraction.from_float(position - lower)
    result = (1 - weight) * Fraction.from_float(values[lower]) + weight * Fraction.from_float(values[upper])
    return _number(float(result), "bootstrap percentile")


def compare_methods(rows, method_a, method_b, *, metric="value", n_resamples=2000,
                    confidence=0.95, seed=0, allow_incomplete=False, value_bounds=None):
    """Compare two methods without silently discarding missing song/method cells.

    Rows require song_id, group_id, method_id and the selected numeric metric;
    reviewer_id is optional but must be nonempty when supplied. Duplicate named
    reviewer ratings in a song/method cell are rejected. Other metric columns and
    JSON metadata are allowed. Integers must be exactly representable in the
    floating-point comparison range rather than silently rounded. No rating scale
    is inferred: value_bounds may be a
    declared [minimum, maximum] with inclusive endpoints.

    The song universe is every song_id appearing in rows, including songs seen
    only for other methods. Strict mode requires A and B for that whole universe.
    allow_incomplete=True explicitly compares their intersection and reports all
    omissions. Songs absent from the input entirely cannot be detected here.

    Ratings need not come from identical reviewer sets; such mismatches and
    missing reviewer identities are disclosed. This is a paired SONG comparison,
    not a claim of reviewer-paired judgments or independently verified grouping.
    Optional origin declarations must be homogeneous human/model_judge across all
    requested-method rows, including rows later omitted by intersection mode.
    All-omitted origins are accepted and reported as unknown, never inferred.
    """
    _require(isinstance(rows, list), "comparison rows must be a list")
    _text(method_a, "method_a")
    _text(method_b, "method_b")
    _require(method_a != method_b, "method_a and method_b must differ")
    _text(metric, "metric")
    _require(metric not in {"song_id", "group_id", "method_id", "reviewer_id"},
             "metric must not name an identity field")
    _require(type(n_resamples) is int and n_resamples > 0, "n_resamples must be a positive integer")
    confidence = _number(confidence, "confidence")
    _require(0 < confidence < 1, "confidence must be between zero and one")
    _require(type(seed) is int and 0 <= seed < 2 ** 63, "seed must be an integer in [0,2**63)")
    _require(type(allow_incomplete) is bool, "allow_incomplete must be boolean")
    bounds = None
    if value_bounds is not None:
        _require(isinstance(value_bounds, (list, tuple)) and len(value_bounds) == 2,
                 "value_bounds must contain exactly minimum and maximum")
        bounds = [_number(value_bounds[0], "minimum bound"), _number(value_bounds[1], "maximum bound")]
        _require(bounds[0] < bounds[1], "minimum bound must be less than maximum bound")

    song_groups, cells, reviewers, unnamed, methods = {}, {}, {}, {}, set()
    selected_origins, selected_origin_missing = [], 0
    named_rows, anonymous_rows = 0, 0
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), "every comparison row must be an object")
        for field in ("song_id", "group_id", "method_id"):
            _text(row.get(field), "row {} {}".format(index, field))
        song_id, group_id, method_id = row["song_id"], row["group_id"], row["method_id"]
        value = _number(row.get(metric), "row {} {}".format(index, metric))
        if bounds is not None:
            _require(bounds[0] <= value <= bounds[1], "rating outside declared value_bounds")
        if song_id in song_groups:
            _require(song_groups[song_id] == group_id,
                     "song {} has inconsistent group_id across rows/methods".format(song_id))
        song_groups[song_id] = group_id
        key = (song_id, method_id)
        methods.add(method_id)
        if method_id in (method_a, method_b):
            if "origin" in row:
                _require(isinstance(row["origin"], str) and row["origin"] in {"human", "model_judge"},
                         "selected rating origin must be human or model_judge")
                selected_origins.append(row["origin"])
            else:
                selected_origin_missing += 1
        cells.setdefault(key, []).append(value)
        reviewers.setdefault(key, set())
        unnamed.setdefault(key, 0)
        if "reviewer_id" in row:
            _text(row["reviewer_id"], "reviewer_id")
            _require(row["reviewer_id"] not in reviewers[key],
                     "duplicate reviewer_id for song {} and method {}".format(song_id, method_id))
            reviewers[key].add(row["reviewer_id"])
            named_rows += 1
        else:
            unnamed[key] += 1
            anonymous_rows += 1
    try:
        finite_json(rows)
    except ValueError as error:
        raise ValidationError("comparison input metadata must be finite JSON") from error
    if selected_origins:
        _require(selected_origin_missing == 0, "selected ratings mix declared origin and unknown origin")
        _require(len(set(selected_origins)) == 1, "selected ratings mix human and model_judge origins")
        rating_origin = selected_origins[0]
    else:
        rating_origin = "unknown"

    requested = (method_a, method_b)
    universe = sorted(song_groups)
    included, omitted = [], []
    for song_id in universe:
        present = [method for method in requested if (song_id, method) in cells]
        missing = [method for method in requested if (song_id, method) not in cells]
        if missing:
            omitted.append({"song_id": song_id, "group_id": song_groups[song_id],
                            "present_methods": present, "missing_methods": missing})
        else:
            included.append(song_id)
    if not allow_incomplete:
        _require(bool(universe), "strict comparison requires at least one song and both methods")
        _require(not omitted, "incomplete paired comparison; missing song/method cells: " +
                 "; ".join(item["song_id"] + "=" + ",".join(item["missing_methods"]) for item in omitted))

    pairs, group_values = [], {}
    for song_id in included:
        key_a, key_b = (song_id, method_a), (song_id, method_b)
        value_a = _finite_mean(cells[key_a], "method A song mean")
        value_b = _finite_mean(cells[key_b], "method B song mean")
        difference = _number(value_b - value_a, "paired B-minus-A difference")
        if unnamed[key_a] or unnamed[key_b]:
            reviewer_status = "not_fully_identified"
        elif reviewers[key_a] == reviewers[key_b]:
            reviewer_status = "same_identified_set"
        else:
            reviewer_status = "different_identified_sets"
        pairs.append({
            "song_id": song_id, "group_id": song_groups[song_id],
            "method_a_mean": value_a, "method_b_mean": value_b, "difference": difference,
            "method_a_rows": len(cells[key_a]), "method_b_rows": len(cells[key_b]),
            "reviewer_pairing_status": reviewer_status,
            "method_a_identified_reviewers": sorted(reviewers[key_a]),
            "method_b_identified_reviewers": sorted(reviewers[key_b]),
            "shared_identified_reviewer_count": len(reviewers[key_a] & reviewers[key_b]),
            "method_a_rows_without_reviewer_id": unnamed[key_a],
            "method_b_rows_without_reviewer_id": unnamed[key_b],
        })
        group_values.setdefault(song_groups[song_id], []).append(difference)
    group_ids = sorted(group_values)
    groups = [{"group_id": group_id, "n_songs": len(group_values[group_id]),
               "mean_difference": _finite_mean(group_values[group_id], "group mean difference")}
              for group_id in group_ids]
    requested_row_count = sum(len(values) for (song_id, method), values in cells.items() if method in requested)
    included_row_count = sum(pair["method_a_rows"] + pair["method_b_rows"] for pair in pairs)
    report = {
        "schema": "choreobench.paired_comparison.v1", "rows_sha256": object_sha(rows),
        "method_a": method_a, "method_b": method_b, "metric": metric,
        "rating_origin": rating_origin,
        "difference_direction": "method_b_minus_method_a", "value_bounds": bounds,
        "estimate": None, "method_a_mean": None, "method_b_mean": None,
        "equal_group_mean_difference": None, "interval": None,
        "status": "no_paired_songs", "n_songs": len(pairs), "n_independent_groups": len(groups),
        "n_resamples": n_resamples, "confidence": confidence, "seed": seed,
        "comparison_unit": "song; mean within song/method, then equal weight per paired song",
        "resampling_unit": "group_id; sample whole source groups with replacement, retaining their songs",
        "coverage": {
            "song_universe": "all song_ids appearing anywhere in the input rows",
            "input_rows": len(rows), "input_songs": len(universe),
            "input_groups": len(set(song_groups.values())),
            "requested_methods": list(requested),
            "method_row_counts": {method: sum(len(values) for (_, name), values in cells.items() if name == method)
                                  for method in requested},
            "method_song_counts": {method: sum((song_id, method) in cells for song_id in universe) for method in requested},
            "paired_songs": len(pairs), "omitted_songs": omitted,
            "omitted_song_count": len(omitted), "paired_fraction": len(pairs) / len(universe) if universe else None,
            "included_rating_rows": included_row_count,
            "omitted_requested_method_rows": requested_row_count - included_row_count,
            "ignored_method_ids": sorted(methods - set(requested)),
            "ignored_method_rows": len(rows) - requested_row_count,
            "rows_with_reviewer_id": named_rows, "rows_without_reviewer_id": anonymous_rows,
            "selected_rows_with_declared_origin": len(selected_origins),
            "selected_rows_without_declared_origin": selected_origin_missing,
            "origin_validation_scope": "all requested method rows before intersection selection",
            "intersection_requested": allow_incomplete,
            "complete_for_observed_song_universe": not omitted and bool(universe),
        },
        "song_pairs": pairs, "groups": groups,
        "limitations": [
            "Positive means B is numerically larger; whether larger is better depends on the selected metric.",
            "Input rows alone cannot identify benchmark songs omitted by every method. Audit against the benchmark manifest.",
            "Source group IDs are externally declared; grouping correctness and true group independence are not verified here.",
            "The comparison is paired by song, not necessarily by reviewer. Shared reviewer effects are not bootstrapped.",
            "Anonymous ratings cannot be deduplicated by reviewer identity; row counts are not independent sample sizes.",
            "Origin labels are external declarations, not authenticated human/model identities. All-unspecified origin remains unknown and cannot verify rating provenance.",
            "Intersection comparisons describe only retained songs and can be biased by method failures or selective missingness.",
            "Percentile cluster bootstrap is descriptive uncertainty; no p-value, significance claim or quality gate is inferred.",
            "External ratings are judgments, not proof of physical feasibility or measured music-motion synchronization.",
        ],
    }
    if not pairs:
        return report
    report["estimate"] = _finite_mean([pair["difference"] for pair in pairs], "paired mean difference")
    report["method_a_mean"] = _finite_mean([pair["method_a_mean"] for pair in pairs], "equal-song method A mean")
    report["method_b_mean"] = _finite_mean([pair["method_b_mean"] for pair in pairs], "equal-song method B mean")
    report["equal_group_mean_difference"] = _finite_mean([group["mean_difference"] for group in groups], "equal-group diagnostic")
    if len(groups) < 2:
        report["status"] = "insufficient_independent_groups"
        return report

    # Exact rational cluster sums avoid intermediate overflow and preserve group
    # multiplicity without materializing a potentially huge repeated song list.
    sums = {group_id: sum((Fraction.from_float(value) for value in group_values[group_id]), Fraction(0))
            for group_id in group_ids}
    counts = {group_id: len(group_values[group_id]) for group_id in group_ids}
    rng, samples = random.Random(seed), []
    for _ in range(n_resamples):
        total, count = Fraction(0), 0
        for _ in group_ids:
            group_id = group_ids[rng.randrange(len(group_ids))]
            total += sums[group_id]
            count += counts[group_id]
        samples.append(_number(float(total / count), "cluster bootstrap replicate"))
    samples.sort()
    alpha = (1 - confidence) / 2
    report["interval"] = [_percentile(samples, alpha), _percentile(samples, 1 - alpha)]
    report["status"] = "computed"
    return report
