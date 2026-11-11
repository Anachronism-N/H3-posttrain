"""Agreement diagnostics for bound human and model reviews, without new inference."""
import argparse
import math
import random
from collections import Counter, defaultdict
from statistics import mean

from choreobench.evaluation import _percentile
from choreobench.io import read_json, write_json
from .choreo_quality import DIMENSIONS, REFERENCE_DIMENSION, summarize_quality


def _ranks(values):
    positions = defaultdict(list)
    for index, value in enumerate(sorted(values), 1):
        positions[value].append(index)
    return [mean(positions[value]) for value in values]


def spearman(left, right):
    """Average ranks for ties; constant or insufficient samples are undefined."""
    if len(left) != len(right):
        raise ValueError("Paired arrays differ in length")
    if len(left) < 2:
        return None
    a, b = _ranks(left), _ranks(right)
    ma, mb = mean(a), mean(b)
    norm = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / norm if norm else None


def quadratic_kappa(left, right):
    """Quadratic weighted Cohen kappa on the frozen ordinal 1..5 scale."""
    if len(left) != len(right):
        raise ValueError("Paired arrays differ in length")
    if any(type(x) is not int or not 1 <= x <= 5 for x in list(left) + list(right)):
        raise ValueError("Kappa requires integer scores 1..5")
    if len(left) < 2:
        return None
    a, b, n = Counter(left), Counter(right), len(left)
    observed = sum((x - y) ** 2 for x, y in zip(left, right)) / n
    expected = sum(a[x] * b[y] * (x - y) ** 2 for x in a for y in b) / n ** 2
    return 1 - observed / expected if expected else None


def _estimate(rows, metric, n_resamples, seed):
    point = metric([row[1] for row in rows], [row[2] for row in rows])
    groups = defaultdict(list)
    for row in rows:
        groups[row[0]].append(row)
    samples, keys, rng = [], sorted(groups), random.Random(seed)
    if len(keys) >= 2 and point is not None:
        for _ in range(n_resamples):
            drawn = [row for _ in keys for row in groups[rng.choice(keys)]]
            value = metric([row[1] for row in drawn], [row[2] for row in drawn])
            if value is not None:
                samples.append(value)
    return {"estimate": point, "interval_95": [_percentile(samples, .025), _percentile(samples, .975)] if samples else None,
            "defined_bootstrap_samples": len(samples), "requested_bootstrap_samples": n_resamples,
            "paired_outputs": len(rows), "source_groups": len(groups),
            "undefined_resamples_are_omitted": True,
            "status": "undefined" if point is None else "point_only" if not samples else "computed"}


def validate_metric(items, reviews, *, humans, judge=None, n_resamples=2000, seed=42):
    """Compare the same outputs. Scores are conditional on complete paired reviews.

    Human means use ALL specified humans; missing values are never imputed. The
    group bootstrap preserves correlated methods within each source group. It
    does not establish song-level independence or authenticate reviewer identity.
    """
    if not isinstance(humans, list) or len(humans) != 2 or len(set(humans)) != 2:
        raise ValueError("Exactly two distinct human reviewer IDs are required")
    if any(not isinstance(h, str) or not h.strip() for h in humans):
        raise ValueError("Invalid human reviewer ID")
    if judge is not None and (not isinstance(judge, str) or not judge.strip() or judge in humans):
        raise ValueError("Judge must be a distinct nonempty reviewer ID")
    # Reuse the authoritative binding, rating, identity and duplicate validation.
    summarize_quality(items, reviews, n_resamples=n_resamples, seed=seed)
    indexed = {(row["item_sha256"], row["reviewer_id"]): row for row in reviews}
    for row in reviews:
        if row["reviewer_id"] in humans and row["origin"] != "human":
            raise ValueError("Expected human reviewer is labelled model_judge")
        if row["reviewer_id"] == judge and row["origin"] != "model_judge":
            raise ValueError("Expected model judge is labelled human")

    def score(item, reviewer, dimension):
        row = indexed.get((item["item_sha256"], reviewer))
        if item["generation_status"] != "ok" or not row or row["review_status"] != "completed":
            return None
        rating = row.get(dimension) if dimension == REFERENCE_DIMENSION else row["ratings"][dimension]
        return rating["score"] if rating and rating["status"] == "rated" else None

    dimensions = {}
    for dimension in DIMENSIONS + (REFERENCE_DIMENSION,):
        human_pairs, judge_pairs = [], []
        for item in items:
            values = [score(item, h, dimension) for h in humans]
            if all(value is not None for value in values):
                human_pairs.append((item["source_group_id"], *values))
                value = score(item, judge, dimension) if judge else None
                if value is not None:
                    judge_pairs.append((item["source_group_id"], mean(values), value))
        dimensions[dimension] = {
            "registered_outputs": len(items),
            "human_pair_coverage": len(human_pairs) / len(items),
            "judge_human_pair_coverage": len(judge_pairs) / len(items) if judge else None,
            "human_quadratic_kappa": _estimate(human_pairs, quadratic_kappa, n_resamples, seed),
            "judge_vs_human_mean_spearman": _estimate(judge_pairs, spearman, n_resamples, seed) if judge else None,
        }
    return {"schema": "captionlab.metric_validation.v1", "humans": humans, "judge": judge,
            "dimensions": dimensions, "conditional_on_complete_pairs": True,
            "resampling_unit": "declared source_group_id; song independence not verified",
            "claim": "Agreement on the supplied frozen evaluation set; not proof of validity or novel metric",
            "seed": seed, "n_resamples": n_resamples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--human", action="append", required=True)
    parser.add_argument("--judge")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    items, reviews = read_json(args.items), read_json(args.reviews)
    items = items["items"] if isinstance(items, dict) else items
    reviews = reviews["reviews"] if isinstance(reviews, dict) else reviews
    write_json(args.out, validate_metric(items, reviews, humans=args.human, judge=args.judge,
                                       n_resamples=args.n_resamples, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
