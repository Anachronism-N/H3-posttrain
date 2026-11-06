"""Analyze ORIGINAL review documents with a complete dataset/method denominator.

No imported summary is trusted, no missing rating becomes zero, and no model or
audio service is called. Grouping and reviewer identities remain declarations.
"""
import argparse
import copy
import json
from pathlib import Path
from statistics import mean

from .comparison import compare_methods
from .dataset import resolve_audio
from .io import file_sha, object_sha, read_json_snapshot, write_json
from .reviews import CRITERIA, import_reviews
from .schema import keys, validate_dataset


GROUP_FIELDS = ("song_group_id", "artist_group_id", "near_duplicate_group_id")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be nonempty text")


def _read(path, role):
    value, digest = read_json_snapshot(path)
    return value, dict(role=role, path=str(Path(path).resolve()), sha256=digest)


def _canonical_song(song, dataset_path):
    result = copy.deepcopy(song)
    result["audio"]["path"] = str(resolve_audio(song, dataset_path))
    return result


def _groups(dataset, selected):
    """Union every declared same-song/artist/near-duplicate/audio relation.

    The graph spans the WHOLE supplied dataset, not just the requested split;
    a transitive bridge must not become two apparently independent groups.
    """
    songs = dataset["songs"]
    parents = {s["song_id"]: s["song_id"] for s in songs}
    def find(sid):
        while parents[sid] != sid:
            parents[sid] = parents[parents[sid]]
            sid = parents[sid]
        return sid
    owners = {}
    for song in songs:
        sid = song["song_id"]
        relations = [(field, song[field]) for field in GROUP_FIELDS]
        relations.append(("audio_sha256", song["audio"]["sha256"]))
        for field, value in relations:
            if value is None:
                continue
            key = (field, value)
            if key in owners:
                parents[find(sid)] = find(owners[key])
            owners[key] = sid
    components = {}
    for song in songs:
        components.setdefault(find(song["song_id"]), []).append(song)
    for component in components.values():
        if len({s["split"] for s in component}) != 1:
            raise ValueError("A source-relation connected component crosses dataset splits")
    missing = [dict(song_id=s["song_id"], fields=[f for f in GROUP_FIELDS if s[f] is None])
               for s in songs if any(s[f] is None for f in GROUP_FIELDS)]
    eligible = dataset["grouping"]["status"] == "reviewed" and not missing
    if not eligible:
        return {s["song_id"]: None for s in selected}, dict(status="unreviewed_or_incomplete_descriptive_only",
            usable_for_group_comparison=False, missing_group_fields=missing, components=None,
            n_selected_components=None, independence_verified=False,
            reason="Require grouping.status=reviewed with evidence and every group field for the complete input dataset")
    mapping, descriptions = {}, []
    for component in sorted(components.values(), key=lambda rows: min(s["song_id"] for s in rows)):
        members = sorted(s["song_id"] for s in component)
        group_id = "component:" + object_sha(members)
        for sid in members:
            mapping[sid] = group_id
        descriptions.append(dict(group_id=group_id, song_ids=members, split=component[0]["split"]))
    return mapping, dict(status="reviewed_declared_connected_components", usable_for_group_comparison=True,
        missing_group_fields=[], components=descriptions,
        n_selected_components=len({mapping[s["song_id"]] for s in selected}), independence_verified=False,
        relation_policy="Connected closure of same song_group_id, artist_group_id, near_duplicate_group_id or audio SHA; relation types are namespaced")


def _reviewer_roster(path, pack):
    document, reference = _read(path, 'reviewer_roster')
    keys(document, ('schema', 'pack_sha256', 'assignments'), where='reviewer roster')
    if document['schema'] != 'choreobench.reviewer_roster.v1' or document['pack_sha256'] != object_sha(pack):
        raise ValueError('Reviewer roster schema or public pack binding differs')
    assignments = document['assignments']
    if not isinstance(assignments, list) or not assignments:
        raise ValueError('Reviewer roster assignments must be a nonempty list')
    items = {item['review_id']: item for item in pack['items']}
    criteria = {c['criterion_id'] for c in CRITERIA}
    identities, seen = {}, set()
    for row in assignments:
        keys(row, ('review_id', 'reviewer_id', 'origin', 'model_id', 'criteria'), where='reviewer assignment')
        for field in ('review_id', 'reviewer_id'):
            _text(row[field], 'assignment.' + field)
        if row['review_id'] not in items or row['reviewer_id'] == 'FILL_REVIEWER_ID':
            raise ValueError('Reviewer assignment requires a known item and a real declared reviewer ID')
        if row['origin'] == 'human':
            if row['model_id'] is not None:
                raise ValueError('Human roster assignment cannot declare model_id')
        elif row['origin'] == 'model_judge':
            _text(row['model_id'], 'assignment.model_id')
        else:
            raise ValueError('Reviewer assignment origin must be human/model_judge')
        identity = (row['origin'], row['model_id'])
        rid = row['reviewer_id']
        if rid in identities and identities[rid] != identity:
            raise ValueError('Roster reviewer changes origin/model identity')
        identities[rid] = identity
        pair = (row['review_id'], rid)
        if pair in seen:
            raise ValueError('Duplicate reviewer/item assignment')
        seen.add(pair)
        requested = row['criteria']
        if (not isinstance(requested, list) or not requested
                or any(not isinstance(c, str) or c not in criteria for c in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError('Assignment criteria must be a nonempty unique list of known criteria')
        if items[row['review_id']]['edit_task'] is None and any(c.startswith('edit_') for c in requested):
            raise ValueError('Cannot assign edit criteria to a generation-only review item')
    return document, reference


def _roster_coverage(assignments, relevant, cells, criterion, origin, judge_model_id):
    by_id = {item['review_id']: key for key, item in relevant.items()}
    expected, excluded, assigned_cells = [], [], set()
    for assignment in assignments:
        key = by_id.get(assignment['review_id'])
        if (key is None or criterion not in assignment['criteria'] or assignment['origin'] != origin
                or origin == 'model_judge' and assignment['model_id'] != judge_model_id):
            excluded.append(copy.deepcopy(assignment))
            continue
        assigned_cells.add(key)
        review = next((row for row in relevant[key]['reviews']
            if row['reviewer_id'] == assignment['reviewer_id'] and row['origin'] == origin
            and row['model_id'] == assignment['model_id']), None)
        status = ('missing_review_record' if review is None else
                  'pending_criterion_rating' if review['ratings'].get(criterion) is None else 'rated')
        expected.append(dict(song_id=key[0], method_id=key[1], review_id=assignment['review_id'],
            reviewer_id=assignment['reviewer_id'], status=status))
    unassigned = [dict(song_id=c['song_id'], method_id=c['method_id'], review_id=c['review_id'])
                  for c in cells if (c['song_id'], c['method_id']) not in assigned_cells]
    reviewers = []
    for rid in sorted({row['reviewer_id'] for row in expected}):
        rows = [row for row in expected if row['reviewer_id'] == rid]
        submitted = sum(row['status'] != 'missing_review_record' for row in rows)
        reviewers.append(dict(reviewer_id=rid, expected_records=len(rows), submitted_records=submitted,
            rated_records=sum(row['status'] == 'rated' for row in rows), entirely_unsubmitted=submitted == 0))
    rated = sum(row['status'] == 'rated' for row in expected)
    return dict(provided=True, expected_assignment_count=len(expected),
        submitted_assignment_count=sum(row['status'] != 'missing_review_record' for row in expected),
        rated_assignment_count=rated, assignments=expected,
        missing_or_unrated_assignments=[row for row in expected if row['status'] != 'rated'],
        unassigned_song_method_cells=unassigned, excluded_assignments=excluded, reviewers=reviewers,
        entirely_unsubmitted_reviewer_ids=[row['reviewer_id'] for row in reviewers if row['entirely_unsubmitted']],
        complete_for_roster=bool(expected) and rated == len(expected) and not unassigned,
        scope='Selected dataset split, two methods, requested criterion and reviewer origin/model only',
        reviewer_identity_or_actual_listening_authenticated=False,
        preregistration_timing_independently_verified=False)


def analyze_reviews(dataset_path, public_path, bindings_path, reviews_paths, *, method_a, method_b,
                    criterion, split, origin, judge_model_id=None, compare=False,
                    n_resamples=2000, confidence=.95, seed=7, reviewer_roster_path=None):
    """Return source-bound rows, complete coverage and optional group comparison.

    A larger pack may be analyzed against each ORIGINAL per-song dataset. Every
    excluded item is reported, never claimed verified against the current
    dataset. Relevant items require its exact byte SHA and full song equality,
    except equivalent absolute/relative audio path representations.
    """
    for name, value in (("method_a", method_a), ("method_b", method_b)):
        _text(value, name)
    if method_a == method_b:
        raise ValueError("method_a and method_b must differ")
    if criterion not in {c["criterion_id"] for c in CRITERIA}:
        raise ValueError("Unknown review criterion")
    if split not in ("train", "dev", "test") or origin not in ("human", "model_judge"):
        raise ValueError("Explicit split and human/model_judge origin are required")
    if origin == "model_judge":
        _text(judge_model_id, "judge_model_id")
    elif judge_model_id is not None:
        raise ValueError("human analysis cannot specify judge_model_id")
    if type(compare) is not bool or type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("compare must be boolean and n_resamples must be positive")
    if type(confidence) not in (int, float) or not 0 < confidence < 1 or type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("Invalid comparison confidence/seed")
    if isinstance(reviews_paths, (str, Path)):
        reviews_paths = [reviews_paths]
    if not isinstance(reviews_paths, (list, tuple)) or not reviews_paths:
        raise ValueError("Provide one or more original reviews files")
    dataset, data_ref = _read(dataset_path, "dataset")
    pack, pack_ref = _read(public_path, "public_pack")
    bindings, bindings_ref = _read(bindings_path, "private_bindings")
    originals, refs = [], []
    for index, path in enumerate(reviews_paths):
        document, ref = _read(path, "reviews:%d" % index)
        import_reviews(pack, bindings, document)  # Validate every original separately.
        originals.extend(copy.deepcopy(document["reviews"]))
        refs.append(ref)
    merged = dict(schema="choreobench.reviews.v1", pack_sha256=object_sha(pack), reviews=originals)
    imported = import_reviews(pack, bindings, merged)  # Reject duplicates across files too.
    validate_dataset(dataset)
    selected = [s for s in dataset["songs"] if s["split"] == split]
    if not selected:
        raise ValueError("No dataset songs in selected split")
    selected_by_id = {s["song_id"]: s for s in selected}
    all_ids = {s["song_id"] for s in dataset["songs"]}
    requested = (method_a, method_b)
    private = {b["review_id"]: b for b in bindings["items"]}
    relevant, excluded_items, source_sets = {}, [], {m: {} for m in requested}
    for item in imported["items"]:
        sid, method = item["song_id"], item["method_id"]
        reason = ("outside_dataset" if sid not in all_ids else "outside_selected_split" if sid not in selected_by_id
                  else "method_not_selected" if method not in requested else None)
        if reason:
            excluded_items.append(dict(review_id=item["review_id"], song_id=sid, method_id=method,
                reason=reason, original_pack_import_validated=True, current_dataset_provenance_checked=False))
            continue
        original = private[item["review_id"]]["original"]
        source = original.get("source_binding")
        if source is None or source["dataset_sha256"] != data_ref["sha256"]:
            raise ValueError("Selected review item lacks the exact current dataset byte binding: " + item["review_id"])
        if object_sha(_canonical_song(original["song"], dataset_path)) != object_sha(_canonical_song(selected_by_id[sid], dataset_path)):
            raise ValueError("Selected original song differs from dataset beyond equivalent audio path: " + sid)
        key = (sid, method)
        if key in relevant:
            raise ValueError("Ambiguous multiple plan/version review items for the same song and method; reviewers must rate one item")
        relevant[key] = item
        source_sets[method][object_sha(source)] = copy.deepcopy(source)
    roster, roster_ref = None, None
    assigned = set()
    if reviewer_roster_path is not None:
        roster, roster_ref = _reviewer_roster(reviewer_roster_path, pack)
        assigned = {(row['review_id'], row['reviewer_id'], row['origin'], row['model_id'])
                    for row in roster['assignments'] if criterion in row['criteria']}
    group_ids, grouping = _groups(dataset, selected)
    rows, cells, excluded_ratings, reviewer_versions = [], [], [], {}
    for song in selected:
        sid = song["song_id"]
        for method in requested:
            item = relevant.get((sid, method))
            scored, pending, selected_reviews = [], [], []
            if item:
                for review in item["reviews"]:
                    rid = review["reviewer_id"]
                    identity = (review["origin"], review["model_id"])
                    if rid in reviewer_versions and reviewer_versions[rid] != identity:
                        raise ValueError("One reviewer_id changes origin/model version across selected review items")
                    reviewer_versions[rid] = identity
                    exclude = ("origin_not_selected" if review["origin"] != origin else
                               "judge_model_not_selected" if origin == "model_judge" and review["model_id"] != judge_model_id else None)
                    if not exclude and roster is not None and (item['review_id'], rid, origin, review['model_id']) not in assigned:
                        exclude = 'reviewer_not_assigned_for_criterion'
                    if exclude:
                        excluded_ratings.append(dict(review_id=item['review_id'], reviewer_id=rid,
                            origin=review["origin"], model_id=review["model_id"], reason=exclude))
                        continue
                    selected_reviews.append(review)
                    value = review["ratings"].get(criterion)
                    if value is None:
                        pending.append(rid)
                        continue
                    flat = dict(song_id=sid, group_id=group_ids[sid], method_id=method, criterion=criterion,
                        value=value, reviewer_id=rid, origin=origin, model_id=review["model_id"],
                        review_id=item["review_id"], plan_sha256=item["plan_sha256"], audio_seen=review["audio_seen"])
                    rows.append(flat)
                    scored.append(flat)
            if item is None:
                status = "missing_plan_in_pack"
            elif item["criteria_status"][criterion] == "not_applicable":
                status = "criterion_not_applicable"
            elif not selected_reviews:
                status = "no_selected_reviewer_ratings"
            elif not scored:
                status = "pending_criterion_rating"
            else:
                status = "rated"
            cells.append(dict(song_id=sid, method_id=method, review_id=item["review_id"] if item else None,
                status=status, n_ratings=len(scored), n_selected_reviews=len(selected_reviews),
                pending_reviewer_ids=sorted(pending), reviewer_ids=sorted(r["reviewer_id"] for r in scored),
                mean_rating=mean(r["value"] for r in scored) if scored else None))
    by_cell = {(c["song_id"], c["method_id"]): c for c in cells}
    descriptions = []
    for song in selected:
        a, b = [by_cell[(song["song_id"], method)] for method in requested]
        descriptions.append(dict(song_id=song["song_id"], group_id=group_ids[song["song_id"]],
            methods={method_a: a, method_b: b}, paired_ratings_available=a["mean_rating"] is not None and b["mean_rating"] is not None,
            difference_b_minus_a=b["mean_rating"] - a["mean_rating"] if a["mean_rating"] is not None and b["mean_rating"] is not None else None))
    complete = all(c['status'] == 'rated' for c in cells)
    roster_report = (_roster_coverage(roster['assignments'], relevant, cells, criterion, origin, judge_model_id)
                     if roster is not None else dict(provided=False, complete_for_roster=None,
                         note='No preregistered roster supplied; complete_for_requested_criterion does not verify every expected reviewer'))
    blockers = []
    if roster is not None and not roster_report['complete_for_roster']:
        blockers.append('Incomplete registered reviewer x item x criterion assignments; no roster-complete comparison')
    if not complete:
        blockers.append("Incomplete full split x methods inventory/criterion ratings; no intersection-only CI")
    if not grouping["usable_for_group_comparison"]:
        blockers.append("Source grouping is not fully reviewed and complete; null group IDs are descriptive only")
    comparison = dict(requested=compare, status="not_requested" if not compare else "blocked" if blockers else "computed",
                      blockers=blockers, result=None)
    if compare and not blockers:
        comparison["result"] = compare_methods(rows, method_a, method_b, n_resamples=n_resamples,
            confidence=confidence, seed=seed, value_bounds=[1, 5])
        comparison["status"] = comparison["result"]["status"]
    sources = [data_ref, pack_ref, bindings_ref] + refs + ([roster_ref] if roster_ref is not None else [])
    for ref in sources:
        if file_sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Review analysis source changed while reading")
    synthetic = bool(dataset["synthetic"] or any(s["synthetic"] for values in source_sets.values() for s in values.values()))
    report = dict(schema="choreobench.review_analysis.v1", sources=sources, source_pack_sha256=object_sha(pack),
        imported_originals_sha256=object_sha(merged), validated_import_sha256=object_sha(imported),
        split=split, method_a=method_a, method_b=method_b, criterion=criterion, origin=origin,
        judge_model_id=judge_model_id, rows=rows, grouping=grouping, songs=descriptions,
        reviewer_roster=roster_report,
        coverage=dict(selected_song_ids=[s["song_id"] for s in selected], expected_cells=len(cells),
            observed_plan_cells=len(relevant), rated_cells=sum(c["status"] == "rated" for c in cells),
            complete_for_requested_criterion=complete, rating_rows=len(rows), cells=cells,
            missing_or_unrated_cells=[copy.deepcopy(c) for c in cells if c["status"] != "rated"],
            pack_total_items=len(imported["items"]), excluded_items=excluded_items, excluded_ratings=excluded_ratings),
        method_sources={m: dict(source_bindings=list(source_sets[m].values()),
            generation_model_identity_verified=False, multiple_source_records=len(source_sets[m]) > 1) for m in requested},
        comparison=comparison, synthetic=synthetic, scientific_quality_verified=False, audio_bytes_checked=False,
        implementation_sha256={name: file_sha(Path(__file__).with_name(name)) for name in
            ("review_analysis.py", "reviews.py", "comparison.py", "dataset.py", "schema.py", "io.py")},
        limitations=[
            "No missing/failed plan, null rating or inapplicable criterion is a numeric zero. Full split denominator comes from the dataset, never the observed rating rows.",
            "A missing plan here may be generation failure or absent packaging; consult the original submission/results for its actual failure cause.",
            "Origins, model IDs, source/grouping declarations and audio_seen are not independent authentication or proof of listening.",
            "Source hashes may differ for legitimate per-song runs; identical generation-model versions must be audited from their original run contracts, not inferred from a method label.",
            "Comparison is paired by song, not necessarily reviewer. A supplied roster gates completion but does not authenticate preregistration or listening; no ICC, rater-effect model, power analysis, significance or scientific victory is inferred.",
            "Three or fewer songs are a small development description, not a population or leaderboard conclusion." if len(selected) <= 3 else
            "Sample size and representativeness require an independent pilot/power and coverage design.",
            "Excluded pack items passed original pack/import validation but were not checked against this dataset; per-song dataset bindings cannot be silently replaced by a merged manifest.",
        ])
    report["record_sha256"] = object_sha(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "public", "private", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--reviews", required=True, nargs="+", type=Path)
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--criterion", required=True, choices=[c["criterion_id"] for c in CRITERIA])
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--origin", required=True, choices=("human", "model_judge"))
    parser.add_argument("--judge-model-id")
    parser.add_argument("--reviewer-roster", type=Path)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    report = analyze_reviews(args.dataset, args.public, args.private, args.reviews,
        method_a=args.method_a, method_b=args.method_b, criterion=args.criterion, split=args.split,
        origin=args.origin, judge_model_id=args.judge_model_id, compare=args.compare,
        n_resamples=args.resamples, confidence=args.confidence, seed=args.seed, reviewer_roster_path=args.reviewer_roster)
    write_json(args.out, report)
    print(json.dumps(dict(out=str(args.out), rating_rows=len(report["rows"]),
        complete=report["coverage"]["complete_for_requested_criterion"], comparison=report["comparison"]["status"])))
    return report


if __name__ == "__main__":
    main()
