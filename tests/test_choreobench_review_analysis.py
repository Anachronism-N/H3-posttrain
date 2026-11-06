"""Original review fixtures; illustrative scores never imply actual human data."""
import copy
from pathlib import Path

import pytest

from choreobench import review_analysis as analysis
from choreobench.dataset import resolve_audio
from choreobench.io import object_sha, read_json, write_json
from choreobench.reviews import blank_review_template, build_review_pack


A, B = "method-alpha", "method-bravo"


def song(index):
    return dict(song_id="song%d" % index, split="dev", song_group_id="source%d" % index,
        artist_group_id="artist%d" % index, near_duplicate_group_id="near%d" % index,
        brief="Fixture planned choreography, not a real benchmark item.",
        audio=dict(path="song%d.wav" % index, sha256=object_sha(index), sample_rate=16000, num_samples=2880000),
        music=dict(source="fixture", beats=[], downbeats=[], sections=[dict(section_id="whole", start=0,
            end=2880000, label="unknown", repeat_group=None)]))


def plan(s, variant=""):
    return dict(schema="choreobench.plan.v1", song_id=s["song_id"], sample_rate=16000,
        num_samples=s["audio"]["num_samples"], motifs=[], provenance={"fixture_version": variant}, events=[dict(
            event_id="e1", start=0, end=s["audio"]["num_samples"], track="body", action="Travel softly and lift both arms. " + variant,
            attributes={}, entry={}, exit={}, motif_id=None, evidence=[])])


def material(tmp_path, *, count=3, reviewed=True, include=None, mutate_songs=None,
             mutate_items=None, source_by_song=False):
    songs = [song(i + 1) for i in range(count)]
    if mutate_songs:
        mutate_songs(songs)
    dataset = dict(schema="choreobench.dataset.v1", dataset_id="fixture", synthetic=True,
        grouping=dict(status="reviewed" if reviewed else "unverified", evidence="Fixture audit declaration" if reviewed else ""), songs=songs)
    dataset_path = tmp_path / "dataset.json"
    write_json(dataset_path, dataset)
    digest = analysis.file_sha(dataset_path)
    items = []
    for s in songs:
        for method in (A, B):
            if include is not None and (s["song_id"], method) not in include:
                continue
            absolute = copy.deepcopy(s)
            absolute["audio"]["path"] = str(resolve_audio(s, dataset_path))
            items.append(dict(song=absolute, plan=plan(s), method_id=method,
                source_binding=dict(kind="run", sha256=object_sha((s["song_id"], method) if source_by_song else method),
                    dataset_sha256=digest, synthetic=True)))
    if mutate_items:
        mutate_items(items)
    paths = publish_pack(tmp_path, items)
    return dataset_path, paths[0], paths[1], paths[2]


def publish_pack(tmp_path, items):
    package = build_review_pack(items, seed=7)
    public, private = tmp_path / "public.json", tmp_path / "private.json"
    write_json(public, package["public"])
    write_json(private, package["private"])
    reviews = blank_review_template(package["public"], reviewer_id="human-one")
    methods = {i["review_id"]: i["method_id"] for i in package["private"]["items"]}
    for row in reviews["reviews"]:
        row["audio_seen"] = True
        row["ratings"]["musical_fit"] = 3 if methods[row["review_id"]] == A else 5
    review_path = tmp_path / "reviews.json"
    write_json(review_path, reviews)
    return public, private, review_path


def run(paths, **kwargs):
    return analysis.analyze_reviews(*paths, method_a=A, method_b=B, criterion="musical_fit",
        split="dev", origin=kwargs.pop("origin", "human"), n_resamples=30, **kwargs)


def replace(path, value):
    write_json(path, value, overwrite=True)


def test_originals_to_complete_rows_and_group_comparison_with_multiple_reviewers(tmp_path):
    paths = material(tmp_path, source_by_song=True)
    second = read_json(paths[3])
    for row in second["reviews"]:
        row["reviewer_id"] = "human-two"
        row["ratings"]["musical_fit"] -= 1
    extra = tmp_path / "second.json"
    write_json(extra, second)
    report = run((*paths[:3], [paths[3], extra]), compare=True)
    assert len(report["rows"]) == 12
    assert report["coverage"]["expected_cells"] == report["coverage"]["rated_cells"] == 6
    assert report["comparison"]["status"] == "computed"
    assert report["comparison"]["result"]["estimate"] == 2
    assert all(c["n_ratings"] == 2 for c in report["coverage"]["cells"])
    assert report["method_sources"][A]["multiple_source_records"]
    assert not report["method_sources"][A]["generation_model_identity_verified"]
    assert report["synthetic"] and not report["scientific_quality_verified"]
    assert not report["audio_bytes_checked"]  # No media files needed for original-score analysis.
    assert any("Three or fewer" in line for line in report["limitations"])
    assert report["record_sha256"] == object_sha({k: v for k, v in report.items() if k != "record_sha256"})


def test_whole_song_absent_from_pack_stays_in_denominator_and_blocks_ci(tmp_path, monkeypatch):
    include = {(s, m) for s in ("song1", "song2") for m in (A, B)}
    paths = material(tmp_path, include=include)
    monkeypatch.setattr(analysis, "compare_methods", lambda *a, **k: pytest.fail("Incomplete inventory cannot get a CI"))
    report = run(paths, compare=True)
    assert report["coverage"]["expected_cells"] == 6 and report["coverage"]["observed_plan_cells"] == 4
    missing = report["songs"][2]
    assert missing["song_id"] == "song3" and missing["difference_b_minus_a"] is None
    assert all(c["mean_rating"] is None and c["status"] == "missing_plan_in_pack" for c in missing["methods"].values())
    assert report["comparison"]["status"] == "blocked" and report["comparison"]["result"] is None


def test_absent_and_null_ratings_do_not_become_zero_or_disappear(tmp_path):
    paths = material(tmp_path)
    reviews = read_json(paths[3])
    absent = reviews["reviews"].pop()
    reviews["reviews"][0]["ratings"]["musical_fit"] = None
    replace(paths[3], reviews)
    report = run(paths, compare=True)
    assert report["coverage"]["expected_cells"] == 6
    statuses = [c["status"] for c in report["coverage"]["cells"]]
    assert "pending_criterion_rating" in statuses and "no_selected_reviewer_ratings" in statuses
    assert len(report["rows"]) == 4 and all(r["value"] > 0 for r in report["rows"])
    assert not report["coverage"]["complete_for_requested_criterion"]
    assert report["comparison"]["result"] is None


@pytest.mark.parametrize("reviewed,missing", [(False, False), (True, True)])
def test_unreviewed_or_incomplete_groups_only_produce_descriptions(tmp_path, reviewed, missing):
    paths = material(tmp_path, reviewed=reviewed,
        mutate_songs=(lambda ss: ss[2].update(near_duplicate_group_id=None)) if missing else None)
    report = run(paths, compare=True)
    assert all(row["group_id"] is None for row in report["rows"])
    assert report["grouping"]["components"] is None
    assert report["comparison"]["status"] == "blocked" and report["comparison"]["result"] is None
    assert report["songs"][0]["difference_b_minus_a"] == 2


def test_transitive_relation_components_are_not_independent_artist_fields(tmp_path):
    def relations(songs):
        songs[1]["artist_group_id"] = songs[0]["artist_group_id"]
        songs[2]["song_group_id"] = songs[1]["song_group_id"]
    paths = material(tmp_path, count=4, mutate_songs=relations)
    report = run(paths, compare=True)
    ids = [s["group_id"] for s in report["songs"]]
    assert ids[0] == ids[1] == ids[2] and ids[3] != ids[0]
    assert report["grouping"]["n_selected_components"] == 2
    assert report["comparison"]["result"]["n_independent_groups"] == 2


def test_one_connected_group_has_no_ci_even_with_multiple_songs(tmp_path):
    paths = material(tmp_path, mutate_songs=lambda songs: [s.update(artist_group_id="shared-artist") for s in songs])
    report = run(paths, compare=True)
    assert report["comparison"]["status"] == "insufficient_independent_groups"
    assert report["comparison"]["result"]["interval"] is None


def test_cross_split_source_relations_are_rejected(tmp_path):
    def cross_split(songs):
        songs[1]["split"] = "test"
        songs[1]["artist_group_id"] = songs[0]["artist_group_id"]
    paths = material(tmp_path, mutate_songs=cross_split)
    with pytest.raises(ValueError, match="Cross-split|crosses dataset splits"):
        run(paths, compare=True)


def test_inapplicable_edit_criterion_is_na_not_a_successful_zero_score(tmp_path):
    paths = material(tmp_path)
    report = analysis.analyze_reviews(*paths, method_a=A, method_b=B, criterion="edit_preservation",
        split="dev", origin="human", compare=True, n_resamples=30)
    assert not report["rows"]
    assert all(c["status"] == "criterion_not_applicable" and c["mean_rating"] is None
               for c in report["coverage"]["cells"])
    assert report["comparison"]["status"] == "blocked"


def test_two_origins_and_judge_versions_require_explicit_selection(tmp_path):
    paths = material(tmp_path)
    pack = read_json(paths[1])
    model = blank_review_template(pack, reviewer_id="judge-one", origin="model_judge", model_id="judge-v1")
    other = blank_review_template(pack, reviewer_id="judge-two", origin="model_judge", model_id="judge-v2")
    for doc in (model, other):
        for row in doc["reviews"]:
            row["audio_seen"] = True
            row["ratings"]["musical_fit"] = 1
    models, others = tmp_path / "models.json", tmp_path / "other-model.json"
    write_json(models, model)
    write_json(others, other)
    inputs = (*paths[:3], [paths[3], models, others])
    human = run(inputs, compare=True)
    assert {r["origin"] for r in human["rows"]} == {"human"}
    assert len(human["coverage"]["excluded_ratings"]) == 12
    chosen = run(inputs, origin="model_judge", judge_model_id="judge-v1", compare=True)
    assert {r["model_id"] for r in chosen["rows"]} == {"judge-v1"}
    assert len(chosen["coverage"]["excluded_ratings"]) == 12
    with pytest.raises(ValueError, match="judge_model_id"):
        run(inputs, origin="model_judge")
    with pytest.raises(ValueError, match="human"):
        run(inputs, judge_model_id="not-a-human")


def test_reviewer_cannot_change_judge_model_version_across_items(tmp_path):
    paths = material(tmp_path)
    model = blank_review_template(read_json(paths[1]), reviewer_id="same-judge", origin="model_judge", model_id="v1")
    for row in model["reviews"]:
        row["audio_seen"] = True
        row["ratings"]["musical_fit"] = 4
    model["reviews"][0]["model_id"] = "v2"
    replace(paths[3], model)
    with pytest.raises(ValueError, match="changes origin/model version"):
        run(paths, origin="model_judge", judge_model_id="v1")


def test_repeated_reviewer_item_across_original_files_is_rejected(tmp_path):
    paths = material(tmp_path)
    second = tmp_path / "duplicate.json"
    write_json(second, read_json(paths[3]))
    with pytest.raises(ValueError, match="duplicate reviewer/item"):
        run((*paths[:3], [paths[3], second]))


@pytest.mark.parametrize("changed", ["public", "private", "reviews", "dataset_bytes", "report_instead_of_original"])
def test_tampered_originals_or_unbound_dataset_are_rejected(tmp_path, changed):
    paths = material(tmp_path)
    if changed == "dataset_bytes":
        paths[0].write_bytes(paths[0].read_bytes() + b" ")
    elif changed == "public":
        doc = read_json(paths[1])
        doc["items"][0]["plan"]["events"][0]["action"] = "Changed public wording."
        replace(paths[1], doc)
    elif changed == "private":
        doc = read_json(paths[2])
        doc["items"][0]["original"]["plan"]["events"][0]["action"] = "Changed original wording."
        replace(paths[2], doc)
    elif changed == "reviews":
        doc = read_json(paths[3])
        doc["reviews"][0]["ratings"]["musical_fit"] = 0
        replace(paths[3], doc)
    else:
        replace(paths[3], {"schema": "choreobench.review_report.v1", "items": []})
    with pytest.raises(ValueError):
        run(paths)


def test_self_consistent_pack_still_cannot_change_dataset_song_metadata(tmp_path):
    paths = material(tmp_path, mutate_items=lambda items: [i["song"]["music"].update(source="different-structure") for i in items])
    with pytest.raises(ValueError, match="differs from dataset"):
        run(paths)


def test_wrong_source_dataset_and_missing_source_binding_are_not_silently_accepted(tmp_path):
    def wrong_binding(items):
        items[0]["source_binding"]["dataset_sha256"] = object_sha("another manifest")
    folder = tmp_path / "wrong"
    folder.mkdir()
    with pytest.raises(ValueError, match="dataset byte binding"):
        run(material(folder, mutate_items=wrong_binding))
    folder = tmp_path / "missing"
    folder.mkdir()
    with pytest.raises(ValueError, match="dataset byte binding"):
        run(material(folder, mutate_items=lambda items: items[0].pop("source_binding")))


def test_multiple_plan_or_model_versions_under_one_song_method_are_ambiguous(tmp_path):
    def duplicate(items):
        extra = copy.deepcopy(items[0])
        extra["plan"] = plan(extra["song"], "A different model version.")
        extra["source_binding"]["sha256"] = object_sha("another-generation-run")
        items.append(extra)
    paths = material(tmp_path, mutate_items=duplicate)
    with pytest.raises(ValueError, match="Ambiguous multiple plan/version"):
        run(paths)


def test_large_pack_can_be_analyzed_against_each_original_dataset_without_fake_merge(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    paths = material(first, count=1)
    private = read_json(paths[2])
    items = [copy.deepcopy(i["original"]) for i in private["items"]]
    outside = song(2)
    for method in (A, B):
        items.append(dict(song=outside, plan=plan(outside), method_id=method,
            source_binding=dict(kind="run", sha256=object_sha(method), dataset_sha256=object_sha("other-per-song-dataset"), synthetic=True)))
    merged = tmp_path / "largepack"
    merged.mkdir()
    public, bindings, reviews = publish_pack(merged, items)
    report = run((paths[0], public, bindings, reviews), compare=True)
    assert report["coverage"]["expected_cells"] == 2 and report["coverage"]["rated_cells"] == 2
    assert len(report["coverage"]["excluded_items"]) == 2
    assert all(i["reason"] == "outside_dataset" and not i["current_dataset_provenance_checked"] for i in report["coverage"]["excluded_items"])
    assert report["comparison"]["status"] == "insufficient_independent_groups"


def test_other_split_and_other_method_items_are_explicitly_excluded(tmp_path):
    def outside_split(songs):
        songs[2]["split"] = "test"
    def extra_method(items):
        extra = copy.deepcopy(items[0])
        extra["method_id"] = "method-charlie"
        items.append(extra)
    paths = material(tmp_path, mutate_songs=outside_split, mutate_items=extra_method)
    report = run(paths)
    assert report["coverage"]["expected_cells"] == 4
    reasons = [row["reason"] for row in report["coverage"]["excluded_items"]]
    assert reasons.count("outside_selected_split") == 2 and reasons.count("method_not_selected") == 1
    assert all(not row["current_dataset_provenance_checked"] for row in report["coverage"]["excluded_items"])


def roster(paths, tmp_path, reviewers=('human-one', 'human-two'), criteria=('musical_fit',)):
    pack = read_json(paths[1])
    assignments = [dict(review_id=item['review_id'], reviewer_id=rid, origin='human', model_id=None,
        criteria=list(criteria)) for item in pack['items'] for rid in reviewers]
    path = tmp_path / 'roster.json'
    write_json(path, dict(schema='choreobench.reviewer_roster.v1', pack_sha256=object_sha(pack), assignments=assignments))
    return path


def test_roster_detects_entire_unsubmitted_reviewer_even_when_all_song_cells_are_rated(tmp_path, monkeypatch):
    paths = material(tmp_path)
    roster_path = roster(paths, tmp_path)
    monkeypatch.setattr(analysis, 'compare_methods', lambda *a, **k: pytest.fail('Missing registered reviewer must block comparison'))
    report = run(paths, reviewer_roster_path=roster_path, compare=True)
    assert report['coverage']['complete_for_requested_criterion']
    registered = report['reviewer_roster']
    assert not registered['complete_for_roster']
    assert registered['expected_assignment_count'] == 12 and registered['rated_assignment_count'] == 6
    assert registered['entirely_unsubmitted_reviewer_ids'] == ['human-two']
    assert all(row['status'] == 'missing_review_record' for row in registered['missing_or_unrated_assignments'])
    assert report['comparison']['status'] == 'blocked' and report['comparison']['result'] is None
    assert report['sources'][-1]['role'] == 'reviewer_roster'


def test_roster_complete_two_reviewers_and_original_review_records(tmp_path):
    paths = material(tmp_path)
    roster_path = roster(paths, tmp_path)
    second = read_json(paths[3])
    for row in second['reviews']:
        row['reviewer_id'] = 'human-two'
    extra = tmp_path / 'second.json';write_json(extra, second)
    report = run((*paths[:3], [paths[3], extra]), reviewer_roster_path=roster_path, compare=True)
    registered = report['reviewer_roster']
    assert registered['complete_for_roster'] and registered['rated_assignment_count'] == 12
    assert not registered['entirely_unsubmitted_reviewer_ids']
    assert not registered['reviewer_identity_or_actual_listening_authenticated']
    assert not registered['preregistration_timing_independently_verified']
    assert report['comparison']['status'] == 'computed' and len(report['rows']) == 12


def test_roster_zero_submission_preserves_every_expected_assignment(tmp_path):
    paths = material(tmp_path)
    roster_path = roster(paths, tmp_path)
    doc = read_json(paths[3]);doc['reviews'] = [];replace(paths[3], doc)
    report = run(paths, reviewer_roster_path=roster_path)
    registered = report['reviewer_roster']
    assert registered['expected_assignment_count'] == 12
    assert registered['submitted_assignment_count'] == registered['rated_assignment_count'] == 0
    assert registered['entirely_unsubmitted_reviewer_ids'] == ['human-one', 'human-two']
    assert len(report['coverage']['cells']) == 6 and not report['rows']


def test_roster_null_criterion_is_not_a_missing_submission_or_a_zero(tmp_path):
    paths = material(tmp_path)
    roster_path = roster(paths, tmp_path, reviewers=('human-one',), criteria=('musical_fit', 'long_range_structure'))
    doc = read_json(paths[3]);doc['reviews'][0]['ratings']['musical_fit'] = None;replace(paths[3], doc)
    report = run(paths, reviewer_roster_path=roster_path, compare=True)
    registered = report['reviewer_roster']
    assert registered['submitted_assignment_count'] == 6 and registered['rated_assignment_count'] == 5
    assert registered['missing_or_unrated_assignments'][0]['status'] == 'pending_criterion_rating'
    assert not registered['entirely_unsubmitted_reviewer_ids']
    assert len(report['rows']) == 5 and report['comparison']['result'] is None
    other = analysis.analyze_reviews(*paths, method_a=A, method_b=B, criterion='long_range_structure',
        split='dev', origin='human', reviewer_roster_path=roster_path)
    assert other['reviewer_roster']['rated_assignment_count'] == 0


def test_unregistered_scores_are_reported_but_cannot_fill_or_change_roster_means(tmp_path):
    paths = material(tmp_path)
    roster_path = roster(paths, tmp_path, reviewers=('human-one',))
    outsider = read_json(paths[3])
    for row in outsider['reviews']:
        row['reviewer_id'] = 'unassigned'
        row['ratings']['musical_fit'] = 1
    extra = tmp_path / 'outside.json';write_json(extra, outsider)
    report = run((*paths[:3], [paths[3], extra]), reviewer_roster_path=roster_path, compare=True)
    assert report['reviewer_roster']['complete_for_roster']
    assert len(report['rows']) == 6 and {row['reviewer_id'] for row in report['rows']} == {'human-one'}
    assert report['comparison']['result']['estimate'] == 2
    assert len(report['coverage']['excluded_ratings']) == 6
    assert all(row['reason'] == 'reviewer_not_assigned_for_criterion' for row in report['coverage']['excluded_ratings'])


def test_incomplete_roster_cannot_silently_shrink_the_song_method_denominator(tmp_path):
    paths = material(tmp_path)
    roster_path = roster(paths, tmp_path, reviewers=('human-one',))
    doc = read_json(roster_path);doc['assignments'].pop();replace(roster_path, doc)
    report = run(paths, reviewer_roster_path=roster_path, compare=True)
    assert len(report['reviewer_roster']['unassigned_song_method_cells']) == 1
    assert not report['reviewer_roster']['complete_for_roster'] and report['comparison']['result'] is None
    assert report['coverage']['expected_cells'] == 6


@pytest.mark.parametrize('change', ['pack', 'schema', 'unknown_field', 'unknown_item', 'placeholder', 'duplicate',
    'empty', 'criteria_empty', 'criteria_unknown', 'criteria_duplicate', 'criteria_edit', 'human_model', 'origin', 'model_identity'])
def test_bad_or_unbound_rosters_are_rejected(tmp_path, change):
    paths = material(tmp_path);path = roster(paths, tmp_path);doc = read_json(path);row = doc['assignments'][0]
    if change == 'pack': doc['pack_sha256'] = object_sha('wrong')
    elif change == 'schema': doc['schema'] = 'other'
    elif change == 'unknown_field': doc['scores'] = []
    elif change == 'unknown_item': row['review_id'] = 'unknown'
    elif change == 'placeholder': row['reviewer_id'] = 'FILL_REVIEWER_ID'
    elif change == 'duplicate': doc['assignments'].append(copy.deepcopy(row))
    elif change == 'empty': doc['assignments'] = []
    elif change == 'criteria_empty': row['criteria'] = []
    elif change == 'criteria_unknown': row['criteria'] = ['unknown']
    elif change == 'criteria_duplicate': row['criteria'] *= 2
    elif change == 'criteria_edit': row['criteria'] = ['edit_preservation']
    elif change == 'human_model': row['model_id'] = 'model'
    elif change == 'origin': row['origin'] = 'unknown'
    else: row.update(origin='model_judge', model_id='v1')
    replace(path, doc)
    with pytest.raises(ValueError):
        run(paths, reviewer_roster_path=path)


def test_roster_filters_other_splits_and_origin_without_fake_missing_reviewer(tmp_path):
    paths = material(tmp_path, mutate_songs=lambda songs: songs[2].update(split='test'))
    path = roster(paths, tmp_path, reviewers=('human-one',))
    doc = read_json(path)
    model_rows = [dict(row, reviewer_id='judge', origin='model_judge', model_id='v1') for row in doc['assignments']]
    doc['assignments'].extend(model_rows);replace(path, doc)
    human = run(paths, reviewer_roster_path=path)
    assert human['reviewer_roster']['expected_assignment_count'] == 4
    assert len(human['reviewer_roster']['excluded_assignments']) == 8
    assert human['reviewer_roster']['complete_for_roster']
    model = run(paths, reviewer_roster_path=path, origin='model_judge', judge_model_id='v1')
    assert model['reviewer_roster']['expected_assignment_count'] == 4
    assert model['reviewer_roster']['entirely_unsubmitted_reviewer_ids'] == ['judge']
    assert not model['rows']


def test_roster_keeps_absent_song_in_denominator(tmp_path):
    include = {(sid, method) for sid in ('song1', 'song2') for method in (A, B)}
    paths = material(tmp_path, include=include)
    path = roster(paths, tmp_path)
    report = run(paths, reviewer_roster_path=path, compare=True)
    assert report['coverage']['expected_cells'] == 6
    assert report['reviewer_roster']['expected_assignment_count'] == 8
    assert len(report['reviewer_roster']['unassigned_song_method_cells']) == 2
    assert {row['song_id'] for row in report['reviewer_roster']['unassigned_song_method_cells']} == {'song3'}
    assert not report['reviewer_roster']['complete_for_roster'] and report['comparison']['result'] is None


def test_roster_criterion_without_assignments_blocks_comparison(tmp_path):
    paths = material(tmp_path)
    path = roster(paths, tmp_path, reviewers=('human-one',), criteria=('long_range_structure',))
    report = run(paths, reviewer_roster_path=path, compare=True)
    registered = report['reviewer_roster']
    assert registered['expected_assignment_count'] == 0 and not registered['complete_for_roster']
    assert len(registered['unassigned_song_method_cells']) == 6
    assert not report['rows'] and report['comparison']['result'] is None


def test_roster_mutation_during_comparison_is_rejected(tmp_path, monkeypatch):
    paths = material(tmp_path);path = roster(paths, tmp_path, reviewers=('human-one',))
    def mutate(*args, **kwargs):
        path.write_bytes(path.read_bytes() + b' ')
        return {'status': 'fixture'}
    monkeypatch.setattr(analysis, 'compare_methods', mutate)
    with pytest.raises(ValueError, match='source changed'):
        run(paths, reviewer_roster_path=path, compare=True)


@pytest.mark.parametrize('module_cli', [True, False])
def test_reviewer_roster_cli_keeps_cell_and_registered_completion_separate(tmp_path, module_cli):
    paths = material(tmp_path);path = roster(paths, tmp_path);output = tmp_path / 'roster-report.json'
    args = ['--dataset', str(paths[0]), '--public', str(paths[1]), '--private', str(paths[2]),
        '--reviews', str(paths[3]), '--method-a', A, '--method-b', B, '--criterion', 'musical_fit',
        '--split', 'dev', '--origin', 'human', '--reviewer-roster', str(path), '--compare', '--out', str(output)]
    if module_cli:
        analysis.main(args)
    else:
        from choreobench.cli import main
        assert main(['analyze-reviews'] + args) == 0
    report = read_json(output)
    assert report['coverage']['complete_for_requested_criterion'] and not report['reviewer_roster']['complete_for_roster']
    assert report['comparison']['status'] == 'blocked'


def test_module_cli_writes_exclusive_source_bound_report(tmp_path):
    paths = material(tmp_path)
    output = tmp_path / "analysis.json"
    args = ["--dataset", str(paths[0]), "--public", str(paths[1]), "--private", str(paths[2]),
        "--reviews", str(paths[3]), "--method-a", A, "--method-b", B, "--criterion", "musical_fit",
        "--split", "dev", "--origin", "human", "--compare", "--resamples", "20", "--out", str(output)]
    report = analysis.main(args)
    assert report == read_json(output)
    assert len(report["sources"]) == 4 and report["comparison"]["status"] == "computed"
    with pytest.raises(FileExistsError):
        analysis.main(args)
