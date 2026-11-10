"""Synthetic evidence only; fake providers never represent musical quality."""
import copy
import json
from pathlib import Path
import re

import pytest

from choreobench import candidate_review, revision
from choreobench.baseline import _sealed
from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.revision_experiment import prepare_revision_experiment
from choreobench.reviews import _validate_bindings
from test_choreobench_revision import World, _nine_event_bounded_source


@pytest.fixture
def experiment(tmp_path, monkeypatch, request):
    world = World(tmp_path)
    scenario = getattr(request, "param", None)
    if scenario == "legacy_diagnosis":
        _nine_event_bounded_source(world)
        monkeypatch.setattr(revision, "_bounded_diagnosis", revision.parse_diagnosis)
    world.issues = [dict(event_id=world.events[0], priority=3, reason="Synthetic issue only.")]
    if scenario == "legacy_diagnosis":
        world.issues = [dict(event_id=eid, priority=3, reason="x" * 401) for eid in world.events]
    if scenario == "fatal_edit":
        world.issues.append(dict(event_id=world.events[1], priority=2, reason="A second synthetic issue."))
    world.verdicts[world.events[0]] = "uncertain"
    monkeypatch.setattr(revision, "create_provider", lambda *a, **kw: world.text)
    monkeypatch.setattr(revision, "create_music_observer", lambda *a, **kw: world.audio)
    repairs = 2 if scenario == "fatal_edit" else 1
    config = world.config(listen=True, max_repairs=repairs, max_calls=3 * repairs)
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    root = tmp_path / "experiment"
    manifest = prepare_revision_experiment(world.dataset, world.plan_path, config_path, root)
    revision.diagnose_plan(world.dataset, world.plan_path, root / "diagnosis", read_json(root / "configs/diagnosis.json"))
    for arm in candidate_review.ARMS:
        arm_config = read_json(root / "configs" / (arm + ".json"))
        selection = revision.select_events(world.plan, dict(schema="choreobench.revision_diagnosis.v1",
            source_plan_sha256=object_sha(world.plan), issues=world.issues), arm_config["selection_policy"], repairs, arm_config["seed"])
        if arm == "random":
            world.edit_behavior[selection[0]["event_id"]] = "no_op"
        else:
            world.edit_behavior[selection[0]["event_id"]] = "edit"
        if arm == "targeted":
            eid = selection[0]["event_id"]
            if scenario == "malformed_edit": world.malformed.add(("patch", eid))
            elif scenario == "forbidden_edit": world.edit_behavior[eid] = "forbidden"
            elif scenario == "failed_judge": world.malformed.add(("revision_verdict", eid))
            elif scenario == "fatal_edit": world.raise_on.add(("patch", eid))
        revision.run_revision(world.dataset, world.plan_path, root / "diagnosis/diagnosis.json", root / "runs" / arm, arm_config)
        world.malformed.clear(); world.raise_on.clear()
    inventory = tmp_path / "inventory.json"
    write_json(inventory, dict(schema=candidate_review.INVENTORY_SCHEMA, experiments=[
        dict(experiment_dir=str(root), manifest_sha256=file_sha(root / "manifest.json"))]))
    calls = len(world.calls)
    monkeypatch.setattr(revision, "create_provider", lambda *a, **kw: pytest.fail("audit constructed model"))
    monkeypatch.setattr(revision, "create_music_observer", lambda *a, **kw: pytest.fail("audit constructed observer"))
    return dict(world=world, root=root, inventory=inventory, calls=calls, manifest=manifest)


def prepare(experiment, tmp_path):
    output = tmp_path / "review"
    receipt = candidate_review.prepare_candidate_review(experiment["inventory"], output)
    return output, receipt


def test_all_nonempty_candidates_and_noops_retained_without_model_calls(experiment, tmp_path):
    output, receipt = prepare(experiment, tmp_path)
    assert receipt["candidate_pair_count"] == 2
    assert receipt["selected_step_count"] == 3
    assert receipt["no_op_count"] == 1
    assert receipt["historical_internal_accepted"] == 0
    assert receipt["new_model_calls"] == receipt["actual_human_reviews"] == receipt["gpu_count"] == 0
    assert len(experiment["world"].calls) == experiment["calls"]
    assert receipt["unique_action_pair_count"] == 1
    assert receipt["latest_guard_rejected_nonempty"] == 0
    pack, bindings = read_json(output / "public/pack.json"), read_json(output / "private/bindings.json")
    _validate_bindings(pack, bindings)
    assert len(pack["items"]) == 4
    assert all(i["edit_task"] is None and i["source_plan"] is None for i in pack["items"])
    assert all(i["plan"]["provenance"] == {} for i in pack["items"])
    page = (output / "public/index.html").read_text()
    assert 'export-preferences' in page and 'reviewed-start' in page
    payload = json.loads(re.search(r'<script id="candidate-data" type="application/json">(.*?)</script>', page, re.S).group(1))
    assert all(r["choice"] is None and not r["reviewed_ranges"] and not r["audio_seen"] for r in payload["preferences"]["reviews"])
    assert 'internal_accepted' not in json.dumps(payload)
    assert 'latest_timing_guard' not in json.dumps(payload)
    assert 'candidate_label' not in json.dumps(payload)
    assert len(read_json(output / "targets/public/pack.json")["items"]) == receipt["target_count"]
    assert 'diagnosis' not in json.dumps(read_json(output / "targets/private/focus.json"))
    for path, sha in receipt["sources"].items():
        assert file_sha(path) == sha
    with pytest.raises(FileExistsError):
        candidate_review.prepare_candidate_review(experiment["inventory"], output)


@pytest.mark.parametrize('artifact', ['steps/00/patch.json', 'steps/00/candidate.json', 'steps/00/decision.json', 'plan.json'])
def test_changed_historical_artifacts_rejected_before_output(experiment, tmp_path, artifact):
    path = experiment["root"] / "runs/targeted" / artifact
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='artifact changed'):
        candidate_review.prepare_candidate_review(experiment["inventory"], tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_duplicate_manifest_and_wrong_manifest_hash_rejected(experiment, tmp_path):
    inventory = read_json(experiment["inventory"])
    inventory["experiments"].append(copy.deepcopy(inventory["experiments"][0]))
    write_json(experiment["inventory"], inventory, overwrite=True)
    with pytest.raises(ValueError, match='Duplicate'):
        prepare(experiment, tmp_path)
    inventory["experiments"] = inventory["experiments"][:1]
    inventory["experiments"][0]["manifest_sha256"] = '0' * 64
    write_json(experiment["inventory"], inventory, overwrite=True)
    with pytest.raises(ValueError, match='Source bytes differ'):
        prepare(experiment, tmp_path)


def test_receipt_preserves_rejected_candidates_and_current_guard_only(experiment, tmp_path, monkeypatch):
    monkeypatch.setattr(candidate_review, '_no_new_timing_literals', lambda *a: (_ for _ in ()).throw(ValueError('fixture timing literal')))
    output, receipt = prepare(experiment, tmp_path)
    assert receipt["latest_guard_rejected_nonempty"] == 2
    assert len(read_json(output / 'public/pack.json')['items']) == 4
    assert receipt['historical_results_unchanged'] and not receipt['quality_verified']


def test_focus_rejects_extra_changes_duplicate_ids_and_excluded_context(experiment, tmp_path):
    output, _ = prepare(experiment, tmp_path)
    pack, focus = read_json(output / 'public/pack.json'), read_json(output / 'private/focus.json')
    broken = copy.deepcopy(focus)
    broken['pairs'][0]['required_range'] = dict(start=1, end=2)
    with pytest.raises(ValueError, match='excludes target'):
        candidate_review.validate_focus(pack, broken)
    broken = copy.deepcopy(focus)
    broken['pairs'][1]['a_review_id'] = broken['pairs'][0]['a_review_id']
    with pytest.raises(ValueError, match='reused'):
        candidate_review.validate_focus(pack, broken)
    broken = copy.deepcopy(focus)
    broken['pairs'].pop()
    with pytest.raises(ValueError, match='entire public inventory'):
        candidate_review.validate_focus(pack, broken)


def filled(output):
    prefs = read_json(output / 'private/preferences.template.json')
    focus = read_json(output / 'private/focus.json')
    for row, pair in zip(prefs['reviews'], focus['pairs']):
        row.update(reviewer_id='fixture-r1', audio_seen=True, choice='tie', reason='Synthetic tie, not human quality evidence.',
                   reviewed_ranges=[pair['required_range']])
    return prefs


def test_missing_reviews_stay_pending_and_roster_is_not_invented(experiment, tmp_path):
    output, receipt = prepare(experiment, tmp_path)
    result = candidate_review.analyze_candidate_preferences(output, [])
    assert result['pair_count'] == receipt['candidate_pair_count']
    assert result['eligible_human_judgments'] == 0
    assert set(result['pair_status'].values()) == {'pending'}
    assert not result['expected_reviewer_roster_supplied'] and result['full_design_complete'] is None
    assert not result['quality_superiority_established']


def test_range_union_and_insufficient_review_do_not_create_quality_success(experiment, tmp_path):
    output, _ = prepare(experiment, tmp_path)
    prefs = filled(output)
    end = prefs['reviews'][0]['reviewed_ranges'][0]['end']
    prefs['reviews'][0]['reviewed_ranges'] = [dict(start=0, end=end // 2), dict(start=end // 2, end=end)]
    prefs['reviews'][1]['reviewed_ranges'] = [dict(start=0, end=1)]
    path = tmp_path / 'prefs.json'
    write_json(path, prefs)
    result = candidate_review.analyze_candidate_preferences(output, [path])
    assert result['eligible_human_judgments'] == 1
    assert result['rows'][0]['range_sufficient'] and not result['rows'][1]['range_sufficient']
    assert all(not r['possible_internal_misaccept'] for r in result['rows'])
    assert not result['quality_superiority_established']


@pytest.mark.parametrize('mutation', ['duplicate', 'wrong_hash', 'missing_reason', 'placeholder', 'model_origin', 'bool_range'])
def test_invalid_external_preferences_rejected(experiment, tmp_path, mutation):
    output, _ = prepare(experiment, tmp_path)
    prefs = filled(output)
    if mutation == 'duplicate': prefs['reviews'].append(copy.deepcopy(prefs['reviews'][0]))
    elif mutation == 'wrong_hash': prefs['reviews'][0]['a_item_sha256'] = '0' * 64
    elif mutation == 'missing_reason': prefs['reviews'][0]['reason'] = ''
    elif mutation == 'placeholder': prefs['reviews'][0]['reviewer_id'] = 'FILL_REVIEWER_ID'
    elif mutation == 'model_origin': prefs['reviews'][0]['model_id'] = 'fixture-model'
    elif mutation == 'bool_range': prefs['reviews'][0]['reviewed_ranges'][0]['start'] = False
    path = tmp_path / 'prefs.json'
    write_json(path, prefs)
    with pytest.raises(ValueError):
        candidate_review.analyze_candidate_preferences(output, [path])


def test_model_judge_is_separate_and_duplicate_reviewer_files_rejected(experiment, tmp_path):
    output, _ = prepare(experiment, tmp_path)
    prefs = filled(output)
    for r in prefs['reviews']:
        r.update(origin='model_judge', model_id='fixture-model')
    path = tmp_path / 'prefs.json'
    write_json(path, prefs)
    result = candidate_review.analyze_candidate_preferences(output, [path])
    assert result['eligible_human_judgments'] == 0
    with pytest.raises(ValueError, match='duplicate reviewer'):
        candidate_review.analyze_candidate_preferences(output, [path, path])


def test_candidate_page_escapes_script_content(experiment, tmp_path):
    output, _ = prepare(experiment, tmp_path)
    pack, focus = read_json(output / 'public/pack.json'), read_json(output / 'private/focus.json')
    # A hostile display ID cannot turn into executable markup.
    focus['pairs'][0]['pair_id'] = '</script><script>alert(1)</script>'
    page = candidate_review.render_candidate_page(pack, focus)
    assert '</script><script>alert(1)</script>' not in page
    assert chr(92) + 'u003c/script>' in page


def test_cli_preparation_and_empty_analysis(experiment, tmp_path, capsys):
    output = tmp_path / 'review'
    candidate_review.main(['prepare', '--inventory', str(experiment['inventory']), '--out-dir', str(output)])
    assert json.loads(capsys.readouterr().out)['candidate_pair_count'] == 2
    dest = tmp_path / 'analysis.json'
    candidate_review.main(['analyze', '--review-dir', str(output), '--out', str(dest)])
    assert read_json(dest)['eligible_human_judgments'] == 0



def test_target_judgments_pending_and_sufficient_range_required(experiment, tmp_path):
    output, _ = prepare(experiment, tmp_path)
    result = candidate_review.analyze_target_judgments(output, [])
    assert result['eligible_human_judgments'] == 0 and result['diagnosis_recall'] is None
    assert set(result['target_status'].values()) == {'pending'}
    document = read_json(output / 'targets/private/judgments.template.json')
    focus = read_json(output / 'targets/private/focus.json')
    for row, target in zip(document['reviews'], focus['targets']):
        row.update(reviewer_id='fixture-r1', audio_seen=True, needs_revision='no',
                   reason='Synthetic only: do not mistake this for independent human evaluation.', reviewed_ranges=[target['required_range']])
    path = tmp_path / 'target-judgments.json'
    write_json(path, document)
    result = candidate_review.analyze_target_judgments(output, [path])
    assert result['eligible_human_judgments'] == len(document['reviews'])
    assert all(r['needs_revision'] == 'no' for r in result['rows'])
    assert result['full_design_complete'] is None
    with pytest.raises(ValueError, match='duplicate reviewer'):
        candidate_review.analyze_target_judgments(output, [path, path])


def test_target_unknown_without_audio_is_pending_and_hash_rejects(experiment, tmp_path):
    output, _ = prepare(experiment, tmp_path)
    document = read_json(output / 'targets/private/judgments.template.json')
    for row in document['reviews']:
        row.update(reviewer_id='fixture-r1', needs_revision='unknown', reason='Unable to listen in synthetic test.')
    path = tmp_path / 'target-judgments.json'
    write_json(path, document)
    result = candidate_review.analyze_target_judgments(output, [path])
    assert result['eligible_human_judgments'] == 0
    document['reviews'][0]['item_sha256'] = '0' * 64
    write_json(path, document, overwrite=True)
    with pytest.raises(ValueError, match='identity mismatch'):
        candidate_review.analyze_target_judgments(output, [path])


def test_private_candidate_label_swap_is_rejected(experiment, tmp_path):
    output, _ = prepare(experiment, tmp_path)
    path = output / 'private/audit.json'
    receipt = read_json(path)
    pair = next(r for r in receipt['rows'] if 'pair_id' in r)
    pair['candidate_label'] = 'b' if pair['candidate_label'] == 'a' else 'a'
    receipt.pop('record_sha256')
    write_json(path, _sealed(receipt), overwrite=True)
    with pytest.raises(ValueError, match='direction'):
        candidate_review.analyze_candidate_preferences(output, [])


def test_browser_pair_and_target_exports_are_blank_until_explicit_input(experiment, tmp_path):
    sync_api = pytest.importorskip('playwright.sync_api')
    from choreobench.review_server import create_review_server
    import threading
    output, _ = prepare(experiment, tmp_path)
    with sync_api.sync_playwright() as engine:
        try:
            browser = engine.chromium.launch(headless=True, args=['--no-sandbox'])
        except sync_api.Error as error:
            if 'Executable doesn' in str(error):
                pytest.skip('Optional Chromium runtime unavailable')
            raise
        try:
            for kind, directory in [('pairs', output), ('targets', output / 'targets')]:
                server = create_review_server(directory / 'public', port=0)
                worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
                page = browser.new_page(accept_downloads=True); errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                try:
                    page.goto('http://127.0.0.1:%d/' % server.server_address[1], wait_until='networkidle')
                    assert not errors
                    page.locator('#reviewer').fill('browser-fixture-not-human')
                    button = '#export-preferences' if kind == 'pairs' else '#export-targets'
                    with page.expect_download() as download:
                        page.locator(button).click()
                    blank = json.loads(Path(download.value.path()).read_text())
                    assert all(not r['audio_seen'] and not r['reviewed_ranges'] for r in blank['reviews'])
                    assert all(r.get('choice', r.get('needs_revision')) is None for r in blank['reviews'])
                    assert page.locator('tr[data-target="true"]').count() > 0
                    assert not errors
                finally:
                    page.close(); server.shutdown(); worker.join(); server.server_close()
        finally:
            browser.close()


def reseal(path, value):
    value.pop("record_sha256", None)
    write_json(path, _sealed(value), overwrite=True)


def change_trace(experiment, kind, change):
    run = experiment["root"] / "runs/targeted"
    for path in (run / "calls").glob("*.started.json"):
        start = read_json(path)
        if start["request"]["response_kind"] != kind:
            continue
        raw_path = path.with_name(path.name.replace(".started.json", ".raw.json"))
        raw = read_json(raw_path)
        change(start, raw)
        start["request_sha256"] = object_sha(start["request"])
        reseal(path, start)
        raw["started_sha256"] = read_json(path)["record_sha256"]
        reseal(raw_path, raw)
        result = read_json(run / "results.json")
        result["call_records"][start["call_id"]] = read_json(raw_path)["record_sha256"]
        reseal(run / "results.json", result)
        return
    raise AssertionError("Fixture lacks requested call")


@pytest.mark.parametrize("product", ["targeted", "diagnosis", "commands"])
def test_prepared_products_are_bound_before_output(experiment, tmp_path, product):
    path = experiment["root"] / ("commands.json" if product == "commands" else "configs/%s.json" % product)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Source bytes differ"):
        prepare(experiment, tmp_path)
    assert not (tmp_path / "review").exists()


def test_correctly_rehashed_command_cannot_reference_another_configuration(experiment, tmp_path):
    root = experiment["root"]
    commands = read_json(root / "commands.json")
    commands["steps"][1]["argv"][9] = str(root / "configs/random.json")
    write_json(root / "commands.json", commands, overwrite=True)
    manifest = read_json(root / "manifest.json")
    manifest["commands"]["sha256"] = file_sha(root / "commands.json")
    reseal(root / "manifest.json", manifest)
    inventory = read_json(experiment["inventory"])
    inventory["experiments"][0]["manifest_sha256"] = file_sha(root / "manifest.json")
    write_json(experiment["inventory"], inventory, overwrite=True)
    with pytest.raises(ValueError, match="Prepared command does not bind"):
        prepare(experiment, tmp_path)


def test_rehashed_run_cannot_change_the_prepared_config(experiment, tmp_path):
    run = experiment["root"] / "runs/targeted"
    contract = read_json(run / "contract.json")
    contract["config"]["seed"] += 100
    write_json(run / "contract.json", contract, overwrite=True)
    contract_sha = object_sha(contract)
    result = read_json(run / "results.json")
    result["contract_sha256"] = contract_sha
    for path in (run / "calls").glob("*.started.json"):
        start = read_json(path); start["contract_sha256"] = contract_sha
        reseal(path, start)
        raw_path = path.with_name(path.name.replace(".started.json", ".raw.json"))
        raw = read_json(raw_path)
        raw.update(contract_sha256=contract_sha, started_sha256=read_json(path)["record_sha256"])
        reseal(raw_path, raw)
        result["call_records"][start["call_id"]] = read_json(raw_path)["record_sha256"]
    reseal(run / "results.json", result)
    with pytest.raises(ValueError, match="Actual run config differs"):
        prepare(experiment, tmp_path)


@pytest.mark.parametrize("contradiction", ["patch_output", "choice_output", "observation_output", "edit_input", "judge_input", "observation_window", "provider_identity", "failed_response"])
def test_rehashed_raw_trace_must_agree_with_saved_artifacts(experiment, tmp_path, contradiction):
    kind = ("local_caption" if contradiction.startswith("observation") else
            "revision_verdict" if contradiction in ("choice_output", "judge_input") else "patch")
    def change(start, raw):
        if contradiction == "patch_output": raw["result"]["text"] = '{"not_a_patch":true}'
        elif contradiction == "choice_output":
            value = json.loads(raw["result"]["text"]); value["choice"] = "a"
            raw["result"]["text"] = json.dumps(value)
        elif contradiction == "observation_output": raw["result"]["text"] = "A different recorded observation."
        elif contradiction == "observation_window": start["request"]["audio"]["parent_end"] -= 1
        elif contradiction == "provider_identity": raw["result"]["identity"]["unexpected"] = "different source"
        elif contradiction == "failed_response": raw.update(success=False, result=None, error={"type": "fixture", "message": "failed"})
        else:
            view = json.loads(start["request"]["messages"][-1]["content"])
            if contradiction == "edit_input": view["source_plan"]["events"][0]["action"] += " Different input."
            else: view["pair"]["a"]["events"][0]["action"] += " Different alternative."
            start["request"]["messages"][-1]["content"] = json.dumps(view)
    change_trace(experiment, kind, change)
    with pytest.raises(ValueError):
        prepare(experiment, tmp_path)
    assert not (tmp_path / "review").exists()


@pytest.mark.parametrize("experiment", ["malformed_edit", "forbidden_edit", "failed_judge", "fatal_edit", "legacy_diagnosis"], indirect=True)
def test_historical_failures_and_legacy_diagnoses_remain_auditable(experiment, tmp_path):
    _, receipt = prepare(experiment, tmp_path)
    targeted = [row for row in receipt["rows"] if row["arm"] == "targeted"]
    original = read_json(experiment["root"] / "runs/targeted/results.json")
    assert [row["status"] for row in targeted] == [step["status"] for step in original["steps"]]
    assert receipt["selected_step_count"] == sum(len(read_json(experiment["root"] / "runs" / arm / "results.json")["steps"])
                                                 for arm in candidate_review.ARMS)
    assert file_sha(experiment["root"] / "runs/targeted/results.json") == receipt["sources"][str(experiment["root"] / "runs/targeted/results.json")]


@pytest.mark.parametrize("target_mode", [False, True])
@pytest.mark.parametrize("identity_change", ["origin", "model"])
def test_reviewer_identity_cannot_drift_between_items_or_files(experiment, tmp_path, target_mode, identity_change):
    output, _ = prepare(experiment, tmp_path)
    document = read_json(output / ("targets/private/judgments.template.json" if target_mode else "private/preferences.template.json"))
    for row in document["reviews"]:
        row.update(reviewer_id="same-reviewer", origin="model_judge", model_id="one-model")
    changed = document["reviews"][1]
    if identity_change == "origin": changed.update(origin="human", model_id=None)
    else: changed["model_id"] = "another-model"
    first, second = copy.deepcopy(document), copy.deepcopy(document)
    first["reviews"], second["reviews"] = document["reviews"][:1], document["reviews"][1:]
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    write_json(one, first); write_json(two, second)
    analyze = candidate_review.analyze_target_judgments if target_mode else candidate_review.analyze_candidate_preferences
    with pytest.raises(ValueError, match="Reviewer origin/model identity changed"):
        analyze(output, [one, two])
    second["reviews"][0]["reviewer_id"] = "different-reviewer"
    write_json(two, second, overwrite=True)
    assert len(analyze(output, [one, two])["rows"]) == len(document["reviews"])


def test_synthetic_status_and_unverified_independence_survive_analysis(experiment, tmp_path):
    output, receipt = prepare(experiment, tmp_path)
    assert receipt["synthetic"] is True and receipt["scientific_result_status"] == "synthetic"
    preferences = filled(output)
    path = tmp_path / "preferences.json"; write_json(path, preferences)
    result = candidate_review.analyze_candidate_preferences(output, [path])
    assert result["synthetic"] is True and result["eligible_human_judgments"] == receipt["candidate_pair_count"]
    for analyze in (candidate_review.analyze_candidate_preferences, candidate_review.analyze_target_judgments):
        report = analyze(output, [])
        assert report["song_id_count"] == receipt["song_count"] and report["independent_song_count"] is None
        assert report["source_group_independence_verified"] is False
    # Legacy private receipts remain readable, with absent provenance unknown.
    path = output / "private/audit.json"
    legacy = read_json(path); legacy.pop("synthetic"); legacy.pop("scientific_result_status")
    reseal(path, legacy)
    for analyze in (candidate_review.analyze_candidate_preferences, candidate_review.analyze_target_judgments):
        report = analyze(output, [])
        assert report["synthetic"] is None and report["scientific_result_status"] == "unknown"


def test_pair_ids_are_assigned_after_seeded_shuffle_without_reordering_audit(experiment, tmp_path):
    roots = [tmp_path / "review-one", tmp_path / "review-again", tmp_path / "review-other"]
    receipts = [candidate_review.prepare_candidate_review(experiment["inventory"], root, seed=seed)
                for root, seed in zip(roots, (1, 1, 0))]
    def pair_order(receipt):
        return [(row["pair_id"], row["arm"], row["candidate_label"])
                for row in sorted((r for r in receipt["rows"] if "pair_id" in r), key=lambda r: r["pair_id"])]
    assert pair_order(receipts[0]) == pair_order(receipts[1])
    assert [row[1] for row in pair_order(receipts[0])] != [row[1] for row in pair_order(receipts[2])]
    assert all([row["arm"] for row in receipt["rows"]] == list(candidate_review.ARMS) for receipt in receipts)
    first, again = (read_json(root / "public/pack.json") for root in roots[:2])
    assert first == again


def test_synthetic_declarations_preserve_nested_taint_and_missing_provenance():
    status = candidate_review._synthetic_status
    assert status([{"synthetic": False, "original_plan": {"provenance": {"synthetic": True}}}]) is True
    assert status([{"synthetic": True}, {}]) is True
    assert status([{"synthetic": False}, {}]) is None
    assert status([{"synthetic": False}, {"synthetic": False}]) is False
