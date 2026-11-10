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
from test_choreobench_revision import World


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.issues = [dict(event_id=world.events[0], priority=3, reason="Synthetic issue only.")]
    world.verdicts[world.events[0]] = "uncertain"
    monkeypatch.setattr(revision, "create_provider", lambda *a, **kw: world.text)
    monkeypatch.setattr(revision, "create_music_observer", lambda *a, **kw: world.audio)
    config = world.config(listen=True, max_repairs=1, max_calls=3)
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    root = tmp_path / "experiment"
    manifest = prepare_revision_experiment(world.dataset, world.plan_path, config_path, root)
    revision.diagnose_plan(world.dataset, world.plan_path, root / "diagnosis", read_json(root / "configs/diagnosis.json"))
    for arm in candidate_review.ARMS:
        arm_config = read_json(root / "configs" / (arm + ".json"))
        selection = revision.select_events(world.plan, dict(schema="choreobench.revision_diagnosis.v1",
            source_plan_sha256=object_sha(world.plan), issues=world.issues), arm_config["selection_policy"], 1, arm_config["seed"])
        if arm == "random":
            world.edit_behavior[selection[0]["event_id"]] = "no_op"
        else:
            world.edit_behavior[selection[0]["event_id"]] = "edit"
        revision.run_revision(world.dataset, world.plan_path, root / "diagnosis/diagnosis.json", root / "runs" / arm, arm_config)
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
