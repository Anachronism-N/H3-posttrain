"""Public HTML tests use synthetic packages, never actual reviewers or scores."""
import copy
import json
import re

import pytest

from choreobench import review_page
from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.reviews import blank_review_template, build_review_pack
from test_choreobench_review_sources import fixture


def pack(count=1, edit=False):
    items = []
    for i in range(count):
        item = fixture()
        sid = 'fixture-' + str(i)
        item['song']['song_id'] = item['plan']['song_id'] = sid
        item['song']['audio']['sha256'] = object_sha(sid)
        item['method_id'] = 'private-method-label'
        item['plan']['provenance'] = {'private': 'PRIVATE_RUNTIME_PATH'}
        item['plan']['events'][0]['evidence'] = ['PRIVATE_CALL_ID']
        if edit:
            source = copy.deepcopy(item['plan'])
            task = dict(schema='choreobench.edit_task.v1', task_id='private-task', song_id=sid,
                source_plan_sha256=object_sha(source), instruction='Change this event only.',
                allowed_event_ids=['e'], repair_event_ids=[], protected=[],
                goals=[dict(event_id='e', field='action', op='contains', value='PRIVATE_GOAL')],
                semantic_goals=['PRIVATE_SEMANTIC_GOAL'])
            item.update(source_plan=source, edit_task=task)
        items.append(item)
    return build_review_pack(items, seed=7)['public']


def embedded(page):
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)[1])


def rehash(item):
    item['item_sha256'] = object_sha({k: v for k, v in item.items() if k != 'item_sha256'})


def test_page_is_public_only_and_always_embeds_fresh_blank_ratings():
    document = pack(4)
    before = copy.deepcopy(document)
    html = review_page.render_review_page(document)
    payload = embedded(html)
    assert payload['pack'] == document and document == before
    assert payload['template'] == blank_review_template(document)
    assert payload['planningMusic'] == {}
    assert 'PRIVATE_' not in html and 'private-method-label' not in html
    assert '#case-([0-9]+)' in html and 'requested<ids.length' in html
    assert 'samples=[' in html and 'comment.value+=' in html
    assert '评分锚点' in html and '校准草案' in html
    assert 'select.disabled=key.startsWith' in html
    assert all(not row['audio_seen'] and all(v is None for v in row['ratings'].values())
               for row in payload['template']['reviews'])


def test_script_closing_text_and_html_in_every_public_string_are_data_not_markup():
    document = pack()
    hostile = '</script><script>alert(1)</script><img src=x onerror=alert(1)> DATA'
    item = document['items'][0]
    item['plan']['events'][0]['action'] = hostile
    item['song']['brief'] = hostile
    rehash(item)
    html = review_page.render_review_page(document)
    payload = embedded(html)
    assert payload['pack']['items'][0]['plan']['events'][0]['action'] == hostile
    assert hostile not in html
    assert html.count('<script') == 2
    assert 'textContent=text' in html


def test_planning_music_is_separate_and_clock_validated_without_mutation():
    document = pack(2)
    music = {item['song']['song_id']: copy.deepcopy(item['song']['music']) for item in document['items']}
    before = copy.deepcopy(music)
    html = review_page.render_review_page(document, music)
    assert embedded(html)['planningMusic'] == music == before
    assert embedded(html)['pack'] == document


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'private', 'clock', 'nonfinite', 'not_map'])
def test_invalid_public_planning_music_rejected(mutation):
    document = pack()
    sid = document['items'][0]['song']['song_id']
    music = {sid: copy.deepcopy(document['items'][0]['song']['music'])}
    if mutation == 'missing': music = {}
    elif mutation == 'extra': music['private-song'] = copy.deepcopy(music[sid])
    elif mutation == 'private': music[sid]['goals'] = ['PRIVATE_GOAL']
    elif mutation == 'clock': music[sid]['sections'][0]['end'] += 1
    elif mutation == 'nonfinite': music[sid]['beats'] = [float('nan')]
    else: music = []
    with pytest.raises(ValueError):
        review_page.render_review_page(document, music)


@pytest.mark.parametrize('path', ['https://example.test/secret.wav', '/absolute/song.wav',
    '../private.wav', 'assets/../private.wav', 'assets/query.wav?private=1', 'assets/%2e%2e', 'assets/a/b.wav'])
def test_external_or_unsafe_audio_routes_rejected(path):
    document = pack()
    item = document['items'][0];item['song']['audio']['path'] = path;rehash(item)
    with pytest.raises(ValueError): review_page.render_review_page(document)


def test_same_song_cannot_have_inconsistent_public_audio_or_music():
    document = pack()
    item = copy.deepcopy(document['items'][0]);item['review_id'] = 'review-extra'
    item['song']['audio']['sha256'] = object_sha('different-waveform');rehash(item)
    document['items'].append(item)
    with pytest.raises(ValueError, match='inconsistent'):
        review_page.render_review_page(document)


def test_plan_cannot_escape_public_song_clock_even_with_recomputed_item_hash():
    document = pack();item = document['items'][0]
    item['plan']['num_samples'] += 1;rehash(item)
    with pytest.raises(ValueError): review_page.render_review_page(document)


def test_private_bindings_or_changed_public_item_rejected_before_publication(tmp_path):
    public = tmp_path / 'pack.json';out = tmp_path / 'index.html'
    write_json(public, {'schema': 'choreobench.review_bindings.v1', 'items': []})
    with pytest.raises(ValueError): review_page.generate_review_page(public, out)
    assert not out.exists()
    document = pack();document['items'][0]['plan']['events'][0]['action'] = 'changed'
    write_json(public, document, overwrite=True)
    with pytest.raises(ValueError): review_page.generate_review_page(public, out)
    assert not out.exists()


def test_edit_source_and_instruction_are_public_but_private_goals_never_embedded():
    document = pack(edit=True)
    payload = embedded(review_page.render_review_page(document))
    assert payload['pack']['items'][0]['edit_task']['instruction'] == 'Change this event only.'
    assert 'PRIVATE_' not in json.dumps(payload)
    assert '编辑指令与原计划' in review_page.render_review_page(document)


@pytest.mark.parametrize('mutation', ['goals', 'source_provenance', 'source_evidence', 'source_clock', 'missing_source'])
def test_forged_edit_public_fields_are_rejected(mutation):
    document = pack(edit=True);item = document['items'][0]
    if mutation == 'goals': item['edit_task']['goals'] = ['PRIVATE_GOAL']
    elif mutation == 'source_provenance': item['source_plan']['provenance'] = {'private': 'secret'}
    elif mutation == 'source_evidence': item['source_plan']['events'][0]['evidence'] = ['secret']
    elif mutation == 'source_clock': item['source_plan']['sample_rate'] += 1
    else: item['source_plan'] = None
    rehash(item)
    with pytest.raises(ValueError): review_page.render_review_page(document)


def test_generate_exclusively_publishes_only_html_and_preserves_all_existing_files(tmp_path):
    public = tmp_path / 'pack.json';out = tmp_path / 'nested/index.html'
    document = pack();write_json(public, document)
    snapshot = public.read_bytes()
    report = review_page.generate_review_page(public, out)
    assert public.read_bytes() == snapshot
    assert report['page_file_sha256'] == file_sha(out)
    assert report['public_pack_sha256'] == object_sha(document)
    assert report['review_items'] == report['songs'] == 1 and report['new_model_calls'] == 0
    assert report['planning_music_file_sha256'] is None and not report['quality_verified']
    old = out.read_bytes()
    with pytest.raises(FileExistsError): review_page.generate_review_page(public, out)
    assert old == out.read_bytes() and not list(out.parent.glob('*.tmp'))


def test_input_drift_is_rejected_before_html_publication(tmp_path, monkeypatch):
    public = tmp_path / 'pack.json';out = tmp_path / 'index.html'
    write_json(public, pack());render = review_page.render_review_page
    def changed(document, music):
        result = render(document, music)
        public.write_bytes(public.read_bytes() + b' ')
        return result
    monkeypatch.setattr(review_page, 'render_review_page', changed)
    with pytest.raises(ValueError, match='input changed'):
        review_page.generate_review_page(public, out)
    assert not out.exists()


@pytest.mark.parametrize('module_cli', [True, False])
def test_page_cli_accepts_only_public_pack_and_optional_planning_music(tmp_path, module_cli):
    public = tmp_path / 'pack.json';out = tmp_path / 'index.html';music_path = tmp_path / 'music.json'
    document = pack();write_json(public, document)
    write_json(music_path, {i['song']['song_id']: i['song']['music'] for i in document['items']})
    args = ['--public', str(public), '--out', str(out), '--planning-music', str(music_path)]
    if module_cli:
        report = review_page.main(args)
        assert report['planning_music_file_sha256'] == file_sha(music_path)
        with pytest.raises(FileExistsError): review_page.main(args)
    else:
        from choreobench.cli import main
        assert main(['review-page'] + args) == 0
        assert main(['review-page'] + args) == 2
    assert embedded(out.read_text(encoding='utf-8'))['pack'] == document
