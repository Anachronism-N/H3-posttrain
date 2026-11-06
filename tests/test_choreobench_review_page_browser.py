"""Optional real Chromium tests. Simulated ratings are discarded, never submitted."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
import wave
from contextlib import contextmanager

from choreobench.io import file_sha, object_sha, write_json
from choreobench.review_page import generate_review_page
from choreobench.review_server import create_review_server
from choreobench.reviews import blank_review_template, build_review_pack


@contextmanager
def running(count=4, edit=False):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original = root / 'fixture.wav'
        with wave.open(str(original), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(bytes([1, 0]) * 81000)
        items = []
        for index in range(count):
            sid = 'fixture-' + str(index)
            song = dict(song_id=sid, split='dev', song_group_id=None, artist_group_id=None,
                near_duplicate_group_id=None, brief='Synthetic browser fixture, not a measured dance.',
                audio=dict(path=str(original), sha256=file_sha(original), sample_rate=16000, num_samples=81000),
                music=dict(source='fixture', beats=[], downbeats=[], sections=[dict(section_id='s', start=0,
                    end=81000, label='unknown', repeat_group=None)]))
            plan = dict(schema='choreobench.plan.v1', song_id=sid, sample_rate=16000, num_samples=81000,
                motifs=[], provenance={}, events=[dict(event_id='e', start=0, end=81000, track='body',
                    action='Step left. </script><script>window.injected=true</script>', attributes={}, entry={},
                    exit={}, motif_id=None, evidence=[])])
            item = dict(song=song, plan=plan, method_id='hidden-method')
            if edit:
                source = copy.deepcopy(plan)
                item.update(source_plan=source, edit_task=dict(schema='choreobench.edit_task.v1', task_id='edit',
                    song_id=sid, source_plan_sha256=object_sha(source), instruction='Change only this action.',
                    allowed_event_ids=['e'], repair_event_ids=[], protected=[], goals=[], semantic_goals=[]))
            items.append(item)
        pack = build_review_pack(items)['public']
        asset = root / pack['items'][0]['song']['audio']['path']
        asset.parent.mkdir(); asset.write_bytes(original.read_bytes())
        public = root / 'pack.json'; write_json(public, pack)
        template = root / 'reviews.template.json'; write_json(template, blank_review_template(pack))
        generate_review_page(public, root / 'index.html')
        expected = template.read_bytes()
        server = create_review_server(root, proxy_prefix='/proxy/8780')
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            yield 'http://127.0.0.1:' + str(server.server_address[1]) + '/proxy/8780/', pack, template, expected
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)
            if thread.is_alive():
                raise AssertionError('Fixture review server failed to stop')


@unittest.skipUnless(importlib.util.find_spec('playwright'), 'Optional Playwright/Chromium runtime not installed')
class ReviewPageBrowserTests(unittest.TestCase):
    def chromium(self, engine):
        from playwright.sync_api import Error
        try:
            return engine.chromium.launch(headless=True, args=['--no-sandbox'])
        except Error as error:
            if "Executable doesn't exist" in str(error):
                self.skipTest('Optional Chromium binary not installed; no download performed')
            raise

    def test_dynamic_cases_comments_escape_export_and_audio_declaration(self):
        from playwright.sync_api import sync_playwright
        with running() as (url, pack, template, expected), sync_playwright() as engine:
            browser = self.chromium(engine)
            try:
                page = browser.new_page(accept_downloads=True); errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(url + '#case-4', wait_until='networkidle')
                self.assertTrue(page.locator('#case-4').is_visible())
                self.assertEqual(page.locator('#cases a').count(), 4)
                self.assertEqual(page.locator('#rubric tr').count(), 5)
                self.assertIsNone(page.evaluate('window.injected'))
                row = page.locator('#case-4 tr[data-start]').first
                self.assertIn('</script><script>', row.inner_text())
                comment = page.locator('#case-4 textarea').first; comment.fill('Existing comment')
                row.click(); drafted = comment.input_value()
                self.assertTrue(drafted.startswith('Existing comment | '))
                self.assertIn('events=event_0001', drafted)
                self.assertIn('samples=[0,81000);', drafted)
                page.locator('#case-4 .card select').nth(1).select_option('3')
                self.assertTrue(page.locator('#case-4 .card select').nth(4).is_disabled())
                self.assertTrue(page.locator('#case-4 .card select').nth(5).is_disabled())
                page.locator('#cases a').first.click()
                page.locator('#cases a').nth(3).click()
                self.assertEqual(comment.input_value(), drafted)
                self.assertEqual(page.locator('#case-4 .card select').nth(1).input_value(), '3')
                audio = page.locator('#case-4 audio')
                audio.evaluate('(a)=>new Promise((resolve,reject)=>{if(a.readyState>=1){resolve();return}a.addEventListener("loadedmetadata",resolve,{once:true});a.addEventListener("error",()=>reject(new Error("audio failed")),{once:true});a.load()})')
                self.assertAlmostEqual(audio.evaluate('(a)=>a.duration'), 5.0625, places=3)
                audio.evaluate('(a)=>new Promise(resolve=>{a.addEventListener("seeked",resolve,{once:true});a.currentTime=2})')
                self.assertEqual(page.locator('#case-4 tr.active').count(), 1)
                page.locator('#reviewer').fill('browser-test-not-actual-reviewer')
                page.locator('#case-4 .card select').first.select_option('4')
                dialogs = []
                page.on('dialog', lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
                page.locator('#export').click()
                self.assertTrue(any('听音声明' in message for message in dialogs))
                page.locator('#case-4 .card select').first.select_option('')
                with page.expect_download() as download:
                    page.locator('#export').click()
                exported = json.loads(Path(download.value.path()).read_text())
                self.assertEqual(len(exported['reviews']), 4)
                self.assertTrue(all(not row['audio_seen'] for row in exported['reviews']))
                self.assertEqual(sum(row['comments'] == drafted for row in exported['reviews']), 1)
                page.goto(url + '#case-999', wait_until='networkidle')
                self.assertTrue(page.locator('#case-1').is_visible())
                self.assertFalse(errors, errors)
                self.assertEqual(template.read_bytes(), expected)
            finally:
                browser.close()

    def test_edit_instruction_source_and_edit_scores_available(self):
        from playwright.sync_api import sync_playwright
        with running(count=1, edit=True) as (url, pack, template, expected), sync_playwright() as engine:
            browser = self.chromium(engine)
            try:
                page = browser.new_page(); page.goto(url, wait_until='networkidle')
                self.assertEqual(page.locator('#case-1 .card').count(), 1)
                self.assertIn('编辑指令与原计划', page.locator('#case-1 .card details').first.inner_text())
                self.assertFalse(page.locator('#case-1 .card select').nth(4).is_disabled())
                self.assertFalse(page.locator('#case-1 .card select').nth(5).is_disabled())
                self.assertEqual(template.read_bytes(), expected)
            finally:
                browser.close()


if __name__ == '__main__':
    unittest.main()
