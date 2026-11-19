"""Real frozen CPU validators and synthetic negative contracts; no actual model calls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
import caption_direct_facts_v1 as direct
bar, pipeline = direct.native()


def facts():
    return [{'fact_id': 'w0:e1', 'window_id': 'w0', 'text': 'The dancer turns the head.',
             'raw_text': 'The dancer turns the head.', 'parts': ['head'], 'frame_indices': [0, 1]},
            {'fact_id': 'w0:e2', 'window_id': 'w0', 'text': 'The dancer opens the left hand.',
             'raw_text': 'The dancer opens the left hand.', 'parts': ['left_hand'], 'frame_indices': [0, 1]},
            {'fact_id': 'w1:e1', 'window_id': 'w1', 'text': 'The dancer raises the right arm.',
             'raw_text': 'The dancer raises the right arm.', 'parts': ['right_arm'], 'frame_indices': [2, 3]}]


def windows():
    return [{'window_id': 'w0', 'start': 0., 'end': 1.},
            {'window_id': 'w1', 'start': 1., 'end': 2.}]


class DirectFactsContracts(unittest.TestCase):
    def test_all_exact_native_text_and_ids_source_order(self):
        original = facts()
        before = copy.deepcopy(original)
        body = direct.direct_caption(windows(), original, [])
        self.assertEqual(body['windows'][0]['caption'], ' '.join(f['text'] for f in original[:2]))
        self.assertEqual(body['windows'][0]['used_fact_ids'], ['w0:e1', 'w0:e2'])
        self.assertEqual(body['issues'], [])
        self.assertEqual(original, before)
        self.assertIs(bar.check_supported_rewrite(body, windows(), original, []), body)

    def test_wrong_parts_rejected_by_actual_native_validator(self):
        original = facts()
        original[0]['text'] = 'The dancer moves the right leg.'
        with self.assertRaisesRegex(ValueError, 'body part absent'):
            direct.direct_caption(windows(), original, [])

    def test_fine_hand_detail_not_discarded(self):
        original = facts()
        original[1]['text'] = original[1]['raw_text'] = 'The dancer curls the left fingers into a closed fist.'
        body = direct.direct_caption(windows(), original, [])
        self.assertIn(original[1]['text'], body['windows'][0]['caption'])
        self.assertIn(original[1]['fact_id'], body['windows'][0]['used_fact_ids'])

    def test_native_subject_conversion_not_raw_gendered_text(self):
        original = facts()
        original[0]['raw_text'] = 'She turns her head.'
        original[0]['text'] = bar.core.neutral_fact_text(original[0]['raw_text'])
        body = direct.direct_caption(windows(), original, [])
        self.assertIn("the dancer turns the dancer's head.", body['windows'][0]['caption'])
        body['windows'][0]['caption'] = original[0]['raw_text'] + ' ' + original[1]['text']
        with self.assertRaisesRegex(ValueError, 'neutral dancer'):
            bar.check_supported_rewrite(body, windows(), original, [])

    def test_empty_window_is_partial_not_stillness(self):
        original = facts()[:2]
        body = direct.direct_caption(windows(), original, [])
        self.assertEqual(body['windows'][1], {'window_id': 'w1', 'caption': '',
            'used_fact_ids': [], 'used_boundary_ids': []})
        body['windows'][1]['caption'] = 'The dancer remains still.'
        with self.assertRaisesRegex(ValueError, 'must remain empty'):
            bar.check_supported_rewrite(body, windows(), original, [])

    def test_all_empty_is_abstain_and_valid(self):
        final = direct.direct_final(windows(), [], [])
        self.assertEqual(final['status'], 'abstain')
        self.assertTrue(all(row['caption'] == '' for row in final['json']['windows']))

    def test_unknown_and_cross_window_fact_refs_rejected(self):
        original = facts()
        for invalid in ['fake:e1', 'w1:e1']:
            body = direct.direct_caption(windows(), original, [])
            body['windows'][0]['used_fact_ids'][0] = invalid
            with self.assertRaisesRegex(ValueError, 'unknown or belongs'):
                bar.check_supported_rewrite(body, windows(), original, [])

    def test_lost_fact_reference_rejected_not_fake_issues(self):
        original = facts()
        body = direct.direct_caption(windows(), original, [])
        body['windows'][0]['used_fact_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'omitted facts require'):
            bar.check_supported_rewrite(body, windows(), original, [])

    def test_duplicate_or_orphan_facts_rejected(self):
        for original in [facts() + [facts()[0]], [dict(facts()[0], window_id='alien')]]:
            with self.assertRaises(ValueError):
                direct.direct_caption(windows(), original, [])

    def test_fake_boundary_rejected_real_boundaries_unused(self):
        boundary = {'boundary_id': 'b:w0:w1:head', 'relation': 'repeat',
                    'left_window_id': 'w0', 'right_window_id': 'w1'}
        body = direct.direct_caption(windows(), facts(), [boundary])
        self.assertTrue(all(row['used_boundary_ids'] == [] for row in body['windows']))
        body['windows'][0]['used_boundary_ids'] = ['fakeboundary']
        with self.assertRaisesRegex(ValueError, 'boundary ref unknown'):
            bar.check_supported_rewrite(body, windows(), facts(), [boundary])

    def test_wrong_frame_refs_rejected_by_actual_observation_validator(self):
        manifest = {'window': windows()[0], 'frames': [{'index': 0, 'time': 0.}, {'index': 1, 'time': .5}]}
        value = bar.empty_observation('unknown')
        value['events'] = [{'id': 'e1', 'description': 'The dancer turns the head.',
                            'parts': ['head'], 'frame_indices': [0, 9]}]
        value['part_support'][bar.PARTS.index('head')]['frame_indices'] = [0, 1]
        value['abstain'] = False
        with self.assertRaisesRegex(ValueError, 'outside actual'):
            bar.validate_observation(value, manifest)

    def test_unadmitted_bad_raw_context_cannot_remove_admitted_hand_fact(self):
        manifest = {'window': windows()[0], 'frames': [{'index': 0, 'time': 0.}, {'index': 1, 'time': .5}]}
        raw = bar.empty_observation('unknown')
        raw['part_support'][bar.PARTS.index('left_hand')]['frame_indices'] = [0, 1]
        raw.update(events=[{'id': 'e1', 'description': 'The dancer opens the left hand. The right foot moves.',
                            'parts': ['left_hand'], 'frame_indices': [0, 1]}], abstain=False)
        replay = bar.filter_supported_observation(raw, manifest, {})
        admitted = bar.facts_from_observations([windows()[0]], [{'status': 'candidate', 'json': replay}])
        self.assertEqual(len(admitted), 1)
        body = direct.direct_caption([windows()[0]], admitted, [])
        self.assertEqual(body['windows'][0]['caption'], admitted[0]['text'])
        self.assertIn('left hand', body['windows'][0]['caption'])
        self.assertNotIn('right foot', body['windows'][0]['caption'])

    def test_validator_error_preserved_not_generic_success(self):
        original = facts()
        original[0]['text'] = 'She turns her head.'
        result = direct.direct_final(windows(), original, [])
        self.assertEqual(result['status'], 'failed')
        self.assertIn('neutral dancer', result['error'])
        self.assertNotIn('json', result)
        self.assertFalse(result['model_call_performed'])

    def test_hook_requires_explicit_boundary_off_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as out:
            args = SimpleNamespace(boundary_review='on', out=out)
            with self.assertRaisesRegex(ValueError, 'boundary-review=off'):
                with direct.direct_recipe(bar, args):
                    pass
            args.boundary_review = 'off'
            Path(out, 'old-generation').write_text('old')
            with self.assertRaisesRegex(ValueError, 'fresh isolated'):
                with direct.direct_recipe(bar, args):
                    pass

    def test_hook_replaces_only_native_boundary_and_final_preserves_identity(self):
        with tempfile.TemporaryDirectory() as out:
            args = SimpleNamespace(boundary_review='off', out=out)
            original_process, original_params = bar.process_source, bar.params
            with direct.direct_recipe(bar, args) as identity:
                self.assertIsNot(bar.process_source, original_process)
                self.assertEqual(identity['caption_recipe'], direct.RECIPE)
                self.assertFalse(identity['training_eligible'])
                self.assertEqual(identity['fulltext_model_calls'], 0)
                globals_ = bar.process_source.__globals__
                self.assertIs(globals_['filter_supported_observation'], bar.filter_supported_observation)
                self.assertIs(globals_['visual_request'], bar.visual_request)
                self.assertIs(globals_['DIRECT_FINAL'], direct.direct_final)
            self.assertIs(bar.process_source, original_process)
            self.assertIs(bar.params, original_params)

    def test_recipe_params_immutable_and_native_sha_not_per_window(self):
        with tempfile.TemporaryDirectory() as out:
            args = bar.parser().parse_args(['--plan', '/unused/plan', '--model', '/unused/model',
                                          '--out', out, '--boundary-review', 'off'])
            with direct.direct_recipe(bar, args):
                original = bar.params(args, {'frozen': 'code'})
                self.assertTrue(original['no_actionclaim_from_unknown_boundaries'])
                original['caption_recipe'] = 'tampered-returned-copy'
                self.assertEqual(bar.params(args, {'frozen': 'code'})['caption_recipe'], direct.RECIPE)
                with patch.object(direct, 'sha', side_effect=AssertionError('no per-window/source code SHA')):
                    direct.direct_caption(windows(), facts(), [])
                    direct.direct_caption(windows(), facts(), [])
                args.visual_max_tokens += 1
                with self.assertRaisesRegex(ValueError, 'parameters changed'):
                    bar.params(args, {'frozen': 'code'})

    def test_hook_synthetic_native_process_dispatches_only_original_rgb_observation(self):
        with tempfile.TemporaryDirectory() as out:
            root = Path(out) / 'source'
            window = windows()[0]
            manifest = {'window': window, 'frames': [
                {'index': 0, 'time': 0., 'source_timestamp': 0., 'file': '0.jpg'},
                {'index': 1, 'time': .5, 'source_timestamp': .5, 'file': '1.jpg'}]}
            prepared = {'forbidden': {}, 'windows': [manifest],
                'boundaries': [{'must_not_dispatch': True}], 'union_manifest_sha256': 'a' * 64,
                'hand_roi': {'available': False}, 'raw2d_mask': {'available': False}}
            args = SimpleNamespace(boundary_review='off', out=out, stage='run', retry_failed=False,
                                   model=str(Path(out) / 'unused-model'), visual_max_tokens=1024)
            row = {'id': 'case1', 'manifest_record': {}, 'source_refs': {}, 'window_manifest': {},
                   'windows': [window], 'plan_status': 'ready'}
            raw = bar.empty_observation('unknown')
            raw['part_support'][bar.PARTS.index('head')]['frame_indices'] = [0, 1]
            raw.update(events=[{'id': 'e1', 'description': 'The dancer turns the head.',
                                'parts': ['head'], 'frame_indices': [0, 1]}], abstain=False)
            admitted = bar.filter_supported_observation(raw, manifest, {})
            requests = []

            def transactions(items, runtime, args):
                requests.extend(item['request'] for item in items)
                self.assertTrue(all(item['request']['messages'][0]['content'] == bar.SYSTEM for item in items))
                return {str(item['root']): {'status': 'candidate', 'json': copy.deepcopy(admitted)} for item in items}

            with patch.object(bar, 'verify_prepared', return_value=prepared), patch.object(bar, 'transactions', side_effect=transactions):
                with direct.direct_recipe(bar, args):
                    (root / 'rgb').mkdir(parents=True)
                    (root / 'prepared.json').write_text('{}')
                    (root / 'rgb/0.jpg').write_bytes(b'synthetic CPU-only image bytes')
                    (root / 'rgb/1.jpg').write_bytes(b'synthetic CPU-only image bytes')
                    expected_request = bar.visual_request(args, root / 'rgb', manifest, {})
                    result = bar.process_source(row, b'{}', root, args, {}, None, {})
                    second_root = Path(out) / 'source2'
                    (second_root / 'rgb').mkdir(parents=True)
                    (second_root / 'prepared.json').write_text('{}')
                    (second_root / 'rgb/0.jpg').write_bytes(b'synthetic CPU-only image bytes')
                    (second_root / 'rgb/1.jpg').write_bytes(b'synthetic CPU-only image bytes')
                    second_request = bar.visual_request(args, second_root / 'rgb', manifest, {})
                    second = bar.process_source(dict(row, id='case2'), b'{}', second_root, args, {}, None, {})
                    self.assertEqual(second['caption_recipe'], result['caption_recipe'])
                    self.assertEqual(second['direct_recipe_execution_ref'], result['direct_recipe_execution_ref'])
                    self.assertEqual(second['boundary_observations'], [])
                    self.assertEqual(second['rewrite_fulltext_calls_per_attempt'], 0)
            self.assertEqual(requests, [expected_request, second_request])
            self.assertEqual(result['caption_recipe'], direct.RECIPE)
            self.assertEqual(result['schema'], direct.RECIPE)
            self.assertFalse(result['native_fulltext_model_output'])
            self.assertTrue(result['no_actionclaim_from_unknown_boundaries'])
            self.assertFalse(result['training_eligible'])
            self.assertEqual(result['boundary_observations'], [])
            self.assertEqual(result['rewrite_fulltext_calls_per_attempt'], 0)
            self.assertEqual(result['final']['json']['windows'][0]['caption'], 'The dancer turns the head.')
            on_disk = json.loads(Path(result['result_ref']['path']).read_bytes())
            self.assertEqual(on_disk['caption_recipe'], direct.RECIPE)
            self.assertFalse(on_disk['training_eligible'])


def run_cpu_tests():
    modules = []
    for filename in ('test_ms_bar_annotation.py', 'test_ms_caption_rewrite.py'):
        path = direct.REPO / 'tests' / filename
        spec = importlib.util.spec_from_file_location('direct_original_' + path.stem, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        modules.append(value)
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(DirectFactsContracts)])
    for value in modules:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(value))
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
    summary = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'success': result.wasSuccessful(),
               'skip_reasons': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
               'original_tests_enabled': [str(direct.REPO / 'tests' / name) for name in
                   ('test_ms_bar_annotation.py', 'test_ms_caption_rewrite.py')],
               'actual_frozen_validator_module': str(Path(bar.__file__).resolve()),
               'actual_model_calls': 0}
    if not result.wasSuccessful():
        raise RuntimeError('Real frozen CPU validator/negative tests failed: ' + repr(summary))
    return summary


if __name__ == '__main__':
    import json
    print(json.dumps(run_cpu_tests(), sort_keys=True))
