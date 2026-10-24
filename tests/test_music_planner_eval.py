"""CPU-only human-written examples; no model weights or private media upload."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts_omnidance/exp/ms/music_planner_eval.py'
spec = importlib.util.spec_from_file_location('music_planner_eval', PATH)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

GOOD = ('[0.000-2.000s] The dancer steps forward and raises both arms. '
        'The hips sway as the feet shift.\n'
        '[2.000-5.062s] The dancer pivots left and bends the knees. '
        'The hands sweep outward and the head turns.')
LOOP = ('[0-2s] The right hand touches the elbow. The right hand touches the elbow. '
        'The right hand touches the elbow.\n'
        '[2-5.0625s] The right hand touches the elbow.')


def row(sid, group=None, sha=None):
    return {'id': sid, 'source_group_id': group or sid, 'audio_sha256': sha or sid,
            'audio_path': '/private/' + sid + '.wav', 'audio_window_seconds': [0.0, 5.0625],
            'target_text': GOOD, 'target_windows': [{'caption': 'GT secret'}],
            'video_path': '/private/GT.mp4', 'reference_image': 'must not pass'}


def record(condition, text=GOOD, sid='a', donor=None, failure=None):
    donor = donor or ('b' if condition == 'mismatched' else 'a')
    return {'id': sid, 'condition': condition, 'raw_text': text, 'source_group_id': 'a',
            'audio_source_group_id': donor, 'audio_sha256': donor,
            'audio_ref': {'path': '/private/' + donor + '.wav'}, 'reference_target': GOOD,
            'metrics': evaluation.text_metrics(text), 'stop_reason': 'eos', 'failure': failure}


class MusicPlannerEvalTest(unittest.TestCase):
    def test_seconds_and_rounding(self):
        result = evaluation.parse_windows(GOOD)
        self.assertTrue(result['matches_fixed_windows'])
        self.assertTrue(result['nonempty_window_text'])
        self.assertEqual(result['window_count'], 2)
        self.assertTrue(evaluation.parse_windows(GOOD.replace('5.062s', '5.100s'))['matches_fixed_windows'])
        self.assertFalse(evaluation.parse_windows(GOOD.replace('5.062s', '5.200s'))['matches_fixed_windows'])

    def test_explicit_millisecond_units(self):
        text = '[0-2000ms] Arms rise. [2000ms-5062.5ms] Feet step.'
        self.assertTrue(evaluation.parse_windows(text)['matches_fixed_windows'])
        self.assertTrue(evaluation.parse_windows('[0s-2s] A. [2s-5.0625s] B.')['matches_fixed_windows'])

    def test_time_failures(self):
        for text in ('Arms rise, feet step.', '[0-2] A. [2-5.0625] B.',
                     '[0-2s] A. [1.8-5.0625s] B.', '[2-5.0625s] B. [0-2s] A.',
                     '[0-0s] A. [2-5.0625s] B.', '[0-2s] A. [2-5.0625s] B. [0-2s] C.',
                     '[0-2frames] A. [2-5.0625frames] B.', '[0-2s] A.'):
            with self.subTest(text=text):
                self.assertFalse(evaluation.parse_windows(text)['matches_fixed_windows'])
        self.assertFalse(evaluation.parse_windows('[0-2s]\n[2-5.0625s]')['nonempty_window_text'])

    def test_loops_and_body_action_diagnostics(self):
        loop = evaluation.text_metrics(LOOP)
        good = evaluation.text_metrics(GOOD)
        self.assertEqual(loop['exact_sentence_max_count'], 4)
        self.assertEqual(loop['repeated_sentence_fraction'], 0.75)
        self.assertGreater(loop['repeated_4gram_fraction'], good['repeated_4gram_fraction'])
        self.assertEqual(good['exact_sentence_max_count'], 1)
        self.assertGreaterEqual(good['body_category_count'], 5)
        self.assertTrue({'step', 'lift', 'turn', 'bend', 'sway', 'sweep'} <= set(good['action_categories']))
        self.assertTrue(good['diagnostic_only_not_quality_score'])
        # The original caption is not modified or de-duplicated.
        self.assertEqual(LOOP.count('The right hand touches the elbow.'), 4)

    def test_actual_word_length_not_character_length(self):
        result = evaluation.text_metrics('Extraordinary choreography. Dance!')
        self.assertEqual(result['word_count'], 3)
        self.assertGreater(result['char_count'], 3)
        self.assertEqual(evaluation.text_metrics('舞者踏步。')['word_count'], 4)
        self.assertEqual(evaluation.text_metrics('')['exact_sentence_max_count'], 0)

    def test_cross_tokenizer_nll_normalization(self):
        a = evaluation.normalize_nll(12.0, 4, 'Step and turn.')
        b = evaluation.normalize_nll(12.0, 8, 'Step and turn.')
        self.assertEqual(a['nll_per_char'], b['nll_per_char'])
        self.assertEqual(a['nll_per_word'], 4.0)
        self.assertNotEqual(a['within_model_nll_per_token'], b['within_model_nll_per_token'])
        for total, count, target in ((float('nan'), 4, 'A'), (1.0, True, 'A'), (-1, 1, 'A'), (1, 1, '')):
            with self.assertRaises(ValueError):
                evaluation.normalize_nll(total, count, target)

    def test_eos_budget_and_native_stop_separate(self):
        self.assertEqual(evaluation.classify_stop([1, 2], [2], 2), 'eos')
        self.assertEqual(evaluation.classify_stop([1, 3], [2], 2), 'budget_truncation')
        self.assertEqual(evaluation.classify_stop([1], [2], 2, native_stop=True), 'native_stop')
        self.assertEqual(evaluation.classify_stop([1], [2], 2), 'unknown_stop')
        with self.assertRaises(ValueError):
            evaluation.classify_stop([1, 2, 3], [2], 2)

    def test_selection_is_model_independent_and_not_test_winner_selection(self):
        rows = [row(str(i)) for i in range(40)]
        a = evaluation.select_rows(rows, 'val')
        b = evaluation.select_rows(list(reversed(rows)), 'val')
        self.assertEqual([r['id'] for r in a], [r['id'] for r in b])
        self.assertEqual(len(a), 32)
        with self.assertRaises(ValueError):
            evaluation.select_rows(rows, 'test')
        self.assertEqual(len(evaluation.select_rows(rows, 'test', limit=500)), 40)
        self.assertEqual([r['id'] for r in evaluation.select_rows(rows, 'test', selection=['3', '1'])], ['3', '1'])
        with self.assertRaises(ValueError):
            evaluation.select_rows(rows, 'val', selection=['3', '3'])

    def test_checkpoint_identity_rejects_changed_portable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'adapter').mkdir()
            (path / 'trainable.pt').write_bytes(b'CPU identity fixture, not real model weights')
            marker = {'schema': 'music-planner-train-v1', 'complete': True,
                      'fingerprint': 'cpu-only', 'cursor': {'update': 1},
                      'trainable_sha256': evaluation.file_ref(path / 'trainable.pt')['sha256']}
            (path / 'trainer_state.json').write_text(json.dumps(marker))
            identity = evaluation.checkpoint_identity(path)
            self.assertEqual(identity['cursor']['update'], 1)
            (path / 'trainable.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                evaluation.checkpoint_identity(path)
            marker['complete'] = False
            (path / 'trainer_state.json').write_text(json.dumps(marker))
            with self.assertRaises(ValueError):
                evaluation.checkpoint_identity(path)

    def test_checkpoint_cannot_mix_another_benchmark_or_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, data = root / 'checkpoint-1', root / 'data'
            (checkpoint / 'adapter').mkdir(parents=True)
            data.mkdir()
            for split in ('train', 'val'):
                (data / f'{split}.jsonl').write_text('CPU benchmark fixture')
            (checkpoint / 'trainable.pt').write_bytes(b'CPU portable fixture')
            config = {'family': 'moss', 'data': {'test_used': False,
                      **{split: {'sha256': evaluation.file_ref(data / f'{split}.jsonl')['sha256']}
                         for split in ('train', 'val')}}}
            fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            marker = {'schema': 'music-planner-train-v1', 'complete': True, 'fingerprint': fingerprint,
                      'cursor': {'update': 1}, 'trainable_sha256': evaluation.file_ref(checkpoint / 'trainable.pt')['sha256']}
            (checkpoint / 'trainer_state.json').write_text(json.dumps(marker))
            (root / 'run_config.json').write_text(json.dumps(dict(config, fingerprint=fingerprint, trainable_parameters=1)))
            self.assertEqual(evaluation.checkpoint_identity(checkpoint, 'moss', data)['fingerprint'], fingerprint)
            with self.assertRaises(ValueError):
                evaluation.checkpoint_identity(checkpoint, 'music_flamingo', data)
            (data / 'val.jsonl').write_text('Another benchmark')
            with self.assertRaises(ValueError):
                evaluation.checkpoint_identity(checkpoint, 'moss', data)

    def test_evaluation_base_and_tokenizer_match_training(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / 'config.json').write_text('{"model_type":"cpu_fixture"}')
            (model / 'tokenizer_config.json').write_text('{"fixture":true}')
            (model / 'model.safetensors').write_bytes(b'CPU fixture, not model weights')
            trained = {'config_sha256': evaluation.file_ref(model / 'config.json')['sha256'],
                       'weights': [{'name': 'model.safetensors', 'size': (model / 'model.safetensors').stat().st_size}],
                       'tokenizer_config.json': evaluation.file_ref(model / 'tokenizer_config.json')['sha256']}
            self.assertFalse(evaluation.verify_base_identity(model, trained)['weight_content_SHA_verified'])
            (model / 'tokenizer_config.json').write_text('{"another_tokenizer":true}')
            with self.assertRaises(ValueError):
                evaluation.verify_base_identity(model, trained)

    def test_prompt_whitelist(self):
        prompt = evaluation.audio_only_input(row('a'), row('b'))
        self.assertEqual(set(prompt), {'id', 'audio_path', 'audio_sha256', 'audio_window_seconds'})
        self.assertEqual(prompt['audio_path'], '/private/b.wav')
        self.assertNotIn('target_text', prompt)
        self.assertNotIn('reference_image', prompt)

    def test_same_known_source_counterfactual_conditions(self):
        conditions = evaluation.make_conditions([row('a'), row('b'), row('c')], True)
        self.assertEqual(len(conditions), 9)
        for source, condition, donor in conditions:
            if condition == 'mismatched':
                self.assertNotEqual(source['source_group_id'], donor['source_group_id'])
                self.assertNotEqual(source['audio_sha256'], donor['audio_sha256'])
            else:
                self.assertEqual(source['id'], donor['id'])
        with self.assertRaises(ValueError):
            evaluation.make_conditions([row('a'), row('b', sha='a')], True)

    def test_counterfactual_is_not_music_sync_score(self):
        records = [record('original'), record('mismatched', LOOP), record('silence', GOOD)]
        result = evaluation.counterfactual_metrics(records)
        self.assertEqual(len(result['pairs']), 2)
        self.assertGreater(result['pairs'][0]['text_edit_change'], 0)
        self.assertEqual(result['pairs'][1]['text_edit_change'], 0)
        self.assertIsNone(result['music_synchronization_score'])
        self.assertIn('NOT better synchrony', result['interpretation'])

    def test_wrong_group_and_failed_counterfactual_are_excluded(self):
        wrong = [record('original'), record('mismatched', LOOP, donor='a'), record('silence')]
        result = evaluation.counterfactual_metrics(wrong)
        self.assertEqual(result['pairs'], [])
        self.assertEqual(result['incomplete_or_invalid_source_ids'], ['a'])
        changed_group = [record('original'), record('mismatched'), record('silence')]
        changed_group[1]['source_group_id'] = 'unrelated_source'
        self.assertEqual(evaluation.counterfactual_metrics(changed_group)['pairs'], [])
        failed = [record('original'), record('mismatched', failure={'type': 'OOM'}), record('silence')]
        self.assertEqual(evaluation.counterfactual_metrics(failed)['pairs'], [])
        with self.assertRaises(ValueError):
            evaluation.counterfactual_metrics([record('original'), record('original')])

    def test_ar_execution_keeps_raw_and_never_passes_gt(self):
        class CPUAdapterContractFixture:
            def generate(self, prompt, max_new_tokens, silence, seed):
                self.prompt = prompt
                self.options = (max_new_tokens, silence, seed)
                return {'raw_text': LOOP, 'new_token_ids': [7, 8, 2], 'eos_token_ids': [2],
                        'id': 'must_not_replace_source', 'stop_reason': 'eos'}

            def teacher_forced(self, source):
                raise RuntimeError('Auxiliary reference scoring fixture failure')

        with tempfile.TemporaryDirectory() as directory:
            source = row('a')
            path = Path(directory) / 'source.wav'
            path.write_bytes(b'CPU contract fixture, not decodable real audio')
            source['audio_path'] = str(path)
            source['audio_sha256'] = evaluation.file_ref(path)['sha256']
            source['target_text'] = 'Tiny GT reference.'
            fixture = CPUAdapterContractFixture()
            result = evaluation.evaluate_one(fixture, source, 'original', source, 2048, teacher_forced=True)
            self.assertEqual(result['raw_text'], LOOP)
            self.assertEqual(result['id'], 'a')
            self.assertEqual(result['native_stop_reason'], 'eos')
            self.assertEqual(result['generated_tokens'], 3)
            self.assertEqual(result['stop_reason'], 'eos')
            self.assertIsNone(result['failure'])
            self.assertEqual(result['teacher_forced_failure']['type'], 'RuntimeError')
            self.assertEqual(set(fixture.prompt), {'id', 'audio_path', 'audio_sha256', 'audio_window_seconds'})
            self.assertEqual(fixture.options, (2048, False, 42))
            silence = evaluation.evaluate_one(fixture, source, 'silence', source, 2048)
            self.assertEqual(fixture.options, (2048, True, 42))
            self.assertEqual(silence['audio_transform'], 'zero_waveform_preserve_length')
            # This checks the contract, not actual model behavior or native silence DSP.

    def test_pending_judge_no_fabricated_rank_or_upload(self):
        records = [record('original'), record('mismatched', LOOP), record('silence')]
        summary = evaluation.summarize(records)
        self.assertIsNone(summary['winner'])
        self.assertEqual(summary['independent_audio_VLM_judge']['status'], 'pending_not_run')
        self.assertIsNone(summary['human_review']['scores'])
        self.assertFalse(summary['automatic_quality_ranking'])
        review = evaluation.review_request(records)
        self.assertFalse(review['external_upload_performed'])
        self.assertTrue(review['authorization_required_before_private_audio_upload'])


if __name__ == '__main__':
    unittest.main()
