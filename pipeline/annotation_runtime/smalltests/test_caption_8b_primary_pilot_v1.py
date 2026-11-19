"""CPU-only fixed-source request preservation and no-retry benchmark tests."""
import copy
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

RT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot', RT / 'caption_8b_primary_pilot_v1.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class FakeLLM:
    _ms_bar_model_sha256 = p.MODEL_SHA
    _ms_8b_pilot_settings = p.SETTINGS

    def __init__(self, records, broken=None):
        self.records, self.broken, self.calls, self.position = records, broken, [], 0

    def chat(self, messages, sampling, use_tqdm=False):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(sampling)))
        result = []
        for message, sample in zip(messages, sampling):
            record = self.records[self.position]
            self.position += 1
            assert message == record['request']['messages']
            assert sample['json_schema'] == record['request']['json_schema']
            assert sample['sampling'] == record['request']['sampling']
            assert sample['disable_any_whitespace'] is True
            indices = [f['index'] for f in record['manifest']['frames']][:2]
            value = dict(events=[dict(id='e1', description='The dancer shifts the torso.',
                parts=['torso'], frame_indices=indices)],
                part_support=[dict(part=part, frame_indices=indices if part == 'torso' else [],
                                   reason='visible' if part == 'torso' else 'unknown') for part in p.original_parts],
                uncertainties=['CPU fake new output; not model quality evidence'], abstain=False)
            if self.broken == 'structure':
                value['part_support'] = value['part_support'][:-1]
            completion = SimpleNamespace(text=json.dumps(value), token_ids=[1, 2, 3], finish_reason='stop')
            result.append(SimpleNamespace(outputs=[completion], prompt_token_ids=[4, 5]))
        return result[:-1] if self.broken == 'cardinality' else result


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = Path(os.environ['CAPTION_8B_PILOT_MANIFEST'])
        cls.manifest = json.loads(cls.fixture.read_bytes())
        cls.core = p.original_filter(RT / 'source_curated_consumer_v1')
        p.original_parts = cls.core.PARTS

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='caption-8b-cpu-')
        self.root = Path(self.temp.name)
        manifest_ref = p.write_once(self.root / 'manifest.json', self.manifest)
        self.prepared = SimpleNamespace(manifest=copy.deepcopy(self.manifest), manifest_ref=manifest_ref, core=self.core)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    @contextmanager
    def admitted(manifest):
        yield dict(admitted=True, node='N6', gpus=[3], tp=1, test_only=True)

    @staticmethod
    def sampling(request):
        return dict(json_schema=request['json_schema'], sampling=request['sampling'], disable_any_whitespace=True)

    def run_fake(self, name='run', broken=None, batch_size=8):
        llm = FakeLLM(self.manifest['requests'], broken)
        receipt = p.benchmark(self.prepared, self.root / name, admission=self.admitted,
                              runtime=lambda: llm, sampling=self.sampling, batch_size=batch_size)
        return llm, receipt

    def test_all_eight_requests_unchanged_except_model(self):
        self.assertEqual(self.manifest['case_ids'], list(p.CASES))
        self.assertEqual(len(self.manifest['requests']), 8)
        for record in self.manifest['requests']:
            original = json.loads(Path(record['original_request_ref']['path']).read_bytes())
            original['model'] = p.MODEL
            self.assertEqual(original, record['request'])
            self.assertEqual(p.digest({k:v for k,v in original.items() if k != 'model'}),
                             record['source_request_without_model_sha256'])
        self.assertEqual(self.manifest['budgets']['new_8B_window_calls'], 8)

    def test_batch8_new_sample_original_filter_and_complete_fact_text(self):
        llm, receipt = self.run_fake()
        self.assertEqual([len(c[0]) for c in llm.calls], [8])
        self.assertEqual(receipt['new_window_calls'], 8)
        self.assertEqual(llm.position, 8)
        self.assertFalse(receipt['measured_real_model'])
        self.assertFalse(receipt['quality_approved'])
        self.assertTrue(all(c['full_window_coverage'] for c in receipt['candidates']))
        for record, result in zip(self.manifest['requests'], receipt['observations']):
            raw = Path(result['raw_notes_ref']['path']).read_bytes()
            original_filter = self.core.filter_supported_observation(
                self.core.parse_response(json.dumps(raw.decode()).encode()), record['manifest'], record['forbidden'])
            self.assertEqual(result['json'], original_filter)
        for candidate in receipt['candidates']:
            for caption in candidate['captions']:
                self.assertEqual(caption['caption'], 'The dancer shifts the torso.')
                self.assertEqual(len(caption['used_fact_ids']), 1)
        with self.assertRaisesRegex(ValueError, 'terminal'):
            p.benchmark(self.prepared, self.root / 'run', admission=self.admitted,
                        runtime=lambda: self.fail('terminal run redispatched'), sampling=self.sampling)

    def test_batch2_each_window_once(self):
        llm, receipt = self.run_fake(batch_size=2)
        self.assertEqual([len(c[0]) for c in llm.calls], [2, 2, 2, 2])
        self.assertEqual(receipt['new_window_calls'], 8)

    def test_defaultdeny_does_not_load_or_write(self):
        with self.assertRaisesRegex(ValueError, 'default deny'):
            p.benchmark(self.prepared, self.root / 'denied')
        self.assertFalse((self.root / 'denied').exists())

    def test_mutated_prompt_denied_before_runtime(self):
        self.prepared.manifest['requests'][0]['request']['messages'][0]['content'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            p.benchmark(self.prepared, self.root / 'mutated', admission=self.admitted,
                        runtime=lambda: self.fail('mutated request reached runtime'), sampling=self.sampling)

    def test_failed_structure_terminal_no_retry(self):
        llm, receipt = self.run_fake(broken='structure')
        self.assertTrue(all(o['status'] == 'failed' for o in receipt['observations']))
        self.assertEqual(llm.position, 8)
        self.assertFalse(any(c['full_window_coverage'] for c in receipt['candidates']))
        with self.assertRaisesRegex(ValueError, 'terminal'):
            p.benchmark(self.prepared, self.root / 'run', admission=self.admitted,
                        runtime=lambda: self.fail('failed request redispatched'), sampling=self.sampling)

    def test_wrong_gpu_and_model_marker_denied(self):
        @contextmanager
        def wrong_gpu(manifest):
            yield dict(admitted=True, node='N7', gpus=[6], tp=1, test_only=True)
        with self.assertRaisesRegex(ValueError, 'N6 GPU3'):
            p.benchmark(self.prepared, self.root / 'wrong_gpu', admission=wrong_gpu,
                        runtime=lambda: self.fail('reviewer GPU load'), sampling=self.sampling)
        llm = FakeLLM(self.manifest['requests'])
        llm._ms_bar_model_sha256 = p.SOURCE_MODEL_SHA
        with self.assertRaisesRegex(ValueError, 'marker mismatch'):
            p.benchmark(self.prepared, self.root / 'wrong_marker', admission=self.admitted,
                        runtime=lambda: llm, sampling=self.sampling)
        self.assertFalse(llm.calls)

    def test_conflicting_completion_count_fails_without_redispatch(self):
        llm, receipt = self.run_fake(broken='cardinality')
        self.assertEqual(llm.position, 8)
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(all(o['status'] == 'failed' for o in receipt['observations']))
        self.assertTrue(all('one completion' in o['error'] for o in receipt['observations']))
        with self.assertRaisesRegex(ValueError, 'terminal'):
            p.benchmark(self.prepared, self.root / 'run', admission=self.admitted,
                        runtime=lambda: self.fail('conflicted run redispatched'), sampling=self.sampling)

    def test_unknown_hand_fine_gesture_rejected_by_original_filter(self):
        r = self.manifest['requests'][0]
        refs = [f['index'] for f in r['manifest']['frames']][:2]
        value = dict(events=[dict(id='e1', description='The dancer makes a V-sign with the left hand.',
                    parts=['left_hand'], frame_indices=refs)],
                    part_support=[dict(part=part, frame_indices=[], reason='unknown') for part in self.core.PARTS],
                    uncertainties=['fingers indistinguishable'], abstain=False)
        checked = self.core.filter_supported_observation(value, r['manifest'], r['forbidden'])
        self.assertTrue(checked['abstain'])
        self.assertTrue(checked['support_filter_audit']['rejected_events'])

    def test_independent_32b_pixel_plan_not_self_review_or_text_agreement(self):
        _, receipt = self.run_fake()
        with self.assertRaisesRegex(ValueError, 'same 8B'):
            p.quality_audit_plan(self.prepared, receipt, reviewer_identity=self.manifest['model_identity'])
        teacher = self.manifest['cases'][0]['teacher_model_identity']
        plan = p.quality_audit_plan(self.prepared, receipt, reviewer_identity=teacher)
        self.assertEqual(len(plan['tasks']), 3)
        self.assertEqual(sum(len(t['original_windows']) for t in plan['tasks']), 8)
        self.assertEqual(plan['budget']['independent_32B_review_window_calls'], 8)
        self.assertFalse(plan['quality_approved'])
        self.assertTrue(plan['label_text_agreement_is_not_semantic_validation'])
        self.assertTrue(all(t['native_pixel_decode_and_review_pending'] for t in plan['tasks']))


if __name__ == '__main__':
    run = unittest.main(verbosity=2, exit=False)
    evidence = os.environ.get('CAPTION_8B_PILOT_EVIDENCE_DIR')
    if evidence:
        p.write_once(Path(evidence) / 'cpu_tests.json', dict(tests_run=run.result.testsRun,
            passed=run.result.wasSuccessful(), failures=len(run.result.failures), errors=len(run.result.errors),
            test_code_ref=p.Reader().read(Path(__file__))[1],
            pilot_code_ref=p.Reader().read(RT / 'caption_8b_primary_pilot_v1.py')[1],
            fixed_source_manifest_ref=p.Reader().read(Path(os.environ['CAPTION_8B_PILOT_MANIFEST']))[1],
            no_actual_GPU_model_load=True, no_production_files_changed=True))
    raise SystemExit(0 if run.result.wasSuccessful() else 1)
