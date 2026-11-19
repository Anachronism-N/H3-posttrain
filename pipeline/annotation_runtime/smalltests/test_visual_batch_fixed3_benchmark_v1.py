"""CPU-only fake native LLM: exact messages/schema/budget, locked one-shot journals."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('visual_bench', RT / 'visual_batch_fixed3_benchmark_v1.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class FakeSampling:
    def __init__(self, **kwargs):
        self.values = kwargs


class FakeStructured:
    def __init__(self, json):
        self.json = json


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.truncate = False
        self.drop = False

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=7)

    def chat(self, messages, sampling, use_tqdm):
        self.calls.append((copy.deepcopy(messages), sampling))
        outputs = []
        for index, (message, sample) in enumerate(zip(messages, sampling)):
            payload = json.loads(message[1]['content'][0]['text'])
            frames = payload['frames']
            value = dict(events=[dict(id='e1', description='The dancer raises the left arm.', parts=['left_arm'], frame_indices=frames[:2])],
                part_support=[dict(part=p, frame_indices=frames, reason='visible') for p in b.pure_bar().PARTS], uncertainties=[], abstain=False)
            outputs.append(SimpleNamespace(request_id=str(index), prompt_token_ids=[1] * 12,
                outputs=[SimpleNamespace(text=json.dumps(value), token_ids=[1] * (1800 if self.truncate else 20),
                    finish_reason='length' if self.truncate else 'stop', stop_reason=None)]))
        return outputs[:-1] if self.drop else outputs


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bar_patch = patch.object(b, 'BAR', self.root)
        self.bar_patch.start()
        self.out = self.root / 'visual_batch_fixed3_benchmark_test'
        self.out.mkdir()
        self.origin = self.root / 'original'
        self.origin.mkdir()
        self.bar = b.pure_bar()
        self.boundary = b.Boundary()
        self.cases, self.items = [], []
        for sid, count in zip(b.SOURCES, (2, 3, 3)):
            windows = []
            for index in range(count):
                wid = 'w%05d' % index
                frames = [dict(index=f, time=float(f) / 8) for f in range(8)]
                manifest = dict(window=dict(window_id=wid, start=0.0, end=1.0), frames=frames)
                windows.append(manifest)
                request = dict(model='/opt/models/Qwen3-VL-32B-Instruct',
                    messages=[dict(role='system', content=self.bar.SYSTEM), dict(role='user', content=[
                        dict(type='text', text=json.dumps(dict(source_id=sid, window_id=wid, frames=list(range(8))))),
                        dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,AA=='))])],
                    sampling=dict(temperature=0.0, seed=0, max_tokens=1800), json_schema=self.bar.observation_schema(list(range(8))))
                refs = {}
                for name, value in [('request.json', request), ('binding.json', dict(source_binding=dict(id=sid), window=manifest['window']))]:
                    path = self.origin / (sid + '_' + wid + '_' + name)
                    path.write_bytes(b.encoded(value))
                    refs[name] = self.boundary.file(path)
                self.items.append(dict(key=sid + ':' + wid, source_id=sid, window_id=wid, refs=refs,
                                       manifest=manifest, forbidden={}))
            prepared = self.origin / (sid + '_prepared.json')
            prepared.write_bytes(b.encoded(dict(windows=windows, forbidden={})))
            self.cases.append(dict(source_id=sid, window_count=count, prepared_ref=self.boundary.file(prepared)))
        self.budget = self.origin / 'tasksource_retrybudget.json'
        self.budget.write_bytes(b'{"used":1}\n')
        self.boundary.file(self.budget)
        self.manifest = dict(schema=b.SCHEMA, output=str(self.out), scope='visual only', cases=self.cases, requests=self.items,
            request_count=8, planned_model_request_count=16, total_completion_token_budget=28800,
            variants=[dict(name='A', batch_size=2), dict(name='B', batch_size=8)], expected_runtime_state=b.EXPECTED_STATE,
            original_boundary=list(self.boundary.refs.values()), **b.FLAGS)
        self.manifest_path = self.out / 'manifest.json'
        self.manifest_path.write_bytes(b.encoded(self.manifest))
        self.fake_vllm = ModuleType('vllm.sampling_params')
        self.fake_vllm.StructuredOutputsParams = FakeStructured
        self.modules = patch.dict(sys.modules, {'vllm.sampling_params': self.fake_vllm})
        self.modules.start()
        self.llm = FakeLLM()
        self.runtime = b.CallerOwnedRuntime(self.llm, FakeSampling, lambda llm: dict(b.EXPECTED_STATE),
            lambda llm: {'cpu_fake': True, 'gpu_measurement_available': False})
        self.admission = lambda manifest, runtime: dict(admitted=True, caller_owned=True, fake_CPU_only=True)

    def tearDown(self):
        self.modules.stop()
        self.bar_patch.stop()
        self.temp.cleanup()

    def run_bench(self):
        return b.run_benchmark(self.manifest_path, self.runtime, self.admission)

    def test_batch_exact_messages_schema_tokens_mapping_and_budget(self):
        original_budget = self.budget.read_bytes()
        result = self.run_bench()
        self.assertEqual([len(messages) for messages, _ in self.llm.calls], [2, 2, 2, 2, 8])
        expected_requests = [b.strict(Path(i['refs']['request.json']['path']).read_bytes()) for i in self.items]
        flattened = [(m, s.values) for messages, samples in self.llm.calls for m, s in zip(messages, samples)]
        for (messages, sampling), request in zip(flattened, expected_requests * 2):
            self.assertEqual(messages, request['messages'])
            self.assertEqual({k: sampling[k] for k in ('temperature', 'seed', 'max_tokens')}, request['sampling'])
            self.assertEqual(sampling['structured_outputs'].json, request['json_schema'])
        self.assertEqual(result['actual_model_request_count'], 16)
        self.assertTrue(result['original_boundary_unchanged'])
        self.assertEqual(self.budget.read_bytes(), original_budget)
        self.assertEqual(result['production_counts'], 0)
        for variant in ('A', 'B'):
            data = result['variants'][variant]
            self.assertEqual([c['window_count'] for c in data['cases']], [2, 3, 3])
            self.assertTrue(all(c['all_window_full_fact_coverage'] for c in data['cases']))
            self.assertEqual(data['completion_tokens'], 160)
            self.assertEqual([r['key'] for r in data['records']], [i['key'] for i in self.items])
        with self.assertRaisesRegex(ValueError, 'already started'):
            self.run_bench()
        self.assertEqual(len(self.llm.calls), 5)

    def test_missing_runtime_and_default_admission_do_not_dispatch(self):
        with self.assertRaisesRegex(ValueError, 'caller-owned'):
            b.run_benchmark(self.manifest_path)
        with self.assertRaises(PermissionError):
            b.run_benchmark(self.manifest_path, self.runtime)
        self.assertEqual(self.llm.calls, [])
        self.assertFalse((self.out / 'started.json').exists())

    def test_live_runtime_configuration_mismatch_denied(self):
        self.runtime.actual_state = lambda llm: {**b.EXPECTED_STATE, 'max_num_seqs': 2}
        with self.assertRaisesRegex(ValueError, 'live native'):
            self.run_bench()
        self.assertEqual(self.llm.calls, [])

    def test_duplicate_request_scope_denied_before_call(self):
        self.manifest['requests'][1] = self.manifest['requests'][0]
        self.manifest_path.write_bytes(b.encoded(self.manifest))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.run_bench()
        self.assertEqual(self.llm.calls, [])

    def test_wrong_response_source_does_not_resubmit(self):
        native_chat = self.runtime.chat
        def wrong(items):
            responses = native_chat(items)
            responses[0]['source_id'] = 'wrong-source'
            return responses
        self.runtime.chat = wrong
        with self.assertRaisesRegex(ValueError, 'mapping'):
            self.run_bench()
        self.assertEqual(len(self.llm.calls), 1)
        result = b.strict((self.out / 'result.json').read_bytes())
        self.assertEqual(result['status'], 'failed_no_resubmit')
        self.assertEqual(result['actual_model_request_count'], 2)
        with self.assertRaisesRegex(ValueError, 'already started'):
            self.run_bench()
        self.assertEqual(len(self.llm.calls), 1)

    def test_missing_output_never_resubmits_on_resume(self):
        self.llm.drop = True
        with self.assertRaisesRegex(ValueError, 'missing/extra'):
            self.run_bench()
        self.assertIn('dispatch_reserved', (self.out / 'calls.jsonl').read_text())
        self.assertEqual(len(self.llm.calls), 1)
        with self.assertRaisesRegex(ValueError, 'already started'):
            self.run_bench()
        self.assertEqual(len(self.llm.calls), 1)

    def test_token_truncation_never_resubmits(self):
        self.llm.truncate = True
        with self.assertRaisesRegex(ValueError, 'truncated'):
            self.run_bench()
        self.assertEqual(len(self.llm.calls), 1)

    def test_exit_original_budget_mutation_fails_closed(self):
        native_chat = self.runtime.chat
        def mutate(items):
            responses = native_chat(items)
            self.budget.write_bytes(b'{"used":2}\n')
            return responses
        self.runtime.chat = mutate
        with self.assertRaisesRegex(ValueError, 'boundary changed'):
            self.run_bench()
        result = b.strict((self.out / 'result.json').read_bytes())
        self.assertEqual(result['status'], 'failed_original_boundary_changed')
        self.assertFalse(result['original_boundary_unchanged'])

    def test_filter_and_schema_remain_original_not_text_equality(self):
        original = self.bar.filter_supported_observation
        seen = []
        def checked(*args):
            seen.append(args)
            return original(*args)
        with patch.object(b, 'pure_bar', return_value=SimpleNamespace(**{**vars(self.bar), 'filter_supported_observation': checked})):
            self.run_bench()
        self.assertEqual(len(seen), 16)

    def test_schema_rejects_extra_keys_bounds_and_duplicate_json(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            b.strict('{"events":[],"events":[]}')
        with self.assertRaises(ValueError):
            b.schema_check('x' * 601, dict(type='string', maxLength=600))
        with self.assertRaises(ValueError):
            b.schema_check({'bad': 1}, dict(type='object', properties={}, required=[], additionalProperties=False))


if __name__ == '__main__':
    unittest.main()
