"""CPU-only fixed3 native transactions; no capture/start/signal/model allocation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
import caption_motion_grounded_manager_v1 as m


def config():
    mm = NS(mm_processor_kwargs={'max_pixels': 200704}, get_limit_per_prompt=lambda k: {'image': 16, 'video': 0}[k])
    return NS(model_config=NS(model=m.MODEL, dtype='torch.bfloat16', quantization=None, seed=0,
        max_model_len=16384, trust_remote_code=False, multimodal_config=mm),
        parallel_config=NS(tensor_parallel_size=2), scheduler_config=NS(max_num_seqs=8),
        structured_outputs_config=NS(backend='xgrammar', disable_any_whitespace=True),
        cache_config=NS(enable_prefix_caching=True, gpu_memory_utilization=.88))


class FakeLLM:
    def __init__(self, entries, bad=None):
        self.entries, self.bad, self.calls = entries, bad, []
        self.vllm_config = config()
        self._grounded_runtime_version = 'CPU_fake_not_actual_GPU'
    def get_tokenizer(self):
        return NS(eos_token_id=2)
    def chat(self, messages, sampling, use_tqdm=False):
        assert (m.OUT / 'claim.json').exists()
        self.calls.append(copy.deepcopy(messages))
        if self.bad == 'exception':
            raise RuntimeError('fake dispatch failed')
        outputs = []
        bar, _ = m.pilot.direct.native()
        for e in self.entries:
            indices = [f['index'] for f in e['record']['manifest']['frames']]
            refs = [i for i in indices if i not in e['record']['forbidden'].get('torso', [])][:2]
            value = dict(events=[dict(id='e1', description='The torso bends sideways and rebounds.',
                parts=['torso'], frame_indices=refs)],
                part_support=[dict(part=p, frame_indices=refs if p == 'torso' else [],
                                   reason='visible' if p == 'torso' else 'unknown') for p in bar.PARTS],
                uncertainties=['CPU fake only, not pixel quality approval'], abstain=False)
            if self.bad == 'structure':
                value['part_support'].pop()
            outputs.append(NS(prompt_token_ids=[9, 10], outputs=[NS(text=json.dumps(value),
                token_ids=[1, 2], finish_reason='stop', stop_reason=None)]))
        return outputs[:-1] if self.bad == 'cardinality' else outputs


class GroundedManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_ref, cls.style_ref = m.ref(m.MANIFEST), m.ref(m.STYLE_REPORT)
        cls.entries, cls.styles, cls.refs = m.inputs(cls.manifest_ref, cls.style_ref)
        cls.bar, cls.pipeline = m.pilot.direct.native()
        cls.model = dict(root=m.MODEL, sha256=m.MODEL_SHA, files={}, CPU_fake=True)

    def test_actual_loaded_state_not_marker_and_original_lifecycle_identity(self):
        m.frozen()
        self.assertEqual(Path(m.base.__file__).name, 'resource_manager.py')
        self.assertEqual(m.parent_gate.__code__, m.helpers.parent_gate.__code__)
        self.assertEqual(m.parent_gate.__globals__['SELF'], m.SELF)
        self.assertTrue(m.actual_state(NS(vllm_config=config()))['verified_actual_vllm_config'])
        with self.assertRaises(AttributeError):
            m.actual_state(NS(model_load_completed=True, _ms_bar_model_sha256=m.MODEL_SHA))
        cfg = config();cfg.model_config.multimodal_config.mm_processor_kwargs = {}
        with self.assertRaises(ValueError):
            m.actual_state(NS(vllm_config=cfg))
        self.assertNotIn("startswith('0.21", Path(m.__file__).read_text())

    def test_same_original_pixels_eight_native_schema_prompt_delta_and_zero_aux_calls(self):
        self.assertEqual(len(self.entries), 8)
        for e in self.entries:
            old = m.checked(e['record']['refs']['request.json']);new = e['request']
            for key in ('sampling', 'model', 'json_schema'):
                self.assertEqual(old[key], new[key])
            self.assertEqual(new['messages'][1], old['messages'][1])
            self.assertEqual(new['messages'][0]['content'], old['messages'][0]['content'] + m.grounded.CONTRACT)
            self.assertEqual(e['body_image_count'], 8)
            self.assertEqual(e['image_count'], e['body_image_count'] + e['crop_image_count'])
            self.assertEqual(new['response_protocol_recipe'], self.bar.OUTPUT_SCHEMA)
            self.assertEqual(new['fact_source_recipe'], m.RECIPE)

    def run_fake(self, root, bad=None):
        with patch.object(m, 'OUT', root):
            entries, _, _ = m.inputs(self.manifest_ref, self.style_ref)
            llm = FakeLLM(entries, bad)
            with patch.object(self.bar.core, 'structured_sampling', side_effect=lambda *a: object()):
                report = m.run_fixed(self.manifest_ref, self.style_ref, self.model,
                    loader=lambda: (llm, object), admission=lambda: True)
            return llm, report

    def test_fake_eight_native_generations_exact_facts_newidentity_and_no_aux_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'trial'
            llm, report = self.run_fake(root)
            self.assertEqual([len(x) for x in llm.calls], [8])
            self.assertEqual(report['window_count'], report['attempted_window_calls'])
            self.assertEqual(report['source_count'], 3)
            for key in ('boundary_model_calls', 'rewrite_model_calls', 'new_style_model_calls', 'production_counts'):
                self.assertEqual(report[key], 0)
            self.assertFalse(report['quality_approved'])
            self.assertEqual(len(list(root.glob('sources/*/windows/*/attempt-0/generation.json'))), 8)
            for r in report['generation_records']:
                g = m.checked(r['generation_ref'])
                self.assertTrue(g['model_call_performed'])
                self.assertEqual(g['generation_recipe'], m.RECIPE)
                self.assertEqual(g['fact_source_recipe'], m.RECIPE)
            for result in report['results']:
                self.assertTrue(result['all_windows_nonempty'])
                self.assertTrue(result['target_text'].startswith('Dance style: '))
                for fact in result['facts']:
                    self.assertEqual(fact['raw_text'], 'The torso bends sideways and rebounds.')
                    self.assertEqual(fact['source'], m.RECIPE)
                self.assertEqual(len(result['uncertainties']), len(result['windows']))
            with patch.object(m, 'OUT', root), self.assertRaisesRegex(ValueError, 'Old output'):
                m.run_fixed(self.manifest_ref, self.style_ref, self.model, loader=lambda: self.fail('redispatch'), admission=lambda: True)

    def test_default_denied_oldout_badoutput_consumed_budget_never_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'denied'
            with patch.object(m, 'OUT', root), self.assertRaisesRegex(ValueError, 'default denied'):
                m.run_fixed(self.manifest_ref, self.style_ref, self.model)
            self.assertFalse(root.exists())
            for bad in ('structure', 'exception', 'cardinality'):
                root = Path(tmp) / bad
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'budget consumed'):
                    self.run_fake(root, bad)
                self.assertEqual(json.loads((root / 'claim.json').read_bytes())['reserved_window_calls'], 8)
                with patch.object(m, 'OUT', root), self.assertRaisesRegex(ValueError, 'Old output'):
                    m.run_fixed(self.manifest_ref, self.style_ref, self.model, loader=lambda: self.fail('retry'), admission=lambda: True)

    def test_wrong_source_mapping_and_native_frame_digest_rejected(self):
        manifest = m.checked(self.manifest_ref)
        real = m.checked
        for kind in ('order', 'source', 'frame'):
            bad = copy.deepcopy(manifest)
            if kind == 'order': bad['requests'][0]['key'] = 'wrong:w00000'
            elif kind == 'source': bad['requests'][0]['source_id'] = m.pilot.SOURCES[1]
            else: bad['requests'][0]['manifest']['frames'][0]['index'] += 1
            with self.subTest(kind=kind), patch.object(m, 'checked', side_effect=lambda r: bad if r == self.manifest_ref else real(r)), self.assertRaises(ValueError):
                m.inputs(self.manifest_ref, self.style_ref)

    def test_zero_preload_safe_binding_and_cpu_site_build_config(self):
        proofs = {str(g): m.ref(m.BAR / 'queues' / m.PRIOR[g][0] / 'resources_safe.json') for g in m.GPUS}
        real_checked = m.checked
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);prior_out = root / 'prior_empty';prior_out.mkdir()
            def checked(reference):
                value = real_checked(reference)
                if Path(reference['path']) == m.PRIOR_CONFIG:
                    value.update(out=str(prior_out), pilot_out=str(root / 'old_absent_pilot'))
                elif Path(reference['path']).name == 'admission.json':
                    value['out'] = str(prior_out)
                return value
            with patch.object(m, 'checked', side_effect=checked), patch.object(m, 'OUT', root / 'trial'):
                holders = [dict(m.restored(g, proofs[str(g)])['holder_identity'], gpu=g,
                                uuid=m.restored(g, proofs[str(g)])['uuid']) for g in m.GPUS]
                capture = dict(schema='visual_bench_native_capture_v1', node='N6', gpus=[2, 3], mode='consumer',
                    captured_unix=m.time.time(), physical_addresses=[m.helpers.HOST], holders=holders,
                    headroom={'headroom': m.base.SAFETY + m.base.LOAD}, signals_sent=False,
                    lock_inventory={str(g): {'held': []} for g in m.GPUS},
                    lease_inventory={str(g): {'exists': False} for g in m.GPUS}, predecessor_safe_refs=proofs)
                self.assertEqual(m.validate_capture(capture), capture)
                old = copy.deepcopy(capture)
                old['predecessor_safe_refs']['2'] = m.ref(m.BAR / 'queues/caption_batch8_direct_full_n6_main_v3/resources_safe.json')
                with self.assertRaisesRegex(ValueError, 'Exact zero-preload'):
                    m.validate_capture(old)
                old = copy.deepcopy(capture);old['holders'][1]['pid'] += 1
                with self.assertRaisesRegex(ValueError, 'actual restored'):
                    m.validate_capture(old)
                cap = self.pipeline.save(root / 'capture.json', capture)
                args = NS(capture=cap, capture_sha=m.base.sha(cap), modelref=Path(m.checked(self.manifest_ref)['model_ref']['path']),
                    modelref_sha=m.checked(self.manifest_ref)['model_ref']['sha256'], config_out=root / 'config.json',
                    execution_root=root / 'execution', queue=m.BAR / 'queues/grounded_CPU_never_started', phase='grounded_CPU')
                with patch.object(m, 'EXECUTION', args.execution_root):
                    cfg = m.checked(m.build_config(args))
                    self.assertEqual(cfg['resource_manager_ref'], m.ref(m.SELF))
                    self.assertEqual(cfg['source_sha256'][str(m.RT / 'resource_manager.py')], m.BASE_SHA)
                    self.assertEqual(cfg['env']['MS_VISUAL_CPU_SITEPATH'], m.CPU_SITEPATH)
                    self.assertEqual(cfg['argv'], [m.PYTHON, '-B', str(m.SELF), 'worker', '--config-file', str(args.config_out)])
                    self.assertFalse(args.execution_root.exists())
                    self.assertFalse((root / 'trial').exists())
                self.assertIn(m.CPU_SITEPATH, sys.path)
                # The old execution must be truly empty, not merely missing a model marker.
                (prior_out / 'generation.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Only zero-preload'):
                    m.restored(2, proofs['2'])
            def wrong_config(reference):
                value = checked(reference)
                if Path(reference['path']).name == 'admission.json':
                    value['config_sha256'] = 'wrong'
                return value
            with patch.object(m, 'checked', side_effect=wrong_config), self.assertRaisesRegex(ValueError, 'failure/safe binding'):
                m.restored(2, proofs['2'])

    def test_runtime_preconsumes_eight_and_denies_second_or_wrong_message(self):
        llm = FakeLLM(self.entries)
        runtime = m.Runtime(llm, lambda: True, self.entries)
        messages = runtime.expected
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'OUT', Path(tmp)):
            (Path(tmp) / 'claim.json').write_text('{}')
            runtime.chat(messages, [])
            self.assertEqual(runtime.calls, 8)
            with self.assertRaisesRegex(ValueError, 'retry'):
                runtime.chat(messages, [])
        runtime = m.Runtime(llm, lambda: True, self.entries)
        with self.assertRaisesRegex(ValueError, 'reserved'):
            runtime.chat([[dict(role='system', content='boundary')]], [])
        self.assertEqual(runtime.calls, 0)


if __name__ == '__main__':
    unittest.main()
