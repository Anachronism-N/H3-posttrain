"""CPU contracts for the diagnosed zero-dispatch configuration-read recovery."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location('bench_v2', RT/'bench_manager_driver_v2.py')
d = importlib.util.module_from_spec(s)
s.loader.exec_module(d)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root/'visual_batch_fixed3_benchmark_v1'
        self.out.mkdir()
        self.queue = self.root/'queues/visual_batch_fixed3_n6g23_main_v1'
        self.queue.mkdir(parents=True)
        self.write(self.queue/'resources_safe.json', dict(phase='visual_batch_fixed3_n6g23_main_v1',
            all_holders_safe=True, locks_released=True, leases_restored=True))
        self.write(self.queue/'admission.json', dict(config_sha256='633b1ee462a9263acf0b763fe0a7b8610621af58b596cf779b1c38dcd89af670',
            node='N6', gpus=[2,3], manager_sha256=d.BASE_SHA))
        self.write(self.queue/'failure.json', dict(error='configuration read failed'))
        (self.queue/'worker.runtime.log').write_text("KeyError: 'max_pixels'\n")
        self.write(self.out/'driver.started.json', dict(legacy=True))
        self.write(self.out/'native_model_load.receipt.json', dict(model_load_completed=True, model_identity_sha256=d.bench.MODEL_SHA))
        self.write(self.out/'original_budget_guard.exit.json', dict(unchanged=True))
        self.prior = dict(node='N6', gpus=[2,3], out=str(self.out), queue=str(self.queue), benchmark_manifest=dict(path='same-original-manifest'))
        self.cfg = dict(out=str(self.out), execution_root=str(self.out/'zero_dispatch_recovery_v2'),
            prior_zero_dispatch_ref=dict(path='prior-launch',sha256='633b1ee462a9263acf0b763fe0a7b8610621af58b596cf779b1c38dcd89af670'),
            benchmark_manifest=self.prior['benchmark_manifest'], resource_safe_refs=[d.ref(self.queue/'resources_safe.json')])
        self.real_checked = d.checked
        self.patches = [patch.object(d,'BAR',self.root), patch.object(d,'checked',side_effect=lambda ref:
            self.prior if ref['path']=='prior-launch' else self.real_checked(ref))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
    def write(self,path,value):
        path.write_bytes(d.bench.encoded(value))
    def test_exact_zero_dispatch_allowed_without_modifying_old_artifacts(self):
        before = {p.name:p.read_bytes() for p in self.out.iterdir()}
        d.verify_zero_dispatch(self.cfg)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.out.iterdir()})
        self.assertFalse(Path(self.cfg['execution_root']).exists())
    def test_any_reserved_call_or_started_benchmark_denied(self):
        for name in ('started.json','calls.jsonl','result.json','generations'):
            with self.subTest(name=name):
                path=self.out/name
                path.write_text('preserve')
                with self.assertRaisesRegex(ValueError,'dispatched|already started'):
                    d.verify_zero_dispatch(self.cfg)
                self.assertEqual(path.read_text(),'preserve')
                path.unlink()
    def test_other_failure_or_changed_budget_guard_denied(self):
        (self.queue/'worker.runtime.log').write_text('different failure')
        with self.assertRaisesRegex(ValueError,'diagnosed'):
            d.verify_zero_dispatch(self.cfg)
        (self.queue/'worker.runtime.log').write_text("KeyError: 'max_pixels'\n")
        self.write(self.out/'original_budget_guard.exit.json',dict(unchanged=False))
        with self.assertRaisesRegex(ValueError,'budget guard'):
            d.verify_zero_dispatch(self.cfg)
    def test_wrong_safe_phase_or_execution_root_denied(self):
        self.cfg['execution_root']=str(self.out/'other')
        with self.assertRaisesRegex(ValueError,'execution'):
            d.verify_zero_dispatch(self.cfg)
        self.cfg['execution_root']=str(self.out/'zero_dispatch_recovery_v2')
        safe=self.queue/'resources_safe.json'
        self.write(safe,dict(phase='wrong',all_holders_safe=True,locks_released=True,leases_restored=True))
        self.cfg['resource_safe_refs']=[d.ref(safe)]
        with self.assertRaisesRegex(ValueError,'not safely restored'):
            d.verify_zero_dispatch(self.cfg)
    def test_actual_new_vllm_multimodal_config_not_deprecated_model_field(self):
        model=SimpleNamespace(model='/opt/models/Qwen3-VL-32B-Instruct',dtype='torch.bfloat16',quantization=None,
            max_model_len=16384,seed=0,mm_processor_kwargs=None,
            multimodal_config=SimpleNamespace(mm_processor_kwargs={'max_pixels':200704},get_limit_per_prompt=lambda key:16))
        vc=SimpleNamespace(model_config=model,scheduler_config=SimpleNamespace(max_num_seqs=8),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            cache_config=SimpleNamespace(gpu_memory_utilization=.88,enable_prefix_caching=True),
            structured_outputs_config=SimpleNamespace(backend='xgrammar',disable_any_whitespace=True))
        llm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=vc),_ms_bar_model_sha256=d.bench.MODEL_SHA)
        load=dict(model_identity_sha256=d.bench.MODEL_SHA,source_reference_sha256=d.bench.MODEL_REF_SHA,model_load_completed=True)
        cfg=dict(modelref=dict(model_identity_sha256=d.bench.MODEL_SHA))
        with patch.object(d,'checked',return_value=load):
            self.assertEqual(d.actual_state(llm,cfg,{}),d.bench.EXPECTED_STATE)
            model.multimodal_config.mm_processor_kwargs={}
            with self.assertRaisesRegex(ValueError,'multimodal processor'):
                d.actual_state(llm,cfg,{})
    def test_load_failure_writes_only_fresh_execution_and_never_resubmits(self):
        cfgpath=self.root/'config.json'
        self.write(cfgpath,{**self.cfg, 'benchmark_manifest':dict(path=str(d.bench.DEFAULT_OUT/'manifest.json'),
            sha256=d.base.sha(d.bench.DEFAULT_OUT/'manifest.json')), 'original_retry_budget_paths':[]})
        before={p.name:p.read_bytes() for p in self.out.iterdir()}
        fake=ModuleType('vllm')
        calls=[]
        def load(**kwargs):
            calls.append(kwargs)
            raise MemoryError('CPU simulated load failure')
        fake.LLM=load
        fake.SamplingParams=object
        with patch.dict(sys.modules,{'vllm':fake}),patch.object(d,'validate'),patch.object(d,'parent_gate',return_value=dict(admitted=True,caller_owned=True)):
            with self.assertRaises(MemoryError):
                d.driver(cfgpath,d.bench.DEFAULT_OUT/'manifest.json')
            with self.assertRaises(FileExistsError):
                d.driver(cfgpath,d.bench.DEFAULT_OUT/'manifest.json')
        self.assertEqual(len(calls),1)
        self.assertTrue((Path(self.cfg['execution_root'])/'driver.started.json').exists())
        self.assertTrue((Path(self.cfg['execution_root'])/'original_budget_guard.exit.json').exists())
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.out.iterdir() if p.is_file()})
        self.assertFalse((self.out/'started.json').exists())
        self.assertFalse((self.out/'calls.jsonl').exists())

if __name__=='__main__':
    unittest.main()
