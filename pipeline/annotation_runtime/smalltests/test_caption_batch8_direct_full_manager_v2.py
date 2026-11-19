"""CPU-only native caller contracts; no capture/start/signal/GPU operations."""
import copy
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import caption_batch8_direct_full_manager_v2 as manager
from test_caption_batch8_direct_full_pilot_v1 import FullFixture, FakeLLM, bar, pipeline


def config():
    mm = NS(mm_processor_kwargs={'max_pixels': 200704}, get_limit_per_prompt=lambda kind: {'image': 16, 'video': 0}[kind])
    return NS(model_config=NS(model=manager.MODEL, dtype='torch.bfloat16', quantization=None,
        max_model_len=16384, seed=0, trust_remote_code=False, mm_processor_kwargs=None, multimodal_config=mm),
        scheduler_config=NS(max_num_seqs=8), parallel_config=NS(tensor_parallel_size=2),
        structured_outputs_config=NS(backend='xgrammar', disable_any_whitespace=True),
        cache_config=NS(enable_prefix_caching=True, gpu_memory_utilization=.88))


class ManagerTests(unittest.TestCase):
    def test_original_base_identity_and_only_native_helper_cloned(self):
        manager.frozen()
        self.assertEqual(Path(manager.base.__file__).resolve(), RT / 'resource_manager.py')
        self.assertEqual(manager.parent_gate.__code__, manager.helpers.parent_gate.__code__)
        self.assertEqual(manager.parent_gate.__globals__['SELF'], manager.SELF)
        self.assertEqual(manager.BASE_SHA, manager.base.sha(manager.base.__file__))

    def predecessor_fixture(self, base):
        holders, refs = [], {}
        for gpu in manager.GPUS:
            name, phase = manager.PRIOR[gpu]
            queue = base / 'queues' / name
            original = {key: 'cpu' for key in manager.base.IDENTITY_KEYS}
            original.update(pid=gpu * 10, start_ticks=gpu, uid=0, argv=manager.base.RESTORE_ARGV)
            current = dict(original, pid=gpu * 10 + 1, start_ticks=gpu + 100, gpu=gpu, uuid='cpu-uuid-%d' % gpu)
            producer, plan = {'path': '/cpu/producer', 'sha256': 'a'}, {'path': '/cpu/plan', 'sha256': 'b'}
            admission = dict(node='N6', gpus=[2, 3], phase=phase, manager_sha256=manager.BASE_SHA,
                model_identity_sha256=manager.MODEL_SHA, config_sha256=manager.FAILED_CONFIG_SHA, out=str(base/'failed-execution'),
                holders=[dict(gpu=gpu, uuid=current['uuid'], identity=original)], producer=producer, plan=plan)
            if not (queue/'admission.json').exists():
                pipeline.save(queue / 'admission.json', {**admission, 'holders': [dict(gpu=g, uuid='cpu-uuid-%d'%g, identity={**original, 'pid':g*10,'start_ticks':g}) for g in manager.GPUS]})
            if not (queue/'failure.json').exists():
                pipeline.save(queue / 'failure.json', dict(phase=phase,worker_started=True))
            (queue/'worker.runtime.log').write_text('Actual vLLM0.21 runtime required')
            pipeline.save(queue / ('holder_restored_gpu%d.json' % gpu), dict(gpu=gpu, uuid=current['uuid'],
                original_identity=original, holder_identity=current, managed_parent_wait=True, restore_argv=manager.base.RESTORE_ARGV))
            safe = pipeline.save(queue / 'resources_safe.json', dict(phase=phase, all_holders_safe=True,
                locks_released=True, leases_restored=True, managed_parent_wait=True, unknown_jobs_not_signaled=True))
            refs[str(gpu)] = pipeline.ref(safe)
            holders.append(current)
        return dict(schema='visual_bench_native_capture_v1', node='N6', gpus=[2, 3], mode='consumer',
            captured_unix=time.time(), physical_addresses=[manager.helpers.HOST], holders=holders, signals_sent=False,
            headroom={'headroom': manager.base.SAFETY + manager.base.LOAD},
            lock_inventory={str(g): {'held': []} for g in manager.GPUS},
            lease_inventory={str(g): {'exists': False} for g in manager.GPUS}, predecessor_safe_refs=refs)

    def test_distinct_gpu2_32b_gpu3_8b_safe_restore_identities_no_old_parent_exit_requirement(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(manager, 'BAR', Path(tmp)), patch.object(manager.pilot, 'DEFAULT_OUT', Path(tmp)/'fresh-pilot'):
            capture = self.predecessor_fixture(Path(tmp))
            self.assertEqual(manager.validate_capture(capture), capture)
            other = copy.deepcopy(capture)
            other['predecessor_safe_refs']['3'] = dict(other['predecessor_safe_refs']['2'],path=str(Path(tmp)/'wrong/resources_safe.json'))
            with self.assertRaisesRegex(ValueError, 'Wrong GPU predecessor'):
                manager.validate_capture(other)
            other = copy.deepcopy(capture)
            other['holders'][1]['pid'] += 1
            with self.assertRaisesRegex(ValueError, 'actual restored'):
                manager.validate_capture(other)
            self.assertFalse(list(Path(tmp).rglob('parent_exit.json')))

    def test_zero_call_recovery_refuses_existing_pilot_or_wrong_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(manager, 'BAR', Path(tmp)), patch.object(manager.pilot, 'DEFAULT_OUT', Path(tmp)/'pilot'):
            capture = self.predecessor_fixture(Path(tmp))
            proof = capture['predecessor_safe_refs']['2']
            (Path(tmp)/'pilot').mkdir()
            with self.assertRaisesRegex(ValueError, 'pre-model/pre-call'):
                manager.restored(2, proof)
            (Path(tmp)/'pilot').rmdir()
            (Path(proof['path']).parent/'worker.runtime.log').write_text('different failure')
            with self.assertRaisesRegex(ValueError, 'pre-model/pre-call'):
                manager.restored(2, proof)

    def test_actual_vllm_multimodal_config_not_static_marker(self):
        llm = NS(vllm_config=config())
        self.assertTrue(manager.actual_state(llm)['verified_actual_vllm_config'])
        llm.vllm_config.model_config.mm_processor_kwargs = {'max_pixels': 200704}
        llm.vllm_config.model_config.multimodal_config.mm_processor_kwargs = {}
        with self.assertRaisesRegex(ValueError, 'Actual loaded'):
            manager.actual_state(llm)
        llm = NS(model_load_completed=True)
        with self.assertRaises(AttributeError):
            manager.actual_state(llm)

    def test_only_native_visual_eight_call_budget(self):
        llm = NS(vllm_config=config(), chat=lambda *a, **kw: [])
        gates = []
        runtime = manager.Runtime(llm, lambda: gates.append(True))
        messages = [[{'content': bar.SYSTEM}]] * 8
        runtime.chat(messages, [])
        self.assertEqual(runtime.calls, 8)
        with self.assertRaisesRegex(ValueError, 'eight new'):
            runtime.chat(messages[:1], [])
        runtime = manager.Runtime(llm, lambda: None)
        with self.assertRaisesRegex(ValueError, 'no boundary'):
            runtime.chat([[{'content': 'boundary'}]], [])
        self.assertEqual(runtime.calls, 0)

    def test_worker_loads_execution_root_before_admission_and_native_full_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp)
            execution = Path(tmp) / 'execution'
            cfg = dict(out=str(execution), config_path=str(Path(tmp) / 'config.json'), modelref={'path': '/cpu/modelref'},
                       plan={'path': str(manager.pilot.__file__), 'sha256': manager.PILOT_SHA})
            pipeline.save(cfg['config_path'], cfg)
            loads, gates = [], []
            class Loaded(FakeLLM):
                def __init__(self, **kw):
                    super().__init__()
                    self.vllm_config = config()
                    loads.append(self)
                    self.assert_out_was_fresh = not f.out.exists()
                    self.assert_execution_started = (execution / 'driver.started.json').exists()
            identity = dict(root=manager.MODEL, sha256=manager.MODEL_SHA, files={}, cpu_fake=True)
            with f.context(), patch.object(manager, 'validate', return_value=None), \
                    patch.object(manager, 'parent_gate', side_effect=lambda *a, **kw: gates.append(kw) or {'cpu_owned': True}), \
                    patch.object(manager.pilot, 'pilot_args', return_value=f.args), \
                    patch.object(bar, 'model_reference', return_value=identity):
                fake = __import__('types').ModuleType('vllm')
                fake.__version__ = '0.31.0'
                fake.LLM, fake.SamplingParams = Loaded, object
                with patch.dict(sys.modules, {'vllm': fake}):
                    report = manager.worker(cfg['config_path'])
                summary = manager.summaries(cfg, execution)
            self.assertEqual(len(loads), 1)
            self.assertTrue(loads[0].assert_out_was_fresh)
            self.assertTrue(loads[0].assert_execution_started)
            self.assertEqual([len(c) for c in loads[0].calls], [8])
            self.assertEqual(report['visual_chat_batch_sizes'], [8])
            self.assertEqual(report['window_count'], 8)
            self.assertEqual(report['boundary_model_calls'], 0)
            self.assertEqual(report['rewrite_model_calls'], 0)
            self.assertGreaterEqual(len(gates), 5)
            self.assertTrue(all(r['joint_candidate']['annotator_model_sha256'] == manager.MODEL_SHA for r in report['results']))
            self.assertEqual(pipeline.load(execution / 'model.identity.json'), pipeline.load(f.out / 'model.identity.json'))
            self.assertEqual(next(iter(summary.values()))['actual_window_calls'], 8)
            self.assertFalse(list(f.original.rglob('generation.json')))

    def test_native_manager_uses_original_base_main_and_no_fixed_benchmark_gate(self):
        with patch.object(manager.base, 'main') as main:
            manager.manager()
            main.assert_called_once_with()
        self.assertIs(manager.base.stage_summaries, manager.summaries)
        self.assertIs(manager.base.selected_sources, manager.selected)
        self.assertEqual(Path(manager.base.__file__).name, 'resource_manager.py')


if __name__ == '__main__':
    unittest.main()
