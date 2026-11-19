"""Finite manager CPU deltas; native media/chat fixtures, no resource actions."""
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
import caption_batch8_production_manager_v1 as manager
from test_caption_batch8_production_consumer_v1 import ProductionFixture, FakeLLM, bar, pipeline


def actual_config():
    mm = NS(mm_processor_kwargs={'max_pixels': 200704}, get_limit_per_prompt=lambda kind: {'image': 16, 'video': 0}[kind])
    return NS(model_config=NS(model=manager.MODEL, dtype='torch.bfloat16', quantization=None,
        max_model_len=16384, seed=0, trust_remote_code=False, multimodal_config=mm),
        scheduler_config=NS(max_num_seqs=8), parallel_config=NS(tensor_parallel_size=2),
        structured_outputs_config=NS(backend='xgrammar', disable_any_whitespace=True),
        cache_config=NS(enable_prefix_caching=True, gpu_memory_utilization=.88))


class ManagerTests(unittest.TestCase):
    def test_base_file_and_actual_parent_helper_retained_no_version_string_gate(self):
        manager.frozen()
        self.assertEqual(Path(manager.base.__file__).resolve(), RT / 'resource_manager.py')
        self.assertEqual(manager.parent_gate.__code__, manager.full.parent_gate.__code__)
        self.assertEqual(manager.parent_gate.__globals__['SELF'], manager.SELF)
        self.assertTrue(manager.actual_state(NS(vllm_config=actual_config()))['verified_actual_vllm_config'])
        wrong = actual_config()
        wrong.model_config.multimodal_config.mm_processor_kwargs = {}
        with self.assertRaises(ValueError):
            manager.actual_state(NS(vllm_config=wrong))

    def test_only_actual_full_success_safe_and_restored_pair_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, holders = Path(tmp), []
            queue = root / 'queues' / manager.PREVIOUS
            producer = {'path': str(manager.full.__file__), 'sha256': manager.FULL_SHA}
            plan = {'path': '/cpu/plan', 'sha256': 'p'}
            old = {key: 'cpu' for key in manager.base.IDENTITY_KEYS}
            old.update(uid=0, argv=manager.base.RESTORE_ARGV)
            for gpu in manager.GPUS:
                identity = dict(old, pid=gpu * 10, start_ticks=gpu)
                holders.append(dict(gpu=gpu, uuid='cpu-%d' % gpu, identity=identity))
            admission = dict(node='N6', gpus=manager.GPUS, phase=manager.PREVIOUS, manager_sha256=manager.BASE_SHA,
                model_identity_sha256=manager.MODEL_SHA, holders=holders, producer=producer, plan=plan)
            pipeline.save(queue / 'admission.json', admission)
            pipeline.save(queue / 'worker_exit.json', dict(phase=manager.PREVIOUS, exit_code=0, source_unchanged=True,
                summaries={'cpu': {}}, producer=producer, plan=plan))
            for h in holders:
                pipeline.save(queue / ('holder_restored_gpu%d.json' % h['gpu']), dict(gpu=h['gpu'], uuid=h['uuid'],
                    managed_parent_wait=True, restore_argv=manager.base.RESTORE_ARGV, original_identity=h['identity'],
                    holder_identity=dict(h['identity'], pid=h['identity']['pid'] + 1)))
            safe = pipeline.save(queue / 'resources_safe.json', dict(phase=manager.PREVIOUS, all_holders_safe=True,
                locks_released=True, leases_restored=True, managed_parent_wait=True, unknown_jobs_not_signaled=True))
            with patch.object(manager, 'BAR', root):
                for gpu in manager.GPUS:
                    self.assertEqual(manager.restored(gpu, pipeline.ref(safe))['gpu'], gpu)
                with self.assertRaisesRegex(ValueError, 'main_v3 predecessor'):
                    manager.restored(2, {'path': str(root / 'queues/old/resources_safe.json'), 'sha256': 'bad'})
                exit_path = queue / 'worker_exit.json'
                failed = pipeline.load(exit_path)
                failed['exit_code'] = 1
                exit_path.write_bytes(pipeline.raw_json(failed))
                with self.assertRaisesRegex(ValueError, 'did not complete'):
                    manager.restored(2, pipeline.ref(safe))

    def test_missing_handoff_refusal_before_any_parser_or_resource_load(self):
        with patch.object(manager, 'consumer_args', side_effect=AssertionError('must not parse')):
            with self.assertRaisesRegex(ValueError, 'handoff artifact'):
                manager.handoff({})

    def test_handoff_actual_native_schema_and_cleanup_exclusion_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=2)
            cfg = dict(handoff_ref=f.args.handoff_ref)
            with patch.object(manager, 'consumer_args', return_value=f.args):
                selected = manager.selected(cfg)
                self.assertEqual(selected, ([row['id'] for row in f.entries], 6))
                path = Path(f.args.handoff_ref['path'])
                value = pipeline.load(path)
                value['exclusive_source_owner'] = False
                path.write_bytes(pipeline.raw_json(value))
                f.args.handoff_ref = pipeline.ref(path)
                cfg['handoff_ref'] = f.args.handoff_ref
                with self.assertRaisesRegex(ValueError, 'Exclusive native handoff'):
                    manager.selected(cfg)

    def test_build_cli_preserves_native_options_after_separator_and_json_output(self):
        from io import StringIO
        argv = ['manager', 'build', '--capture', '/cpu/cap', '--capture-sha', 'cap', '--modelref', '/cpu/model',
                '--modelref-sha', 'model', '--handoff-ref', '/cpu/handoff', '--handoff-sha', 'handoff',
                '--config-out', '/cpu/config', '--execution-root', '/cpu/execution', '--queue', '/cpu/queue', '--phase', 'cpu',
                '--', '--out', '/cpu/consumer', '--member', 'n4c3', '--handoff', '/cpu/handoff', '--handoff-sha', 'handoff']
        stream = StringIO()
        with patch.object(sys, 'argv', argv), patch.object(manager, 'build_config', return_value={'path': '/cpu/config', 'sha256': 'cpu'}) as build, \
                patch.object(sys, 'stdout', stream):
            manager.main()
        self.assertEqual(build.call_args.args[0].consumer_args, argv[argv.index('--'):])
        self.assertIn('"sha256": "cpu"', stream.getvalue())

    def test_new_capture_refuses_existing_output_before_any_resource_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'capture.json'
            path.write_bytes(b'original caller capture')
            with patch.object(manager, '_capture', side_effect=AssertionError('no resources')):
                with self.assertRaisesRegex(ValueError, 'Fresh capture'):
                    manager.capture_current(path)
            self.assertEqual(path.read_bytes(), b'original caller capture')

    def long_fixture(self, tmp):
        f = ProductionFixture(tmp, groups=2)
        first = f.entries[0]
        windows = [dict(start=a, end=b, raw_window_indices=[i]) for i, (a, b) in enumerate(
            ((0., .25), (.25, .5), (.5, .75), (.75, 1.), (1., 1.5)))]
        first['windows'] = windows
        first['window_manifest']['windows'] = copy.deepcopy(windows)
        source = pipeline.source_dir(f.original, first['manifest_record'])
        plan_path = source / 'bar_plan.json'
        plan_path.write_bytes(pipeline.raw_json(first))
        terminal_path = source / 'producer.terminal.json'
        producer = pipeline.load(terminal_path)
        producer['plan_ref'] = pipeline.ref(plan_path)
        terminal_path.write_bytes(pipeline.raw_json(producer))
        queue = f.original / 'ready' / ('producer-%04d.jsonl' % (first['manifest_record']['source_ordinal'] % 4))
        hints = [bar.core.strict_json(line) for line in queue.read_bytes().splitlines()]
        hints = [producer if value['id'] == first['id'] else value for value in hints]
        queue.write_bytes(b''.join(pipeline.raw_json(value) for value in hints))
        return f

    def test_worker_six_native_sources_whole_long_source_batch8_tail_same_recipe_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.long_fixture(tmp)
            execution = Path(tmp) / 'execution'
            cfg = dict(out=str(execution), config_path=str(Path(tmp) / 'config.json'),
                plan=f.args.handoff_ref, handoff_ref=f.args.handoff_ref, consumer_argv=[], modelref={'path': '/cpu/modelref'})
            pipeline.save(cfg['config_path'], cfg)
            loads, saved, gates = [], [], []
            class Loaded(FakeLLM):
                def __init__(self, **kwargs):
                    super().__init__()
                    self.vllm_config = actual_config()
                    self.fresh_out = not f.out.exists()
                    loads.append(self)
            model = dict(root=manager.MODEL, sha256=manager.MODEL_SHA, files={}, cpu_fake=True)
            original_save = bar.core.save_json
            def counted_save(path, value):
                if Path(path) == f.out / 'visual/model.identity.json':
                    saved.append(path)
                    self.assertTrue((f.out / 'protocol.json').exists())
                return original_save(path, value)
            ids = [r['id'] for r in f.entries]
            with f.context(), patch.object(manager, 'validate', return_value=None), \
                    patch.object(manager, 'parent_gate', side_effect=lambda *a, **kw: gates.append(kw) or {'cpu_owned': True}), \
                    patch.object(manager, 'consumer_args', return_value=f.args), \
                    patch.object(bar, 'model_reference', return_value=model), \
                    patch.object(bar.core, 'save_json', counted_save):
                report = manager.worker(cfg['config_path'], factory=(Loaded, object))
                summaries = manager.summaries(cfg, execution)
                before = (f.out / 'completed.jsonl').read_bytes()
                cfg2 = dict(cfg, out=str(Path(tmp) / 'new-execution'), config_path=str(Path(tmp) / 'new-config.json'))
                pipeline.save(cfg2['config_path'], cfg2)
                second = manager.worker(cfg2['config_path'], factory=(Loaded, object))
                self.assertEqual(second['terminal_count'], 6)
                self.assertEqual((f.out / 'completed.jsonl').read_bytes(), before)
                f.args.visual_max_tokens += 1
                cfg3 = dict(cfg, out=str(Path(tmp) / 'third-execution'), config_path=str(Path(tmp) / 'third-config.json'))
                pipeline.save(cfg3['config_path'], cfg3)
                with self.assertRaisesRegex(ValueError, 'owned resume'):
                    manager.worker(cfg3['config_path'], factory=(Loaded, object))
            self.assertEqual([len(c) for c in loads[0].calls], [8, 2, 8])
            self.assertEqual(loads[1].calls, [])
            self.assertEqual(loads[2].calls, [])
            self.assertTrue(loads[0].fresh_out)
            self.assertEqual(len(saved), 1)
            self.assertEqual(report['selected_source_count'], 6)
            self.assertEqual(report['terminal_count'], 6)
            self.assertEqual(report['boundary_model_calls'], 0)
            self.assertEqual(report['rewrite_model_calls'], 0)
            self.assertEqual(next(iter(summaries.values()))['summary']['total_source_count'], 6)
            self.assertGreaterEqual(len(gates), 7)

    def test_native_manager_main_not_wrapped_resource_lifecycle(self):
        with patch.object(manager.base, 'main') as main:
            manager.manager()
            main.assert_called_once_with()
        self.assertEqual(Path(manager.base.__file__).name, 'resource_manager.py')
        self.assertIs(manager.base.stage_summaries, manager.summaries)


if __name__ == '__main__':
    unittest.main()
