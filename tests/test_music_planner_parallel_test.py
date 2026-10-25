"""CPU only: fake receipts/tensors, real frozen eval and resource helper functions.

These tests never query CUDA/nvidia-smi, load an official model, capture/release a
real holder, SSH, or prove real GPU quality/resource restoration.
"""
import copy
from contextlib import ExitStack
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'scripts_omnidance/exp/ms'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


w = load('_test_new_parallel_worker', SCRIPTS / 'music_planner_parallel_test.py')
m = load('_test_new_parallel_parent', SCRIPTS / 'music_planner_parallel_test_manager.py')
b = m.load_base()
resource, fixed, ev = b.load_helpers()
# Reuse the existing real-helper fake-weight receipt builder, not a second recipe.
old = load('_parallel_existing_CPU_receipts', REPO / 'tests/test_music_planner_eval_manager.py')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return dict(path=str(path), sha256=w.sha(path))


class ParallelCPU(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'outputs'
        self.out.mkdir()
        self.sources = {key: w.file_ref(SCRIPTS / 'music_planner_parallel_test.py') for key in
                        ('producer', 'frozen_eval_source', 'frozen_adapter_source', 'generation_patch')}
        self.dispatch = dict(family='moss', invocation_nonce='CPU-test-current-invocation',
            world_size=8, generation_patch_entrypoint='apply_moss_generation_patch',
            source_ids=['test-%04d' % i for i in range(500)], source_refs=self.sources,
            checkpoint_identity=dict(fingerprint='CPU-fake-not-trained'), data_refs={},
            created_unix=time.time(), loading_timeout_seconds=10, completion_timeout_seconds=20,
            config=dict(path='/CPU-fake-config'), model='/CPU-fake-model',
            base_audit=dict(identity={'CPU_fake': True}),
            checkpoint='/CPU-fake-checkpoint', protocol=dict(path='/CPU-fake-protocol'),
            final_selection=dict(path='/CPU-fake-selection'), checkpoint_selection=dict(path='/CPU-fake-best'),
            checkpoint_marker=dict(path='/CPU-fake-marker'))
        self.dispatch['source_ids_sha256'] = w.ids_sha(self.dispatch['source_ids'])
        self.patch_evidence = dict(requested_ref=self.sources['generation_patch'],
                                   applied_ref=self.sources['generation_patch'], native_eos_token_ids=[9])

    def identity(self, rank):
        return dict(w.identity_fields(self.dispatch, rank), pid=80000 + rank, pid_start_ticks=90000 + rank)

    def shards(self):
        for rank in range(8):
            directory = self.out / ('rank%d' % rank)
            directory.mkdir()
            ident = self.identity(rank)
            ready = dict(ident, device='cuda:%d' % rank, model_parameters_on_assigned_GPU=True,
                         generation_patch=self.patch_evidence)
            write_json(self.out / ('start-rank%d.json' % rank), ident)
            write_json(self.out / ('loaded-rank%d.json' % rank), ready)
            write_json(directory / 'native_source_loaded.json', dict(ready, actual_imported_source_refs=self.sources))
            indices = list(range(rank, 500, 8))
            records = []
            with (directory / 'raw.jsonl').open('wb') as stream:
                for index in indices:
                    text = '  CPU fake raw\nnot GPU evidence; index %d  ' % index
                    native = dict(raw_text=text, new_token_ids=[7, 9], eos_token_ids=[9], native_stop=False,
                                  applied_generation_patch_ref=self.sources['generation_patch'])
                    record = dict(id=self.dispatch['source_ids'][index], condition='original',
                        source_group_id='CPU-fake-group-%d' % index, audio_source_id=self.dispatch['source_ids'][index],
                        audio_source_group_id='CPU-fake-group-%d' % index, audio_sha256='CPU-fake-audio',
                        audio_transform='none', audio_ref=dict(path='/CPU-fake-audio', sha256='CPU-fake-audio'),
                        raw_text=text, native_output=native, new_token_ids=[7, 9], eos_token_ids=[9],
                        generation_budget=2048, generated_tokens=2, stop_reason='eos', failure=None,
                        metrics=ev.text_metrics(text), global_index=index, rank=rank, device='cuda:%d' % rank,
                        invocation_nonce=self.dispatch['invocation_nonce'], source_refs=self.sources,
                        data_refs={}, generation_patch=self.patch_evidence, actual_forward_count=2,
                        finite_native_logits_forward_count=2, native_logits_all_finite=True)
                    records.append(record)
                    # Nondefault separators make a rewrite detectable, including escaped raw text.
                    stream.write((json.dumps(record, ensure_ascii=False, separators=(', ', ': ')) + '\n').encode())
            raw = directory / 'raw.jsonl'
            manifest = dict(ident, source_ids=[self.dispatch['source_ids'][i] for i in indices],
                global_indices=indices, generation_patch=self.patch_evidence, device='cuda:%d' % rank,
                complete=True, pretrained_GPU_generation_executed=True, human_quality_accepted=False,
                attempted=len(indices), actual_forward_count=2 * len(indices),
                finite_native_logits_forward_count=2 * len(indices), native_logits_all_finite=True, raw=w.file_ref(raw))
            write_json(directory / 'manifest.json', manifest)
            write_json(self.out / ('exit-rank%d.json' % rank), dict(ident, exit_code=0))

    def rewrite_record(self, rank, offset, modify):
        path = self.out / ('rank%d' % rank) / 'raw.jsonl'
        records = [json.loads(line) for line in path.read_bytes().splitlines()]
        modify(records[offset])
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        manifest_path = path.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['raw'] = w.file_ref(path)
        write_json(manifest_path, manifest)

    def test_exact500_partition_63_63_63_63_62_62_62_62(self):
        rows = [dict(id=id_) for id_ in self.dispatch['source_ids']]
        shards = [w.partition(rows, rank) for rank in range(8)]
        self.assertEqual([len(s) for s in shards], [63, 63, 63, 63, 62, 62, 62, 62])
        indices = [index for shard in shards for index, _ in shard]
        self.assertEqual(sorted(indices), list(range(500)))
        self.assertEqual(len(set(indices)), 500)
        for world, rank in ((7, 0), (8, 8), (8, False)):
            with self.assertRaises(RuntimeError):
                w.partition(rows, rank, world)
        with self.assertRaises(RuntimeError):
            w.partition(rows[:-1], 0)
        with self.assertRaises(RuntimeError):
            w.partition(rows[:-1] + rows[:1], 0)

    def test_native_seed42_sha_sort_selection_same_across_ranks(self):
        rows = [dict(id=id_) for id_ in reversed(self.dispatch['source_ids'])]
        selected = ev.select_rows(rows, 'test', limit=500, seed=42)
        self.assertEqual(selected, ev.select_rows(rows, 'test', limit=500, seed=42))
        self.assertEqual(len({index for rank in range(8) for index, _ in w.partition(selected, rank)}), 500)

    def test_serial_loader_does_not_load_without_actual_predecessor_ready(self):
        self.assertIsNone(w.wait_previous_loaded(self.out, self.dispatch, 0))
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            w.wait_previous_loaded(self.out, self.dispatch, 1, clock=lambda: 10**20, sleep=lambda _: None)
        ident = self.identity(0)
        write_json(self.out / 'start-rank0.json', ident)
        with patch.object(w, 'process_alive', return_value=False), self.assertRaisesRegex(RuntimeError, 'died'):
            w.wait_previous_loaded(self.out, self.dispatch, 1)
        write_json(self.out / 'loaded-rank0.json', dict(ident, device='cuda:0', model_parameters_on_assigned_GPU=True))
        with patch.object(w, 'process_alive', return_value=True):
            loaded = w.wait_previous_loaded(self.out, self.dispatch, 1)
            self.assertEqual(loaded['rank'], 0)

    def test_serial_loader_wrong_nonce_PID_or_GPU_rejected(self):
        ident = self.identity(0)
        write_json(self.out / 'start-rank0.json', ident)
        for change in (dict(invocation_nonce='CPU-old-invocation'), dict(pid=123), dict(device='cuda:1'),
                       dict(model_parameters_on_assigned_GPU=False)):
            write_json(self.out / 'loaded-rank0.json', dict(dict(ident, device='cuda:0', model_parameters_on_assigned_GPU=True), **change))
            with patch.object(w, 'process_alive', return_value=True), self.assertRaises(RuntimeError):
                w.wait_previous_loaded(self.out, self.dispatch, 1)

    def test_any_failed_rank_stops_load_and_dead_completion(self):
        write_json(self.out / 'exit-rank4.json', dict(self.identity(4), exit_code=1, error='CPU-fake-technical-failure'))
        with self.assertRaisesRegex(RuntimeError, 'rank failed'):
            w.wait_previous_loaded(self.out, self.dispatch, 1)
        with self.assertRaisesRegex(RuntimeError, 'rank failed'):
            w.await_other_completions(self.out, self.dispatch)

    def test_aggregate_global_order_byte_preserving_new_producer_and_summary(self):
        self.shards()
        records, original_lines, receipts, diagnostics = w.collect_shards(self.out, self.dispatch, ev)
        self.assertEqual(len(records), 500)
        self.assertEqual([r['global_index'] for r in records], list(range(500)))
        self.assertEqual(len(receipts), 8)
        self.assertEqual(diagnostics, [])
        manifest = w.aggregate(self.out, self.dispatch, ev)
        self.assertEqual((self.out / 'raw.jsonl').read_bytes(), b''.join(original_lines))
        self.assertEqual(manifest['mode'], 'finaltest_parallel')
        self.assertEqual(manifest['code'], self.sources['producer'])
        self.assertEqual(manifest['frozen_adapter_source'], self.sources['frozen_adapter_source'])
        self.assertFalse(manifest['human_quality_accepted'])
        self.assertEqual(json.loads((self.out / 'summary.json').read_text()), ev.summarize(records))
        original = (self.out / 'raw.jsonl').read_bytes()
        with self.assertRaises(FileExistsError):
            w.aggregate(self.out, self.dispatch, ev)
        self.assertEqual((self.out / 'raw.jsonl').read_bytes(), original)
        cfg = dict(out=str(self.out), family='moss')
        evidence = m.check_worker_outputs(cfg, self.dispatch, w, ev)
        self.assertTrue(evidence['each_source_once'])
        self.assertFalse(evidence['human_quality_accepted'])

    def test_duplicate_extra_missing_or_wrong_rank_raw_rejected_before_combined_write(self):
        self.shards()
        for change in (dict(global_index=8), dict(global_index=500), dict(rank=1),
                       dict(id='CPU-wrong-ID'), dict(device='cpu'), dict(invocation_nonce='stale'),
                       dict(actual_forward_count=0), dict(finite_native_logits_forward_count=0),
                       dict(native_logits_all_finite=False)):
            path = self.out / 'rank0/raw.jsonl'
            saved = path.read_bytes()
            self.rewrite_record(0, 0, lambda r: r.update(change))
            with self.assertRaises(RuntimeError):
                w.aggregate(self.out, self.dispatch, ev)
            self.assertFalse((self.out / 'raw.jsonl').exists())
            path.write_bytes(saved)
            manifest_path = path.parent / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['raw'] = w.file_ref(path)
            write_json(manifest_path, manifest)
        path = self.out / 'rank7/raw.jsonl'
        path.write_bytes(b''.join(path.read_bytes().splitlines(keepends=True)[:-1]))
        with self.assertRaises(RuntimeError):
            w.aggregate(self.out, self.dispatch, ev)

    def test_technical_failure_retained_and_no_combined_complete(self):
        self.shards()
        self.rewrite_record(3, 0, lambda r: r.update(failure=dict(type='CPUFakeFailure', message='no rerun')))
        preserved = (self.out / 'rank3/raw.jsonl').read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'technical interface failure'):
            w.aggregate(self.out, self.dispatch, ev)
        self.assertEqual((self.out / 'rank3/raw.jsonl').read_bytes(), preserved)
        self.assertFalse((self.out / 'raw.jsonl').exists())

    def test_budget_unknown_and_empty_are_quality_diagnostics_not_retry(self):
        self.shards()
        def change_cap(record):
            record.update(new_token_ids=[7] * 2048, generated_tokens=2048, stop_reason='budget_truncation')
            record['native_output']['new_token_ids'] = record['new_token_ids']
        self.rewrite_record(0, 0, change_cap)
        def change_unknown_empty(record):
            record.update(new_token_ids=[7], generated_tokens=1, stop_reason='unknown_stop', raw_text='  ')
            record['native_output'].update(new_token_ids=[7], raw_text='  ')
            record['metrics'] = ev.text_metrics('  ')
        self.rewrite_record(1, 0, change_unknown_empty)
        manifest = w.aggregate(self.out, self.dispatch, ev)
        self.assertTrue(manifest['complete'])
        self.assertEqual(manifest['diagnostic_count'], 2)
        self.assertFalse(manifest['automatic_retry'])
        self.assertTrue(manifest['diagnostics_do_not_trigger_retry'])

    def test_wrong_loaded_patch_native_stamp_or_source_manifest_rejected(self):
        self.shards()
        self.rewrite_record(0, 0, lambda r: r['native_output'].update(applied_generation_patch_ref=dict(path='CPU-other-patch')))
        with self.assertRaisesRegex(RuntimeError, 'native output patch'):
            w.collect_shards(self.out, self.dispatch, ev)
        self.rewrite_record(0, 0, lambda r: r['native_output'].update(applied_generation_patch_ref=self.sources['generation_patch']))
        path = self.out / 'rank3/native_source_loaded.json'
        loaded = json.loads(path.read_text())
        loaded['actual_imported_source_refs'] = {}
        write_json(path, loaded)
        with self.assertRaisesRegex(RuntimeError, 'actual GPU'):
            w.collect_shards(self.out, self.dispatch, ev)

    def test_real_rank_worker_uses_previous_GPU_barrier_then62_original_native_calls(self):
        self.run_fake_rank(nonfinite=False)

    def test_real_rank_worker_nonfinite_logits_raise_technical_failure_keep_partial_raw_no_retry(self):
        self.run_fake_rank(nonfinite=True)

    def run_fake_rank(self, nonfinite):
        # Explicit fake CUDA/model only: exercises the real worker loop and frozen
        # evaluate_one, not an actual GPU pass or quality verdict.
        audio = self.root / 'CPU-audio.bin'
        audio.write_bytes(b'CPU-fake-audio-not-an-official-dataset')
        rows = [dict(id=id_, source_group_id='CPU-group-' + id_, audio_path=str(audio),
                     audio_sha256=w.sha(audio), audio_window_seconds=[0.0, 5.0625],
                     target_text='CPU-target-never-passed-to-generator', target_windows=[]) for id_ in self.dispatch['source_ids']]
        rows = ev.select_rows(rows, 'test', limit=500, seed=42)
        self.dispatch.update(out=str(self.out), data=str(self.root), model='/CPU-fake-model',
            source_ids=[r['id'] for r in rows], checkpoint_identity=dict(training_config=dict(max_length=8192)))
        self.dispatch['source_ids_sha256'] = w.ids_sha(self.dispatch['source_ids'])
        self.sources['frozen_eval_source'] = w.file_ref(b.FROZEN / 'music_planner_eval.py')
        self.sources['frozen_adapter_source'] = w.file_ref(b.FROZEN / 'music_planner_adapters.py')
        previous = dict(self.identity(6), device='cuda:6', model_parameters_on_assigned_GPU=True)
        write_json(self.out / 'start-rank6.json', self.identity(6))
        write_json(self.out / 'loaded-rank6.json', previous)
        events, calls, hooks = [], [], []
        model = types.SimpleNamespace(
            parameters=lambda: iter([types.SimpleNamespace(device=types.SimpleNamespace(type='cuda', index=7))]),
            register_forward_pre_hook=lambda hook: (hooks.append(hook) or types.SimpleNamespace(remove=lambda: None)),
            register_forward_hook=lambda hook: (hooks.append(hook) or types.SimpleNamespace(remove=lambda: None)))
        class FakeTensor:
            device = types.SimpleNamespace(type='cuda', index=7)
            def numel(self):
                return 9
        def generate(audio_input, max_new_tokens, silence, seed):
            calls.append((audio_input, max_new_tokens, silence, seed))
            self.assertNotIn('target_text', audio_input)
            hooks[0](model, ())
            hooks[1](model, (), types.SimpleNamespace(logits=FakeTensor()))
            return dict(raw_text='CPU fake native output', new_token_ids=[7,9], eos_token_ids=[9], native_stop=False,
                        applied_generation_patch_ref=self.sources['generation_patch'])
        adapter = types.SimpleNamespace(model=model, generate=generate, eos_token_ids=lambda: [9])
        def load_adapter(**options):
            events.append('load_adapter')
            self.assertEqual(options['device'], 'cuda:7')
            return adapter
        fake_adapters = types.SimpleNamespace(__file__=self.sources['frozen_adapter_source']['path'], load_planner_adapter=load_adapter)
        fake_eval = types.SimpleNamespace(__file__=self.sources['frozen_eval_source']['path'],
            validate_data=lambda _: ({'test': rows}, {}), select_rows=ev.select_rows,
            evaluate_one=ev.evaluate_one, classify_stop=ev.classify_stop)
        fake_patch = types.SimpleNamespace(__file__=self.sources['generation_patch']['path'],
                                           apply_moss_generation_patch=lambda _: self.sources['generation_patch'])
        fake_cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 8,
            set_device=lambda rank: events.append('device%d' % rank), manual_seed_all=lambda _: None,
            synchronize=lambda _: None, get_device_name=lambda _: 'CPU-fake-CUDA', memory_allocated=lambda _: 1)
        fake_torch = types.SimpleNamespace(cuda=fake_cuda, manual_seed=lambda _: None, Tensor=FakeTensor,
            isfinite=lambda _: types.SimpleNamespace(all=lambda: types.SimpleNamespace(item=lambda: not (nonfinite and len(calls) > 1))))
        original_barrier = w.wait_previous_loaded
        def barrier(*args):
            result = original_barrier(*args)
            events.append('actual_previous_loaded_verified')
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, LOCAL_RANK='7', RANK='7', WORLD_SIZE='8',
                                            CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'))
            stack.enter_context(patch.dict(__import__('sys').modules, torch=fake_torch))
            stack.enter_context(patch.object(w, 'verify_dispatch'))
            stack.enter_context(patch.object(w.signal, 'signal'))
            stack.enter_context(patch.object(w.signal, 'setitimer'))
            stack.enter_context(patch.object(w, 'check_inventory'))
            stack.enter_context(patch.object(w, 'load_module', side_effect=[fake_eval, fake_adapters, fake_patch]))
            stack.enter_context(patch.object(w, 'process_alive', return_value=True))
            stack.enter_context(patch.object(w, 'wait_previous_loaded', side_effect=barrier))
            if nonfinite:
                with self.assertRaisesRegex(RuntimeError, 'technical interface failure'):
                    w.run_rank(self.dispatch, self.sources['producer']['sha256'])
            else:
                w.run_rank(self.dispatch, self.sources['producer']['sha256'])
        if nonfinite:
            raw = [json.loads(line) for line in (self.out / 'rank7/raw.jsonl').read_text().splitlines()]
            self.assertEqual(len(raw), 2)
            self.assertIsNone(raw[0]['failure'])
            self.assertTrue(raw[0]['native_logits_all_finite'])
            self.assertEqual(raw[1]['failure']['type'], 'FloatingPointError')
            self.assertIn('nonfinite native logits', raw[1]['failure']['message'])
            self.assertFalse(raw[1]['native_logits_all_finite'])
            self.assertEqual(raw[1]['finite_native_logits_forward_count'], 0)
            self.assertEqual(raw[1]['actual_forward_count'], 1)
            exit_receipt = json.loads((self.out / 'exit-rank7.json').read_text())
            self.assertEqual(exit_receipt['exit_code'], 1)
            self.assertFalse(exit_receipt['automatic_retry'])
            self.assertFalse((self.out / 'rank7/manifest.json').exists())
            self.assertFalse((self.out / 'raw.jsonl').exists())
            return
        self.assertLess(events.index('actual_previous_loaded_verified'), events.index('load_adapter'))
        self.assertEqual(len(calls), 62)
        self.assertTrue(all(budget == 2048 and silence is False and seed == 42 for _, budget, silence, seed in calls))
        raw = [json.loads(line) for line in (self.out / 'rank7/raw.jsonl').read_text().splitlines()]
        self.assertEqual([r['global_index'] for r in raw], list(range(7,500,8)))
        self.assertTrue(all(r['condition'] == 'original' and r['actual_forward_count'] == 1 for r in raw))
        self.assertTrue(all('teacher_forced' not in r for r in raw))
        self.assertTrue(all(r['native_logits_all_finite'] and r['finite_native_logits_forward_count'] == 1 for r in raw))
        self.assertEqual(json.loads((self.out / 'exit-rank7.json').read_text())['exit_code'], 0)
        self.assertTrue((self.out / 'rank7/native_source_loaded.json').is_file())

    def test_atomic_receipt_nonce_cannot_be_overwritten(self):
        path = self.root / 'receipt.json'
        w.new_json(path, dict(invocation_nonce='CPU-invocation-one'))
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            w.new_json(path, dict(invocation_nonce='CPU-invocation-two'))
        self.assertEqual(path.read_bytes(), original)

    def test_sources_rejected_before_import_even_on_CPU(self):
        path = self.root / 'fake.py'
        path.write_text('raise AssertionError("must never execute")')
        with self.assertRaisesRegex(RuntimeError, 'SHA changed'):
            w.load_module('CPU_must_not_import', dict(path=str(path), sha256='0' * 64))

    def test_all_eight_lock_busy_inode_and_symlink_guards(self):
        for gpu in range(8):
            (self.root / ('ms12k_gpu_%d.lock' % gpu)).touch()
        with patch.object(m, 'LOCK_ROOT', self.root):
            locks = m.open_all_locks()
            try:
                with self.assertRaises(BlockingIOError):
                    m.open_all_locks()
                path = self.root / 'ms12k_gpu_7.lock'
                path.rename(self.root / 'CPU-old-lock')
                path.touch()
                with self.assertRaisesRegex(RuntimeError, 'inode'):
                    m.check_lock(locks[7], 7)
            finally:
                for fd in locks:
                    os.close(fd)
            path = self.root / 'ms12k_gpu_7.lock'
            path.unlink()
            path.symlink_to(self.root / 'ms12k_gpu_6.lock')
            with self.assertRaises(RuntimeError):
                m.open_all_locks()

    def test_native_finish_resources_restores_all8_old_leases_and_unlocks(self):
        entries, locks, saved = [], [], {}
        for gpu in range(8):
            lock_path = self.root / ('ms12k_gpu_%d.lock' % gpu)
            lock_path.touch()
            lease_path = self.root / ('od_reserve_%d' % gpu)
            lease_path.touch()
            os.utime(lease_path, (1,1))
            old_lease = resource.lease_snapshot(lease_path)
            expected = dict(pid=100+gpu, uid=os.getuid(), argv=resource.RESTORE_ARGV, start_ticks=200+gpu,
                cgroup='CPU-same-cgroup', CVD=['CUDA_VISIBLE_DEVICES=%d' % gpu], OCCUPY=['OCCUPY_GPU=0'],
                exe='/CPU-python', sig_ign=0, sig_blk=0, uuid='CPU-GPU-%d' % gpu, gpu=gpu, state='S')
            entry = dict(gpu=gpu, uuid=expected['uuid'], expected=expected,
                reserve=lease_path, old=old_lease, leasefd=None, released=False,
                fd=os.open(lock_path, os.O_RDONLY), queue=self.root)
            entries.append(entry)
            resource.acquire_lease(entry)
        def identify(pid):
            return next((e['expected'] for e in entries if e['expected']['pid'] == pid), entries[0]['expected'])
        with ExitStack() as stack:
            stack.enter_context(patch.object(m, 'LOCK_ROOT', self.root))
            stack.enter_context(patch.object(resource.signal, 'signal'))
            stack.enter_context(patch.object(resource, 'identity', side_effect=identify))
            stack.enter_context(patch.object(resource, 'uuid_for', side_effect=lambda gpu: entries[gpu]['uuid']))
            stack.enter_context(patch.object(resource, 'apps', return_value=[dict(pid=e['expected']['pid'], uuid=e['uuid']) for e in entries]))
            locks = m.open_all_locks()
            resource.finish_resources(entries, locks, self.root, lambda name,value: saved.update({name:value}),
                                      [], dict(child=None, started=False))
            self.assertEqual(saved['resources_safe.json']['admitted_gpu_count'], 8)
            self.assertTrue(saved['resources_safe.json']['all_holders_safe'])
            for entry in entries:
                self.assertEqual(entry['reserve'].stat().st_mtime_ns, entry['old'].st_mtime_ns)
            reacquired = m.open_all_locks()
            for fd in reacquired:
                os.close(fd)

    def test_release_requires_all8_lease_locks_and_last_all8_identity_before_pidfd(self):
        entries = []
        for gpu in range(8):
            expected = dict(pid=100 + gpu, uid=os.getuid(), argv=resource.RESTORE_ARGV, start_ticks=200+gpu,
                cgroup='CPU-same-cgroup', CVD=['CUDA_VISIBLE_DEVICES=%d' % gpu], OCCUPY=['OCCUPY_GPU=0'],
                exe='/CPU-python', sig_ign=0, sig_blk=0, uuid='CPU-GPU-%d' % gpu, gpu=gpu, state='S')
            entries.append(dict(gpu=gpu, uuid=expected['uuid'], expected=expected, fd=1000+gpu,
                                leasefd=2000+gpu, lease_inode=(1, gpu), released=False))
        events = []
        def identify(pid):
            return next((e['expected'] for e in entries if e['expected']['pid'] == pid), entries[0]['expected'])
        def selected(fds, writes, errors, timeout):
            return (fds if timeout else [], [], [])
        with ExitStack() as stack:
            stack.enter_context(patch.object(resource, 'identity', side_effect=identify))
            stack.enter_context(patch.object(resource, 'uuid_for', side_effect=lambda gpu: entries[gpu]['uuid']))
            stack.enter_context(patch.object(resource, 'apps', return_value=[dict(pid=e['expected']['pid'], uuid=e['uuid']) for e in entries]))
            stack.enter_context(patch.object(resource, 'refresh_leases', side_effect=lambda _: events.append('all8-heartbeat')))
            stack.enter_context(patch.object(resource, 'headroom', return_value=dict(headroom=160 * 2**30)))
            stack.enter_context(patch.object(m, 'check_lock', side_effect=lambda fd, gpu: events.append('lock%d' % gpu)))
            stack.enter_context(patch.object(m.select, 'select', side_effect=selected))
            send = stack.enter_context(patch.object(resource.signal, 'pidfd_send_signal', side_effect=lambda fd, sig: events.append('signal%d' % fd)))
            m.release_all(entries, list(range(8)), resource, lambda: events.append('last-source-check'))
            self.assertEqual(send.call_count, 8)
            self.assertLess(events.index('last-source-check'), events.index('signal1000'))
            self.assertTrue(all(e['released'] for e in entries))
            send.reset_mock()
            with self.assertRaises(RuntimeError):
                m.release_all(entries[:-1], list(range(8)), resource, lambda: None)
            send.assert_not_called()
            entries[-1]['expected']['sig_ign'] = 1 << (signal.SIGTERM-1)
            with self.assertRaises(RuntimeError):
                m.release_all(entries, list(range(8)), resource, lambda: None)
            send.assert_not_called()


class FrozenReceiptCPU(unittest.TestCase):
    def setUp(self):
        self.fixture = old.CPUHandoff(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (m, b):
            for key in ('EXP', 'RT', 'DATA', 'HANDOFF'):
                self.stack.enter_context(patch.object(module, key, getattr(self.fixture, {'EXP':'exp','RT':'rt','DATA':'data','HANDOFF':'handoff'}[key])))
        self.stack.enter_context(patch.object(m, 'FROZEN', self.fixture.rt / 'source_v11'))
        self.final = self.fixture.final_config()
        row = self.final['families']['omni_captioner']
        self.cfg = dict(self.fixture.cfg, **row, schema='music-planner-parallel-test-manager-v1',
            node='N4', host=m.SCOPE['N4'][0], family='omni_captioner', gpus=list(range(8)), job='test',
            queue=str(self.fixture.handoff / 'parallel-captioner-queue'), out=str(self.fixture.handoff / 'parallel-captioner-out'),
            frozen_eval_source=dict(path=str(m.FROZEN / 'music_planner_eval.py'), sha256='CPU-fake'),
            frozen_adapter_source=dict(path=str(m.FROZEN / 'music_planner_adapters.py'), sha256='CPU-fake'),
            producer=dict(path=str(self.fixture.rt / 'new_parallel/music_planner_parallel_test.py'), sha256='CPU-fake'),
            generation_patch=dict(path=str(self.fixture.rt / 'new_patch/generation_eos_patch.py'), sha256='CPU-fake'),
            generation_patch_entrypoint='apply_generation_eos_patch', loading_timeout_seconds=60, completion_timeout_seconds=120)

    def test_explicit_8GPU_scope_never_reuses_old_GPU0_authorization(self):
        m.validate_scope(self.cfg)
        for change in (dict(node='N2', host='28.216.17.70', family='music_flamingo'),
                       dict(node='N7'), dict(gpus=[0]), dict(gpus=[False,1,2,3,4,5,6,7]),
                       dict(job='fixed3'), dict(release_authorized=False), dict(loading_timeout_seconds=0),
                       dict(generation_patch_entrypoint='apply_moss_generation_patch')):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                m.validate_scope(dict(self.cfg, **change))
        mutable = dict(self.cfg, generation_patch=dict(path=str(SCRIPTS / 'music_planner_generation_eos_patch.py'), sha256='CPU-fake'))
        with self.assertRaisesRegex(RuntimeError, 'separately frozen'):
            m.validate_scope(mutable)

    def test_real_dispatch_source_and_fake_inventory_SHA_guards(self):
        checkpoint = self.fixture.root / 'CPU-portable-checkpoint'
        checkpoint.mkdir()
        tensor = checkpoint / 'CPU-fake-tensor.bin'
        tensor.write_bytes(b'CPU-fake-tensor-not-real-weights')
        base_file = self.fixture.root / 'CPU-base.bin'
        base_file.write_bytes(b'CPU-fake-official-base')
        patch_source = self.fixture.root / 'CPU-patch.py'
        patch_source.write_text('raise AssertionError("source validation must not execute this")')
        refs = dict(producer=w.file_ref(SCRIPTS / 'music_planner_parallel_test.py'),
            frozen_eval_source=w.file_ref(b.FROZEN / 'music_planner_eval.py'),
            frozen_adapter_source=w.file_ref(b.FROZEN / 'music_planner_adapters.py'),
            generation_patch=w.file_ref(patch_source))
        cfg = dict(family='moss', fingerprint='CPU-only-fingerprint',
            generation_patch_entrypoint='apply_moss_generation_patch', protocol=self.cfg['protocol'],
            final_selection=self.cfg['final_selection'], checkpoint_selection=self.cfg['checkpoint_selection'],
            checkpoint_marker=self.cfg['checkpoint_marker'])
        config = write_json(self.fixture.root / 'CPU-dispatch-config.json', cfg)
        dispatch = dict(cfg, schema='music-planner-parallel-test-dispatch-v1', world_size=8,
            source_refs=refs, config=config, source_ids=['CPU-id-%d' % i for i in range(500)],
            checkpoint_identity=dict(training_config=dict(family='moss', fingerprint='CPU-only-fingerprint',
                adapter_source_sha256=refs['frozen_adapter_source']['sha256'])),
            test_used_for_checkpoint_selection=False, decoding=w.DECODE,
            data_refs={}, checkpoint=str(checkpoint), checkpoint_stat_snapshot={str(tensor): w.file_stat(tensor)},
            base_audit=dict(files=[dict(path=str(base_file), stat=w.file_stat(base_file))],
                            current_full_content_SHA_and_official_digest_verified=True))
        dispatch['source_ids_sha256'] = w.ids_sha(dispatch['source_ids'])
        w.verify_dispatch(dispatch, refs['producer']['sha256'])
        wrong = copy.deepcopy(dispatch)
        wrong['checkpoint_identity']['training_config']['adapter_source_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'no migration'):
            w.verify_dispatch(wrong, refs['producer']['sha256'])
        patch_source.write_text('CPU-source-changed')
        with self.assertRaisesRegex(RuntimeError, 'SHA changed'):
            w.verify_dispatch(dispatch, refs['producer']['sha256'])
        self.assertFalse((checkpoint / 'model_loaded.json').exists())

    def test_real_all3_complete_selection_gate_and_per_family_once_claim(self):
        b.check_final_selection(self.cfg, resource, fixed, ev, self.fixture.protocol)
        path = m.claim_test(self.cfg, 'CPU-config-sha', {'producer': self.cfg['producer']}, b)
        original = path.read_bytes()
        self.assertEqual(path.name, 'test_once_omni_captioner.json')
        self.assertEqual(json.loads(original)['world_size'], 8)
        with self.assertRaises(FileExistsError):
            m.claim_test(dict(self.cfg, out='/CPU-other-out'), 'CPU-other-config', {}, b)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(json.loads(original)['automatic_retry'])

    def test_real_frozen_helper_rejects_missing_family_or_corrupted_full_completion(self):
        final = copy.deepcopy(self.final)
        final['families'].pop('moss')
        self.cfg['final_selection'] = write_json(self.fixture.handoff / 'final_selection.json', final)
        with self.assertRaises(RuntimeError):
            b.check_final_selection(self.cfg, resource, fixed, ev, self.fixture.protocol)

    def test_standalone_local8_torchrun_no_retry_or_sampling(self):
        argv = m.worker_argv(self.cfg, dict(path='/CPU-dispatch', sha256='CPU-sha'))
        self.assertEqual(argv[:3], ['/opt/venv_music_planner_v1/bin/python', '-m', 'torch.distributed.run'])
        self.assertIn('--nproc_per_node=8', argv)
        self.assertIn('--max_restarts=0', argv)
        self.assertNotIn('--teacher-forced', argv)
        self.assertNotIn('--counterfactual', argv)
        self.assertNotIn('--clean', argv)


class LeaseWatchdogCPU(unittest.TestCase):
    def test_heartbeats_continue_during_long_checks_and_join_before_fd_cleanup(self):
        calls = []
        manager = types.SimpleNamespace(refresh_leases=lambda entries: calls.append(list(entries)))
        watchdog = m.start_lease_watchdog(['owned-only'], manager, interval=.005)
        time.sleep(.035)
        m.stop_lease_watchdog(watchdog)
        self.assertGreater(len(calls), 0)
        self.assertTrue(all(item == ['owned-only'] for item in calls))
        self.assertFalse(watchdog[2].is_alive())
        count = len(calls)
        time.sleep(.02)
        self.assertEqual(len(calls), count)
    def test_lease_error_is_recorded_not_silently_ignored(self):
        def fail(entries):
            raise RuntimeError('owned lease replaced')
        watchdog = m.start_lease_watchdog(['owned-only'], types.SimpleNamespace(refresh_leases=fail), interval=.005)
        time.sleep(.02)
        m.stop_lease_watchdog(watchdog)
        self.assertEqual(len(watchdog[1]), 1)
        self.assertIn('replaced', str(watchdog[1][0]))
        self.assertFalse(watchdog[2].is_alive())


if __name__ == '__main__':
    unittest.main()
