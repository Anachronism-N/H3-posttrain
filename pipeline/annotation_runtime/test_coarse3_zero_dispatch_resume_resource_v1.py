"""CPU-only gates. No admitted GPU result, checkpoint load, signal, or launch build."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import coarse3_zero_dispatch_resume_resource_v1 as r


def fixture(out):
    sources = [{'source_id': sid, 'root': str(out / 'sources' / ('train-' + sid) / 'fixed'),
                'candidate_ref': {'path': '/synthetic/' + sid, 'sha256': sid}} for sid in r.IDS]
    plan = {'sources': sources, 'maximum_review_calls': 3, 'population_ref': {'path': '/synthetic/pop', 'sha256': 'pop'}}
    flags = {'semantic_approval': False, 'training_eligible': False}
    summary = {**flags, 'population_ref': plan['population_ref'], 'selected_source_ids': list(r.IDS),
               'pending_source_ids': list(r.IDS), 'population_source_count': 3, 'maximum_review_calls': 3,
               'statuses': {'pending': 3}, 'review_dispatches': 0, 'actual_VLM_calls': 0,
               'GPU_model_load_performed': False, 'sampled_reviewed_denominator': 0,
               'independent_sample_review_completed': False, 'original_attempt_budget_consumed': False}
    rows = [{**flags, 'source_id': s['source_id'], 'candidate_ref': s['candidate_ref'], 'status': 'pending',
             'error': r.ERROR, 'review_dispatches': 0, 'actual_VLM_calls': 0, 'label_scope': 'complete'} for s in sources]
    (out / r.HISTORY_NAMES[0]).write_bytes(r.encoded(summary))
    (out / r.HISTORY_NAMES[1]).write_bytes(b''.join(r.encoded(x).replace(b'\n', b' ') + b'\n' for x in rows))
    stage = {**flags, 'launch_config_ref': r.PREVIOUS, 'source_count': 3, 'terminal_count': 3,
             'population_ref': plan['population_ref'], 'review_dispatches': 0, 'score_refs': [],
             'independent_summary_ref': r.ref(out / r.HISTORY_NAMES[0]),
             'independent_ledger_ref': r.ref(out / r.HISTORY_NAMES[1])}
    (out / r.HISTORY_NAMES[2]).write_bytes(r.encoded(stage))
    return plan


class MountBridgeTests(unittest.TestCase):
    def test_only_device_not_content_path_inode_or_times(self):
        bridge = r.load_module('cpu_test_mount_bridge', r.BRIDGE, r.BRIDGE_SHA)
        old = [{'path': '/same/path', 'sha256': 'content-a', 'stat':
                {'device': 10, 'inode': 42, 'size': 100, 'mtime_ns': 1, 'ctime_ns': 2}}]
        now = copy.deepcopy(old)
        now[0]['stat']['device'] = 20
        preserved, proof = bridge.bridge_references(old, now)
        self.assertEqual(preserved, old)
        self.assertEqual(proof[0]['current'], now[0])
        for container, key, value in [('ref', 'path', '/other/path'), ('ref', 'sha256', 'content-b'),
                                      ('stat', 'inode', 43), ('stat', 'size', 101),
                                      ('stat', 'mtime_ns', 2), ('stat', 'ctime_ns', 3)]:
            changed = copy.deepcopy(now)
            (changed[0] if container == 'ref' else changed[0]['stat'])[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bridge.bridge_references(old, changed)
        self.assertEqual(old[0]['stat']['device'], 10)


class ZeroHistoryTests(unittest.TestCase):
    def test_exact_zero_history_original_bytes_and_changed_bytes_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plan = fixture(out)
            before = {name: (out / name).read_bytes() for name in r.HISTORY_NAMES}
            history_ref = r.zero_history(out, plan, r.PREVIOUS, preserve=True)
            self.assertEqual(history_ref, r.zero_history(out, plan, r.PREVIOUS, history_ref=history_ref))
            for name, data in before.items():
                self.assertEqual((out / name).read_bytes(), data)
                self.assertEqual(r.checked_bytes(r.checked(history_ref)['files'][name]['snapshot_ref']), data)
            (out / r.HISTORY_NAMES[0]).write_bytes(before[r.HISTORY_NAMES[0]] + b' ')
            with self.assertRaises(RuntimeError):
                r.zero_history(out, plan, r.PREVIOUS, history_ref=history_ref)
            for name in r.HISTORY_NAMES:
                self.assertEqual(r.checked_bytes(r.checked(history_ref)['files'][name]['snapshot_ref']), before[name])

    def test_any_dispatch_budget_model_or_claim_denied(self):
        mutations = [('review_dispatches', 1), ('actual_VLM_calls', 1), ('GPU_model_load_performed', True),
                     ('original_attempt_budget_consumed', True), ('sampled_reviewed_denominator', 1)]
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                plan = fixture(out)
                summary = json.loads((out / r.HISTORY_NAMES[0]).read_bytes())
                summary[key] = value
                (out / r.HISTORY_NAMES[0]).write_bytes(r.encoded(summary))
                with self.assertRaises(RuntimeError):
                    r.zero_history(out, plan, r.PREVIOUS, preserve=True)
        for name in ('review-attempts', 'loads', 'reviewer.model.identity.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                plan = fixture(out)
                (out / name).mkdir()
                with self.assertRaises(RuntimeError):
                    r.zero_history(out, plan, r.PREVIOUS, preserve=True)
                self.assertTrue((out / name).exists())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plan = fixture(out)
            root = Path(plan['sources'][0]['root'])
            (root / 'review').mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                r.zero_history(out, plan, r.PREVIOUS, preserve=True)
            self.assertTrue((root / 'review').exists())


    def test_per_source_dispatch_or_review_budget_marker_denied(self):
        for name, change in [('ledger-dispatch', {'review_dispatches': 1}),
                             ('ledger-claim', {'review_budget_consumed': True}),
                             ('stage-dispatch', {'review_dispatches': 1})]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                plan = fixture(out)
                if name.startswith('ledger'):
                    path = out / r.HISTORY_NAMES[1]
                    rows = [json.loads(x) for x in path.read_bytes().splitlines()]
                    rows[0].update(change)
                    path.write_bytes(b''.join(json.dumps(x).encode() + bytes([10]) for x in rows))
                    stage_path = out / r.HISTORY_NAMES[2]
                    stage = json.loads(stage_path.read_bytes())
                    stage['independent_ledger_ref'] = r.ref(path)
                    stage_path.write_bytes(r.encoded(stage))
                else:
                    path = out / r.HISTORY_NAMES[2]
                    stage = json.loads(path.read_bytes())
                    stage.update(change)
                    path.write_bytes(r.encoded(stage))
                with self.assertRaises(RuntimeError):
                    r.zero_history(out, plan, r.PREVIOUS, preserve=True)


class RealInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c, cls.cache, cls.previous, cls.plan = r.configured()
        with r.private_import('prepare_coarse3_n6_v1', cls.c):
            cls.driver = cls.cache.module('cpu_test_native_driver', r.RT / 'coarse3_n6_driver_v1.py')
        cls.worker = cls.c.module('cpu_test_native_worker', cls.c.WORKER)

    def test_original_execute_and_admission_deny_without_callback(self):
        with self.assertRaisesRegex(ValueError, 'verified caller admission'):
            self.worker.execute(prepare_only=False, reviewer=object(), admission_callback=None)
        with self.assertRaisesRegex(ValueError, 'caller-owned'):
            self.worker.verified_admission(None, None, {})

    def test_real_owned_fd_flock_and_integer_lease(self):
        with tempfile.TemporaryFile() as stream:
            value = self.c.lease_value(os.fstat(stream.fileno()))
            self.c.verify_lease_value(value)
            self.assertTrue(self.c.fd_inode_owned(os.getpid(), value['st_dev'], value['st_ino']))
            value['st_mtime_ns'] = float(value['st_mtime_ns'])
            with self.assertRaises(RuntimeError):
                self.c.verify_lease_value(value)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cpu-lock'
            with path.open('w+b') as stream:
                self.assertFalse(self.c.proc_lock_owned(os.getpid(), str(path)))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(self.c.proc_lock_owned(os.getpid(), str(path)))

    def test_actual_parent_not_history_alone_no_GPU_probe(self):
        # Use the original real validator and actual frozen config, not an admission mock.
        with r.private_import('prepare_curated_coarse_empty_n5_v1', self.c.h):
            c = self.cache.module('cpu_test_raw_prepare', r.RT / 'prepare_coarse3_n6_v1.py')
        c.MANAGER = c.DRIVER = r.CACHE
        with r.private_import('prepare_coarse3_n6_v1', c):
            driver = self.cache.module('cpu_test_real_parent_driver', r.RT / 'coarse3_n6_driver_v1.py')
        base = c.base_module()
        with patch.object(base, 'apps', side_effect=AssertionError('GPU probe forbidden')), \
             patch.object(base, 'uuid_for', side_effect=AssertionError('GPU probe forbidden')):
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                driver.verify_actual_parent(self.previous, base, self.plan)

    def test_current_three_context_and_fourth_dispatch(self):
        source = self.plan['sources'][0]
        context = {'schema': self.worker.SCHEMA, 'caller_pid': os.getpid(), 'out': str(self.c.OUT),
                   'population_ref': self.plan['population_ref'], 'jobs_manifest_ref': self.plan['jobs_manifest_ref'],
                   'source_job_refs': self.plan['source_job_refs'], 'candidate_ref': source['candidate_ref'],
                   'input_row_sha256': source['row_sha256'], 'model_sha256': self.c.MODEL_SHA,
                   'tensor_parallel_size': 1, 'maximum_review_calls': 3}
        self.assertEqual(self.driver.verify_context(context, self.plan), source)
        for key in ('candidate_ref', 'input_row_sha256', 'population_ref', 'model_sha256'):
            changed = dict(context, **{key: 'changed'})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.driver.verify_context(changed, self.plan)
        counter = SimpleNamespace(_coarse3_dispatches=0)
        for sid in r.IDS:
            r.dispatch_guard({'id': sid}, counter)
            counter._coarse3_dispatches += 1
        with self.assertRaisesRegex(RuntimeError, 'fourth dispatch'):
            r.dispatch_guard({'id': r.IDS[0]}, counter)
        # No synthetic review status or fabricated admission is produced.

    def test_native_empty_CVD_exact_title_and_owned_tree_algorithm(self):
        real = self.c.base_module()
        own = real.identity(os.getpid())
        child = dict(own, pid=2147483000, ppid=own['pid'], start_ticks=own['start_ticks'] + 1,
                     argv=['VLLM::EngineCore'], CVD=[])
        table = {own['pid']: own, child['pid']: child}
        base = SimpleNamespace(identity=lambda pid: copy.deepcopy(table[pid]), IDENTITY_KEYS=real.IDENTITY_KEYS)
        tracker = self.driver.ModelDescendants(base)
        tracker.verify(child['pid'])
        for key, value in [('argv', ['VLLM::Worker_TP1']), ('argv', ['vllm-unknown-title']),
                           ('ppid', 1), ('CVD', ['CUDA_VISIBLE_DEVICES=2'])]:
            altered = dict(child, **{key: value})
            table[child['pid']] = altered
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.driver.ModelDescendants(base).verify(child['pid'])
        # This tests only the original native-title/tree algorithm, not admission.

    def test_real_native_tracker_rejects_current_non_vllm_process(self):
        tracker = self.driver.ModelDescendants(self.c.base_module())
        with self.assertRaises(RuntimeError):
            tracker.verify(os.getppid())

    def test_missing_safe_refuses_and_CLI_SELF_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'unsafe-resource.json'
            path.write_bytes(r.encoded({'node': 'N6', 'gpus': [3], 'modelref': self.c.modelref()}))
            with patch.object(self.c, 'fresh_holder', side_effect=AssertionError('GPU forbidden')):
                with self.assertRaises(RuntimeError):
                    r.resource_chain(self.c, r.ref(path), r.PREVIOUS)
        result = subprocess.run([sys.executable, '-B', str(r.SELF), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--from-launch', result.stdout)
        for mode in ('--config', '--launch'):
            help_result = subprocess.run([sys.executable, '-B', str(r.SELF), mode, '/tmp/CPU-help-never-read', '--help'], capture_output=True, text=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn(mode, help_result.stdout)
        self.assertEqual(r.manager_argv(self.c, self.previous)[3], str(r.SELF))
        self.assertEqual(self.c.MANAGER, r.SELF)
        self.assertEqual(self.c.DRIVER, r.SELF)
        self.assertEqual(self.cache.SELF, r.SELF)
        with r.private_import('prepare_coarse3_n6_v1', self.c):
            manager = self.cache.module('cpu_test_manager_argv', r.RT / 'coarse3_n6_resource_manager_v1.py')
        self.assertEqual(manager.c.MANAGER, r.SELF)
        self.assertEqual(manager.c.DRIVER, r.SELF)


if __name__ == '__main__':
    unittest.main()
