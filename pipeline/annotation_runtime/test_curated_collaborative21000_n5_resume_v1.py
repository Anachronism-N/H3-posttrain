"""CPU receipts-only tests: no model import/load, real GPU query, capture or signals."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_curated_collaborative21000_n5_resume_v1 as c


def identity(pid, argv, gpu=None):
    return {'pid': pid, 'uid': os.getuid(), 'start_ticks': pid * 10, 'argv': argv,
            'cgroup': 'fixture', 'CVD': [] if gpu is None else ['CUDA_VISIBLE_DEVICES=%d' % gpu],
            'exe': '/usr/bin/python3.11', 'ppid': 1}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.temp.name)
        self.rt, self.q, self.out = self.root / 'runtime', self.root / 'finiteq', self.root / 'finiteout'
        self.t = 1700000000000000000
        for name, value in {'RT': self.rt}.items():
            self.stack.enter_context(patch.object(c, name, value))
        for name, value in {'LAUNCH': self.rt / 'finite.json', 'QUEUE': self.q, 'OUT': self.out,
                            'MANAGER': self.rt / 'finite_manager.py'}.items():
            self.stack.enter_context(patch.object(c.finite, name, value))
        self.source = self.put(self.rt / 'source.json', {})
        self.put(c.finite.MANAGER, {})
        self.stack.enter_context(patch.object(c, 'FINITE_MANAGER_SHA', c.finite.sha(c.finite.MANAGER)))
        self.originals, self.prior = [], {'previous_pairs': [], 'prior_producer_completion_not_claimed': True}
        holders = []
        for gpu in (0, 1):
            holder = dict(identity(100 + gpu, c.base.RESTORE_ARGV, gpu), gpu=gpu, uuid='fixture-gpu-%d' % gpu)
            holders.append(holder)
            oldq = self.root / ('oldq%d' % gpu)
            old = {'node': 'N5', 'gpus': [gpu], 'queue': str(oldq), 'phase': 'old%d' % gpu,
                   'argv': ['fixture-producer', str(gpu)]}
            oldpath = self.put(self.rt / ('old%d.json' % gpu), old)
            oldref = c.finite.ref(oldpath)
            self.originals.append((oldref, old))
            manager = identity(200 + gpu, ['fixture-old-manager'])
            admitted = {'manager_identity': manager}
            admission_path = self.put(oldq / 'admission.json', admitted, self.t)
            safe_path = self.put(oldq / 'resources_safe.json', {'phase': old['phase'], **dict.fromkeys(c.SAFE_KEYS, True)}, self.t + 2)
            restored = {'holder_identity': holder, 'uuid': holder['uuid']}
            self.prior['previous_pairs'].append({'gpu': gpu, 'restored': restored,
                'history_refs': [oldref, c.finite.ref(admission_path), c.finite.ref(safe_path)]})
            worker = dict(identity(300 + gpu, old['argv'], gpu), ppid=manager['pid'])
            self.put(oldq / 'live.json', {'phase': old['phase'], 'argv': old['argv'],
                                        'manager_identity': manager, 'child_identity': worker})
            self.put(self.rt / ('n5_coarse_empty_transition_gpu%d_v1.json' % gpu), {
                'schema': 'owned_producer_resource_transition_v1', 'node': 'N5', 'gpus': [gpu],
                'old_config_ref': oldref, 'original_queue': str(oldq), 'manager_identity': manager,
                'worker_identity': worker, 'signal': 'SIGTERM', 'signal_target': 'exact_owned_manager_pidfd',
                'reason': 'temporary_finite_three_coarse_window_validation', 'original_outputs_preserved': True,
                'incomplete_sources_remain_pending': True, 'old_source_selection_completed': False,
                'unknown_jobs_not_signaled': True, 'observed_unix_ns': self.t + 1})
        self.stack.enter_context(patch.object(c.finite, 'prior_safe', return_value=self.prior))
        self.stack.enter_context(patch.object(c, 'old_config', side_effect=lambda gpu: self.originals[gpu]))
        prior_ref = c.finite.ref(self.put(self.rt / 'prior.json', self.prior))
        hints, terminals = [], []
        for number in range(3):
            source_key = 'train/s%d' % number
            jobref = c.finite.ref(self.put(self.rt / ('job%d.json' % number), {'source_key': source_key}))
            hints.append({'source_key': source_key, 'job_ref': jobref})
            terminal = {'source_key': source_key, 'original_job_ref': jobref, 'global_revision': c.finite.REVISION,
                        'status': 'failed_preserved_no_redispatch', 'semantic_approval': False, 'training_eligible': False}
            terminals.append(c.finite.ref(self.put(self.out / 'worker-terminals' / (source_key + '.json'), terminal, self.t + 11)))
        manifest_ref = c.finite.ref(self.put(self.rt / 'manifest.json', {'jobs': hints}))
        self.cfg = {'node': 'N5', 'gpus': [0, 1], 'phase': c.finite.PHASE, 'queue': str(self.q), 'out': str(self.out),
                    'targetcount': 3, 'input_count': 222939, 'global_revision': c.finite.REVISION,
                    'maximum_total_visual_calls': 3, 'maximum_text_model_calls': 0,
                    'previous_resources_ref': prior_ref, 'jobs_manifest_ref': manifest_ref, 'holders': holders,
                    'producer': c.finite.ref(self.source), 'plan': c.finite.ref(self.source),
                    'source_sha256': {}, 'modelref': {'model_identity_sha256': 'fixture-model'}}
        self.put(c.finite.LAUNCH, self.cfg)
        launch_ref = c.finite.ref(c.finite.LAUNCH)
        self.stack.enter_context(patch.object(c, 'FINITE_LAUNCH_SHA', launch_ref['sha256']))
        manager = identity(400, ['/opt/venv_eval/bin/python', '-B', '-u', str(c.finite.MANAGER), '--config',
                                 str(c.finite.LAUNCH), '--config-sha', launch_ref['sha256'], '--manager-sha', c.finite.BASE_SHA])
        self.admission = {'config_sha256': launch_ref['sha256'], 'manager_sha256': c.finite.BASE_SHA,
                          'phase': c.finite.PHASE, 'node': 'N5', 'gpus': [0, 1], 'targetcount': 3,
                          'producer': self.cfg['producer'], 'plan': self.cfg['plan'], 'out': str(self.out),
                          'source_sha256': {}, 'model_identity_sha256': 'fixture-model',
                          'manager_identity': manager, 'budget': {'headroom': 160 * 2**30},
                          'holders': [{'gpu': gpu, 'uuid': h['uuid'], 'identity': h} for gpu, h in enumerate(holders)]}
        admission_ref = c.finite.ref(self.put(self.q / 'admission.json', self.admission, self.t + 10))
        held_ref = c.finite.ref(self.put(self.q / 'actual_leases_held.json', {
            'schema': 'coarse_empty_n5_actual_leases_held_v1', 'config_ref': launch_ref,
            'resource_admission_ref': admission_ref, 'node': 'N5', 'gpus': [0, 1], 'manager_identity': manager,
            'leases': [{'gpu': gpu, 'path': '/tmp/od_reserve_%d' % gpu, 'uuid': h['uuid'],
                        'stat': dict.fromkeys(c.finite.LEASE_KEYS, 1)} for gpu, h in enumerate(holders)]}, self.t + 11))
        self.put(self.q / 'actual_worker_admission.json', {'launch_config_ref': launch_ref,
            'resource_admission_ref': admission_ref, 'actual_held_leases_ref': held_ref,
            'source_job_refs': [h['job_ref'] for h in hints], 'jobs_manifest_ref': manifest_ref,
            'global_revision': c.finite.REVISION, 'gpus': [0, 1], 'tensor_parallel_size': 2, 'out': str(self.out)}, self.t + 11)
        self.put(self.q / 'last_result_proof.json', {'terminal_refs': terminals, 'partial_outputs_preserved': True,
            'actual_model_dispatches': 0, 'text_model_calls': 0, 'semantic_approval': False, 'evidence_refs': []}, self.t + 12)
        self.summary = {'terminals': terminals, 'jobs_manifest_ref': manifest_ref,
                        'source_job_refs': [h['job_ref'] for h in hints], 'source_count': 3, 'terminal_count': 3,
                        'global_revision': c.finite.REVISION, 'text_model_calls': 0, 'visual_model_calls': 0,
                        'partial_outputs_preserved': True, 'semantic_approval': False, 'training_eligible': False}
        self.put(self.out / 'finite.stage.summary.json', self.summary, self.t + 13)
        self.put(self.q / 'failure.json', {'phase': c.finite.PHASE, 'partial_outputs_preserved': True,
                                         'unknown_jobs_not_signaled': True, 'worker_started': True}, self.t + 14)
        for gpu, holder in enumerate(self.admission['holders']):
            self.put(self.q / ('holder_restored_gpu%d.json' % gpu), {'gpu': gpu, 'uuid': holder['uuid'],
                'holder_identity': holders[gpu], 'original_identity': holder['identity'],
                'restore_argv': c.base.RESTORE_ARGV, 'managed_parent_wait': True}, self.t + 15)
        self.put(self.q / 'resources_safe.json', {'phase': c.finite.PHASE, **dict.fromkeys(c.SAFE_KEYS, True)}, self.t + 20)

    def put(self, path, value, timestamp=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        if timestamp is not None:
            os.utime(path, ns=(timestamp, timestamp))
        return path

    def change(self, path, key, value):
        previous = c.finite.load(path)
        previous[key] = value
        self.put(path, previous, path.stat().st_mtime_ns)

    def test_actual_failure_without_exit_preserves_pending(self):
        gate = c.resource_gate()
        self.assertFalse(gate['finite_worker_exit_present'])
        self.assertIsNone(gate['finite_worker_exit_code'])
        self.assertFalse(gate['old_source_selection_completed'])
        self.assertTrue(gate['source_failures_preserved'])
        self.assertFalse(gate['semantic_approval'])

    def test_partial_zero_exit_is_not_semantic_success(self):
        self.put(self.q / 'worker_exit.json', {'phase': c.finite.PHASE, 'exit_code': 0, 'targetcount': 3,
            'input_count': 222939, 'producer': self.cfg['producer'], 'plan': self.cfg['plan'],
            'source_unchanged': True, 'semantic_approval': False, 'training_entry_updated': False,
            'summaries': {str(self.out / 'finite.stage.summary.json'): {'actual_finite_result': self.summary}}}, self.t + 14)
        gate = c.resource_gate()
        self.assertEqual(gate['finite_worker_exit_code'], 0)
        self.assertTrue(gate['finite_failure_present'])
        self.assertFalse(gate['semantic_approval'])

    def test_reject_incomplete_safe(self):
        for key in c.SAFE_KEYS:
            with self.subTest(key=key):
                self.change(self.q / 'resources_safe.json', key, False)
                with self.assertRaises(RuntimeError):
                    c.resource_gate()
                self.change(self.q / 'resources_safe.json', key, True)

    def test_reject_wrong_transition(self):
        self.change(self.rt / 'n5_coarse_empty_transition_gpu0_v1.json', 'signal_target', 'process_group')
        with self.assertRaises(RuntimeError):
            c.resource_gate()

    def test_reject_failure_without_source_proof(self):
        (self.q / 'last_result_proof.json').unlink()
        with self.assertRaises(RuntimeError):
            c.resource_gate()

    def test_reject_changed_restore_uuid(self):
        self.change(self.q / 'holder_restored_gpu1.json', 'uuid', 'wrong')
        with self.assertRaises(RuntimeError):
            c.resource_gate()

    def test_reject_changed_actual_admission(self):
        self.change(self.q / 'admission.json', 'config_sha256', 'wrong')
        with self.assertRaises(RuntimeError):
            c.resource_gate()

    def test_reject_no_terminal_evidence(self):
        (self.out / 'worker-terminals/train/s1.json').unlink()
        with self.assertRaises(FileNotFoundError):
            c.resource_gate()

    def test_build_fails_before_fresh_capture(self):
        with patch.object(c.finite, 'check_node'), patch.object(c, 'resource_gate', side_effect=RuntimeError('not safe')), patch.object(c.old, 'capture_current') as capture:
            with self.assertRaises(RuntimeError):
                c.build(0)
            capture.assert_not_called()
            self.assertFalse(c.paths(0)['launch'].exists())


class ExactConfigTests(unittest.TestCase):
    def test_only_transaction_metadata_changes(self):
        original_ref, previous = c.old_config(0)
        cfg = copy.deepcopy(previous)
        p = c.paths(0)
        gate = {'references': [], 'restored_holders': [{'uuid': cfg['holders'][0]['uuid'], 'holder_identity': cfg['holders'][0]}]}
        cfg.update(phase=c.phase(0), queue=str(p['queue']), resume_original_config=original_ref,
                   completed_resource_gate=gate, resource_manager=c.finite.ref(c.MANAGER),
                   holder_capture={'path': str(p['capture']), 'sha256': 'fixture-capture'})
        cfg['source_sha256'].update({str(c.BUILDER): c.finite.sha(c.BUILDER), str(c.MANAGER): c.finite.sha(c.MANAGER),
            original_ref['path']: original_ref['sha256'], str(p['capture']): 'fixture-capture',
            str(c.finite.__file__): c.FROZEN_PREPARE_SHA})
        with patch.object(c, 'resource_gate', return_value=gate):
            c.verify_resume_config(cfg)
            for key, value in {'reuse_existing_results': False, 'targetcount': 4999, 'argv': [],
                               'env': {}, 'out': '/tmp/wrong', 'source_selection': {}}.items():
                with self.subTest(key=key):
                    changed = copy.deepcopy(cfg)
                    changed[key] = value
                    with self.assertRaises(RuntimeError):
                        c.verify_resume_config(changed)

    def test_exact_gpu_scope(self):
        for gpu in (-1, 2, True, 0.0):
            with self.subTest(gpu=gpu), self.assertRaises(RuntimeError):
                c.phase(gpu)


class WrapperLeaseTests(unittest.TestCase):
    def test_capture_requires_exact_integer_ns(self):
        from types import SimpleNamespace
        import curated_collaborative21000_n5_resume_resource_manager_v1 as wrapper
        owner = dict(identity(999, c.base.RESTORE_ARGV, 0), gpu=0, uuid='fixture')
        cfg = {'node': 'N5', 'mode': 'producer', 'gpus': [0], 'holders': [owner], 'holder_capture': {}}
        stat = SimpleNamespace(st_dev=1, st_ino=2, st_uid=os.getuid(), st_mode=33188,
                               st_size=0, st_mtime_ns=1792915625795513301)
        lease = c.old.lease_value(stat)
        captured = {'node': 'N5', 'mode': 'producer', 'gpus': [0], 'holders': [owner],
                    'signals_sent': False, 'captured_unix_ns': 1, 'leases': [lease]}
        with patch.object(c.finite, 'checked', return_value=captured), patch.object(wrapper, 'original_snapshot', return_value=stat):
            wrapper.verify_capture_leases(cfg)
            captured['leases'] = [lease[:-1] + [float(lease[-1])]]
            with self.assertRaises(RuntimeError):
                wrapper.verify_capture_leases(cfg)
            captured['leases'] = [lease[:-1] + [lease[-1] + 1]]
            with self.assertRaises(RuntimeError):
                wrapper.verify_capture_leases(cfg)

    def test_native_signal_cleanup_budget_functions_unchanged(self):
        import curated_collaborative21000_n5_resume_resource_manager_v1 as wrapper
        for name in ('release_holder', 'acquire_lease', 'refresh_leases', 'restore_step',
                     'required_budget', 'check_holder', 'headroom', 'main'):
            self.assertEqual(getattr(wrapper.base, name).__module__, 'curated_base_resource')


if __name__ == '__main__':
    unittest.main()
