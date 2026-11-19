"""CPU/mock resource checks; no model, GPU query, signal, capture or manager main."""
import contextlib
import copy
import fcntl
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock

import prepare_curated_coarse_empty_n5_v1 as c
import curated_coarse_empty_n5_driver_v1 as d
import curated_coarse_empty_n5_resource_manager_v1 as m


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='empty-n5-resource-cpu-')
        self.root = Path(self.tmp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.tmp.cleanup)
        self.queue = self.root / 'queue'
        self.out = self.root / 'newout'
        self.queue.mkdir()
        self.out.mkdir()
        self.launch = self.root / 'launch.json'
        self.manager = self.root / 'manager.py'
        self.driver = self.root / 'driver.py'
        self.consumer = self.root / 'consumer.json'
        self.worker = self.root / 'worker.py'
        for p in (self.manager, self.driver, self.worker):
            p.write_text('cpu fixture')
        for name, value in [('QUEUE', self.queue), ('OUT', self.out), ('LAUNCH', self.launch),
                            ('MANAGER', self.manager), ('DRIVER', self.driver), ('CONSUMER', self.consumer),
                            ('PRIMARY', self.root / 'primary')]:
            self.stack.enter_context(mock.patch.object(c, name, value))
        self.modelref = {'path': str(self.root / 'model-reference.json'), 'sha256': 'fixture-content', 'model_identity_sha256': 'verified-model'}
        self.consumer_value = {'env': {'CUDA_VISIBLE_DEVICES': '4,5'}, 'modelref': self.modelref,
                               'plan': {'path': str(self.root / 'plan.jsonl'), 'sha256': 'fixture-plan'}}
        c.save_once(self.consumer, self.consumer_value)
        self.jobs = []
        self.hints = []
        for key, wid in c.TARGETS:
            p = self.out / 'jobs' / (key + '.json')
            job = {'source_key': key, 'id': key.split('/')[1], 'target_window_id': wid,
                   'primary_budget_root': str(c.PRIMARY), 'queue_root': str(c.PRIMARY)}
            c.save_once(p, job)
            self.jobs.append(job)
            self.hints.append({'source_key': key, 'job_ref': c.ref(p)})
        self.manifest = {'source_count': 3, 'jobs': self.hints}
        manifest_path = self.out / 'jobs.manifest.json'
        c.save_once(manifest_path, self.manifest)
        for name in ('protocol.json', 'cpu.prepare.evidence.json'):
            c.save_once(self.out / name, {'CPU_fixture': True})
        self.contract = {'consumer': self.consumer_value, 'jobs': self.jobs, 'manifest': self.manifest,
                         'manifest_ref': c.ref(manifest_path), 'protocol_ref': c.ref(self.out / 'protocol.json'),
                         'evidence_ref': c.ref(self.out / 'cpu.prepare.evidence.json'),
                         'bindings': {str(self.worker): c.sha(self.worker)}}
        self.cfg = {'node': 'N5', 'gpus': [0, 1], 'targetcount': 3, 'input_count': 222939,
                    'phase': c.PHASE, 'global_revision': c.REVISION, 'queue': str(self.queue), 'out': str(self.out),
                    'reuse_existing_results': True, 'release_authorized': True, 'cwd': str(c.RT),
                    'env': c.driver_env(self.consumer_value), 'producer': c.ref(self.driver),
                    'modelref': self.modelref, 'plan': self.consumer_value['plan'],
                    'consumer_config_ref': c.ref(self.consumer), 'jobs_manifest_ref': self.contract['manifest_ref'],
                    'protocol_ref': self.contract['protocol_ref'], 'cpu_evidence_ref': self.contract['evidence_ref'],
                    'source_sha256': self.contract['bindings'],
                    'argv': ['/opt/venv_vllm/bin/python', '-B', '-u', str(self.driver), '--launch', str(self.launch)]}
        c.save_once(self.launch, self.cfg)
        self.own = {'pid': 101, 'uid': os.getuid(), 'argv': ['own-driver'], 'start_ticks': 100,
                    'cgroup': 'fixture-cgroup', 'CVD': ['CUDA_VISIBLE_DEVICES=0,1'], 'exe': '/usr/bin/python3.11',
                    'ppid': 100, 'state': 'R'}
        self.parent = {**self.own, 'pid': 100, 'ppid': 99, 'start_ticks': 90, 'CVD': [],
                       'argv': ['/opt/venv_eval/bin/python', '-B', '-u', str(self.manager), '--config', str(self.launch),
                                '--config-sha', c.sha(self.launch), '--manager-sha', c.BASE_SHA]}
        self.admission = {'config_sha256': c.sha(self.launch), 'manager_sha256': c.BASE_SHA,
                          'source_sha256': self.cfg['source_sha256'], 'producer': self.cfg['producer'],
                          'plan': self.cfg['plan'], 'model_identity_sha256': 'verified-model', 'node': 'N5',
                          'gpus': [0, 1], 'out': str(self.out), 'targetcount': 3,
                          'budget': {'headroom': 160 * 2**30}, 'manager_identity': self.parent,
                          'holders': [{'gpu': g, 'uuid': 'fixture-uuid-%d' % g} for g in [0, 1]]}
        self.proof = {'schema': 'coarse_empty_n5_actual_leases_held_v1', 'node': 'N5', 'gpus': [0, 1],
                      'manager_identity': self.parent, 'config_ref': c.ref(self.launch),
                      'observed_unix_ns': time.time_ns() - 3600 * 10**9, 'leases': []}
        self.mapping = {}
        for gpu in [0, 1]:
            lease, lock = self.root / ('lease%d' % gpu), self.root / ('lock%d' % gpu)
            lease.touch()
            lock.touch()
            self.mapping['/tmp/od_reserve_%d' % gpu] = lease
            self.mapping['/tmp/ms12k_gpu_%d.lock' % gpu] = lock
            self.proof['leases'].append({'gpu': gpu, 'uuid': 'fixture-uuid-%d' % gpu,
                                        'path': str(lease), 'stat': c.lease_value(lease.stat())})
        self.rewrite_receipts()
        self.base = types.SimpleNamespace(IDENTITY_KEYS=('pid', 'uid', 'argv', 'start_ticks', 'cgroup', 'CVD', 'exe'),
                    identity=mock.Mock(side_effect=lambda p: copy.deepcopy(self.parent if p == 100 else self.own)),
                    apps=mock.Mock(return_value=[]), uuid_for=mock.Mock(side_effect=lambda g: 'fixture-uuid-%d' % g))
        self.stack.enter_context(mock.patch.object(d.os, 'getppid', return_value=100))
        self.stack.enter_context(mock.patch.object(d.os, 'getpid', return_value=101))
        self.stack.enter_context(mock.patch.object(d, 'Path', side_effect=lambda p: self.mapping.get(str(p), Path(p))))
        self.fd_mock = self.stack.enter_context(mock.patch.object(c, 'fd_inode_owned', return_value=True))
        self.kernel_mock = self.stack.enter_context(mock.patch.object(c, 'proc_lock_owned', return_value=True))
        self.stack.enter_context(mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1', 'COARSE_EMPTY_N5_CONFIG': str(self.launch)}))
        # A real accidental GPU query, process launch, pidfd or signal is a test failure.
        self.stack.enter_context(mock.patch.object(c.subprocess, 'check_output', side_effect=AssertionError('real external query forbidden')))
        self.stack.enter_context(mock.patch.object(c.subprocess, 'Popen', side_effect=AssertionError('real process/model forbidden')))
        self.stack.enter_context(mock.patch.object(c.os, 'pidfd_open', side_effect=AssertionError('real capture forbidden')))

    def rewrite_receipts(self):
        path = self.queue / 'admission.json'
        path.write_text(c.json.dumps(self.admission))
        self.proof['resource_admission_ref'] = c.ref(path)
        (self.queue / 'actual_leases_held.json').write_text(c.json.dumps(self.proof))

    def verify(self, **kwargs):
        return d.verify_actual_parent(self.cfg, self.base, self.contract, **kwargs)

    def test_live_leases_accept_old_historical_proof(self):
        result = self.verify()
        self.assertEqual(len(result['current_lease_refs']), 2)
        self.assertEqual(result['source_job_refs'], [h['job_ref'] for h in self.hints])
        self.assertEqual(self.base.apps.call_count, 1)

    def test_wrong_direct_parent_identity(self):
        self.parent['start_ticks'] += 1
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_wrong_parent_argv(self):
        self.parent['argv'] = ['not-the-owned-manager']
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_wrong_config_sha(self):
        self.admission['config_sha256'] = 'changed'
        self.rewrite_receipts()
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_changed_wrapper_or_source_sha(self):
        self.worker.write_text('changed source')
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_missing_second_lease(self):
        self.proof['leases'] = self.proof['leases'][:1]
        self.rewrite_receipts()
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_missing_actual_parent_fd(self):
        self.fd_mock.return_value = False
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_wrong_kernel_flock_owner(self):
        self.kernel_mock.return_value = False
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_expired_heartbeat(self):
        path = self.mapping['/tmp/od_reserve_1']
        os.utime(path, ns=(time.time_ns(), time.time_ns() - 61 * 10**9))
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_wrong_lease_inode(self):
        self.proof['leases'][1]['stat']['st_ino'] += 1
        self.rewrite_receipts()
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_nonempty_lease(self):
        self.mapping['/tmp/od_reserve_1'].write_text('not a lease')
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_unknown_gpu_app_before_model(self):
        self.base.apps.return_value = [{'uuid': 'fixture-uuid-0', 'pid': 555}]
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_unknown_gpu_app_after_model(self):
        self.base.apps.return_value = [{'uuid': 'fixture-uuid-0', 'pid': 555}]
        tracker = types.SimpleNamespace(verify=mock.Mock(side_effect=RuntimeError('unknown app')))
        with self.assertRaises(RuntimeError):
            self.verify(tracker=tracker, allow_compute=True)
        tracker.verify.assert_called_once_with(555)

    def test_only_known_model_app_after_load(self):
        self.base.apps.return_value = [{'uuid': 'fixture-uuid-0', 'pid': 555}]
        tracker = types.SimpleNamespace(verify=mock.Mock())
        self.verify(tracker=tracker, allow_compute=True)
        tracker.verify.assert_called_once_with(555)

    def test_under_160gib(self):
        self.admission['budget']['headroom'] = 160 * 2**30 - 1
        self.rewrite_receipts()
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_wrong_cuda_devices(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,0'}):
            with self.assertRaises(RuntimeError):
                self.verify()

    def test_changed_physical_uuid(self):
        self.base.uuid_for.side_effect = lambda g: 'wrong-uuid'
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_reserved_canonical_budget_across_edition(self):
        path = c.PRIMARY / 'attempt-budget' / (self.jobs[0]['source_key'] + '.json')
        c.save_once(path, {'older_edition': True})
        with self.assertRaises(RuntimeError):
            c.require_unreserved(self.jobs)

    def test_dangling_budget_symlink_still_reserved(self):
        path = c.PRIMARY / 'attempt-budget' / (self.jobs[0]['source_key'] + '.json')
        path.parent.mkdir(parents=True)
        path.symlink_to(self.root / 'missing')
        with self.assertRaises(RuntimeError):
            c.require_unreserved(self.jobs)

    def test_half_job_manifest_callback_rejected(self):
        runtime = object.__new__(d.FiniteRuntime)
        runtime.contract = self.contract
        with self.assertRaises(RuntimeError):
            runtime.verify_current_admission([self.hints[0]['job_ref']])

    def test_reordered_manifest_callback_rejected(self):
        runtime = object.__new__(d.FiniteRuntime)
        runtime.contract = self.contract
        with self.assertRaises(RuntimeError):
            runtime.verify_current_admission([h['job_ref'] for h in reversed(self.hints)])

    def make_six_file_result(self, failed=False):
        hint, job = self.hints[0], self.jobs[0]
        root = self.out / 'revisions' / c.REVISION / job['source_key']
        attempt = root / 'windows' / job['target_window_id'] / 'attempt-000001'
        request = {'sampling': {'max_tokens': 321}, 'json_schema': {'type': 'object'}, 'messages': []}
        c.save_once(attempt / 'request.json', request)
        c.save_once(self.out / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json', request)
        binding = {'current_job_ref': hint['job_ref']}
        c.save_once(attempt / 'binding.json', binding)
        (attempt / 'response.raw.txt').write_text('mock native model completion')
        digest = lambda v: c.hashlib.sha256(c.json.dumps(v, sort_keys=True).encode()).hexdigest()
        if not failed:
            c.save_once(attempt / 'generation.json', {'model_call_performed': True,
                'request_sha256': c.sha(attempt / 'request.json'), 'raw_response_sha256': c.sha(attempt / 'response.raw.txt'),
                'model_identity_sha256': 'verified-model', 'max_tokens': 321,
                'structured_output_schema_sha256': digest(request['json_schema'])})
            c.save_once(attempt / 'response.json', {'mock_response': True})
        c.save_once(attempt / 'final.json', {'status': 'failed' if failed else 'candidate'})
        names = ('request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json')
        receipt = {'status': 'failed' if failed else 'candidate', 'binding': binding,
                   'files_sha256': {n: c.sha(attempt / n) for n in names if (attempt / n).exists()}}
        c.save_once(attempt / 'receipt.json', receipt)
        original = {'window_observations': [{'window': {'window_id': 'untargeted'}, 'status': 'old'},
                                            {'window': {'window_id': job['target_window_id']}, 'status': 'abstain'}]}
        old_joint = {'style_result': {'style': 'same'}, 'dance_style_label': 'same'}
        visual = {'window_observations': copy.deepcopy(original['window_observations'])}
        c.save_once(root / 'visual_result.json', visual)
        result = {'new_text_model_requests': 0, 'new_visual_requests': 1, 'original_job_ref': hint['job_ref'],
                  'new_label_review_required': True, 'training_eligible': False, 'style_label_changed': False,
                  'visual_result_ref': c.ref(root / 'visual_result.json'), **old_joint}
        c.save_once(root / 'result.json', result)
        terminal = {'joint_result_ref': c.ref(root / 'result.json')}
        c.save_once(self.out / 'joint/review-required' / (job['source_key'] + '.json'),
                    {'result_ref': terminal['joint_result_ref'], 'new_label_review_required': True})
        worker = types.SimpleNamespace(verify_job=mock.Mock(return_value=(original, old_joint, {}, {})),
                                       q=types.SimpleNamespace(digest=digest))
        self.stack.enter_context(mock.patch.object(c, 'worker_module', return_value=worker))
        return hint, terminal, attempt, receipt

    def test_actual_six_file_candidate_generation_binding(self):
        hint, terminal, attempt, receipt = self.make_six_file_result()
        self.assertEqual(len(receipt['files_sha256']), 6)
        m.verify_result(self.cfg, self.out, hint, terminal)

    def test_failed_dispatch_missing_generation_preserved(self):
        hint, terminal, attempt, receipt = self.make_six_file_result(failed=True)
        self.assertLess(len(receipt['files_sha256']), 6)
        m.verify_result(self.cfg, self.out, hint, terminal)

    def test_candidate_missing_generation_not_accepted(self):
        hint, terminal, attempt, receipt = self.make_six_file_result(failed=True)
        receipt['status'] = 'candidate'
        (attempt / 'receipt.json').write_text(c.json.dumps(receipt))
        with self.assertRaises(RuntimeError):
            m.verify_result(self.cfg, self.out, hint, terminal)

    def test_generation_wrong_model_refused_even_with_fresh_file_sha(self):
        hint, terminal, attempt, receipt = self.make_six_file_result()
        generation = c.load(attempt / 'generation.json')
        generation['model_identity_sha256'] = 'old-or-different-model'
        (attempt / 'generation.json').write_text(c.json.dumps(generation))
        receipt['files_sha256']['generation.json'] = c.sha(attempt / 'generation.json')
        (attempt / 'receipt.json').write_text(c.json.dumps(receipt))
        with self.assertRaises(RuntimeError):
            m.verify_result(self.cfg, self.out, hint, terminal)

    def write_partial_stage(self):
        refs = []
        for hint, job in zip(self.hints, self.jobs):
            p = self.out / 'worker-terminals' / (hint['source_key'] + '.json')
            c.save_once(p, {'source_key': hint['source_key'], 'id': job['id'], 'global_revision': c.REVISION,
                           'original_job_ref': hint['job_ref'], 'status': 'failed_preserved_no_redispatch',
                           'semantic_approval': False, 'training_eligible': False})
            refs.append(c.ref(p))
        report = {'source_count': 3, 'terminal_count': 3, 'terminals': refs, 'visual_model_calls': 0,
                  'text_model_calls': 0, 'global_revision': c.REVISION, 'jobs_manifest_ref': self.contract['manifest_ref'],
                  'source_job_refs': [h['job_ref'] for h in self.hints], 'evidence_refs': []}
        c.save_once(self.out / 'finite.stage.summary.json', report)
        return report

    def test_partial_three_terminals_do_not_claim_joint_complete(self):
        self.write_partial_stage()
        summary = m.summaries(self.cfg, self.out)
        self.assertEqual(len(summary), 1)
        self.assertIs(next(iter(summary.values()))['semantic_approval'], False)

    def test_half_stage_terminal_count_refused(self):
        report = self.write_partial_stage()
        report['terminal_count'] = 2
        (self.out / 'finite.stage.summary.json').write_text(c.json.dumps(report))
        with self.assertRaises(RuntimeError):
            m.summaries(self.cfg, self.out)

    def test_stage_text_model_calls_refused(self):
        report = self.write_partial_stage()
        report['text_model_calls'] = 1
        (self.out / 'finite.stage.summary.json').write_text(c.json.dumps(report))
        with self.assertRaises(RuntimeError):
            m.summaries(self.cfg, self.out)

    def test_source_job_ref_stage_mismatch_refused(self):
        report = self.write_partial_stage()
        report['source_job_refs'] = report['source_job_refs'][:2]
        (self.out / 'finite.stage.summary.json').write_text(c.json.dumps(report))
        with self.assertRaises(RuntimeError):
            m.summaries(self.cfg, self.out)

    def test_flags_only_not_authority(self):
        self.proof = {'launch_authorized': True, 'resource_manager_verified_and_leases_held': True,
                      'fresh_pair_safety_verified': True}
        self.rewrite_receipts()
        with self.assertRaises(RuntimeError):
            self.verify()


class NativeKernelAndBudgetTests(unittest.TestCase):
    def test_real_cpu_flock_and_open_descriptor(self):
        with tempfile.TemporaryDirectory(prefix='empty-n5-kernel-cpu-') as tmp:
            path = Path(tmp) / 'empty.lock'
            with path.open('x+b') as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                st = path.stat()
                self.assertTrue(c.fd_inode_owned(os.getpid(), st.st_dev, st.st_ino))
                self.assertTrue(c.proc_lock_owned(os.getpid(), str(path)))
                self.assertFalse(c.proc_lock_owned(os.getppid(), str(path)))
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                self.assertFalse(c.proc_lock_owned(os.getpid(), str(path)))

    def test_model_descendant_rejects_foreign_tree(self):
        own = {'pid': 10, 'uid': os.getuid(), 'ppid': 1, 'start_ticks': 10, 'cgroup': 'same',
               'CVD': ['CUDA_VISIBLE_DEVICES=0,1'], 'exe': 'python', 'argv': ['driver'], 'state': 'R'}
        app = {**own, 'pid': 30, 'ppid': 1, 'start_ticks': 20, 'argv': ['vLLM::Worker_TP0']}
        base = types.SimpleNamespace(IDENTITY_KEYS=('pid', 'uid', 'argv', 'start_ticks', 'cgroup', 'CVD', 'exe'),
                                     identity=lambda pid: own if pid == 10 else app)
        with mock.patch.object(d.os, 'getpid', return_value=10):
            tracker = d.ModelDescendants(base)
        with self.assertRaises(RuntimeError):
            tracker.verify(30)

    def test_native_vllm_empty_reported_environment_with_owned_parent_tree(self):
        own = {'pid': 10, 'uid': os.getuid(), 'ppid': 1, 'start_ticks': 10, 'cgroup': 'same',
               'CVD': ['CUDA_VISIBLE_DEVICES=0,1'], 'exe': 'python', 'argv': ['driver'], 'state': 'R'}
        engine = {**own, 'pid': 20, 'ppid': 10, 'start_ticks': 20, 'argv': ['VLLM::EngineCore'], 'CVD': []}
        app = {**own, 'pid': 30, 'ppid': 20, 'start_ticks': 30, 'argv': ['VLLM::Worker_TP0'], 'CVD': []}
        values = {10: own, 20: engine, 30: app}
        base = types.SimpleNamespace(IDENTITY_KEYS=('pid', 'uid', 'argv', 'start_ticks', 'cgroup', 'CVD', 'exe'),
                                     identity=lambda pid: copy.deepcopy(values[pid]))
        with mock.patch.object(d.os, 'getpid', return_value=10):
            tracker = d.ModelDescendants(base)
        tracker.verify(30)
        app['CVD'] = ['CUDA_VISIBLE_DEVICES=7']
        with self.assertRaises(RuntimeError):
            tracker.verify(30)
        app['CVD'], app['ppid'] = [], 1
        with self.assertRaises(RuntimeError):
            tracker.verify(30)

    def test_model_descendant_rejects_reused_pid(self):
        own = {'pid': 10, 'uid': os.getuid(), 'ppid': 1, 'start_ticks': 10, 'cgroup': 'same',
               'CVD': ['CUDA_VISIBLE_DEVICES=0,1'], 'exe': 'python', 'argv': ['driver'], 'state': 'R'}
        app = {**own, 'pid': 30, 'ppid': 10, 'start_ticks': 20, 'argv': ['vLLM::Worker_TP0']}
        base = types.SimpleNamespace(IDENTITY_KEYS=('pid', 'uid', 'argv', 'start_ticks', 'cgroup', 'CVD', 'exe'),
                                     identity=lambda pid: copy.deepcopy(own if pid == 10 else app))
        with mock.patch.object(d.os, 'getpid', return_value=10):
            tracker = d.ModelDescendants(base)
        tracker.verify(30)
        app['start_ticks'] = 21
        with self.assertRaises(RuntimeError):
            tracker.verify(30)

    def test_failed_load_never_retried(self):
        runtime = object.__new__(d.FiniteRuntime)
        runtime.contract = {'manifest': {'jobs': []}}
        runtime.loaded, runtime.model_started = None, True
        runtime.verify_current_admission = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'never retry'):
            runtime()

    def test_fourth_visual_call_refused_without_model(self):
        runtime = types.SimpleNamespace(contract={'manifest': {'jobs': [{'job_ref': {'path': 'job', 'sha256': 'x'}}]}},
                    verify_current_admission=mock.Mock(), active_job_ref={'path': 'job', 'sha256': 'x'},
                    visual_calls=3, seen_jobs=set())
        model = types.SimpleNamespace(_ms_bar_model_sha256='same', chat=mock.Mock())
        guard = d.FiniteModel(model, runtime)
        with self.assertRaises(RuntimeError):
            guard.chat([[]], [object()])
        model.chat.assert_not_called()

    def test_duplicate_source_dispatch_refused_without_model(self):
        reference = {'path': 'job', 'sha256': 'x'}
        runtime = types.SimpleNamespace(contract={'manifest': {'jobs': [{'job_ref': reference}]}},
                    verify_current_admission=mock.Mock(), active_job_ref=reference, visual_calls=1, seen_jobs={'job'})
        model = types.SimpleNamespace(_ms_bar_model_sha256='same', chat=mock.Mock())
        with self.assertRaises(RuntimeError):
            d.FiniteModel(model, runtime).chat([[]], [object()])
        model.chat.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
