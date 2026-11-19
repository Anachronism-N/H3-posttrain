"""CPU-only refusal/identity/ownership tests. Never launch or signal a GPU job."""
import copy
import json
import fcntl
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import prepare_coarse3_n6_v1 as c
import coarse3_n6_driver_v1 as d


class GuardTests(unittest.TestCase):
    def test_unchanged_base_and_no_ms_import_pollution(self):
        before = {k for k in sys.modules if k.startswith(('ms_', 'motionscript_v2'))}
        base = c.base_module()
        c.module('test_storage_only', c.RT / 'curated_review_extension_resource_manager_v1.py')
        self.assertEqual(base.sha(c.RT / 'resource_manager.py'), c.BASE_SHA)
        self.assertEqual(before, {k for k in sys.modules if k.startswith(('ms_', 'motionscript_v2'))})

    def test_real_FD_and_kernel_FLOCK(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'lock'
            with p.open('w+b') as stream:
                st = os.fstat(stream.fileno())
                self.assertTrue(c.fd_inode_owned(os.getpid(), st.st_dev, st.st_ino))
                self.assertFalse(c.proc_lock_owned(os.getpid(), str(p)))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(c.proc_lock_owned(os.getpid(), str(p)))
                fcntl.flock(stream, fcntl.LOCK_UN)
                self.assertFalse(c.proc_lock_owned(os.getpid(), str(p)))

    def test_nanosecond_integer_lease_contract(self):
        with tempfile.TemporaryFile() as f:
            value = c.lease_value(os.fstat(f.fileno()))
            c.verify_lease_value(value)
            value['st_mtime_ns'] = float(value['st_mtime_ns'])
            with self.assertRaises(RuntimeError):
                c.verify_lease_value(value)

    def test_full_model_SHA_not_size_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'weight.safetensors'
            p.write_bytes(b'weight-a')
            identity = {'files': {'weight.safetensors': c.ref(p)}}
            with patch.object(c, 'verify_model_reference', return_value=(identity, None)), patch.object(c, 'module', return_value=SimpleNamespace(base=SimpleNamespace(), check_model=lambda cfg: None)):
                c.verify_local_model()
                before = p.stat()
                p.write_bytes(b'weight-b')
                os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))
                with self.assertRaisesRegex(RuntimeError, 'full local checkpoint'):
                    c.verify_local_model()

    def test_different_root_same_weight_not_independent(self):
        a = {'root': '/tmp/model-a', 'files': {'w.safetensors': {'sha256': 'a' * 64}}}
        b = {'root': '/tmp/model-b', 'files': {'another.safetensors': {'sha256': 'a' * 64}}}
        self.assertEqual(c.checkpoint(a), c.checkpoint(b))
        b['files']['another.safetensors']['sha256'] = 'b' * 64
        self.assertNotEqual(c.checkpoint(a), c.checkpoint(b))

    def test_missing_actual_safe_refuses_without_GPU_probe(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'fresh_holder', side_effect=AssertionError('GPU access forbidden')):
            p = Path(tmp) / 'transition.json'
            c.save_once(p, {'schema': 'owned_salvage_finite_review_transition_v1',
                            'admission_ref': {'path': str(Path(tmp) / 'not-actual-admission'), 'sha256': 'a' * 64}})
            with self.assertRaisesRegex(RuntimeError, 'exact original salvage queue'):
                c.prior_safe(c.ref(p))

    def test_previous_attempt_denies_without_erasing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OUT', Path(tmp)):
            root = Path(tmp) / 'source'
            claim = Path(tmp) / 'review-attempts/train-a.json'
            c.save_once(claim, {'ambiguous_dispatch_spent': True})
            with self.assertRaisesRegex(RuntimeError, 'no redispatch'):
                c.require_unreviewed({'sources': [{'root': str(root), 'source_id': 'a'}]})
            self.assertTrue(claim.exists())

    def test_exact_context_mutations(self):
        source = dict(candidate_ref={'path': '/tmp/recovered.json', 'sha256': 'a' * 64}, row_sha256='b' * 64)
        plan = dict(sources=[source], population_ref={'path': '/tmp/pop.json', 'sha256': 'c' * 64},
                    jobs_manifest_ref={'path': '/tmp/jobs.json', 'sha256': 'd' * 64}, source_job_refs=[{'path': '/tmp/job.json', 'sha256': 'e' * 64}])
        ctx = dict(schema='coarse3_filter_recovery_independent_review_v1', caller_pid=os.getpid(), out=str(c.OUT),
                   population_ref=plan['population_ref'], jobs_manifest_ref=plan['jobs_manifest_ref'],
                   source_job_refs=plan['source_job_refs'], candidate_ref=source['candidate_ref'],
                   input_row_sha256=source['row_sha256'], model_sha256=c.MODEL_SHA, tensor_parallel_size=1, maximum_review_calls=3)
        self.assertEqual(d.verify_context(ctx, plan), source)
        for key, value in [('caller_pid', 0), ('out', '/tmp/other'), ('candidate_ref', {}), ('model_sha256', 'x'),
                           ('input_row_sha256', 'x'), ('tensor_parallel_size', 2), ('maximum_review_calls', 4), ('source_job_refs', [])]:
            with self.subTest(key=key):
                changed = dict(ctx, **{key: value})
                with self.assertRaises(RuntimeError):
                    d.verify_context(changed, plan)

    def tracker(self):
        own = dict(pid=os.getpid(), uid=os.getuid(), ppid=1, start_ticks=100, state='R',
                   argv=['native-driver'], cgroup='same', CVD=['CUDA_VISIBLE_DEVICES=3'], exe='/usr/bin/python3.11')
        child = dict(own, pid=123456789, ppid=own['pid'], start_ticks=101, argv=['VLLM::EngineCore'], CVD=[])
        table = {own['pid']: own, child['pid']: child}
        base = SimpleNamespace(identity=lambda pid: copy.deepcopy(table[pid]), IDENTITY_KEYS=c.base_module().IDENTITY_KEYS)
        return d.ModelDescendants(base), table, child['pid']

    def test_native_empty_CVD_only_exact_title_and_parent(self):
        tracker, table, pid = self.tracker()
        tracker.verify(pid)
        table[pid]['argv'] = ['vllm-unknown-title']
        with self.assertRaises(RuntimeError):
            tracker.verify(pid)

    def test_unknown_parent_and_wrong_physical_CVD_denied(self):
        for key, value in [('ppid', 1), ('CVD', ['CUDA_VISIBLE_DEVICES=2']), ('uid', -1), ('exe', '/other'), ('cgroup', '/other'), ('start_ticks', 99), ('state', 'Z')]:
            with self.subTest(key=key):
                tracker, table, pid = self.tracker()
                table[pid][key] = value
                with self.assertRaises(RuntimeError):
                    tracker.verify(pid)

    def test_current_tracker_rejects_PID_reuse(self):
        tracker, table, pid = self.tracker()
        tracker.verify(pid)
        table[pid]['start_ticks'] += 1
        with self.assertRaisesRegex(RuntimeError, 'tracked model PID'):
            tracker.verify(pid)

    def test_TP1_rejects_native_worker_TP1(self):
        tracker, table, pid = self.tracker()
        table[pid]['argv'] = ['VLLM::Worker_TP1']
        with self.assertRaises(RuntimeError):
            tracker.verify(pid)

    def test_plan_mode_does_not_use_resources_or_build(self):
        plan = dict(source_sha256={}, population_ref={}, modelref={})
        argv = ['prepare', '--check-plan']
        with patch.object(c.sys if hasattr(c, 'sys') else sys, 'argv', argv), patch.object(c, 'source_plan', return_value=plan), patch.object(c, 'require_unreviewed'), patch.object(c, 'fresh_holder', side_effect=AssertionError('GPU forbidden')), patch.object(c, 'prior_safe', side_effect=AssertionError('transition forbidden')), patch.object(c, 'build', side_effect=AssertionError('build forbidden')):
            c.main()

    def parent_fixture(self):
        launch, queue = '/tmp/unit-memory-config', '/tmp/unit-memory-queue'
        parent = dict(pid=987654321, uid=os.getuid(), ppid=1, start_ticks=100, state='S',
                      argv=['/opt/venv_eval/bin/python', '-B', '-u', str(c.MANAGER), '--config', launch,
                            '--config-sha', 'a' * 64, '--manager-sha', c.BASE_SHA],
                      cgroup='own', CVD=[], exe='/usr/bin/python3.11')
        own = dict(parent, pid=os.getpid(), ppid=parent['pid'], argv=['driver'], CVD=['CUDA_VISIBLE_DEVICES=3'])
        cfg = dict(queue=queue, launch_path=launch, source_sha256={}, producer={}, plan={},
                   modelref={'model_identity_sha256': c.MODEL_SHA}, jobs_manifest_ref={})
        state = SimpleNamespace(st_dev=1, st_ino=42, st_uid=os.getuid(), st_mode=0o100600, st_size=0,
                                st_mtime_ns=d.time.time_ns(), st_ctime_ns=d.time.time_ns())
        refs = {launch: {'path': launch, 'sha256': 'a' * 64}}
        def reference(path):
            return refs.get(str(path), {'path': str(path), 'sha256': 'b' * 64})
        admission = dict(config_sha256='a' * 64, manager_sha256=c.BASE_SHA, manager_identity=parent,
                         source_sha256={}, producer={}, plan={}, model_identity_sha256=c.MODEL_SHA,
                         node='N6', gpus=[3], out=str(c.OUT), targetcount=3,
                         budget={'headroom': 160 * 2**30}, holders=[{'gpu': 3, 'uuid': 'unitGPU3'}])
        proof = dict(schema='coarse3_n6_actual_leases_held_v1', resource_admission_ref=reference(queue + '/admission.json'),
                     config_ref=reference(launch), manager_identity=copy.deepcopy(parent), node='N6', gpus=[3],
                     observed_unix_ns=1, leases=[{'gpu': 3, 'uuid': 'unitGPU3', 'path': '/tmp/od_reserve_3', 'stat': c.lease_value(state)}])
        documents = {queue + '/admission.json': admission, queue + '/actual_leases_held.json': proof}
        base = SimpleNamespace(identity=lambda pid: copy.deepcopy(parent if pid == parent['pid'] else own),
                               IDENTITY_KEYS=c.base_module().IDENTITY_KEYS, uuid_for=lambda gpu: 'unitGPU3', apps=lambda: [])
        return cfg, parent, own, state, admission, proof, documents, reference, base

    def call_parent(self, fixture, **kwargs):
        cfg, parent, own, state, admission, proof, documents, reference, base = fixture
        with patch.object(c, 'verify_config'), patch.object(c, 'ref', side_effect=reference), patch.object(c, 'checked', side_effect=lambda r: documents[r['path']]), patch.object(d.os, 'getppid', return_value=parent['pid']), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='3', COARSE3_N6_CONFIG=cfg['launch_path']), patch.object(Path, 'lstat', return_value=state), patch.object(c, 'fd_inode_owned', return_value=kwargs.pop('fd', True)), patch.object(c, 'proc_lock_owned', return_value=kwargs.pop('flock', True)):
            return d.verify_actual_parent(cfg, base, {'source_job_refs': []}, **kwargs)

    def test_live_heartbeat_not_historical_proof_age(self):
        fixture = self.parent_fixture()
        result = self.call_parent(fixture)
        self.assertEqual(result['gpus'], [3])
        self.assertEqual(result['tensor_parallel_size'], 1)
        self.assertTrue(result['current_lease_refs'])

    def test_parent_current_FD_and_kernel_FLOCK_required(self):
        for argument in ('fd', 'flock'):
            with self.subTest(argument=argument), self.assertRaisesRegex(RuntimeError, 'parent lease FD'):
                self.call_parent(self.parent_fixture(), **{argument: False})

    def test_parent_wrong_inode_stale_heartbeat_wrong_UUID_denied(self):
        for mutation in ('inode', 'stale', 'uuid', 'proof-node', 'physical-gpu', 'holders'):
            fixture = self.parent_fixture()
            cfg, parent, own, state, admission, proof, documents, reference, base = fixture
            if mutation == 'inode':
                state.st_ino += 1
            elif mutation == 'stale':
                state.st_mtime_ns -= 61 * 10**9
            elif mutation == 'uuid':
                base.uuid_for = lambda gpu: 'otherGPU'
            elif mutation == 'proof-node':
                proof['node'] = 'N5'
            elif mutation == 'physical-gpu':
                admission['gpus'] = [2]
            else:
                admission['holders'] = []
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.call_parent(fixture)

    def test_parent_exact_argv_identity_and_direct_parent_required(self):
        for key, value in [('argv', ['unrelated-manager']), ('uid', -1), ('start_ticks', 999), ('cgroup', 'other'), ('state', 'Z')]:
            fixture = self.parent_fixture()
            fixture[1][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.call_parent(fixture)

    def test_unknown_compute_is_never_accepted_or_signaled(self):
        fixture = self.parent_fixture()
        fixture[-1].apps = lambda: [{'uuid': 'unitGPU3', 'pid': 444}]
        with self.assertRaisesRegex(RuntimeError, 'unexpected GPU apps'):
            self.call_parent(fixture)
        tracker = SimpleNamespace(verify=lambda pid: (_ for _ in ()).throw(RuntimeError('unknown descendant')))
        with self.assertRaisesRegex(RuntimeError, 'unknown descendant'):
            self.call_parent(fixture, allow_compute=True, tracker=tracker)


    def stream_fixture(self, root):
        stream_path = root / 'stream.jsonl'
        sources, data = {}, b''
        watcher, _ = c.salvage_modules()
        for sid in ('a', 'b'):
            key = 'train/' + sid
            candidate_path = root / (sid + '.candidate.json')
            c.save_once(candidate_path, {'label': {'dance_style': 'test', 'timed_motion_script': sid}})
            notice = {'source_key': key, 'joint_candidate_ref': c.ref(candidate_path)}
            raw = (json.dumps(notice) + '\n').encode()
            range_ref = {'path': str(stream_path), 'byte_offset': len(data), 'byte_length': len(raw),
                         'sha256': c.hashlib.sha256(raw).hexdigest()}
            notice_path = root / (sid + '.notice.json')
            c.save_once(notice_path, notice)
            row_path = root / (sid + '.input.json')
            row = {'id': sid, 'split': 'train', 'label_path': str(candidate_path), 'label': c.load(candidate_path)['label'],
                   'review_provenance': {'original_notice_ref': c.ref(notice_path), 'original_notice_stream_ref': range_ref}}
            c.save_once(row_path, row)
            row = c.load(row_path)
            native_digest = lambda value: c.hashlib.sha256((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
            txn_path = root / (sid + '.transaction.json')
            c.save_once(txn_path, {'identity': {'source_key': key, 'notice_sha256': watcher.digest(notice),
                           'input_row_sha256': native_digest(row), 'semantic_label_sha256': native_digest(row['label']),
                           'label_result_ref': c.ref(candidate_path)}, 'input_ref': c.ref(row_path), 'status': 'pending'})
            sources[key] = {'transaction_ref': c.ref(txn_path), 'status': 'pending'}
            data += raw
        stream_path.write_bytes(data)
        st = stream_path.stat()
        state_path = root / 'state.json'
        c.save_once(state_path, {'input_stream': str(stream_path), 'sources': sources,
             'cursor': {'offset': len(data), 'identity': [st.st_dev, st.st_ino], 'observed_size': len(data),
                        'committed_prefix_sha256': c.hashlib.sha256(data).hexdigest()}})
        return stream_path, state_path, data

    def test_real_v5_prefix_allows_append_without_rehash_signing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, state, original = self.stream_fixture(root)
            with patch.object(c, 'STREAM', path), patch.object(c, 'RT', root):
                proof, prefix, lines = c.committed_stream_proof(c.ref(state))
                frozen = c.save_prefix_snapshot(proof, prefix)
                row = c.load(root / 'a.input.json')
                refs = c.preserved_references(row, lines, 'train/a')
                self.assertNotIn(str(path), refs)
                historical = copy.deepcopy(row['review_provenance']['original_notice_stream_ref'])
                with path.open('ab') as stream:
                    stream.write(b'{"unconsumed_append":true}\n')
                c.verify_stream_prefix(frozen)
                after, raw, current_lines = c.committed_stream_proof(c.ref(state))
                self.assertEqual(raw, original)
                self.assertEqual(after, proof)
                self.assertEqual(historical, row['review_provenance']['original_notice_stream_ref'])
                self.assertEqual(refs, c.preserved_references(row, current_lines, 'train/a'))

    def test_real_v5_midprefix_mutation_truncate_and_inode_replacement_denied(self):
        for mutation in ('middle', 'truncate', 'replace'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path, state, original = self.stream_fixture(root)
                with patch.object(c, 'STREAM', path), patch.object(c, 'RT', root):
                    proof, prefix, _ = c.committed_stream_proof(c.ref(state))
                    frozen = c.save_prefix_snapshot(proof, prefix)
                    if mutation == 'middle':
                        with path.open('r+b') as stream:
                            stream.seek(len(original) // 2)
                            stream.write(b'X')
                    elif mutation == 'truncate':
                        with path.open('r+b') as stream:
                            stream.truncate(len(original) - 1)
                    else:
                        other = root / 'replacement'
                        other.write_bytes(original)
                        other.replace(path)
                    with self.assertRaises((ValueError, RuntimeError)):
                        c.verify_stream_prefix(frozen)

    def test_prefix_source_mismatch_and_wrong_range_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, state_path, _ = self.stream_fixture(root)
            with patch.object(c, 'STREAM', path), patch.object(c, 'RT', root):
                proof, prefix, lines = c.committed_stream_proof(c.ref(state_path))
                row = c.load(root / 'a.input.json')
                with self.assertRaisesRegex(RuntimeError, 'another transaction'):
                    c.preserved_references(row, lines, 'train/b')
                changed = copy.deepcopy(row)
                changed['review_provenance']['original_notice_stream_ref']['byte_offset'] += 1
                with self.assertRaisesRegex(RuntimeError, 'byte-range'):
                    c.preserved_references(changed, lines, 'train/a')
                state = c.load(state_path)
                state['sources']['train/a'], state['sources']['train/b'] = state['sources']['train/b'], state['sources']['train/a']
                other_state = root / 'mismatched-state.json'
                c.save_once(other_state, state)
                with self.assertRaisesRegex(RuntimeError, 'notice/source/input/label'):
                    c.committed_stream_proof(c.ref(other_state))

    def test_snapshot_bytes_and_prefix_SHA_not_freshly_re_signed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, state, _ = self.stream_fixture(root)
            with patch.object(c, 'STREAM', path), patch.object(c, 'RT', root):
                proof, prefix, _ = c.committed_stream_proof(c.ref(state))
                frozen = c.save_prefix_snapshot(proof, prefix)
                snapshot = Path(frozen['prefix_snapshot_ref']['path'])
                snapshot.write_bytes(b'fake-history')
                with self.assertRaisesRegex(RuntimeError, 'snapshot changed'):
                    c.verify_stream_prefix(frozen)

    def test_salvage_prefix_import_has_no_ms_namespace_side_effect(self):
        before = {k for k in sys.modules if k.startswith(('ms_', 'motionscript_v2'))}
        c.salvage_modules()
        self.assertEqual(before, {k for k in sys.modules if k.startswith(('ms_', 'motionscript_v2'))})


    def test_per_call_exact_config_does_not_sweep_historical_files(self):
        launch = c.RT / 'coarse3_n6_launch_unit_memory.json'
        plan = {'jobs_manifest_ref': {}, 'population_ref': {}, 'source_sha256': {'/tmp/current-input': 'b' * 64}}
        reference = lambda path: {'path': str(path), 'sha256': 'a' * 64}
        cfg = dict(launch_path=str(launch), node='N6', gpus=[3], targetcount=3, input_count=222939, phase=c.PHASE,
                   out=str(c.OUT), cwd=str(c.RT), release_authorized=True, reuse_existing_results=True,
                   maximum_review_calls=3, tensor_parallel_size=1, semantic_approval=False, training_eligible=False,
                   env=c.driver_env(launch), argv=['/opt/venv_vllm/bin/python', '-B', '-u', str(c.DRIVER), '--launch', str(launch)],
                   producer=reference(c.DRIVER), resource_manager=reference(c.MANAGER), modelref=c.modelref(),
                   jobs_manifest_ref={}, population_ref={}, plan=reference('/tmp/plan'),
                   queue=str(c.BAR / 'queues/coarse3_n6_independent_unit'),
                   source_sha256={'/tmp/current-input': 'b' * 64, '/tmp/historical-score': 'c' * 64})
        with patch.object(c, 'ref', side_effect=reference), patch.object(c, 'checked', return_value=plan), patch.object(c, 'sha', side_effect=AssertionError('no historical stat/SHA sweep per call')), patch.object(c, 'verify_stream_prefix', side_effect=AssertionError('no prefix sweep per call')):
            c.verify_config(cfg, plan, verify_history=False)
            cfg['source_sha256']['/tmp/current-input'] = 'changed'
            with self.assertRaisesRegex(RuntimeError, 'source SHA binding missing'):
                c.verify_config(cfg, plan, verify_history=False)


if __name__ == '__main__':
    unittest.main()
