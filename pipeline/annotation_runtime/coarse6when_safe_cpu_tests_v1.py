"""Synthetic CPU fixtures; never opens actual GPU leases or launches a worker."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import coarse6when_safe_common_v1 as c
import coarse6when_safe_driver_v1 as driver
import coarse6when_safe_manager_v1 as manager


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return c.ref(path)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='coarse6when_safe_CPU_fixture_')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def predecessor(self):
        queue = self.root / 'queues/curated_collaborative20000_n3g01_v1'
        out = self.root / 'curated222939_collaborative_v1'
        selected = [{'id': 'fixture%04d' % i, 'split': 'train', 'batch_id': 'fixture', 'source_ordinal': 20000+i} for i in range(1000)]
        selected_ref = write(self.root / 'selected.json', selected)
        source = write(self.root / 'fixture_source.json', {'fixture': True})
        config = {'fixture_params': True}
        params = hashlib.sha256((json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
        previous = {'queue': str(queue), 'out': str(out), 'gpus': [0,1], 'node': 'N3',
                    'modelref': {**source, 'model_identity_sha256': 'fixture-model'}, 'source_sha256': {},
                    'producer': source, 'plan': source, 'selected_sources': selected_ref,
                    'manifest_summary': source, 'consumer_params_sha256': params}
        write(queue / 'admission.json', {'config_sha256': c.PREVIOUS_SHA, 'model_identity_sha256': 'fixture-model',
              'source_sha256': {}, 'producer': source, 'out': str(out), 'node': 'N3', 'gpus': [0,1]})
        for row in selected:
            write(out / 'sources/train' / row['id'] / 'consumer.terminal.json',
                  {**row, 'params_sha256': params, 'status': 'incomplete_candidate'})
        marker = out / 'runs/v0-fixture/completed.json'
        write(marker, {'selected_total': 1000, 'queue_sources_consumed': 1000})
        write(marker.with_name('protocol.json'), {'consumer_wrapper': source, 'configuration': config, 'manifest_summary': source})
        exit_value = {'exit_code': 0, 'targetcount': 1000, 'input_count': 222939, 'source_unchanged': True,
                      'producer': source, 'plan': source, 'summaries': {str(marker): {'summary': {
                      'selected_source_count': 1000, 'processed_source_count': 1000, 'total_source_count': 222939,
                      'plan_sha256': source['sha256']}, 'selection': {'selected_ids': [r['id'] for r in selected]}}}}
        write(queue / 'worker_exit.json', exit_value)
        safe = {k: True for k in ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')}
        write(queue / 'resources_safe.json', safe)
        for gpu in (0,1):
            write(queue / ('holder_restored_gpu%d.json' % gpu), {'CPU_fixture_only': True})
        return previous, queue, marker

    def test_complete1000_but_partial_labels_is_not_semantic_approval(self):
        previous, queue, marker = self.predecessor()
        with patch.object(c, 'ROOT', self.root):
            value = c.previous_completed(previous)
        self.assertEqual(value['terminal_count'], 1000)
        self.assertNotIn('semantic_approval', value)

    def test_previous_partial_completion_forbids_launch(self):
        previous, queue, marker = self.predecessor()
        write(marker, {'selected_total': 1000, 'queue_sources_consumed': 999})
        with patch.object(c, 'ROOT', self.root), self.assertRaisesRegex(RuntimeError, 'partial'):
            c.previous_completed(previous)

    def test_previous_failure_forbids_success(self):
        previous, queue, marker = self.predecessor()
        write(queue / 'failure.json', {'CPU_fixture_failure': True})
        with patch.object(c, 'ROOT', self.root), self.assertRaisesRegex(RuntimeError, 'failure'):
            c.previous_completed(previous)

    def test_one_nanosecond_change_rejected(self):
        path = self.root / 'lease'
        path.touch()
        os.utime(path, ns=(1, 1))
        base = c.base_module()
        expected = c.lease_value(base.lease_snapshot(path))
        os.utime(path, ns=(1, 2))
        with self.assertRaisesRegex(RuntimeError, 'nanosecond'):
            c.unchanged_lease(base, path, expected)

    def test_boolean_or_float_ns_rejected(self):
        value = {k: 1 for k in c.LEASE_KEYS}
        for bad in (True, 1.0):
            value['st_mtime_ns'] = bad
            with self.assertRaisesRegex(RuntimeError, 'integer'):
                c.verify_lease_value(value)

    def test_new_output_cannot_reset_canonical_once_budget(self):
        job = {'source_key': 'train/fixture'}
        primary = self.root / 'canonical_primary'
        with patch.object(c, 'PRIMARY', primary):
            c.require_unreserved([job])
            path = primary / 'attempt-budget/train/fixture.json'
            write(path, {'CPU_fixture': True, 'status': 'reserved'})
            for out in ('independent-v1', 'independent-v2'):
                (self.root / out).mkdir()
                with self.assertRaisesRegex(RuntimeError, 'no call/redispatch'):
                    c.require_unreserved([job])
            self.assertTrue(path.exists())

    def actual_parent_fixture(self):
        queue, launch, out, manager_path = self.root / 'queue', self.root / 'launch.json', self.root / 'out', self.root / 'manager.py'
        identity = {'pid': os.getppid(), 'uid': os.getuid(), 'argv': [], 'start_ticks': 5,
                    'cgroup': 'CPU-fixture-cgroup', 'CVD': [], 'exe': '/fixture/python'}
        source = write(self.root / 'source.json', {'fixture_only': True})
        cfg = {'queue': str(queue), 'out': str(out), 'node': 'N3', 'gpus': [0,1], 'producer': source, 'plan': source,
               'modelref': {'model_identity_sha256': 'fixture-model'}, 'source_sha256': {source['path']: source['sha256']}}
        write(launch, cfg)
        identity['argv'] = ['/opt/venv_eval/bin/python', '-B', '-u', str(manager_path), '--config', str(launch), '--config-sha', c.sha(launch), '--manager-sha', c.BASE_SHA]
        admission = {'config_sha256': c.sha(launch), 'manager_sha256': c.BASE_SHA, 'source_sha256': cfg['source_sha256'],
                     'producer': source, 'plan': source, 'model_identity_sha256': 'fixture-model', 'node': 'N3', 'gpus': [0,1],
                     'out': str(out), 'targetcount': 2, 'manager_identity': identity, 'holders': [{'uuid': 'CPUfixture0'}, {'uuid': 'CPUfixture1'}],
                     'budget': {'headroom': 160 * 2**30}}
        resource = write(queue / 'admission.json', admission)
        lease = {k: 1 for k in c.LEASE_KEYS}
        proof = {'schema': 'coarse6_actual_manager_leases_held_v1', 'resource_admission_ref': resource, 'config_ref': c.ref(launch),
                 'node': 'N3', 'gpus': [0,1], 'manager_identity': identity, 'observed_unix_ns': c.time.time_ns(),
                 'leases': [{'gpu': gpu, 'path': '/tmp/od_reserve_%d' % gpu, 'stat': lease} for gpu in (0,1)]}
        write(queue / 'actual_leases_held.json', proof)
        base = types.SimpleNamespace(IDENTITY_KEYS=c.base_module().IDENTITY_KEYS, identity=lambda pid: identity,
                                     check_model=lambda cfg: None, apps=lambda: [])
        fake_stat = types.SimpleNamespace(st_mode=c.stat.S_IFREG | 0o600, st_uid=os.getuid(), st_size=0, st_dev=1, st_ino=1, st_mtime_ns=c.time.time_ns())
        return cfg, base, queue, launch, out, manager_path, fake_stat, resource

    def verify_actual(self, fixture, owned=True):
        cfg, base, queue, launch, out, manager_path, fake_stat, resource = fixture
        with patch.multiple(c, QUEUE=queue, LAUNCH=launch, OUT=out, MANAGER=manager_path), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1', 'COARSE6_MANAGER_CONFIG': str(launch)}), patch.object(Path, 'lstat', return_value=fake_stat), patch.object(c, 'proc_lock_owned', return_value=owned), patch.object(c, 'fd_inode_owned', return_value=owned):
            return driver.verify_actual_parent(cfg, base)

    def test_actual_ref_sha_is_verified_and_returned(self):
        fixture = self.actual_parent_fixture()
        resource, proof = self.verify_actual(fixture)
        self.assertEqual(resource, fixture[-1])
        self.assertEqual(resource['sha256'], c.sha(resource['path']))
        changed = c.load(resource['path'])
        changed['targetcount'] = 1
        write(Path(resource['path']), changed)
        with self.assertRaisesRegex(RuntimeError, 'evidence'):
            self.verify_actual(fixture)

    def test_fake_true_booleans_without_actual_proof_are_not_authority(self):
        fixture = self.actual_parent_fixture()
        fixture[0].update(launch_authorized=True, fresh_pair_safety_verified=True, resource_manager_verified_and_leases_held=True)
        (fixture[2] / 'actual_leases_held.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify_actual(fixture)

    def test_actual_parent_without_all_owned_fds_and_locks_rejected(self):
        fixture = self.actual_parent_fixture()
        with self.assertRaisesRegex(RuntimeError, 'actually hold'):
            self.verify_actual(fixture, owned=False)

    def test_proc_lock_and_open_descriptor_are_real_kernel_evidence(self):
        path = self.root / 'CPU_fixture_real_lock'
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            c.fcntl.flock(fd, c.fcntl.LOCK_EX | c.fcntl.LOCK_NB)
            s = os.fstat(fd)
            self.assertTrue(c.proc_lock_owned(os.getpid(), path))
            self.assertTrue(c.fd_inode_owned(os.getpid(), s.st_dev, s.st_ino))
            self.assertFalse(c.proc_lock_owned(-12345, path))
        finally:
            os.close(fd)

    def test_independent_manager_has_no_other_gpu_authorization(self):
        self.assertEqual(manager.base.AUTHORIZED_PAIRS, {'N3': ((0,1),)})
        self.assertIsNot(manager.base, c.base_module())


if __name__ == '__main__':
    unittest.main(verbosity=2)
