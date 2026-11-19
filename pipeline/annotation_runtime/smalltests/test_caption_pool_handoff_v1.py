"""CPU-only metadata/handoff tests; no real signal, GPU, source copy or model."""
import copy
from contextlib import contextmanager
import importlib.util
import json
import io
from contextlib import redirect_stdout
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SOURCE_RT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dependencies = tempfile.TemporaryDirectory(prefix='pool-handoff-deps-')
        cls.dep = Path(cls.dependencies.name)
        names = ('caption_pool_handoff_v1.py', 'caption_pool_live_manager_v1.py', 'caption_ready_pool_v1.py',
                 'caption_local_rebalance_transition_v1.py', 'caption_light_to_local_transition_v1.py',
                 'caption_light_to_local_transition_v2.py', 'caption_local_live_manager_v1.py', 'resource_manager.py')
        for name in names:
            shutil.copyfile(SOURCE_RT / name, cls.dep / name)
        cls.base = module('handoff_test_base', cls.dep / 'resource_manager.py')
        cls.pool = module('caption_ready_pool_v1', cls.dep / 'caption_ready_pool_v1.py')
        old = types.ModuleType('caption_local_live_manager_v1')
        old.base = cls.base
        old.BASE_SHA = cls.base.sha(cls.dep / 'resource_manager.py')
        old.SELF = cls.dep / 'caption_local_live_manager_v1.py'
        old.BAR = cls.dep.parent
        old.HOSTS = {'N4': 'host-n4', 'N3': 'host-n3'}
        old.MODEL = 'never-loaded'
        old.h = types.SimpleNamespace(subprocess=subprocess)
        old.ref = cls.pool.ref
        old.selected = lambda cfg: ([r['id'] for r in cls.pool.load(cfg['plan']['path'])],
                                     len(cls.pool.load(cfg['plan']['path'])))
        old.local = types.SimpleNamespace(light=types.SimpleNamespace(need=cls.pool.need, load=cls.pool.load,
                                          write_once=cls.pool.exclusive, FLAGS=cls.pool.FLAGS))
        # Native GPU validation/gate are replaced at their hardware boundary below.
        old.validate = lambda cfg: None
        old.gate = lambda cfg, before=False: None
        sys.modules[old.__name__] = old
        cls.native = module('caption_light_to_local_transition_v1', cls.dep / 'caption_light_to_local_transition_v1.py')
        module('caption_light_to_local_transition_v2', cls.dep / 'caption_light_to_local_transition_v2.py')
        cls.rebalance = module('caption_local_rebalance_transition_v1', cls.dep / 'caption_local_rebalance_transition_v1.py')
        cls.manager = module('caption_pool_live_manager_v1', cls.dep / 'caption_pool_live_manager_v1.py')
        cls.H = module('caption_pool_handoff_v1', cls.dep / 'caption_pool_handoff_v1.py')

    @classmethod
    def tearDownClass(cls):
        cls.dependencies.cleanup()

    def setUp(self):
        capture = redirect_stdout(io.StringIO())
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)
        self.tmp = tempfile.TemporaryDirectory(prefix='pool-handoff-case-')
        self.addCleanup(self.tmp.cleanup)
        self.bar = Path(self.tmp.name)
        self.rt = self.bar / 'runtime'
        self.rt.mkdir()
        for source in self.dep.iterdir():
            if source.is_file():
                (self.rt / source.name).symlink_to(source)
        self.h = self.H
        self.specs = copy.deepcopy(self.h.SPECS)
        self.olds = []
        self.all_rows = []
        for i, spec in enumerate(self.specs):
            rows = [dict(id='lane%d-source%d' % (i, j), split='train', batch_id='train-b0', source_ordinal=i * 10 + j)
                    for j in range(4)]
            spec['count'] = len(rows)
            path = self.rt / (spec['old'] + '_launch.json')
            pp = self.rt / (spec['old'] + '_selected.json')
            self.pool.exclusive(pp, rows)
            root = self.bar / spec['old']
            root.mkdir()
            queue = self.bar / 'queues' / spec['old']
            queue.mkdir(parents=True)
            cfg = dict(config_path=str(path), cwd=str(self.rt), phase=spec['old'], node=spec['node'],
                       gpus=spec['gpus'], targetcount=len(rows), input_count=len(rows), plan=self.pool.ref(pp),
                       producer=self.pool.ref(self.rt / 'caption_local_live_manager_v1.py'),
                       resource_manager_ref=self.pool.ref(self.rt / 'caption_local_live_manager_v1.py'),
                       argv=[self.native.PYTHON, '-B', str(self.dep / 'caption_local_live_manager_v1.py'), 'worker', '--config', str(path)],
                       release_authorized=True, env={'CUDA_VISIBLE_DEVICES': ','.join(map(str, spec['gpus']))},
                       draft_out=str(root), out=str(self.bar / (spec['old'] + '_execution')), queue=str(queue),
                       source_root=str(self.bar / 'curated222939_collaborative_v3'), modelref={'model_identity_sha256': 'synthetic'},
                       source_sha256={str(self.dep / 'resource_manager.py'): self.manager.BASE_SHA,
                                      str(self.dep / 'caption_local_live_manager_v1.py'): self.pool.sha(self.dep / 'caption_local_live_manager_v1.py')})
            self.pool.exclusive(path, cfg)
            spec['sha'] = self.pool.sha(path)
            self.pool.exclusive(root / 'progress.json', dict(completed_source_keys=[[rows[0]['id'], 'train']], processed=1, selected=4))
            for j in (0, 1):
                name = 'batch-lane%d-%d' % (i, j)
                self.pool.exclusive(root / 'budgets' / name / 'claim.json', dict(metas=[rows[j]], reserved_sources=1))
                if j == 0:
                    self.pool.exclusive(root / (name + '.committed.json'), dict(batch_id=name, reliable_writeback=True,
                                        archive_path=str(root / (name + '.tar'))))
            (root / 'preserved.tmp').write_bytes(b'old-cache-do-not-touch')
            self.olds.append(cfg)
            self.all_rows.extend(rows)
        self.patches = []
        frozen = {name: self.pool.sha(self.rt / name) for name in self.h.FROZEN}
        for obj, key, value in ((self.h, 'RT', self.rt), (self.h, 'BAR', self.bar),
                                (self.h, 'POOL_ROOT', self.bar / 'caption_two_stage_dynamic_pool_v1'),
                                (self.h, 'SPECS', self.specs), (self.h, 'FROZEN', frozen),
                                (self.native, 'RT', self.rt), (self.native.m, 'BAR', self.bar)):
            p = patch.object(obj, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.node = 'N4'
        self.hardware_validated = []
        self.park_calls = []
        self.capacity_value = dict(memory={'headroom': 161 * 2**30}, local_disk_free_bytes=9 * 2**30)
        self.handoff_lock = patch.object(self.native, 'handoff_lock', self.lock)
        self.handoff_lock.start()
        self.addCleanup(self.handoff_lock.stop)
        hardware = ((self.native, 'capacity', lambda: self.capacity_value),
                    (self.native, 'park_old', self.fake_park),
                    (self.native, 'restored_owners', self.fake_owners),
                    (self.native, 'old_tree_absent', lambda expected: self.pool.need(all(x['synthetic_exited'] for x in expected), 'Worker still alive')),
                    (self.native, 'safe_parent_wait', lambda old, parent: self.native.safe_proof(Path(old['queue']) / 'resources_safe.json')),
                    (self.manager, 'validate', lambda cfg: self.hardware_validated.append((self.node, cfg['node'], cfg['phase']))))
        for obj, key, value in hardware:
            p = patch.object(obj, key, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(subprocess, 'check_output', lambda *a, **k: 'host-' + self.node.lower())
        p.start()
        self.addCleanup(p.stop)
        # Metadata receipt/argv checks remain the actual native CPU implementation.
        for i, cfg in enumerate(self.olds):
            parent = dict(pid=100 + i, uid=os.getuid(), argv=self.native.manager_argv(Path(cfg['config_path']), self.dep / 'caption_local_live_manager_v1.py', self.pool.sha(cfg['config_path'])),
                          start_ticks=i + 1, cgroup='synthetic', CVD=[], exe='python', ppid=1)
            child = dict(pid=200 + i, uid=os.getuid(), argv=cfg['argv'], start_ticks=i + 1, cgroup='synthetic',
                         CVD=['CUDA_VISIBLE_DEVICES=' + cfg['env']['CUDA_VISIBLE_DEVICES']], exe='python', ppid=parent['pid'], synthetic_exited=True)
            admission = dict(config_sha256=self.pool.sha(cfg['config_path']), manager_sha256=self.manager.BASE_SHA,
                             producer=cfg['producer'], plan=cfg['plan'], source_sha256=cfg['source_sha256'],
                             out=cfg['out'], node=cfg['node'], gpus=cfg['gpus'], phase=cfg['phase'], targetcount=4,
                             model_identity_sha256='synthetic', manager_identity=parent)
            live = dict(child_identity=child, manager_identity=parent, argv=cfg['argv'], phase=cfg['phase'], targetcount=4)
            self.pool.exclusive(Path(cfg['queue']) / 'admission.json', admission)
            self.pool.exclusive(Path(cfg['queue']) / 'live.json', live)

    @contextmanager
    def lock(self, phase):
        yield

    def fake_owners(self, old):
        self.native.safe_proof(Path(old['queue']) / 'resources_safe.json')
        return [dict(self.pool.load(Path(old['queue']) / ('holder_restored_gpu%d.json' % gpu))['holder_identity'], gpu=gpu,
                     uuid=self.pool.load(Path(old['queue']) / ('holder_restored_gpu%d.json' % gpu))['uuid']) for gpu in old['gpus']]

    def fake_park(self, old, parent, child, proofs, tree_path):
        self.park_calls.append(old['phase'])
        self.native.check_files(proofs)
        queue = Path(old['queue'])
        self.pool.exclusive(queue / 'resources_safe.json', dict(all_holders_safe=True, locks_released=True, leases_restored=True))
        for gpu in old['gpus']:
            self.pool.exclusive(queue / ('holder_restored_gpu%d.json' % gpu),
                                dict(gpu=gpu, uuid=old['node'] + '-gpu%d' % gpu,
                                     holder_identity=dict(pid=1000 + gpu, uid=os.getuid())))
        self.pool.exclusive(tree_path, dict(manager=parent, child=child, confirmed_gpu_tree=[]))
        return [child], self.native.safe_proof(queue / 'resources_safe.json')

    def park_all(self):
        for spec in self.specs:
            self.node = spec['node']
            self.h.park(self.h.cfg_path(spec))
        self.node = 'N4'

    def prepared(self):
        self.park_all()
        return self.h.prepare()

    def test_shared_union_claims_and_unready_retained(self):
        configs = self.prepared()
        rows = self.pool.load(self.h.artifact('_remaining'))
        self.assertEqual(len(rows), 8)
        self.assertEqual({r['id'] for r in rows}, {r['id'] for r in self.all_rows if r['source_ordinal'] % 10 in (2, 3)})
        self.assertEqual({cfg['targetcount'] for cfg in configs}, {8})
        self.assertEqual(len({cfg['plan']['sha256'] for cfg in configs}), 1)
        detail = self.pool.load(self.h.artifact('_handoff'))
        self.assertEqual(len(detail['original_owner_map']), 16)
        self.assertEqual(len(detail['excluded_source_keys']), 8)
        self.assertTrue(detail['all_unready_retained'])
        self.assertFalse(self.h.POOL_ROOT.exists())
        for old in self.olds:
            self.assertEqual((Path(old['draft_out']) / 'preserved.tmp').read_bytes(), b'old-cache-do-not-touch')

    def test_manifest_identity_no_cycle_and_each_node_validation(self):
        configs = self.prepared()
        self.assertEqual(len(self.hardware_validated), 3)
        self.assertTrue(all(actual == expected == 'N4' for actual, expected, phase in self.hardware_validated))
        manifest = self.pool.load(self.h.artifact('_manifest'))
        self.assertEqual(len(manifest['participants']), 4)
        for cfg, participant in zip(configs, manifest['participants']):
            self.assertEqual(participant['config_identity_sha256'], self.pool.config_identity(cfg))
            self.assertNotIn(str(self.h.artifact('_prepared')), cfg['source_sha256'])
            self.pool.check_manifest(cfg)
        self.node = 'N3'
        path = self.h.cfg_path(self.specs[1], True)
        self.h.validate_new(path)
        receipt = self.rt / (self.specs[1]['new'] + '_validated.json')
        before = receipt.read_bytes(), receipt.stat().st_mtime_ns
        self.h.validate_new(path)
        self.assertEqual((receipt.read_bytes(), receipt.stat().st_mtime_ns), before)
        self.assertTrue(all(a == e for a, e, phase in self.hardware_validated))

    def test_missing_park_seals_prepare(self):
        self.h.park(self.h.cfg_path(self.specs[0]))
        with self.assertRaises(FileNotFoundError):
            self.h.prepare()
        self.assertTrue(self.h.artifact('_failed').exists())
        with self.assertRaises(ValueError):
            self.h.park(self.h.cfg_path(self.specs[2]))

    def test_incomplete_latest_safe_rejected(self):
        self.park_all()
        path = Path(self.olds[1]['queue']) / 'resources_safe.json'
        path.write_text(json.dumps(dict(all_holders_safe=True, locks_released=False, leases_restored=True)))
        with self.assertRaises(ValueError):
            self.h.prepare()
        self.assertFalse(self.h.artifact('_remaining').exists())

    def test_history_safe_cannot_replace_current(self):
        self.park_all()
        old = self.olds[1]
        group = self.pool.ref(self.h.artifact('_claim'))
        historical = self.rt / 'historical-safe.json'
        self.pool.exclusive(historical, dict(all_holders_safe=True, locks_released=True, leases_restored=True, history=True))
        path = self.h.phase_path(self.specs[1], 'parked')
        record = self.pool.load(path)
        record['resource_previous_safe'] = self.pool.ref(historical)
        path.write_bytes(self.pool.encoded(record))
        with self.assertRaises(ValueError):
            self.h.parked_records(self.olds, group)

    def test_real_native_receipt_wrong_argv_seals_no_park(self):
        path = Path(self.olds[0]['queue']) / 'live.json'
        live = self.pool.load(path)
        live['argv'] = ['not-the-native-worker']
        path.write_bytes(self.pool.encoded(live))
        with self.assertRaises(ValueError):
            self.h.park(self.h.cfg_path(self.specs[0]))
        self.assertFalse(self.park_calls)
        self.assertTrue(self.h.artifact('_failed').exists())

    def test_wrong_old_producer_node_gpu_output_refused(self):
        for field, bad in (('producer', {'path': '/wrong', 'sha256': 'bad'}), ('node', 'N6'),
                           ('gpus', [0, 1]), ('draft_out', '/wrong')):
            with self.subTest(field=field):
                path = self.h.cfg_path(self.specs[0])
                original = path.read_bytes()
                cfg = self.pool.load(path)
                cfg[field] = bad
                path.write_bytes(self.pool.encoded(cfg))
                self.specs[0]['sha'] = self.pool.sha(path)
                with self.assertRaises(ValueError):
                    self.h.load_olds()
                path.write_bytes(original)
                self.specs[0]['sha'] = self.pool.sha(path)

    def test_wrong_new_node_and_tamper_refused(self):
        self.prepared()
        self.node = 'N4'
        with self.assertRaises(ValueError):
            self.h.validate_new(self.h.cfg_path(self.specs[1], True))
        self.assertFalse(self.h.artifact('_failed').exists())
        path = self.h.cfg_path(self.specs[0], True)
        cfg = self.pool.load(path)
        cfg['gpus'] = [4, 5]
        path.write_bytes(self.pool.encoded(cfg))
        with self.assertRaises(ValueError):
            self.h.validate_new(path)
        self.assertTrue(self.h.artifact('_failed').exists())

    def test_cross_device_only_allowed_other_proof_changes_rejected(self):
        path = self.rt / 'proof.json'
        self.pool.exclusive(path, {'value': 1})
        proof = self.native.file_identity(path)
        foreign = dict(proof, device=proof['device'] + 1)
        self.rebalance.check_shared_files([foreign])
        with self.assertRaises(ValueError):
            self.native.check_files([foreign])
        for key in ('inode', 'size', 'mtime_ns', 'uid', 'sha256', 'path'):
            changed = dict(proof)
            changed[key] = proof[key] + 1 if isinstance(proof[key], int) else proof[key] + 'x'
            with self.subTest(key=key), self.assertRaises((ValueError, FileNotFoundError)):
                self.rebalance.check_shared_files([changed])

    def test_prefetch_failed_and_commit_without_progress_excluded(self):
        old = self.olds[0]
        rows = self.pool.load(old['plan']['path'])
        root = Path(old['draft_out'])
        self.pool.exclusive(root / 'budgets/batch-extra/claim.json', dict(metas=[rows[2]], reserved_sources=1))
        self.pool.exclusive(root / 'batch-extra.committed.json', dict(batch_id='batch-extra', reliable_writeback=True,
                            archive_path=str(root / 'batch-extra.tar')))
        self.pool.exclusive(root / 'budgets/batch-lane0-1/failure.json', {'failed': True})
        excluded, detail, proofs = self.rebalance.local_exclusions(old, rows)
        self.assertEqual(len(excluded), 3)
        self.assertEqual(detail['committed_without_progress'], [(rows[2]['id'], 'train')])
        self.assertEqual(detail['reserved_uncommitted'], [(rows[1]['id'], 'train')])

    def test_gate_160_not_reduced_and_no_repeat_park_prepare(self):
        self.capacity_value['memory']['headroom'] = 159 * 2**30
        with self.assertRaises(ValueError):
            self.h.park(self.h.cfg_path(self.specs[0]))
        self.assertFalse(self.park_calls)
        with self.assertRaises(ValueError):
            self.h.park(self.h.cfg_path(self.specs[0]))

    def test_repeat_successful_park_seals_without_replay(self):
        spec = self.specs[0]
        self.h.park(self.h.cfg_path(spec))
        before = self.h.phase_path(spec, 'parked').read_bytes()
        with self.assertRaises(ValueError):
            self.h.park(self.h.cfg_path(spec))
        self.assertEqual(self.h.phase_path(spec, 'parked').read_bytes(), before)
        self.assertEqual(len(self.park_calls), 1)
        self.assertTrue(self.h.artifact('_failed').exists())

    def test_repeat_prepare_rejected_and_budget_fixed(self):
        self.prepared()
        with self.assertRaises(ValueError):
            self.h.prepare()
        self.assertTrue(self.h.artifact('_failed').exists())
        value = dict(self.h.BUDGET, memory={'headroom': 163 * 2**30}, local_disk_free_bytes=9 * 2**30)
        value['new_copy_threads'] = 5
        with self.assertRaises(ValueError):
            self.h.check_budget(value)


if __name__ == '__main__':
    unittest.main()
