"""CPU-only four-lane handoff regressions; code copied to /tmp exactly once."""
import atexit
from contextlib import redirect_stdout
import copy
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent
SANDBOX = tempfile.TemporaryDirectory(prefix='caption-handoff-cpu-', dir='/tmp')
atexit.register(SANDBOX.cleanup)
RT = Path(SANDBOX.name) / 'runtime'
RT.mkdir()
for name in ('caption_pool_handoff_v1', 'caption_local_rebalance_transition_v1',
             'caption_light_to_local_transition_v1', 'caption_light_to_local_transition_v2',
             'caption_ready_pool_v1', 'caption_pool_live_manager_v1', 'caption_local_live_manager_v1',
             'caption_local_source_bundle_v1', 'caption_two_stage_light_v1', 'caption_direct_facts_v1',
             'visual_batch_fixed3_benchmark_v1', 'bench_manager_driver_v2', 'curated_resource_manager', 'resource_manager'):
    shutil.copyfile(SOURCE / (name + '.py'), RT / (name + '.py'))
sys.path.insert(0, str(RT))
import caption_pool_handoff_v1 as h

p, m, n, new = h.pool, h.m, h.native, h.manager
CODE_REFS = {str(path): p.sha(path) for path in RT.glob('*.py')}


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='assets-', dir=SANDBOX.name)
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.addCleanup(patch.stopall)
        self.addCleanup(self.clear_artifacts)
        patch.object(h, 'BAR', self.root).start()
        patch.object(m, 'BAR', self.root).start()
        patch.object(new, 'BAR', self.root).start()
        patch.object(h, 'POOL_ROOT', self.root / 'caption_two_stage_dynamic_pool_v1').start()
        self.node = 'N4'
        self.identities, self.owners = {}, []
        self.headroom, self.diskfree = 161 * 2**30, 9 * 2**30
        self.identity_calls = []
        patch.object(m.h.subprocess, 'check_output', side_effect=self.hostname).start()
        patch.object(m.base, 'identity', side_effect=self.identity).start()
        patch.object(m.base, 'uuid_for', side_effect=lambda gpu: self.uuid(self.node, gpu)).start()
        patch.object(m.base, 'apps', side_effect=self.apps).start()
        patch.object(m.h, 'lock_snapshot', return_value=dict(held=False)).start()
        patch.object(m.h, 'lease_snapshot', return_value=None).start()
        patch.object(m.base, 'headroom', side_effect=lambda: dict(headroom=self.headroom)).start()
        patch.object(n.shutil, 'disk_usage', side_effect=lambda path: types.SimpleNamespace(free=self.diskfree)).start()
        self.output = io.StringIO()
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def clear_artifacts(self):
        for path in RT.iterdir():
            if path.suffix != '.py':
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    def hostname(self, argv, **kwargs):
        self.assertEqual(argv, ['hostname', '-I'])
        return m.HOSTS[self.node] + '\n'

    def uuid(self, node, gpu):
        return 'synthetic-' + node + '-gpu-' + str(gpu)

    def identity(self, pid):
        self.identity_calls.append(pid)
        if pid not in self.identities:
            raise FileNotFoundError(pid)
        return self.identities[pid]

    def apps(self):
        return [dict(pid=o['pid'], uuid=o['uuid']) for i, owners in enumerate(self.owners)
                if h.SPECS[i]['node'] == self.node for o in owners]

    def fixture(self, parked=True):
        specs = copy.deepcopy(h.SPECS)
        for spec in specs:
            spec['count'] = 7
        patch.object(h, 'SPECS', tuple(specs)).start()
        source = self.root / 'curated222939_collaborative_v3'
        source.mkdir()
        model_path = self.root / 'modelref.json'
        p.exclusive(model_path, dict(synthetic=True))
        modelref = dict(p.ref(model_path), model_identity_sha256='synthetic-model')
        self.rows, olds = [], []
        for i, spec in enumerate(specs):
            rows = [dict(id='lane%d-source%d' % (i, j), split='train', payload=dict(lane=i, index=j)) for j in range(7)]
            self.rows.append(rows)
            plan_path = RT / (spec['old'] + '_selected.json')
            p.exclusive(plan_path, rows)
            out = self.root / spec['old']
            queue = self.root / 'queues' / spec['old']
            p.exclusive(out / 'progress.json', dict(processed=1, selected=7, completed_source_keys=[[rows[0]['id'], 'train']]))
            for batch, metas in [('batch-prefetch', rows[:2]), ('batch-failed', rows[2:3]), ('batch-late', rows[3:4])]:
                p.exclusive(out / 'budgets' / batch / 'claim.json', dict(metas=metas, reserved_sources=len(metas)))
            p.exclusive(out / 'batch-late.committed.json', dict(batch_id='batch-late', reliable_writeback=True,
                        archive_path=str(out / 'batch-late.tar')))
            p.exclusive(out / 'batch-failed.failed.json', dict(reason='synthetic failure'))
            prior = self.root / ('historical-' + str(i))
            p.exclusive(prior / 'resources_safe.json', dict(all_holders_safe=True, locks_released=True, leases_restored=True))
            p.exclusive(queue / 'resources_safe.json', dict(all_holders_safe=True, locks_released=True, leases_restored=True, current_lane=i))
            owners = []
            for gpu in spec['gpus']:
                pid = (i + 1) * 1000 + gpu
                owner = dict(pid=pid, uid=os.getuid(), argv=['python', str(m.base.HOLDER)], start_ticks=pid,
                    cgroup='synthetic-' + spec['node'], CVD=['CUDA_VISIBLE_DEVICES=' + str(gpu)], exe='/fake/python',
                    sig_ign=0, sig_blk=0, gpu=gpu, uuid=self.uuid(spec['node'], gpu))
                owners.append(owner)
                self.identities[pid] = owner
                p.exclusive(queue / ('holder_restored_gpu%d.json' % gpu), dict(gpu=gpu, uuid=owner['uuid'], holder_identity=owner))
            self.owners.append(owners)
            producer = p.ref(m.SELF)
            cfg_path = h.cfg_path(spec)
            cfg = dict(node=spec['node'], gpus=spec['gpus'], phase=spec['old'], targetcount=7, input_count=7,
                producer=producer, resource_manager_ref=producer, config_path=str(cfg_path), cwd=str(RT),
                release_authorized=True, env=dict(CUDA_VISIBLE_DEVICES=','.join(map(str, spec['gpus'])), OMP_NUM_THREADS='1'),
                argv=[n.PYTHON, '-B', str(m.SELF), 'worker', '--config', str(cfg_path)], holders=owners,
                out=str(self.root / (spec['old'] + '_execution')), draft_out=str(out), queue=str(queue),
                source_root=str(source), modelref=modelref, plan=p.ref(plan_path), source_sha256=dict(CODE_REFS),
                resource_previous_safe=p.ref(prior / 'resources_safe.json'))
            cfg['source_sha256'][str(plan_path)] = p.sha(plan_path)
            p.exclusive(cfg_path, cfg)
            spec['sha'] = p.sha(cfg_path)
            parent, child = dict(pid=90000+i, synthetic=True), dict(pid=91000+i, synthetic=True)
            p.exclusive(queue / 'admission.json', dict(synthetic=True, manager_identity=parent))
            p.exclusive(queue / 'live.json', dict(synthetic=True, child_identity=child))
            olds.append(cfg)
        h.load_olds()
        group = h.group_claim(olds, create=True)
        if parked:
            for old, spec, owners in zip(olds, specs, self.owners):
                attempt = h.phase_path(spec, 'park_claim')
                p.exclusive(attempt, dict(group_claim=group, old_config=p.ref(h.cfg_path(spec)), transition=p.ref(h.SELF)))
                tree_path = h.phase_path(spec, 'old_tree')
                p.exclusive(tree_path, dict(synthetic=True))
                queue = Path(old['queue'])
                paths = [h.cfg_path(spec), Path(old['plan']['path']), queue/'admission.json', queue/'live.json',
                         tree_path, queue/'resources_safe.json'] + [queue/('holder_restored_gpu%d.json'%g) for g in old['gpus']]
                record = dict(group_claim=group, old_config=p.ref(h.cfg_path(spec)), old_phase=old['phase'],
                    manager=p.load(queue/'admission.json')['manager_identity'],
                    expected_old_tree=[p.load(queue/'live.json')['child_identity']], old_worker_tree_absent=True,
                    old_worker_tree_absence_scope=h.ABSENCE_SCOPE, safe_old_manager_quiescent_wait_allowed=True,
                    resource_previous_safe=p.ref(queue/'resources_safe.json'), owners=owners,
                    proofs=[n.file_identity(path) for path in paths], source_bindings=old['source_sha256'],
                    resource_budget=h.capacity(), park_attempt=p.ref(attempt))
                p.exclusive(h.phase_path(spec, 'parked'), record)
        return olds, group

    def prepared(self):
        olds, group = self.fixture()
        return olds, group, h.prepare()

    def replace_json(self, path, change):
        value = p.load(path)
        change(value)
        p.atomic(path, value)
        return value

    def test_union_retains_all_unready_and_excludes_all_budget_metas(self):
        olds, _ = self.fixture()
        rows, detail, proofs = h.remaining_union(olds)
        expected = [row for lane in self.rows for row in lane[4:]]
        self.assertEqual(rows, expected)
        self.assertEqual(len(rows), 12)
        self.assertEqual(detail['original_union_count'], 28)
        self.assertEqual(len(detail['excluded_source_keys']), 16)
        self.assertTrue(detail['all_unready_retained'])
        self.assertEqual(len(detail['original_owner_map']), 28)
        for lane in detail['per_lane_exclusions']:
            info = lane['details']
            self.assertEqual(len(info['reserved_pending']), 3)
            self.assertEqual(len(info['committed_without_progress']), 1)
            self.assertEqual(len(info['reserved_uncommitted']), 3)
        n.check_files(proofs)
        self.assertFalse(any(Path(olds[0]['source_root']).glob('ready*')))

    def test_prepare_actual_native_hooks_only_local_three_then_n3_one(self):
        olds, group = self.fixture()
        with patch.object(new, 'validate', wraps=new.validate) as validate, patch.object(h, 'check_sources', wraps=h.check_sources) as sources:
            cfgs = h.prepare()
            self.assertEqual([call.args[0]['node'] for call in validate.call_args_list], ['N4'] * 3)
            self.assertEqual([call.args[0]['gpus'] for call in validate.call_args_list], [[6,7], [4,5], [2,3]])
            self.assertEqual(sources.call_count, 3)
            self.assertTrue(all(call.args[2] is new.selected for call in sources.call_args_list))
            n3_receipt = RT / (h.SPECS[1]['new'] + '_validated.json')
            self.assertFalse(n3_receipt.exists())
            n3_ids = {o['pid'] for o in self.owners[1]} | {90001, 91001}
            self.assertFalse(n3_ids & set(self.identity_calls))
            self.node = 'N3'
            h.validate_new(cfgs[1]['config_path'])
            self.assertEqual(validate.call_count, 4)
            self.assertEqual(sources.call_count, 4)
            self.assertTrue(n3_receipt.exists())
        for cfg in cfgs:
            self.assertEqual(cfg['targetcount'], 12)
            self.assertEqual(cfg['input_count'], 12)
            self.assertEqual(cfg['resource_previous_safe']['path'], str(Path(olds[h.select(cfg['config_path'], True)]['queue'])/'resources_safe.json'))
            self.assertFalse('processed' in cfg)

    def test_prepare_on_n3_validates_one_then_n4_validates_three(self):
        self.fixture()
        self.node = 'N3'
        with patch.object(new, 'validate', wraps=new.validate) as validate:
            cfgs = h.prepare()
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(validate.call_args.args[0]['node'], 'N3')
            n4_ids = {owner['pid'] for i, owners in enumerate(self.owners) if i != 1 for owner in owners}
            self.assertFalse(n4_ids & set(self.identity_calls))
            self.node = 'N4'
            for i in (0, 2, 3):
                h.validate_new(cfgs[i]['config_path'])
            self.assertEqual(validate.call_count, 4)
        self.assertTrue(all((RT / (s['new'] + '_validated.json')).exists() for s in h.SPECS))

    def test_partial_xb_attempt_cannot_be_rewritten(self):
        path = self.root / 'partial.json'
        path.write_bytes(b'{')
        self.assertIs(h.write_once, p.exclusive)
        with self.assertRaises(FileExistsError):
            h.write_once(path, dict(replay=True))
        self.assertEqual(path.read_bytes(), b'{')

    def test_missing_park_attempt_refused_and_seals_group(self):
        self.fixture()
        h.phase_path(h.SPECS[0], 'park_claim').unlink()
        with self.assertRaises(FileNotFoundError):
            h.prepare()
        self.assertTrue(h.artifact('_failed').exists())
        self.assertFalse(h.artifact('_remaining').exists())

    def test_manifest_cycle_identity_and_bindings(self):
        _, _, cfgs = self.prepared()
        handoff = p.load(h.artifact('_handoff'))
        self.assertNotIn('pool_manifest', handoff)
        self.assertNotIn('new_configs', handoff)
        manifest = p.load(h.artifact('_manifest'))
        self.assertEqual(len(manifest['participants']), 4)
        for cfg, participant in zip(cfgs, manifest['participants']):
            self.assertEqual(participant['config_identity_sha256'], p.config_identity(cfg))
            changed = copy.deepcopy(cfg)
            changed['pool_manifest']['sha256'] = 'x'
            changed['source_sha256'][changed['pool_manifest']['path']] = 'y'
            changed['_config_sha'], changed['_config_path'] = 'runtime', '/runtime'
            self.assertEqual(p.config_identity(changed), p.config_identity(cfg))
            self.assertFalse(any(h.cfg_path(s, True).as_posix() in cfg['source_sha256'] for s in h.SPECS))
            h.check_sources(cfg, Path(cfg['config_path']), new.selected)
            p.check_manifest(cfg)

    def test_missing_park_record_seals_group(self):
        self.fixture(parked=False)
        with self.assertRaises(FileNotFoundError):
            h.prepare()
        self.assertTrue(h.artifact('_failed').exists())
        self.assertFalse(h.artifact('_manifest').exists())
        with self.assertRaisesRegex(ValueError, 'sealed'):
            h.prepare()

    def test_incomplete_current_safe_cannot_use_historical_safe(self):
        olds, _ = self.fixture()
        p.atomic(Path(olds[1]['queue'])/'resources_safe.json', dict(all_holders_safe=True, locks_released=False, leases_restored=True))
        self.assertTrue(Path(olds[1]['resource_previous_safe']['path']).exists())
        with self.assertRaises(ValueError):
            h.prepare()
        self.assertTrue(h.artifact('_failed').exists())

    def test_failed_group_no_automatic_retry_or_claim_cleanup(self):
        olds, group = self.fixture()
        before = n.file_identity(h.artifact('_claim'))
        h.fail(ValueError('synthetic'))
        with self.assertRaisesRegex(ValueError, 'sealed'):
            h.group_claim(olds, create=True)
        self.assertEqual(n.file_identity(h.artifact('_claim')), before)
        self.assertTrue(all(h.old_claim(s).exists() for s in h.SPECS))

    def test_later_lane_claim_is_legal_same_group(self):
        olds, group = self.fixture(parked=False)
        self.assertEqual(h.group_claim(olds, create=True), group)
        self.assertEqual(h.group_claim(olds), group)

    def test_native_capacity_exact_160_gate_and_disk_gate(self):
        self.headroom, self.diskfree = 160*2**30, 8*2**30
        h.capacity()
        self.headroom -= 1
        with self.assertRaisesRegex(ValueError, '160'):
            h.capacity()
        self.headroom += 1
        self.diskfree -= 1
        with self.assertRaisesRegex(ValueError, '8 GiB'):
            h.capacity()

    def test_prepare_fresh_capacity_fail_before_new_plan(self):
        self.fixture()
        self.headroom = 160*2**30-1
        with self.assertRaises(ValueError):
            h.prepare()
        self.assertFalse(h.artifact('_remaining').exists())
        self.assertTrue(h.artifact('_failed').exists())

    def test_wrong_real_node_refused_without_remote_proc(self):
        _, _, cfgs = self.prepared()
        before = list(self.identity_calls)
        with self.assertRaisesRegex(ValueError, 'node'):
            h.validate_new(cfgs[1]['config_path'])
        self.assertEqual(self.identity_calls, before)
        self.assertTrue(h.artifact('_failed').exists())

    def test_wrong_producer_gpu_and_output_refs_refused(self):
        olds, group, cfgs = self.prepared()
        records = h.parked_records(olds, group)
        path = Path(cfgs[0]['config_path'])
        original = copy.deepcopy(cfgs[0])
        for field, value in [('producer', p.ref(m.SELF)), ('gpus', [0,1]), ('draft_out', olds[0]['draft_out']),
                             ('queue', olds[0]['queue']), ('plan', olds[0]['plan'])]:
            bad = copy.deepcopy(original)
            bad[field] = value
            p.atomic(path, bad)
            self.replace_json(h.artifact('_prepared'), lambda receipt: receipt['configs'].__setitem__(0, p.ref(path)))
            with self.assertRaises(ValueError):
                h._validate(path, olds, group, records)
        p.atomic(path, original)

    def test_actual_base_source_binding_rejects_changed_digest(self):
        _, _, cfgs = self.prepared()
        cfg = copy.deepcopy(cfgs[0])
        cfg['source_sha256'][str(RT/'caption_ready_pool_v1.py')] = '0'*64
        with self.assertRaisesRegex(RuntimeError, 'source/input changed'):
            h.check_sources(cfg, Path(cfg['config_path']), new.selected)

    def test_repeated_validate_is_read_only_same_receipt(self):
        _, _, cfgs = self.prepared()
        path = RT/(h.SPECS[0]['new']+'_validated.json')
        before = n.file_identity(path)
        argv = h.validate_new(cfgs[0]['config_path'])
        self.assertEqual(n.file_identity(path), before)
        self.assertEqual(argv[:2], [n.PYTHON, '-B'])
        self.assertNotIn('-u', argv)
        self.assertEqual(argv[-2:], ['--manager-sha', m.BASE_SHA])

    def test_shared_proof_only_device_difference_allowed(self):
        self.fixture()
        proof = n.file_identity(h.artifact('_claim'))
        changed = dict(proof, device=proof['device']+99)
        h.rebalance.check_shared_files([changed])
        with self.assertRaises(ValueError):
            n.check_files([changed])
        for field in ('sha256', 'inode', 'size', 'mtime_ns', 'uid', 'path'):
            changed = dict(proof)
            changed[field] = '/tmp/nonexistent-proof' if field == 'path' else ('wrong' if field == 'sha256' else proof[field]+1)
            with self.assertRaises((ValueError, FileNotFoundError)):
                h.rebalance.check_shared_files([changed])


if __name__ == '__main__':
    unittest.main()
