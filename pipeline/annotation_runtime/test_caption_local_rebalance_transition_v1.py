"""CPU tests: real old receipts/native validators, claim exclusion, complete union."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import caption_local_rebalance_transition_v1 as t


def budget():
    return dict(t.BUDGET, memory=dict(headroom=t.native.MIN_HEADROOM),
                local_disk_free_bytes=t.native.MIN_DISK)


class CurrentIdentityTests(unittest.TestCase):
    def test_actual_two_cfg_receipts_different_worker_paths_no_signal(self):
        with patch.object(t.native.signal, 'pidfd_send_signal', autospec=True) as signaling, \
             patch.object(t.native.os, 'pidfd_open', autospec=True) as opening:
            olds = t.load_pair()
            workers = []
            for old, spec in zip(olds, t.SPECS):
                scope = t.lane_scope(old)
                parent, child = scope['check_old_receipts'](old, t.old_path(spec),
                    t.load(Path(old['queue']) / 'admission.json'), t.load(Path(old['queue']) / 'live.json'))
                workers.append(child['argv'][2])
                self.assertEqual(parent['argv'][2], old['producer']['path'])
                bindings = scope['code_bindings'](old)
                self.assertEqual(bindings[old['producer']['path']], old['producer']['sha256'])
            self.assertNotEqual(*workers)
            self.assertEqual(t.native.LIGHT_MANAGER.name, 'caption_two_stage_live_manager_v5.py')
            self.assertEqual(t.native.SELF, Path(t.native.__file__).resolve())
        signaling.assert_not_called()
        opening.assert_not_called()

    def test_wrong_old_config_and_new_name_refused(self):
        with self.assertRaisesRegex(ValueError, 'two fixed'):
            t.select_spec(t.RT / 'wrong_launch.json')
        with self.assertRaisesRegex(ValueError, 'two fixed'):
            t.select_spec(t.RT / 'caption_two_stage_local_wrong_launch.json', new=True)
        with patch.object(t.m.base, 'sha', autospec=True, return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                t.load_pair()

    def test_budget_each_field_and_capacity_refused(self):
        for key in t.BUDGET:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'copy budget'):
                t.check_budget(dict(budget(), **{key: t.BUDGET[key] + 1}))
        for key in ('memory', 'local_disk_free_bytes'):
            bad = budget()
            bad[key] = dict(headroom=t.native.MIN_HEADROOM - 1) if key == 'memory' else t.native.MIN_DISK - 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'capacity gate'):
                t.check_budget(bad)


class SharedProofTests(unittest.TestCase):
    def test_cross_node_device_difference_does_not_change_shared_identity(self):
        proof = dict(path='/shared/proof.json', sha256='abc', device=1048631,
                     inode=123, size=42, mtime_ns=456, uid=os.getuid())
        current = dict(proof, device=1048634)
        with patch.object(t.native, 'file_identity', return_value=current):
            t.check_shared_files([proof])

    def test_all_other_shared_identity_changes_are_rejected(self):
        proof = dict(path='/shared/proof.json', sha256='abc', device=1048631,
                     inode=123, size=42, mtime_ns=456, uid=os.getuid())
        for key in ('path', 'sha256', 'inode', 'size', 'mtime_ns', 'uid'):
            changed = dict(proof, device=1048634)
            changed[key] = str(proof[key]) + '-changed' if isinstance(proof[key], str) else proof[key] + 1
            with self.subTest(key=key), patch.object(t.native, 'file_identity', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'Shared proof'):
                    t.check_shared_files([proof])


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [dict(id=name, split='train') for name in ('done', 'prefetch', 'failed', 'commit_lag', 'fresh')]
        self.old = dict(draft_out=str(self.root), targetcount=len(self.rows))
        (self.root / 'budgets').mkdir()
        t.write_once(self.root / 'progress.json', dict(processed=1, selected=len(self.rows), completed_source_keys=[['done', 'train']]))

    def claim(self, sid, committed=False):
        batch = 'batch-' + sid
        t.write_once(self.root / 'budgets' / batch / 'claim.json', dict(metas=[dict(id=sid, split='train')], reserved_sources=1))
        if committed:
            t.write_once(self.root / (batch + '.committed.json'), dict(batch_id=batch,
                reliable_writeback=True, archive_path=str(self.root / (batch + '.tar'))))

    def test_old_local_uncommitted_prefetch_failed_and_commit_lag_excluded_once(self):
        for sid in ('done', 'prefetch', 'failed', 'commit_lag'):
            self.claim(sid, committed=sid in ('done', 'commit_lag'))
        t.write_once(self.root / 'budgets/batch-failed/failure.json', dict(budget_not_reset=True))
        marker = self.root / 'inputs.tmp'
        marker.write_bytes(b'keep owned cache')
        with patch.object(t.os, 'scandir', wraps=os.scandir) as scanning:
            excluded, detail, proofs = t.local_exclusions(self.old, self.rows)
        self.assertEqual(excluded, {(sid, 'train') for sid in ('done', 'prefetch', 'failed', 'commit_lag')})
        self.assertEqual(set(map(tuple, detail['reserved_pending'])), {('prefetch', 'train'), ('failed', 'train'), ('commit_lag', 'train')})
        self.assertEqual(set(map(tuple, detail['reserved_uncommitted'])), {('prefetch', 'train'), ('failed', 'train')})
        self.assertEqual(detail['committed_without_progress'], [('commit_lag', 'train')])
        scanning.assert_called_once_with(self.root / 'budgets')
        self.assertTrue(detail['budget_not_reset'] and detail['old_inputs_tmp_preserved'])
        self.assertEqual(marker.read_bytes(), b'keep owned cache')
        t.native.check_files(proofs)

    def test_unknown_local_claim_refused(self):
        self.claim('not_in_plan')
        with self.assertRaisesRegex(ValueError, 'claim keys'):
            t.local_exclusions(self.old, self.rows)

    def test_missing_claim_is_failclosed_not_reassigned(self):
        (self.root / 'budgets/batch-interrupted').mkdir()
        with self.assertRaises(FileNotFoundError):
            t.local_exclusions(self.old, self.rows)

    def test_ready_balanced_445_and_all_763_n6_unready_retained(self):
        ready_rows = [dict(id='ready_%04d' % i, split='train') for i in range(445)]
        n4_unready = [dict(id='n4_future_%04d' % i, split='train') for i in range(21)]
        n6_unready = [dict(id='n6_future_%04d' % i, split='train') for i in range(763)]
        remaining = ready_rows + n4_unready + n6_unready
        owners = {t.native.row_key(row): int(row in n6_unready) for row in remaining}
        ready = {t.native.row_key(row) for row in ready_rows}
        plans, counts = t.balanced_plans(remaining, owners, ready)
        self.assertEqual(counts, [223, 222])
        self.assertTrue(all(row in plans[1] for row in n6_unready))
        self.assertTrue(all(row in plans[0] for row in n4_unready))
        keys = [{t.native.row_key(row) for row in plan} for plan in plans]
        self.assertFalse(keys[0] & keys[1])
        self.assertEqual(keys[0] | keys[1], {t.native.row_key(row) for row in remaining})

    def test_ready_index_reads_complete_prefix_once_each_no_media(self):
        (self.root / 'ready').mkdir()
        paths = []
        for i in range(4):
            path = self.root / 'ready' / ('producer-%04d.jsonl' % i)
            path.write_bytes((json.dumps(dict(id='source_%d' % i, split='train')) + '\n').encode() + b'{"incomplete":')
            paths.append(path)
        original = Path.read_bytes
        seen = []
        def read(path):
            seen.append(path)
            return original(path)
        with patch.object(Path, 'read_bytes', read):
            ready, prefixes = t.ready_index(self.root)
        self.assertEqual(seen, paths)
        self.assertEqual(ready, {('source_%d' % i, 'train') for i in range(4)})
        self.assertEqual(len(prefixes), 4)


class NativePairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rt = self.root / 'runtime'
        self.rt.mkdir()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(t, 'RT', self.rt))
        self.stack.enter_context(patch.object(t.native, 'RT', self.rt))
        self.stack.enter_context(patch.object(t.m, 'BAR', self.root))
        self.olds, self.owners = [], []
        for lane, (spec, node, gpus, ids) in enumerate(zip(t.SPECS, ('N4', 'N6'), ([0, 1], [2, 3]),
                                                     (('direct_done', 'ready_a', 'ready_b', 'n4_future'),
                                                      ('local_done', 'prefetch', 'n6_future')))):
            producer = t.m.ref(Path(t.m.SELF).parent / spec['producer'])
            rows = [dict(id=sid, split='train') for sid in ids]
            plan = self.rt / (spec['old'] + '_selected.json')
            t.write_once(plan, rows)
            path = t.old_path(spec)
            queue = self.root / 'queues' / spec['old']
            queue.mkdir(parents=True)
            safe = queue / 'resources_safe.json'
            t.write_once(safe, dict.fromkeys(t.native.SAFE_KEYS, True))
            owners = []
            for gpu in gpus:
                owner = dict(pid=800000 + lane * 100 + gpu, uid=os.getuid(), ppid=1,
                    argv=list(t.m.base.RESTORE_ARGV), start_ticks=1000 + gpu, cgroup='cpu-' + node,
                    CVD=['CUDA_VISIBLE_DEVICES=' + str(gpu)], exe='/cpu/python', sig_ign=0, sig_blk=0,
                    gpu=gpu, uuid='cpu-' + node + '-gpu-' + str(gpu))
                owners.append(owner)
                t.write_once(queue / ('holder_restored_gpu%d.json' % gpu),
                    dict(gpu=gpu, uuid=owner['uuid'], holder_identity=owner))
            self.owners.append(owners)
            out = self.root / spec['old']
            old = dict(node=node, gpus=gpus, queue=str(queue), phase=spec['old'], holders=owners,
                draft_out=str(out), out=str(self.root / (spec['old'] + '_execution')), plan=t.m.ref(plan),
                targetcount=len(rows), input_count=len(rows), config_path=str(path), cwd=str(self.rt),
                source_root=str(self.root / 'shared_source'), modelref=dict(model_identity_sha256='cpu-model'),
                env=dict(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), HF_HUB_OFFLINE='1'),
                producer=producer, resource_manager_ref=producer, release_authorized=True)
            old['argv'] = [t.native.PYTHON, '-B', producer['path'], 'worker', '--config', str(path)]
            old['source_sha256'] = {producer['path']: producer['sha256'], str(plan): t.m.base.sha(plan)}
            t.write_once(path, old)
            self.olds.append(old)
        t.write_once(Path(self.olds[0]['draft_out']) / 'sources/train/direct_done/prepared.draft.json', dict(started=True))
        local_root = Path(self.olds[1]['draft_out'])
        t.write_once(local_root / 'progress.json', dict(processed=1, selected=3, completed_source_keys=[['local_done', 'train']]))
        t.write_once(local_root / 'budgets/batch-prefetch/claim.json', dict(metas=[dict(id='prefetch', split='train')], reserved_sources=1))
        ready = Path(self.olds[0]['source_root']) / 'ready'
        ready.mkdir(parents=True)
        (ready / 'producer-0000.jsonl').write_text(''.join(json.dumps(dict(id=sid, split='train')) + '\n' for sid in ('ready_a', 'ready_b', 'prefetch')))
        self.stack.enter_context(patch.object(t, 'load_pair', return_value=self.olds))
        self.stack.enter_context(patch.object(t, 'capacity', side_effect=lambda: budget()))
        self.host = self.stack.enter_context(patch.object(t.m.h.subprocess, 'check_output', autospec=True,
                                                    return_value=t.m.HOSTS['N4']))
        self.current_lane = 0
        def identity(pid):
            for owner in self.owners[self.current_lane]:
                if owner['pid'] == pid:
                    return copy.deepcopy(owner)
            raise FileNotFoundError(pid)
        self.identity = self.stack.enter_context(patch.object(t.m.base, 'identity', autospec=True, side_effect=identity))
        self.stack.enter_context(patch.object(t.m.base, 'uuid_for', autospec=True,
            side_effect=lambda gpu: 'cpu-' + self.olds[self.current_lane]['node'] + '-gpu-' + str(gpu)))
        self.stack.enter_context(patch.object(t.m.base, 'apps', autospec=True,
            side_effect=lambda: [dict(uuid=owner['uuid'], pid=owner['pid'], used_MiB='1') for owner in self.owners[self.current_lane]]))
        self.stack.enter_context(patch.object(t.m.h, 'lock_snapshot', autospec=True, return_value=dict(held=[])))
        self.stack.enter_context(patch.object(t.m.h, 'lease_snapshot', autospec=True, return_value={}))
        self.group = t.group_claim(self.olds, create=True)
        for old, spec, owners in zip(self.olds, t.SPECS, self.owners):
            queue = Path(old['queue'])
            proofs = [t.native.file_identity(t.old_path(spec)), t.native.file_identity(old['plan']['path']),
                t.native.file_identity(queue / 'resources_safe.json')]
            proofs += [t.native.file_identity(queue / ('holder_restored_gpu%d.json' % gpu)) for gpu in old['gpus']]
            code_scope = t.lane_scope(old)
            code_scope['RT'] = Path(t.m.SELF).parent
            record = dict(group_claim=self.group, old_config=t.m.ref(t.old_path(spec)), old_phase=old['phase'],
                manager=dict(pid=900000, start_ticks=1), expected_old_tree=[], old_worker_tree_absent=True,
                old_worker_tree_absence_scope=t.ABSENCE_SCOPE, resource_previous_safe=t.m.ref(queue / 'resources_safe.json'),
                owners=owners, proofs=proofs, source_bindings=code_scope['code_bindings'](old), resource_budget=budget())
            t.write_once(self.rt / (spec['old'] + '_rebalance_parked.json'), record)

    def test_union_excludes_live_started_and_local_completed_prefetch(self):
        remaining, owners, detail, proofs = t.remaining_union(self.olds)
        self.assertEqual({t.native.row_key(row) for row in remaining},
                         {(sid, 'train') for sid in ('ready_a', 'ready_b', 'n4_future', 'n6_future')})
        self.assertEqual(detail['reserved_pending'], [('prefetch', 'train')])
        self.assertEqual(set(map(tuple, detail['excluded_started_without_draft'])),
                         {('direct_done', 'train'), ('prefetch', 'train')})
        self.assertEqual(owners[('n6_future', 'train')], 1)
        t.native.check_files(proofs)

    def test_prepare_two_real_native_cfg_validators_two_distinct_safes(self):
        with patch.object(t.native.signal, 'pidfd_send_signal', autospec=True) as signaling:
            configs = t.prepare()
            self.assertEqual([cfg['targetcount'] for cfg in configs], [2, 2])
            self.assertEqual(configs[0]['source_handoff'], configs[1]['source_handoff'])
            self.assertNotEqual(configs[0]['resource_previous_safe'], configs[1]['resource_previous_safe'])
            original_selector = t.m.base.selected_sources
            for i, cfg in enumerate(configs):
                self.current_lane = i
                self.host.return_value = t.m.HOSTS[self.olds[i]['node']]
                argv = t.validate_new(cfg['config_path'])
                self.assertEqual(cfg['producer'], t.m.ref(t.m.SELF))
                self.assertEqual(cfg['resource_manager_ref'], t.m.ref(t.m.SELF))
                self.assertEqual(argv[2], str(t.m.SELF))
                self.assertNotIn('-u', argv)
                self.assertEqual(argv[-1], t.m.BASE_SHA)
                self.assertIs(t.m.base.selected_sources, original_selector)
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                t.prepare()
        signaling.assert_not_called()

    def test_park_bad_budget_claims_failclosed_before_any_pidfd(self):
        bad = dict(budget(), total_copy_threads=7)
        with patch.object(t, 'capacity', return_value=bad), \
             patch.object(t.native.os, 'pidfd_open', autospec=True) as opening:
            with self.assertRaisesRegex(ValueError, 'copy budget'):
                t.park(t.old_path(t.SPECS[0]))
            self.assertTrue(t.artifact('_failed').exists())
            self.assertTrue(t.load(t.artifact('_failed'))['locks_leases_budgets_not_deleted'])
            with self.assertRaisesRegex(ValueError, 'failed; no automatic'):
                t.park(t.old_path(t.SPECS[0]))
        opening.assert_not_called()

    def test_already_native_safe_park_real_helpers_no_signal_then_second_park_refused(self):
        old, spec = self.olds[0], t.SPECS[0]
        scope = t.lane_scope(old)
        parent = dict(pid=900000, uid=os.getuid(), ppid=1, start_ticks=1, cgroup='cpu-N4',
            argv=t.native.manager_argv(t.old_path(spec), Path(old['producer']['path']), t.m.base.sha(t.old_path(spec))),
            CVD=[], exe='/cpu/python', sig_ign=0, sig_blk=0)
        child = dict(parent, pid=900001, ppid=parent['pid'], argv=old['argv'],
                     CVD=['CUDA_VISIBLE_DEVICES=' + old['env']['CUDA_VISIBLE_DEVICES']])
        queue = Path(old['queue'])
        admission = dict(config_sha256=t.m.base.sha(t.old_path(spec)), manager_sha256=t.m.BASE_SHA,
            producer=old['producer'], plan=old['plan'], source_sha256=old['source_sha256'],
            out=old['out'], node=old['node'], gpus=old['gpus'], phase=old['phase'],
            targetcount=old['targetcount'], model_identity_sha256=old['modelref']['model_identity_sha256'],
            manager_identity=parent)
        live = dict(manager_identity=parent, child_identity=child, argv=old['argv'],
                    phase=old['phase'], targetcount=old['targetcount'])
        t.write_once(queue / 'admission.json', admission)
        t.write_once(queue / 'live.json', live)
        parked = self.rt / (spec['old'] + '_rebalance_parked.json')
        original_write = t.write_once
        def write(path, value):
            return original_write(self.root / 'new_cpu_parked.json' if Path(path) == parked else path, value)
        with patch.object(t, 'write_once', side_effect=write), \
             patch.object(t.native.signal, 'pidfd_send_signal', autospec=True) as signaling, \
             patch.object(t.native.os, 'pidfd_open', autospec=True) as opening:
            record = t.park(t.old_path(spec))
            self.assertEqual(record['expected_old_tree'], [child])
            self.assertTrue(record['old_worker_tree_absent'])
            self.assertEqual(record['owners'], self.owners[0])
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                t.park(t.old_path(spec))
        signaling.assert_not_called()
        opening.assert_not_called()

    def test_changed_shared_group_claim_budget_refuses(self):
        path = t.artifact('_claim')
        claim = t.load(path)
        claim['resource_budget']['total_copy_threads'] += 1
        path.write_text(json.dumps(claim))
        with self.assertRaisesRegex(ValueError, 'claim/config/budget'):
            t.group_claim(self.olds)

    def test_second_safe_incomplete_blocks_both_configs(self):
        safe = Path(self.olds[1]['queue']) / 'resources_safe.json'
        safe.write_text(json.dumps(dict.fromkeys(t.native.SAFE_KEYS, True) | {'leases_restored': False}))
        with self.assertRaises(ValueError):
            t.prepare()
        self.assertTrue(t.artifact('_failed').exists())
        self.assertFalse((self.rt / (t.SPECS[0]['new'] + '_launch.json')).exists())


if __name__ == '__main__':
    unittest.main()
