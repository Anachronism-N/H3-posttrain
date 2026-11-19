"""CPU-only checks for caption_pool_consumer_handoff_v1 (no signals, no GPU, tmp dirs only)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import caption_pool_consumer_handoff_v1 as hand

IDS = ['AAAAAAAAAAAAAAAA', 'BBBBBBBBBBBBBBBB', 'CCCCCCCCCCCCCCCC', 'DDDDDDDDDDDDDDDD', 'EEEEEEEEEEEEEEEE', 'FFFFFFFFFFFFFFFF']


def rows():
    return [dict(id=i, split='train', batch_id='train-b000001', source_ordinal=n) for n, i in enumerate(IDS)]


class Exclusions(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='pool2-test-'))
        self.out = self.tmp / 'shard-0000'
        (self.out / 'sources' / 'train' / IDS[0]).mkdir(parents=True)
        (self.out / 'visual' / 'shard-0000' / 'sources' / IDS[1]).mkdir(parents=True)
        (self.out / 'completed.jsonl').write_text(json.dumps(dict(id=IDS[2], split='train')) + '\n')
        (self.out / 'batches' / 'train-b000001').mkdir(parents=True)
        (self.out / 'batches' / 'train-b000001' / 'terminal_index.jsonl').write_text(json.dumps(dict(id=IDS[3], split='train')) + '\n')
        self.old = dict(out=str(self.out), source_selection=dict(shard=0, nshard=9, limit=0))

    def test_started_and_terminal_excluded(self):
        denied, detail, proofs = hand.consumer_exclusions(self.old, rows())
        self.assertEqual({k[0] for k in denied}, set(IDS[:4]))
        self.assertEqual({k[0] for k in map(tuple, detail['started_without_terminal'])}, {IDS[0], IDS[1]})
        self.assertTrue(detail['budget_not_reset'])
        self.assertEqual(len(proofs), 2)

    def test_foreign_started_rejected(self):
        (self.out / 'sources' / 'train' / 'ZZZZZZZZZZZZZZZZ').mkdir()
        with self.assertRaises(Exception):
            hand.consumer_exclusions(self.old, rows())

    def test_foreign_visual_rejected(self):
        (self.out / 'visual' / 'shard-0000' / 'sources' / 'ZZZZZZZZZZZZZZZZ').mkdir()
        with self.assertRaises(Exception):
            hand.consumer_exclusions(self.old, rows())

    def test_unknown_visual_shard_rejected(self):
        (self.out / 'visual' / 'shard-0001').mkdir()
        with self.assertRaises(Exception):
            hand.consumer_exclusions(self.old, rows())

    def test_partial_journal_rejected(self):
        with (self.out / 'completed.jsonl').open('a') as f:
            f.write('{"id":"EEEE')
        with self.assertRaises(Exception):
            hand.consumer_exclusions(self.old, rows())

    def test_unknown_split_rejected(self):
        (self.out / 'sources' / 'bogus').mkdir()
        with self.assertRaises(Exception):
            hand.consumer_exclusions(self.old, rows())

    def test_empty_output_excludes_nothing(self):
        empty = dict(out=str(self.tmp / 'none'), source_selection=dict(shard=0, nshard=9, limit=0))
        denied, _, proofs = hand.consumer_exclusions(empty, rows())
        self.assertEqual(denied, set())
        self.assertEqual(proofs, [])


class Receipts(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='pool2-receipt-'))
        self.cfg_file = self.tmp / 'old_launch.json'
        self.cfg_file.write_text('{}')
        uid = os.getuid()
        self.old = dict(producer=dict(path='/p/consumer.py', sha256='a'), plan=dict(path='/p/plan', sha256='b'),
                        out='/o', node='N2', gpus=[2, 3], phase='old_phase', targetcount=3333, queue=str(self.tmp / 'q'),
                        modelref=dict(model_identity_sha256='m'), resource_manager=dict(path='/p/rm.py', sha256='c'),
                        argv=['/opt/venv_vllm/bin/python', '-B', '-u', '/p/consumer.py', 'consumer'],
                        env=dict(CUDA_VISIBLE_DEVICES='2,3'))
        digest = hand.m.base.sha(self.cfg_file)
        self.parent = dict(pid=10, uid=uid, ppid=1, start_ticks=5, state='S', cgroup='cg', CVD=[], exe='/opt/venv_vllm/bin/python3',
                           argv=['/opt/venv_vllm/bin/python', '-B', '-u', '/p/rm.py', '--config', str(self.cfg_file),
                                 '--config-sha', digest, '--manager-sha', hand.m.BASE_SHA])
        self.child = dict(pid=11, uid=uid, ppid=10, start_ticks=6, state='S', cgroup='cg', exe='/opt/venv_vllm/bin/python3',
                          CVD=['CUDA_VISIBLE_DEVICES=2,3'], argv=self.old['argv'])
        self.admission = dict(config_sha256=digest, manager_sha256=hand.m.BASE_SHA, producer=self.old['producer'],
                              plan=self.old['plan'], out='/o', node='N2', gpus=[2, 3], phase='old_phase', targetcount=3333,
                              model_identity_sha256='m', manager_identity=self.parent,
                              holders=[dict(gpu=2, uuid='U2', identity=dict(pid=20)), dict(gpu=3, uuid='U3', identity=dict(pid=21))])
        self.live = dict(manager_identity=dict(self.parent), child_identity=self.child, argv=self.old['argv'], phase='old_phase')

    def run_check(self):
        q = Path(self.old['queue'])
        q.mkdir(exist_ok=True)
        (q / 'admission.json').write_text(json.dumps(self.admission))
        (q / 'live.json').write_text(json.dumps(self.live))
        with mock.patch.object(hand, 'cfg_path', lambda spec, new=False: self.cfg_file):
            return hand.check_old_receipts(self.old, hand.SPECS[0])

    def test_exact_binding_passes(self):
        parent, child, lane = self.run_check()
        self.assertEqual(parent['pid'], 10)
        self.assertEqual(child['pid'], 11)
        self.assertEqual([h['uuid'] for h in lane['holders']], ['U2', 'U3'])

    def test_parent_argv_mismatch_rejected(self):
        self.parent['argv'] = self.parent['argv'][:-1] + ['0' * 64]
        with self.assertRaises(Exception):
            self.run_check()

    def test_child_not_direct_rejected(self):
        self.child['ppid'] = 99
        with self.assertRaises(Exception):
            self.run_check()

    def test_cvd_mismatch_rejected(self):
        self.child['CVD'] = ['CUDA_VISIBLE_DEVICES=4,5']
        with self.assertRaises(Exception):
            self.run_check()

    def test_config_sha_mismatch_rejected(self):
        self.admission['config_sha256'] = '0' * 64
        with self.assertRaises(Exception):
            self.run_check()

    def test_holder_order_rejected(self):
        self.admission['holders'].reverse()
        with self.assertRaises(Exception):
            self.run_check()


class Scope(unittest.TestCase):
    def test_only_n2_lanes(self):
        self.assertEqual({s['node'] for s in hand.SPECS}, {'N2'})
        self.assertEqual(sorted(g for s in hand.SPECS for g in s['gpus']), [2, 3, 4, 5, 6, 7])
        self.assertTrue(all(s['new'].startswith('caption_two_stage_local_') for s in hand.SPECS))
        self.assertEqual(hand.POOL_ROOT.name, 'caption_two_stage_dynamic_pool_v2')

    def test_n7_config_not_selectable(self):
        with self.assertRaises(Exception):
            hand.select(hand.RT / 'curated_collaborative3_n7c7_v1_launch.json')

    def test_manager_v2_hosts_and_identity(self):
        self.assertEqual(hand.manager.old.HOSTS['N2'], '28.216.17.70')
        self.assertEqual(hand.manager.old.HOSTS['N7'], '28.216.18.144')
        self.assertEqual(hand.manager.SELF.name, 'caption_pool_live_manager_v2.py')

    def test_union_overlap_with_foreign_rejected(self):
        olds = [dict(selected_sources=dict(path='x%d' % i), targetcount=2, phase='p%d' % i) for i in range(3)]
        plans = {('x%d' % i): [dict(id=IDS[2 * i], split='train'), dict(id=IDS[2 * i + 1], split='train')] for i in range(3)}
        with mock.patch.object(hand, 'ref', lambda p: dict(path=p)), \
             mock.patch.object(hand, 'cfg_path', lambda s, new=False: 'c'), \
             mock.patch.object(hand, 'load', lambda p: plans[p]), \
             mock.patch.object(hand, 'consumer_exclusions', lambda old, rows: (set(), {}, [])):
            with mock.patch.object(hand, 'foreign_keys', lambda: set()):
                remaining, detail, _ = hand.remaining_union(olds)
                self.assertEqual(len(remaining), 6)
            with mock.patch.object(hand, 'foreign_keys', lambda: {(IDS[5], 'train')}):
                with self.assertRaises(Exception):
                    hand.remaining_union(olds)


if __name__ == '__main__':
    unittest.main(verbosity=2)
