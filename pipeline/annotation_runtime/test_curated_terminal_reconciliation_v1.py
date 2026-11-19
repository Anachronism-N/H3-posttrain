#!/usr/bin/env python3
"""Real original-source smoke test plus isolated, full-chain negative/replay tests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import curated_terminal_reconciliation_v1 as m

q = m.q


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(q.encoded(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def binary(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


class FullChainFixture:
    """Every admitted source traverses the real frozen Evidence.terminal implementation."""
    def __init__(self, root):
        self.root, self.production, self.out = root, root / 'production', root / 'derived'
        self.production.mkdir()
        csv = binary(root / 'original.csv', b'id,caption\noriginal,true\n')
        code = binary(root / 'ms_bar_annotation.py', b"def params():\n    return {'configuration': True, 'code_sha256': {}}\n")
        code_map = {code['path']: code['sha256']}
        configuration = {'configuration': True, 'code_sha256': code_map, 'style_module_sha256': 'fixture-style'}
        params_sha = m.recovery.visual_digest(configuration)
        self.media, self.rows, self.joints, self.plans = {}, {}, {}, {}
        descriptors, batch_members = [], {}
        assignment = {'G': 0, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'H': 2, 'E': 3, 'F': 3}
        for index in range(448):
            batch = 'train-b%06d' % index
            count = 500 if index < 444 else [200, 200, 200, 339][index - 444]
            path = root / 'inputs' / (batch + '.jsonl')
            names = sorted(s for s, b in assignment.items() if b == index)
            members = []
            if index < 4:
                for sid in names:
                    video = binary(root / 'media' / (sid + '.mp4'), ('video-' + sid).encode())
                    audio = binary(root / 'media' / (sid + '.wav'), ('audio-' + sid).encode())
                    self.media[sid] = {'video_path': video, 'audio_path': audio}
                    members.append({'id': sid, 'split': 'train', 'batch_id': batch,
                                    'video_path': video['path'], 'audio_path': audio['path'], 'source_refs': {'csv': csv}})
                for number in range(count - len(members)):
                    members.append({'id': 'unused%06d_%03d' % (index, number), 'split': 'train', 'batch_id': batch,
                                    'video_path': '/tmp/not-a-candidate', 'audio_path': '/tmp/not-a-candidate', 'source_refs': {'csv': csv}})
                members = [json.loads(q.encoded(row)) for row in members]
                raw = b''.join(q.encoded(row) for row in members)
                ref = binary(path, raw)
                batch_members[batch] = {r['id']: r for r in members}
            else:
                ref = {'path': str(path), 'sha256': 'not-selected-and-never-read'}
            descriptors.append({'batch_id': batch, 'count': count, 'split': 'train', **ref})
        self.manifest = root / 'inputs' / 'manifest.summary.json'
        summary_ref = put(self.manifest, {'total': 222939, 'batch_size': 500, 'batches': descriptors})
        put(self.production / 'runs/v0-original/protocol.json', {'mode': 'consumer',
            'configuration': configuration, 'manifest_summary': summary_ref})
        for sid, batch_number in assignment.items():
            batch = 'train-b%06d' % batch_number
            member = batch_members[batch][sid]
            directory = self.production / 'sources/train' / sid
            visual_root = self.production / 'visual/shard-0000/sources' / sid
            put(self.production / 'visual/shard-0000/load.receipt.json', {'code_sha256': code_map})
            manifest_ref = {k: descriptors[batch_number][k] for k in ('path', 'sha256')}
            plan = {'id': sid, 'manifest_record': member,
                    'source_refs': {'manifest': manifest_ref, 'original_csv': csv},
                    'binding': {'video_sha256': self.media[sid]['video_path']['sha256'],
                                'audio_sha256': self.media[sid]['audio_path']['sha256']}}
            plan = json.loads(q.encoded(plan))
            plan_ref = put(directory / 'bar_plan.json', plan)
            self.plans[sid] = plan_ref
            binding = {'source_binding': plan['binding'], 'plan_sha256': plan_ref['sha256'],
                       'full_source_row_sha256': m.recovery.visual_digest(plan), 'raw_line_sha256': plan_ref['sha256'],
                       'manifest_record_sha256': m.recovery.visual_digest(member), 'params_sha256': params_sha}
            put(directory / 'producer.terminal.json', {'id': sid, 'split': 'train', 'batch_id': batch,
                                                       'plan_ref': plan_ref, 'source_binding': plan['binding']})
            put(visual_root / 'binding.json', binding)
            put(visual_root / 'source.refs.json', [manifest_ref, csv])
            source_identity = {'sha256': 'original-source-' + sid, 'files': copy.deepcopy(self.media[sid])}
            put(visual_root / 'source.identity.json', source_identity)
            prepared_ref = put(visual_root / 'prepared.json', {'binding': binding})
            put(visual_root / 'prepared.receipt.json', {'binding': binding, 'prepared_sha256': prepared_ref['sha256']})
            visual = {'id': sid, 'manifest_record': member, 'binding': binding}
            visual_ref = put(visual_root / 'results' / (q.digest(visual) + '.json'), visual)
            style = {'id': sid, 'status': 'candidate'}
            style_ref = put(directory / 'style-results' / (q.digest(style) + '.json'), style)
            joint = {'id': sid, 'split': 'train', 'batch_id': batch, 'joint_complete': True,
                     'style_status': 'candidate', 'visual_status': 'candidate', 'source_refs': member['source_refs'],
                     'source_binding': binding, 'visual_result_ref': visual_ref, 'style_result_ref': style_ref,
                     'source_identity': source_identity}
            joint_ref = put(directory / 'joint-results' / (q.digest(joint) + '.json'), joint)
            self.joints[sid] = joint_ref
            row = {'id': sid, 'split': 'train', 'batch_id': batch, 'status': 'joint_candidate',
                   'joint_complete': True, 'joint_candidate_ref': joint_ref,
                   'style_status': 'candidate', 'visual_status': 'candidate', 'params_sha256': params_sha,
                   'semantic_approval': False, 'human_review_completed': False, 'formal_training': False}
            put(directory / 'consumer.terminal.json', row)
            self.rows[sid] = row
        prefix = q.encoded(self.rows['G']) + q.encoded(self.rows['H'])
        splice = q.encoded(self.rows['C'])[:85] + q.encoded(self.rows['D'])
        self.bad_offset = len(prefix)
        self.empty_offset = len(prefix) + len(splice)
        self.original_raw = prefix + splice + bytes([10]) + q.encoded(self.rows['G'])
        binary(self.production / 'completed.jsonl', self.original_raw)
        self.rules = {self.bad_offset: {'sha256': hashlib.sha256(splice).hexdigest(), 'size': len(splice),
                                       'split': 'train', 'batch_id': 'train-b000001', 'ids': ['C', 'D']}}
        for batch in m.DEFAULT_BATCHES:
            hints = [{'id': sid, 'split': 'train'} for sid in assignment if self.rows[sid]['batch_id'] == batch]
            binary(self.production / 'batches' / batch / 'terminal_index.jsonl', b''.join(q.encoded(r) for r in hints))

    def run(self):
        with patch.dict(m.recovery.KNOWN, self.rules, clear=True), patch.object(m, 'EMPTY_OFFSET', self.empty_offset):
            return m.reconcile(self.production, self.manifest, self.out)

    def terminal(self, sid):
        return self.production / 'sources/train' / sid / 'consumer.terminal.json'


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='terminal-reconciliation-test-', dir='/tmp')
        self.fixture = FullChainFixture(Path(self.temp.name))
        self.addCleanup(self.temp.cleanup)

    def test_four_real_chain_missing_and_complete_denominator(self):
        f = self.fixture
        r = f.run()
        self.assertEqual(r['missing_source_keys'], ['train/A', 'train/B', 'train/E', 'train/F'])
        self.assertEqual(r['new_reconciled_source_count'], 4)
        self.assertEqual(r['selected_original_member_count'], 2000)
        self.assertEqual(r['original_journal_unique_sources'], 4)
        self.assertEqual(r['terminal_index_unique_sources'], 8)
        self.assertEqual(r['full_original_manifest_total'], 222939)
        self.assertEqual([b['raw_size'] for b in r['badlines']], [f.rules[f.bad_offset]['size'], 1])
        self.assertFalse(r['entire_dataset_success_claimed'])
        self.assertEqual((f.production / 'completed.jsonl').read_bytes(), f.original_raw)
        for sid in ('A', 'B', 'E', 'F'):
            row = json.loads((f.out / 'records/train' / (sid + '.json')).read_bytes())
            self.assertEqual(row['hint'], f.rows[sid])
            self.assertIsNone(row['original_byte_offset'])
            self.assertFalse(row['training_eligible'])
            self.assertTrue((f.out / 'receipts/train' / (sid + '.json')).exists())

    def test_replay_and_original_append_do_not_duplicate_sources(self):
        f = self.fixture
        first = f.run()
        before = {p.name: p.read_bytes() for p in (f.out / 'records/train').glob('*.json')}
        second = f.run()
        self.assertEqual(second['new_reconciled_source_count'], 0)
        with (f.production / 'completed.jsonl').open('ab') as stream:
            stream.write(q.encoded(f.rows['A']))
        third = f.run()
        self.assertEqual(third['new_reconciled_source_count'], 0)
        self.assertEqual(third['cumulative_reconciled_unique_sources'], 4)
        self.assertEqual(third['missing_source_keys'], ['train/B', 'train/E', 'train/F'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in (f.out / 'records/train').glob('*.json')})
        lines = [json.loads(l) for l in (f.out / 'records.jsonl').read_bytes().splitlines()]
        self.assertEqual(len({v['source_dedup_key'] for v in lines}), 4)
        self.assertNotEqual(first['snapshot_ref']['path'], third['snapshot_ref']['path'])

    def test_missing_journal_not_inferred_from_emptyline(self):
        f = self.fixture
        r = f.run()
        blank = next(b for b in r['badlines'] if b['kind'] == 'verified_empty_physical_row')
        self.assertEqual(blank['contained_source_records'], 0)
        self.assertEqual(blank['reconstructed_source_records'], 0)
        for p in (f.out / 'records/train').glob('*.json'):
            self.assertIsNone(json.loads(p.read_bytes())['original_byte_offset'])

    def test_wrong_terminal_vs_journal_rejected(self):
        f = self.fixture
        row = copy.deepcopy(f.rows['G'])
        row['status'] = 'incorrect'
        put(f.terminal('G'), row)
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_wrong_joint_sha_rejected(self):
        f = self.fixture
        path = Path(f.joints['A']['path'])
        value = json.loads(path.read_bytes())
        value['visual_status'] = 'tampered'
        put(path, value)
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_wrong_csv_bytes_rejected(self):
        f = self.fixture
        (f.root / 'original.csv').write_bytes(b'tampered,csv\n')
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_wrong_split_rejected(self):
        f = self.fixture
        path = f.production / 'batches/train-b000001/terminal_index.jsonl'
        raw = path.read_bytes().replace(b'"train"', b'"val"', 1)
        path.write_bytes(raw)
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_wrong_media_bytes_rejected(self):
        f = self.fixture
        Path(f.media['A']['video_path']['path']).write_bytes(b'changed-video')
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_wrong_plan_bytes_rejected(self):
        f = self.fixture
        Path(f.plans['A']['path']).write_bytes(b'{}\n')
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_unknown_journal_badline_keeps_raw_and_reason(self):
        f = self.fixture
        with (f.production / 'completed.jsonl').open('ab') as stream:
            stream.write(b'{unknown-badline}\n')
        with self.assertRaises(m.recovery.UnrecoverableLine):
            f.run()
        rejection = list((f.out / 'snapshots').glob('*/rejection.json'))
        self.assertEqual(len(rejection), 1)
        self.assertIn('Unreviewed malformed journal row', json.loads(rejection[0].read_bytes())['reason'])
        self.assertFalse((f.out / 'records/train/A.json').exists())

    def test_unknown_index_badline_rejected(self):
        f = self.fixture
        with (f.production / 'batches/train-b000001/terminal_index.jsonl').open('ab') as stream:
            stream.write(b'\n')
        with self.assertRaises(m.recovery.UnrecoverableLine):
            f.run()
        self.assertTrue(list((f.out / 'snapshots').glob('*/rejection.json')))

    def test_previous_prefix_modification_rejected_even_same_length(self):
        f = self.fixture
        f.run()
        path = f.production / 'completed.jsonl'
        raw = bytearray(path.read_bytes())
        raw[3] ^= 1
        path.write_bytes(raw)
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_previous_stream_inode_replacement_rejected(self):
        f = self.fixture
        f.run()
        path = f.production / 'completed.jsonl'
        replacement = f.production / 'new.completed.jsonl'
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_virtual_original_offset_rejected(self):
        f = self.fixture
        f.run()
        path = f.out / 'records/train/A.json'
        value = json.loads(path.read_bytes())
        value['original_byte_offset'] = 1192186
        put(path, value)
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_terminal_mutation_after_first_reconciliation_rejected(self):
        f = self.fixture
        f.run()
        row = copy.deepcopy(f.rows['A'])
        row['joint_complete'] = False
        put(f.terminal('A'), row)
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_concurrent_original_prefix_change_rejected_before_record_publish(self):
        f = self.fixture
        original = m.complete_evidence
        changed = False
        def check(evidence, row):
            nonlocal changed
            result = original(evidence, row)
            if not changed:
                changed = True
                path = f.production / 'completed.jsonl'
                raw = bytearray(path.read_bytes())
                raw[3] ^= 1
                path.write_bytes(raw)
            return result
        with patch.object(m, 'complete_evidence', check), self.assertRaises(q.SourceChanged):
            f.run()
        self.assertFalse((f.out / 'records/train/A.json').exists())

    def test_first_verified_full_sha_cache_refuses_changed_stat(self):
        path = self.fixture.root / 'cache-artifact'
        path.write_bytes(b'original')
        cache = m.ByteCache()
        self.assertEqual(cache.sha(path), hashlib.sha256(b'original').hexdigest())
        path.write_bytes(b'tampered-longer')
        with self.assertRaises(q.SourceChanged):
            cache.sha(path)

    def test_partial_index_tail_rejected(self):
        f = self.fixture
        path = f.production / 'batches/train-b000003/terminal_index.jsonl'
        with path.open('ab') as stream:
            stream.write(b'{')
        with self.assertRaises(m.recovery.StreamBusy):
            f.run()

    def test_immutable_receipt_tamper_rejected(self):
        f = self.fixture
        f.run()
        path = f.out / 'receipts/train/A.json'
        value = json.loads(path.read_bytes())
        value['original_byte_offset'] = 1192186
        put(path, value)
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_derived_projection_duplicate_rejected(self):
        f = self.fixture
        f.run()
        path = f.out / 'records.jsonl'
        raw = path.read_bytes()
        path.write_bytes(raw + raw.splitlines(keepends=True)[0])
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_changed_known_empty_row_rejected(self):
        f = self.fixture
        path = f.production / 'completed.jsonl'
        raw = path.read_bytes()
        path.write_bytes(raw[:f.empty_offset] + b' ' + raw[f.empty_offset:])
        with self.assertRaises(m.recovery.UnrecoverableLine):
            f.run()

    def test_non_full_manifest_rejected(self):
        f = self.fixture
        summary = json.loads(f.manifest.read_bytes())
        summary['batch_size'] = 499
        put(f.manifest, summary)
        with self.assertRaises(q.SourceChanged):
            f.run()

    def test_out_cannot_overlap_original_production(self):
        f = self.fixture
        with self.assertRaises(ValueError):
            m.reconcile(f.production, f.manifest, f.production / 'wrong')


@unittest.skipUnless(os.environ.get('TERMINAL_RECONCILIATION_REAL') == '1', 'Opt-in real CPU SHA read, outputs only /tmp')
class OriginalProductionTests(unittest.TestCase):
    def test_actual_missing_terminals_full_chain_snapshot_and_review_namespace(self):
        root = m.RT.parent
        production = root / 'curated222939_production_v1'
        manifest = root / 'curated222939_inputs_v1/manifest.summary.json'
        with tempfile.TemporaryDirectory(prefix='terminal-reconciliation-real-test-', dir='/tmp') as temporary:
            out = Path(temporary)
            result = m.reconcile(production, manifest, out)
            self.assertFalse(result['entire_dataset_success_claimed'])
            self.assertEqual(result['selected_original_member_count'], 2000)
            snapshot = json.loads(Path(result['snapshot_ref']['path']).read_bytes())
            before = m.verify_prefix(snapshot['original_journal_prefix'])
            with m.verified_hash_cache():
                evidence = m.recovery.Evidence(production, manifest)
                journal, _, _ = m.decode_journal(before, production, evidence)
                indexed = {}
                for batch, ref in snapshot['terminal_index_prefixes'].items():
                    indexed.update(m.decode_index(m.verify_prefix(ref), production, batch, evidence)[0])
                self.assertEqual(result['missing_source_keys'], sorted(set(indexed) - set(journal)))
                for sid in ('BBucGIEtwcrDIngP', 'dBloPhzUQlCooRtD', 'pbUXtdkTIWrvjeXw', 'tYhkFFYDNlaEJimA'):
                    key = 'train/' + sid
                    self.assertIn(key, indexed)
                    self.assertNotIn(key, journal)
                    value = json.loads((out / 'records' / (key + '.json')).read_bytes())
                    m.check_record(value, production, evidence)
                    self.assertIsNone(value['original_byte_offset'])
                from main_review_watch_v5 import Membership, terminal_record
                membership = Membership(manifest)
                for path in (out / 'records/train').glob('*.json'):
                    value = json.loads(path.read_bytes())
                    terminal_record(production, value['hint'], membership)
            self.assertEqual(m.verify_prefix(snapshot['original_journal_prefix']), before)
            self.assertEqual((Path(result['snapshot_ref']['path']).parent / 'completed.prefix.raw').read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
