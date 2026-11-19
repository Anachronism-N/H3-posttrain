#!/usr/bin/env python3
"""CPU-only exact incident/cursor/immutable budget and concurrent-write tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import curated_repair_collector_recovery_v1 as r
q = r.q


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.production = self.root / 'production'
        self.out = self.root / 'repair'
        self.production.mkdir()
        self.out.mkdir()
        self.csv = self.root / 'train.csv'
        self.csv.write_bytes(b'one immutable original row\n')
        self.member_rows = []
        self.split, self.batch = 'train', 'train-b000001'
        self.revision = 'repair-20261025-v1'
        self.bar = self.root / 'ms_bar_annotation.py'
        self.bar.write_text("def params(args, code):\n    return {'model_path': args.model, 'code_sha256': code}\n")
        self.code = {str(self.bar): q.sha(self.bar)}
        self.configuration = {'model_path': '/not-loaded/CPU-test', 'code_sha256': self.code,
                              'style_module_sha256': 'fake_style_value_not_a_ref'}
        self.params_sha = r.visual_digest(self.configuration)
        for sid in ('Before', 'A', 'B'):
            video, audio = self.root / (sid + '.mp4'), self.root / (sid + '.wav')
            video.write_bytes((sid + 'video').encode())
            audio.write_bytes((sid + 'audio').encode())
            self.member_rows.append({'id': sid, 'source_id': sid, 'split': self.split, 'batch_id': self.batch,
                'source_ordinal': len(self.member_rows), 'video_path': str(video), 'audio_path': str(audio),
                'source_refs': {'csv': {**q.identity(self.csv), 'csv_line': len(self.member_rows) + 2,
                                       'raw_row_sha256': hashlib.sha256(sid.encode()).hexdigest()}}})
        self.manifest = self.root / 'train-b000001.manifest.jsonl'
        self.manifest.write_bytes(b''.join(q.encoded(row) for row in self.member_rows))
        self.summary = self.root / 'manifest.summary.json'
        q.save_once(self.summary, {'total': 3, 'batches': [{'batch_id': self.batch, 'split': self.split,
                      'count': 3, **q.identity(self.manifest)}]})
        q.save_once(self.production / 'runs/v0-test/protocol.json', {'mode': 'consumer',
                     'configuration': self.configuration, 'manifest_summary': q.identity(self.summary)})
        self.rows = {}
        for member in self.member_rows:
            member = json.loads(q.encoded(member))
            sid = member['id']
            directory = self.production / 'sources/train' / sid
            binding = {'id': sid, 'video_sha256': q.sha(member['video_path']),
                       'audio_sha256': q.sha(member['audio_path'])}
            plan = {'id': sid, 'manifest_record': member, 'binding': binding,
                    'source_refs': {'manifest': q.identity(self.manifest), 'original_csv': member['source_refs']['csv']}}
            q.save_once(directory / 'bar_plan.json', plan)
            plan = q.load(directory / 'bar_plan.json')
            producer = {'id': sid, 'split': self.split, 'batch_id': self.batch, 'status': 'ready',
                        'source_binding': binding, 'plan_ref': q.identity(directory / 'bar_plan.json')}
            q.save_once(directory / 'producer.terminal.json', producer)
            visual_root = self.production / 'visual/shard-0000/sources' / sid
            visual_root.mkdir(parents=True)
            receipt_path = visual_root.parent.parent / 'load.receipt.json'
            q.save_once(receipt_path, {'code_sha256': self.code})
            visual_binding = {'source_binding': binding, 'plan_sha256': producer['plan_ref']['sha256'],
                'full_source_row_sha256': r.visual_digest(plan), 'raw_line_sha256': producer['plan_ref']['sha256'],
                'manifest_record_sha256': r.visual_digest(member), 'params_sha256': self.params_sha}
            source = {'sha256': hashlib.sha256(sid.encode()).hexdigest(),
                      'files': {'video_path': q.identity(member['video_path']),
                                'audio_path': q.identity(member['audio_path']), 'nlf_path': None}}
            q.save_once(visual_root / 'binding.json', visual_binding)
            q.save_once(visual_root / 'source.identity.json', source)
            q.save_once(visual_root / 'source.refs.json',
                        [{k: ref[k] for k in ('path', 'sha256')} for ref in plan['source_refs'].values()])
            visual = {'id': sid, 'manifest_record': member, 'binding': visual_binding,
                      'window_observations': [], 'final': {'status': 'candidate'}}
            q.save_once(visual_root / 'results/visual.json', visual)
            joint = {'id': sid, 'source_id': sid, 'split': self.split, 'batch_id': self.batch,
                'source_binding': visual_binding, 'source_identity': source, 'source_refs': member['source_refs'],
                'visual_result_ref': q.identity(visual_root / 'results/visual.json'),
                'joint_complete': sid != 'A', 'all_windows_nonempty': sid != 'A',
                'visual_status': 'candidate', 'style_status': 'resolved', 'dance_style_label': 'hiphop',
                'style_result': {'status': 'resolved'}}
            q.save_once(directory / 'joint-results/joint.json', joint)
            terminal = {'id': sid, 'split': self.split, 'batch_id': self.batch, 'status': 'joint_candidate',
                        'joint_complete': sid != 'A', 'visual_status': 'candidate', 'style_status': 'resolved',
                        'params_sha256': self.params_sha, 'joint_candidate_ref': q.identity(directory / 'joint-results/joint.json')}
            q.save_once(directory / 'consumer.terminal.json', terminal)
            self.rows[sid] = q.load(directory / 'consumer.terminal.json')
        self.before = q.encoded(self.rows['Before'])
        a, b = q.encoded(self.rows['A']), q.encoded(self.rows['B'])
        cut = a.index(b'"path":"') + len(b'"path":"') + 10
        self.badline = a[:cut] + b
        self.queue = self.production / 'completed.jsonl'
        self.queue.write_bytes(self.before + self.badline)
        self.offset = len(self.before)
        self.rules = {self.offset: {'sha256': hashlib.sha256(self.badline).hexdigest(),
            'size': len(self.badline), 'split': self.split, 'batch_id': self.batch, 'ids': ['A', 'B']}}
        stat = self.queue.stat()
        self.origin = {'input_path': str(self.queue), 'input_identity': [stat.st_dev, stat.st_ino],
            'offset': self.offset, 'seen': {'train/Before': q.digest(self.rows['Before'])},
            'counts': {'joint_complete': 1}, 'stage_counts': {}}
        q.save_once(self.out / 'collector.state.json', self.origin)
        q.save_once(self.out / 'attempt-budget/train/A.json', {'source_key': 'train/A',
                      'maximum_extra_source_attempts': 1, 'extra_source_attempt': 1})
        self.budget_bytes = (self.out / 'attempt-budget/train/A.json').read_bytes()
        self.origin_bytes = (self.out / 'collector.state.json').read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def collect(self):
        return r.collect(self.production, self.out, self.revision, self.summary, stable_seconds=0, rules=self.rules)

    def test_two_sources_real_classification_and_resume_budget(self):
        result = self.collect()
        self.assertEqual(result['counts'], {'joint_complete': 2, 'incomplete_or_failed': 1})
        self.assertEqual(result['new_queued_sources'], 1)
        self.assertEqual(result['input_byte_offset'], len(self.before + self.badline))
        self.assertEqual(q.load(self.out / 'jobs/train/A.json')['completed_row_sha256'], q.digest(self.rows['A']))
        self.assertFalse((self.out / 'jobs/train/B.json').exists())
        hints = [json.loads(line) for line in (self.out / 'collector-recovery-v1/derived-hints.jsonl').read_bytes().splitlines()]
        self.assertEqual([h['hint'] for h in hints], [self.rows['A'], self.rows['B']])
        self.assertEqual(Path(hints[0]['badline_ref']['path']).read_bytes(), self.badline)
        self.assertEqual(hints[0]['completed_row_sha256'], q.digest(self.rows['A']))
        self.assertEqual(hints[0]['evidence']['consumer_terminal_ref']['sha256'], q.sha(self.production / 'sources/train/A/consumer.terminal.json'))
        ready = (self.out / 'ready.jsonl').read_bytes()
        resumed = self.collect()
        self.assertEqual(resumed['new_queued_sources'], 0)
        self.assertEqual(resumed['processed_source_hints'], 0)
        self.assertEqual((self.out / 'ready.jsonl').read_bytes(), ready)
        self.assertEqual((self.out / 'collector.state.json').read_bytes(), self.origin_bytes)
        self.assertEqual((self.out / 'attempt-budget/train/A.json').read_bytes(), self.budget_bytes)

    def test_unknown_offset_or_sha_rejected_with_evidence_cursor(self):
        self.rules = {}
        with self.assertRaises(r.UnrecoverableLine), patch.object(r.Evidence, 'terminal', side_effect=AssertionError('must not fetch guessed ID')):
            self.collect()
        state = q.load(self.out / 'collector.recovery.state.json')
        self.assertEqual(state['offset'], self.offset)
        self.assertEqual(state['seen'], self.origin['seen'])
        self.assertEqual(len(list((self.out / 'collector-recovery-v1/badlines').glob('*.raw'))), 1)
        self.assertFalse((self.out / 'jobs/train/A.json').exists())

    def test_reviewed_sha_cannot_admit_unknown_source(self):
        self.rules[self.offset]['ids'][0] = 'NotInManifest'
        with self.assertRaises((r.UnrecoverableLine, FileNotFoundError)):
            self.collect()
        self.assertEqual(q.load(self.out / 'collector.recovery.state.json')['offset'], self.offset)

    def test_partial_line_waits_then_two_sources_recovered(self):
        self.queue.write_bytes(self.before + self.badline[:-1])
        result = self.collect()
        self.assertEqual(result['waiting'], 'partial_line')
        self.assertEqual(result['input_byte_offset'], self.offset)
        with self.queue.open('ab') as stream:
            stream.write(b'\n')
        self.assertEqual(self.collect()['new_queued_sources'], 1)

    def test_existing_same_row_job_reused_without_budget_or_revision_change(self):
        job = q.classify(self.rows['A'], self.production)
        job.update(global_revision=self.revision, queue_root=str(self.out))
        q.save_once(self.out / 'jobs/train/A.json', job)
        original = (self.out / 'jobs/train/A.json').read_bytes()
        result = self.collect()
        self.assertEqual(result['new_queued_sources'], 0)
        self.assertEqual((self.out / 'jobs/train/A.json').read_bytes(), original)
        self.assertEqual((self.out / 'attempt-budget/train/A.json').read_bytes(), self.budget_bytes)

    def test_normal_terminal_hint_mismatch_rejected(self):
        wrong = copy.deepcopy(self.rows['Before'])
        wrong['joint_complete'] = False
        self.queue.write_bytes(self.before + q.encoded(wrong))
        with self.assertRaises(q.SourceChanged):
            self.collect()
        self.assertEqual(q.load(self.out / 'collector.recovery.state.json')['offset'], self.offset)

    def test_origin_state_drift_rejected(self):
        self.collect()
        changed = copy.deepcopy(self.origin)
        changed['counts']['joint_complete'] = 100
        q.replace_state(self.out / 'collector.state.json', changed)
        with self.assertRaises(q.SourceChanged):
            self.collect()

    def test_prefix_mutation_and_truncation_rejected(self):
        self.collect()
        data = bytearray(self.queue.read_bytes())
        data[20] = ord('Z')
        self.queue.write_bytes(data)
        with self.assertRaises(q.SourceChanged):
            self.collect()
        self.queue.write_bytes(self.before)
        with self.assertRaises(q.SourceChanged):
            self.collect()

    def test_inherited_seen_mismatch_rejected(self):
        changed = copy.deepcopy(self.origin)
        changed['seen']['train/Before'] = '0' * 64
        q.replace_state(self.out / 'collector.state.json', changed)
        with self.assertRaises(q.SourceChanged):
            self.collect()
        self.assertFalse((self.out / 'collector.recovery.state.json').exists())

    def test_concurrent_badline_snapshot_waits_no_false_recovery(self):
        def writer(_):
            with self.queue.open('ab') as stream:
                stream.write(b' ')  # Shared writer changed size between stable snapshots.
        with patch.object(r.time, 'sleep', side_effect=writer):
            result = self.collect()
        self.assertEqual(result['waiting'], 'badline_snapshot_busy')
        self.assertEqual(result['input_byte_offset'], self.offset)
        self.assertEqual(result['new_queued_sources'], 0)
        self.assertFalse((self.out / 'jobs/train/A.json').exists())

    def test_terminal_joint_sha_or_binding_mutation_rejected(self):
        joint_path = Path(self.rows['A']['joint_candidate_ref']['path'])
        joint_path.write_bytes(joint_path.read_bytes() + b' ')
        with self.assertRaises(q.SourceChanged):
            self.collect()
        self.assertFalse((self.out / 'jobs/train/A.json').exists())

    def test_newly_read_bytes_overwrite_before_checkpoint_rejected(self):
        line = q.encoded(self.rows['B'])
        self.queue.write_bytes(self.before + line)
        original_admit = r.admit
        def overwrite_after_admit(*args, **kwargs):
            result = original_admit(*args, **kwargs)
            changed = line.replace(b'"resolved"', b'"changed_"')
            self.assertEqual(len(changed), len(line))
            self.queue.write_bytes(self.before + changed)
            return result
        with patch.object(r, 'admit', side_effect=overwrite_after_admit):
            with self.assertRaisesRegex(q.SourceChanged, 'Read source bytes changed'):
                self.collect()
        state = q.load(self.out / 'collector.recovery.state.json')
        self.assertEqual(state['offset'], self.offset)
        self.assertEqual(state['seen'], self.origin['seen'])

    def test_unknown_later_badline_checkpoints_previous_coverage(self):
        with self.queue.open('ab') as stream:
            stream.write(b'{"id":"NotInManifest",bad}\n')
        with self.assertRaises(r.UnrecoverableLine):
            self.collect()
        state = q.load(self.out / 'collector.recovery.state.json')
        self.assertEqual(state['offset'], len(self.before + self.badline))
        self.assertEqual(len(state['seen']), 3)
        self.assertEqual(state['counts'], {'joint_complete': 2, 'incomplete_or_failed': 1})


if __name__ == '__main__':
    unittest.main(verbosity=2)
