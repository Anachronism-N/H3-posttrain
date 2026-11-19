"""CPU-only temporary fixtures; never open production media or model weights."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('collector_offset4579239_quarantine_v1.py')
spec = importlib.util.spec_from_file_location('tested_quarantine4579239', PATH)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class ExactQuarantineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='collector4579239-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.production = self.root / 'curated222939_production_v1'
        self.out = self.root / 'curated222939_repair_v1'
        self.production.mkdir()
        self.out.mkdir()
        inputs = self.root / 'curated222939_inputs_v1'
        inputs.mkdir()
        self.manifest = inputs / 'manifest.summary.json'
        self.manifest.write_bytes(w.q.encoded({'total': 0, 'batches': []}))
        self.queue = self.production / 'completed.jsonl'
        with self.queue.open('wb') as stream:
            stream.seek(w.OFFSET - 1)
            stream.write(b'\n' + w.RAW)
        stat = self.queue.stat()
        identity = [stat.st_dev, stat.st_ino]
        prefix = w.r.prefix_sha(self.queue, w.OFFSET, identity)
        self.patches = [patch.object(w, 'RT', self.root / 'runtime'), patch.object(w, 'INPUT_IDENTITY', identity),
                        patch.object(w, 'PREFIX_SHA', prefix)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.revision = 'fixture-revision-v1'
        origin = {'offset': 0, 'input_path': str(self.queue), 'input_identity': identity,
                  'seen': {}, 'counts': {}, 'stage_counts': {}}
        self.origin_path = self.out / 'collector.state.json'
        self.origin_path.write_bytes(w.q.encoded(origin))
        self.state_path = self.out / 'collector.recovery.state.json'
        self.state = {'schema': w.r.SCHEMA, 'offset': w.OFFSET, 'input_path': str(self.queue),
            'input_identity': identity, 'input_prefix_sha256': prefix, 'global_revision': self.revision,
            'origin_state_ref': w.q.identity(self.origin_path), 'manifest_summary_ref': w.q.identity(self.manifest),
            'seen': {'train/existingfixture': 'a' * 64}, 'counts': {'joint_complete': 1},
            'stage_counts': {'rewrite': 2}, 'recovered_badline_offsets': list(w.PRIOR_OFFSETS)}
        self.state_path.write_bytes(w.q.encoded(self.state))
        self.budget = self.out / 'budget.fixture.json'
        self.budget.write_bytes(b'{"spent":7,"remaining":2}\n')
        self.evidence = w.r.Evidence(self.production, self.manifest)
        self.metadata = self.out / 'collector-offset4579239-quarantine-v1'

    def samples(self):
        return w.r.stable_badline(self.queue, w.OFFSET, w.RAW, w.INPUT_IDENTITY, 0)

    def recover(self, samples=None, begin=None, raw=None):
        return w.recover_line(self.production, self.out, w.OFFSET if begin is None else begin,
                              w.RAW if raw is None else raw, self.evidence,
                              self.samples() if samples is None else samples)

    def files(self):
        return {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}

    def test_exact_bytes_offset_and_unknown_association(self):
        before = self.files()
        self.assertEqual(self.recover(), [])
        saved = w.q.load(self.metadata / 'snapshot.json')
        self.assertEqual(saved['physical_byte_begin'], 4579239)
        self.assertEqual(saved['raw_size'], 5)
        self.assertEqual(saved['raw_sha256'], hashlib.sha256(b'te"}\n').hexdigest())
        self.assertEqual((self.metadata / 'badline-4579239.raw').read_bytes(), b'te"}\n')
        self.assertTrue(saved['source_association_unknown'])
        self.assertEqual(saved['fragment_source_association'], 'unknown')
        self.assertEqual(saved['contained_complete_sources'], 0)
        self.assertEqual(saved['reconstructed_source_records'], 0)
        for name in ('source_population_completeness_not_proven', 'lost_sources_need_separate_reconciliation',
                     'no_model_calls', 'no_budget_reset'):
            self.assertTrue(saved[name])
        for name in ('GPU_dispatch_performed', 'formal_training', 'training_eligible', 'semantic_approval'):
            self.assertFalse(saved[name])
        self.assertNotIn('source_key', saved)
        for path, data in before.items():
            self.assertEqual((self.root / path).read_bytes(), data)

    def test_original_collect_owns_cursor_without_seen_count_budget_changes(self):
        before_log, before_origin, before_budget = self.queue.read_bytes(), self.origin_path.read_bytes(), self.budget.read_bytes()
        result = w.r.collect(self.production, self.out, self.revision, self.manifest, 0)
        after = w.q.load(self.state_path)
        self.assertEqual(result['input_byte_offset'], w.OFFSET + 5)
        self.assertEqual(result['processed_source_hints'], 0)
        self.assertEqual(result['new_queued_sources'], 0)
        self.assertEqual(result['waiting'], 'tail')
        self.assertFalse(result['GPU_dispatch_performed'])
        for name in ('seen', 'counts', 'stage_counts'):
            self.assertEqual(after[name], self.state[name])
        self.assertEqual(after['recovered_badline_offsets'], list(w.PRIOR_OFFSETS) + [w.OFFSET])
        self.assertEqual(after['input_prefix_sha256'], hashlib.sha256(before_log).hexdigest())
        self.assertEqual(self.queue.read_bytes(), before_log)
        self.assertEqual(self.origin_path.read_bytes(), before_origin)
        self.assertEqual(self.budget.read_bytes(), before_budget)
        self.assertFalse((self.out / 'ready.jsonl').exists())
        self.assertFalse((self.out / 'jobs').exists())
        self.assertEqual(w.check_only(self.production, self.out, self.revision, self.manifest, 0)['input_byte_offset'], w.OFFSET + 5)

    def test_wrong_raw_rejected_without_any_writes(self):
        before = self.files()
        with self.assertRaises(w.r.UnrecoverableLine):
            self.recover(raw=b'xx"}\n')
        self.assertEqual(self.files(), before)

    def test_wrong_offset_same_raw_rejected_without_writes(self):
        before = self.files()
        with self.assertRaises(w.r.UnrecoverableLine):
            self.recover(begin=w.OFFSET + 1)
        self.assertEqual(self.files(), before)

    def test_same_raw_at_prior_offset_not_quarantined_as_new_incident(self):
        with self.assertRaises(w.r.UnrecoverableLine):
            w.verify_incident(self.production, 2053650, w.RAW, self.samples())

    def test_three_equal_snapshots_required(self):
        for mutate in ('missing', 'sha', 'identity'):
            samples = copy.deepcopy(self.samples())
            if mutate == 'missing':
                samples.pop()
            elif mutate == 'sha':
                samples[1]['line_sha256'] = '0' * 64
            else:
                samples[1]['identity'][3] += 1
            with self.assertRaises(w.r.StreamBusy):
                self.recover(samples=samples)
        self.assertFalse(self.metadata.exists())

    def test_snapshot_wrong_inode_rejected(self):
        samples = self.samples()
        for sample in samples:
            sample['identity'][1] += 1
        with self.assertRaises(w.q.SourceChanged):
            self.recover(samples=samples)
        self.assertFalse(self.metadata.exists())

    def test_truncated_snapshot_rejected(self):
        samples = self.samples()
        for sample in samples:
            sample['identity'][2] = w.OFFSET + 4
        with self.assertRaises(w.q.SourceChanged):
            self.recover(samples=samples)

    def test_different_namespace_rejected_without_writes(self):
        before = self.files()
        for production, out in ((self.root / 'other', self.out), (self.production, self.root / 'other')):
            with self.assertRaises(w.q.SourceChanged):
                w.recover_line(production, out, w.OFFSET, w.RAW, self.evidence, self.samples())
        self.assertEqual(self.files(), before)

    def test_unknown_incident_failclosed_without_general_raw_capture(self):
        before = self.files()
        with patch.object(w, 'old_recover_line', side_effect=AssertionError('must not delegate unknown')):
            with self.assertRaises(w.r.UnrecoverableLine):
                self.recover(begin=w.OFFSET + 100, raw=b'new malformed\n')
        self.assertEqual(self.files(), before)

    def test_known_incidents_delegate_saved_handler(self):
        with patch.object(w, 'old_recover_line', return_value=['delegated']) as handler:
            for begin in w.PRIOR_OFFSETS:
                self.assertEqual(self.recover(begin=begin), ['delegated'])
            self.assertEqual(handler.call_count, len(w.PRIOR_OFFSETS))
        self.assertFalse(self.metadata.exists())

    def test_check_only_is_readonly_and_has_no_collect_or_write_calls(self):
        before = self.files()
        with patch.object(w.r, 'collect', side_effect=AssertionError('collect forbidden')), \
             patch.object(w.q, 'save_once', side_effect=AssertionError('metadata writes forbidden')), \
             patch.object(w.q, 'replace_state', side_effect=AssertionError('state writes forbidden')), \
             patch.object(w.r, 'save_raw_once', side_effect=AssertionError('raw writes forbidden')):
            result = w.check_only(self.production, self.out, self.revision, self.manifest, 0)
        self.assertTrue(result['check_only'])
        self.assertFalse(result['state_modified'])
        self.assertFalse(result['metadata_written'])
        self.assertEqual(self.files(), before)
        self.assertFalse((self.out / 'collector.lock').exists())

    def test_check_only_wrong_revision_and_manifest_namespace_rejected(self):
        with self.assertRaises(w.q.SourceChanged):
            w.check_only(self.production, self.out, 'other-revision', self.manifest, 0)
        other = self.root / 'different-manifest.json'
        other.write_bytes(self.manifest.read_bytes())
        with self.assertRaises(w.q.SourceChanged):
            w.check_only(self.production, self.out, self.revision, other, 0)

    def test_prefix_change_is_rejected(self):
        with self.queue.open('r+b') as stream:
            stream.seek(10)
            stream.write(b'x')
        with self.assertRaises(w.q.SourceChanged):
            self.recover()
        self.assertFalse(self.metadata.exists())

    def test_snapshot_replay_preserves_first_evidence_when_stream_grows(self):
        self.recover()
        before = {path.name: path.read_bytes() for path in self.metadata.iterdir()}
        with self.queue.open('ab') as stream:
            stream.write(b'\n')
        self.recover()
        self.assertEqual({path.name: path.read_bytes() for path in self.metadata.iterdir()}, before)

    def test_receipt_unknown_association_cannot_be_rewritten(self):
        self.recover()
        path = self.metadata / 'snapshot.json'
        value = w.q.load(path)
        value['source_association_unknown'] = False
        path.write_bytes(w.q.encoded(value))
        with self.assertRaises(w.q.SourceChanged):
            self.recover()


if __name__ == '__main__':
    unittest.main()
