"""CPU-only raw/verified separation and original byte-watermark regression tests."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('watch_v6_tests', RT / 'main_review_watch_v6.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class Members:
    descriptors = {'train-b000000': {'split': 'train'}, 'train-b000001': {'split': 'train'}}
    def batch(self, bid):
        return {'first': {'id': 'first'}, 'second': {'id': 'second'}}


def row(sid, batch='train-b000000'):
    return {'id': sid, 'split': 'train', 'batch_id': batch, 'status': 'joint_candidate'}


def encoded(v):
    return (json.dumps(v) + '\n').encode()


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.root.joinpath('completed.jsonl').write_bytes(encoded(row('first')) + encoded(row('second', 'train-b000001')))
        self.index = w.RawIndex(self.root, Members())

    def tearDown(self):
        self.tmp.cleanup()

    def test_pending_masks_all_source_labels_completed(self):
        base = {'all_source_labels_completed': True, 'sampled_sources_reviewed': 0}
        with patch.object(w.v5, 'context_progress', return_value=base.copy()):
            self.index.pending = {'offset': 0, 'reason': 'unknown_malformed_line'}
            self.assertFalse(w.progress(self.index, self.root, 'test', set(), 0)['all_source_labels_completed'])
        with patch.object(w.v5, 'context_progress', return_value=base.copy()):
            self.index.pending = None
            self.index.batch_pending = {'train-b000000': {'reason': 'unpublished'}}
            self.assertFalse(w.progress(self.index, self.root, 'test', set(), 0)['all_source_labels_completed'])
        with patch.object(w.v5, 'context_progress', return_value=base.copy()):
            self.index.batch_pending = {}
            self.assertTrue(w.progress(self.index, self.root, 'test', set(), 0)['all_source_labels_completed'])

    def test_builder_worker_export_api(self):
        tree = ast.parse((RT / 'build_main_review_v6_fresh.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'worker':
                self.assertTrue(hasattr(w, node.attr), 'Missing worker API: ' + node.attr)
        self.assertTrue(callable(w.v5.checked))

    def test_raw_index_does_not_verify_or_advance_consumer(self):
        with patch.object(w, 'terminal_record', side_effect=AssertionError('raw scan cannot read terminal metadata')):
            self.assertEqual(self.index.scan(), {'train-b000000', 'train-b000001'})
        self.assertEqual(sum(len(x) for x in self.index.raw.values()), 2)
        self.assertFalse(self.index.completed)
        self.assertEqual(self.index.offset, 0)
        self.assertGreater(self.index.indexed_offset, 0)

    def test_batch_laziness_and_contiguous_watermark(self):
        self.index.scan()
        called = []
        def verify(prod, value, members):
            called.append(value['id'])
            return value
        with patch.object(w, 'terminal_record', side_effect=verify):
            self.assertTrue(self.index.verify('train-b000001'))
            self.assertEqual(called, ['second'])
            self.assertEqual(self.index.offset, 0)
            self.assertTrue(self.index.verify('train-b000000'))
        self.assertEqual(self.index.offset, self.index.indexed_offset)

    def test_same_inode_changed_prefix_rejected(self):
        self.index.scan()
        p = self.root / 'completed.jsonl'
        data = p.read_bytes()
        p.write_bytes(data.replace(b'first', b'FIRST'))
        with self.assertRaises(ValueError):
            self.index.scan()

    def test_unknown_bad_line_stops_without_skip(self):
        p = self.root / 'completed.jsonl'
        first = encoded(row('first'))
        p.write_bytes(first + b'{bad}\n' + encoded(row('second')))
        self.index.scan()
        self.assertEqual(self.index.indexed_offset, len(first))
        self.assertEqual(self.index.pending['reason'], 'unknown_malformed_line')
        self.assertNotIn('second', self.index.raw['train-b000000'])
        self.assertEqual(self.index.offset, 0)

    def test_half_line_retried_at_original_byte_boundary(self):
        p = self.root / 'completed.jsonl'
        first = encoded(row('first'))
        second = encoded(row('second'))
        p.write_bytes(first + second[:-1])
        self.index.scan()
        self.assertEqual(self.index.indexed_offset, len(first))
        self.assertEqual(self.index.pending['reason'], 'partial_original_line')
        with p.open('ab') as f:
            f.write(b'\n')
        self.index.scan()
        self.assertIn('second', self.index.raw['train-b000000'])
        self.assertEqual(self.index.offset, 0)

    def test_missing_terminal_is_pending_without_partial_commit(self):
        self.index.scan()
        with patch.object(w, 'terminal_record', side_effect=FileNotFoundError('unpublished')):
            self.assertFalse(self.index.verify('train-b000000'))
        self.assertFalse(self.index.completed['train-b000000'])
        self.assertIn('train-b000000', self.index.batch_pending)

    def test_source_sha_failure_is_hard_failure(self):
        self.index.scan()
        with patch.object(w, 'terminal_record', side_effect=ValueError('source SHA mismatch')):
            with self.assertRaises(ValueError):
                self.index.verify('train-b000000')
        self.assertEqual(self.index.offset, 0)

    def test_edition_indices_and_duplicate_revision_do_not_mix(self):
        other = self.root / 'other'
        other.mkdir()
        other.joinpath('completed.jsonl').write_bytes(encoded(row('first')))
        second = w.RawIndex(other, Members())
        self.index.scan()
        second.scan()
        with patch.object(w, 'terminal_record', side_effect=lambda p, v, m: {**v, 'origin': str(p)}):
            self.index.verify('train-b000000')
            self.assertFalse(second.completed)
            second.verify('train-b000000')
        self.assertNotEqual(self.index.completed['train-b000000']['first']['origin'], second.completed['train-b000000']['first']['origin'])
        with other.joinpath('completed.jsonl').open('ab') as f:
            f.write(encoded({**row('first'), 'status': 'failed'}))
        with self.assertRaises(ValueError):
            second.scan()

    def test_changed_bytes_between_read_and_commit_rejected(self):
        original = w.recovery.prefix_sha
        calls = 0
        def changed(path, end, identity):
            nonlocal calls
            calls += 1
            if end:
                return '0' * 64
            return original(path, end, identity)
        with patch.object(w.recovery, 'prefix_sha', side_effect=changed):
            with self.assertRaises(ValueError):
                self.index.scan()
        self.assertEqual(self.index.indexed_offset, 0)
        self.assertEqual(self.index.offset, 0)


if __name__ == '__main__':
    unittest.main()
