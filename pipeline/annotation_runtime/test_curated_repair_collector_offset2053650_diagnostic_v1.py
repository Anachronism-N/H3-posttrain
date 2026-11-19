#!/usr/bin/env python3
"""CPU-only positive diagnosis and fail-closed exact stream guards."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import curated_repair_collector_offset2053650_diagnostic_v1 as d


class IncidentDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / 'completed.jsonl'
        self.prefix = b'original prefix\n'
        self.path.write_bytes(self.prefix + d.RAW)
        stat = self.path.stat()
        self.offset = len(self.prefix)
        self.identity = [stat.st_dev, stat.st_ino]
        self.patches = [patch.object(d, 'OFFSET', self.offset),
                        patch.object(d, 'INPUT_IDENTITY', self.identity),
                        patch.object(d, 'PREFIX_SHA', hashlib.sha256(self.prefix).hexdigest())]
        for item in self.patches:
            item.start()
        self.protected = {}
        for name in ('collector.state.json', 'collector.recovery.state.json', 'seen.json',
                     'attempt-budget.json', 'jobs.json', 'main.log'):
            path = self.root / name
            path.write_bytes(b'original protected bytes\n')
            self.protected[path] = path.read_bytes()

    def tearDown(self):
        for path, content in self.protected.items():
            self.assertEqual(path.read_bytes(), content)
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def verify(self):
        return d.verify_incident(self.path, self.offset, 0)

    def test_exact_positive_is_diagnosis_not_source_row(self):
        before = self.path.read_bytes()
        result = self.verify()
        self.assertEqual(result['raw_hex'], '7465227d0a')
        self.assertEqual(result['byte_begin'], self.offset)
        self.assertEqual(result['byte_end'], self.offset + 5)
        self.assertEqual(len(result['stable_badline_snapshots']), 3)
        self.assertNotIn('hint', result)
        self.assertEqual(self.path.read_bytes(), before)

    def test_wrong_offset_rejected_before_stream_read(self):
        with patch.object(d.r, 'prefix_sha', side_effect=AssertionError('must not read unknown offset')):
            with self.assertRaises(d.r.UnrecoverableLine):
                d.verify_incident(self.path, self.offset + 1, 0)

    def test_wrong_byte_rejected(self):
        self.path.write_bytes(self.prefix + b'teX}\n')
        with self.assertRaises(d.r.UnrecoverableLine):
            self.verify()

    def test_blank_not_admitted(self):
        self.path.write_bytes(self.prefix + b'\n')
        with self.assertRaises(d.r.UnrecoverableLine):
            self.verify()

    def test_truncated_line_rejected(self):
        self.path.write_bytes(self.prefix + d.RAW[:-1])
        with self.assertRaises(d.r.UnrecoverableLine):
            self.verify()

    def test_truncated_prefix_rejected(self):
        self.path.write_bytes(self.prefix[:-1])
        with self.assertRaises(d.r.q.SourceChanged):
            self.verify()

    def test_prefix_mutation_rejected(self):
        self.path.write_bytes(b'X' + self.prefix[1:] + d.RAW)
        with self.assertRaises(d.r.q.SourceChanged):
            self.verify()

    def test_legal_tail_append_allowed(self):
        with self.path.open('ab') as stream:
            stream.write(b'{"another":"source"}\n')
        result = self.verify()
        self.assertEqual(result['byte_end'], self.offset + 5)

    def test_unstable_snapshots_rejected(self):
        samples = d.r.stable_badline(self.path, self.offset, d.RAW, self.identity, 0)
        samples[-1]['identity'] = list(samples[-1]['identity'])
        samples[-1]['identity'][2] += 1
        with patch.object(d.r, 'stable_badline', return_value=samples):
            with self.assertRaises(d.r.StreamBusy):
                self.verify()

    def test_wrong_snapshot_sha_rejected(self):
        samples = d.r.stable_badline(self.path, self.offset, d.RAW, self.identity, 0)
        samples[1]['line_sha256'] = '0' * 64
        with patch.object(d.r, 'stable_badline', return_value=samples):
            with self.assertRaises(d.r.StreamBusy):
                self.verify()

    def test_incomplete_snapshot_set_rejected(self):
        samples = d.r.stable_badline(self.path, self.offset, d.RAW, self.identity, 0)
        with patch.object(d.r, 'stable_badline', return_value=samples[:2]):
            with self.assertRaises(d.r.StreamBusy):
                self.verify()

    def test_concurrent_append_requires_new_stable_snapshots(self):
        def append(_):
            with self.path.open('ab') as stream:
                stream.write(b'new tail\n')
        with patch.object(d.r.time, 'sleep', side_effect=append):
            with self.assertRaises(d.r.StreamBusy):
                self.verify()

    def test_replaced_inode_rejected(self):
        replacement = self.root / 'replacement.jsonl'
        replacement.write_bytes(self.path.read_bytes())
        replacement.replace(self.path)
        with self.assertRaises(d.r.q.SourceChanged):
            self.verify()


if __name__ == '__main__':
    unittest.main()
