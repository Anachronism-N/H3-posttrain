import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import takeover_throughput_monitor_v1 as mod

ARTIFACT_ROOT = Path(__file__).resolve().parent.parent / 'takeover_monitor_v1'


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def wall(self):
        return 1700000000.0 + self.now


class MonitorTests(unittest.TestCase):
    def setUp(self):
        ARTIFACT_ROOT.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='fixture-', dir=ARTIFACT_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clock = Clock()
        self.paths = []
        for name in mod.POOL_NAMES:
            pool = self.root / name
            (pool / 'terminals').mkdir(parents=True)
            index = pool / 'terminal_index.jsonl'
            index.write_bytes(b'')
            self.paths.append(index)
        self.serial = 0

    def make(self, pool=0, sid='source', status='draft_generated', worker='worker-0', value=None):
        self.serial += 1
        receipt = value or dict(split='train', id=sid, status=status,
            owner=dict(worker_id=worker), batch_id='batch-' + str(self.serial),
            archive_path='/must-not-open-media/archive.tar', training_eligible=False,
            quality_approved=False, semantic_approval=False)
        data = (json.dumps(receipt, sort_keys=True) + '\n').encode()
        path = self.paths[pool].parent / 'terminals' / ('receipt-%d.json' % self.serial)
        path.write_bytes(data)
        os.utime(path, (self.clock.wall(), self.clock.wall()))
        reference = str(path) if pool == 2 else dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
        line = (json.dumps(dict(receipt=reference)) + '\n').encode()
        return path, line, receipt

    def append(self, pool, line):
        with self.paths[pool].open('ab') as handle:
            handle.write(line)

    def monitor(self):
        return mod.Monitor(self.root, self.clock.monotonic, self.clock.wall)

    def add(self, pool=0, **kwargs):
        path, line, value = self.make(pool, **kwargs)
        self.append(pool, line)
        return path, line, value

    def test_three_pool_formats_and_baseline(self):
        self.add(0, sid='a')
        self.add(1, sid='b', status='draft_partial', worker='worker-1')
        self.add(2, sid='c', status='draft_needs_regeneration', worker='worker-2')
        monitor = self.monitor()
        first = monitor.poll()
        self.assertEqual(first['global_counts'], dict(draft_generated=1, partial=1, needs_regeneration=1))
        self.assertEqual(first['success_total'], 2)
        self.assertEqual(first['rolling_15m']['success'], 0)
        self.assertEqual(first['interval']['success'], 0)
        self.assertIsNone(first['rolling_15m']['success_per_minute'])
        self.assertEqual(sum(w['baseline_success'] for w in first['workers'].values()), 2)
        self.assertEqual(first['warnings'], [])
        self.assertTrue(first['terminal_receipts_are_not_approval'])

    def test_cross_pool_same_source_dedup_and_worker_scope(self):
        _, _, value = self.add(0, sid='same')
        self.add(1, value=value)
        monitor = self.monitor()
        monitor.poll()
        self.clock.now = 60
        _, _, fresh = self.add(2, sid='new', worker='union-worker')
        self.add(0, value=fresh)
        result = monitor.poll()
        self.assertEqual(result['success_total'], 2)
        self.assertEqual(result['duplicate_records'], 2)
        self.assertEqual(result['interval']['success'], 1)
        self.assertEqual(result['workers']['union-worker']['new_success_since_start'], 1)
        self.assertEqual(result['rolling_15m']['success_per_minute'], 1)
        unchanged = monitor.poll()
        self.assertEqual(unchanged['interval']['success'], 0)
        self.assertEqual(unchanged['duplicate_records'], 2)

    def test_failures_never_count_as_success(self):
        monitor = self.monitor()
        monitor.poll()
        self.clock.now = 60
        self.add(0, sid='failed-a', status='draft_needs_regeneration')
        self.add(2, sid='failed-b', status='needs_regeneration')
        result = monitor.poll()
        self.assertEqual(result['success_total'], 0)
        self.assertEqual(result['global_counts']['needs_regeneration'], 2)
        self.assertEqual(result['rolling_15m']['success'], 0)
        self.assertEqual(result['interval']['success'], 0)

    def test_conflicting_duplicate_is_separate_and_excluded(self):
        monitor = self.monitor()
        monitor.poll()
        self.clock.now = 60
        self.add(0, sid='conflict')
        self.assertEqual(monitor.poll()['success_total'], 1)
        self.clock.now = 120
        self.add(2, sid='conflict', status='draft_needs_regeneration')
        result = monitor.poll()
        self.assertEqual(result['conflicted_sources'], 1)
        self.assertEqual(result['conflict_duplicate_records'], 1)
        self.assertEqual(result['success_total'], 0)
        self.assertEqual(result['rolling_15m']['success'], 0)
        self.assertEqual(result['unique_terminal_sources'], 1)
        self.assertEqual(result['global_counts']['needs_regeneration'], 0)

    def test_raw_nul_keeps_cursor_then_rereads_valid_suffix(self):
        monitor = self.monitor()
        monitor.poll()
        self.clock.now = 60
        _, good, _ = self.make(0, sid='good')
        _, suffix, _ = self.make(0, sid='suffix')
        self.paths[0].write_bytes(b'\x00' + good[1:] + suffix)
        result = monitor.poll()
        self.assertEqual(monitor.streams[0].offset, 0)
        self.assertEqual(result['success_total'], 0)
        self.assertTrue(result['warnings'])
        self.paths[0].write_bytes(good + suffix)
        result = monitor.poll()
        self.assertEqual(result['success_total'], 2)
        self.assertEqual(monitor.streams[0].offset, len(good + suffix))

    def test_unicode_nul_and_invalid_utf8_keep_cursor(self):
        for bad in (b'{"receipt":"bad\\u0000path"}\n', b'{"receipt":"\xff"}\n'):
            self.paths[0].write_bytes(bad)
            monitor = self.monitor()
            result = monitor.poll()
            self.assertEqual(monitor.streams[0].offset, 0)
            self.assertTrue(result['warnings'])
            self.assertEqual(result['success_total'], 0)

    def test_half_line_cursor_preserves_complete_prefix(self):
        monitor = self.monitor()
        monitor.poll()
        _, first, _ = self.make(0, sid='first')
        _, second, _ = self.make(0, sid='second')
        self.paths[0].write_bytes(first + second[:-1])
        result = monitor.poll()
        self.assertEqual(result['success_total'], 1)
        self.assertEqual(monitor.streams[0].offset, len(first))
        self.append(0, b'\n')
        result = monitor.poll()
        self.assertEqual(result['success_total'], 2)
        self.assertEqual(monitor.streams[0].offset, len(first + second))

    def test_malformed_json_keeps_cursor_without_skip(self):
        _, good, _ = self.make(0, sid='fixed')
        self.paths[0].write_bytes(b'!' + good[1:])
        monitor = self.monitor()
        self.assertTrue(monitor.poll()['warnings'])
        self.assertEqual(monitor.streams[0].offset, 0)
        self.paths[0].write_bytes(good)
        result = monitor.poll()
        self.assertEqual(result['success_total'], 1)
        self.assertEqual(result['interval']['success'], 0)
        self.assertEqual(result['baseline_counts']['draft_generated'], 1)

    def test_bad_terminal_keeps_index_cursor_and_recovers(self):
        path, _, _ = self.add(2, sid='repair')
        correct = path.read_bytes()
        path.write_bytes(b'\x00' + correct[1:])
        monitor = self.monitor()
        result = monitor.poll()
        self.assertTrue(result['warnings'])
        self.assertEqual(monitor.streams[2].offset, 0)
        path.write_bytes(correct)
        self.clock.now = 60
        result = monitor.poll()
        self.assertEqual(result['success_total'], 1)
        self.assertEqual(result['interval']['success'], 0)

    def test_terminal_missing_newline_keeps_cursor(self):
        path, _, _ = self.add(2)
        correct = path.read_bytes()
        path.write_bytes(correct[:-1])
        monitor = self.monitor()
        self.assertTrue(monitor.poll()['warnings'])
        self.assertEqual(monitor.streams[2].offset, 0)
        path.write_bytes(correct)
        self.assertEqual(monitor.poll()['success_total'], 1)

    def test_inode_change_stops_only_that_stream_permanently(self):
        self.add(0, sid='old')
        monitor = self.monitor()
        monitor.poll()
        _, new, _ = self.make(0, sid='replacement')
        replacement = self.root / 'replacement.jsonl'
        replacement.write_bytes(new)
        os.replace(replacement, self.paths[0])
        self.clock.now = 60
        self.add(2, sid='union')
        result = monitor.poll()
        self.assertEqual(monitor.streams[0].stopped, 'index_inode_changed')
        self.assertEqual(result['success_total'], 2)
        self.assertEqual(result['interval']['success'], 1)
        self.append(0, new)
        self.assertEqual(monitor.poll()['success_total'], 2)

    def test_shrink_stops_stream_even_unconsumed_bytes(self):
        self.paths[0].write_bytes(b'broken\n' * 10)
        monitor = self.monitor()
        monitor.poll()
        self.paths[0].write_bytes(b'broken\n')
        result = monitor.poll()
        self.assertEqual(monitor.streams[0].offset, 0)
        self.assertEqual(monitor.streams[0].stopped, 'index_shrank')
        self.assertTrue(result['warnings'])

    def test_rolling_15min_actual_duration_and_baseline_exclusion(self):
        self.add(0, sid='baseline')
        monitor = self.monitor()
        monitor.poll()
        self.clock.now = 60
        self.add(2, sid='new-60', worker='union')
        result = monitor.poll()
        self.assertEqual(result['rolling_15m']['seconds'], 60)
        self.assertEqual(result['rolling_15m']['success_per_minute'], 1)
        self.clock.now = 900
        self.add(2, sid='new-900', worker='union', status='draft_partial')
        result = monitor.poll()
        self.assertEqual(result['rolling_15m']['seconds'], 900)
        self.assertEqual(result['rolling_15m']['success'], 2)
        self.assertAlmostEqual(result['rolling_15m']['success_per_minute'], 2 / 15)
        self.assertTrue(result['rolling_15m']['full_window'])
        self.assertEqual(result['interval']['seconds'], 840)
        self.assertEqual(result['workers']['union']['rolling_15m_success'], 2)
        self.assertEqual(result['workers']['union']['last_new_success_observed_at'], mod.utc(self.clock.wall()))
        self.clock.now = 960
        result = monitor.poll()
        self.assertEqual(result['rolling_15m']['success'], 1)
        self.assertEqual(result['workers']['union']['new_success_since_start'], 2)
        self.assertEqual(result['workers']['union']['rolling_15m_success'], 1)

    def test_ten_minute_alarm_no_automatic_action(self):
        monitor = self.monitor()
        monitor.poll()
        self.clock.now = 599
        self.assertFalse(any(w['code'] == 'no_new_success_10min' for w in monitor.poll()['warnings']))
        self.clock.now = 600
        result = monitor.poll()
        alarm = next(w for w in result['warnings'] if w['code'] == 'no_new_success_10min')
        self.assertFalse(alarm['automatic_action'])
        self.clock.now = 660
        self.add(2)
        self.assertEqual(monitor.poll()['idle_seconds'], 0)

    def test_receipt_outside_terminals_rejected(self):
        outside = self.root / 'outside.json'
        outside.write_text('{}\n')
        self.append(2, (json.dumps(dict(receipt=str(outside))) + '\n').encode())
        monitor = self.monitor()
        self.assertTrue(monitor.poll()['warnings'])
        self.assertEqual(monitor.streams[2].offset, 0)

    def cli(self, out, *extra):
        return subprocess.run([sys.executable, '-B', str(Path(mod.__file__)), '--root', str(self.root),
            '--out', str(out), '--once', *extra], capture_output=True, text=True, timeout=10)

    def test_once_publishes_only_summary_log_lock_and_heartbeat(self):
        self.add(2)
        out = self.root / 'out'
        result = self.cli(out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(p.name for p in out.iterdir()), {'summary_latest.json', 'monitor.jsonl', 'monitor.lock'})
        summary = json.loads((out / 'summary_latest.json').read_text())
        log = (out / 'monitor.jsonl').read_text().splitlines()
        self.assertEqual(len(log), 1)
        self.assertEqual(json.loads(log[0]), summary)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)['success_total'], 1)

    def test_flock_refuses_second_instance_and_stop_prevents_poll(self):
        out = self.root / 'out'
        out.mkdir()
        with (out / 'monitor.lock').open('a+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.cli(out)
            self.assertEqual(result.returncode, 2)
            self.assertIn('already running', result.stderr)
            self.assertFalse((out / 'summary_latest.json').exists())
        (out / 'STOP').touch()
        result = self.cli(out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertFalse((out / 'summary_latest.json').exists())

    def test_invalid_interval_rejected(self):
        for interval in ('0', '-1', 'nan', 'inf'):
            result = self.cli(self.root / 'out', '--interval', interval)
            self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
