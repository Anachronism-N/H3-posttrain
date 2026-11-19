import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
import takeover_throughput_monitor_v2 as mod

ARTIFACT_ROOT = Path(__file__).resolve().parent.parent / 'takeover_monitor_v2'


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
            quality_approved=False, semantic_approval=False, needs_visual_revision=True,
            pixel_quality_review_completed=False)
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

    def test_startup_skips_all_historical_terminals_and_publishes_boundary(self):
        for pool in range(3):
            path, _, _ = self.add(pool, sid='old-%d' % pool)
            path.unlink()  # Historical terminal must never be opened.
        with mock.patch.object(mod.Monitor, 'receipt', side_effect=AssertionError('history read')):
            monitor = self.monitor()
            result = monitor.poll(scan=False)
            monitor.poll()
        self.assertEqual(result['success_total'], 0)
        self.assertEqual(result['historical_global_total_status'], 'unknown_not_scanned')
        self.assertEqual(result['success_total_scope'], 'observed_since_start')
        self.assertEqual(result['phase'], 'baseline_captured')
        for stream in monitor.streams:
            self.assertEqual(stream.offset, stream.baseline_bytes)

    def test_startup_incomplete_record_preserved_not_counted(self):
        _, old, _ = self.make(0, sid='old')
        self.paths[0].write_bytes(old[:-1])
        monitor = self.monitor()
        self.assertEqual(monitor.streams[0].offset, 0)
        self.append(0, b'\n')
        self.add(0, sid='new')
        self.clock.now = 60
        result = monitor.poll()
        self.assertEqual(result['success_total'], 1)
        self.assertEqual(result['unique_terminal_sources'], 1)
        self.assertEqual(result['interval']['success'], 1)
        self.assertEqual(monitor.streams[0].offset, self.paths[0].stat().st_size)

    def test_three_pool_new_formats_success_and_failure(self):
        monitor = self.monitor()
        monitor.poll(scan=False)
        self.clock.now = 60
        self.add(0, sid='a')
        self.add(1, sid='b', status='draft_partial', worker='second')
        self.add(2, sid='c', status='draft_needs_regeneration', worker='union')
        result = monitor.poll()
        self.assertEqual(result['global_counts'], dict(draft_generated=1, partial=1, needs_regeneration=1))
        self.assertEqual(result['success_total'], 2)
        self.assertEqual(result['interval']['success'], 2)
        self.assertEqual(result['warnings'], [])
        self.assertEqual(result['workers']['union']['new_success_since_start'], 0)

    def test_cross_new_pool_dedup_and_conflict(self):
        monitor = self.monitor()
        monitor.poll(scan=False)
        self.clock.now = 60
        _, _, value = self.add(0, sid='same')
        self.add(2, value=value)
        result = monitor.poll()
        self.assertEqual(result['success_total'], 1)
        self.assertEqual(result['duplicate_records'], 1)
        self.assertEqual(result['interval']['success'], 1)
        self.add(1, sid='same', status='draft_needs_regeneration')
        result = monitor.poll()
        self.assertEqual(result['conflicted_sources'], 1)
        self.assertEqual(result['success_total'], 0)
        self.assertEqual(result['rolling_15m']['success'], 0)

    def test_invalid_nul_utf8_json_cursor_reread(self):
        monitor = self.monitor()
        monitor.poll(scan=False)
        _, good, _ = self.make(0, sid='repaired')
        _, later, _ = self.make(0, sid='later')
        for bad in (b'\x00' + good[1:], b'!' + good[1:],
                    b'{"receipt":"\\u0000"}\n', b'{"receipt":"\xff"}\n'):
            padded = bad + b' ' * (len(good) - len(bad) - 1) + b'\n' if len(bad) < len(good) else bad
            self.paths[0].write_bytes(padded + later)
            result = monitor.poll()
            self.assertEqual(monitor.streams[0].offset, 0)
            self.assertEqual(result['success_total'], 0)
            self.assertTrue(result['warnings'])
        self.paths[0].write_bytes(good + later)
        result = monitor.poll()
        self.assertEqual(result['success_total'], 2)
        self.assertEqual(monitor.streams[0].offset, len(good + later))

    def test_incremental_half_line_keeps_prefix_cursor(self):
        monitor = self.monitor()
        monitor.poll(scan=False)
        _, first, _ = self.make(0, sid='first')
        _, second, _ = self.make(0, sid='second')
        self.paths[0].write_bytes(first + second[:-1])
        self.assertEqual(monitor.poll()['success_total'], 1)
        self.assertEqual(monitor.streams[0].offset, len(first))
        self.append(0, b'\n')
        self.assertEqual(monitor.poll()['success_total'], 2)

    def test_receipt_raw_reads_at_most_4096_and_bad_flags_stop(self):
        monitor = self.monitor()
        monitor.poll(scan=False)
        self.add(2, sid='ok')
        original = os.read
        with mock.patch.object(mod.os, 'read', wraps=original) as reads:
            self.assertEqual(monitor.poll()['success_total'], 1)
        self.assertTrue(reads.call_args_list)
        self.assertTrue(all(call.args[1] <= 4096 for call in reads.call_args_list))
        _, _, value = self.make(2, sid='bad')
        value['quality_approved'] = True
        self.add(2, value=value)
        before = monitor.streams[2].offset
        result = monitor.poll()
        self.assertEqual(monitor.streams[2].offset, before)
        self.assertTrue(any('FLAGS' in w.get('detail', '') for w in result['warnings']))

    def test_terminal_nul_missing_newline_repair(self):
        monitor = self.monitor()
        monitor.poll(scan=False)
        path, _, _ = self.add(2, sid='repair')
        good = path.read_bytes()
        for bad in (b'\x00' + good[1:], good[:-1]):
            path.write_bytes(bad)
            self.assertTrue(monitor.poll()['warnings'])
            self.assertEqual(monitor.streams[2].offset, 0)
        path.write_bytes(good)
        self.assertEqual(monitor.poll()['success_total'], 1)

    def test_inode_and_shrink_fail_closed(self):
        monitor = self.monitor()
        monitor.poll(scan=False)
        self.add(0, sid='old')
        monitor.poll()
        replacement = self.root / 'replace'
        replacement.write_bytes(self.paths[0].read_bytes())
        os.replace(replacement, self.paths[0])
        monitor.poll()
        self.assertEqual(monitor.streams[0].stopped, 'index_inode_changed')
        self.add(1, sid='one')
        monitor.poll()
        self.paths[1].write_bytes(b'')
        monitor.poll()
        self.assertEqual(monitor.streams[1].stopped, 'index_shrank')

    def test_rolling_15min_and_idle_use_observation_only(self):
        self.add(0, sid='historical')
        monitor = self.monitor()
        monitor.poll(scan=False)
        self.clock.now = 60
        self.add(2, sid='new60', worker='union')
        result = monitor.poll()
        self.assertEqual(result['rolling_15m']['seconds'], 60)
        self.assertEqual(result['rolling_15m']['success_per_minute'], 1)
        self.clock.now = 900
        self.add(2, sid='new900', worker='union', status='draft_partial')
        result = monitor.poll()
        self.assertAlmostEqual(result['rolling_15m']['success_per_minute'], 2 / 15)
        self.assertEqual(result['success_total'], 2)
        self.assertEqual(result['workers']['union']['rolling_15m_success'], 2)
        self.clock.now = 960
        self.assertEqual(monitor.poll()['rolling_15m']['success'], 1)
        self.clock.now = 1500
        result = monitor.poll()
        self.assertTrue(any(w['code'] == 'no_new_success_10min' and not w['automatic_action'] for w in result['warnings']))

    def test_bounded_scan_128_records_per_stream(self):
        monitor = self.monitor()
        monitor.poll(scan=False)
        path, line, _ = self.make(2, sid='one')
        self.append(2, line * 130)
        result = monitor.poll()
        self.assertEqual(monitor.streams[2].offset, len(line) * 128)
        self.assertEqual(result['duplicate_records'], 127)
        result = monitor.poll()
        self.assertEqual(monitor.streams[2].offset, len(line) * 130)
        self.assertEqual(result['duplicate_records'], 129)

    def test_cli_baseline_heartbeat_zero_terminal_reads_flock_and_stop(self):
        for pool in range(3):
            path, _, _ = self.add(pool, sid='old%d' % pool)
            path.unlink()
        out = self.root / 'out'
        command = [sys.executable, '-B', mod.__file__, '--root', str(self.root), '--out', str(out), '--once']
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        heartbeats = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(heartbeats[0]['phase'], 'baseline_loading')
        self.assertEqual(heartbeats[0]['terminal_reads'], 0)
        self.assertEqual(heartbeats[1]['phase'], 'baseline_captured')
        self.assertEqual(heartbeats[1]['success_total'], 0)
        summary = json.loads((out / 'summary_latest.json').read_text())
        self.assertEqual(summary['historical_terminals_read'], 0)
        self.assertEqual(len((out / 'monitor.jsonl').read_text().splitlines()), 1)
        with (out / 'monitor.lock').open('a+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            denied = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(denied.returncode, 2)
        (out / 'STOP').touch()
        stopped = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertEqual(stopped.returncode, 0)
        self.assertEqual(stopped.stdout, '')


if __name__ == '__main__':
    unittest.main()
