"""Read-only receipt monitor; never imports production workers or touches GPU state."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

POOL_NAMES = ('caption_two_stage_dynamic_pool_v1',
              'caption_two_stage_dynamic_pool_v2', 'caption_two_stage_union_pool_v3')
STATUSES = {'draft_generated': 'draft_generated', 'draft_partial': 'partial',
            'partial': 'partial', 'draft_needs_regeneration': 'needs_regeneration',
            'needs_regeneration': 'needs_regeneration'}
SUCCESS = {'draft_generated', 'partial'}
MAX_BYTES = 262144


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def decode(data):
    if b'\x00' in data:
        raise ValueError('NUL in JSON bytes')
    value = json.loads(data.decode('utf-8'))

    def check(item):
        if isinstance(item, str) and '\x00' in item:
            raise ValueError('Unicode NUL in JSON string')
        if isinstance(item, dict):
            for key, child in item.items():
                check(key)
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
    check(value)
    return value


class Stream:
    def __init__(self, root, baseline_bytes, identity):
        self.root = root
        self.path = root / 'terminal_index.jsonl'
        self.offset = 0
        self.baseline_bytes = baseline_bytes
        self.identity = identity
        self.largest_size = baseline_bytes
        self.stopped = None


class Monitor:
    def __init__(self, root, clock=time.monotonic, wall_clock=time.time):
        self.root = Path(root).resolve()
        self.clock, self.wall_clock = clock, wall_clock
        self.streams = []
        # Capture all initial byte boundaries before reading any stream.
        for name in POOL_NAMES:
            path = self.root / name / 'terminal_index.jsonl'
            try:
                info = path.stat()
                size, identity = info.st_size, (info.st_dev, info.st_ino)
            except FileNotFoundError:
                size, identity = 0, None
            self.streams.append(Stream(path.parent, size, identity))
        self.sources = {}
        self.conflicts = set()
        self.duplicate_records = 0
        self.conflict_duplicate_records = 0
        self.events = deque()
        self.started = self.previous = None
        self.sequence = 0

    def receipt(self, stream, row):
        reference = row['receipt']
        path_value = reference['path'] if isinstance(reference, dict) else reference
        if not isinstance(path_value, str) or not path_value:
            raise ValueError('receipt path must be a nonempty string')
        path = Path(path_value)
        if not path.is_absolute():
            path = stream.root / path
        if path.is_symlink() or path.parent.resolve() != (stream.root / 'terminals').resolve() or path.suffix != '.json':
            raise ValueError('receipt outside pool terminals or symlink')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
                raise ValueError('receipt must be a small regular file')
            data = handle.read(MAX_BYTES + 1)
            after = os.fstat(handle.fileno())
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('receipt changed during read')
        if not data.endswith(b'\n') or len(data) > MAX_BYTES:
            raise ValueError('incomplete/oversized terminal JSON')
        if isinstance(reference, dict) and 'sha256' in reference:
            if hashlib.sha256(data).hexdigest() != reference['sha256']:
                raise ValueError('receipt SHA mismatch')
        value = decode(data)
        split, sid, status = value['split'], value['id'], value['status']
        if split not in ('train', 'val', 'test') or not isinstance(sid, str) or not sid:
            raise ValueError('invalid source key')
        if status not in STATUSES:
            raise ValueError('unknown terminal status: ' + str(status))
        worker = value['owner']['worker_id']
        if not isinstance(worker, str) or not worker:
            raise ValueError('invalid worker_id')
        fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                allow_nan=False).encode()).hexdigest()
        return dict(key=(split, sid), status=STATUSES[status], worker=worker,
                    fingerprint=fingerprint, terminal_mtime=info.st_mtime,
                    pool=stream.root.name)

    def read_stream(self, stream, warnings):
        if stream.stopped:
            warnings.append(dict(pool=stream.root.name, code=stream.stopped, cursor=stream.offset))
            return []
        parsed = []
        next_offset = stream.offset
        try:
            with stream.path.open('rb') as handle:
                info = os.fstat(handle.fileno())
                identity = (info.st_dev, info.st_ino)
                if stream.identity is not None and identity != stream.identity:
                    stream.stopped = 'index_inode_changed'
                    raise ValueError(stream.stopped)
                stream.identity = identity
                if info.st_size < stream.largest_size:
                    stream.stopped = 'index_shrank'
                    raise ValueError(stream.stopped)
                stream.largest_size = info.st_size
                handle.seek(stream.offset)
                while handle.tell() < info.st_size:
                    position = handle.tell()
                    line = handle.readline(min(MAX_BYTES + 1, info.st_size - position))
                    if not line.endswith(b'\n') or len(line) > MAX_BYTES:
                        warnings.append(dict(pool=stream.root.name, code='incomplete_or_oversized_index_line', cursor=position))
                        break
                    try:
                        record = self.receipt(stream, decode(line))
                    except (ValueError, KeyError, TypeError, OSError) as error:
                        warnings.append(dict(pool=stream.root.name, code='index_or_receipt_read_error',
                                             cursor=position, detail=str(error)[:240]))
                        break
                    record['baseline'] = position < stream.baseline_bytes
                    parsed.append(record)
                    next_offset = handle.tell()
                current = stream.path.stat()
                if (current.st_dev, current.st_ino) != identity:
                    stream.stopped = 'index_inode_changed'
                    raise ValueError(stream.stopped)
                if current.st_size < stream.largest_size:
                    stream.stopped = 'index_shrank'
                    raise ValueError(stream.stopped)
        except (ValueError, OSError) as error:
            warnings.append(dict(pool=stream.root.name, code=stream.stopped or 'index_unavailable',
                                 cursor=stream.offset, detail=str(error)[:240]))
            return []
        stream.offset = next_offset
        return parsed

    def poll(self):
        warnings, pending = [], []
        for stream in self.streams:
            pending.extend(self.read_stream(stream, warnings))
        now, wall = self.clock(), self.wall_clock()
        if self.started is None:
            self.started = self.previous = now
        newly_successful = []
        for record in pending:
            key = record['key']
            first = self.sources.get(key)
            if first is not None:
                if first['fingerprint'] == record['fingerprint']:
                    self.duplicate_records += 1
                else:
                    self.conflict_duplicate_records += 1
                    self.conflicts.add(key)
                continue
            record['observed_monotonic'] = now
            record['observed_at'] = utc(wall)
            self.sources[key] = record
            if not record['baseline'] and record['status'] in SUCCESS:
                self.events.append((now, key))
                newly_successful.append(key)
        while self.events and self.events[0][0] <= now - 900:
            self.events.popleft()
        valid = [record for key, record in self.sources.items() if key not in self.conflicts]
        totals = Counter(record['status'] for record in valid)
        baseline = Counter(record['status'] for record in valid if record['baseline'])
        new_total = Counter(record['status'] for record in valid if not record['baseline'])
        window_keys = [key for _, key in self.events if key not in self.conflicts]
        window_counts = Counter(self.sources[key]['status'] for key in window_keys)
        elapsed = max(0.0, now - self.started)
        duration = min(900.0, elapsed)
        interval_seconds = max(0.0, now - self.previous)
        interval_keys = [key for key in newly_successful if key not in self.conflicts]
        workers = {}
        for record in valid:
            worker = workers.setdefault(record['worker'], dict(scope='worker', counts=Counter(),
                baseline_success=0, new_success_since_start=0, new_success_this_poll=0,
                rolling_15m_success=0, last_new_success_observed_at=None,
                latest_success_terminal_mtime_utc=None))
            worker['counts'][record['status']] += 1
            if record['status'] in SUCCESS:
                stamp = utc(record['terminal_mtime'])
                if worker['latest_success_terminal_mtime_utc'] is None or stamp > worker['latest_success_terminal_mtime_utc']:
                    worker['latest_success_terminal_mtime_utc'] = stamp
                if record['baseline']:
                    worker['baseline_success'] += 1
                else:
                    worker['new_success_since_start'] += 1
                    worker['last_new_success_observed_at'] = record['observed_at']
        for key in interval_keys:
            workers[self.sources[key]['worker']]['new_success_this_poll'] += 1
        for key in window_keys:
            workers[self.sources[key]['worker']]['rolling_15m_success'] += 1
        latest = max((record['observed_monotonic'] for record in valid
                      if not record['baseline'] and record['status'] in SUCCESS), default=self.started)
        idle = max(0.0, now - latest)
        if idle >= 600:
            warnings.append(dict(code='no_new_success_10min', idle_seconds=idle,
                                 automatic_action=False))
        self.sequence += 1
        summary = dict(schema='takeover_throughput_monitor_v1', observed_at=utc(wall),
            sequence=self.sequence, elapsed_seconds=elapsed, observation_monotonic=now,
            count_scope='global_unique_split_id_nonconflicting_terminal_receipts',
            successful_statuses=['draft_generated', 'partial'],
            terminal_receipts_are_not_approval=True, media_or_claims_inspected=False,
            global_counts={name: totals[name] for name in ('draft_generated', 'partial', 'needs_regeneration')},
            success_total=totals['draft_generated'] + totals['partial'],
            baseline_counts=dict(baseline), new_counts_since_start=dict(new_total),
            unique_terminal_sources=len(self.sources), duplicate_records=self.duplicate_records,
            conflict_duplicate_records=self.conflict_duplicate_records, conflicted_sources=len(self.conflicts),
            conflict_examples=[dict(split=key[0], id=key[1]) for key in sorted(self.conflicts)[:20]],
            interval=dict(seconds=interval_seconds, success=len(interval_keys),
                          success_per_minute=len(interval_keys) * 60 / interval_seconds if interval_seconds else None),
            rolling_15m=dict(seconds=duration, success=len(window_keys), counts=dict(window_counts),
                success_per_minute=len(window_keys) * 60 / duration if duration else None,
                full_window=elapsed >= 900),
            workers=workers, idle_seconds=idle, warnings=warnings,
            streams={stream.root.name: dict(cursor=stream.offset, baseline_bytes=stream.baseline_bytes,
                startup_baseline_pending=stream.offset < stream.baseline_bytes,
                identity=list(stream.identity) if stream.identity else None, stopped=stream.stopped)
                for stream in self.streams})
        self.previous = now
        return summary


def publish(out, summary):
    data = (json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    temporary = out / '.summary_latest.json.tmp'
    with temporary.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, out / 'summary_latest.json')
    with (out / 'monitor.jsonl').open('ab') as handle:
        handle.write(data)
        handle.flush()
    print(json.dumps(dict(observed_at=summary['observed_at'], success_total=summary['success_total'],
        new_success=summary['interval']['success'], success_per_minute_15m=summary['rolling_15m']['success_per_minute'],
        needs_regeneration=summary['global_counts']['needs_regeneration'],
        conflicts=summary['conflicted_sources'], warnings=len(summary['warnings'])), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be a positive finite number')
    out = (args.out or args.root / 'takeover_monitor_v1').resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Advisory flock only: never acquire production pool/GPU locks.
    with (out / 'monitor.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, 'monitor already running for this output directory\n')
        monitor = Monitor(args.root)
        while not (out / 'STOP').exists():
            tick = time.monotonic()
            publish(out, monitor.poll())
            if args.once:
                break
            deadline = tick + args.interval
            while time.monotonic() < deadline:
                if (out / 'STOP').exists():
                    return
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
