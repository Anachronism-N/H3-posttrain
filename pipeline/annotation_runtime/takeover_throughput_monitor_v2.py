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
import re
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
        self.offset = baseline_bytes
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
        self.startup_warnings = []
        for stream in self.streams:
            if not stream.baseline_bytes:
                continue
            fd = None
            try:
                fd = os.open(stream.path, os.O_RDONLY | os.O_NOFOLLOW)
                info = os.fstat(fd)
                if (info.st_dev, info.st_ino) != stream.identity:
                    stream.stopped = 'index_inode_changed'
                    raise ValueError(stream.stopped)
                if info.st_size < stream.baseline_bytes:
                    stream.stopped = 'index_shrank'
                    raise ValueError(stream.stopped)
                begin = max(0, stream.baseline_bytes - 4096)
                tail = os.pread(fd, stream.baseline_bytes - begin, begin)
                if len(tail) != stream.baseline_bytes - begin:
                    raise ValueError('short startup tail read')
                if not tail.endswith(b'\n'):
                    boundary = tail.rfind(b'\n')
                    if boundary < 0 and begin:
                        stream.stopped = 'startup_tail_boundary_unknown'
                        raise ValueError(stream.stopped)
                    stream.offset = begin + boundary + 1
                    self.startup_warnings.append(dict(pool=stream.root.name,
                        code='startup_partial_line_preserved', cursor=stream.offset))
            except (OSError, ValueError) as error:
                stream.stopped = stream.stopped or 'startup_boundary_unavailable'
                self.startup_warnings.append(dict(pool=stream.root.name,
                    code=stream.stopped, cursor=stream.offset, detail=str(error)[:240]))
            finally:
                if fd is not None:
                    os.close(fd)
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
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
                raise ValueError('receipt must be a small regular file')
            chunks, remaining = [], info.st_size
            while remaining:
                chunk = os.read(fd, min(4096, remaining))
                if not chunk:
                    raise ValueError('short terminal read')
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b''.join(chunks)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('receipt changed during read')
        if not data.endswith(b'\n') or len(data) > MAX_BYTES:
            raise ValueError('incomplete/oversized terminal JSON')
        if isinstance(reference, dict) and 'sha256' in reference:
            if hashlib.sha256(data).hexdigest() != reference['sha256']:
                raise ValueError('receipt SHA mismatch')
        value = decode(data)
        split, sid, status = value['split'], value['id'], value['status']
        if split not in ('train', 'val', 'test') or not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', sid):
            raise ValueError('invalid source key')
        if status not in STATUSES:
            raise ValueError('unknown terminal status: ' + str(status))
        expected_flags = dict(training_eligible=False, semantic_approval=False, quality_approved=False,
            needs_visual_revision=True, pixel_quality_review_completed=False)
        if any(value.get(name) is not expected for name, expected in expected_flags.items()):
            raise ValueError('terminal FLAGS missing or changed; no approval inferred')
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
        deadline = self.clock() + 5.0
        scanned = 0
        try:
            with stream.path.open('rb', buffering=0) as handle:
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
                while handle.tell() < info.st_size and scanned < 128 and self.clock() < deadline:
                    position = handle.tell()
                    chunks, length = [], 0
                    while handle.tell() < info.st_size and length <= MAX_BYTES:
                        block = os.read(handle.fileno(), min(4096, info.st_size - handle.tell(), MAX_BYTES + 1 - length))
                        if not block:
                            break
                        newline = block.find(b'\n')
                        if newline >= 0:
                            handle.seek(newline + 1 - len(block), os.SEEK_CUR)
                            block = block[:newline + 1]
                        chunks.append(block)
                        length += len(block)
                        if newline >= 0:
                            break
                    line = b''.join(chunks)
                    scanned += 1
                    if not line.endswith(b'\n') or len(line) > MAX_BYTES:
                        warnings.append(dict(pool=stream.root.name, code='incomplete_or_oversized_index_line', cursor=position))
                        break
                    try:
                        row = decode(line)
                        if position < stream.baseline_bytes:
                            next_offset = handle.tell()
                            continue  # A startup partial record is historical, not a new terminal.
                        record = self.receipt(stream, row)
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

    def poll(self, scan=True):
        warnings, pending = list(self.startup_warnings), []
        if scan:
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
        summary = dict(schema='takeover_throughput_monitor_v2', observed_at=utc(wall),
            sequence=self.sequence, elapsed_seconds=elapsed, observation_monotonic=now,
            phase='observing_incremental' if scan else 'baseline_captured',
            count_scope='observed_since_start_unique_split_id_nonconflicting_terminal_receipts',
            historical_global_total=None, historical_global_total_status='unknown_not_scanned',
            success_total_scope='observed_since_start',
            historical_terminals_read=0,
            source_population_completeness='unknown', resource_safety_status='not_assessed',
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
                startup_partial_pending=stream.offset < stream.baseline_bytes,
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
    print(json.dumps(dict(observed_at=summary['observed_at'], phase=summary['phase'],
        count_scope=summary['success_total_scope'], historical_global_total=None, success_total=summary['success_total'],
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
    out = (args.out or args.root / 'takeover_monitor_v2').resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Advisory flock only: never acquire production pool/GPU locks.
    with (out / 'monitor.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, 'monitor already running for this output directory\n')
        if (out / 'STOP').exists():
            return
        print(json.dumps(dict(phase='baseline_loading', count_scope='observed_since_start',
            historical_global_total=None, terminal_reads=0)), flush=True)
        monitor = Monitor(args.root)
        initial = True
        while not (out / 'STOP').exists():
            tick = time.monotonic()
            publish(out, monitor.poll(scan=not initial))
            initial = False
            if args.once:
                break
            deadline = tick + args.interval
            while time.monotonic() < deadline:
                if (out / 'STOP').exists():
                    return
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
